"""Consolidated source PDF validator.

Single entry point for validating ANY retrieved PDF — whether from web search,
academic databases, student URLs, or instructor uploads. Consolidates three
previously-duplicate identity checks (pdf_verifier, source_resolver,
web_search) and adds completeness + text-quality checks that were missing
from the web-search path.

Three validation layers:
  1. IDENTITY: is this the RIGHT source? (DOI + title + author + year match)
  2. COMPLETENESS: is this a COMPLETE document? (not truncated, has back matter)
  3. TEXT QUALITY: is the text READABLE? (digital vs scan vs pure-scan)

Usage:
    from app.services.source_validator import validate_retrieved_pdf, ValidationResult
    result = validate_retrieved_pdf(pdf_bytes, doi="10.1234/foo", title="Some Paper")
    if result.accept:
        # use the PDF for verification
    else:
        # reject — result.reason explains why
"""

import logging
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Optional

from app.services.source_type import (
    SourceKindAssessment,
    classify_content_source_kind,
    compare_source_kinds,
    normalize_source_kind,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a retrieved PDF."""
    accept: bool               # True = use this PDF; False = reject
    identity_confidence: str   # "high" | "medium" | "low" | "rejected"
    completeness: str          # "complete" | "incomplete" | "uncertain" | "skipped"
    text_quality: str          # "digital" | "scan_ocr" | "pure_scan" | "skipped"
    reason: str                # human-readable explanation
    page_count: int = 0        # detected page count (for logging)
    observed_source_kind: str = "unknown"
    source_kind_verdict: str = "unknown"
    source_inspection: dict | None = None
    reason_code: str | None = None


def validate_ocr_derivative_text(
    text: str,
    *,
    expected_doi: str | None = None,
    expected_title: str | None = None,
    expected_author: str | None = None,
    expected_year: str | None = None,
    expected_source_kind: str | None = None,
    expected_source_kind_confidence: str = "unknown",
    expected_source_kind_evidence: tuple[str, ...] = (),
    completeness: str = "uncertain",
    page_count: int = 0,
    _layout_spans: list[dict] | None = None,
    _layout_lines: list[tuple[str, float]] | None = None,
    _layout_height: float = 0,
    _layout_pages: list[tuple[list[dict], float]] | None = None,
) -> ValidationResult:
    """Revalidate identity/type against bounded OCR text without PDF metadata.

    Uses the same identity rules as native PDF validation. Plain OCR text has
    no font/position or embedded-metadata observations: do not manufacture title
    prominence from occurrence alone. Completeness is independently established.
    """
    if not text.strip() or len(text) > 20_000_000:
        return ValidationResult(
            accept=False,
            identity_confidence="rejected",
            completeness=completeness,
            text_quality="scan_ocr",
            reason="OCR derivative text is empty or exceeds its validation bound.",
            page_count=page_count,
        )
    pages = text.split("\f")
    front_text = "\n".join(pages[:3])[:24_000]
    listing_conflict = _detect_nonwork_listing(front_text, expected_title)
    if listing_conflict:
        return ValidationResult(
            accept=False,
            identity_confidence="rejected",
            completeness=completeness,
            text_quality="scan_ocr",
            reason=f"OCR representation is {listing_conflict}, not the cited work.",
            page_count=page_count,
        )
    observed_kind = classify_content_source_kind(front_text)
    expected_kind = SourceKindAssessment(
        normalize_source_kind(expected_source_kind),
        expected_source_kind_confidence,
        expected_source_kind_evidence,
    )
    compatibility = compare_source_kinds(expected_kind, observed_kind)
    if compatibility.verdict == "incompatible":
        return ValidationResult(
            accept=False,
            identity_confidence="rejected",
            completeness=completeness,
            text_quality="scan_ocr",
            reason=f"Bibliographic type conflict — {compatibility.reason}.",
            page_count=page_count,
            observed_source_kind=observed_kind.kind,
            source_kind_verdict=compatibility.verdict,
        )

    identity = _check_identity_observations(
        front_text, "", expected_doi, expected_title, expected_author,
        expected_year,
        prominent_title=any(_title_in_layout_spans(spans, height, expected_title or "")
                            for spans, height in (_layout_pages if _layout_pages is not None
                                                  else [(_layout_spans or [], _layout_height)])[:3]),
        prominent_years=_document_years_in_lines(_layout_lines or [], _layout_height),
    )
    accept = identity == "high" and completeness == "complete"
    has_layout = _layout_spans is not None or bool(
        _layout_pages and any(spans for spans, _ in _layout_pages[:3])
    )
    return ValidationResult(
        accept=accept,
        identity_confidence=identity,
        completeness=completeness,
        text_quality="scan_ocr",
        reason=("Accepted" if accept else "Needs review")
        + ": OCR derivative identity="
        + identity
        + f", completeness={completeness}"
        + ("; OCR text has no title-layout observations" if not has_layout else
           "; bounded OCR positions available; font and embedded metadata unavailable"),
        page_count=page_count,
        observed_source_kind=observed_kind.kind,
        source_kind_verdict=compatibility.verdict,
    )


def validate_retrieved_pdf(pdf_bytes, *args, inspection_provider=None,
                           inspection_reconciliation=False, **kwargs) -> ValidationResult:
    """Existing deterministic verifier with an optional shared observation hook.

    The authorized caller supplies the hook after file safety checks. Inspection
    cannot change admission or completeness until its quality gate is accepted.
    """
    validation = _validate_retrieved_pdf_deterministic(pdf_bytes, *args, **kwargs)
    if (inspection_provider is not None and (validation.identity_confidence != 'rejected'
            or validation.reason_code == 'identity_insufficient_observations')
            and (validation.identity_confidence != 'high' or validation.completeness not in {'complete', 'skipped'})):
        try:
            validation.source_inspection = inspection_provider(pdf_bytes, validation)
            if inspection_reconciliation and not args:
                validation = _reconcile_inspected_title(pdf_bytes, validation, kwargs)
        except Exception:
            validation.source_inspection = {'status': 'incomplete', 'decision_applied': False,
                                            'reason_code': 'inspection_unavailable'}
    return validation


def _reconcile_inspected_title(pdf_bytes, validation, expected):
    """Narrow typo proposal, independently checked by the original validator.

    Never substitute model confidence for bibliographic or admission checks.
    Only one alphabetic edit in a sufficiently long title is eligible; numeric
    changes, short titles and all other reported differences abstain.
    """
    import hashlib
    from difflib import SequenceMatcher
    finding = validation.source_inspection
    if not isinstance(finding, dict):
        return validation
    if (finding.get('version') != 'source-inspection-v3'
            or finding.get('status') != 'complete'
            or finding.get('identity') != 'same_work'
            or finding.get('representation_role') != 'source_text'
            or finding.get('completeness') != 'not_established'
            or finding.get('content_sha256') != hashlib.sha256(pdf_bytes).hexdigest()
            or (validation.identity_confidence not in {'low', 'medium'}
                and validation.reason_code != 'identity_insufficient_observations')
            or expected.get('expected_doi') or expected.get('expected_isbn')
            or not expected.get('expected_author') or not expected.get('expected_year')):
        return validation
    differences = finding.get('differences', [])
    if not any(d.get('field') == 'title' and d.get('kind') == 'typographic' for d in differences):
        return validation
    for difference in differences:
        field = difference.get('field')
        if difference.get('kind') != 'typographic' or field not in {'title', 'author', 'year'}:
            return validation
        if field != 'title':
            # Models sometimes label identical author/date wording as a
            # difference. Independently identical fields require no repair;
            # retain the model record without letting it rewrite these fields.
            values = {_normalize_identity_text(o.get('quote', ''))
                      for o in finding.get('observations', []) if o.get('field') == field}
            if values != {_normalize_identity_text(expected.get('expected_'+field) or '')}:
                return validation
    titles = {o.get('quote') for o in finding.get('observations', [])
              if o.get('field') == 'title' and isinstance(o.get('quote'), str)}
    if len(titles) != 1:
        return validation
    observed = titles.pop()
    original = expected.get('expected_title') or ''
    a, b = _normalize_identity_text(original), _normalize_identity_text(observed)
    edits = [(a[i:j], b[k:l]) for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
             if op != 'equal']
    if (min(len(a), len(b)) < 20 or len(edits) != 1
            or max(map(len, edits[0])) != 1
            or not all(not part or part.isalpha() for part in edits[0])):
        return validation
    prominent = _has_prominent_front_title_support(pdf_bytes, observed)
    if not prominent and expected.get('expected_source_kind') in {'monograph', 'edited_collection'}:
        prominent = _late_book_title_identity(pdf_bytes, observed,
            expected['expected_author'], expected['expected_year'])
    if not prominent:
        return validation
    checked = _validate_retrieved_pdf_deterministic(
        pdf_bytes, **{**expected, 'expected_title': observed})
    if (checked.identity_confidence != 'high'
            or checked.source_kind_verdict == 'incompatible'
            or (validation.completeness != 'skipped'
                and checked.completeness != validation.completeness)
            or checked.text_quality != validation.text_quality):
        return validation
    if checked.completeness != 'complete' or checked.text_quality not in {'digital', 'scan_ocr'}:
        checked.accept = False
    checked.source_inspection = {**finding, 'decision_applied': True,
        'reconciliation_version': 'single-title-typo-v1',
        'original_identity_confidence': validation.identity_confidence,
        'bibliographic_identity': 'confirmed_with_minor_differences',
        'original_expected_title': original, 'observed_title': observed}
    return checked


def _validate_retrieved_pdf_deterministic(
    pdf_bytes: bytes,
    expected_doi: Optional[str] = None,
    expected_title: Optional[str] = None,
    expected_author: Optional[str] = None,
    expected_year: Optional[str] = None,
    expected_isbn: Optional[str] = None,
    is_article: bool = True,
    document_kind: str | None = None,
    expected_page_range: tuple[int, int] | None = None,
    expected_source_kind: str | None = None,
    expected_source_kind_confidence: str = "unknown",
    expected_source_kind_evidence: tuple[str, ...] = (),
    skip_completeness: bool = False,
    skip_text_quality: bool = False,
) -> ValidationResult:
    """Validate a retrieved PDF before accepting it for verification.

    Combines three checks:
      1. Identity (right source?) — DOI/title/author/year multi-field match
      2. Completeness (full document?) — TOC x-ref, back matter, page count
      3. Text quality (readable?) — digital vs scan vs pure-scan

    Args:
        pdf_bytes: The PDF file content.
        expected_doi: DOI from the cited reference (strongest identity signal).
        expected_title: Title from the cited reference.
        expected_author: Author from the cited reference.
        expected_year: Publication year from the cited reference.
        is_article: True for journal articles (skips book-only completeness
            heuristics). False for books/book chapters.
        skip_completeness: Skip completeness check (faster, less safe).
        skip_text_quality: Skip text quality check (faster, less safe).

    Returns:
        ValidationResult with accept/reject + confidence + reason.
    """
    if not pdf_bytes or len(pdf_bytes) < 100:
        return ValidationResult(
            accept=False, identity_confidence="rejected",
            completeness="skipped", text_quality="skipped",
            reason="PDF too small or empty",
        )

    # Layer 1: Text quality — can we extract text? (MUST come before identity,
    # because identity check needs extractable text to find DOI/title/author.
    # A pure scan has no text → can't check identity → reject or OCR first.
    # Future: if OCR is implemented, pure_scan → OCR → then identity check
    # on the OCR'd text.)
    text_quality = "skipped"
    if not skip_text_quality:
        text_quality = _check_text_quality(pdf_bytes)
        if text_quality == "pure_scan":
            return ValidationResult(
                accept=False, identity_confidence="skipped",
                completeness="skipped", text_quality=text_quality,
                reason="PDF is a pure scan (no text layer) — cannot extract "
                       "text for identity check or verification. Requires OCR "
                       "or instructor upload of a digital copy.",
            )

    # Work type is part of source identity, not merely a completeness hint.
    # A book review can repeat the reviewed book's title, author, year and DOI,
    # so field overlap alone is not a safe identity boundary.
    front_text = _extract_pdf_front_text(pdf_bytes)
    observed_kind = classify_content_source_kind(front_text)
    expected_kind = SourceKindAssessment(
        normalize_source_kind(expected_source_kind),
        expected_source_kind_confidence,
        expected_source_kind_evidence,
    )
    kind_compatibility = compare_source_kinds(expected_kind, observed_kind)
    if kind_compatibility.verdict == "incompatible":
        return ValidationResult(
            accept=False,
            identity_confidence="rejected",
            completeness="skipped",
            text_quality=text_quality,
            reason=f"Bibliographic type conflict — {kind_compatibility.reason}.",
            observed_source_kind=observed_kind.kind,
            source_kind_verdict=kind_compatibility.verdict,
        )

    listing_conflict = _detect_nonwork_listing(front_text, expected_title)
    if listing_conflict:
        return ValidationResult(
            accept=False,
            identity_confidence="rejected",
            completeness="skipped",
            text_quality=text_quality,
            reason=(
                "Representation identity conflict — the PDF is "
                f"{listing_conflict}, not the cited work itself."
            ),
            observed_source_kind=observed_kind.kind,
            source_kind_verdict=kind_compatibility.verdict,
        )

    # Layer 2: Identity check — is this the RIGHT source?
    # (Now safe to run — text quality check confirmed text is extractable.)
    identity = _check_identity(pdf_bytes, expected_doi, expected_title,
                                expected_author, expected_year)
    if identity != "high" and expected_kind.kind in {"monograph", "edited_collection"}:
        if _late_book_title_identity(pdf_bytes, expected_title, expected_author, expected_year):
            identity = "high"
    if identity == "rejected":
        return ValidationResult(
            accept=False, identity_confidence="rejected",
            completeness="skipped", text_quality=text_quality,
            reason="The inspected front matter does not provide enough information "
                   "to establish this source's identity.",
            reason_code="identity_insufficient_observations",
        )

    # Layer 3: Completeness — is this a full document?
    completeness = "skipped"
    page_count = 0
    if not skip_completeness:
        completeness, page_count = _check_completeness(
            pdf_bytes,
            is_article,
            document_kind=document_kind,
            expected_page_range=expected_page_range,
            isbn=expected_isbn,
            title=expected_title,
            author=expected_author,
        )
        if completeness == "incomplete":
            return ValidationResult(
                accept=False, identity_confidence=identity,
                completeness=completeness, text_quality=text_quality,
                page_count=page_count,
                reason="PDF appears incomplete (truncated, missing back matter, "
                       "or significantly shorter than expected).",
            )

    # Only high-confidence identity can be used automatically.  Medium/low
    # results remain inspectable but require review and must not become the
    # accepted representation merely because their bytes were downloadable.
    reasons = [f"identity={identity}"]
    if completeness != "skipped":
        reasons.append(f"completeness={completeness}")
    if text_quality != "skipped":
        reasons.append(f"text_quality={text_quality}")

    return ValidationResult(
        accept=identity == "high",
        identity_confidence=identity,
        completeness=completeness,
        text_quality=text_quality,
        page_count=page_count,
        reason=("Accepted: " if identity == "high" else "Needs review: ")
        + ", ".join(reasons),
        observed_source_kind=observed_kind.kind,
        source_kind_verdict=kind_compatibility.verdict,
    )


# ── Layer 1: Identity check ─────────────────────────────────────────────

def _late_book_title_identity(pdf_bytes, title, author, year) -> bool:
    """Conservative pages 4–6 fallback, never whole-book title occurrence.

    Require a short, prominent title page with its author and adjacent
    publication-year evidence. Reissues/conflicting copyright dates abstain.
    Existing representation/type checks have already run before this fallback.
    """
    if not title or not author or not year:
        return False
    wanted = _normalize_identity_text(title)
    surname = _normalize_identity_text(author.split(",")[0])
    if len(wanted) < 10 or len(surname) < 3:
        return False
    try:
        import fitz
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            texts = [doc[i].get_text() for i in range(min(6, len(doc)))]
            sample = "\n".join(texts)[:20000]
            if _detect_nonwork_listing(sample, title):
                return False
            if classify_content_source_kind(sample).kind in {"book_review", "journal_article", "book_section"}:
                return False
            if _explicit_first_page_document_years(pdf_bytes) - {year}:
                return False
            for index in range(3, len(texts)):
                text = texts[index]
                if len(text.split()) > 160 or surname not in _normalize_identity_text(text):
                    continue
                # Reuse the established geometric title check on this page only.
                with fitz.open() as single:
                    single.insert_pdf(doc, from_page=index, to_page=index)
                    if not _has_prominent_front_title_support(single.tobytes(), title, title_zone=0.60):
                        continue
                adjacent = "\n".join(texts[index:min(index + 2, 6)])
                if not re.search(r"\b" + re.escape(year) + r"\b", adjacent):
                    continue
                dated_lines = [line for line in adjacent.splitlines()
                               if re.search(r"copyright|©|published|reprint|edition", line, re.I)]
                dates = set(re.findall(r"\b(?:19|20)\d{2}\b", "\n".join(dated_lines)))
                if dates - {year}:
                    continue
                return True
    except Exception:
        return False
    return False


def _extract_pdf_front_text(pdf_bytes: bytes) -> str:
    """Extract a bounded front-matter sample for identity/type checks."""
    import fitz

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return "\n".join(
                document[index].get_text() for index in range(min(3, len(document)))
            )[:20000]
        finally:
            document.close()
    except Exception:
        return ""


def _extract_pdf_identity_metadata(pdf_bytes: bytes) -> str:
    """Extract bounded standard embedded identity fields."""
    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            metadata = document.metadata or {}
            return "\n".join(
                str(metadata.get(field, ""))[:1000]
                for field in ("title", "author", "subject", "keywords")
                if metadata.get(field)
            )
        finally:
            document.close()
    except Exception:
        return ""


def _has_prominent_front_title_support(
    pdf_bytes: bytes,
    expected_title: str | None,
    *, title_zone: float = 0.30,
) -> bool:
    """Return whether the expected title appears as visible front matter.

    Plain full-text occurrence is unsafe identity evidence: a later document by
    the same author can name the target work in its prose or bibliography. A
    title receives the strong identity weight only when it appears in the upper
    title zone or at the page's most prominent font size on one of the first
    three pages. Embedded metadata is handled separately by ``_check_identity``.
    """
    title = _normalize_identity_text(expected_title or "")
    if len(title) < 10:
        return False
    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for index in range(min(3, len(document))):
                page = document[index]
                spans = [
                    span
                    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES).get("blocks", [])
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]
                if not spans:
                    continue
                if _title_in_layout_spans(spans, float(page.rect.height), title,
                                          title_zone=title_zone):
                    return True
        finally:
            document.close()
    except Exception:
        return False
    return False


def _title_in_layout_spans(spans: list[dict], height: float, title: str,
                           *, title_zone: float = 0.30) -> bool:
    """Use actual positions; OCR word-box height is NOT a font-size estimate."""
    if not spans or len(_normalize_identity_text(title)) < 10:
        return False
    sizes = [float(span.get("size", 0.0)) for span in spans]
    max_size = max(sizes)
    display_type = max_size >= statistics.median(sizes) + 1.5
    prominent = " ".join(
        span["text"] for span in spans
        if float(span["bbox"][1]) <= height * title_zone
        or (display_type and float(span.get("size", 0.0)) >= max_size - 0.1)
    )
    return _normalize_identity_text(title) in _normalize_identity_text(prominent)


def _explicit_first_page_document_years(pdf_bytes: bytes) -> set[str]:
    """Extract explicitly dated document-version years from the first page."""
    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if not len(document):
                return set()
            page = document[0]
            lines = [
                line
                for block in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES).get("blocks", [])
                for line in block.get("lines", [])
                if line.get("spans")
            ]
            spans = [
                span
                for line in lines
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            if not spans:
                return set()
            positioned_lines = []
            for line in lines:
                line_text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                line_top = min(
                    float(span["bbox"][1]) for span in line.get("spans", [])
                )
                positioned_lines.append((line_text, line_top))
            return _document_years_in_lines(positioned_lines, float(page.rect.height))
        finally:
            document.close()
    except Exception:
        return set()


def _document_years_in_lines(lines: list[tuple[str, float]], height: float) -> set[str]:
    """Shared explicit document-version date rule over positioned first-page lines."""
    cues = re.compile(r"\b(?:working paper|draft|version|revised|dated)\b", re.I)
    date = re.compile(
        r"\b(?:january|february|march|april|(?-i:May)|june|july|august|"
        r"september|october|november|december)\b.{0,24}\b(?:18|19|20)\d{2}\b", re.I,
    )
    return {year for text, top in lines
            if top <= height * 0.55 and (cues.search(text) or date.search(text))
            for year in re.findall(r"\b(?:18|19|20)\d{2}\b", text)}


# An entry line in a bibliography: a capitalised opening, a four-digit year,
# and enough text around it to be a reference rather than a heading.
_CITATION_ENTRY = re.compile(r"(?m)^[A-Z\u2018\u2019\"'][^\n]{8,240}\b(?:19|20)\d{2}[a-z]?\b")


# Executable script and markup output are not document text. Two cookie-consent
# JavaScript bundles and a WordPress RSS feed were admitted as full-text
# representations on 2026-09-23 because the only check on retrieved text was a
# 500-character minimum. Densities measured over the stored text objects: the
# script markers appear 2.20 times per 1,000 characters in the JavaScript and
# at most 0.05 in every genuine source; tags appear 5.15 times per 1,000 in the
# feed and at most 0.24 in genuine sources. Both thresholds sit an order of
# magnitude clear of real prose.
#
# The alphabetic-character ratio is deliberately NOT used: a genuine source in
# the same set sits at 0.67, below the JavaScript's 0.72, so that signal would
# reject real documents.
_SCRIPT_MARKER = re.compile(
    r"(?:\bfunction\s*\(|=>|\bvar\s|\bconst\s|\btypeof\b|\};|===)")
_MARKUP_TAG = re.compile(r"</?[a-zA-Z][\w:-]*(?:\s[^<>]{0,200})?/?>")
_FEED_OR_OBJECT_OPENING = re.compile(r"^\s*(?:<\?xml|<rss\b|<feed\b|<!DOCTYPE\s+html|\{\s*\")", re.I)
_SCRIPT_MARKERS_PER_1K = 1.0
_MARKUP_TAGS_PER_1K = 2.0
# Density alone rejects a short work that quotes three tags or one function.
# The real payloads carry roughly 44 script markers and 103 tags across their
# first 20,000 characters, so an absolute floor separates them with a wide
# margin and spares a passing example.
_MIN_SCRIPT_MARKERS = 8
_MIN_MARKUP_TAGS = 12


def detect_nonprose_payload(text: str) -> str | None:
    """Say why retrieved text is not a document's prose, or None if it is.

    Fail-closed on strong structural markers only. A work that quotes a line of
    code or shows an XML example stays far below these densities.
    """
    sample = (text or "")[:20000]
    if len(sample) < 200:
        return None
    if _FEED_OR_OBJECT_OPENING.match(sample):
        return "a feed or structured data payload"
    per_1k = 1000 / len(sample)
    scripts = len(_SCRIPT_MARKER.findall(sample))
    if scripts >= _MIN_SCRIPT_MARKERS and scripts * per_1k >= _SCRIPT_MARKERS_PER_1K:
        return "executable script rather than document text"
    tags = len(_MARKUP_TAG.findall(sample))
    if tags >= _MIN_MARKUP_TAGS and tags * per_1k >= _MARKUP_TAGS_PER_1K:
        return "markup or feed output rather than document text"
    return None


def _detect_nonwork_listing(
    front_text: str, expected_title: str | None = None
) -> str | None:
    """Detect documents that only list or cite the expected work.

    Exact title and author overlap is not source identity when the containing
    representation is a CV/publication list, awards listing or publisher
    accessibility document about the work. Keep this
    fail-closed rule restricted to strong representation-level markers so an
    ordinary article that merely discusses awards is not rejected.
    """
    visible = " ".join(front_text.split())[:12000]
    normalized = _normalize_identity_text(visible)
    leading = normalized[:4000]

    # A download directory can reproduce a book's exact title/authors/ISBN on
    # page one. Repeated download headings PLUS several book-sized catalog
    # entries and URLs establish the document role, not merely topic overlap.
    download_headings = len(re.findall(r'(?im)^\s*DOWNLOAD\s*$', front_text[:12000]))
    catalog_entries = len(re.findall(r'\b(?:19|20)\d{2},\s*[^.;]{0,80}?\b\d{1,5}\s+pages\b', visible, re.I))
    if (download_headings >= 2 and catalog_entries >= 4
            and len(re.findall(r'https?://', front_text[:12000])) >= 3):
        return "a download/catalog listing"

    # A repository metadata export reproduces a work's exact title and authors
    # as labelled fields. It is a record about the work, never the work's text.
    # Require several distinct Dublin Core element labels so that ordinary
    # prose using one or two of these words is unaffected, and match without
    # line anchors because callers may supply whitespace-compacted text.
    metadata_labels = {
        match.group(1).lower()
        for match in re.finditer(
            r"(?i)\b(title|creator|contributor|subject|description|publisher|"
            r"date|identifier|relation|coverage|rights|language|format)\s*:\s*\S",
            visible[:12000],
        )
    }
    if len(metadata_labels) >= 5:
        return "a bibliographic metadata record"

    # A course reading list reproduces dozens of works' exact titles, authors
    # and years, and is about none of them. One was acquired as the source for
    # a monograph on 2026-09-23: a 32-page Talis Aspire list whose page one is
    # a bibliography, from which a naive reader took the first entry's DOI as
    # the document's own. Talis stamps "readinglists@<institution>" in the page
    # header; other platforms title the document a reading or resource list.
    # Either marker could appear in passing inside an ordinary work, so require
    # it together with a body of citation entries. Measured over the 88 stored
    # source objects, the marker matches exactly one document -- that list --
    # and no accepted source.
    reading_list_marker = bool(
        re.search(r"\breading\s*lists?\s*@", visible, re.I)
        or re.search(r"\b(?:rl|readinglists)\.[a-z0-9.-]*talis\b", visible, re.I)
        or re.search(r"(?im)^[^\n]{0,80}\b(?:reading|resource)\s+list\b[^\n]{0,80}$",
                     front_text[:2000])
    )
    if reading_list_marker and len(_CITATION_ENTRY.findall(front_text[:12000])) >= 8:
        return "a course reading list"

    cv_marker = bool(
        re.search(r"\b(?:curriculum vitae|abridged c v)\b", leading)
        or (
            "academic employment" in leading
            and "education" in leading
            and "publications" in normalized
        )
    )
    if cv_marker:
        return "a curriculum vitae or publication list"

    award_count = len(re.findall(r"\baward\b", leading))
    if "award winners" in leading or (
        award_count >= 4 and "outstanding" in leading
    ):
        return "an awards or contents listing"

    # A support PDF may prominently repeat the book's author/title/ISBN. Require
    # an opening purpose heading, standards language AND a first-party platform
    # declaration; accessibility vocabulary alone is ordinary scholarly content.
    purpose = re.search(
        r"\b(?:our commitment to accessibility|accessibility conformance report|"
        r"voluntary product accessibility template|product accessibility report|"
        r"accessibility statement)\b",
        leading[:1200],
    )
    standards = bool(re.search(r"\b(?:wcag|vpats?|section 508)\b", normalized))
    platform_declaration = bool(
        re.search(
            r"\bthis report (?:reflects|describes|documents) the accessibility "
            r"(?:level|status|conformance)\b",
            normalized,
        )
        or re.search(
            r"\b(?:our|this) (?:product|platform) (?:conforms|supports|complies)\b",
            normalized,
        )
    )
    if purpose and standards and platform_declaration:
        cited_title = _normalize_identity_text(expected_title or "")
        # This is not an identity approval. An explicitly cited support report
        # still has to pass the normal field, type and completeness checks.
        cites_report_itself = bool(
            cited_title
            and (cited_title == purpose.group() or cited_title.startswith(purpose.group() + " "))
            and cited_title in leading
        )
        if not cites_report_itself:
            return "a publisher accessibility or conformance document"

    return None


def _visible_identity_support_count(
    front_text: str,
    expected_title: str | None,
    expected_author: str | None,
    expected_year: str | None,
) -> int:
    """Count independent visible front-matter support for embedded metadata."""
    normalized = _normalize_identity_text(front_text)
    support = 0
    if expected_title:
        title_tokens = {
            token
            for token in _normalize_identity_text(expected_title).split()
            if len(token) >= 4
        }
        if title_tokens:
            matches = sum(token in normalized for token in title_tokens)
            support += int(matches / len(title_tokens) >= 0.6)
    if expected_author:
        surname = _normalize_identity_text(_identity_surname(expected_author))
        support += int(bool(surname and len(surname) >= 3 and surname in normalized))
    if expected_year:
        support += int(expected_year in front_text)
    return support

def _check_identity(
    pdf_bytes: bytes,
    expected_doi: Optional[str],
    expected_title: Optional[str],
    expected_author: Optional[str],
    expected_year: Optional[str],
) -> str:
    """Check if the PDF matches the cited reference via multi-field triangulation.

    Uses triangulation across multiple reference fields (title + author + year)
    rather than relying on title token overlap alone. A student may have errors
    in one field, but author + year + approximate title together is strong evidence.

    Returns: "high" | "medium" | "low" | "rejected"
    """
    front_text = _extract_pdf_front_text(pdf_bytes)
    identity_metadata = _extract_pdf_identity_metadata(pdf_bytes)
    return _check_identity_observations(
        front_text, identity_metadata, expected_doi, expected_title,
        expected_author, expected_year,
        prominent_title=_has_prominent_front_title_support(pdf_bytes, expected_title),
        prominent_years=_explicit_first_page_document_years(pdf_bytes),
    )


def _check_identity_observations(
    front_text: str,
    identity_metadata: str,
    expected_doi: Optional[str],
    expected_title: Optional[str],
    expected_author: Optional[str],
    expected_year: Optional[str],
    *,
    prominent_title: bool,
    prominent_years: set[str],
) -> str:
    """Shared native/OCR identity policy; callers supply only observed signals.

    Layout absence is not title prominence. This internal scorer neither
    authorizes source access nor decides completeness, readability or admission.
    Representation-level listing/type guards remain in the consolidated entries.
    """
    text = f"{front_text}\n{identity_metadata}"[:24000]

    if not text.strip():
        return "rejected"

    text_identity = _normalize_identity_text(text)

    # ── DOI match (definitive — unique identifier) ──
    if expected_doi:
        normalized_doi = expected_doi.lower()
        if normalized_doi in front_text.lower():
            return "high"
        if (
            normalized_doi in identity_metadata.lower()
            and _visible_identity_support_count(
                front_text, expected_title, expected_author, expected_year
            )
            >= 2
        ):
            return "high"

    # ── Multi-field triangulation ──
    # Collect signals from each available field
    signals = []

    # Signal 1: Title — check for EXACT title string (contiguous phrase),
    # not just scattered tokens. Much stronger than token overlap.
    title_exact = False
    title_fuzzy = False
    if expected_title:
        title_clean = _normalize_identity_text(expected_title)
        title_in_visible_text = bool(
            title_clean and len(title_clean) >= 10 and title_clean in text_identity
        )
        title_in_metadata = bool(
            title_clean
            and len(title_clean) >= 10
            and title_clean in _normalize_identity_text(identity_metadata)
        )
        # Strong title identity requires title-zone/prominent front matter. A
        # same-author later work may repeat the exact target title and year in
        # its prose or bibliography; that plain occurrence is supporting only.
        title_exact = prominent_title
        if (
            not title_exact
            and title_in_metadata
            and _visible_identity_support_count(
                front_text, expected_title, expected_author, expected_year
            )
            >= 2
        ):
            title_exact = True
        # Fuzzy: token overlap (only as a supporting signal, never standalone)
        title_tokens = {
            token for token in title_clean.split() if len(token) >= 4
        }
        if title_tokens:
            matches = sum(1 for token in title_tokens if token in text_identity)
            if title_in_visible_text or matches / len(title_tokens) >= 0.6:
                title_fuzzy = True

    if title_exact:
        signals.append(("title_exact", 3))
    elif title_fuzzy:
        signals.append(("title_fuzzy", 1))

    # Signal 2: Author surname
    author_match = False
    if expected_author:
        # Shared convention: "Croteau, D." or "Jane Smith" supplies a surname.
        author_raw = _identity_surname(expected_author)
        surname = _normalize_identity_text(author_raw)
        if surname and len(surname) >= 3 and surname in text_identity:
            author_match = True
            signals.append(("author", 2))

    # Signal 3: Year
    year_match = False
    if expected_year and expected_year in text:
        year_match = True
        signals.append(("year", 1))

    # ── Scoring: require triangulation (2+ signals) for acceptance ──
    total_score = sum(s[1] for s in signals)

    # A prominent conflicting document year distinguishes an earlier/later
    # working paper from the cited publication. The expected year may still
    # occur in prose or references, so title+author overlap cannot override it.
    visible_year_conflict = bool(
        expected_year
        and prominent_years
        and expected_year not in prominent_years
    )

    if total_score >= 5:  # e.g., title_exact(3) + author(2) = 5
        return "medium" if visible_year_conflict else "high"
    elif total_score >= 3:  # e.g., title_exact(3) alone, or title_fuzzy(1) + author(2)
        return "medium"
    elif total_score >= 2:  # e.g., author(2) alone, or title_fuzzy(1) + year(1)
        return "low"
    else:
        return "rejected"


def _identity_surname(author: str) -> str:
    """Share the existing OCR comma/surname convention with native checking."""
    value = author.strip()
    if "," in value:
        return value.split(",", 1)[0].strip()
    tokens = value.split()
    return tokens[-1] if tokens else ""


def _normalize_identity_text(value: str) -> str:
    """Fold diacritics and punctuation variants for bibliographic comparison."""
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    ascii_like = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_like).split())


# ── Layer 2: Text quality ───────────────────────────────────────────────

def _check_text_quality(pdf_bytes: bytes) -> str:
    """Check if the PDF has readable text.

    Returns: "digital" | "scan_ocr" | "pure_scan" | "unknown"
    - "digital": born-digital PDF with extractable text (best)
    - "scan_ocr": scanned but has OCR text layer (usable, medium confidence)
    - "pure_scan": scanned image with no text layer (unusable without OCR)
    - "unknown": classification raised an exception (honest label, not "digital")
    """
    from app.services.page_layout import classify_text_quality
    try:
        result = classify_text_quality(pdf_bytes, sample_size=8)
        return result.verdict  # extract the string ("digital"/"scan_ocr"/"pure_scan")
    except Exception as e:
        logger.debug("Text quality check failed (type=%s)", type(e).__name__)
        # Report honestly — don't claim "digital" for a PDF we couldn't classify
        # (REVIEW §3.2). Callers see text_quality="unknown"; identity/completeness
        # checks still run, so this only affects the reported quality label.
        return "unknown"


# ── Layer 3: Completeness ───────────────────────────────────────────────

def _check_completeness(
    pdf_bytes: bytes,
    is_article: bool = True,
    *,
    document_kind: str | None = None,
    expected_page_range: tuple[int, int] | None = None,
    isbn: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> tuple[str, int]:
    """Check if the PDF is complete (not truncated).

    Returns: (verdict, page_count)
    - verdict: "complete" | "incomplete" | "uncertain"
    """
    from app.services.completeness_checker import check_completeness
    try:
        report = check_completeness(
            pdf_bytes,
            is_article=is_article,
            document_kind=document_kind,
            expected_page_range=expected_page_range,
            isbn=isbn,
            title=title,
            author=author,
        )
        page_count = (
            report.n_up_layout.logical_pages if report.n_up_layout else 0
        )
        return report.verdict.lower(), page_count
    except Exception as e:
        logger.debug("Completeness check failed (type=%s)", type(e).__name__)
        return "uncertain", 0
