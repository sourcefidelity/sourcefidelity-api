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
) -> ValidationResult:
    """Revalidate identity/type against bounded OCR text without PDF metadata.

    OCR cannot preserve font prominence or embedded metadata. Automatic high
    confidence therefore requires two visible bibliographic fields, including
    an exact normalized title or DOI, and completeness must be established
    independently from the immutable parent PDF.
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

    normalized = _normalize_identity_text(front_text)
    normalized_title = _normalize_identity_text(expected_title or "")
    title_match = bool(
        len(normalized_title) >= 10 and normalized_title in normalized
    )
    author_value = (expected_author or "").strip()
    if "," in author_value:
        surname_value = author_value.split(",", 1)[0].strip()
    else:
        author_tokens = author_value.split()
        surname_value = author_tokens[-1] if author_tokens else ""
    surname = _normalize_identity_text(surname_value)
    author_match = bool(len(surname) >= 3 and surname in normalized)
    year_match = bool(expected_year and expected_year in front_text)
    normalized_doi = (expected_doi or "").strip().casefold()
    doi_match = bool(normalized_doi and normalized_doi in front_text.casefold())
    identity = (
        "high"
        if (doi_match and (title_match or author_match))
        or (title_match and author_match)
        else "medium"
        if doi_match or title_match or (author_match and year_match)
        else "low"
        if author_match or year_match
        else "rejected"
    )
    accept = identity == "high" and completeness == "complete"
    return ValidationResult(
        accept=accept,
        identity_confidence=identity,
        completeness=completeness,
        text_quality="scan_ocr",
        reason=("Accepted" if accept else "Needs review")
        + ": OCR derivative identity="
        + identity
        + f", completeness={completeness}",
        page_count=page_count,
        observed_source_kind=observed_kind.kind,
        source_kind_verdict=compatibility.verdict,
    )


def validate_retrieved_pdf(
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
    if identity == "rejected":
        return ValidationResult(
            accept=False, identity_confidence="rejected",
            completeness="skipped", text_quality=text_quality,
            reason="Identity check failed — PDF does not match the cited reference "
                   "(no DOI match and title overlap too low). Likely wrong source.",
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
                    for block in page.get_text("dict").get("blocks", [])
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]
                if not spans:
                    continue
                max_size = max(float(span.get("size", 0.0)) for span in spans)
                median_size = statistics.median(
                    float(span.get("size", 0.0)) for span in spans
                )
                upper_limit = float(page.rect.height) * 0.30
                has_distinct_display_type = max_size >= median_size + 1.5
                prominent = " ".join(
                    span["text"]
                    for span in spans
                    if float(span["bbox"][1]) <= upper_limit
                    or (
                        has_distinct_display_type
                        and float(span.get("size", 0.0)) >= max_size - 0.1
                    )
                )
                if title in _normalize_identity_text(prominent):
                    return True
        finally:
            document.close()
    except Exception:
        return False
    return False


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
                for block in page.get_text("dict").get("blocks", [])
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
            explicit_date_years: set[str] = set()
            document_date_cues = re.compile(
                r"\b(?:working paper|draft|version|revised|dated)\b",
                re.IGNORECASE,
            )
            month_date = re.compile(
                r"\b(?:january|february|march|april|(?-i:May)|june|july|august|"
                r"september|october|november|december)\b.{0,24}"
                r"\b(?:18|19|20)\d{2}\b",
                re.IGNORECASE,
            )
            for line in lines:
                line_text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                line_top = min(
                    float(span["bbox"][1]) for span in line.get("spans", [])
                )
                if line_top <= float(page.rect.height) * 0.55 and (
                    document_date_cues.search(line_text)
                    or month_date.search(line_text)
                ):
                    explicit_date_years.update(
                        re.findall(r"\b(?:18|19|20)\d{2}\b", line_text)
                    )
            return explicit_date_years
        finally:
            document.close()
    except Exception:
        return set()


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
        surname = _normalize_identity_text(expected_author.split(",", 1)[0])
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
        title_exact = _has_prominent_front_title_support(pdf_bytes, expected_title)
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
        # Extract surname: "Croteau, D." → "croteau"; "Smith J" → "smith"
        author_raw = expected_author.split(",")[0].strip()
        if not author_raw:
            author_raw = expected_author.split()[0]
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
    prominent_years = _explicit_first_page_document_years(pdf_bytes)
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
