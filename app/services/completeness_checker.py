"""Book completeness checker — detects truncated / incomplete PDFs.

Instructors sometimes upload truncated PDFs (a single chapter, a download
that cut off mid-sentence). Storing these pollutes the source repository
with partial texts that produce false "source not found" results during
citation verification.

This module combines several independent signals into a verdict. It is
LOCAL-FIRST: the three primary signals need no network access and work
even when external APIs are unavailable from the deployment network.
External page-count lookup (Google Books, Open Library) is an optional
enhancement signal.

Signals
-------
A. TOC cross-reference (local): when the PDF has a table of contents,
   does the last referenced page fall within the document?
B. Last-page terminal check (local): does the final page end mid-word
   or mid-sentence, or with proper terminal content?
C. Back-matter presence (local): is there an index / references /
   bibliography near the end of the book?
D. External page-count (optional): does logical page count roughly match
   an expected count from Google Books / Open Library?
E. Manual entry (optional): an instructor-supplied expected page count.

The verdict is ADVISORY by default. The caller decides what to do with it
based on the configured strictness mode (see app.config.STRICTNESS_MODE).
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Literal

import fitz  # PyMuPDF

from app.services.book_metadata import extract_isbn_candidates
from app.services.page_layout import detect_n_up, NUpLayout

logger = logging.getLogger(__name__)

# Verdict constants
COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
UNCERTAIN = "UNCERTAIN"

DocumentKind = Literal["article", "book", "chapter", "unknown"]
_DOCUMENT_KINDS = {"article", "book", "chapter", "unknown"}

# Back-matter markers (lowercase). Presence near the end suggests a complete book.
_BACK_MATTER_MARKERS = (
    "index",
    "references",
    "bibliography",
    "works cited",
    "reference list",
    "author index",
    "subject index",
)

# What fraction of the final pages to scan for back matter.
_BACK_MATTER_WINDOW = 0.15

_TERMINAL_HEADING_RE = re.compile(
    r"^(?:references|bibliography|works cited|reference list|index|"
    r"author index|subject index)\s*$",
    re.IGNORECASE,
)
_EXPLICIT_PARTIAL_RE = re.compile(
    r"\b(?:book preview|document preview|preview copy|excerpt|extract only|"
    r"selected pages|sample chapter|sample pages)\b",
    re.IGNORECASE,
)
_PAGE_RANGE_RE = re.compile(
    # Local inference deliberately accepts the publication-metadata word
    # "pages", not citation-style "pp.": references in an article's opening
    # pages routinely contain unrelated cited-work ranges.
    r"\bpages?\s*[:.]?\s*(\d{1,5})\s*[-–—]\s*(\d{1,5})",
    re.IGNORECASE,
)

_RASTER_CROP_SAMPLE_LIMIT = 8
_RASTER_CROP_MIN_HIDDEN_FRACTION = 0.03
_RASTER_CROP_MIN_TOTAL_HIDDEN_FRACTION = 0.08
_RASTER_CROP_MIN_INK_DENSITY = 0.02


@dataclass
class CompletenessReport:
    """Result of the completeness check.

    Attributes:
        verdict: COMPLETE | INCOMPLETE | UNCERTAIN
        confidence: high | medium | low
        signals: list of human-readable signal descriptions (what fired)
        messages: list of human-readable summary messages for the instructor
        n_up_layout: the detected page layout (for caller reference)
        expected_pages: the externally- or manually-supplied expected count, if any
    """

    verdict: str
    confidence: str
    signals: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    n_up_layout: NUpLayout | None = None
    expected_pages: int | None = None
    document_kind: str = "unknown"


def check_completeness(
    file_bytes: bytes,
    *,
    isbn: str | None = None,
    title: str | None = None,
    author: str | None = None,
    expected_pages: int | None = None,
    expected_page_range: tuple[int, int] | None = None,
    external_lookup: bool = True,
    is_article: bool = False,
    document_kind: DocumentKind | None = None,
) -> CompletenessReport:
    """Check whether a PDF is complete or truncated.

    Args:
        file_bytes: Raw PDF bytes.
        isbn: Optional ISBN for external page-count lookup.
        title: Optional title for external lookup fallback.
        author: Optional author for external lookup fallback.
        expected_pages: Optional instructor-supplied expected page count (Signal E).
        expected_page_range: Optional inclusive first/last page range for an
            article or chapter, ideally from edition-specific metadata.
        external_lookup: If True, attempt Google Books / Open Library lookup (Signal D).
        is_article: If True, skip book-specific heuristics (the short-document /
            no-back-matter check is meaningless for journal articles. Retained
            for compatibility; prefer ``document_kind`` in new callers.
        document_kind: ``article``, ``book``, ``chapter`` or ``unknown``.
            Unknown and chapter inputs never inherit book-only pagination rules.

    Returns:
        CompletenessReport with a verdict and supporting detail.
    """
    signals: list[str] = []
    messages: list[str] = []
    kind = _normalize_document_kind(document_kind, is_article=is_article)

    # ── First: determine the true logical page count (N-up aware) ───────
    try:
        layout = detect_n_up(file_bytes)
    except Exception as exc:
        return CompletenessReport(
            verdict=UNCERTAIN,
            confidence="low",
            signals=[f"PDF layout could not be assessed ({type(exc).__name__})"],
            messages=[
                "Could not parse the PDF for completeness. The file must pass "
                "structural safety inspection before completeness is assessed."
            ],
            document_kind=kind,
        )
    logical_pages = layout.logical_pages
    if layout.is_n_up:
        signals.append(
            f"N-up layout detected: {layout.pages_per_sheet} pages/sheet "
            f"({layout.captured_pages} captured → {layout.logical_pages} logical pages)"
        )

    if logical_pages == 0:
        return CompletenessReport(
            verdict=INCOMPLETE,
            confidence="high",
            signals=["document has 0 pages"],
            messages=["PDF contains no pages."],
            n_up_layout=layout,
            document_kind=kind,
        )

    # Track votes toward incomplete vs complete.
    incomplete_votes: list[str] = []
    complete_votes: list[str] = []

    # ── Signal A: TOC cross-reference ───────────────────────────────────
    # An in-bounds PDF bookmark tree is internally consistent, but it is not
    # proof that the represented work is complete: publishers often generate
    # excerpt-specific bookmarks. It may vote INCOMPLETE, never COMPLETE.
    toc_msg = _signal_toc_cross_reference(file_bytes, logical_pages)
    if toc_msg:
        signals.append(toc_msg["detail"])
        if toc_msg["vote"] == INCOMPLETE:
            incomplete_votes.append(toc_msg["detail"])

    # ── Signal C: back-matter presence ──────────────────────────────────
    # (Computed before Signal B because B uses back-matter as context.)
    back_msg = _signal_back_matter(file_bytes, logical_pages)
    signals.append(back_msg["detail"])
    if back_msg["vote"] == COMPLETE:
        complete_votes.append(back_msg["detail"])

    # ── Signal B: explicit partial markers and advertised page ranges ──
    partial_msg = _signal_explicit_partial_marker(file_bytes)
    signals.append(partial_msg["detail"])
    if partial_msg["vote"] == INCOMPLETE:
        incomplete_votes.append(partial_msg["detail"])

    # A page crop can hide a substantial part of a scanned/OCR representation
    # even though its text ends cleanly and its page sequence is intact.  Only
    # repeated, ink-bearing crop of an axis-aligned full-page raster votes
    # incomplete; ordinary trimming of blank scanner margins does not.
    crop_msg = _signal_rendered_raster_coverage(file_bytes)
    signals.append(crop_msg["detail"])
    if crop_msg["vote"] == INCOMPLETE:
        incomplete_votes.append(crop_msg["detail"])

    range_msg = _signal_advertised_page_range(
        file_bytes,
        logical_pages,
        expected_page_range=expected_page_range,
    )
    signals.append(range_msg["detail"])
    if range_msg["vote"] == INCOMPLETE:
        incomplete_votes.append(range_msg["detail"])
    elif range_msg["vote"] == COMPLETE:
        complete_votes.append(range_msg["detail"])

    # ── Signal C: book-only captured pagination start ──────────────────
    # A stable Arabic page sequence beginning well after page 1 is strong
    # evidence that a *book* representation starts mid-work. It is normal for
    # journal articles and complete chapters, so those kinds never use it.
    if kind == "book" and not layout.is_n_up:
        start_msg = _signal_book_pagination_start(file_bytes)
        signals.append(start_msg["detail"])
        if start_msg["vote"] == INCOMPLETE:
            incomplete_votes.append(start_msg["detail"])
    else:
        signals.append(f"Book pagination-start check: skipped ({kind})")

    # Length without affirmative edition/terminal evidence is not a verdict.
    # Short articles, reports, pamphlets, chapters and monographs are common.
    signals.append(
        f"Document length: {logical_pages} logical pages (advisory only)"
    )

    # ── Signal D / E: expected page count (external or manual) ──────────
    expected: int | None = None
    expected_source: str | None = None
    is_manual_expected = False
    auto_isbn_candidates = ()

    if expected_pages is not None:
        # Signal E: manual entry (instructor knows the exact edition — reliable)
        expected = expected_pages
        expected_source = "instructor-supplied"
        is_manual_expected = True
    elif external_lookup:
        # Signal D: external lookup (Google Books → Open Library)
        # Edition-dependent: a mismatch often means a different edition, not
        # truncation. Treated as advisory unless the gap is severe.
        ext = _lookup_expected_pages(isbn=isbn, title=title, author=author)
        if ext is None and isbn is None and kind == "book":
            isbn_report = extract_isbn_candidates(file_bytes)
            auto_isbn_candidates = isbn_report.candidates
            if auto_isbn_candidates:
                signals.append(
                    "Automatic ISBN extraction: found "
                    f"{len(auto_isbn_candidates)} checksum-valid edition "
                    "candidate(s); metadata must match the cited book"
                )
            else:
                signals.append("Automatic ISBN extraction: no valid candidate found")
            # Do not send identifier candidates to external services unless
            # cited-work title evidence is available to reject unrelated ISBNs
            # printed in references, advertisements, or series front matter.
            if title and auto_isbn_candidates:
                ext = _lookup_expected_pages_for_candidates(
                    auto_isbn_candidates,
                    title=title,
                    author=author,
                )
        if ext is not None:
            expected = ext["pages"]
            expected_source = ext["source"]

    if expected is not None:
        page_msg = _signal_page_count(
            logical_pages, expected, expected_source, is_manual=is_manual_expected
        )
        signals.append(page_msg["detail"])
        if page_msg["vote"] == INCOMPLETE:
            incomplete_votes.append(page_msg["detail"])
        elif page_msg["vote"] == COMPLETE:
            complete_votes.append(page_msg["detail"])

    # ── Compute verdict ─────────────────────────────────────────────────
    verdict, confidence = _compute_verdict(incomplete_votes, complete_votes)

    if verdict == INCOMPLETE:
        messages.append(
            f"This PDF appears INCOMPLETE (confidence: {confidence}). "
            "It may be truncated — please verify before relying on it."
        )
    elif verdict == UNCERTAIN:
        messages.append(
            "Could not determine completeness with confidence. "
            "No strong signals either way — please verify manually."
        )
    else:
        messages.append("Completeness checks passed.")

    return CompletenessReport(
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        messages=messages,
        n_up_layout=layout,
        expected_pages=expected,
        document_kind=kind,
    )


# ── Signal implementations ─────────────────────────────────────────────


def _signal_toc_cross_reference(file_bytes: bytes, logical_pages: int) -> dict:
    """Signal A: does the TOC reference pages beyond the document?"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            toc = doc.get_toc()
        finally:
            doc.close()
    except Exception as e:
        return {"vote": None, "detail": f"TOC check skipped (extraction failed: {e})"}

    if not toc:
        return {"vote": None, "detail": "TOC cross-reference: no TOC present (unavailable)"}

    max_toc_page = max(e[2] for e in toc)
    # For N-up, TOC page numbers refer to book pages (logical), so compare against logical_pages.
    if max_toc_page > logical_pages:
        return {
            "vote": INCOMPLETE,
            "detail": (
                f"TOC cross-reference: INCOMPLETE — TOC references page {max_toc_page} "
                f"but document has {logical_pages} logical pages"
            ),
        }
    return {
        "vote": None,
        "detail": (
            f"TOC cross-reference: all {len(toc)} entries within {logical_pages} "
            "logical pages (internally consistent; not proof of full-work completeness)"
        ),
    }


def _signal_back_matter(file_bytes: bytes, logical_pages: int) -> dict:
    """Signal C: is there back matter (index/references) near the end?"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            n = len(doc)
            if n == 0:
                return {"vote": None, "detail": "Back-matter check: no pages"}
            # Scan the last 15% of pages
            window = max(1, int(n * _BACK_MATTER_WINDOW))
            start = max(0, n - window)
            tail_text = ""
            for page in doc[start:]:
                tail_text += page.get_text()
        finally:
            doc.close()
    except Exception as e:
        return {"vote": None, "detail": f"Back-matter check skipped (extraction failed: {e})"}

    lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
    found = [line for line in lines if _TERMINAL_HEADING_RE.fullmatch(line)]
    if found:
        return {
            "vote": COMPLETE,
            "detail": f"Back-matter check: found heading {found[0]!r} in final {window} pages",
        }
    return {"vote": None, "detail": f"Back-matter check: no index/references found in final {window} pages (weak)"}


def _normalize_document_kind(
    document_kind: DocumentKind | None,
    *,
    is_article: bool,
) -> DocumentKind:
    if document_kind is None:
        return "article" if is_article else "book"
    normalized = document_kind.strip().lower()
    if normalized not in _DOCUMENT_KINDS:
        raise ValueError(f"Unsupported document_kind: {document_kind}")
    return normalized  # type: ignore[return-value]


def _first_page_text(file_bytes: bytes, page_limit: int = 2) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(doc[i].get_text() for i in range(min(page_limit, len(doc))))
    finally:
        doc.close()


def _signal_explicit_partial_marker(file_bytes: bytes) -> dict:
    try:
        text = _first_page_text(file_bytes)
    except Exception as exc:
        return {
            "vote": None,
            "detail": f"Explicit partial-document check unavailable ({type(exc).__name__})",
        }
    match = _EXPLICIT_PARTIAL_RE.search(text)
    if match:
        return {
            "vote": INCOMPLETE,
            "detail": f"Explicit partial-document marker found ({match.group(0).lower()!r})",
        }
    return {"vote": None, "detail": "Explicit partial-document marker: none found"}


def _sample_page_indexes(page_count: int, limit: int) -> list[int]:
    """Return bounded, evenly distributed page indexes."""
    if page_count <= limit:
        return list(range(page_count))
    return sorted(
        {
            round(index * (page_count - 1) / (limit - 1))
            for index in range(limit)
        }
    )


def _ink_density(pixmap: fitz.Pixmap, start_x: int, end_x: int) -> float:
    """Measure dark-pixel density in a vertical image band."""
    width = end_x - start_x
    if width <= 0 or pixmap.height <= 0:
        return 0.0
    samples = pixmap.samples
    row_width = pixmap.width
    dark = 0
    for y_pos in range(pixmap.height):
        row_start = y_pos * row_width
        dark += sum(
            value < 180
            for value in samples[
                row_start + start_x : row_start + end_x
            ]
        )
    return dark / (width * pixmap.height)


def _signal_rendered_raster_coverage(file_bytes: bytes) -> dict:
    """Detect repeated page crops that remove ink-bearing raster content.

    This deliberately does not treat a CropBox difference by itself as an
    error.  Cropping blank scanner margins is normal.  The signal requires a
    full-page, unrotated raster to extend materially beyond the visible page
    and the hidden horizontal band to contain ink at a density comparable to
    the retained page image.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            sampled = _sample_page_indexes(
                len(doc), _RASTER_CROP_SAMPLE_LIMIT
            )
            assessable_pages = 0
            clipped_pages = 0
            max_hidden = 0.0

            for page_index in sampled:
                page = doc[page_index]
                candidates: list[tuple[float, int, fitz.Rect, fitz.Matrix]] = []
                for image in page.get_images(full=True):
                    xref = image[0]
                    for bbox, transform in page.get_image_rects(
                        xref, transform=True
                    ):
                        # Restrict the inference to an axis-aligned image whose
                        # source x-axis maps directly onto the page x-axis.
                        if (
                            transform.a <= 0
                            or transform.d <= 0
                            or abs(transform.b) > 0.01
                            or abs(transform.c) > 0.01
                        ):
                            continue
                        visible_width = min(bbox.x1, page.rect.width) - max(
                            bbox.x0, 0
                        )
                        visible_height = min(bbox.y1, page.rect.height) - max(
                            bbox.y0, 0
                        )
                        if (
                            visible_width < page.rect.width * 0.70
                            or visible_height < page.rect.height * 0.70
                        ):
                            continue
                        candidates.append(
                            (bbox.width * bbox.height, xref, bbox, transform)
                        )

                if not candidates:
                    continue
                _area, xref, bbox, _transform = max(candidates)
                hidden_left = max(0.0, -bbox.x0 / bbox.width)
                hidden_right = max(
                    0.0, (bbox.x1 - page.rect.width) / bbox.width
                )
                total_hidden = hidden_left + hidden_right
                if total_hidden < _RASTER_CROP_MIN_TOTAL_HIDDEN_FRACTION:
                    continue

                pixmap = fitz.Pixmap(doc, xref)
                if pixmap.colorspace != fitz.csGRAY:
                    pixmap = fitz.Pixmap(fitz.csGRAY, pixmap)
                if pixmap.alpha:
                    pixmap = fitz.Pixmap(pixmap, 0)
                while max(pixmap.width, pixmap.height) > 800:
                    pixmap.shrink(1)

                left_end = round(pixmap.width * hidden_left)
                right_start = round(pixmap.width * (1.0 - hidden_right))
                inner_density = _ink_density(pixmap, left_end, right_start)
                side_densities: list[float] = []
                if hidden_left >= _RASTER_CROP_MIN_HIDDEN_FRACTION:
                    side_densities.append(_ink_density(pixmap, 0, left_end))
                if hidden_right >= _RASTER_CROP_MIN_HIDDEN_FRACTION:
                    side_densities.append(
                        _ink_density(pixmap, right_start, pixmap.width)
                    )

                assessable_pages += 1
                hidden_has_content = any(
                    density >= _RASTER_CROP_MIN_INK_DENSITY
                    and density >= inner_density * 0.40
                    for density in side_densities
                )
                if hidden_has_content:
                    clipped_pages += 1
                    max_hidden = max(max_hidden, total_hidden)
        finally:
            doc.close()
    except Exception as exc:
        return {
            "vote": None,
            "detail": (
                "Rendered raster-coverage check unavailable "
                f"({type(exc).__name__})"
            ),
        }

    if assessable_pages == 0:
        return {
            "vote": None,
            "detail": "Rendered raster coverage: no material ink-bearing crop found",
        }

    required_pages = min(
        assessable_pages,
        max(1, math.ceil(len(sampled) * 0.50)),
    )
    if clipped_pages >= required_pages:
        return {
            "vote": INCOMPLETE,
            "detail": (
                "Rendered raster coverage: INCOMPLETE — page crop repeatedly "
                f"hides ink-bearing content ({clipped_pages}/{len(sampled)} "
                f"sampled pages; up to {max_hidden:.0%} of image width hidden)"
            ),
        }
    return {
        "vote": None,
        "detail": (
            "Rendered raster coverage: isolated/ambiguous crop evidence "
            f"({clipped_pages}/{len(sampled)} sampled pages; advisory only)"
        ),
    }


def _signal_advertised_page_range(
    file_bytes: bytes,
    logical_pages: int,
    *,
    expected_page_range: tuple[int, int] | None,
) -> dict:
    source = "provided metadata"
    page_range = expected_page_range
    if page_range is None:
        source = "first-page text"
        try:
            first_text = _first_page_text(file_bytes)
        except Exception as exc:
            return {
                "vote": None,
                "detail": f"Advertised page-range check unavailable ({type(exc).__name__})",
            }
        candidates = []
        for start_raw, end_raw in _PAGE_RANGE_RE.findall(first_text):
            start, end = int(start_raw), int(end_raw)
            span = end - start + 1
            if 1 < span <= 500:
                candidates.append((start, end))
        page_range = max(candidates, key=lambda item: item[1] - item[0], default=None)

    if page_range is None:
        return {"vote": None, "detail": "Advertised page range: unavailable"}

    start, end = page_range
    if start < 1 or end < start:
        return {"vote": None, "detail": "Advertised page range: invalid metadata ignored"}
    expected_span = end - start + 1
    ratio = logical_pages / expected_span
    if ratio < 0.8:
        return {
            "vote": INCOMPLETE,
            "detail": (
                f"Advertised page-range check ({source}): INCOMPLETE — "
                f"{logical_pages} logical pages captured for advertised pp. "
                f"{start}-{end} ({expected_span} pages, {ratio:.0%} captured)"
            ),
        }
    if 0.8 <= ratio <= 1.25:
        return {
            "vote": COMPLETE,
            "detail": (
                f"Advertised page-range check ({source}): {logical_pages} logical "
                f"pages are consistent with pp. {start}-{end}"
            ),
        }
    return {
        "vote": None,
        "detail": (
            f"Advertised page-range check ({source}): captured length exceeds "
            "the advertised span; edition/layout requires review"
        ),
    }


def _margin_page_numbers(page: fitz.Page) -> set[int]:
    numbers: set[int] = set()
    height = page.rect.height
    for _x0, y0, _x1, y1, word, *_rest in page.get_text("words"):
        if y0 >= height * 0.18 and y1 <= height * 0.82:
            continue
        if not re.fullmatch(r"\d{1,4}", word):
            continue
        value = int(word)
        if 1800 <= value <= 2099:  # likely a year, not pagination
            continue
        numbers.add(value)
    return numbers


def _signal_book_pagination_start(file_bytes: bytes) -> dict:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            samples = [
                _margin_page_numbers(doc[i]) for i in range(min(12, len(doc)))
            ]
        finally:
            doc.close()
    except Exception as exc:
        return {
            "vote": None,
            "detail": f"Book pagination-start check unavailable ({type(exc).__name__})",
        }

    estimated_starts: list[int] = []
    for index in range(len(samples) - 2):
        for number in samples[index]:
            if number + 1 in samples[index + 1] and number + 2 in samples[index + 2]:
                estimated_starts.append(number - index)

    if not estimated_starts:
        return {
            "vote": None,
            "detail": "Book pagination-start check: no stable early Arabic sequence",
        }
    estimated_first = min(estimated_starts)
    if estimated_first > 5:
        return {
            "vote": INCOMPLETE,
            "detail": (
                "Book pagination-start check: INCOMPLETE — captured sequence "
                f"appears to begin near printed page {estimated_first}"
            ),
        }
    return {
        "vote": None,
        "detail": (
            "Book pagination-start check: early sequence is compatible with "
            "front matter plus page 1"
        ),
    }


def _lookup_expected_pages(
    *,
    isbn: str | None,
    title: str | None,
    author: str | None,
) -> dict | None:
    """Signal D: look up expected page count from Google Books → Open Library."""
    # Try Google Books first
    gb = _lookup_google_books(isbn=isbn, title=title, author=author)
    if gb is not None:
        return gb

    # Fallback: Open Library
    ol = _lookup_open_library(isbn=isbn, title=title, author=author)
    if ol is not None:
        return ol

    return None


def _lookup_expected_pages_for_candidates(
    candidates,
    *,
    title: str,
    author: str | None,
) -> dict | None:
    """Resolve bounded automatically-extracted ISBN candidates conservatively."""
    matches: list[dict] = []
    # Equivalent ISBN-10/13 forms are already collapsed by the extractor.
    for candidate in candidates[:4]:
        result = _lookup_google_books(
            isbn=candidate.canonical_isbn13 or candidate.value,
            title=title,
            author=author,
        )
        if result is not None:
            matches.append(result)

    if not matches:
        return None
    pages = [int(match["pages"]) for match in matches]
    # Multiple formats of the same work may legitimately differ.  A bounded
    # cluster is useful; divergent records remain review evidence only.
    if max(pages) / min(pages) > 1.20:
        return None
    return {
        "pages": min(pages),
        "source": "Google Books exact ISBN extracted from PDF",
    }


def _signal_page_count(actual: int, expected: int, source: str, *, is_manual: bool) -> dict:
    """Compare actual logical pages vs expected.

    For MANUAL entry (instructor-supplied), use the configured threshold —
    the instructor knows the edition, so a real shortfall is meaningful.

    For EXTERNAL lookup (Google Books / Open Library), use a much looser
    threshold (severe mismatch only). Book editions routinely vary by 30%+
    in page count, so an external mismatch usually means a different edition,
    not truncation. Only flag at < 50% (a clear, severe truncation).
    """
    from app.config import settings

    ratio = actual / expected if expected else 1.0
    min_ratio = settings.COMPLETENESS_MIN_PAGE_RATIO  # 0.70 by default
    # External lookups: only flag severe mismatches (edition variation is common)
    effective_min = min_ratio if is_manual else 0.50

    if ratio < effective_min:
        return {
            "vote": INCOMPLETE,
            "detail": (
                f"Page-count check ({source}): INCOMPLETE — {actual} logical pages vs "
                f"{expected} expected ({ratio:.0%}, below {effective_min:.0%} threshold)"
            ),
        }
    if ratio < min_ratio and not is_manual:
        # Moderate shortfall against an external source — advisory note only.
        # Edition variation is the likely cause, so don't vote INCOMPLETE.
        return {
            "vote": None,
            "detail": (
                f"Page-count check ({source}): {actual} logical pages vs {expected} expected "
                f"({ratio:.0%}, below {min_ratio:.0%} but likely edition variation — advisory only)"
            ),
        }
    if ratio > 1.0 / min_ratio:
        return {
            "vote": None,
            "detail": (
                f"Page-count check ({source}): {actual} logical pages vs {expected} expected "
                f"({ratio:.0%}, more than expected — verify edition)"
            ),
        }
    return {
        "vote": COMPLETE,
        "detail": f"Page-count check ({source}): {actual} logical pages vs {expected} expected ({ratio:.0%}) — within range",
    }


# ── External lookup helpers (imported lazily so the module works offline) ──


def _lookup_google_books(
    *, isbn: str | None, title: str | None, author: str | None
) -> dict | None:
    """Look up expected page count from Google Books API."""
    try:
        from app.services.retrieval.google_books import GoogleBooksRetriever

        retriever = GoogleBooksRetriever()
        result = retriever.lookup_page_count(isbn=isbn, title=title, author=author)
        if (
            result.success
            and result.expected_pages
            and result.match_confidence == "high"
        ):
            return {"pages": result.expected_pages, "source": "Google Books"}
    except Exception as e:
        logger.debug("Google Books page-count lookup failed: %s", e)
    return None


def _lookup_open_library(
    *, isbn: str | None, title: str | None, author: str | None
) -> dict | None:
    """Look up expected page count from Open Library API."""
    try:
        from app.services.retrieval.open_library import OpenLibraryRetriever

        retriever = OpenLibraryRetriever()
        result = retriever.lookup_page_count(isbn=isbn, title=title, author=author)
        if result.success and result.expected_pages:
            return {"pages": result.expected_pages, "source": "Open Library"}
    except Exception as e:
        logger.debug("Open Library page-count lookup failed: %s", e)
    return None


def _compute_verdict(incomplete_votes: list[str], complete_votes: list[str]) -> tuple[str, str]:
    """Combine signal votes into a final verdict + confidence."""
    if not incomplete_votes and not complete_votes:
        return UNCERTAIN, "low"

    if incomplete_votes and not complete_votes:
        # No positive signals, at least one negative — likely incomplete
        confidence = "high" if len(incomplete_votes) >= 2 else "medium"
        return INCOMPLETE, confidence

    if incomplete_votes and complete_votes:
        # Conflicting signals — uncertain
        return UNCERTAIN, "medium"

    # Only complete votes
    confidence = "high" if len(complete_votes) >= 2 else "medium"
    return COMPLETE, confidence
