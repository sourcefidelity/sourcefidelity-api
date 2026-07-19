"""Book completeness checker — detects truncated / incomplete PDFs.

Instructors sometimes upload truncated PDFs (a single chapter, a download
that cut off mid-sentence). Storing these pollutes the source repository
with partial texts that produce false "source not found" results during
citation verification.

This module combines several independent signals into a verdict. It is
LOCAL-FIRST: the three primary signals need no network access and work
even when external APIs are unavailable or regionally blocked (e.g.
Chinese-web-optimized deployments).
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
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.services.page_layout import detect_n_up, NUpLayout

logger = logging.getLogger(__name__)

# Verdict constants
COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
UNCERTAIN = "UNCERTAIN"

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


def check_completeness(
    file_bytes: bytes,
    *,
    isbn: str | None = None,
    title: str | None = None,
    author: str | None = None,
    expected_pages: int | None = None,
    external_lookup: bool = True,
    is_article: bool = False,
) -> CompletenessReport:
    """Check whether a PDF is complete or truncated.

    Args:
        file_bytes: Raw PDF bytes.
        isbn: Optional ISBN for external page-count lookup.
        title: Optional title for external lookup fallback.
        author: Optional author for external lookup fallback.
        expected_pages: Optional instructor-supplied expected page count (Signal E).
        external_lookup: If True, attempt Google Books / Open Library lookup (Signal D).
        is_article: If True, skip book-specific heuristics (the short-document /
            no-back-matter check is meaningless for journal articles, which are
            legitimately short).

    Returns:
        CompletenessReport with a verdict and supporting detail.
    """
    signals: list[str] = []
    messages: list[str] = []

    # ── First: determine the true logical page count (N-up aware) ───────
    layout = detect_n_up(file_bytes)
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
        )

    # Track votes toward incomplete vs complete.
    incomplete_votes: list[str] = []
    complete_votes: list[str] = []

    # ── Signal A: TOC cross-reference ───────────────────────────────────
    toc_msg = _signal_toc_cross_reference(file_bytes, logical_pages)
    if toc_msg:
        signals.append(toc_msg["detail"])
        if toc_msg["vote"] == INCOMPLETE:
            incomplete_votes.append(toc_msg["detail"])
        elif toc_msg["vote"] == COMPLETE:
            complete_votes.append(toc_msg["detail"])

    # ── Signal C: back-matter presence ──────────────────────────────────
    # (Computed before Signal B because B uses back-matter as context.)
    back_msg = _signal_back_matter(file_bytes, logical_pages)
    signals.append(back_msg["detail"])
    if back_msg["vote"] == COMPLETE:
        complete_votes.append(back_msg["detail"])

    # ── Signal B: short-document / missing-back-matter heuristic ────────
    # A last-page terminal check is unreliable for books: page-boundary
    # truncation still ends with a period, and publisher footers/colophons
    # confuse mid-word detection. Instead, treat a SHORT book that also
    # lacks back matter as a suspicious combination.
    # SKIPPED for articles: journal articles are legitimately short and rarely
    # have an index, so the heuristic would flag every article as incomplete.
    if is_article:
        signals.append("Short-document check: skipped (journal article)")
    else:
        last_msg = _signal_short_or_missing_backmatter(
            logical_pages, has_back_matter=(back_msg["vote"] == COMPLETE)
        )
        signals.append(last_msg["detail"])
        if last_msg["vote"] == INCOMPLETE:
            incomplete_votes.append(last_msg["detail"])
        elif last_msg["vote"] == COMPLETE:
            complete_votes.append(last_msg["detail"])

    # ── Signal D / E: expected page count (external or manual) ──────────
    expected: int | None = None
    expected_source: str | None = None
    is_manual_expected = False

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
        "vote": COMPLETE,
        "detail": f"TOC cross-reference: all {len(toc)} entries within {logical_pages} logical pages",
    }


def _signal_short_or_missing_backmatter(logical_pages: int, has_back_matter: bool) -> dict:
    """Signal B: a short book with no back matter is suspicious.

    A last-page terminal check is unreliable for books (page-boundary
    truncation still ends with a period; publisher footers confuse mid-word
    detection). Instead, we flag books that are both SHORT and lack back
    matter — the combination that characterizes a truncated upload like a
    single chapter.

    Thresholds:
      - < 50 logical pages AND no back matter → INCOMPLETE (medium confidence)
      - < 50 logical pages WITH back matter → COMPLETE (a legitimate short book)
      - >= 50 logical pages, no back matter → weak note, no vote (many complete
        books lack an index, e.g. novels)
    """
    if logical_pages < 50:
        if has_back_matter:
            return {
                "vote": COMPLETE,
                "detail": f"Short-document check: {logical_pages} pages but back matter present (legitimate short work)",
            }
        return {
            "vote": INCOMPLETE,
            "detail": (
                f"Short-document check: INCOMPLETE — only {logical_pages} logical pages "
                "and no index/references/bibliography detected (likely a chapter, not a full book)"
            ),
        }
    if not has_back_matter:
        return {
            "vote": None,
            "detail": f"Short-document check: {logical_pages} pages, no back matter (weak note — many books lack an index)",
        }
    return {
        "vote": None,
        "detail": f"Short-document check: {logical_pages} pages with back matter (normal)",
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

    tail_lower = tail_text.lower()
    found = [m for m in _BACK_MATTER_MARKERS if m in tail_lower]
    if found:
        return {"vote": COMPLETE, "detail": f"Back-matter check: found {found[0]!r} in final {window} pages"}
    return {"vote": None, "detail": f"Back-matter check: no index/references found in final {window} pages (weak)"}


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
        if result.success and result.expected_pages:
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
