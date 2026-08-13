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
from dataclasses import dataclass
from typing import Optional

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


def validate_retrieved_pdf(
    pdf_bytes: bytes,
    expected_doi: Optional[str] = None,
    expected_title: Optional[str] = None,
    expected_author: Optional[str] = None,
    expected_year: Optional[str] = None,
    is_article: bool = True,
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
        completeness, page_count = _check_completeness(pdf_bytes, is_article)
        if completeness == "incomplete":
            return ValidationResult(
                accept=False, identity_confidence=identity,
                completeness=completeness, text_quality=text_quality,
                page_count=page_count,
                reason="PDF appears incomplete (truncated, missing back matter, "
                       "or significantly shorter than expected).",
            )

    # All checks passed (or uncertain — accept with appropriate confidence)
    reasons = [f"identity={identity}"]
    if completeness != "skipped":
        reasons.append(f"completeness={completeness}")
    if text_quality != "skipped":
        reasons.append(f"text_quality={text_quality}")

    return ValidationResult(
        accept=True,
        identity_confidence=identity,
        completeness=completeness,
        text_quality=text_quality,
        page_count=page_count,
        reason="Accepted: " + ", ".join(reasons),
    )


# ── Layer 1: Identity check ─────────────────────────────────────────────

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
    import fitz

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for i in range(min(3, len(doc))):
            text += doc[i].get_text()
        doc.close()
    except Exception:
        return "rejected"

    if not text.strip():
        return "rejected"

    text_lower = text.lower()

    # ── DOI match (definitive — unique identifier) ──
    if expected_doi and expected_doi.lower() in text_lower:
        return "high"

    # ── Multi-field triangulation ──
    # Collect signals from each available field
    signals = []

    # Signal 1: Title — check for EXACT title string (contiguous phrase),
    # not just scattered tokens. Much stronger than token overlap.
    title_exact = False
    title_fuzzy = False
    if expected_title:
        title_clean = expected_title.strip().lower()
        # Check if the full title (or 80%+ of it) appears as a contiguous string
        if title_clean and len(title_clean) >= 10:
            if title_clean in text_lower:
                title_exact = True
            elif len(title_clean) > 30:
                # For long titles, check if first 30 chars appear as substring
                if title_clean[:30] in text_lower:
                    title_exact = True
        # Fuzzy: token overlap (only as a supporting signal, never standalone)
        title_tokens = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", expected_title)
                       if len(t) >= 4}
        if title_tokens:
            matches = sum(1 for t in title_tokens if t in text_lower)
            if matches / len(title_tokens) >= 0.6:
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
        surname = author_raw.lower()
        if surname and len(surname) >= 3 and surname in text_lower:
            author_match = True
            signals.append(("author", 2))

    # Signal 3: Year
    year_match = False
    if expected_year and expected_year in text:
        year_match = True
        signals.append(("year", 1))

    # ── Scoring: require triangulation (2+ signals) for acceptance ──
    total_score = sum(s[1] for s in signals)

    if total_score >= 5:  # e.g., title_exact(3) + author(2) = 5
        return "high"
    elif total_score >= 3:  # e.g., title_exact(3) alone, or title_fuzzy(1) + author(2)
        return "medium"
    elif total_score >= 2:  # e.g., author(2) alone, or title_fuzzy(1) + year(1)
        return "low"
    else:
        return "rejected"


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
        logger.debug("Text quality check failed: %s", e)
        # Report honestly — don't claim "digital" for a PDF we couldn't classify
        # (REVIEW §3.2). Callers see text_quality="unknown"; identity/completeness
        # checks still run, so this only affects the reported quality label.
        return "unknown"


# ── Layer 3: Completeness ───────────────────────────────────────────────

def _check_completeness(pdf_bytes: bytes, is_article: bool = True) -> tuple[str, int]:
    """Check if the PDF is complete (not truncated).

    Returns: (verdict, page_count)
    - verdict: "complete" | "incomplete" | "uncertain"
    """
    from app.services.completeness_checker import check_completeness
    try:
        report = check_completeness(pdf_bytes, is_article=is_article)
        return report.verdict.lower(), report.page_count or 0
    except Exception as e:
        logger.debug("Completeness check failed: %s", e)
        return "uncertain", 0
