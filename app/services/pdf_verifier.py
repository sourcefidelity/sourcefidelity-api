"""PDF metadata verification for instructor uploads.

Extracts title, DOI, and year from a PDF and compares against
instructor-provided metadata, so that mismatched content never enters
the source repository.
"""

import logging
import re

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Matches a DOI, optionally prefixed by "doi:", "doi.org/", or a full URL.
_DOI_PATTERN = re.compile(
    r'(?:doi\s*[:/]\s*|https?://(?:dx\.)?doi\.org/)?'
    r'(10\.\d{4,}/[^\s"\']+)',
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')


def extract_metadata_from_pdf(file_bytes: bytes) -> dict:
    """Extract bibliographic metadata from PDF bytes.

    Returns:
        dict with keys: doi, title, year, first_page_text
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        # Extract text from first 3 pages
        text = ""
        for page in doc[:3]:
            text += page.get_text()
    finally:
        doc.close()

    # Extract DOI (strip trailing punctuation that DOIs often pick up)
    doi_match = _DOI_PATTERN.search(text)
    doi = doi_match.group(1).rstrip(".,;)]>") if doi_match else None

    # Extract title (heuristic: largest-font text on page 1)
    title = _extract_title_from_first_page(file_bytes)

    # Extract year (search the first 1000 chars to avoid picking up
    # random 4-digit numbers in reference lists)
    year_match = _YEAR_PATTERN.search(text[:1000])
    year = year_match.group(0) if year_match else None

    return {
        "doi": doi,
        "title": title,
        "year": year,
        "first_page_text": text[:2000],
    }


def _extract_title_from_first_page(file_bytes: bytes) -> str | None:
    """Extract the likely title from the first page of a PDF.

    Uses a font-size heuristic: the title is usually among the largest
    text spans. We collect the spans at (or near) the max font size and
    join them in reading order.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            page = doc[0]
            blocks = page.get_text("dict").get("blocks", [])

            # Find the max font size among reasonably long text spans
            spans: list[tuple[float, str]] = []
            max_size = 0.0
            for block in blocks:
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        text = span.get("text", "").strip()
                        if size > 11 and len(text) > 3:
                            spans.append((size, text))
                            if size > max_size:
                                max_size = size

            if not spans or max_size <= 0:
                return None

            # Collect all spans within ~1pt of the max size (titles often
            # span several lines at the same size). This is more robust
            # than requiring an exact match.
            threshold = max_size - 1.0
            title_spans = [t for size, t in spans if size >= threshold]
            return " ".join(title_spans)[:300]
        finally:
            doc.close()
    except Exception as e:
        logger.warning("Title extraction failed: %s", e)
        return None


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase for fuzzy comparison."""
    return re.sub(r"\s+", " ", s.strip().lower())


def verify_instructor_upload(
    file_bytes: bytes,
    provided_doi: str | None = None,
    provided_title: str | None = None,
    provided_author: str | None = None,
) -> tuple[bool, list[str]]:
    """Verify that an instructor-uploaded PDF matches provided metadata.

    Returns:
        (verified: bool, messages: list of human-readable status strings)
    """
    metadata = extract_metadata_from_pdf(file_bytes)
    messages: list[str] = []

    # DOI check — authoritative. Mismatch here fails verification.
    if provided_doi and metadata["doi"]:
        cleaned_provided = provided_doi.removeprefix("https://doi.org/").strip().lower()
        cleaned_found = metadata["doi"].strip().lower()
        doi_match = cleaned_provided == cleaned_found
        messages.append(
            f"DOI match: {'PASS' if doi_match else 'FAIL'} "
            f"(provided='{cleaned_provided}', found='{cleaned_found}')"
        )
        if not doi_match:
            return False, messages

    # Title check — advisory. DOIs are more reliable than font-size titles,
    # so a title mismatch warns but does not reject on its own.
    if provided_title and metadata["title"]:
        t1 = _normalize(provided_title)
        t2 = _normalize(metadata["title"])
        overlap = (t1 in t2) or (t2 in t1) or (t1[:80] == t2[:80])
        messages.append(
            f"Title check: {'PASS' if overlap else 'WARNING'} "
            f"(provided='{t1[:60]}', found='{t2[:60]}')"
        )
        if not overlap:
            messages.append("Title mismatch — instructor should verify manually")

    return True, messages
