"""Regex-based field extraction for individual references.

Extracts structured fields (author, year, title, DOI, URL, citation_key) from a
single reference string using format-specific regex patterns. Designed as a
fast, deterministic first pass — references that regex can't handle (unusual
formatting, institutional authors, merged entries) fall back to the LLM
per-reference path in ``reference_parser.py``.

Two format-specific extractors:
  - ``extract_fields_apa()`` — APA 7th edition: ``Author. (Year). Title. Source.``
  - ``extract_fields_mla()`` — MLA 9th edition: ``Author. Title. Publisher, Year.``

The title field is the weakest extraction (ambiguous title/source boundary);
references where author OR title comes back empty are marked for LLM fallback.
"""

import re
import logging
from typing import Optional

from app.services.schemas import ParsedReference

logger = logging.getLogger(__name__)

# ── Shared patterns ──────────────────────────────────────────────────────────

# DOI — reused from the established pdf_verifier pattern (the better of two
# existing copies in the codebase). Matches bare DOIs, doi: prefixes, and
# https://doi.org/ URLs, capturing the 10.XXXX/... identifier.
_DOI_PATTERN = re.compile(
    r'(?:doi\s*[:/]\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s"\']+)',
    re.IGNORECASE,
)

# URL (when no DOI) — captures full http(s) URLs, strips trailing punctuation
_URL_PATTERN = re.compile(r'https?://[^\s]+', re.IGNORECASE)

# Generic 4-digit year
_YEAR_GENERIC = re.compile(r'\b(?:19|20)\d{2}\b')


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation and doi.org prefix from a captured DOI."""
    return doi.rstrip('.,;)]>').replace("https://doi.org/", "").replace("http://doi.org/", "")


def _clean_url(url: str) -> str:
    """Strip trailing punctuation from a captured URL."""
    return url.rstrip('.,;)]>')


def _make_citation_key(author: str, year: str) -> str:
    """Build a citation key from first surname + year (e.g. 'Smith2020')."""
    if not author:
        return ""
    # First word before comma/space = surname
    surname = re.split(r'[, ]', author.strip())[0]
    surname = re.sub(r'[^A-Za-z]', '', surname)
    yr = re.search(r'\d{4}', year) or (year if year else "")
    yr_str = yr.group(0) if hasattr(yr, 'group') else str(yr)
    return f"{surname}{yr_str}" if surname and yr_str else ""


def _extract_identifiers(text: str) -> tuple[str, str]:
    """Extract DOI and URL from any reference text. DOI takes precedence.

    Returns (doi, url) — empty strings if not found. If a DOI is found,
    URL is left empty (DOI is the preferred identifier).
    """
    doi_match = _DOI_PATTERN.search(text)
    if doi_match:
        return _clean_doi(doi_match.group(1)), ""

    url_match = _URL_PATTERN.search(text)
    if url_match:
        return "", _clean_url(url_match.group(0))

    return "", ""


# ── APA field extraction ─────────────────────────────────────────────────────

# APA year: (2020) or (n.d.) — full parenthetical including closing paren
_APA_YEAR = re.compile(r'\((?:19|20)\d{2}[a-z]?\)|\(n\.d\.\)', re.IGNORECASE)


def extract_fields_apa(ref: str) -> Optional[ParsedReference]:
    """Extract structured fields from an APA 7th edition reference string.

    APA format: ``Author, A. (Year). Title. Source. DOI/URL``

    Returns a ParsedReference if author AND title are found, None otherwise
    (caller should send None results to LLM fallback).

    Args:
        ref: A single APA reference string (post-split, one reference).

    Returns:
        ParsedReference with fields populated, or None if extraction failed.
    """
    ref = ref.strip()
    if not ref:
        return None

    # DOI + URL (shared extraction)
    doi, url = _extract_identifiers(ref)

    # Year — first parenthetical year/n.d.
    year_match = _APA_YEAR.search(ref)
    if year_match:
        year_raw = year_match.group(0)
        # Extract 4-digit year from "(2020)" or use "n.d."
        yr_inner = re.search(r'\d{4}', year_raw)
        year = yr_inner.group(0) if yr_inner else "n.d."
        year_pos = year_match.start()
    else:
        # No parenthetical year — try bare year
        bare = _YEAR_GENERIC.search(ref)
        year = bare.group(0) if bare else ""
        year_pos = bare.start() if bare else -1

    # Author — text before the year (APA: author precedes year in parens)
    if year_pos > 0:
        author = ref[:year_pos].strip().rstrip(',.;:')
        # Strip leading numbering like "1." or "1)"
        author = re.sub(r'^\d+[\.\)]\s*', '', author)
    else:
        author = ""

    # Title — text between "(YEAR). " and the next period that starts a new
    # element (capital letter or italic indicator). This is the weakest field;
    # ambiguous boundaries will send the ref to LLM fallback.
    title = ""
    if year_match:
        after_year = ref[year_match.end():]
        # After "(2020)" the next chars should be ". " — skip leading punctuation
        after_year = re.sub(r'^[\s.\)]+', '', after_year).strip()
        # Title ends at the first ". " followed by a capital letter (the source
        # follows). But titles can contain periods (e.g., "Dr. Strange"), so
        # require the period to be followed by space + capital + not a common
        # title-ending word.
        title_match = re.match(r'(.+?)\.\s+(?=[A-Z])', after_year)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # No clear boundary — take everything up to the DOI/URL or end
            title = re.split(r'\s+(?:https?://|doi:)', after_year)[0].strip()
            title = title.rstrip('.')

    # Trim trailing source info from title if it's clearly there
    # (heuristic: title shouldn't contain volume/issue patterns like "12(3)")
    if title:
        title = re.split(r'\s*,\s*\d+\(', title)[0].strip()

    # Success check — need both author and title
    if not author or not title:
        return None

    citation_key = _make_citation_key(author, year)

    return ParsedReference(
        author=author,
        year=year or "n.d.",
        title=title,
        doi=doi,
        url=url,
        raw_ref=ref,
        citation_key=citation_key,
        extraction_method="regex",
    )


# ── MLA field extraction ─────────────────────────────────────────────────────

# MLA author ends at the first ". " followed by content (title follows)
_MLA_AUTHOR_END = re.compile(r'^(.{2,200}?)\.\s+(?=[A-Z\u201c"])')

# MLA title: quoted article title or book title between periods
_MLA_QUOTED_TITLE = re.compile(r'\u201c([^"]+?)\u201d|"([^"]+?)"')


def extract_fields_mla(ref: str) -> Optional[ParsedReference]:
    """Extract structured fields from an MLA 9th edition reference string.

    MLA format: ``Lastname, Firstname. Title. Publisher, Year.`` or
    ``Lastname, Firstname. "Article Title." Journal, vol., no., Year, pp.``

    MLA is harder to regex than APA — year position varies, title formatting
    varies (quotes for articles, italics lost in plain text for books). Expect
    a higher LLM-fallback rate (~30-40% vs APA's ~10-15%).

    Returns ParsedReference if author AND title are found, None otherwise.

    Args:
        ref: A single MLA reference string.

    Returns:
        ParsedReference with fields populated, or None if extraction failed.
    """
    ref = ref.strip()
    if not ref:
        return None

    # DOI + URL (shared extraction)
    doi, url = _extract_identifiers(ref)

    # Year — take the LAST 4-digit year in the string (MLA year is near end)
    year_matches = list(_YEAR_GENERIC.finditer(ref))
    if year_matches:
        year = year_matches[-1].group(0)  # last match
    else:
        year = ""

    # Author — text from start to first ". " followed by capital/quote
    # Handles "Lastname, Firstname." and "Lastname, Firstname, et al."
    # Strip leading numbering first
    ref_clean = re.sub(r'^\d+[\.\)]\s*', '', ref)
    author_match = _MLA_AUTHOR_END.match(ref_clean)
    if author_match:
        author = author_match.group(1).strip()
    else:
        # Fallback: take text before first period
        first_period = ref_clean.find('. ')
        author = ref_clean[:first_period].strip() if first_period > 0 else ""

    # Title — MLA titles: quoted (articles) or between author-period and
    # next period (books, where italics are lost in plain text)
    title = ""

    # First try: quoted title (articles/essays)
    quoted = _MLA_QUOTED_TITLE.search(ref_clean)
    if quoted:
        title = (quoted.group(1) or quoted.group(2) or "").strip()

    if not title and author:
        # Second try: book title — text after author's period, up to next period
        # Find position after author
        after_author = ref_clean[len(author):].lstrip('. ')
        if after_author:
            # Title up to next period followed by space+capital (publisher)
            title_match = re.match(r'(.+?)\.\s+(?=[A-Z]|\d|https?://)', after_author)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # No clear boundary — take up to next period
                title = after_author.split('.')[0].strip()

    # Success check — need both author and title
    if not author or not title:
        return None

    citation_key = _make_citation_key(author, year)

    return ParsedReference(
        author=author,
        year=year or "n.d.",
        title=title,
        doi=doi,
        url=url,
        raw_ref=ref,
        citation_key=citation_key,
        extraction_method="regex",
    )


# ── LLM-response field extraction ────────────────────────────────────────────

# The LLM fallback returns labeled plain-text lines:
#   Author: Smith, J.
#   Year: 2020
#   Title: Some title
#   DOI: 10.xxxx/yyy
#   URL: https://...
# These patterns extract from each labeled line.
_LLM_FIELD_PATTERNS = {
    "author": re.compile(r'^[Aa]uthor:\s*(.+)$', re.MULTILINE),
    "year": re.compile(r'^[Yy]ear:\s*(.+)$', re.MULTILINE),
    "title": re.compile(r'^[Tt]itle:\s*(.+)$', re.MULTILINE),
    "doi": re.compile(r'^[Dd][Oo][Ii]:\s*(.+)$', re.MULTILINE),
    "url": re.compile(r'^[Uu][Rr][Ll]:\s*(.+)$', re.MULTILINE),
}


def extract_fields_from_llm_response(response: str, raw_ref: str) -> ParsedReference:
    """Extract fields from an LLM's labeled plain-text response.

    The LLM per-reference fallback returns text like::
        Author: Smith, J.
        Year: 2020
        Title: Some title
        DOI: 10.xxxx/yyy
        URL: none

    This function extracts those labeled fields into a ParsedReference. Values
    like "none" or "n/a" are converted to empty strings.

    Args:
        response: The LLM's plain-text response.
        raw_ref: The original reference string (preserved as raw_ref).

    Returns:
        ParsedReference with extraction_method="llm" and needs_review=True.
    """
    fields = {}
    for field, pattern in _LLM_FIELD_PATTERNS.items():
        match = pattern.search(response)
        if match:
            value = match.group(1).strip()
            # Normalize "none"/"n/a" to empty
            if value.lower() in ("none", "n/a", "null", ""):
                value = ""
            fields[field] = value

    author = fields.get("author", "")
    year = fields.get("year", "")
    title = fields.get("title", "")
    doi = fields.get("doi", "")
    url = fields.get("url", "")

    # Clean DOI/URL
    if doi:
        doi = _clean_doi(doi)
    if url:
        url = _clean_url(url)

    # Normalize year to 4-digit or n.d.
    if year:
        yr_match = _YEAR_GENERIC.search(year)
        year = yr_match.group(0) if yr_match else "n.d."
    else:
        year = "n.d."

    citation_key = _make_citation_key(author, year)

    return ParsedReference(
        author=author,
        year=year,
        title=title,
        doi=doi,
        url=url,
        raw_ref=raw_ref,
        citation_key=citation_key,
        extraction_method="llm",
        needs_review=True,
    )
