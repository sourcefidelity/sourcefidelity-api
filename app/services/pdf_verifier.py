"""PDF metadata verification for instructor uploads.

Extracts title, DOI, and year from a PDF and compares against
instructor-provided metadata, so that mismatched content never enters
the source repository.
"""

from difflib import SequenceMatcher
import logging
import re
from typing import NamedTuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Matches a DOI, optionally prefixed by "doi:", "doi.org/", or a full URL.
_DOI_PATTERN = re.compile(
    r'(?:doi\s*[:/]\s*|https?://(?:dx\.)?doi\.org/)?'
    r'(10\.\d{4,}/[^\s"\']+)',
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')
_ISBN_PATTERN = re.compile(
    r"(?:ISBN(?:-1[03])?\s*:?[\s-]*)?"
    r"((?:97[89][\s-]*)?(?:\d[\s-]*){8,11}[\dXx])"
)
_TITLE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "with",
}
_AUTHOR_STOPWORDS = {
    "and", "author", "authors", "by", "editor", "editors", "et", "al",
}


def extract_metadata_from_pdf(file_bytes: bytes) -> dict:
    """Extract bibliographic metadata from PDF bytes.

    Returns:
        dict with keys: doi, title, year, first_page_text
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        # Bound identity evidence to front matter. Books may have an image
        # cover, series/blurb leaf and blank verso before a page-four title.
        # Do not search the whole source, where cited works could match.
        text = ""
        for page in doc[:5]:
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

    isbn_candidates = {
        normalized
        for match in _ISBN_PATTERN.finditer(text[:6000])
        if len(normalized := _normalize_isbn(match.group(1))) in {10, 13}
    }

    return {
        "doi": doi,
        "title": title,
        "year": year,
        "isbn_candidates": sorted(isbn_candidates),
        "first_page_text": text[:6000],
    }


class TitleObservation(NamedTuple):
    """What the front page showed, and when it showed nothing, why.

    A bare `None` conflated several different situations: a scanned cover with
    no text layer, a uniformly typeset document where no span stands out, a
    page whose largest text is a structural label, and an unreadable file. They
    look identical to a caller but mean opposite things. Failing to read a page
    is not evidence about what is on it, while reading a page and finding no
    work title is.
    """

    title: str | None
    reason: str

    @property
    def observed(self) -> bool:
        return self.reason == "title_observed"

    @property
    def read_the_page(self) -> bool:
        """Did we get far enough to say anything about this document?"""
        return self.reason not in {"page_unreadable", "no_text_layer"}


# Structural labels are the largest text on many front pages. "PART I
# INTRODUCTION" is a section heading, not the title of a work.
_FRONT_MATTER_LABELS = frozenset({
    "part", "section", "chapter", "volume", "book", "introduction", "contents",
    "table", "preface", "foreword", "afterword", "appendix", "index", "abstract",
    "summary", "prologue", "epilogue", "acknowledgments", "acknowledgements",
    "notes", "bibliography", "glossary", "references", "i", "ii", "iii", "iv",
    "v", "vi", "vii", "viii", "ix", "x", "one", "two", "three", "four", "five",
})


def _is_front_matter_label(value: str) -> bool:
    words = re.findall(r"[^\W\d_]+", value.casefold())
    return bool(words) and all(word in _FRONT_MATTER_LABELS for word in words)


def _is_illegible(value: str) -> bool:
    """OCR of a stylised or handwritten cover returns characters, not words."""
    letters = sum(character.isalpha() for character in value)
    return letters < 0.6 * len(value)


def describe_first_page_title(
    file_bytes: bytes, *, page_index: int = 0
) -> TitleObservation:
    """Read the front page's title, or report why none could be read.

    Uses a font-size heuristic: the title is usually among the largest text
    spans. Spans at (or near) the max size are joined in reading order.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            page = doc[page_index]
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES).get("blocks", [])
            page_has_text = bool(page.get_text().strip())

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
                # An image-only cover was never read; a uniformly typeset page
                # was read and simply has no typographic title.
                return TitleObservation(
                    None, "no_text_layer" if not page_has_text else "uniform_typography"
                )

            # Collect all spans within ~1pt of the max size (titles often
            # span several lines at the same size). This is more robust
            # than requiring an exact match.
            threshold = max_size - 1.0
            title_spans = [t for size, t in spans if size >= threshold]
            title = " ".join(title_spans)[:300]
            if _is_front_matter_label(title):
                return TitleObservation(None, "front_matter_label_only")
            if _is_illegible(title):
                return TitleObservation(None, "illegible_title")
            return TitleObservation(title, "title_observed")
        finally:
            doc.close()
    except Exception as e:
        logger.warning("Title extraction failed (type=%s)", type(e).__name__)
        return TitleObservation(None, "page_unreadable")


def _extract_title_from_first_page(file_bytes: bytes, *, page_index: int = 0) -> str | None:
    """Back-compatible title accessor; use describe_first_page_title for the reason."""
    return describe_first_page_title(file_bytes, page_index=page_index).title


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase for fuzzy comparison."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _normalize_doi(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)", "", normalized)
    return normalized.rstrip(".,;)]>")


def _normalize_isbn(value: str) -> str:
    return "".join(char for char in value.upper() if char.isdigit() or char == "X")


def _significant_tokens(value: str, *, stopwords: set[str]) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(value))
        if len(token) > 1 and token not in stopwords
    ]


_TITLE_TAIL_BOUNDARY = re.compile(r"^\s*[-:;.,/?!|\u2013\u2014(\[]")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _aligned_title_containment(expected: str, found: str) -> bool:
    """True when one title extends the other at a subtitle boundary.

    A cited title routinely drops its subtitle, so "The studio system" and
    "The studio system: a history" are the same work. A cited title that merely
    occurs as a phrase inside a longer, differently headed title is not: bare
    containment matched Belton's chapter "The Studio System" against a thesis
    titled "Working Below the Line in the Studio System: Exploring Labour
    Processes in the UK Film Industry 1927-1950", and the report then offered a
    stranger's thesis as a possible match for the chapter.
    """
    expected = _LEADING_ARTICLE.sub("", expected)
    found = _LEADING_ARTICLE.sub("", found)
    if not expected or not found:
        return False
    if expected == found:
        return True
    longer, shorter = (found, expected) if len(found) > len(expected) else (expected, found)
    if not longer.startswith(shorter):
        return False
    return bool(_TITLE_TAIL_BOUNDARY.match(longer[len(shorter):]))


def _title_matches(provided: str, metadata: dict) -> bool:
    # A line-break hyphen kept inside a word ("automa-tion") is also tried
    # without it; a real hyphen still matches as written (2026-10-07).
    joined = re.sub(r"(?<=[a-z])-(?=[a-z])", "", provided)
    return _title_matches_as_written(provided, metadata) or (
        joined != provided and _title_matches_as_written(joined, metadata))


def _title_matches_as_written(provided: str, metadata: dict) -> bool:
    expected = _normalize(provided)
    found_title = _normalize(metadata.get("title") or "")
    page_text = _normalize(metadata.get("first_page_text") or "")
    if expected and (_aligned_title_containment(expected, found_title)
                     or expected in page_text):
        return True
    if found_title and SequenceMatcher(None, expected, found_title).ratio() >= 0.82:
        return True
    tokens = _significant_tokens(expected, stopwords=_TITLE_STOPWORDS)
    if len(tokens) < 3:
        return False
    found_tokens = set(_significant_tokens(found_title, stopwords=_TITLE_STOPWORDS))
    page_tokens = set(_significant_tokens(page_text[:2500], stopwords=_TITLE_STOPWORDS))
    coverage = len(set(tokens) & (found_tokens | page_tokens)) / len(set(tokens))
    return coverage >= 0.9


# APA bracketed descriptors after a web title: "[Online]", "[Web log post]".
_BRACKET_DESCRIPTOR = re.compile(r"\s*\[[^\]]{2,40}\]\s*")


def _web_title_near_match(provided: str, metadata: dict) -> bool:
    """A web page's printed heading often drops words the student added
    ("Star Trek: The Motion Picture movie review" vs the film title): accept
    similarity 0.75 or 80% word coverage once bracketed descriptors are removed.
    Only used with an author match (owner decision 2026-10-01)."""
    expected = _normalize(_BRACKET_DESCRIPTOR.sub(" ", provided))
    if not expected:
        return False
    found_title = _normalize(metadata.get("title") or "")
    page_text = _normalize(metadata.get("first_page_text") or "")
    if _aligned_title_containment(expected, found_title) or expected in page_text:
        return True
    if found_title and SequenceMatcher(None, expected, found_title).ratio() >= 0.75:
        return True
    tokens = set(_significant_tokens(expected, stopwords=_TITLE_STOPWORDS))
    if len(tokens) < 3:
        return False
    found = set(_significant_tokens(found_title, stopwords=_TITLE_STOPWORDS)) | set(
        _significant_tokens(page_text[:2500], stopwords=_TITLE_STOPWORDS))
    return len(tokens & found) / len(tokens) >= 0.8


def _author_matches(provided: str, metadata: dict) -> bool:
    expected = set(_significant_tokens(provided, stopwords=_AUTHOR_STOPWORDS))
    if not expected:
        return False
    page_tokens = set(
        _significant_tokens(
            str(metadata.get("first_page_text") or "")[:2500],
            stopwords=_AUTHOR_STOPWORDS,
        )
    )
    return bool(expected & page_tokens)


def verify_instructor_upload(
    file_bytes: bytes,
    provided_doi: str | None = None,
    provided_title: str | None = None,
    provided_author: str | None = None,
    provided_year: str | None = None,
    provided_isbn: str | None = None,
    web_page: bool = False,
) -> tuple[bool, list[str]]:
    """Verify that an instructor-uploaded PDF matches provided metadata.

    ``web_page`` (a reference typed as a web page) allows a near title match
    when the author also matches.

    Returns:
        (verified: bool, messages: list of human-readable status strings)
    """
    try:
        metadata = extract_metadata_from_pdf(file_bytes)
    except Exception:
        logger.warning("PDF identity extraction failed")
        return False, ["pdf_identity_extraction_failed"]
    messages: list[str] = []

    corroborated: set[str] = set()

    # Exact identifiers are authoritative when they are present in the file.
    if provided_doi:
        expected_doi = _normalize_doi(provided_doi)
        found_doi = _normalize_doi(metadata["doi"]) if metadata["doi"] else None
        if found_doi and expected_doi != found_doi:
            return False, ["doi_conflict"]
        if found_doi == expected_doi:
            corroborated.add("doi")
            messages.append("doi_match")

    if provided_isbn:
        expected_isbn = _normalize_isbn(provided_isbn)
        found_isbns = set(metadata.get("isbn_candidates") or [])
        if found_isbns and expected_isbn not in found_isbns:
            return False, ["isbn_conflict"]
        if expected_isbn in found_isbns:
            corroborated.add("isbn")
            messages.append("isbn_match")

    if provided_title:
        if _title_matches(provided_title, metadata):
            corroborated.add("title")
            messages.append("title_match")
        elif metadata.get("title"):
            messages.append("title_not_corroborated")

    if provided_author:
        if _author_matches(provided_author, metadata):
            corroborated.add("author")
            messages.append("author_match")
        else:
            messages.append("author_not_corroborated")

    if (web_page and provided_title and "title" not in corroborated and "author" in corroborated
            and _web_title_near_match(provided_title, metadata)):
        corroborated.add("title")
        messages.append("web_page_title_near_match")

    if provided_year:
        normalized_year = provided_year.strip()
        if metadata.get("year") == normalized_year:
            corroborated.add("year")
            messages.append("year_match")
        elif metadata.get("year"):
            messages.append("year_not_corroborated")

    # A matched persistent identifier is sufficient. Without one, require a
    # strong title match plus another identity-bearing field. A distinctive
    # title-only upload remains possible only when the normalized title is long
    # enough to be unlikely to identify an unrelated work accidentally.
    if corroborated & {"doi", "isbn"}:
        return True, messages
    if "title" in corroborated and corroborated & {"author", "year"}:
        return True, messages
    if "title" in corroborated:
        tokens = _significant_tokens(provided_title or "", stopwords=_TITLE_STOPWORDS)
        if len(set(tokens)) >= 6 and len(_normalize(provided_title or "")) >= 30:
            messages.append("distinctive_title_match")
            return True, messages
    messages.append("insufficient_identity_evidence")
    return False, messages
