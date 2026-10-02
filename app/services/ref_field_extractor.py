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
import unicodedata
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
_TRAILING_PAGE_RANGE = re.compile(
    r"\s*\((?:pp?\.?|pages?)\s*\d{1,5}\s*[-–—]\s*\d{1,5}\)\s*$",
    re.IGNORECASE,
)


def decode_doi(doi: str) -> str:
    """A DOI copied from a link arrives percent-encoded (``%28`` for ``(``);
    decode it once so the resolver and the DOI link see the registered form."""
    from urllib.parse import unquote
    decoded = unquote(doi) if "%" in doi else doi
    return decoded if decoded.startswith("10.") and not any(c.isspace() for c in decoded) else doi


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation and doi.org prefix from a captured DOI."""
    doi = decode_doi(doi.replace("https://doi.org/", "").replace("http://doi.org/", "").rstrip('.,;]>'))
    # A closing parenthesis belongs to the DOI only when it closes one inside it.
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip('.,;]>')
    return doi


def _clean_url(url: str) -> str:
    """Strip trailing punctuation from a captured URL."""
    return url.rstrip('.,;)]>')


def _make_citation_key(author: str, year: str) -> str:
    """Build a display alias, preserving year suffixes such as ``2020b``."""
    if not author:
        return ""
    # First word before comma/space = surname
    surname = re.split(r'[, ]', author.strip())[0]
    surname = re.sub(
        r'[^A-Za-z]',
        '',
        unicodedata.normalize("NFKD", surname),
    )
    yr = re.search(r'\d{4}[a-z]?', year, re.IGNORECASE) or (year if year else "")
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
_APA_YEAR = re.compile(
    r'\((?:19|20)\d{2}[a-z]?(?:\s*[-–—]\s*(?:19|20)\d{2})?(?:,\s*[^)]{1,60}|'
    r'\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+(?:[1-9]|[12]\d|3[01]))?)?\)|\(n\.d\.\)',
    re.IGNORECASE,
)


def _explicit_source_boundary(text: str, index: int) -> bool:
    """Recognize structural source cues missed by the whitespace boundary.

    Literal emphasis markers are not always stripped from submitted text.
    Require the following journal volume/pages, not just an emphasized phrase.
    A joined publisher must be the terminal element and use the existing
    publisher vocabulary; do not guess arbitrary missing sentence spaces.
    """
    tail = text[index + 1:]
    from app.services.source_type import _BOOK_PUBLISHER_RE, _MARKED_JOURNAL_STRUCTURE_RE
    if tail[:1].isspace() and _MARKED_JOURNAL_STRUCTURE_RE.match(tail.lstrip()):
        return True
    if (re.match(r'[A-Z]', tail) and re.search(r'[a-z]{2,}$', text[:index])
            and len(text[:index].split()) >= 2):
        return bool(_BOOK_PUBLISHER_RE.fullmatch(tail.rstrip('. ')))
    return False


# A journal prints a book review's title as the reviewed work followed by its
# author, imprint and whole-work extent. That entire span is the review
# article's title, not a separate publisher element.
_APA_REVIEW_HEADER_RE = re.compile(
    r'\.\s+'
    r'(?P<attribution>[^.\n]{2,160})\.\s+'
    r'(?P<imprint>[^.\n]{2,160}),\s*(?:1[5-9]|20)\d{2}\.\s*'
    r'Pp\.\s*(?:[ivxlcdm]{1,12}\s*\+\s*)?\d{1,5}\.'
)


def _review_header_end(text: str, index: int) -> int | None:
    """Extend a review-header title across its reviewed-work description.

    Require every structural element before extending: the reviewed work's
    attribution, an imprint using the existing publisher vocabulary with its
    own year, a whole-work extent statement (an extent, never a page range),
    and a following journal container. An ordinary title followed by a
    publisher element has no extent statement and no journal container, so it
    is unaffected.
    """
    match = _APA_REVIEW_HEADER_RE.match(text, index)
    if not match:
        return None
    from app.services.source_type import _BOOK_PUBLISHER_RE, _JOURNAL_STRUCTURE_RE
    if not _BOOK_PUBLISHER_RE.search(match['imprint']):
        return None
    if not _JOURNAL_STRUCTURE_RE.search(text[match.end():]):
        return None
    return match.end()


def _apa_title(after_year: str) -> str:
    """Find a source boundary outside balanced parenthetical title material."""
    text = after_year.strip()
    # Literal Markdown-style delimiters occur in submitted plain-text
    # references. They delimit the field, not observed italic typography.
    marked = re.match(r'^\*([^*\n]+)\*(.*)$', text)
    if marked and (not marked.group(2).strip() or marked.group(2).startswith('.')
                   or re.search(r'[.!?]$', marked.group(1))):
        return marked.group(1).rstrip('.')
    depth = 0
    start = 0
    groups = []
    end = len(text)
    for index, char in enumerate(text):
        if char == '(':
            if depth == 0:
                start = index
            depth += 1
        elif char == ')':
            if depth == 0:
                return ''
            depth -= 1
            if depth == 0:
                groups.append((start, index + 1))
        elif char == '.' and depth == 0 and (
            re.match(r'\s+[A-Z]', text[index + 1:]) or _explicit_source_boundary(text, index)
        ):
            # A personal-name initial inside a title is not a source boundary
            # (e.g. "Pearl S. Buck" or "David O. Selznick").
            if (re.search(r'\b[A-Z][a-z]{1,30}\s+(?:[A-Z]\.\s*)*[A-Z]$', text[:index])
                    and re.match(r'\s+[A-Z][a-z]{1,30}\b', text[index+1:])):
                continue
            extended = _review_header_end(text, index)
            end = index if extended is None else extended
            break
        elif char in '?!' and depth == 0 and re.match(
            r'\s+[A-Z][^.!?\n]{1,180},\s*\d{1,4}\s*(?:\([^)]{1,30}\))?\s*[,.;]',
            text[index + 1:],
        ):
            # A terminal question/exclamation mark can separate an article
            # from its journal/volume element without an extra period. Require
            # that structural source cue: a question followed by a subtitle
            # is not, by itself, a safe title boundary. Preserve the mark.
            end = index + 1
            break
    if depth:
        return ''
    title = text[:end].strip()
    if end == len(text):
        # Preserve the existing no-boundary fallback without changing how
        # unrelated merged/ambiguous entries are handled in this repair.
        title = re.split(r'\s+(?:https?://|doi:)', title)[0].strip()
    title = title.rstrip('.')
    # Contributor credits describe the edition, not the work's title. Keep
    # their exact spelling in raw_ref; do not remove ordinary subtitles or
    # edition parentheses such as "(2nd ed.)".
    for left, right in reversed(groups):
        if right != len(title):
            break
        credit = title[left + 1:right - 1]
        if not re.fullmatch(r'.+(?:,\s*|\s+\()(?:Eds?|Trans)\.\)?', credit, re.IGNORECASE):
            break
        title = title[:left].rstrip()
    # A thesis descriptor, "(Bachelor's thesis, University)" in APA 6 or
    # "[Doctoral dissertation, University]" in APA 7, describes the work; it is
    # not part of the title. raw_ref keeps it.
    return _THESIS_DESCRIPTOR.sub("", title).rstrip() or title


_THESIS_DESCRIPTOR = re.compile(
    r"\s*[(\[](?:(?:unpublished\s+)?(?:bachelor|master|honou?rs|doctoral|ph\.?\s?d\.?|mphil|undergraduate)"
    r"(?:['’]s)?\s+)?(?:thesis|dissertation)\b[^()\[\]]*[)\]]$", re.IGNORECASE)


def extract_authorless_apa_journal(ref: str) -> Optional[ParsedReference]:
    """A complete date/title/journal structure can be parsed without an author.

    This says where the fields are, not that the work actually has an author.
    Incomplete titles/books and contributor-credit entries use review fallback.
    """
    text = ref.strip()
    date = _APA_YEAR.match(text)
    if not date or not re.match(r'^\((?:19|20)\d{2}[a-z]?\)\.\s+', text, re.I):
        return None
    remainder = text[date.end():].lstrip('. ')
    title = _apa_title(remainder)
    if len(title.split()) < 4 or not remainder.startswith(title):
        return None
    tail = remainder[len(title):].lstrip('. ')
    if not re.fullmatch(
        r'[^\n.!?]{2,160},\s*\d{1,4}\s*\(\d{1,4}\)\s*,\s*'
        r'\d{1,5}\s*[-–]\s*\d{1,5}\.\s*https?://\S+', tail):
        return None
    doi, url = _extract_identifiers(text)
    return ParsedReference(raw_ref=ref, title=title, author='',
        year=date.group()[1:-1].lower(), doi=doi, url=url,
        needs_review=False, extraction_method='authorless_journal_regex')


def extract_authorless_apa_fields(ref: str) -> Optional[ParsedReference]:
    """Retain visible fields in an authorless entry without proving omission."""
    from app.services.source_type import _BOOK_PUBLISHER_RE, _JOURNAL_STRUCTURE_RE
    text = ref.strip()
    date = _APA_YEAR.match(text)
    remainder = text[date.end():].lstrip('. ') if date else text
    if not date and re.match(r'^[^,]{1,80},\s*[A-Z]\.', remainder):
        return None
    title = _apa_title(remainder)
    if len(title.split()) < 4 or not remainder.startswith(title):
        return None
    tail = remainder[len(title):].lstrip('. ')
    if not (_JOURNAL_STRUCTURE_RE.search(tail) or _BOOK_PUBLISHER_RE.search(tail)):
        return None
    doi, url = _extract_identifiers(text)
    if not (doi or url):
        return None
    year = re.search(r'\d{4}[a-z]?', date.group(), re.I) if date else None
    return ParsedReference(raw_ref=ref, title=title, author='',
        year=year.group().lower() if year else 'n.d.', doi=doi, url=url,
        needs_review=True, extraction_method='partial_regex')


# "In J. Belton (Ed.), American cinema/American culture (pp. 64-86)."
# The containing work is the part of a miscited chapter reference a student
# usually gets right. Discarding it left discovery searching only for the
# chapter title, which in Belton's case named no work that exists, so the
# merge attached whichever record shared those two words.
_CONTAINER_IN_RE = re.compile(
    r'\bIn\s+[^()]{2,120}?\(\s*Eds?\.?\s*\)\s*,\s*(?P<container>[^()]{3,300}?)\s*'
    r'(?:\(\s*pp?\.\s*(?P<pages>[\dixvlc]{1,6}\s*[-–—]\s*[\dixvlc]{1,6})\s*\)|[.,])',
    re.IGNORECASE)


# "Chapter title. In Book title (pp. 75-116). Publisher." A student often
# drops the editors; the page range after the book title still marks a part of
# a book (Franchise 1, 2026-09-29: read as a whole book, the chapter title was
# searched as a book and the containing book was never looked up).
_CONTAINER_IN_NO_EDITOR_RE = re.compile(
    r'\.\s+In\s+(?P<container>[^()]{3,300}?)\s*'
    r'\(\s*pp?\.\s*(?P<pages>[\dixvlc]{1,6}\s*[-–—]\s*[\dixvlc]{1,6})\s*\)',
    re.IGNORECASE)


# "Chapter title. In Book title (A. Name & B. Name, Eds.) (pp. 5-20)." The
# editors follow the book title inside its parentheses; pages are optional.
_CONTAINER_IN_EDITORS_AFTER_RE = re.compile(
    r'\.\s+In\s+(?P<container>[^()]{3,300}?)\s*\([^()]{2,160}?,\s*Eds?\.\s*\)'
    r'(?:\s*\(\s*pp?\.\s*(?P<pages>[\dixvlc]{1,6}\s*[-–—]\s*[\dixvlc]{1,6})\s*\))?',
    re.IGNORECASE)


# "… article title. Journal Name, 40(2), 125-139." The journal is the other
# half of the identity combination for an article, and it was never extracted:
# `_CONTAINER_IN_RE` recognises only the edited-collection shape, so all 155
# journal articles in the development corpus carried an empty container and a
# wrong journal could not be compared, let alone reported. The volume, or the
# issue in parentheses, is the anchor: a bare trailing number is not enough to
# call the text before it a journal.
_JOURNAL_RE = re.compile(
    r'(?:^|(?<=\.)|(?<=\?)|(?<=!))\s*(?P<container>[^\W\d_][^.\n]{2,120}?)\s*,?\s*'
    r'(?P<volume>\d{1,4})\s*'
    r'(?:\(\s*(?P<issue>[^)\n]{1,24})\s*\))?\s*'
    r'(?:,\s*(?P<pages>[A-Za-z]?\d{1,6}\s*[-\u2013\u2014]\s*[A-Za-z]?\d{1,6}|e\d{3,8}))?',
    re.UNICODE)
# Emphasis marks survive docx conversion around "*Journal, 131*(5)" and would
# otherwise split the journal from its volume.
_EMPHASIS = re.compile(r'[*_]+')


def _journal_parts(ref: str) -> tuple[str, str, str, str]:
    """Recover journal, volume, issue and page range from an article."""
    text = _EMPHASIS.sub('', ref or '')
    match = None
    for candidate in _JOURNAL_RE.finditer(text):
        # Require an issue or a page range; the last such element in the
        # reference is the periodical, not a number inside the title.
        if candidate.group('issue') or candidate.group('pages'):
            match = candidate
    if not match:
        return '', '', '', ''
    container = re.sub(r'\s+', ' ', match.group('container') or '').strip(' .,')
    pages = re.sub(r'\s*[-\u2013\u2014]\s*', '-', (match.group('pages') or '').strip())
    volume = (match.group('volume') or '').strip()
    issue = re.sub(r'\s+', ' ', (match.group('issue') or '')).strip(' .,')
    return container, volume, issue, pages


def _journal_and_pages(ref: str) -> tuple[str, str]:
    """Recover the journal and page range from an article-shaped reference."""
    container, _volume, _issue, pages = _journal_parts(ref)
    return container, pages


# "… (pp. 64-86). New York: McGraw Hill." or "… (pp. 1-10). Routledge."
# APA 7 drops the place; APA 6 and many student references keep it.
_PUBLISHER_TAIL_RE = re.compile(
    r'(?:\)|\.)\s*(?:(?P<place>[A-Z][A-Za-z.\s]{2,40}?)\s*:\s*)?'
    r'(?P<publisher>[A-Z][^.]{2,80}?)\s*\.\s*$')
_NOT_A_PUBLISHER = re.compile(
    r'https?://|\bdoi\b|\bpp?\.|\bretrieved\b|\bvol\b|\d{4}', re.IGNORECASE)


def _publisher(ref: str, source_kind: str) -> str:
    """Take the trailing publisher element of a book-shaped reference.

    Classification cannot gate this: a book is often typed `unknown` precisely
    because its publisher has not been recognised yet. The element is accepted
    on structure instead — known imprint vocabulary, a chapter's page range or
    place prefix, or a final element that is not the title, which is what
    distinguishes "… itself. Basic Books." from "… (2019). An article title."
    """
    text = (ref or '').strip()
    match = _PUBLISHER_TAIL_RE.search(text)
    if not match:
        return ''
    publisher = re.sub(r'\s+', ' ', match.group('publisher')).strip(' .,')
    if not publisher or _NOT_A_PUBLISHER.search(publisher):
        return ''
    from app.services.source_type import _BOOK_PUBLISHER_RE
    year = _APA_YEAR.search(text)
    between = text[year.end():match.start()] if year else ''
    accepted = (
        bool(_BOOK_PUBLISHER_RE.search(publisher))
        or bool(match.group('place'))
        or bool(_CONTAINER_IN_RE.search(text)) or bool(_CONTAINER_IN_NO_EDITOR_RE.search(text))
        or bool(re.search(r'\.\s', between))  # the title stands between year and imprint
    )
    return publisher if accepted else ''


def _container_and_pages(ref: str) -> tuple[str, str]:
    """Recover the containing work and page range from a reference.

    The chapter shape is checked first because it is the more specific: an
    edited collection carries its own page range, and its container is named
    after an explicit "In ... (Ed.),". An article names its journal instead.
    """
    match = (_CONTAINER_IN_RE.search(ref or '') or _CONTAINER_IN_NO_EDITOR_RE.search(ref or '')
             or _CONTAINER_IN_EDITORS_AFTER_RE.search(ref or ''))
    if not match:
        return _journal_and_pages(ref)
    container = re.sub(r'\s+', ' ', match.group('container') or '').strip(' .,')
    pages = re.sub(r'\s*[-–—]\s*', '-', (match.group('pages') or '').strip())
    return container, pages


def extract_incomplete_apa_chapter(ref: str) -> Optional[ParsedReference]:
    """Preserve explicit title/container/pages without inventing author or date.

    Review-only recovery after ordinary parsing failed. This is not a complete
    title assertion or a finding that missing metadata exists in the source.
    """
    match = re.fullmatch(
        r'(?P<title>[^\n]{12,500})\.\s+In\s+(?P<container>[^\n]{12,700}?)\s*'
        r'\(pp\.\s*(?P<pages>\d{1,5}\s*[-–]\s*\d{1,5})\)\.\s+'
        r'[^\n]{3,250}', ref.strip())
    if not match or _APA_YEAR.search(match['title']):
        return None
    if re.search(r'\b(?:https?://|www\.)|\bIn\s|[()]', match['title']):
        return None
    # Do not absorb a malformed leading author into a proposed title.
    if re.match(r'^[^,]{1,80},\s*[A-Z]\.', match['title']):
        return None
    doi, url = _extract_identifiers(ref)
    return ParsedReference(raw_ref=ref, title=match['title'], author='', year='n.d.',
        container_title=match['container'], pages=match['pages'], doi=doi, url=url,
        source_kind='book_section', source_kind_confidence='medium',
        source_kind_evidence=['explicit incomplete title/container/page structure'],
        needs_review=True, extraction_method='partial_regex')


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
    # A title-led pressbook entry can put the studio in both a parenthetical
    # credit and the representation label. Do not parse its title as author.
    pressbook = re.match(
        r'^(?P<title>[^\n()]{3,250}?)\s*\((?P<studio>[^()]{2,100})\)\.\s*'
        r'\((?P<year>(?:19|20)\d{2})\)\.\s*(?P<credit>[^\n]{2,100}?)\s+Pressbook\.', ref, re.I)
    if pressbook and re.sub(r'\W', '', pressbook['studio']).casefold() == re.sub(r'\W', '', pressbook['credit']).casefold():
        return ParsedReference(raw_ref=ref, title=pressbook['title'].strip(),
            author=pressbook['studio'], year=pressbook['year'], doi=doi, url=url,
            citation_key=_make_citation_key(pressbook['studio'], pressbook['year']),
            extraction_method='regex', source_kind='report', source_kind_confidence='high',
            source_kind_evidence=['explicit pressbook label and repeated studio credit'])
    year_match = _APA_YEAR.search(ref)
    if year_match:
        year_raw = year_match.group(0)
        # Preserve disambiguation suffixes from "(2020a)" / "(2020b)".
        yr_inner = re.search(r'\d{4}[a-z]?(?:\s*[-–—]\s*\d{4})?', year_raw, re.IGNORECASE)
        year = yr_inner.group(0).lower() if yr_inner else "n.d."
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
        author = re.sub(r'^[-•]\s+', '', author)
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
        title = _apa_title(after_year)

    # Trim trailing source info from title if it's clearly there
    # (heuristic: title shouldn't contain volume/issue patterns like "12(3)")
    if title:
        title = re.split(r'\s*,\s*\d+\(', title)[0].strip()
        # Preserve a cited component locator in raw_ref, but keep it out of
        # canonical title identity and retrieval queries.
        title = _TRAILING_PAGE_RANGE.sub("", title).strip()

    # Success check — need both author and title
    if not author or not title:
        return None

    citation_key = _make_citation_key(author, year)
    container_title, pages = _container_and_pages(ref)
    # Volume and issue belong to the periodical shape. When the chapter route
    # supplied the container, the periodical numbers are not this reference's,
    # so one shape's numbers are never attached to the other's title.
    journal, volume, issue, _ = _journal_parts(ref)
    if container_title and container_title != journal:
        volume = issue = ''
    # An article is published in a periodical, not by a publisher. The
    # trailing-element rule that finds a book's imprint reads the journal
    # name instead, which put a false publisher on 59 of 179 articles and
    # fed bogus publisher agreement into the merge score.
    periodical = bool(journal and volume)
    from app.services.source_type import classify_reference_source_kind
    kind = classify_reference_source_kind(ref, title=title, url=url).kind

    return ParsedReference(
        author=author,
        year=year or "n.d.",
        title=title,
        doi=doi,
        url=url,
        raw_ref=ref,
        citation_key=citation_key,
        container_title=container_title,
        volume=volume,
        issue=issue,
        pages=pages,
        publisher='' if periodical else _publisher(ref, kind),
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

    # Normalize year to four digits plus an optional disambiguation suffix.
    if year:
        yr_match = re.search(r'\b(?:19|20)\d{2}[a-z]?\b', year, re.IGNORECASE)
        year = yr_match.group(0).lower() if yr_match else "n.d."
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
