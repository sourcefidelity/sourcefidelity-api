"""Observed webpage identity fields; never fill them from the citation."""

import hashlib
import re
from datetime import datetime
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import trafilatura


_SITE_SUFFIX_SEPARATORS = (' | ', ' - ', ' \u2013 ', ' \u2014 ')
_ISSUE_LABEL_RE = re.compile(
    r':\s*(?:Vol(?:ume)?|No|Number|Issue|Iss)\b.*', re.IGNORECASE,
)


def _strip_declared_site_suffix(title: str, declared_names: list[str]) -> str:
    """Remove a trailing site/journal name the page declares about itself.

    The suffix must repeat a declared name exactly and must either end the
    title or be followed by that name's issue label. A title that merely
    contains such a name keeps it.
    """
    for name in declared_names:
        name = name.strip()
        if not name:
            continue
        for separator in _SITE_SUFFIX_SEPARATORS:
            index = title.rfind(separator + name)
            if index <= 0:
                continue
            remainder = title[index + len(separator) + len(name):]
            if remainder and not _ISSUE_LABEL_RE.fullmatch(remainder):
                continue
            return title[:index].strip()
    return title


# A page can be fetched cleanly and still say nothing about what work it holds.
# "We never reached it", "it has no title element", "its title is site chrome"
# and "it is a contents listing" are four different facts that all used to
# arrive as an empty title. See TitleObservation in pdf_verifier for the PDF
# equivalent; the vocabulary is deliberately parallel.
_BOILERPLATE_TITLE_RE = re.compile(
    r"^\s*(?:just a moment|checking your browser|access denied|forbidden|"
    r"page not found|not found|404|error|sign in|log in|login|register|"
    r"search results?|results|home ?page|home|redirecting|loading|untitled|"
    r"document|index of /|robot check|are you a robot)\b",
    re.IGNORECASE,
)


def _title_reason(title, soup) -> str:
    """Say why a page yielded no work title, or that it yielded one."""
    if title:
        return "boilerplate_title_only" if _BOILERPLATE_TITLE_RE.match(str(title)) else "title_observed"
    # Only refine an already-empty observation. A page carrying a real title is
    # never reclassified by its body text.
    from app.services.abstract_shape import looks_like_contents_listing

    try:
        visible = soup.get_text(" ", strip=True)[:12000]
    except Exception:
        visible = ""
    if visible and looks_like_contents_listing(visible):
        return "navigation_listing_page"
    return "no_title_element"


def extract_web_source_metadata(page_html: str, page_url: str) -> dict:
    soup = BeautifulSoup(page_html, "html.parser")
    document = trafilatura.extract_metadata(
        page_html, default_url=page_url, extensive=False,
    )
    fields = document.as_dict() if document else {}
    heading = soup.find("h1")
    header_nodes = heading.find_all_next(["p", "div", "span", "time"], limit=24) if heading else []
    meta: dict[str, list[str]] = {}
    for node in soup.find_all("meta"):
        key = str(node.get("name") or node.get("property") or "").lower()
        value = str(node.get("content") or "").strip()
        if not value:
            continue
        # Digital Commons (bepress) emits the same Highwire tags under its own
        # prefix, and hosts a large share of university repository content.
        # Reading only the unprefixed names meant those pages returned no
        # observed title, author, journal or volume at all -- the record was
        # sitting in the page and was never looked at.
        if key.startswith("bepress_citation_"):
            meta.setdefault(key[len("bepress_"):], []).append(value)
        meta.setdefault(key, []).append(value)
    title = next(iter(meta.get("citation_title", []) or meta.get("og:title", [])), None)
    title_method = ("citation_title" if meta.get("citation_title")
                    else "og_title" if title else None)
    # An older page with no metadata can state its work in the title element:
    # "Genre Films and the Status Quo by Judith Hess" (Jump Cut archive,
    # 2026-09-30), where the h1 is only the magazine banner.
    element = soup.title.get_text(" ", strip=True) if soup.title else ""
    stated = None if title else re.fullmatch(
        r"(?P<title>.{8,300}?)\s+by\s+(?P<author>[A-Z][\w.'’\-]+(?:\s+[A-Z][\w.'’\-]+){1,3})", element)
    title = (stated.group("title") if stated else None) or title or fields.get("title")
    title_method = title_method or ("title_element_byline" if stated else "page_metadata" if title else None)
    authors = meta.get("citation_author", []) or meta.get("author", [])
    author_method = "html_meta" if authors else None
    if stated and not authors:
        authors, author_method = [stated.group("author")], "title_element_byline"
    # A site's generic author meta can name the publisher rather than this
    # article's byline. Prefer one explicit article-header author in that case;
    # never override a citation_author or search body mentions for a name.
    # A site can append its own name to og:title even when no article heading
    # exposes the work title on its own. Strip only a suffix that repeats a
    # name the page itself declares, optionally followed by its issue label.
    if title and not meta.get('citation_title'):
        declared = [value for key in ('og:site_name', 'application-name', 'citation_journal_title')
                    for value in meta.get(key, [])]
        title = _strip_declared_site_suffix(str(title), declared)
    from app.services.source_type import visible_journal_masthead
    if heading and not meta.get('citation_title') and visible_journal_masthead(soup):
        visible_title = heading.get_text(' ', strip=True)
        if any(str(title).casefold() == (visible_title + separator + site).casefold()
               for site in meta.get('og:site_name', []) for separator in (' - ', ' | ', ' – ')):
            title = visible_title
    if heading and not meta.get('citation_author') and visible_journal_masthead(soup):
        site_names = {v.casefold() for key in ('og:site_name', 'application-name') for v in meta.get(key, [])}
        if not authors or all(a.casefold() in site_names for a in authors):
            bylines = [n.get_text(' ', strip=True).rstrip('*').strip()
                       for n in heading.find_all_next(['h2', 'h3', 'p'], limit=12)
                       if any(re.search(r'(?:^|-)author(?:-|$)', c) for c in n.get('class', []))]
            if len(bylines) == 1 and 3 <= len(bylines[0]) <= 200:
                authors = [bylines[0]]
                author_method = 'visible_journal_header_byline'
    if not authors and fields.get("author"):
        authors = re.split(r"\s*;\s*", fields["author"])
        author_method = "trafilatura_metadata"
    # Some article headers use an ordinary visible byline, not author metadata.
    # Restrict fallback to a short explicit byline near the article heading;
    # never search arbitrary body mentions for the expected author.
    if not authors:
        if heading:
            for node in header_nodes:
                if node.find(["p", "div"]):
                    continue
                text = node.get_text(" ", strip=True)
                match = re.fullmatch(r"(?:Posted|Written) by\s+(.{3,200})", text, re.I)
                if match:
                    names = match.group(1).split(",", 1)[0]
                    authors = re.split(r"\s+(?:and|&)\s+|\s*;\s*", names)
                    author_method = "visible_header_byline"
                    break
    parsed_url = urlsplit(page_url)
    archive_item = parsed_url.hostname in {"archive.org", "www.archive.org"} and parsed_url.path.startswith("/details/")
    date = next(iter(meta.get("citation_publication_date", []) or meta.get("citation_date", [])
                     or meta.get("article:published_time", [])), None)
    generic_date = not date and bool(fields.get("date"))
    date = date or fields.get("date")
    date_method = "metadata" if date else None
    # A byline slot holding interface text ("Author Notes Loading Author Notes",
    # "Name") is page furniture, not a credit; on such a page the generic date
    # describes the page, not the work (paper 5, Bond, 2026-10-01).
    labels = [a for a in authors if _label_only_author(str(a))]
    if labels:
        authors = [a for a in authors if not _label_only_author(str(a))]
        if not authors:
            author_method = "interface_label_rejected"
        if generic_date:
            date, date_method = None, "page_furniture_date_rejected"
    if archive_item:
        # The item webpage's date is often its upload/scan date, not the
        # publication year of the described book. Require the labelled field.
        date = None
        date_method = "catalog_publication_date_unavailable"
        publication_values = []
        for term in soup.find_all("dt"):
            if term.get_text(" ", strip=True).casefold() == "publication date":
                value = term.find_next_sibling("dd")
                if value:
                    publication_values.append(value.get_text(" ", strip=True))
        if len(set(publication_values)) == 1 and re.fullmatch(r"(?:18|19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?", publication_values[0]):
            date = publication_values[0]
            date_method = "catalog_publication_date"
    if not date and not archive_item:
        for node in header_nodes:
            value = node.get_text(" ", strip=True)
            value = re.sub(r"^(?:Published|Posted)(?: on)?\s+", "", value, flags=re.I)
            if len(value) > 40:
                continue
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"):
                try:
                    date = datetime.strptime(value, fmt).date().isoformat()
                except ValueError:
                    continue
                date_method = "visible_header_date"
                break
            if date:
                break
    if not date and stated:
        # The same pages open with their source statement: "from Jump Cut,
        # no. 1, 1974, pp. 1, 16, 18". Only that statement, in the opening
        # text of a page that already stated its title and author, gives a year.
        opening = soup.get_text(" ", strip=True)[:800]
        source_year = re.search(r"\bfrom\s+[^.]{2,80}?,\s*(?:(?:no|vol|issue)\.?\s*\d{1,4},\s*)?((?:18|19|20)\d{2})\b",
                                opening, re.IGNORECASE)
        if source_year:
            date, date_method = source_year.group(1), "visible_source_statement"
    year = re.search(r"\b(?:18|19|20)\d{2}\b", str(date or ""))
    # Catalog volume/pressbook dates describe the item, not the webpage.
    # Do not generalize to arbitrary historical years in article titles.
    if not date and parsed_url.hostname == 'lantern.mediahist.org' and title:
        volume_date = re.search(
            r'\((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+'
            r'((?:18|19|20)\d{2})\s*[-–—]\s*'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+'
            r'((?:18|19|20)\d{2})\)', str(title), re.I)
        pressbook_date = re.search(r'\([^()]*\bPressbook,\s*((?:18|19|20)\d{2})\)', str(title), re.I)
        if volume_date and int(volume_date[1]) <= int(volume_date[2]) <= int(volume_date[1])+1:
            date = volume_date[1]+'–'+volume_date[2]
        elif pressbook_date:
            date = pressbook_date[1]
        if date:
            date_method = 'catalog_title_publication_interval'
            year = re.search(r'\d{4}', date)
    doi = next(iter(meta.get("citation_doi", []) or meta.get("dc.identifier", [])), None)
    doi_match = re.search(r"10\.\d{4,9}/\S+", doi or "", re.I)
    # Preserve explicit component observations without guessing from body text.
    # Conflicting repeated values are unresolved, not first-value wins.
    def unique_meta(name):
        values = set(meta.get(name, []))
        return next(iter(values)) if len(values) == 1 else None

    # A journal article names its journal here; a book chapter names its book
    # in the other. Reading only the second meant no journal article ever
    # carried an observed container, so a reference naming the wrong journal
    # had nothing to disagree with.
    container = unique_meta("citation_journal_title") or unique_meta("citation_inbook_title")
    volume = unique_meta("citation_volume")
    issue = unique_meta("citation_issue")
    first = unique_meta("citation_firstpage")
    last = unique_meta("citation_lastpage")
    pages = None
    if (first and last and re.fullmatch(r"\d{1,5}", first)
            and re.fullmatch(r"\d{1,5}", last) and int(first) <= int(last)):
        pages = f"{first}-{last}"
    return {
        "container_title": container[:1000] if container else None,
        "volume": volume[:40] if volume else None,
        "issue": issue[:40] if issue else None,
        "pages": pages,
        "title": str(title)[:1000] if title else None,
        "title_method": title_method,
        "title_reason": _title_reason(title, soup),
        "authors": [_without_job_title(str(author).strip())[:500] for author in authors[:64] if str(author).strip()],
        "year": year.group(0) if year else None,
        "doi": doi_match.group(0)[:255] if doi_match else None,
        "publication_date": str(date)[:40] if date else None,
        "date_method": date_method,
        "author_method": author_method,
        "html_sha256": hashlib.sha256(page_html.encode("utf-8")).hexdigest(),
    }


# A news byline often carries the writer's job title: "Ariana Brockington
# Trending News Reporter" (TODAY.com, Franchise 1, 2026-09-29) was read as
# the surname "Reporter" and the student's own link was rejected.
_JOB_TITLE = re.compile(
    r"^(?P<name>\S+(?:\s+\S+)+?)\s+"
    r"(?:(?:senior|staff|contributing|associate|deputy|chief|managing|executive|lead|freelance|"
    r"trending|breaking|news|entertainment|features?|digital|political|politics|business|culture|"
    r"pop|tv|film|movies?|television|music|health|sports?|tech|technology|science|social|media|"
    r"assistant|general|principal|national|global|international|weekend|editorial|commerce|"
    r"shopping|lifestyle|opinion|review|reviews|video|audio|investigative|special|senior)\s+)*"
    r"(?:reporter|writer|editor|correspondent|contributor|columnist|journalist|producer|critic)s?$",
    re.IGNORECASE)


def _without_job_title(author: str) -> str:
    """The name without a trailing job title; unchanged when none is present."""
    match = _JOB_TITLE.match(author)
    return match.group("name").rstrip(" ,") if match else author


# Words that make up interface labels rather than names.
_AUTHOR_LABEL_WORDS = frozenset({"author", "authors", "note", "notes", "loading", "name", "names",
                                 "by", "unknown", "details", "more", "information", "info"})


def _label_only_author(value: str) -> bool:
    words = re.findall(r"[a-z]+", value.casefold())
    return bool(words) and all(word in _AUTHOR_LABEL_WORDS for word in words)
