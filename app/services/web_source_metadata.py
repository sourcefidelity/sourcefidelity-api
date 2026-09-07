"""Observed webpage identity fields; never fill them from the citation."""

import hashlib
import re
from datetime import datetime
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import trafilatura


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
        if value:
            meta.setdefault(key, []).append(value)
    title = next(iter(meta.get("citation_title", []) or meta.get("og:title", [])), None)
    title = title or fields.get("title")
    authors = meta.get("citation_author", []) or meta.get("author", [])
    author_method = "html_meta" if authors else None
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
    date = next(iter(meta.get("citation_publication_date", []) or meta.get("article:published_time", [])), None)
    date = date or fields.get("date")
    date_method = "metadata" if date else None
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
    year = re.search(r"\b(?:18|19|20)\d{2}\b", str(date or ""))
    doi = next(iter(meta.get("citation_doi", []) or meta.get("dc.identifier", [])), None)
    doi_match = re.search(r"10\.\d{4,9}/\S+", doi or "", re.I)
    return {
        "title": str(title)[:1000] if title else None,
        "authors": [str(author).strip()[:500] for author in authors[:64] if str(author).strip()],
        "year": year.group(0) if year else None,
        "doi": doi_match.group(0)[:255] if doi_match else None,
        "publication_date": str(date)[:40] if date else None,
        "date_method": date_method,
        "author_method": author_method,
        "html_sha256": hashlib.sha256(page_html.encode("utf-8")).hexdigest(),
    }
