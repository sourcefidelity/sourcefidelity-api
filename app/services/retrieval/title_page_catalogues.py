"""Title-page statements of responsibility from national library catalogues.

Whether a cited book is single-authored or edited cannot be read from author
lists: Google Books and Open Library list a volume's editors as its authors.
The title page can tell ("by Richard Dyer" against "edited by David
Neumeyer"), and library catalogues transcribe it in MARC field 245 $c.
Open Library holds that statement for some editions only; these catalogues
are asked when it does not (`title-page-catalogues-v1`, owner decision
2026-10-16):

* the Library of Congress (SRU, https://lx2.loc.gov/sru/lcdb; no account;
  the server's published limits are technical: 50 records a response);
* the National Library of Scotland (Alma SRU, 44NLS_INST; no account). A
  legal deposit library, it holds UK books the Library of Congress may not.
  Its terms let BNB-derived records be used one at a time for non-commercial
  resource discovery; commercial deployments must arrange use with the
  British Library (metadata@bl.uk).

Measured 2026-10-16 over the 44 books cited as chapter containers in stored
papers: wherever Open Library and the Library of Congress both answered they
agreed (10 of 10, the one edited volume included); the Library of Congress
settled 8 more that Open Library left unresolved. Metadata only; no text is
acquired. A catalogue that fails or holds nothing leaves the book unresolved.
"""
from __future__ import annotations

import html
import re

from app.services.processing_metrics import record_provider_request

VERSION = "title-page-catalogues-v1"
_PACE_SECONDS = 1.0
_MAX_RECORDS = 5
CATALOGUES = (
    ("library_of_congress", "https://lx2.loc.gov/sru/lcdb",
     {"version": "1.1", "recordSchema": "marcxml"}, 'dc.title="{title}" and dc.creator="{surname}"'),
    ("national_library_of_scotland", "https://nls.alma.exlibrisgroup.com/view/sru/44NLS_INST",
     {"version": "1.2", "recordSchema": "marcxml"}, 'alma.title all "{title}" and alma.creator all "{surname}"'),
)
_RECORD_245 = re.compile(r'tag="245".*?</(?:marc:)?datafield>', re.S)


def _words(value: str) -> list[str]:
    value = str(value or "").replace("’", "'").replace("‘", "'")
    return re.findall(r"\w+", value.casefold())


def main_title(title: str) -> str:
    """The title before its subtitle (":" or a spaced dash)."""
    return re.split(r"\s*[:—–]\s*|\s+-\s+", str(title or ""), maxsplit=1)[0].strip()


def statements_from_marcxml(payload: str, title: str) -> list[str]:
    """245 $c statements of the records whose title proper is the cited main title."""
    wanted = _words(main_title(title))
    statements = []
    for record in _RECORD_245.findall(payload or ""):
        proper = re.search(r'code="a">([^<]*)<', record)
        statement = re.search(r'code="c">([^<]*)<', record)
        if not proper or not statement:
            continue
        if _words(html.unescape(proper.group(1))) != wanted:
            continue
        statements.append(" ".join(html.unescape(statement.group(1)).split())[:300])
    return statements


def catalogue_statements(title: str, surname: str, get) -> tuple[str, list[str]] | None:
    """(catalogue, statements) from the first catalogue holding the book, or None.

    `get(provider, url, params)` is a paced GET. The query uses the main title
    with straight apostrophes ("Hollywood's second sex" is not found with a
    curly one) and the cited author's surname.
    """
    query_title = re.sub(r'["\\]', " ", main_title(title).replace("’", "'").replace("‘", "'")).strip()
    if len(_words(query_title)) < 1 or not surname.strip():
        return None
    for provider, url, base, template in CATALOGUES:
        params = {**base, "operation": "searchRetrieve", "maximumRecords": str(_MAX_RECORDS),
                  "query": template.format(title=query_title, surname=surname.strip())}
        try:
            record_provider_request(provider)
            response = get(provider, url, params)
        except Exception:  # noqa: BLE001 - a catalogue that fails only leaves the book unresolved
            continue
        if getattr(response, "status_code", None) != 200:
            continue
        statements = statements_from_marcxml(response.text, title)
        if statements:
            return provider, statements
    return None
