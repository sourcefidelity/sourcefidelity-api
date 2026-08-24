from io import BytesIO
from zipfile import ZipFile

import httpx

from app.services.retrieval.gutenberg import (
    GutenbergRetriever,
    _extract_epub_text,
    _parse_item_feed,
    _parse_search_feed,
)
from app.services.retrieval.wikisource import WikisourceRetriever, _extract_rendered_text


SEARCH_FEED = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>Hard Times</title><content type="text">Charles Dickens</content>
    <link rel="subsection" href="/ebooks/786.opds" /></entry>
</feed>"""

ITEM_FEED = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>Hard Times</title><published>1997-01-01T00:00:00Z</published>
    <rights>Public domain in the USA.</rights><author><name>Dickens, Charles</name></author>
    <link rel="http://opds-spec.org/acquisition" type="application/epub+zip"
      href="https://www.gutenberg.org/ebooks/786.epub.noimages" /></entry>
</feed>"""


def test_gutenberg_parses_official_opds_search_and_item():
    candidate = _parse_search_feed(SEARCH_FEED)[0]
    assert candidate == {
        "title": "Hard Times",
        "authors": ["Charles Dickens"],
        "item_url": "https://www.gutenberg.org/ebooks/786.opds",
    }
    edition = _parse_item_feed(ITEM_FEED)
    assert edition["title"] == "Hard Times"
    assert edition["rights"] == "Public domain in the USA."
    assert edition["acquisition_url"].endswith("786.epub.noimages")


def test_gutenberg_extracts_epub_text_and_strips_boilerplate():
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr(
            "chapter.xhtml",
            "<html><body>*** START OF THE PROJECT GUTENBERG EBOOK TEST ***"
            "<p>Readable chapter text.</p>"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***</body></html>",
        )
    assert _extract_epub_text(payload.getvalue()) == "Readable chapter text."


def test_gutenberg_search_uses_opds_and_records_rights(monkeypatch):
    responses = [
        httpx.Response(200, content=SEARCH_FEED, request=httpx.Request("GET", "https://example")),
        httpx.Response(200, content=ITEM_FEED, request=httpx.Request("GET", "https://example")),
    ]
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: responses.pop(0))
    result = GutenbergRetriever().search_by_title_author("Hard Times", "Charles Dickens")
    assert result.success
    assert result.metadata["license_class"] == "public_domain"
    assert result.metadata["rights_jurisdiction"] == "USA"


def test_wikisource_uses_root_work_for_subpage_and_rejects_short_page(monkeypatch):
    search = httpx.Response(
        200,
        json={"query": {"search": [{"title": "Hard Times/Book I/Chapter I"}]}},
        request=httpx.Request("GET", "https://example"),
    )
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: search)
    retriever = WikisourceRetriever()
    monkeypatch.setattr(retriever, "_fetch_rendered_page", lambda lang, title: ("too short", "Charles Dickens"))
    result = retriever._search_edition("en", "Hard Times", "Charles Dickens", "Hard Times Dickens")
    assert not result.success


def test_wikisource_rendered_html_is_cleaned():
    html = "<div><style>x</style><table class='metadata'><tr><td>header</td></tr></table>" + (
        "<p>Readable primary text sentence.</p>" * 10
    ) + "</div>"
    cleaned = _extract_rendered_text(html)
    assert "header" not in cleaned
    assert "Readable primary text sentence." in cleaned
