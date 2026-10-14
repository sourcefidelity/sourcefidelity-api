"""Magazine pages from the Media History Digital Library (lantern-pages-v1, owner decision 2026-10-14)."""
import json

import httpx

from app.services.retrieval import lantern
from app.services.retrieval.lantern import cited_pages, lantern_target, read_article, title_in_text

VOLUME = "fanmag22unse"
PAGES = {31: "25", 32: "26", 33: "27", 156: "26", 157: "27", 333: "61", 334: "62", 98: "90"}
OCR = {
    32: "26 FANMAG Meet the NEW BLONDE! She's Hara Voss, the heroine of the season.",
    33: "27 FANMAG continued: Hara Voss came to the studio with a borrowed coat.",
    156: "26 FANMAG Golf widows of the hills meet the wives who wait.",
    157: "27 FANMAG More golf widows.",
    333: "A caption about another star entirely.",
    334: "62 FANMAG letters page.",
    98: "90 FANMAG the end of the Voss story.",
}


def _response(url: str, *, text: str = "", payload=None) -> httpx.Response:
    request = httpx.Request("GET", url)
    if payload is not None:
        return httpx.Response(200, json=payload, request=request)
    return httpx.Response(200, text=text, request=request)


def _fake_get(url: str) -> httpx.Response:
    if url.endswith(".json") and "/catalog/" in url:
        return _response(url, payload={"data": {"attributes": {
            "title": {"attributes": {"value": "Fanmag"}},
            "dateStart": {"attributes": {"value": "Nov 1930-Apr 1931"}}}}})
    if url.endswith("_page_numbers.json"):
        return _response(url, payload={"pages": [{"leafNum": leaf, "pageNumber": page} for leaf, page in PAGES.items()]})
    leaf = int(url.rsplit("_", 1)[1])
    return _response(url, text=f'<html><div class="card card-body ">{OCR.get(leaf, "")}</div></html>')


def test_links_pages_and_titles_are_read_tolerantly():
    assert lantern_target("https://lantern.mediahist.org/catalog/screenland22unse_0333") == ("screenland22unse", 333)
    assert lantern_target("https://archive.org/details/screenland22unse") is None
    assert cited_pages("Talbott, T. (1930). Meet the new blonde. Screenland, 26-27.") == (26, 27)
    assert cited_pages("Whitchel, J. A. (1932). The girl. Silver Screen, 22-50.") == (22, 50)
    # A display headline the OCR broke ("th s TARS") still matches; an unrelated page does not.
    assert title_in_text("The Girl Who Changed the Stars", "envious blanched The Girl Who Changed th s TARS Tossed into the Stream")
    assert not title_in_text("The Girl Who Changed the Stars", "Golf widows of the hills meet the wives")


def test_the_article_is_found_on_the_cited_page_not_the_linked_leaf():
    article = read_article(f"https://lantern.mediahist.org/catalog/{VOLUME}_0333", title="Meet the new blonde",
                           raw_ref="Talbott, T. (1930). Meet the new blonde. Fanmag, 26-27.",
                           get=_fake_get, get_archive=_fake_get)
    assert article.leaves == [32, 33] and article.printed_pages == ["26", "27"] and article.complete
    assert article.magazine == "Fanmag" and "Hara Voss came to the studio" in article.text
    assert article.first_leaf_url.endswith(f"{VOLUME}_0032")


def test_a_long_range_reads_the_opening_and_the_last_page_and_is_not_complete():
    article = read_article(f"https://lantern.mediahist.org/catalog/{VOLUME}_0032", title="Meet the new blonde",
                           raw_ref="Talbott, T. (1930). Meet the new blonde. Fanmag, 26-90.",
                           get=_fake_get, get_archive=_fake_get)
    assert article.leaves == [32, 33, 98] and not article.complete
    assert article.notes == ["non_contiguous_page_range"]


def test_a_title_on_no_candidate_page_finds_nothing():
    assert read_article(f"https://lantern.mediahist.org/catalog/{VOLUME}_0333", title="A story never printed here",
                        raw_ref="X. (1930). A story never printed here. Fanmag, 26-27.",
                        get=_fake_get, get_archive=_fake_get) is None


def test_the_resolver_confirms_on_title_magazine_and_year(monkeypatch):
    from app.services.source_resolver import SourceResolver
    resolver = SourceResolver.__new__(SourceResolver)
    monkeypatch.setattr("app.services.retrieval.provider_gate.paced_get",
                        lambda provider, interval, url, **kwargs: _fake_get(url))
    url = f"https://lantern.mediahist.org/catalog/{VOLUME}_0333"
    ref = "Talbott, T. (1930). Meet the new blonde. Fanmag, 26-27."
    found = resolver._try_lantern(url, title="Meet the new blonde", author="Talbott, T.", year="1930",
                                  container_title="Fanmag", raw_ref=ref)
    assert found.success and found.metadata["identity_confidence"] == "high"
    assert found.metadata["text_quality"] == "scan_ocr" and found.representation.completeness == "complete"
    assert found.authors == []   # the byline is not in the scanned text
    wrong_year = resolver._try_lantern(url, title="Meet the new blonde", author="Talbott, T.", year="1950",
                                       container_title="Fanmag", raw_ref=ref)
    assert not wrong_year.success and wrong_year.metadata["identity_confidence"] == "medium"
    assert json.dumps(wrong_year.metadata["lantern"])  # the record stays inspectable


def test_lantern_ocr_text_gets_ocr_tolerant_quotation_matching():
    from types import SimpleNamespace
    from app.services.verification_evidence import _ocr_tolerant
    assert _ocr_tolerant(SimpleNamespace(text_quality="scan_ocr", derivation_method=None, provenance="lantern"))
    assert _ocr_tolerant(SimpleNamespace(text_quality="scan_ocr", derivation_method="ocr-v1", provenance="upload"))
    # A scan's own text layer without a bound derivation stays strict, as before.
    assert not _ocr_tolerant(SimpleNamespace(text_quality="scan_ocr", derivation_method=None, provenance="web_fetch"))
    assert not _ocr_tolerant(SimpleNamespace(text_quality="digital", derivation_method=None, provenance="lantern"))
