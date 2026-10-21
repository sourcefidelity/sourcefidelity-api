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
    # A single page after the magazine needs no "p." (owner review 2026-10-14, Harlow reference 7).
    assert cited_pages("Rowe, K. (1932). Strikes. Movie Classic, 41. https://lantern.mediahist.org/catalog/x_0041") == (41, 41)
    assert cited_pages("Rowe, K. (1932). Strikes. Movie Classic, 41.") == (41, 41)
    assert cited_pages("Rowe, K. (1932). Strikes. Photoplay. https://lantern.mediahist.org/catalog/x_0041") is None
    assert cited_pages("Rowe, K. (1932). Strikes. Photoplay, 1932.") is None
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
    # Printed page numbers travel with the exact text they label (2026-10-14).
    import hashlib
    labels = found.metadata["printed_page_labels"]
    assert labels["content_sha256"] == hashlib.sha256(found.representation.content).hexdigest()
    assert len(labels["labels"]) == found.representation.content.count(b"\f") + 1 == 2
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


def test_an_earlier_result_without_text_under_older_link_handling_is_searched_again():
    from app.services.source_reuse import STUDENT_LINK_ROUTES_VERSION, earlier_result
    _reusable = lambda item, reference=None: earlier_result(item, reference).reusable
    reference = {"reference_id": "ref-1", "url": "https://lantern.mediahist.org/catalog/fanmag22unse_0333",
                 "raw_ref": "Talbott, T. (1930). Meet the new blonde. Fanmag, 26-27.", "source_kind": "magazine_article"}
    completed = {"outcome": "unlocated_after_search", "created_at": "t0",
                 "attempts": [{"route_category": "student_url", "provider": "student_url_html", "outcome": "candidate_found"}]}
    item = {"status": "unavailable", "reason_code": "full_text_unavailable", "reference_discovery": completed}
    assert not _reusable(item, reference)
    assert _reusable({**item, "student_link_routes": STUDENT_LINK_ROUTES_VERSION}, reference)
    # A reference without a student link is unaffected.
    assert _reusable(item, {**reference, "url": ""})



def test_a_link_to_the_second_page_of_a_spread_with_a_damaged_headline():
    pages = {10: "40", 11: "41", 12: "42"}
    ocr = {10: "Scenes from life during the tr nside the studio.",
           11: "J ressed in Di nity and ]S[ot Much Else. Fans demand to know.",
           12: "Another article entirely."}

    def get(url):
        if url.endswith(".json") and "/catalog/" in url:
            return _response(url, payload={"data": {"attributes": {"title": {"attributes": {"value": "Fanmag"}},
                                                                   "dateStart": {"attributes": {"value": "1933"}}}}})
        if url.endswith("_page_numbers.json"):
            return _response(url, payload={"pages": [{"leafNum": k, "pageNumber": v} for k, v in pages.items()]})
        leaf = int(url.rsplit("_", 1)[1])
        return _response(url, text=f'<div class="card card-body ">{ocr.get(leaf, "")}</div>')
    article = read_article("https://lantern.mediahist.org/catalog/vol_0011", title="Dressed in Dignity and Not Much Else",
                           raw_ref="X. (1933). Dressed in dignity and not much else. Fanmag, 40-41.", get=get, get_archive=get)
    assert article.leaves == [10, 11] and article.complete
    # Without cited pages only the page carrying the title is read, not the next article.
    alone = read_article("https://lantern.mediahist.org/catalog/vol_0011", title="Dressed in Dignity and Not Much Else",
                         raw_ref="X. (1933). Dressed in dignity and not much else. Fanmag.", get=get, get_archive=get)
    assert alone.leaves == [11] and not alone.complete and alone.notes == ["no_cited_pages"]


def test_a_title_not_found_is_an_answered_check(monkeypatch):
    from app.services.source_resolver import SourceResolver, _student_link_disposition
    monkeypatch.setattr("app.services.retrieval.provider_gate.paced_get",
                        lambda provider, interval, url, **kwargs: _fake_get(url))
    missing = SourceResolver.__new__(SourceResolver)._try_lantern(
        f"https://lantern.mediahist.org/catalog/{VOLUME}_0333", title="A story never printed here", author="X.",
        year="1930", container_title="Fanmag", raw_ref="X. (1930). A story never printed here. Fanmag, 26-27.")
    assert not missing.success and _student_link_disposition(missing) == ("no_match", "link_page_title_differs")


def test_pages_carry_the_printed_number_only_where_the_scan_shows_it():
    # Owner decision 2026-10-14: the archive's page numbers label the pages,
    # each confirmed by the page number printed on that page; pages are
    # separated by a form feed so a locator can be checked.
    from app.services.retrieval.lantern import page_labels
    ocr = {10: "40\nDRESSED in dignity and not much else, she said.", 11: "The rest of the story.\n:o"}
    def get(url):
        if url.endswith(".json") and "/catalog/" in url:
            return _response(url, payload={"data": {"attributes": {"title": {"attributes": {"value": "Fanmag"}},
                                                                   "dateStart": {"attributes": {"value": "1933"}}}}})
        if url.endswith("_page_numbers.json"):
            return _response(url, payload={"pages": [{"leafNum": 10, "pageNumber": "40"},
                                                     {"leafNum": 11, "pageNumber": "41"}]})
        leaf = int(url.rsplit("_", 1)[1])
        return _response(url, text=f'<div class="card card-body ">{ocr.get(leaf, "")}</div>')
    article = read_article("https://lantern.mediahist.org/catalog/vol_0010", title="Dressed in Dignity and Not Much Else",
                           raw_ref="X. (1933). Dressed in dignity and not much else. Fanmag, 40-41.", get=get, get_archive=get)
    assert article.text.count("\f") == 1 and "\n\n" not in article.text
    assert page_labels(article) == ["40", None]


def test_a_cached_lantern_copy_without_page_numbers_is_read_again():
    # Owner decision 2026-10-14: the local copy is skipped once, so the Lantern
    # route stores the pages with their printed numbers.
    from types import SimpleNamespace
    from app.services.source_reuse import lantern_copy_stale as _stale_lantern_copy
    assert _stale_lantern_copy(SimpleNamespace(provenance="lantern", validation_evidence={}))
    labelled = {"printed_page_labels": {"labels": ["26", None]}}
    assert not _stale_lantern_copy(SimpleNamespace(provenance="lantern", validation_evidence=labelled))
    assert not _stale_lantern_copy(SimpleNamespace(provenance="crossref", validation_evidence={}))


def test_an_article_continued_at_the_back_of_the_issue_is_read_from_its_first_page():
    from app.services.retrieval.lantern import cited_page_ranges
    assert cited_page_ranges("Q. (1947, March). Ask the editor. Fanmag, 50–51, 90–92.") == [(50, 51), (90, 92)]
    assert cited_page_ranges("Q. (1945). A title. Fanmag, 49(10), 30–31, 74–75.") == [(30, 31), (74, 75)]
    assert cited_page_ranges("Rowe, K. (1932). Strikes. Movie Classic, 41.") == [(41, 41)]
    assert cited_page_ranges("Rowe, K. (1932). Strikes. Photoplay, 1932.") == []
    pages = {20: "50", 21: "51", 30: "90", 31: "91", 32: "92"}
    ocr = {20: "Ask the Editor. By Orla Quill. The studio answers.", 21: "More answers.",
           30: "Ask the Editor (Continued from page 51)", 31: "and so on", 32: "the end"}

    def get(url):
        if url.endswith(".json") and "/catalog/" in url:
            return _response(url, payload={"data": {"attributes": {"title": {"attributes": {"value": "Fanmag"}},
                                                                   "dateStart": {"attributes": {"value": "1947"}}}}})
        if url.endswith("_page_numbers.json"):
            return _response(url, payload={"pages": [{"leafNum": k, "pageNumber": v} for k, v in pages.items()]})
        leaf = int(url.rsplit("_", 1)[1])
        return _response(url, text=f'<div class="card card-body ">{ocr.get(leaf, "")}</div>')
    article = read_article("https://lantern.mediahist.org/catalog/vol_0030", title="Ask the editor",
                           raw_ref="Quill, O. (1947, March). Ask the editor. Fanmag, 50–51, 90–92.",
                           get=get, get_archive=get)
    assert article.leaves == [20, 21, 30, 31, 32] and article.complete
    assert article.printed_pages == ["50", "51", "90", "91", "92"]


def test_the_issue_nearest_after_the_linked_page_is_tried_first():
    # Ten bound issues each print a page 30; the linked contents page is in the ninth.
    pages = {leaf: "30" for leaf in range(10, 110, 10)}
    ocr = {90: "How the Star Has Changed! The studio speaks."}

    def get(url):
        if url.endswith(".json") and "/catalog/" in url:
            return _response(url, payload={"data": {"attributes": {"title": {"attributes": {"value": "Fanmag"}},
                                                                   "dateStart": {"attributes": {"value": "1945"}}}}})
        if url.endswith("_page_numbers.json"):
            return _response(url, payload={"pages": [{"leafNum": k, "pageNumber": v} for k, v in pages.items()]})
        leaf = int(url.rsplit("_", 1)[1])
        return _response(url, text=f'<div class="card card-body ">{ocr.get(leaf, "")}</div>')
    article = read_article("https://lantern.mediahist.org/catalog/vol_0085", title="How the star has changed",
                           raw_ref="Q. (1945, August). How the star has changed. Fanmag, 30.", get=get, get_archive=get)
    assert article is not None and article.leaves == [90]


def test_a_searched_article_is_read_from_the_first_page_after_its_contents_line():
    from app.services.retrieval.lantern import article_first_leaf
    pages = {876: "4", 902: "30", 903: "31", 1002: "30"}

    def get_archive(url):
        return _response(url, payload={"pages": [{"leafNum": k, "pageNumber": v} for k, v in pages.items()]})
    assert article_first_leaf("vol", 876, 30, get_archive) == 902
    assert article_first_leaf("vol", 902, 30, get_archive) == 902
    assert article_first_leaf("vol", 876, None, get_archive) == 876
    # Beyond one issue's span the found page itself is read.
    assert article_first_leaf("vol", 700, 30, get_archive) == 700


def test_a_headline_counts_in_order_and_little_words_do_not():
    from app.services.retrieval.lantern import searched_article_on_spread, title_in_order
    assert title_in_order("Doin' the Shorty-George!", "SHORTY. GEORGE! Fred Astaire and his partner")
    scattered = ("the stars of the week " + "filler words about the studio " * 4
                 + "a girl from Ohio " + "more filler about pictures " * 4 + "who changed her name")
    assert not title_in_order("The Girl Who Changed the Stars", scattered)
    assert not title_in_order("Exciting woman", "an exciting week for a woman")
    # A graphic headline OCR cannot read: the cited author's byline confirms a searched page.
    assert searched_article_on_spread("Exciting woman", "Vidor", "Days to remember from an ace director",
                                      "CHARLES VIDOR As told to a writer")
    assert not searched_article_on_spread("Exciting woman", "Vidor", "Days to remember", "As told to a writer")


def test_a_year_with_an_apa_letter_falls_within_the_volume_dates():
    from app.services.retrieval.lantern import year_in_dates
    assert year_in_dates("1946b", "Photoplay (Jul-Dec 1946)") and not year_in_dates("1947a", "Jul-Dec 1946")
