"""Magazine articles found in the Media History Digital Library (media-history-archive-v1).

Fixtures are invented: no real magazine page, person or student text.
"""
import httpx
import pytest

from app.services.retrieval import media_history_archive as archive


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            raise httpx.HTTPStatusError("error", request=request,
                                        response=httpx.Response(self.status_code, request=request))

    def json(self):
        return self.payload


TOC = ("Contents\nThe Editor's Page Mara Quill 9\nHow Ada Lark Has Changed! Tamsin 30\n"
       "Shirley, Today. Orla Penn 32\n")


def serve(monkeypatch, pages, hits, publications=("Lanternland", "Other Weekly")):
    calls = []

    def get(url, params=None, **kwargs):
        calls.append((url, params))
        if url == archive._PUBLICATIONS:
            return Response(list(publications))
        if url == archive._SEARCH:
            return Response({"data": hits if params.get("page") == 1 else [], "meta": {"pages": {}}})
        page_id = url.rsplit("/", 2)[-2]
        return Response({"body": [pages[page_id]], "read": f"http://archive.org/stream/{page_id}"})
    monkeypatch.setattr(archive.httpx, "get", get)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    monkeypatch.setattr(archive, "_publications_cache", None)
    return calls


def hit(page_id, label):
    return {"id": page_id, "attributes": {"title": label}}


def test_a_contents_line_with_the_cited_author_and_page_locates_the_article(monkeypatch):
    serve(monkeypatch, {"lantern49_0876": TOC}, [hit("lantern49_0876", "Lanternland (Nov 1944-Oct 1945)")])
    found = archive.find_periodical_article("How Ada Lark has change!", "Lanternland", "1945", "Tamsin", "30")
    assert found.outcome == "results"
    assert found.match.volume_dates == "Lanternland (Nov 1944-Oct 1945)"
    assert found.match.author_beside and found.match.first_page_beside
    assert "How Ada Lark Has Changed! Tamsin 30" in found.match.excerpt


@pytest.mark.parametrize("year,author,page", [("1952", "Tamsin", "30"), ("1945", "Nobody", "99")])
def test_a_page_outside_the_cited_year_or_without_author_or_page_is_not_accepted(monkeypatch, year, author, page):
    serve(monkeypatch, {"lantern49_0876": TOC}, [hit("lantern49_0876", "Lanternland (Nov 1944-Oct 1945)")])
    found = archive.find_periodical_article("How Ada Lark has change!", "Lanternland", year, author, page)
    assert found.outcome == "no_results_low_coverage" and found.match is None


def test_another_magazine_is_never_the_cited_one(monkeypatch):
    serve(monkeypatch, {"other_0001": TOC}, [hit("other_0001", "Other Weekly (1945)")])
    found = archive.find_periodical_article("How Ada Lark has change!", "Lanternland", "1945", "Tamsin", "30")
    assert found.match is None


def test_a_magazine_the_library_does_not_hold_is_not_searched(monkeypatch):
    calls = serve(monkeypatch, {}, [])
    found = archive.find_periodical_article("A title here", "Weekly Pictorial", "1947", "Quill", "80")
    assert found.outcome == "declined" and [url for url, _ in calls] == [archive._PUBLICATIONS]


def test_a_failed_request_is_a_failure_not_an_empty_answer(monkeypatch):
    def get(url, params=None, **kwargs):
        raise httpx.ConnectTimeout("slow")
    monkeypatch.setattr(archive.httpx, "get", get)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    monkeypatch.setattr(archive, "_publications_cache", None)
    assert archive.find_periodical_article("A title here", "Lanternland", "1945").outcome == "timeout"


def test_lookalike_letters_from_another_script_are_folded_for_search():
    from app.services.lookalike_letters import fold_lookalikes
    assert fold_lookalikes("The cult оf love") == "The cult of love"
    assert fold_lookalikes("Москва") == "Москва"


def test_a_located_archive_page_makes_the_reference_a_possible_match_never_cannot_be_verified():
    from types import SimpleNamespace
    from app.services.reference_verification import assess_reference_verification
    ref = SimpleNamespace(reference_id="r", source_kind="journal_article", title="How Ada Lark has change!",
                          raw_ref="Tamsin. (1945, August). How Ada Lark has change! Lanternland, 49(10), 30-31.")
    discovery = {"outcome": "search_incomplete", "expected": {"title": ref.title, "source_kind": "journal_article"},
                 "attempts": [{"route_category": "academic_adapter", "provider": "crossref"}], "queries": [],
                 "candidates": [], "periodical_archive_match": {
                     "page_id": "lantern49_0876", "publication": "Lanternland",
                     "volume_dates": "Lanternland (Nov 1944-Oct 1945)", "read_url": "http://archive.org/stream/x",
                     "excerpt": "How Ada Lark Has Changed! Tamsin 30"}}
    result = assess_reference_verification(ref, discovery)
    assert result["status"] == "possible_match" and result["reason_code"] == "periodical_archive_page_located"
    assert result["archive_page"]["publication"] == "Lanternland"


@pytest.mark.parametrize("read", [True, False])
def test_the_search_step_reads_the_found_page_like_a_student_link(monkeypatch, read):
    from app.services.reference_discovery import ExpectedBibliographicFields
    from app.services.retrieval.base import RetrievalResult
    from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
    from app.services.source_type import SourceKindAssessment
    serve(monkeypatch, {"lantern49_0876": TOC}, [hit("lantern49_0876", "Lanternland (Nov 1944-Oct 1945)")])
    expected = ExpectedBibliographicFields(title="How Ada Lark has change!", authors=["Tamsin"], year="1945",
                                           container_title="Lanternland", pages="30-31",
                                           source_kind="journal_article")
    trace = {'reference_id': 'fixture', 'expected': expected, 'required': {'academic_adapter'},
             'queries': [], 'attempts': [], 'candidates': [], 'limitations': [],
             'search_policy_version': 'api-first-search-v2'}
    links = []
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._try_lantern = lambda url, **kwargs: (links.append(url) or RetrievalResult(
        source_name="lantern", success=read, error=None if read else "Cited title not found"))
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        found = resolver._search_periodical_archive(
            title=expected.title, author="Tamsin", year="1945", container_title="Lanternland",
            raw_ref="x", expected_kind=SourceKindAssessment("journal_article", "high", ("fixture",)))
        discovery = {}
        SourceResolver._attach_periodical_archive({}, discovery)
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    assert links == ["https://lantern.mediahist.org/catalog/lantern49_0876"]
    assert (found is not None) is read
    search, reading = trace['attempts']
    assert search.provider == archive.PROVIDER and not search.required and search.outcome == 'candidate_found'
    assert reading.provider == 'lantern' and not reading.required
    # Only the reading of the cited pages records a located record.
    assert bool(trace['candidates']) is read
    assert discovery['periodical_archive_match']['publication'] == 'Lanternland'


def test_a_lookalike_letter_does_not_stop_a_title_from_agreeing_or_counting_as_searched():
    from app.services.reference_discovery import _normalize_text
    from app.services.reference_review_scope import text_key
    cited, found = "The cult оf the love goddess in America", "The Cult of the Love Goddess in America"
    assert _normalize_text(cited) == _normalize_text(found)
    assert text_key(cited) in text_key("title:the cult of the love goddess in america")


def test_the_reference_window_shows_a_located_archive_page_under_located_record():
    from app.services.evidence_report import _render_reference_window_template
    entry = {"number": 7, "template_id": "reference-panel-7", "reference_id": "r",
             "source": {"raw_reference": "Tamsin. (1945, August). How Ada Lark has change! Lanternland, 49(10), 30-31."}}
    page = {"page_id": "lantern49_0876", "publication": "Lanternland", "volume_dates": "Lanternland (Nov 1944-Oct 1945)"}
    html = _render_reference_window_template(entry, [], [], archive_page=page)
    assert '<h3>Located Record</h3><p class="full-reference">Lanternland (Nov 1944-Oct 1945) <a class="reference-url" ' \
           'href="https://lantern.mediahist.org/catalog/lantern49_0876"' in html
    assert "Located Record" not in _render_reference_window_template(entry, [], [])
