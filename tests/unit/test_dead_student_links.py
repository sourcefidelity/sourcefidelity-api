"""A student's link that answers 404 or 410 is a dead link, not a network failure (2026-10-14)."""
from app.services.retrieval.base import RetrievalResult
from app.services.search.rerun_reuse import _network_failed
from app.services.source_resolver import _link_gone, _student_link_disposition
from app.services.web_fetch_diagnostics import WebFetchDiagnostic


def _observed(url: str, outcome: str):
    """The link checker's record of one request for a submitted link."""
    from types import SimpleNamespace
    from app.services.submitted_links import ACTIVE_LINKS, binding
    return ACTIVE_LINKS.set([SimpleNamespace(request_sha256=binding(url), requests=[SimpleNamespace(outcome=outcome)])])


def test_the_search_record_reads_the_link_checkers_outcome():
    from app.services.submitted_links import ACTIVE_LINKS
    url = "https://publisher.example/file.pdf"
    for outcome, gone in [("not_found", True), ("removed", True), ("timeout", False),
                          ("connection_failure", False), ("access_refused", False)]:
        token = _observed(url, outcome)
        try:
            assert _link_gone(url) is gone
            assert not _link_gone("https://publisher.example/other.pdf")
        finally:
            ACTIVE_LINKS.reset(token)
    assert not _link_gone(url)   # never requested


def test_a_dead_link_is_an_answered_check():
    result = RetrievalResult(source_name="student_url", success=False,
                             error="Student URL download failed (HTTPStatusError)",
                             metadata={"web_fetch_diagnostic": WebFetchDiagnostic(reason="link_not_found").model_dump()})
    assert _student_link_disposition(result) == ("no_match", "link_not_found")
    # Without the diagnostic (a network error) it is still not an answered check.
    assert _student_link_disposition(RetrievalResult(source_name="student_url", success=False,
                                                     error="Student URL download failed (ConnectError)")) is None


def test_a_dead_link_does_not_force_a_new_search_on_the_next_run():
    adapters = [{"route_category": "academic_adapter", "outcome": "no_match"} for _ in range(4)]
    dead = {"outcome": "bibliographic_conflict",
            "attempts": [{"route_category": "student_url", "outcome": "no_match", "reason_code": "link_not_found"},
                         *adapters]}
    failed = {"outcome": "bibliographic_conflict",
              "attempts": [{"route_category": "student_url", "outcome": "operational_failure"}, *adapters]}
    assert not _network_failed(dead)
    assert _network_failed(failed)


def test_a_pdf_link_answered_by_a_web_page_is_checked_as_a_web_page(monkeypatch):
    from app.services.source_resolver import SourceResolver
    resolver = SourceResolver.__new__(SourceResolver)

    def not_pdf(url, **kwargs):
        raise ValueError("Not a PDF: response content type is not permitted for this route")
    monkeypatch.setattr(resolver, "_safe_download", not_pdf, raising=False)
    result = resolver._try_student_url("https://publisher.example/paper.pdf", None, "A title", "Ames, A.", "2020")
    # The wording resolve() reads to send the link on to the web route, and an answered check.
    assert "not a pdf" in result.error.lower()
    assert _student_link_disposition(result) == ("no_match", "link_not_pdf")


def test_a_pdf_link_the_checker_found_missing_is_a_dead_link(monkeypatch):
    from app.services.source_resolver import SourceResolver
    from app.services.submitted_links import ACTIVE_LINKS
    resolver = SourceResolver.__new__(SourceResolver)
    url = "https://publisher.example/paper.pdf"

    def missing(address, **kwargs):
        raise ValueError("Download failed (HTTPStatusError status=404)")
    monkeypatch.setattr(resolver, "_safe_download", missing, raising=False)
    token = _observed(url, "not_found")
    try:
        result = resolver._try_student_url(url, None, "A title", "Ames, A.", "2020")
    finally:
        ACTIVE_LINKS.reset(token)
    assert _student_link_disposition(result) == ("no_match", "link_not_found")
