"""P3, 2026-10-07: a student link or DOI page that answered without identifying
the cited work is a completed check, not a provider failure."""
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE, _student_link_disposition
from app.services.web_fetch_diagnostics import WebFetchDiagnostic


def _page(reason):
    return RetrievalResult(source_name="web_fetch", success=False, error="Page title does not match the cited reference",
                           metadata={"web_fetch_diagnostic": WebFetchDiagnostic(reason=reason).model_dump()})


def test_answered_links_and_doi_pages_are_typed_and_failures_stay_failures():
    assert _student_link_disposition(_page("page_title_mismatch_unconfirmed")) == ("no_match", "link_page_title_differs")
    assert _student_link_disposition(_page("site_homepage")) == ("no_match", "link_site_home_page")
    doi = lambda error: RetrievalResult(source_name="doi_resolver", success=False, error=error)
    assert _student_link_disposition(doi("Malformed DOI")) == ("unavailable", "doi_malformed")
    assert _student_link_disposition(doi("DOI resolver HTML title does not match cited source")) == (
        "no_match", "doi_page_title_differs")
    # A network failure, a refusal and a success are left to the ordinary rules.
    assert _student_link_disposition(doi("DOI resolver failed (ConnectTimeout)")) is None
    assert _student_link_disposition(_page("access_restricted")) is None
    assert _student_link_disposition(RetrievalResult(source_name="web_fetch", success=True)) is None


def _record(provider, result):
    token = _ACTIVE_DISCOVERY_TRACE.set({
        "reference_id": "fixture", "expected": ExpectedBibliographicFields(title="A source title", authors=["Rivera"], year="2024"),
        "required": set(), "queries": [], "attempts": [], "candidates": [], "limitations": [],
        "search_policy_version": "api-first-search-v2"})
    try:
        SourceResolver._record_discovery_attempt(category="student_url", provider=provider, result=result, required=False)
        trace, _record = SourceResolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    [attempt] = trace["attempts"]
    return attempt


def test_the_recorded_attempt_says_what_happened():
    attempt = _record("student_url_html", _page("page_title_mismatch_unconfirmed"))
    assert (attempt["outcome"], attempt["reason_code"], attempt["error_code"]) == ("no_match", "link_page_title_differs", None)
    malformed = _record("public_doi", RetrievalResult(source_name="doi_resolver", success=False, error="Malformed DOI"))
    assert (malformed["outcome"], malformed["reason_code"], malformed["error_code"]) == ("unavailable", "doi_malformed", None)
    failed = _record("public_doi", RetrievalResult(source_name="doi_resolver", success=False,
                                                   error="DOI resolver failed (ConnectTimeout)"))
    assert failed["outcome"] == "operational_failure"
