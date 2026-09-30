"""Temporary PDF identity inspection never admits content or invents absence."""
import json
from unittest.mock import Mock

import fitz
import httpx
import pytest

from app.services import identity_pdf as pdf
from app.services.file_safety import FileSafetyReport, SafetyVerdict
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.search.transient import finalize_transient_brave, BRAVE_TRANSIENT_POLICY
from app.services.candidate_budget import candidate_budget_scope


def content(title="Competition and cultural institutions", author="Anna River", byline=True):
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 70), title, fontsize=18)
        if byline:
            page.insert_text((40, 100), "By " + author, fontsize=12)
        page.insert_text((40, 125), "Revised January 2020", fontsize=12)
        page.insert_text((40, 200), "PRIVATE BODY TEXT", fontsize=10)
        return doc.tobytes()


def loc():
    return AcquisitionLocation(url="https://publisher.example/item.pdf", provider="exa",
        representation_kind=RepresentationKind.PDF,
        metadata={"search_provider": "exa", "search_title": "SECRET SEARCH TITLE"})


def expected():
    return ExpectedBibliographicFields(title="Competition and cultural institutions", authors=["River, A."], year="2020")


@pytest.fixture
def setup(monkeypatch):
    safety = Mock(return_value=FileSafetyReport(SafetyVerdict.CLEAN, SafetyVerdict.CLEAN, SafetyVerdict.CLEAN))
    monkeypatch.setattr(pdf, "inspect_uploaded_pdf", safety)
    def install(body):
        responses = []
        def request(*args, **kwargs):
            response = httpx.Response(200, content=body, headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", loc().url))
            responses.append(response)
            return response
        fetch = Mock(side_effect=request)
        monkeypatch.setattr(pdf, "safe_request", fetch)
        return fetch, responses, safety
    return install


@pytest.mark.parametrize("title,outcome", [
    ("Competition and cultural institutions", "unavailable"),
    ("Botanical classification of tropical trees", "identity_rejected"),
])
def test_native_metadata_and_disposal(setup, title, outcome):
    fetch, responses, safety = setup(content(title))
    attempt = pdf.inspect(loc(), expected())
    assert attempt["outcome"] == outcome
    observation = attempt.get("landing_metadata_identity") or attempt["landing_metadata_observation"]
    assert observation["observed"]["title"] == title
    assert observation["identity_evidence_kind"].startswith("pdf_front_matter_")
    assert len(observation["content_sha256"]) == 64
    assert all(r.is_closed and not r.content for r in responses)
    assert "PRIVATE BODY" not in json.dumps(attempt) and "SECRET SEARCH" not in json.dumps(attempt)
    assert fetch.call_args.kwargs["max_bytes"] == 25 * 1024 * 1024
    assert fetch.call_args.kwargs["allowed_media_types"] == frozenset({"application/pdf"})
    safety.assert_called_once()


def test_missing_author_not_a_different_work(setup):
    setup(content("Botanical classification of tropical trees", byline=False))
    assert pdf.inspect(loc(), expected())["outcome"] == "identity_unconfirmed"


@pytest.mark.parametrize("blank_pages,observed", [(1, True), (2, True), (3, False)])
def test_cover_pages_within_native_front_matter_bound(setup, blank_pages, observed):
    with fitz.open() as doc, fitz.open(stream=content(), filetype="pdf") as title_page:
        for _ in range(blank_pages):
            doc.new_page()
        doc.insert_pdf(title_page)
        setup(doc.tobytes())
    attempt = pdf.inspect(loc(), expected())
    assert bool(attempt.get("landing_metadata_observation")) is observed
    assert attempt["outcome"] == "identity_unconfirmed"  # No first-page publication year.


def test_pdf_with_embedded_but_invisible_author_is_unresolved(setup):
    with fitz.open(stream=content(byline=False), filetype="pdf") as doc:
        doc.set_metadata({"author": "Anna River"})
        setup(doc.tobytes())
    assert pdf.inspect(loc(), expected())["reason_code"] == "identity_pdf_bibliography_unavailable"


def test_configured_smaller_byte_cap_is_preserved(setup, monkeypatch):
    fetch, _, _ = setup(content())
    monkeypatch.setattr(pdf.settings, "MAX_FILE_SIZE_MB", 5)
    pdf.inspect(loc(), expected())
    assert fetch.call_args.kwargs["max_bytes"] == 5 * 1024 * 1024


def test_safety_precedes_native_parsing(setup, monkeypatch):
    _, responses, safety = setup(content())
    safety.return_value = FileSafetyReport(SafetyVerdict.NOT_ASSESSED, SafetyVerdict.CLEAN, SafetyVerdict.NOT_ASSESSED)
    parse = Mock(); monkeypatch.setattr(pdf, "_observed_fields", parse)
    assert pdf.inspect(loc(), expected())["outcome"] == "identity_unconfirmed"
    parse.assert_not_called()
    assert not responses[0].content


def test_shared_two_pdf_limit_and_new_reference_reset(setup):
    fetch, _, _ = setup(content())
    @pdf.bounded_pdf_inspection
    def nested():
        return pdf.inspect(loc(), expected())
    @pdf.bounded_pdf_inspection
    def run():
        return [nested() for _ in range(5)]
    attempts = run()
    assert fetch.call_count == 2
    assert all(a["reason_code"] == "identity_pdf_limit" for a in attempts[2:])
    nested()
    assert fetch.call_count == 3


@pytest.mark.parametrize("error", [pdf.UnsafeUrlError("secret"), pdf.ResponseTooLargeError("secret"),
    pdf.ResponseMediaTypeError("secret"), pdf.FileSafetyUnavailable("secret"), TimeoutError("secret"),
    httpx.ConnectError("secret")])
def test_failures_are_unresolved_content_free(monkeypatch, error):
    monkeypatch.setattr(pdf, "safe_request", Mock(side_effect=error))
    attempt = pdf.inspect(loc(), expected())
    assert attempt["outcome"] not in {"identity_rejected", "acquired"}
    assert "secret" not in json.dumps(attempt)


def test_no_capacity_no_download(setup):
    fetch, _, _ = setup(content())
    with candidate_budget_scope(0):
        assert pdf.inspect(loc(), expected())["reason_code"] == "candidate_budget_exhausted"
    fetch.assert_not_called()


def test_interruption_disposes_body(setup, monkeypatch):
    _, responses, _ = setup(content())
    monkeypatch.setattr(pdf, "_observed_fields", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        pdf.inspect(loc(), expected())
    assert not responses[0].content


@pytest.mark.parametrize("title", ["Competition and cultural institutions", "Botanical classification of tropical trees"])
def test_transient_independent_metadata_only(setup, title):
    setup(content(title))
    attempt = pdf.inspect(loc(), expected())
    result = RetrievalResult(source_name="web_search", success=True, locations=[loc()],
        metadata={"search_retention_policy": BRAVE_TRANSIENT_POLICY, "location_attempts": [attempt]})
    finalize_transient_brave(result)
    text = json.dumps(result.metadata)
    assert "pdf_front_matter_" in text
    assert "SECRET SEARCH TITLE" not in text and "PRIVATE BODY TEXT" not in text
    assert not result.locations and not result.full_text and not result.representation


def test_resolver_pdf_observation_binds_without_source_admission(setup, monkeypatch):
    from tests.unit.test_bibliography_identity import resolver, reference
    from app.services.reference_discovery import assess_reference_discovery_trace
    setup(content())
    class Web:
        name = "web_search"
        capabilities = {"web_discovery"}
        _policy_providers = {"exa": object()}
        _policy_query_cache = {}
        def search_reference(self, **kwargs):
            location = loc()
            location.metadata["search_title"] = kwargs["title"]
            return RetrievalResult(source_name=self.name, success=True, locations=[location], metadata={
                "search_attempts": [{"provider": "exa", "query": kwargs["title"], "outcome": "results", "result_count": 1}]})
        def search_after_failed_candidates(self, **kwargs):
            return RetrievalResult(source_name=self.name, success=False, error="No further providers")
    result = resolver(monkeypatch, [Web()]).resolve_reference(reference(title=expected().title, year="2020"), identity_only=True)
    assert not result.full_text and not result.representation and not result.abstract
    trace = result.metadata["reference_discovery_trace"]
    assert trace["candidates"][0]["identity_evidence_kind"] == "pdf_front_matter_observation"
    assert assess_reference_discovery_trace(trace).ready
