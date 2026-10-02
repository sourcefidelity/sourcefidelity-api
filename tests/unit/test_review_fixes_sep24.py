"""Regressions for the 2026-09-24 code-review fixes."""
import pytest
from pydantic import BaseModel, ValidationError


class TestDoiRequestSegment:
    """A DOI comes from parsed reference text and is untrusted.

    It was interpolated straight into the request path by both the DataCite
    and Crossref adapters, so "?", "#" or "/../" inside it rewrote the request
    a provider received, while a legitimate DOI containing "#" or "?" (both
    allowed by the DOI syntax) was cut at the query boundary and looked up as
    a different identifier.
    """

    def test_query_and_fragment_characters_are_encoded(self):
        from app.services.doi_cache import doi_request_segment

        assert doi_request_segment("10.1234/abc?page[size]=1000") == (
            "10.1234/abc%3Fpage%5Bsize%5D%3D1000"
        )
        assert doi_request_segment("10.1002/(SICI)1097-0258#x") == (
            "10.1002/%28sici%291097-0258%23x"
        )

    @pytest.mark.parametrize("doi", [
        "10.1234/../other", "10.1234/a/./b", "10.1234//x", "10.1234/", "not a doi", "", None,
    ])
    def test_path_climbing_and_non_dois_are_refused(self, doi):
        from app.services.doi_cache import doi_request_segment

        assert doi_request_segment(doi) is None

    def test_the_suffix_separator_survives(self):
        """Legacy arXiv DOIs carry a slash inside the suffix."""
        from app.services.doi_cache import doi_request_segment

        assert doi_request_segment("10.48550/arXiv.math/0309136") == "10.48550/arxiv.math/0309136"
        assert doi_request_segment("https://doi.org/10.5555/ok") == "10.5555/ok"

    def test_adapters_do_not_call_out_for_a_non_doi(self, monkeypatch):
        from app.services.retrieval.datacite import DataCiteRetriever
        from app.services.retrieval.crossref import CrossrefRetriever

        called = []
        datacite = DataCiteRetriever()
        monkeypatch.setattr(datacite, "_request", lambda *a, **k: called.append("datacite"))
        assert datacite.search_by_doi("10.1234/../x").success is False
        crossref = CrossrefRetriever()
        monkeypatch.setattr(crossref, "_get", lambda *a, **k: called.append("crossref"))
        assert crossref.search_by_doi("not a doi").success is False
        assert called == []


class TestRouteErrorCode:
    """The field exists to tell a local network loss from a provider fault.

    The first version tested "connect" before "timeout" (so ConnectTimeout was
    a connection error), and matched bare digits anywhere in the text (so
    "Expected 4000 bytes" was an HTTP 400). Status codes are read only from the
    two shapes the adapters produce.
    """

    @pytest.mark.parametrize("error,expected", [
        ("ConnectTimeout", "timeout"),
        ("ReadTimeout", "timeout"),
        ("connect_error", "connect_error"),
        ("http_503", "http_server_error"),
        ("DataCite HTTP 404", "http_client_error"),
        ("ERIC HTTP 429", "rate_limited"),
        ("http_401", "access_restricted"),
        ("Expected 4000 bytes", "unclassified"),
        ("value 500 exceeded", "unclassified"),
        ("provider_call_skipped", "circuit_open"),
        ("Name or service not known (dns)", "network_error"),
        ("429 rate limit exceeded", "rate_limited"),
        ("response_invalid", "invalid_response"),
        ("", None),
        (None, None),
    ])
    def test_classification(self, error, expected):
        from app.services.source_resolver import _route_error_code

        class Result:
            pass

        result = Result()
        result.error = error
        assert _route_error_code(result) == expected


class TestFailureRecordReachesEveryPath:
    """The targeted-refresh rollback kept only the exception class name.

    That is the path the owner's queued rerun uses, so a failure there would
    have been exactly as undiagnosable as the one the detail was added to fix.
    """

    def test_a_validation_error_carries_its_field(self):
        from app.services.paper_workflow import _failure_record

        class M(BaseModel):
            cited_author_label: str

        try:
            M(cited_author_label=None)
        except ValidationError as exc:
            record = _failure_record(exc)
        assert record.startswith("ValidationError (")
        assert "cited_author_label" in record

    def test_the_rollback_receives_the_same_record(self, monkeypatch):
        from app.services import paper_workflow as pw

        class Job:
            id = "job"
            status = "running"
            upload_evidence = {pw.TARGETED_SOURCE_REFRESH_KEY: {"attempt_id": "a"}}

        received = {}
        monkeypatch.setattr(pw, "_job", lambda session, job_id: Job())
        monkeypatch.setattr(
            pw, "rollback_targeted_source_refresh",
            lambda session, job_id, *, error_code, commit=True: received.setdefault("code", error_code),
        )

        class M(BaseModel):
            field: int

        with pytest.raises(ValidationError) as caught:
            M(field="x")
        pw.fail_paper_job(object(), object(), "job", caught.value)
        assert received["code"] == pw._failure_record(caught.value)
        assert "field" in received["code"]


BIG = "X" * 50_000


class TestBoundedFieldsAreRecorded:
    """Review finding 5: bounding validators altered evidence with no provenance.

    AGENTS.md requires findings to expose their limitations. Each model a run
    constructs from paper or provider data now records the names of fields the
    application truncated or degraded in `bounded_fields`, serialized only when
    non-empty so an untouched record and every stored hash of one is unchanged.
    """

    def test_an_untouched_record_dumps_exactly_as_before(self):
        from app.services.schemas import ParsedReference
        from app.services.reference_discovery import ExpectedBibliographicFields

        assert "bounded_fields" not in ParsedReference(raw_ref="r", publisher="ok").model_dump()
        assert "bounded_fields" not in ExpectedBibliographicFields(title="t").model_dump()

    def test_every_altering_validator_names_what_it_altered(self):
        from app.services.schemas import CitationMarkerMember, ParsedReference
        from app.services.reference_discovery import ExpectedBibliographicFields
        from app.services.verification_evidence import ClaimEvidence, CitationSourceBinding
        from app.services.relationship_stage_evaluation import StageCandidateCase

        assert ParsedReference(raw_ref="r", container_title=BIG).bounded_fields == ["container_title"]
        assert ExpectedBibliographicFields(title=BIG, authors=["A"] * 500).bounded_fields == ["title", "authors"]
        assert ClaimEvidence(
            claim_id="c", paper_version_id="p", text="t", page_locator=BIG, reference_ids=["r"]
        ).bounded_fields == ["page_locator"]
        binding = CitationSourceBinding(
            status="exact", reference_id="r", cited_author_label="A" * 300,
            marker_text="M" * 2000, marker_local_start=0, marker_local_end=2000,
        )
        assert binding.status == "unresolved"
        assert binding.bounded_fields == ["marker_text", "cited_author_label"]
        assert CitationMarkerMember(text=BIG, local_start=3, local_end=3 + len(BIG)).bounded_fields == ["text"]

    def test_the_record_survives_a_round_trip(self):
        from app.services.schemas import ParsedReference

        altered = ParsedReference(raw_ref="r", container_title=BIG)
        assert ParsedReference(**altered.model_dump()).bounded_fields == ["container_title"]


class TestNormalizationPrecedesTheCache:
    """Review finding 6: the cache was written before identifier normalization,
    so it held the polluted title and an empty DOI and the repair was redone on
    every hit."""

    def test_the_cache_receives_the_corrected_reference(self, monkeypatch):
        from app.services import reference_parser as rp
        from app.services import doi_cache

        cached = []
        monkeypatch.setattr(doi_cache, "cache_reference", lambda **kw: cached.append(kw))
        monkeypatch.setattr(rp.settings, "CACHE_ENABLED", True)
        monkeypatch.setattr(doi_cache, "get_cached_reference", lambda *a, **k: None, raising=False)
        raw = "Wu, Y. (2016). GNMT system: Bridging the gap. arXiv:1609.08144"
        results = rp._extract_fields_regex_first([raw], "apa", use_llm_fallback=False)
        parsed = results[0]
        assert parsed is not None
        assert parsed.doi == "10.48550/arXiv.1609.08144"
        assert "arxiv" not in parsed.title.lower()
        if cached:
            assert cached[0]["doi"] == "10.48550/arXiv.1609.08144"
            assert "arxiv" not in cached[0]["data"]["title"].lower()


class TestOneDetailShape:
    """Review finding 7: two definitions of what a safe detail looks like."""

    def test_the_exception_and_the_helper_agree(self):
        from app.log_safety import bounded_detail, safe_exception_detail
        from app.services.verification_report import ReportAuthorizationError

        detail = "stored=body_prose; derived=reference_list; method=bm25_concept"
        exc = ReportAuthorizationError("Candidate passage role is not application-derived", detail=detail)
        assert exc.detail == detail
        assert safe_exception_detail(exc) == detail
        excerpt = "student wrote: “the regulator’s finding”"
        assert bounded_detail(excerpt) is None
        assert ReportAuthorizationError("m", detail=excerpt).detail is None

    def test_the_limit_reader_is_defined_once(self):
        import inspect

        from app.services import reference_discovery

        assert not hasattr(reference_discovery, "_declared_max_length")
        assert "declared_max_length" in inspect.getsource(reference_discovery)


class TestOneAuthorLimit:
    """Review finding 8: author caps duplicated at call sites as magic numbers."""

    def test_adapters_and_the_model_share_the_constant(self):
        from app.services.retrieval.base import OBSERVED_AUTHOR_LIMIT
        from app.services.retrieval import datacite, eric, open_library, semantic_scholar
        from app.services import reference_discovery

        for module in (datacite, eric, open_library, semantic_scholar, reference_discovery):
            assert module.OBSERVED_AUTHOR_LIMIT is OBSERVED_AUTHOR_LIMIT
        assert not hasattr(reference_discovery, "_bounded_observed_authors")
        creators = [{"name": f"A{n}"} for n in range(OBSERVED_AUTHOR_LIMIT + 50)]
        assert len(datacite._attribute_authors({"creators": creators})) == OBSERVED_AUTHOR_LIMIT


class TestClassifierBoundaries:
    """Review findings 9 and 10."""

    def test_a_url_path_that_looks_like_a_doi_is_still_a_webpage(self):
        from app.services.source_type import classify_reference_source_kind

        raw = "Someone, A. (2022). A page. https://example.org/reports/10.2019/summary"
        assert classify_reference_source_kind(raw).kind == "webpage"

    def test_an_actual_doi_statement_still_keeps_the_reference_reviewable(self):
        from app.services.source_type import classify_reference_source_kind

        for raw in ("Smith, J. (2020). Invented. https://doi.org/10.9999/fake.2020",
                    "Smith, J. (2020). Invented. doi:10.9999/fake.2020"):
            assert classify_reference_source_kind(raw).kind != "webpage", raw

    def test_org_authored_work_with_a_publisher_statement_stays_reviewable(self):
        """The rule once excluded this, removing org-authored grey literature -
        including fabricated grey literature - from review altogether."""
        from app.services.source_type import classify_reference_source_kind

        raw = "World Health Organization (2021). Global status report on alcohol. World Health Organization."
        # Owner decision 2026-10-02: an organisation's dated, titled document is a searchable report.
        assert classify_reference_source_kind(raw).kind == "report"

    def test_a_document_with_nothing_after_its_title_is_now_a_report(self):
        """Owner decision 2026-10-02 reverses 2026-09-23: such documents are online and searched."""
        from app.services.source_type import classify_reference_source_kind

        for raw in ("Northfield State University (2015). NSU language policy.",
                    "Ministry of Education (2019). National curriculum framework."):
            assert classify_reference_source_kind(raw).kind == "report", raw
