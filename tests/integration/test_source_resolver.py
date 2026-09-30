import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import fitz
import httpx

import pytest

from app.config import settings
from app.services.retrieval.base import AcquisitionLocation, RetrievalResult, RetrievalSource
from app.services.retrieval.base import RepresentationKind
from app.services.retrieval.base import SourceRepresentation
from app.services.file_safety import FileSafetyReport, SafetyVerdict
from app.services.source_resolver import (
    SourceResolutionError,
    SourceResolver,
    _document_kind_for_result,
    _expected_page_range,
    _normalize_cited_url,
    _url_prefers_pdf,
)
from app.services.source_validator import validate_retrieved_pdf
from app.services.source_type import SourceKindAssessment
from app.services.schemas import ParsedReference


class _MetadataOnlySource(RetrievalSource):
    name = "metadata"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)


class _DeferredSource(_MetadataOnlySource):
    name = "deferred"
    deferred = True

    def __init__(self) -> None:
        self.prefetch_dois = Mock(return_value=1)

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            abstract="Deferred abstract",
        )


class _CompletedNoMatchWebSource(_MetadataOnlySource):
    name = "web_search"
    capabilities = frozenset({"web_discovery"})

    @staticmethod
    def _miss(query: str) -> RetrievalResult:
        return RetrievalResult(
            source_name="web_search",
            success=False,
            error="no search results",
            metadata={
                "search_attempts": [
                    {
                        "provider": "bounded-control",
                        "query": query,
                        "outcome": "no_results",
                        "result_count": 0,
                    }
                ]
            },
        )

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return self._miss(f'"{doi}"')

    def search_by_title_author(
        self,
        title: str,
        author: str | None = None,
        year: str | None = None,
    ) -> RetrievalResult:
        return self._miss(f'"{title}"')


class _LocatedSource(_MetadataOnlySource):
    def __init__(self, name: str, url: str, *, is_best: bool = False) -> None:
        self.name = name
        self.url = url
        self.is_best = is_best

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title="Canonical work",
            year="2024",
            authors=["Rivera, Alex"],
            locations=[
                AcquisitionLocation(
                    url=self.url,
                    provider=self.name,
                    representation_kind=RepresentationKind.PDF,
                    is_best=self.is_best,
                )
            ],
            metadata={"provider_record": self.name},
        )


class _BarrierMetadataSource(_MetadataOnlySource):
    def __init__(self, name: str, barrier: threading.Barrier) -> None:
        self.name = name
        self.barrier = barrier

    def search_by_doi(self, doi: str) -> RetrievalResult:
        self.barrier.wait(timeout=2)
        return RetrievalResult(source_name=self.name, success=True, doi=doi)


@pytest.fixture
def resolver(monkeypatch) -> SourceResolver:
    from app.services.retrieval.google_books import BookMetadataSearch
    monkeypatch.setattr(
        "app.services.retrieval.google_books.GoogleBooksRetriever.search_metadata_result",
        lambda *args, **kwargs: BookMetadataSearch("title:test", "no_results"),
    )
    instance = SourceResolver.__new__(SourceResolver)
    instance._backend = None
    instance._retrieval_sources = []
    instance._acquisition_capabilities = None
    return instance


def test_structured_metadata_adapters_overlap_but_keep_configured_order(resolver):
    barrier = threading.Barrier(2)
    sources = [
        _BarrierMetadataSource("first", barrier),
        _BarrierMetadataSource("second", barrier),
    ]

    results = resolver._lookup_structured_sources(
        sources,
        "10.1234/example",
        "Example",
        "Author",
        "2024",
    )

    assert [source.name for source, _result in results] == ["first", "second"]
    assert all(result.success for _source, result in results)


def _resolver_response(content: bytes, content_type: str) -> httpx.Response:
    request = httpx.Request("GET", "https://resolver.example/10.1234/example")
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": content_type},
        request=request,
    )


def _pdf_with_text(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 555, 780), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def test_provider_metadata_supplies_kind_and_page_range():
    result = RetrievalResult(
        source_name="crossref",
        success=True,
        doi="10.1234/example",
        metadata={
            "provider_metadata": {
                "crossref": {
                    "message": {
                        "type": "book-chapter",
                        "page": "120-147",
                    }
                }
            }
        },
    )

    assert _document_kind_for_result(result, has_doi=True) == "chapter"
    assert _expected_page_range(result) == (120, 147)


def test_resolve_reference_preserves_parsed_bibliographic_type_contract(
    resolver: SourceResolver,
) -> None:
    reference = ParsedReference(
        author="Wu, T.",
        year="2010",
        title="The master switch",
        raw_ref="Wu, T. (2010). The master switch. Knopf.",
    )
    expected = RetrievalResult(source_name="test", success=False)
    resolver.resolve = Mock(return_value=expected)

    result = SourceResolver.resolve_reference(resolver, reference)

    assert result is expected
    assert resolver.resolve.call_args.kwargs["source_kind"] == "monograph"
    assert resolver.resolve.call_args.kwargs["source_kind_confidence"] == "high"
    assert resolver.resolve.call_args.kwargs["source_kind_evidence"]


def test_resolve_reference_attaches_non_decisive_route_trace_on_success(
    resolver: SourceResolver,
) -> None:
    cached = RetrievalResult(
        source_name="local_cache",
        success=True,
        full_text=b"%PDF-trace-control",
        title="Traceable scholarly source",
        authors=["Scholar"],
        year="2024",
    )
    resolver._check_local_cache = Mock(return_value=cached)
    reference = ParsedReference(
        reference_id="ref-trace-success",
        author="Scholar",
        year="2024",
        title="Traceable scholarly source",
    )

    result = resolver.resolve_reference(reference)
    trace = result.metadata["reference_discovery_trace"]

    assert trace["trace_version"] == "reference-discovery-trace-v1"
    assert trace["reference_id"] == "ref-trace-success"
    assert trace["attempts"][0]["route_category"] == "durable_repository"
    assert trace["attempts"][0]["outcome"] == "candidate_found"
    assert len(trace["candidates"]) == 1
    assert trace["candidates"][0]["attempt_id"] == trace["attempts"][0]["attempt_id"]
    assert trace["candidates_complete"] is True
    assert trace["outcome_derived"] is True
    assert result.metadata["reference_discovery"]["outcome"] == "confirmed"


def test_resolve_reference_failure_retains_each_attempt_without_deriving_absence(
    resolver: SourceResolver,
) -> None:
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._retrieval_sources = [_MetadataOnlySource()]
    reference = ParsedReference(
        reference_id="ref-trace-failure",
        author="Scholar",
        year="2024",
        title="Unlocated but traceable source",
    )

    with pytest.raises(SourceResolutionError) as raised:
        resolver.resolve_reference(reference)

    trace = raised.value.reference_discovery_trace
    assert trace is not None
    assert [attempt["route_category"] for attempt in trace["attempts"]] == [
        "durable_repository",
        "academic_adapter",
    ]
    assert trace["required_route_categories"] == [
        "academic_adapter",
        "bounded_web",
    ]
    assert trace["candidates_complete"] is True
    assert trace["outcome_derived"] is True
    assert raised.value.reference_discovery["outcome"] == "search_incomplete"


def test_live_unlocated_outcome_remains_suppressed_pending_real_control(
    resolver: SourceResolver,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", "configured-search-v1")
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._retrieval_sources = [
        _MetadataOnlySource(),
        _CompletedNoMatchWebSource(),
    ]
    reference = ParsedReference(
        reference_id="ref-unlocated-suppressed",
        author="Scholar",
        year="2024",
        title="A sufficiently specific scholarly source",
    )

    with pytest.raises(SourceResolutionError) as raised:
        resolver.resolve_reference(reference)

    trace = raised.value.reference_discovery_trace
    assert trace["candidates_complete"] is True
    assert trace["outcome_derived"] is False
    assert raised.value.reference_discovery is None
    assert any(
        "completed no-match outcome is suppressed" in limitation
        for limitation in trace["limitations"]
    )


@pytest.mark.integration
def test_accepted_cache_result_precedes_all_acquisition(resolver: SourceResolver) -> None:
    cached = RetrievalResult(
        source_name="local_cache",
        success=True,
        full_text=b"%PDF-cached representation",
        doi="10.1234/example",
    )
    resolver._check_local_cache = Mock(return_value=cached)
    resolver._try_student_url = Mock()

    result = resolver.resolve(
        doi="10.1234/example",
        student_url="https://student.example/source.pdf",
    )

    assert result is cached
    resolver._try_student_url.assert_not_called()


@pytest.mark.integration
def test_archive_reference_fails_before_cache_or_network(resolver: SourceResolver) -> None:
    resolver._check_local_cache = Mock()

    with pytest.raises(SourceResolutionError, match="Physical archive source"):
        resolver.resolve(
            title="Author papers",
            raw_ref="University Archive, Special Collections, Box 4, Folder 2.",
        )

    resolver._check_local_cache.assert_not_called()


@pytest.mark.integration
def test_traditional_media_skips_academic_adapters(resolver: SourceResolver) -> None:
    adapter = Mock()
    adapter.name = "academic_database"
    resolver._retrieval_sources = [adapter]
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._try_source = Mock()

    with pytest.raises(SourceResolutionError, match="Source not found"):
        resolver.resolve(
            title="Spirited Away",
            raw_ref="Miyazaki, H. (Director). (2001). Spirited Away [Film].",
        )

    resolver._try_source.assert_not_called()


@pytest.mark.integration
def test_malformed_doi_is_rejected_without_request(resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch) -> None:
    safe_request = Mock()
    monkeypatch.setattr("app.services.source_resolver.safe_request", safe_request)

    result = resolver._try_doi_resolver(
        "10.1234/example?url=http://internal.example",
        "Expected title",
    )

    assert result.success is False
    assert result.error == "Malformed DOI"
    safe_request.assert_not_called()


@pytest.mark.parametrize('direct_success', [True, False])
def test_supplied_doi_metadata_and_direct_resolution_precede_paid_search(resolver, monkeypatch, direct_success):
    from app.config import settings
    monkeypatch.setattr(settings, 'DOI_RESOLVER_URL', None)
    events = []
    web = Mock(); web.name = 'web_search'; web.capabilities = {'web_discovery'}
    metadata_adapter = Mock(); metadata_adapter.name = 'crossref'; metadata_adapter.capabilities = {'doi', 'title_author'}
    resolver._retrieval_sources = [metadata_adapter, web]
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name='local_cache', success=False))
    def metadata(*args):
        assert args[0] == [metadata_adapter]
        # An unconfirming DOI is followed by a title-only search of the
        # required indexes (reference-verification-v1), still before paid search.
        events.append('metadata' if args[1] == '10.1234/example' else 'title' if args[1] is None else 'other')
        return []
    resolver._lookup_structured_sources = metadata
    def direct(doi, title, **kw):
        events.append('direct')
        assert kw['public'] and doi == '10.1234/example'
        return RetrievalResult(source_name='doi_resolver', success=direct_success,
            full_text=b'%PDF-source' if direct_success else None, error=None if direct_success else 'timeout')
    resolver._try_doi_resolver = direct
    def paid(*args, **kwargs):
        events.append('paid')
        return RetrievalResult(source_name='web_search', success=False, error='No results')
    resolver._try_source = paid
    if direct_success:
        result = resolver.resolve(doi='10.1234/example', title='Specific article title', author='Author')
        assert result.full_text
        assert events == ['metadata', 'title', 'direct']
    else:
        with pytest.raises(SourceResolutionError):
            resolver.resolve(doi='10.1234/example', title='Specific article title', author='Author')
        assert events == ['metadata', 'title', 'direct', 'paid']


def test_public_doi_never_inherits_operator_trusted_prefix(resolver, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, 'DOI_RESOLVER_URL', 'https://operator.example/')
    fetch = Mock(return_value=_resolver_response(b'%PDF-test', 'application/pdf'))
    monkeypatch.setattr('app.services.source_resolver.safe_request', fetch)
    resolver._validated_doi_resolver_pdf = Mock(return_value=RetrievalResult(source_name='doi_resolver', success=False))
    result = resolver._try_doi_resolver('10.1234/example', 'Expected', public=True)
    assert fetch.call_args.args[0] == 'https://doi.org/10.1234/example'
    assert fetch.call_args.kwargs['trust_prefix'] is None
    resolver._validated_doi_resolver_pdf.assert_called_once()
    assert resolver._validated_doi_resolver_pdf.call_args.kwargs['access_type'] is None
    assert fetch.call_args.kwargs['max_bytes'] == settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024
    assert not result.success


def test_supplied_doi_url_is_not_fetched_twice_before_search(resolver, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, 'DOI_RESOLVER_URL', None)
    web = Mock(); web.name = 'web_search'; web.capabilities = {'web_discovery'}
    resolver._retrieval_sources = [web]
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name='local_cache', success=False))
    resolver._lookup_structured_sources = Mock(return_value=[])
    resolver._try_web_fetch = Mock(return_value=RetrievalResult(source_name='web_fetch', success=False, error='timeout'))
    resolver._try_doi_resolver = Mock(side_effect=AssertionError('Duplicate DOI fetch'))
    resolver._try_source = Mock(return_value=RetrievalResult(source_name='web_search', success=False, error='No results'))
    with pytest.raises(SourceResolutionError):
        resolver.resolve(doi='10.1234/example', title='Specific article',
                         student_url='https://doi.org/10.1234/example')
    resolver._try_web_fetch.assert_called_once()
    resolver._try_doi_resolver.assert_not_called()


def test_public_doi_html_requires_observed_identity_not_body_doi(resolver, monkeypatch):
    html = b'<html><head><title>Expected article title</title></head><body>10.1234/example</body></html>'
    monkeypatch.setattr('app.services.source_resolver.safe_request', Mock(return_value=_resolver_response(html, 'text/html')))
    resolver._landing_metadata_identity = Mock(return_value=None)
    result = resolver._try_doi_resolver('10.1234/example', 'Expected article title', public=True)
    assert not result.success and not result.full_text
    assert 'identity remains unconfirmed' in result.error


@pytest.mark.integration
def test_doi_resolver_rejects_login_or_menu_html(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    login_html = (
        "<html><head><title>Library sign in</title></head><body><main>"
        + "Sign in to select a database or search the library catalogue. " * 20
        + "</main></body></html>"
    ).encode()
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request",
        Mock(return_value=_resolver_response(login_html, "text/html; charset=utf-8")),
    )

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is False
    assert "title does not match" in (result.error or "")
    assert result.metadata['identity_rejected'] is True
    assert set(result.metadata) == {'identity_rejected', 'operation_timing', 'operation_timings'}
    assert result.metadata['operation_timing']['elapsed_seconds'] >= 0


@pytest.mark.integration
def test_doi_resolver_accepts_only_identity_matching_html(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    article_html = (
        "<html><head>"
        '<meta name="citation_title" content="Identity-Gated Resolver Article">'
        "<title>Identity-Gated Resolver Article</title></head><body><article>"
        "<h1>Identity-Gated Resolver Article</h1>"
        "<p>DOI 10.1234/example. "
        + "This is the complete scholarly article text used for verification. " * 20
        + "</p></article></body></html>"
    ).encode()
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request",
        Mock(return_value=_resolver_response(article_html, "text/html; charset=utf-8")),
    )

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is True
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PLAIN_TEXT
    assert result.representation.original_kind is RepresentationKind.HTML
    assert result.locations[0].provider == "doi_resolver"
    assert result.locations[0].representation_kind is RepresentationKind.HTML
    assert result.metadata["identity_confidence"] == "high"


@pytest.mark.integration
def test_doi_resolver_rejects_wrong_pdf_through_common_validator(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    wrong_pdf = _pdf_with_text(
        "An unrelated publication by Different Author, 2018. " * 15
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request",
        Mock(return_value=_resolver_response(wrong_pdf, "application/pdf")),
    )

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is False
    assert "failed source identity" in (result.error or "")
    # This fixture supplies unrelated body wording, not independently observed
    # conflicting bibliographic fields. It stays unusable without claiming an
    # affirmative source conflict from missing identity observations.
    assert result.metadata["identity_unconfirmed"] is True
    assert result.metadata["identity_reason_code"] == "identity_insufficient_observations"
    assert result.full_text is None


@pytest.mark.integration
def test_doi_resolver_accepts_identity_matching_pdf_as_typed_location(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    matching_pdf = _pdf_with_text(
        "Identity-Gated Resolver Article\nDOI 10.1234/example\n"
        + "Complete article content for verification. " * 20
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request",
        Mock(return_value=_resolver_response(matching_pdf, "application/pdf")),
    )

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is True
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PDF
    assert result.locations[0].representation_kind is RepresentationKind.PDF
    assert result.metadata["identity_confidence"] == "high"
    assert result.metadata["completeness"] == "uncertain"


@pytest.mark.integration
def test_doi_resolver_rejects_advertised_range_excerpt(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    excerpt = _pdf_with_text(
        "Proceedings of Example Research, pages 12076-12100\n"
        "Identity-Gated Resolver Article\nDOI 10.1234/example\n"
        "Only the opening page of this article is present."
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request",
        Mock(return_value=_resolver_response(excerpt, "application/pdf")),
    )

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is False
    assert "completeness" in (result.error or "")
    assert result.metadata["completeness_rejected"] is True


@pytest.mark.integration
def test_doi_resolver_landing_page_discovers_and_validates_pdf(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    landing_html = (
        "<html><head>"
        '<meta name="citation_title" content="Identity-Gated Resolver Article">'
        '<meta name="citation_pdf_url" content="https://publisher.example/article.pdf">'
        "</head><body>Article landing page</body></html>"
    ).encode()
    matching_pdf = _pdf_with_text(
        "Identity-Gated Resolver Article\nDOI 10.1234/example\n"
        + "Complete article content for verification. " * 20
    )
    request = Mock(side_effect=[
        _resolver_response(landing_html, "text/html; charset=utf-8"),
        _resolver_response(matching_pdf, "application/pdf"),
    ])
    monkeypatch.setattr("app.services.source_resolver.safe_request", request)

    result = resolver._try_doi_resolver(
        "10.1234/example", "Identity-Gated Resolver Article"
    )

    assert result.success is True
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PDF
    assert result.locations[0].landing_page_url is not None
    assert result.locations[0].metadata["discovery_signal"] == "citation_pdf_url"
    assert request.call_count == 2


@pytest.mark.integration
def test_bounded_capabilities_suppress_student_url(resolver: SourceResolver) -> None:
    resolver._acquisition_capabilities = {"academic_adapters"}
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._try_student_url = Mock()

    with pytest.raises(SourceResolutionError, match="Source not found"):
        resolver.resolve(
            title="Example source",
            student_url="https://student.example/source.pdf",
        )

    resolver._try_student_url.assert_not_called()


@pytest.mark.integration
def test_deferred_provider_batches_only_explicit_unresolved_dois(
    resolver: SourceResolver,
) -> None:
    ordinary = _MetadataOnlySource()
    deferred = _DeferredSource()
    resolver._retrieval_sources = [ordinary, deferred]

    prefetched = resolver.prefetch_deferred_dois(["10.1234/unresolved"])
    result = resolver.resolve_deferred_doi("10.1234/unresolved", "Expected title")

    assert prefetched == {"deferred": 1}
    deferred.prefetch_dois.assert_called_once_with(["10.1234/unresolved"])
    assert result.source_name == "deferred"
    assert result.abstract == "Deferred abstract"


@pytest.mark.integration
def test_canonical_graph_falls_through_locations_and_preserves_provenance(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _LocatedSource(
        "first_provider", "https://first.example/paper.pdf", is_best=True
    )
    second = _LocatedSource(
        "second_provider", "https://second.example/paper.pdf"
    )
    resolver._retrieval_sources = [first, second]
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._safe_download = Mock(
        side_effect=[ValueError("location unavailable"), b"%PDF-second"]
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        lambda *args, **kwargs: SimpleNamespace(
            identity_confidence="high",
            reason="exact DOI in acquired PDF",
            completeness="complete",
        ),
    )

    result = resolver.resolve(
        doi="10.1234/example",
        title="Canonical work",
        author="Rivera, Alex",
        year="2024",
    )

    assert result.source_name == "second_provider"
    assert result.full_text == b"%PDF-second"
    assert result.metadata["canonical_work"]["accepted_providers"] == [
        "first_provider",
        "second_provider",
    ]
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "unavailable",
        "acquired",
    ]
    assert [attempt["provider"] for attempt in result.metadata["location_attempts"]] == [
        "first_provider",
        "second_provider",
    ]


@pytest.mark.integration
def test_canonical_graph_rejects_wrong_download_and_resumes_remaining_locations(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver._retrieval_sources = [
        _LocatedSource(
            "wrong_provider",
            "https://wrong.example/paper.pdf",
            is_best=True,
        ),
        _LocatedSource(
            "correct_provider",
            "https://correct.example/paper.pdf",
        ),
    ]
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._safe_download = Mock(
        side_effect=[b"%PDF-wrong-work", b"%PDF-correct-work"]
    )
    validator = Mock(
        side_effect=[
            SimpleNamespace(
                identity_confidence="rejected",
                reason="downloaded representation is a different work",
                completeness="skipped",
            ),
            SimpleNamespace(
                identity_confidence="high",
                reason="exact DOI belongs to the downloaded representation",
                completeness="complete",
            ),
        ]
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        validator,
    )

    result = resolver.resolve(
        doi="10.1234/example",
        title="Canonical work",
        author="Rivera, Alex",
        year="2024",
    )

    assert result.source_name == "correct_provider"
    assert result.full_text == b"%PDF-correct-work"
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "identity_rejected",
        "acquired",
    ]
    assert validator.call_count == 2


@pytest.mark.integration
def test_type_conflict_rejects_book_review_and_resumes_retrieval_graph(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver._retrieval_sources = [
        _LocatedSource("review_host", "https://wrong.example/review.pdf", is_best=True),
        _LocatedSource("book_host", "https://correct.example/book.pdf"),
    ]
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._safe_download = Mock(side_effect=[b"%PDF-review", b"%PDF-book"])
    validator = Mock(
        side_effect=[
            SimpleNamespace(
                identity_confidence="rejected",
                reason="Bibliographic type conflict — expected monograph, observed book_review",
                completeness="skipped",
                observed_source_kind="book_review",
                source_kind_verdict="incompatible",
            ),
            SimpleNamespace(
                identity_confidence="high",
                reason="matching book identity",
                completeness="complete",
                observed_source_kind="monograph",
                source_kind_verdict="compatible",
            ),
        ]
    )
    monkeypatch.setattr("app.services.source_resolver.validate_retrieved_pdf", validator)

    result = resolver.resolve(
        doi="10.1234/example",
        title="Canonical work",
        author="Rivera, Alex",
        year="2024",
        source_kind="monograph",
        source_kind_confidence="high",
    )

    assert result.source_name == "book_host"
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "type_rejected",
        "acquired",
    ]
    assert validator.call_count == 2


@pytest.mark.integration
def test_legacy_cache_requires_fresh_high_confidence_validation(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver._backend = Mock()
    resolver._backend.download.return_value = b"%PDF-legacy-uncertain"
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_ENABLED", False)
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        Mock(
            return_value=SimpleNamespace(
                identity_confidence="medium",
                reason="title appears but ownership is unconfirmed",
                completeness="complete",
            )
        ),
    )

    result = resolver._check_local_cache(
        None,
        None,
        "Expected work",
        author="Rivera, Alex",
        year="2024",
    )

    assert result.success is False


def test_validator_does_not_accept_medium_identity_for_automatic_use() -> None:
    pdf = _pdf_with_text("Expected work\nGeneral discussion without author metadata.")

    validation = validate_retrieved_pdf(
        pdf,
        expected_title="Expected work",
        skip_completeness=True,
    )

    assert validation.identity_confidence == "medium"
    assert validation.accept is False
    assert validation.reason.startswith("Needs review:")


@pytest.mark.integration
def test_opt_in_pure_scan_preflight_replaces_evidence_with_bound_derivative(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = SourceRepresentation(
        kind=RepresentationKind.PDF,
        media_type="application/pdf",
        content=b"%PDF-pure-scan-parent",
        source_url="https://example.org/source.pdf",
        completeness="complete",
    )
    derivative = SourceRepresentation(
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
        content=b"validated OCR derivative text",
        source_url=parent.source_url,
        original_kind=RepresentationKind.PDF,
        completeness="complete",
        metadata={"ocr_derivative_version": "local-pdf-ocr-derivative-v1"},
    )
    result = RetrievalResult(
        source_name="test",
        success=True,
        representation=parent,
        title="Expected title",
        authors=["Scholar"],
    )
    safety = FileSafetyReport(
        verdict=SafetyVerdict.CLEAN,
        structural_verdict=SafetyVerdict.CLEAN,
        malware_verdict=SafetyVerdict.CLEAN,
    )
    prepared = SimpleNamespace(
        status="ready",
        reason="Accepted OCR derivative",
        parent=parent,
        derivative=derivative,
        derivative_record=SimpleNamespace(
            parent_content_sha256=hashlib.sha256(parent.content).hexdigest(),
            content_sha256=hashlib.sha256(derivative.content).hexdigest(),
            manifest_sha256="b" * 64,
            manifest={
                "version": "local-pdf-ocr-derivative-v1",
                "engine": "tesseract",
                "engine_version": "test",
                "language": "eng",
            },
        ),
        validation=SimpleNamespace(
            identity_confidence="high",
            reason="Accepted OCR derivative",
            completeness="complete",
            text_quality="scan_ocr",
            observed_source_kind="journal_article",
            source_kind_verdict="compatible",
        ),
        page_labels=("1",),
        page_mapping_method="observed_only",
        page_mapping_sha256="c" * 64,
    )
    monkeypatch.setattr(settings, "PURE_SCAN_OCR_ENABLED", True)
    monkeypatch.setattr(resolver, "_durable_repository_active", lambda: True)
    monkeypatch.setattr(
        "app.services.source_resolver.inspect_uploaded_pdf", lambda _content: safety
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        lambda *_args, **_kwargs: SimpleNamespace(
            identity_confidence="skipped",
            reason="pure scan",
            completeness="skipped",
            text_quality="pure_scan",
            observed_source_kind="unknown",
            source_kind_verdict="unknown",
        ),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.prepare_pure_scan_ocr",
        lambda *_args, **_kwargs: prepared,
    )

    accepted, outcome, _reason = resolver._preflight_acquired_representation(
        result,
        expected_doi=None,
        expected_title="Expected title",
        expected_author="Scholar",
        expected_year="2024",
        expected_source_kind=SourceKindAssessment(
            "journal_article", "high", ("test",)
        ),
    )

    assert accepted and outcome == "acquired"
    assert result.representation is derivative
    assert result.parent_representation is parent
    assert result.metadata["text_quality"] == "scan_ocr"
    assert result.metadata["ocr_derivative"]["parent_file_safety"]["verdict"] == "clean"


@pytest.mark.integration
def test_bounded_capabilities_suppress_publisher_constructor(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver._acquisition_capabilities = {"academic_adapters"}
    source = _MetadataOnlySource()
    publisher_download = Mock()
    monkeypatch.setattr(
        "app.services.publisher_urls.try_download_publisher_pdf", publisher_download
    )
    result = RetrievalResult(
        source_name="metadata",
        success=True,
        doi="10.1234/example",
        title="Example source",
    )

    resolved = resolver._download_and_cache(source, result)

    assert resolved.full_text is None
    publisher_download.assert_not_called()


@pytest.mark.integration
def test_web_url_uses_html_route_before_pdf(resolver: SourceResolver) -> None:
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    web_result = RetrievalResult(
        source_name="web_fetch", success=True, abstract="Readable web article"
    )
    resolver._try_web_fetch = Mock(return_value=web_result)
    resolver._try_student_url = Mock()

    result = resolver.resolve(
        title="News article",
        student_url="https://news.example/article/123",
    )

    assert result is web_result
    resolver._try_student_url.assert_not_called()


@pytest.mark.integration
def test_web_fetch_returns_typed_html_text_and_limited_evidence_compatibility(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = " ".join(["Readable article sentence with substantive evidence."] * 20)
    response = httpx.Response(
        200,
        text=(
            "<html><head><meta name='citation_title' content='News article'></head>"
            f"<body><article>{body}</article></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://news.example/article"),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request", lambda *args, **kwargs: response
    )

    result = resolver._try_web_fetch("https://news.example/article", "News article")

    assert not result.success  # Title alone cannot stop the discovery chain.
    assert result.metadata['identity_confidence'] != 'high'
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PLAIN_TEXT
    assert result.representation.original_kind is RepresentationKind.HTML
    assert result.abstract == result.representation.content.decode("utf-8")


@pytest.mark.integration
def test_web_fetch_rejects_news_article_when_citation_expects_report(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = " ".join(["Readable news coverage of the agency report."] * 20)
    response = httpx.Response(
        200,
        text=(
            "<html><head><meta name='citation_title' content='Annual findings'>"
            "<script type='application/ld+json'>{\"@type\":\"NewsArticle\"}</script>"
            f"</head><body><article>{body}</article></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://news.example/annual-findings"),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request", lambda *args, **kwargs: response
    )

    result = resolver._try_web_fetch(
        "https://news.example/annual-findings",
        "Annual findings",
        expected_source_kind=SourceKindAssessment(
            "report", "high", ("explicit report number",)
        ),
    )

    assert result.success is False
    assert result.metadata["observed_source_kind"] == "news_article"
    assert result.metadata["source_kind_verdict"] == "incompatible"


@pytest.mark.integration
def test_web_fetch_rejects_generic_landing_page_when_citation_expects_monograph(
    resolver: SourceResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = " ".join(["Repository metadata describing the cited book."] * 20)
    response = httpx.Response(
        200,
        text=(
            "<html><head><meta name='citation_title' content='The cited book'></head>"
            f"<body><main>{body}</main></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://repository.example/books/63"),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request", lambda *args, **kwargs: response
    )

    result = resolver._try_web_fetch(
        "https://repository.example/books/63",
        "The cited book",
        expected_source_kind=SourceKindAssessment(
            "monograph", "high", ("terminal publisher structure",)
        ),
    )

    assert result.success is False
    assert result.metadata["observed_source_kind"] == "webpage"
    assert result.metadata["source_kind_verdict"] == "unknown"


@pytest.mark.integration
def test_explicit_pdf_url_uses_pdf_route_before_html(resolver: SourceResolver) -> None:
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    pdf_result = RetrievalResult(
        source_name="student_url", success=True, full_text=b"%PDF-example"
    )
    resolver._try_student_url = Mock(return_value=pdf_result)
    resolver._try_web_fetch = Mock()

    result = resolver.resolve(
        title="Paper",
        student_url="https://repository.example/paper.pdf",
    )

    assert result is pdf_result
    resolver._try_web_fetch.assert_not_called()


def test_pdf_url_detection_is_explicit_and_conservative() -> None:
    assert _url_prefers_pdf("https://example.org/paper.PDF?download=1") is True
    assert _url_prefers_pdf("https://example.org/download?format=pdf") is True
    assert _url_prefers_pdf("https://example.org/news/pdf-policy") is False


def test_scheme_less_www_citation_is_normalized_conservatively() -> None:
    assert _normalize_cited_url("www.example.org/article") == "https://www.example.org/article"
    assert _normalize_cited_url("https://example.org/article") == "https://example.org/article"
    assert _normalize_cited_url("javascript:alert(1)") == "javascript:alert(1)"


@pytest.mark.integration
def test_library_locator_is_not_treated_as_source_content(resolver: SourceResolver) -> None:
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._try_web_fetch = Mock()
    resolver._try_student_url = Mock()

    with pytest.raises(SourceResolutionError, match="Library locator"):
        resolver.resolve(
            title="Catalogued book",
            student_url=(
                "https://search-ebscohost-com.proxy.example.edu/login.aspx"
                "?direct=true&db=example&AN=example.0000259848"
            ),
        )

    resolver._try_web_fetch.assert_not_called()
    resolver._try_student_url.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "allowed"),
    [
        ("unknown", True),
        ("monograph", True),
        ("edited_collection", True),
        ("book_section", True),
        ("journal_article", False),
        ("conference_paper", False),
        ("report", False),
        ("webpage", False),
    ],
)
def test_public_domain_fallback_is_bounded_to_book_shaped_or_unknown_works(
    kind: str, allowed: bool
) -> None:
    assessment = SourceKindAssessment(kind, "high" if kind != "unknown" else "unknown")

    assert SourceResolver._public_domain_fallback_allowed(assessment) is allowed
