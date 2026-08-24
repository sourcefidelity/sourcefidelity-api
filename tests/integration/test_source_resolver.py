from types import SimpleNamespace
from unittest.mock import Mock

import fitz
import httpx

import pytest

from app.config import settings
from app.services.retrieval.base import AcquisitionLocation, RetrievalResult, RetrievalSource
from app.services.retrieval.base import RepresentationKind
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

@pytest.fixture
def resolver() -> SourceResolver:
    instance = SourceResolver.__new__(SourceResolver)
    instance._backend = None
    instance._retrieval_sources = []
    instance._acquisition_capabilities = None
    return instance


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
    assert result.metadata == {"identity_rejected": True}


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
    assert result.metadata["identity_rejected"] is True


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

    assert result.success
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
