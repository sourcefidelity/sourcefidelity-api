from unittest.mock import Mock

import httpx

from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.retrieval.crossref import _parse_full_text_links
from app.services.retrieval.landing_page import discover_scholarly_locations
from app.services.retrieval.openalex import OpenAlexRetriever, _parse_locations
from app.services.source_resolver import SourceResolver
from app.services.source_resolver import _verify_text_content_identity
from app.services.source_type import SourceKindAssessment


def _resolver() -> SourceResolver:
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = None
    return resolver


def test_legacy_result_bytes_are_typed_without_assuming_everything_is_pdf() -> None:
    pdf = RetrievalResult(source_name="test", success=True, full_text=b"%PDF-example")
    text = RetrievalResult(source_name="test", success=True, full_text=b"article text")

    assert pdf.representation is not None
    assert pdf.representation.kind is RepresentationKind.PDF
    assert text.representation is not None
    assert text.representation.kind is RepresentationKind.PLAIN_TEXT


def test_representation_populates_compatibility_field_and_location() -> None:
    result = RetrievalResult(
        source_name="test",
        success=True,
        full_text_url="https://example.org/article",
        representation=SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT,
            media_type="text/plain",
            content=b"article text",
            original_kind=RepresentationKind.HTML,
        ),
    )

    assert result.full_text == b"article text"
    assert result.locations[0].representation_kind is RepresentationKind.PLAIN_TEXT


def test_openalex_preserves_multiple_pdf_and_landing_locations() -> None:
    data = {
        "best_oa_location": {
            "pdf_url": "https://repository.example/article.pdf",
            "landing_page_url": "https://repository.example/article",
            "is_oa": True,
            "version": "acceptedVersion",
            "source": {"id": "S1", "display_name": "Repository"},
        },
        "primary_location": {
            "landing_page_url": "https://publisher.example/article",
            "version": "publishedVersion",
            "source": {"id": "S2", "display_name": "Publisher"},
        },
        "locations": [
            {
                "pdf_url": "https://mirror.example/article.pdf",
                "landing_page_url": "https://mirror.example/article",
                "is_oa": True,
                "source": {"id": "S3", "display_name": "Mirror"},
            }
        ],
    }

    locations = _parse_locations(data)

    assert {location.url for location in locations} == {
        "https://repository.example/article.pdf",
        "https://repository.example/article",
        "https://publisher.example/article",
        "https://mirror.example/article.pdf",
        "https://mirror.example/article",
    }
    assert sum(location.representation_kind is RepresentationKind.PDF for location in locations) == 2
    assert locations[0].is_best


def test_openalex_accepts_nullable_nested_records() -> None:
    result = OpenAlexRetriever()._parse_work(
        {
            "doi": "https://doi.org/10.1234/example",
            "title": "Nullable OpenAlex record",
            "publication_year": 2024,
            "authorships": [{"author": None}, None],
            "best_oa_location": {
                "landing_page_url": "https://publisher.example/article",
                "source": None,
            },
            "primary_location": None,
            "locations": [None],
            "abstract_inverted_index": {"Abstract": [0], "gap": None},
        }
    )

    assert result.success
    assert result.authors == []
    assert result.locations[0].url == "https://publisher.example/article"
    assert result.locations[0].host_type is None
    assert result.abstract == "Abstract"


def test_crossref_preserves_tdm_format_version_and_license() -> None:
    locations = _parse_full_text_links(
        {
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
            "link": [
                {
                    "URL": "https://publisher.example/full.xml",
                    "content-type": "application/xml",
                    "content-version": "vor",
                    "intended-application": "text-mining",
                },
                {
                    "URL": "https://publisher.example/full.pdf",
                    "content-type": "application/pdf",
                },
            ],
        }
    )

    assert [location.representation_kind for location in locations] == [
        RepresentationKind.XML,
        RepresentationKind.PDF,
    ]
    assert locations[0].version == "vor"
    assert locations[0].access_type == "tdm"
    assert locations[0].license == "https://creativecommons.org/licenses/by/4.0/"


def test_landing_page_discovers_standard_and_structured_full_text_links() -> None:
    page = """
    <html><head>
      <meta name="citation_pdf_url" content="/files/paper.pdf">
      <link rel="alternate" type="application/xml" href="/files/paper.xml">
      <script type="application/ld+json">
        {"encoding": {"encodingFormat": "text/html", "contentUrl": "/fulltext"}}
      </script>
    </head><body><a href="/files/paper.pdf">Download PDF</a></body></html>
    """

    locations = discover_scholarly_locations(page, "https://journal.example/article")

    assert [(location.url, location.representation_kind) for location in locations] == [
        ("https://journal.example/files/paper.pdf", RepresentationKind.PDF),
        ("https://journal.example/files/paper.xml", RepresentationKind.XML),
        ("https://journal.example/fulltext", RepresentationKind.HTML),
    ]


def test_landing_page_does_not_treat_oembed_xml_as_full_text() -> None:
    page = """
    <html><head>
      <link rel="alternate" type="text/xml+oembed" href="/wp-json/oembed.xml">
      <link rel="alternate" type="application/json+oembed" href="/wp-json/oembed.json">
    </head><body>Publication record</body></html>
    """

    assert discover_scholarly_locations(
        page, "https://journal.example/article"
    ) == []


def test_location_acquisition_tries_an_alternate_after_first_pdf_fails() -> None:
    resolver = _resolver()
    resolver._safe_download = Mock(
        side_effect=[ValueError("first location unavailable"), b"%PDF-second"]
    )
    resolver._preflight_acquired_representation = Mock(
        return_value=(True, "acquired", "verified")
    )
    result = RetrievalResult(
        source_name="openalex",
        success=True,
        title="Example",
        locations=[
            AcquisitionLocation(
                url="https://first.example/article.pdf",
                provider="openalex",
                representation_kind=RepresentationKind.PDF,
                is_best=True,
            ),
            AcquisitionLocation(
                url="https://second.example/article.pdf",
                provider="openalex",
                representation_kind=RepresentationKind.PDF,
            ),
        ],
    )

    assert resolver._acquire_from_locations(result)
    assert result.representation is not None
    assert result.representation.source_url == "https://second.example/article.pdf"
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "unavailable",
        "acquired",
    ]


def test_pdf_download_retries_one_transient_transport_failure(monkeypatch) -> None:
    resolver = _resolver()
    request = httpx.Request("GET", "https://repository.example/source.pdf")
    fetch = Mock(
        side_effect=[
            httpx.RemoteProtocolError("server disconnected", request=request),
            b"%PDF-recovered",
        ]
    )
    monkeypatch.setattr("app.services.source_resolver.safe_fetch_bytes", fetch)

    assert resolver._safe_download("https://repository.example/source.pdf") == b"%PDF-recovered"
    assert fetch.call_count == 2


def test_pdf_download_does_not_retry_http_status_failure(monkeypatch) -> None:
    resolver = _resolver()
    request = httpx.Request("GET", "https://repository.example/source.pdf")
    response = httpx.Response(403, request=request)
    fetch = Mock(
        side_effect=httpx.HTTPStatusError(
            "forbidden", request=request, response=response
        )
    )
    monkeypatch.setattr("app.services.source_resolver.safe_fetch_bytes", fetch)

    try:
        resolver._safe_download("https://repository.example/source.pdf")
    except ValueError as exc:
        assert "Download failed" in str(exc)
    else:
        raise AssertionError("HTTP status failure should be rejected")
    assert fetch.call_count == 1


def test_web_graph_uses_next_search_provider_after_candidates_are_rejected() -> None:
    resolver = _resolver()
    source = Mock()
    source.name = "web_search"
    source.capabilities = frozenset({"web_discovery"})
    source.search_by_doi.return_value = RetrievalResult(
        source_name="web_search",
        success=True,
        locations=[
            AcquisitionLocation(
                url="https://wrong.example/doi.pdf",
                provider="web_search",
                representation_kind=RepresentationKind.PDF,
            )
        ],
        metadata={
            "search_attempts": [
                {"provider": "tavily", "outcome": "results", "result_count": 1}
            ]
        },
    )
    source.search_by_title_author.return_value = RetrievalResult(
        source_name="web_search",
        success=True,
        locations=[
            AcquisitionLocation(
                url="https://wrong.example/title.pdf",
                provider="web_search",
                representation_kind=RepresentationKind.PDF,
            )
        ],
        metadata={
            "search_attempts": [
                {"provider": "tavily", "outcome": "results", "result_count": 1}
            ]
        },
    )
    source.search_after_failed_candidates.return_value = RetrievalResult(
        source_name="web_search",
        success=True,
        locations=[
            AcquisitionLocation(
                url="https://official.example/report.pdf",
                provider="web_search",
                representation_kind=RepresentationKind.PDF,
            )
        ],
        metadata={
            "search_attempts": [
                {"provider": "exa", "outcome": "results", "result_count": 1}
            ]
        },
    )

    calls = 0

    def acquire(_source, result, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        result.metadata = result.metadata or {}
        url = result.locations[0].url
        if calls < 3:
            result.metadata["location_attempts"] = [
                {"url": url, "provider": "web_search", "outcome": "identity_rejected"}
            ]
        else:
            result.set_representation(
                SourceRepresentation(
                    kind=RepresentationKind.PDF,
                    media_type="application/pdf",
                    content=b"%PDF-official",
                    source_url=url,
                )
            )
            result.metadata["location_attempts"] = [
                {"url": url, "provider": "web_search", "outcome": "acquired"}
            ]
        return result

    resolver._download_and_cache = Mock(side_effect=acquire)

    result = resolver._try_source(
        source,
        "10.1234/report",
        "Official report",
        "OECD",
        "2021",
    )

    assert result.full_text == b"%PDF-official"
    assert [entry["phase"] for entry in result.metadata["retrieval_trace"]] == [
        "doi",
        "title_author",
        "post_validation_escalation",
    ]
    assert source.search_after_failed_candidates.call_args.kwargs["tried_providers"] == {
        "tavily",
        "exa",
    }


def test_explicit_full_text_html_is_acquired_as_typed_text(monkeypatch) -> None:
    resolver = _resolver()
    article = (
        "Example article by Rivera, 2024. "
        + " ".join(["Substantive article body sentence."] * 30)
    )
    response = httpx.Response(
        200,
        text=(
            "<html><head><meta name='citation_title' content='Example article'></head>"
            f"<body><article><p>{article}</p></article></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://journal.example/fulltext"),
    )
    monkeypatch.setattr("app.services.source_resolver.safe_request", lambda *args, **kwargs: response)
    result = RetrievalResult(
        source_name="crossref",
        success=True,
        title="Example article",
        year="2024",
        authors=["Rivera, Alex"],
        locations=[
            AcquisitionLocation(
                url="https://journal.example/fulltext",
                provider="crossref",
                representation_kind=RepresentationKind.HTML,
                intended_application="text-mining",
            )
        ],
    )

    assert resolver._acquire_from_locations(result)
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PLAIN_TEXT
    assert result.representation.original_kind is RepresentationKind.HTML
    assert b"Substantive article body" in result.representation.content


def test_web_search_html_hint_alone_does_not_promote_record_page(monkeypatch) -> None:
    resolver = _resolver()
    response = httpx.Response(
        200,
        text=(
            "<html><head><meta name='citation_title' content='Example article'></head>"
            "<body><main>Abstract and bibliographic metadata only.</main></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://repository.example/publications/example"),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.safe_request", lambda *args, **kwargs: response
    )
    result = RetrievalResult(
        source_name="web_search",
        success=True,
        title="Example article",
        locations=[
            AcquisitionLocation(
                url="https://repository.example/publications/example",
                provider="web_search",
                representation_kind=RepresentationKind.HTML,
                metadata={"may_contain_full_text": True},
            )
        ],
    )

    assert resolver._acquire_from_locations(result) is False
    assert result.representation is None
    assert result.metadata["location_attempts"][0]["outcome"] == "unavailable"


def test_location_acquisition_rejects_wrong_pdf_then_accepts_next_location() -> None:
    resolver = _resolver()
    resolver._safe_download = Mock(side_effect=[b"%PDF-wrong", b"%PDF-correct"])
    resolver._preflight_acquired_representation = Mock(
        side_effect=[
            (False, "identity_rejected", "wrong work"),
            (True, "acquired", "verified work"),
        ]
    )
    result = RetrievalResult(
        source_name="canonical_work_graph",
        success=True,
        title="Expected work",
        locations=[
            AcquisitionLocation(
                url="https://first.example/wrong.pdf",
                provider="first_provider",
                representation_kind=RepresentationKind.PDF,
                is_best=True,
            ),
            AcquisitionLocation(
                url="https://second.example/correct.pdf",
                provider="second_provider",
                representation_kind=RepresentationKind.PDF,
            ),
        ],
    )

    assert resolver._acquire_from_locations(result)
    assert result.full_text == b"%PDF-correct"
    assert result.representation is not None
    assert result.representation.source_url == "https://second.example/correct.pdf"
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "identity_rejected",
        "acquired",
    ]


def test_location_acquisition_rejects_incomplete_pdf_then_accepts_next() -> None:
    resolver = _resolver()
    resolver._safe_download = Mock(side_effect=[b"%PDF-fragment", b"%PDF-complete"])
    resolver._preflight_acquired_representation = Mock(
        side_effect=[
            (False, "completeness_rejected", "truncated representation"),
            (True, "acquired", "complete representation"),
        ]
    )
    result = RetrievalResult(
        source_name="canonical_work_graph",
        success=True,
        title="Expected work",
        locations=[
            AcquisitionLocation(
                url="https://first.example/fragment.pdf",
                provider="first_provider",
                representation_kind=RepresentationKind.PDF,
                is_best=True,
            ),
            AcquisitionLocation(
                url="https://second.example/complete.pdf",
                provider="second_provider",
                representation_kind=RepresentationKind.PDF,
            ),
        ],
    )

    assert resolver._acquire_from_locations(result)
    assert result.full_text == b"%PDF-complete"
    assert [attempt["outcome"] for attempt in result.metadata["location_attempts"]] == [
        "completeness_rejected",
        "acquired",
    ]


def test_non_pdf_content_identity_requires_evidence_inside_the_representation() -> None:
    confidence, reason = _verify_text_content_identity(
        b"The Exact Article Title. Smith. Published in 2024. Body text.",
        expected_doi=None,
        expected_title="The Exact Article Title",
        expected_author="Smith",
        expected_year="2024",
    )
    weak_confidence, _ = _verify_text_content_identity(
        b"An unrelated document with no matching bibliographic evidence.",
        expected_doi=None,
        expected_title="The Exact Article Title",
        expected_author="Smith",
        expected_year="2024",
    )

    assert confidence == "high"
    assert "author/year support" in reason
    assert weak_confidence == "low"


def test_html_landing_page_cannot_satisfy_typed_journal_reference() -> None:
    resolver = _resolver()
    body = (
        "The Exact Article Title. Smith. Published in 2024. "
        + " ".join(["Repository metadata record."] * 30)
    ).encode()
    result = RetrievalResult(
        source_name="web_search",
        success=True,
        title="The Exact Article Title",
        year="2024",
        authors=["Smith"],
        representation=SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT,
            media_type="text/plain",
            content=body,
            source_url="https://repository.example/record",
            original_kind=RepresentationKind.HTML,
            metadata={
                "observed_source_kind": "webpage",
                "observed_source_kind_confidence": "medium",
                "page_title_match": True,
            },
        ),
    )

    accepted, outcome, _ = resolver._preflight_acquired_representation(
        result,
        expected_doi=None,
        expected_title="The Exact Article Title",
        expected_author="Smith",
        expected_year="2024",
        expected_source_kind=SourceKindAssessment(
            "journal_article", "high", ("journal container structure",)
        ),
    )

    assert accepted is False
    assert outcome == "type_unconfirmed"


def test_structurally_typed_html_article_can_satisfy_journal_reference() -> None:
    resolver = _resolver()
    body = (
        "The Exact Article Title. Smith. Published in 2024. "
        + " ".join(["Substantive article evidence."] * 30)
    ).encode()
    result = RetrievalResult(
        source_name="crossref",
        success=True,
        title="The Exact Article Title",
        year="2024",
        authors=["Smith"],
        representation=SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT,
            media_type="text/plain",
            content=body,
            source_url="https://journal.example/fulltext",
            original_kind=RepresentationKind.HTML,
            metadata={
                "observed_source_kind": "journal_article",
                "observed_source_kind_confidence": "medium",
                "page_title_match": True,
            },
        ),
    )

    accepted, outcome, _ = resolver._preflight_acquired_representation(
        result,
        expected_doi=None,
        expected_title="The Exact Article Title",
        expected_author="Smith",
        expected_year="2024",
        expected_source_kind=SourceKindAssessment(
            "journal_article", "high", ("journal container structure",)
        ),
    )

    assert accepted is True
    assert outcome == "acquired"
