from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.retrieval.canonical_work import (
    CanonicalWorkGraph,
    canonicalize_location_url,
)


def _result(
    provider: str,
    *,
    doi: str = "10.1234/example",
    title: str = "A Canonical Example Work",
    year: str = "2024",
    locations: list[AcquisitionLocation] | None = None,
    abstract: str | None = None,
    representation: SourceRepresentation | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        source_name=provider,
        success=True,
        doi=doi,
        title=title,
        year=year,
        authors=["Rivera, Alex"],
        locations=locations or [],
        abstract=abstract,
        representation=representation,
        metadata={"provider_record": provider},
    )


def test_merges_duplicate_locations_without_losing_provider_provenance() -> None:
    graph = CanonicalWorkGraph(expected_doi="10.1234/example")
    graph.add(
        _result(
            "openalex",
            locations=[
                AcquisitionLocation(
                    url="https://REPO.example:443/paper.pdf?a=1&utm_source=test#page=2",
                    provider="openalex",
                    representation_kind=RepresentationKind.PDF,
                )
            ],
        )
    )
    graph.add(
        _result(
            "crossref",
            locations=[
                AcquisitionLocation(
                    url="https://repo.example/paper.pdf?a=1",
                    provider="crossref",
                    representation_kind=RepresentationKind.PDF,
                    is_best=True,
                )
            ],
        )
    )

    merged = graph.to_result()

    assert len(merged.locations) == 1
    assert merged.locations[0].metadata["providers"] == ["openalex", "crossref"]
    assert merged.locations[0].is_best is True
    assert merged.metadata["canonical_work"]["accepted_providers"] == [
        "openalex",
        "crossref",
    ]


def test_rejects_conflicting_doi_and_excludes_its_location() -> None:
    graph = CanonicalWorkGraph(expected_doi="10.1234/example")
    assessment = graph.add(
        _result(
            "wrong_provider",
            doi="10.9999/different",
            locations=[
                AcquisitionLocation(
                    url="https://wrong.example/paper.pdf",
                    provider="wrong_provider",
                    representation_kind=RepresentationKind.PDF,
                )
            ],
        )
    )

    merged = graph.to_result()

    assert assessment.confidence == "rejected"
    assert merged.success is False
    assert merged.locations == []
    assert merged.metadata["canonical_work"]["rejected_candidates"][0][
        "provider"
    ] == "wrong_provider"


def test_rejects_provider_record_with_incompatible_bibliographic_type() -> None:
    graph = CanonicalWorkGraph(
        expected_title="The Master Switch",
        expected_author="Wu, Tim",
        expected_year="2010",
        expected_source_kind="monograph",
        expected_source_kind_confidence="high",
        expected_source_kind_evidence=("book publisher citation structure",),
    )
    candidate = _result(
        "crossref",
        doi="",
        title="The Master Switch",
        year="2010",
        locations=[
            AcquisitionLocation(
                url="https://example.org/review.pdf",
                provider="crossref",
                representation_kind=RepresentationKind.PDF,
            )
        ],
    )
    candidate.metadata = {"message": {"type": "journal-article"}}

    assessment = graph.add(candidate)

    assert assessment.confidence == "rejected"
    assert "type conflict" in assessment.reason
    assert graph.to_result().locations == []


def test_exact_doi_is_not_overridden_by_coarse_provider_type_taxonomy() -> None:
    graph = CanonicalWorkGraph(
        expected_doi="10.1234/example",
        expected_source_kind="journal_article",
        expected_source_kind_confidence="high",
    )
    candidate = _result("crossref")
    candidate.metadata = {"message": {"type": "proceedings-article"}}

    assessment = graph.add(candidate)

    assert assessment.confidence == "high"
    assert assessment.reason == "exact DOI match"


def test_preserves_conflicts_and_selects_richest_accepted_evidence() -> None:
    graph = CanonicalWorkGraph(
        expected_title="A Canonical Example Work",
        expected_author="Rivera, Alex",
    )
    graph.add(_result("first", year="2023", abstract="Short abstract."))
    graph.add(
        _result(
            "second",
            year="2024",
            abstract="A much longer abstract containing the richer available evidence.",
        )
    )

    merged = graph.to_result()

    assert merged.abstract.startswith("A much longer")
    assert "year" in merged.metadata["canonical_work"]["metadata_conflicts"]


def test_representation_selection_prefers_completeness_before_format() -> None:
    graph = CanonicalWorkGraph(expected_doi="10.1234/example")
    graph.add(
        _result(
            "pdf_provider",
            representation=SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=b"%PDF-partial",
                completeness="partial",
            ),
        )
    )
    graph.add(
        _result(
            "text_provider",
            representation=SourceRepresentation(
                kind=RepresentationKind.PLAIN_TEXT,
                media_type="text/plain",
                content=b"complete text",
                completeness="complete",
            ),
        )
    )

    merged = graph.to_result()

    assert merged.source_name == "text_provider"
    assert merged.representation.kind is RepresentationKind.PLAIN_TEXT


def test_url_canonicalization_preserves_possible_access_parameters() -> None:
    canonical = canonicalize_location_url(
        "HTTPS://Example.org:443/text?download=1&token=abc&utm_campaign=test#section"
    )

    assert canonical == "https://example.org/text?download=1&token=abc"


def test_url_canonicalization_fails_safe_on_invalid_port() -> None:
    url = "https://example.org:not-a-port/paper.pdf"

    assert canonicalize_location_url(url) == url
