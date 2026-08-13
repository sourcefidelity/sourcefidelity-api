from unittest.mock import Mock

import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import SourceResolutionError, SourceResolver


@pytest.fixture
def resolver() -> SourceResolver:
    instance = SourceResolver.__new__(SourceResolver)
    instance._backend = None
    instance._retrieval_sources = []
    return instance


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
