"""A validated public copy is kept whatever route found it (owner decision 2026-10-09)."""
from types import SimpleNamespace

import pytest

from app.services.retrieval.base import RepresentationKind, RetrievalResult, SourceRepresentation
from app.services.source_resolver import SourceResolver

REFERENCE = SimpleNamespace(title="Harbour cranes in river ports", doi=None, author="Ames", year="2020",
                            source_kind="journal_article", source_kind_confidence="high", source_kind_evidence=[],
                            raw_ref="Ames (2020). Harbour cranes in river ports.", url=None)


def _result(**changes):
    representation = SourceRepresentation(
        kind=RepresentationKind.HTML, media_type="text/html", content=b"<p>Harbour cranes in river ports.</p>",
        source_url=changes.pop("url", "https://ports.example/ames"))
    representation.completeness = changes.pop("completeness", "complete")
    metadata = {"identity_confidence": "high", "identity_reason": "fixture", **changes.pop("metadata", {})}
    return RetrievalResult(source_name="web_search", success=True, metadata=metadata, representation=representation)


@pytest.fixture
def resolver(monkeypatch):
    resolver = object.__new__(SourceResolver)
    resolver.persisted = []
    monkeypatch.setattr(resolver, "_durable_repository_active", lambda: True)
    monkeypatch.setattr(resolver, "_persist_retrieved_representation",
                        lambda result, **kwargs: resolver.persisted.append(kwargs))
    return resolver


def test_a_web_search_copy_is_offered_for_keeping(resolver):
    resolver.retain_validated_copy(REFERENCE, _result())
    assert len(resolver.persisted) == 1
    kept = resolver.persisted[0]
    # Never claimed as a publisher download: the retention policy classifies it.
    assert kept["downloaded_via_publisher"] is False and kept["identity_confidence"] == "high"
    assert kept["ref_title"] == REFERENCE.title


@pytest.mark.parametrize("change", [
    {"metadata": {"identity_confidence": "medium"}},
    {"metadata": {"durable_admission": {"state": "not_stored"}}},
    {"metadata": {"repository_representation_id": "r1"}},
    {"completeness": "incomplete"},
    {"url": "file:///tmp/ames.html"},
])
def test_unverified_partial_already_decided_or_unaddressed_copies_are_not(resolver, change):
    resolver.retain_validated_copy(REFERENCE, _result(**change))
    assert resolver.persisted == []


def test_a_failure_to_keep_never_fails_the_run(resolver, monkeypatch):
    def fail(result, **kwargs):
        raise RuntimeError("storage down")
    monkeypatch.setattr(resolver, "_persist_retrieved_representation", fail)
    result = _result()
    resolver.retain_validated_copy(REFERENCE, result)
    assert result.metadata["durable_admission"] == {"state": "not_stored", "reason": "retention_failed"}


def test_nothing_is_kept_when_the_source_library_is_off(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_durable_repository_active", lambda: False)
    resolver.retain_validated_copy(REFERENCE, _result())
    assert resolver.persisted == []
