import json
from unittest.mock import Mock

import httpx
import pytest

from app.config import settings
from app.services.retrieval import get_retrieval_sources, installed_retrieval_providers
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy


def _response(status: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://api.crossref.org/works/example")
    return httpx.Response(status, json=payload, request=request)


def test_installed_provider_registry_exposes_capabilities_not_secrets() -> None:
    registry = installed_retrieval_providers()

    assert "semantic_scholar" in registry
    assert "batch_doi" in registry["semantic_scholar"]["capabilities"]
    assert registry["core"]["documentation_url"]
    assert "api_key" not in json.dumps(registry).lower()


def test_unknown_configured_provider_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_SOURCES", "openalex,unreviewed_remote")

    with pytest.raises(ValueError, match="install a trusted adapter"):
        get_retrieval_sources()


def test_provider_policy_applies_valid_safe_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "RETRIEVAL_PROVIDER_CONFIG",
        '{"semantic_scholar":{"batch_size":3,"max_batches":2,"enabled":false}}',
    )

    policy = provider_policy("semantic_scholar", ProviderPolicy(batch_size=5))

    assert policy.batch_size == 3
    assert policy.max_batches == 2
    assert policy.enabled is False


def test_crossref_records_calls_and_outcomes(monkeypatch) -> None:
    payload = {
        "message": {
            "DOI": "10.1234/example",
            "title": ["Example"],
            "issued": {"date-parts": [[2024]]},
        }
    }
    request = Mock(side_effect=[_response(200, payload), _response(404)])
    monkeypatch.setattr("app.services.retrieval.crossref.httpx.get", request)

    retriever = CrossrefRetriever()
    assert retriever.search_by_doi("10.1234/example").success is True
    assert retriever.search_by_doi("10.1234/missing").success is False
    assert retriever.provider_metrics == {
        "calls": 2,
        "successes": 1,
        "not_found": 1,
        "client_errors": 0,
        "server_errors": 0,
        "network_errors": 0,
    }
