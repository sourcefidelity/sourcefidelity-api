from unittest.mock import Mock

import httpx

from app.config import settings
from app.services.retrieval.elsevier import (
    ElsevierRetriever,
    _extract_article_text,
    _is_elsevier_doi,
)


def _response(status_code: int, json_data: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://api.elsevier.com/content/article/doi/example")
    return httpx.Response(status_code, request=request, json=json_data)


def test_non_elsevier_doi_skips_article_api(monkeypatch) -> None:
    request = Mock()
    monkeypatch.setattr("app.services.retrieval.elsevier.httpx.get", request)

    result = ElsevierRetriever().search_by_doi("10.1017/S1474745611000231")

    assert result.success is False
    assert "skipped" in (result.error or "").lower()
    request.assert_not_called()


def test_elsevier_doi_uses_metadata_then_native_xml(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ELSEVIER_API_KEY", "synthetic-test-key")
    metadata = {
        "full-text-retrieval-response": {
            "coredata": {
                "dc:title": "Example article",
                "prism:doi": "10.1016/j.example.2026.1",
                "dc:description": "<p>Example abstract</p>",
                "openaccess": "1",
            }
        }
    }
    request = Mock(
        side_effect=[
            _response(200, metadata),
            httpx.Response(
                200,
                request=httpx.Request("GET", "https://api.elsevier.com/article"),
                text=(
                    "<ja:body><ce:para>Complete article body text with enough "
                    "content to satisfy the full-text minimum length used by the adapter. "
                    "This is representative article prose, not metadata.</ce:para></ja:body>"
                ),
            ),
        ]
    )
    monkeypatch.setattr("app.services.retrieval.elsevier.httpx.get", request)

    result = ElsevierRetriever().search_by_doi("10.1016/j.example.2026.1")

    assert result.success is True
    assert result.full_text is not None
    assert result.full_text.startswith(b"Complete article body text")
    assert request.call_count == 2
    assert request.call_args_list[1].kwargs["headers"]["Accept"] == "text/xml"
    assert "params" not in request.call_args_list[1].kwargs


def test_metadata_only_xml_is_not_full_text() -> None:
    xml = "<full-text-retrieval-response><coredata><dc:title>Only metadata</dc:title></coredata></full-text-retrieval-response>"

    assert _extract_article_text(xml) == ""


def test_elsevier_prefix_gate_is_conservative() -> None:
    assert _is_elsevier_doi("10.1016/j.example.2026.1") is True
    assert _is_elsevier_doi("10.1017/S1474745611000231") is False
