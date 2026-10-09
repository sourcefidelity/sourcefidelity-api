import logging

import httpx

from app.services.search import mojeek
from app.services.search.mojeek import MojeekSearch

KEY = "synthetic-mojeek-key-000"


def _response(status, body=None):
    request = httpx.Request("GET", f"https://api.mojeek.com/search?api_key={KEY}")
    return httpx.Response(status, json=body or {}, request=request)


def test_results_are_parsed(monkeypatch):
    body = {"response": {"status": "OK", "results": [
        {"url": "https://example.org/a.pdf", "title": "A title", "desc": "A snippet"}]}}
    monkeypatch.setattr(mojeek.httpx, "get", lambda *a, **k: _response(200, body))
    monkeypatch.setattr(mojeek.time, "sleep", lambda s: None)
    provider = MojeekSearch(KEY)
    [result] = provider.search('"A title"', 5)
    assert result.is_pdf and result.snippet == "A snippet"
    assert provider.last_status == "completed"


def test_a_refused_key_never_reaches_the_log(monkeypatch, caplog):
    monkeypatch.setattr(mojeek.httpx, "get", lambda *a, **k: _response(401))
    monkeypatch.setattr(mojeek.time, "sleep", lambda s: None)
    provider = MojeekSearch(KEY)
    with caplog.at_level(logging.DEBUG):
        assert provider.search("query", 5) == []
        assert provider.search("query", 5) == []      # circuit: no second call
    assert KEY not in caplog.text and "api_key" not in caplog.text
    assert provider.last_status != "completed"


def test_a_provider_error_status_is_a_failure(monkeypatch):
    body = {"response": {"status": "ERROR", "message": "daily limit"}}
    monkeypatch.setattr(mojeek.httpx, "get", lambda *a, **k: _response(200, body))
    monkeypatch.setattr(mojeek.time, "sleep", lambda s: None)
    provider = MojeekSearch(KEY)
    assert provider.search("query", 5) == []
    assert provider.last_status != "completed"
