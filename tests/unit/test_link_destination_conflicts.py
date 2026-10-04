"""Submitted links that lead to a different work (owner request 2026-10-02, paper 8)."""
from types import SimpleNamespace

import app.services.source_resolver as resolver
from app.services.retrieval.base import RetrievalResult


def capture(monkeypatch):
    calls = []
    monkeypatch.setattr(resolver, "identity_observed",
                        lambda url, sha, reason, fields, differences=(): calls.append((reason, list(fields), differences)))
    return calls


def test_a_page_titled_with_a_different_work_is_a_title_conflict(monkeypatch):
    calls = capture(monkeypatch)
    resp = SimpleNamespace(content=b"<html></html>")
    resolver._record_link_title_conflict(
        "https://www.example.test/stats", resp, "The Economic Contribution of the Australian Film Industry",
        ["Film, Television and Digital Games, Australia, 2021-22 financial year | Bureau of Statistics"])
    assert calls and calls[0][0] == "bibliographic_fields_conflict" and "title" in calls[0][1]
    calls.clear()
    resolver._record_link_title_conflict("https://www.example.test/", resp, "A Cited Work Title", ["Home"])
    assert calls == []


def test_a_semantic_scholar_link_is_read_from_its_registered_record(monkeypatch):
    calls = capture(monkeypatch)

    class Fake:
        def paper_by_id(self, paper_id):
            assert paper_id == "2cbdab0f6dfe26cb53c71b9bd4acc33b595269e6"
            return RetrievalResult(source_name="semantic_scholar", success=True,
                                   title="Culture in Australia: Policies, publics, and programs",
                                   authors=["T. Bennett", "D. Carter"], year="2001")

    import app.services.retrieval.semantic_scholar as s2
    monkeypatch.setattr(s2, "SemanticScholarRetriever", Fake)
    url = ("https://www.semanticscholar.org/paper/Culture-in-Australia-%3A-Policies%2C-publics%2C-and-Bennett-Carter/"
           "2cbdab0f6dfe26cb53c71b9bd4acc33b595269e6")
    resolver._record_registry_link_identity(url, SimpleNamespace(content=b"challenge"),
                                            "Cultural Policy and Cultural Institutions in Australia",
                                            expected_author="Bennett, T.", expected_year="2007")
    assert calls and calls[0][0] == "bibliographic_fields_conflict" and "title" in calls[0][1]
    calls.clear()
    resolver._record_registry_link_identity("https://www.example.test/paper/x", SimpleNamespace(content=b""), "A title")
    assert calls == []


def test_a_bot_check_or_script_shell_page_title_is_never_a_conflict(monkeypatch):
    calls = capture(monkeypatch)
    resp = SimpleNamespace(content=b"<html></html>")
    for shell in ("JavaScript is disabled", "Just a moment...", "Access Denied | Example Site Security Check"):
        resolver._record_link_title_conflict("https://www.example.test/t", resp, "The Day the Earth Stood Still (1951)", [shell])
    assert calls == []
    assert resolver.interstitial_page_title("JavaScript is disabled")
    assert not resolver.interstitial_page_title("Film, Television and Digital Games, Australia")
