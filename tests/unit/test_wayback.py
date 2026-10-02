"""Archived copies of a submitted page (`wayback-snapshot-v1`, owner decision 2026-10-02)."""
from types import SimpleNamespace

import app.services.safe_fetch as safe_fetch
from app.services.retrieval.wayback import RETRY_REASONS, archived_snapshot


def respond(payload, status=200):
    return lambda url, **kwargs: SimpleNamespace(status_code=status, json=lambda: payload)


def test_the_closest_capture_is_read_as_captured(monkeypatch):
    payload = {"archived_snapshots": {"closest": {"available": True, "status": "200", "timestamp": "20251113172748"}}}
    monkeypatch.setattr(safe_fetch, "safe_request", respond(payload))
    snapshot = archived_snapshot("https://www.example.test/reviews/a-film-1988")
    assert snapshot["snapshot_url"] == "https://web.archive.org/web/20251113172748id_/https://www.example.test/reviews/a-film-1988"
    assert snapshot["policy_version"] == "wayback-snapshot-v1"


def test_no_capture_or_a_failed_capture_is_no_copy(monkeypatch):
    monkeypatch.setattr(safe_fetch, "safe_request", respond({"archived_snapshots": {}}))
    assert archived_snapshot("https://www.example.test/a") is None
    failed = {"archived_snapshots": {"closest": {"available": True, "status": "404", "timestamp": "20200101000000"}}}
    monkeypatch.setattr(safe_fetch, "safe_request", respond(failed))
    assert archived_snapshot("https://www.example.test/a") is None
    assert archived_snapshot("doi:10.1/x") is None


def test_only_refused_missing_or_unreadable_pages_are_retried_from_the_archive():
    assert {"access_restricted", "fetch_unavailable"} <= RETRY_REASONS
    assert "bibliographic_fields_conflict" not in RETRY_REASONS and "page_title_mismatch_unconfirmed" not in RETRY_REASONS
