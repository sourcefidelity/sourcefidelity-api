"""Automatic re-runs fire only for providers that were blocking completion.

Owner decision, 2026-09-24: keep automatic re-runs on by default, trigger them
only for providers the completion gate counts as blocking, and allow them to be
switched off. Measured basis: every one of the four re-run waves on 2026-09-23
(4, 90, 47 and 51 reference re-runs) was triggered by SearXNG, which cannot
complete or invalidate a search under `api-first-search-v2`. Those waves
changed potentially-fabricated-reference flags back and forth between report
versions -- 15 fabricated references were flagged in some versions and not
others -- without changing a single completion gate.
"""
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from app.services.paper_workflow import _blocking_providers_from_record
from app.services.reference_discovery import (
    ReferenceDiscoveryRecord, _search_is_incomplete, search_blocking_providers)
from tests.unit.blocked_discovery import blocked_record


@pytest.fixture(autouse=True)
def _institutional_deployment(monkeypatch):
    """Automatic re-searching happens only in an Institutional deployment
    (owner decision 2026-09-29); these tests exercise that behaviour."""
    import app.services.assessment_marks as marks
    monkeypatch.setattr(marks, "institutional_deployment", lambda: True)



def _gate(record: dict) -> bool:
    r = ReferenceDiscoveryRecord.model_validate(record)
    return _search_is_incomplete(r.required_route_categories, r.attempts,
                                 r.queries, r.search_policy_version)


@pytest.mark.parametrize("record", [
    blocked_record("a", adapter="core"),
    blocked_record("b", adapter="crossref"),
    blocked_record("c", policy="api-first-search-v2",
                   web_outcomes={"brave": "timeout", "exa": "results"}),
    blocked_record("d", policy="api-first-search-v2",
                   web_outcomes={"brave": "results", "exa": "rate_limited"}),
    blocked_record("e", web_outcomes={"searxng": "cooldown_skipped"}),
])
def test_a_named_blocker_always_means_the_search_is_incomplete(record) -> None:
    """The trigger must never name a provider on a search that already completed."""
    assert _blocking_providers_from_record(record)
    assert _gate(record) is True


def test_an_optional_web_provider_never_triggers_under_api_first() -> None:
    record = blocked_record("r", policy="api-first-search-v2", web_outcomes={
        "brave": "results", "exa": "results", "searxng": "operational_failure"})

    assert _blocking_providers_from_record(record) == []


def test_a_required_web_api_that_never_ran_is_blocking() -> None:
    """Exa absent entirely is as blocking as Exa failing."""
    record = blocked_record("r", policy="api-first-search-v2",
                            web_outcomes={"brave": "results"})

    assert _blocking_providers_from_record(record) == ["exa"]


def test_an_optional_adapter_that_failed_is_not_blocking() -> None:
    record = blocked_record("r", adapter="semantic_scholar")
    record["attempts"][0]["required"] = False

    assert _blocking_providers_from_record(record) == []


def test_our_own_budget_running_out_does_not_name_the_provider() -> None:
    """A provider we never called cannot be what recovery would cure."""
    record = blocked_record("r", adapter="crossref")
    record["attempts"][0].update(outcome="unavailable",
                                 reason_code="route_elapsed_budget_exhausted")
    record["queries"][0]["execution_outcome"] = "budget_skipped"

    assert _gate(record) is True          # still incomplete, honestly
    assert _blocking_providers_from_record(record) == []   # but not a re-run trigger


def test_a_provider_that_answered_is_not_a_recovery_target() -> None:
    """An empty result the gate declines to certify is not an outage."""
    record = blocked_record("r", adapter="openalex")
    record["attempts"][0].update(outcome="no_match", reason_code="no_candidate_returned")
    record["queries"][0].update(execution_outcome="no_results",
                                reason_code="metadata_candidates_filtered")

    assert _gate(record) is True
    assert _blocking_providers_from_record(record) == []


def test_the_refresh_ignores_a_stale_stored_trigger_list() -> None:
    """Jobs stored before this change list every queried provider.

    The refresh recomputes from the record, so an old list naming SearXNG does
    not re-run a reference SearXNG was never blocking.
    """
    from app.services.paper_workflow import prepare_provider_recovery_refresh

    record = blocked_record("ref-1", policy="api-first-search-v2", web_outcomes={
        "brave": "results", "exa": "timeout", "searxng": "cooldown_skipped"})
    from types import SimpleNamespace

    from app.models.job import JobStatus

    def fresh_job():
        return SimpleNamespace(
            status=JobStatus.COMPLETED, store_only=False, extraction_payload={"x": 1},
            upload_evidence={}, verification_summary={}, stage="completed",
            source_results=[{"reference_id": "ref-1", "reference_discovery": record,
                             "retryable_provider_dependencies": ["searxng", "exa"]}])
    session = Mock()

    import app.services.paper_workflow as workflow
    original = workflow._job
    try:
        workflow._job = lambda _s, _id: fresh_job()
        assert prepare_provider_recovery_refresh(session, "j", provider="searxng",
                                                 commit=False) == []
        workflow._job = lambda _s, _id: fresh_job()
        assert prepare_provider_recovery_refresh(session, "j", provider="exa",
                                                 commit=False) == ["ref-1"]
    finally:
        workflow._job = original


def test_automatic_reruns_are_on_by_default() -> None:
    from app.config import Settings

    assert Settings.model_fields["PROVIDER_RECOVERY_REFRESH_ENABLED"].default is True


def test_turning_reruns_off_stops_the_requeue(monkeypatch) -> None:
    from app.tasks import provider_recovery

    monkeypatch.setattr(provider_recovery.settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", False)
    session_factory = Mock(side_effect=AssertionError("no job may be read when disabled"))
    monkeypatch.setattr(provider_recovery, "SessionLocal", session_factory)
    dispatch = Mock()
    monkeypatch.setattr(provider_recovery, "dispatch_paper_workflow", dispatch)

    result = provider_recovery.requeue_recovered_provider_work.run("brave")

    assert result["jobs_requeued"] == 0
    assert result["disabled"] == "PROVIDER_RECOVERY_REFRESH_ENABLED"
    dispatch.assert_not_called()
