import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.models.job import Job, JobStatus
from app.tasks import provider_recovery


@pytest.fixture(autouse=True)
def _institutional_deployment(monkeypatch):
    """Automatic re-searching happens only in an Institutional deployment
    (owner decision 2026-09-29); these tests exercise that behaviour."""
    import app.services.assessment_marks as marks
    monkeypatch.setattr(marks, "institutional_deployment", lambda: True)



def _job(*, paper_version_id: str, provider: str | None) -> Job:
    result = {
        "reference_id": f"ref-{paper_version_id}",
        "status": "unavailable",
        "reason_code": "source_not_found",
    }
    if provider:
        from tests.unit.blocked_discovery import blocked_record
        # The refresh recomputes from the stored record, so it must genuinely be
        # blocked by `provider`: SearXNG as a required web route under the
        # legacy policy, anything else as a failed required academic adapter.
        result["reference_discovery"] = (
            blocked_record(result["reference_id"], web_outcomes={provider: "cooldown_skipped"})
            if provider == "searxng"
            else blocked_record(result["reference_id"], adapter=provider))
        result["retryable_provider_dependencies"] = [provider]
    return Job(
        filename=f"{paper_version_id}.pdf",
        status=JobStatus.COMPLETED,
        stage="completed",
        paper_version_id=paper_version_id,
        scope_type="personal_owner",
        scope_id="owner-1",
        input_sha256="a" * 64,
        input_media_type="application/pdf",
        input_byte_size=100,
        input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        upload_evidence={},
        extraction_payload={"checkpoint": "present"},
        source_results=[result],
        verification_summary={"reports_persisted": 0, "report_ids": []},
    )


def test_recovery_task_requeues_only_matching_completed_jobs(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        matching = _job(paper_version_id="matching", provider="searxng")
        unrelated = _job(paper_version_id="unrelated", provider="core")
        session.add_all([matching, unrelated])
        session.commit()
        matching_id = matching.id
        unrelated_id = unrelated.id

    scheduled = Mock()
    monkeypatch.setattr(provider_recovery, "SessionLocal", factory)
    monkeypatch.setattr(provider_recovery, "dispatch_paper_workflow", scheduled)
    monkeypatch.setattr(
        provider_recovery.settings, "PROVIDER_RECOVERY_MAX_JOBS", 25
    )

    result = provider_recovery.requeue_recovered_provider_work.run("searxng")

    assert result["jobs_requeued"] == 1
    assert result["reference_members"] == 1
    scheduled.assert_called_once()
    with factory() as session:
        matching = session.get(Job, matching_id)
        unrelated = session.get(Job, unrelated_id)
        assert matching.status == JobStatus.RUNNING
        assert scheduled.call_args.args == (
            str(matching_id), matching.upload_evidence["workflow_dispatch_v1"]["attempt_id"]
        )
        assert matching.source_results == []
        assert matching.upload_evidence["provider_refresh_reference_ids"] == [
            "ref-matching"
        ]
        assert unrelated.status == JobStatus.COMPLETED
        assert unrelated.source_results[0]["retryable_provider_dependencies"] == [
            "core"
        ]


def test_recovery_task_fails_closed_for_blank_provider() -> None:
    assert provider_recovery.requeue_recovered_provider_work.run("  ") == {
        "provider": "",
        "jobs_requeued": 0,
        "reference_members": 0,
    }


def test_due_health_probe_closes_incident_and_schedules_targeted_refresh(
    monkeypatch,
) -> None:
    monkeypatch.setattr(provider_recovery.settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", True)
    # Pin the engine the probe looks for, so the test asserts probe behaviour
    # rather than whichever engines happen to be configured today.
    monkeypatch.setattr(provider_recovery.settings, "SEARXNG_ENGINE_GROUPS", "google cse")
    store = Mock()
    store.incident_providers.return_value = ["searxng:google cse"]
    store.cooldown_remaining.return_value = 0
    store.claim_recovery_probe.return_value = True
    store.record_success.return_value = True
    search = Mock()
    search.search.return_value = []
    search.last_status = "completed"
    scheduled = Mock()
    monkeypatch.setattr(provider_recovery, "ProviderHealthStore", lambda: store)
    monkeypatch.setattr(
        provider_recovery, "get_search_provider", lambda _name: search
    )
    monkeypatch.setattr(provider_recovery, "schedule_provider_recovery", scheduled)

    result = provider_recovery.probe_retrieval_provider_recovery.run()

    assert result == {
        "provider": "managed_web_search",
        "incidents_probed": 1,
        "incidents_recovered": 1,
    }
    search.search.assert_called_once_with(
        "sourcefidelity provider availability probe",
        num_results=1,
        engines="google cse",
    )
    store.record_success.assert_called_once_with("searxng:google cse")
    scheduled.assert_called_once_with("searxng")


def test_active_cooldown_is_not_probed(monkeypatch) -> None:
    monkeypatch.setattr(provider_recovery.settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", True)
    store = Mock()
    store.incident_providers.return_value = ["searxng:brave"]
    store.cooldown_remaining.return_value = 30
    search = Mock()
    monkeypatch.setattr(provider_recovery, "ProviderHealthStore", lambda: store)
    monkeypatch.setattr(
        provider_recovery, "get_search_provider", lambda _name: search
    )

    result = provider_recovery.probe_retrieval_provider_recovery.run()

    assert result["incidents_probed"] == 0
    search.search.assert_not_called()


def test_unconfigured_searx_engine_is_neither_probed_nor_cleared(monkeypatch):
    monkeypatch.setattr(provider_recovery.settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", True)
    monkeypatch.setattr(provider_recovery.settings, "SEARXNG_ENGINE_GROUPS", "google cse;brave")
    store = Mock()
    store.incident_providers.return_value = ["searxng:example trial", "searxng:old group"]
    monkeypatch.setattr(provider_recovery, "ProviderHealthStore", lambda: store)
    search_factory = Mock()
    monkeypatch.setattr(provider_recovery, "get_search_provider", search_factory)

    result = provider_recovery.probe_retrieval_provider_recovery.run()

    assert result["incidents_probed"] == 0
    search_factory.assert_not_called()
    store.record_success.assert_not_called()
    store.record_unavailable.assert_not_called()
    store.claim_recovery_probe.assert_not_called()


def test_removed_provider_incident_generates_no_probe_traffic(monkeypatch) -> None:
    """A stale incident for a removed provider is kept, never probed.

    The DuckDuckGo adapter was removed on 2026-09-22 because it used an
    endpoint its terms do not permit. Health records written before that
    must not keep sending the application back to it.
    """
    monkeypatch.setattr(
        provider_recovery.settings, "SEARCH_POLICY_VERSION", "configured-search-v1"
    )
    store = Mock()
    store.incident_providers.return_value = ["duckduckgo"]
    store.cooldown_remaining.return_value = 0
    store.claim_recovery_probe.return_value = True
    search = Mock()
    scheduled = Mock()
    monkeypatch.setattr(provider_recovery, "ProviderHealthStore", lambda: store)
    monkeypatch.setattr(
        provider_recovery, "get_search_provider", lambda name: search
    )
    monkeypatch.setattr(provider_recovery, "schedule_provider_recovery", scheduled)

    result = provider_recovery.probe_retrieval_provider_recovery.run()

    assert result["incidents_probed"] == 0
    search.search.assert_not_called()
    store.claim_recovery_probe.assert_not_called()
    scheduled.assert_not_called()


def test_a_brave_recovery_requeues_a_job_brave_was_blocking(monkeypatch) -> None:
    """Closes the chain added 2026-09-24: Brave reports its own recovery.

    A reference held incomplete because Brave failed under `api-first-search-v2`
    is re-run when Brave recovers. One that Brave did not block -- Exa failed
    there, Brave answered -- is left alone.
    """
    from tests.unit.blocked_discovery import blocked_record

    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        blocked = _job(paper_version_id="brave-blocked", provider=None)
        blocked.source_results = [{
            "reference_id": "ref-brave", "status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": blocked_record(
                "ref-brave", policy="api-first-search-v2",
                web_outcomes={"brave": "operational_failure", "exa": "results"})}]
        other = _job(paper_version_id="exa-blocked", provider=None)
        other.source_results = [{
            "reference_id": "ref-exa", "status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": blocked_record(
                "ref-exa", policy="api-first-search-v2",
                web_outcomes={"brave": "results", "exa": "timeout"})}]
        session.add_all([blocked, other])
        session.commit()
        blocked_id, other_id = blocked.id, other.id

    dispatch = Mock()
    monkeypatch.setattr(provider_recovery, "SessionLocal", factory)
    monkeypatch.setattr(provider_recovery, "dispatch_paper_workflow", dispatch)
    monkeypatch.setattr(provider_recovery.settings, "PROVIDER_RECOVERY_MAX_JOBS", 25)
    monkeypatch.setattr(provider_recovery.settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", True)

    result = provider_recovery.requeue_recovered_provider_work.run("brave")

    assert result["jobs_requeued"] == 1
    dispatch.assert_called_once()
    with factory() as session:
        assert session.get(Job, blocked_id).upload_evidence[
            "provider_refresh_reference_ids"] == ["ref-brave"]
        assert session.get(Job, other_id).status == JobStatus.COMPLETED
