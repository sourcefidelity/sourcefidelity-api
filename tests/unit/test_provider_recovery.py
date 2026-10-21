import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.models.job import Job, JobStatus
from app.tasks import provider_recovery





def _job(*, paper_version_id: str, provider: str | None) -> Job:
    result = {
        "reference_id": f"ref-{paper_version_id}",
        "status": "unavailable",
        "reason_code": "source_not_found",
    }
    if provider:
        from tests.unit.blocked_discovery import blocked_record
        # The refresh recomputes from the stored record, so it must genuinely be
        # blocked by `provider`: Tavily as a required web route under the
        # legacy policy, anything else as a failed required academic adapter.
        result["reference_discovery"] = (
            blocked_record(result["reference_id"], web_outcomes={provider: "cooldown_skipped"})
            if provider == "tavily"
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
        matching = _job(paper_version_id="matching", provider="tavily")
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

    result = provider_recovery.requeue_recovered_provider_work.run("tavily")

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
                # Exa failing too: with Exa answering, Brave's provider-side
                # failure alone would leave the search complete
                # (search-completion-v1, owner decision 2026-10-19).
                web_outcomes={"brave": "operational_failure", "exa": "timeout"})}]
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


@pytest.fixture(autouse=True)
def _automatic_searching_on(monkeypatch, request):
    # These papers' scopes finish searches on their own; the Personal default
    # (off, owner decision 2026-10-18) is tested on its own.
    if "owners_setting" not in request.node.name:
        monkeypatch.setattr("app.services.storage_policy.finishes_searches_automatically",
                            lambda session, scope_type, scope_id: True)
