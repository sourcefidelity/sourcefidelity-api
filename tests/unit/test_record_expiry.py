"""A paper's whole record is deleted once its scope's report period ends (owner decision 2026-10-09)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.judgment import JudgmentCandidateResult, JudgmentRun
from app.models.report import Report, ReportPaperArtifactRecord, VerificationReportRecord
from app.services.record_expiry import expire_paper_records
from app.services.storage_policy import set_policy

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _paper(session, name, *, scope_id="owner-1", age_days=40, status=JobStatus.COMPLETED):
    built = NOW - timedelta(days=age_days)
    job = Job(filename=f"{name}.pdf", status=status, stage=JobStage.COMPLETED, paper_version_id=f"paper:{name}",
              scope_type="personal_owner", scope_id=scope_id, input_sha256="a" * 64,
              input_media_type="application/pdf", input_byte_size=1, input_expires_at=built,
              upload_evidence={}, created_at=built, updated_at=built)
    session.add(job)
    session.flush()
    report = Report(job_id=job.id, report_json={}, report_version=1, created_at=built)
    session.add(report)
    session.flush()
    check = VerificationReportRecord(verification_id=f"v-{name}", paper_version_id=job.paper_version_id,
                                     scope_type="personal_owner", scope_id=scope_id, report_version=1,
                                     artifact_version="x", verdict="x", evidence_sha256="b" * 64, report_payload={})
    session.add(check)
    session.flush()
    run = JudgmentRun(report_id=report.id, scope_type="personal_owner", scope_id=scope_id, requested_by="t",
                      notice_version="n", panel_version="p")
    session.add(run)
    session.flush()
    session.add(JudgmentCandidateResult(run_id=run.id, seq=1, verification_report_id=check.id, candidate_id="c1",
                                        display_state="supported", reason_code="r", panel={}))
    session.add(ReportPaperArtifactRecord(
        job_id=job.id, report_id=report.id, paper_version_id=job.paper_version_id, scope_type="personal_owner",
        scope_id=scope_id, storage_key=f"{name}.pdf", content_sha256="c" * 64, media_type="application/pdf",
        byte_size=1, artifact_kind="submitted_pdf", presentation_status="page_faithful_ready",
        sanitization_evidence={}, expires_at=built + timedelta(days=7)))
    session.commit()
    return job.id


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def _backend(deletes=True):
    backend = Mock()
    backend.delete.return_value = deletes
    backend.exists.return_value = not deletes
    return backend


def test_only_records_past_their_scopes_period_are_deleted(session):
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}})
    old = _paper(session, "old")
    recent = _paper(session, "recent", age_days=10)
    untouched = _paper(session, "other-scope", scope_id="owner-2", age_days=400)
    assert expire_paper_records(session, _backend(), now=NOW, dry_run=True) == {"expired": 1, "deleted": 0}
    assert _count(session, Job) == 3
    assert expire_paper_records(session, _backend(), now=NOW) == {"expired": 1, "deleted": 1}
    assert {job.id for job in session.scalars(select(Job))} == {recent, untouched}
    assert old not in {r.job_id for r in session.scalars(select(Report))}
    for model in (Report, ReportPaperArtifactRecord, VerificationReportRecord, JudgmentRun, JudgmentCandidateResult):
        assert _count(session, model) == 2


def test_returning_a_report_again_restarts_its_period(session):
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}})
    job_id = _paper(session, "reused")
    job = session.get(Job, job_id)
    job.upload_evidence = {"paper_reuse": [{"at": (NOW - timedelta(days=2)).isoformat(), "report_id": "r"}]}
    session.commit()
    assert expire_paper_records(session, _backend(), now=NOW)["expired"] == 0


def test_nothing_is_deleted_by_default_or_while_a_paper_is_running(session):
    _paper(session, "default", age_days=4000)
    assert expire_paper_records(session, _backend(), now=NOW)["expired"] == 0
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}})
    _paper(session, "running", status=JobStatus.RUNNING)
    assert expire_paper_records(session, _backend(), now=NOW)["deleted"] == 1
    assert {job.filename for job in session.scalars(select(Job))} == {"running.pdf"}


def test_a_paper_copy_that_cannot_be_deleted_keeps_the_whole_record(session):
    set_policy(session, "personal_owner", "owner-1", {"reports": {"report_days": 30}})
    _paper(session, "stuck")
    assert expire_paper_records(session, _backend(deletes=False), now=NOW) == {"expired": 1, "deleted": 0}
    assert _count(session, Job) == 1 and _count(session, Report) == 1
