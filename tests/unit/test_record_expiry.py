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


# --- Only the latest report (owner decision 2026-10-20) ----------------------
from app.models.report import ReportRedirect  # noqa: E402
from app.services.record_expiry import current_report_id, keep_latest_reports  # noqa: E402


def _version(session, job_id, number, *, judged=True):
    report = Report(job_id=job_id, report_json={}, report_version=number, created_at=NOW - timedelta(days=10 - number))
    session.add(report)
    session.flush()
    if judged:
        session.add(JudgmentRun(report_id=report.id, scope_type="personal_owner", scope_id="owner-1",
                                requested_by="t", notice_version="n", panel_version="p", status="completed"))
    session.commit()
    return report.id


def _judged(session):
    """The fixture's Judgment runs finished (a queued run keeps every version)."""
    for run in session.scalars(select(JudgmentRun).where(JudgmentRun.status != "running")):
        run.status = "completed"
    session.commit()


def _reports(session, job_id):
    return set(session.scalars(select(Report.id).where(Report.job_id == job_id)))


def test_older_versions_go_and_their_addresses_open_the_latest(session):
    job = _paper(session, "p")
    first = next(iter(_reports(session, job)))
    second = _version(session, job, 2)
    third = _version(session, job, 3)
    _judged(session)
    assert keep_latest_reports(session, _backend(), dry_run=True) == {"older_versions_deleted": 2,
                                                                      "replaced_runs_deleted": 0}
    assert _count(session, Report) == 3
    assert keep_latest_reports(session, _backend())["older_versions_deleted"] == 2
    assert _reports(session, job) == {third}
    assert {r.report_id for r in session.scalars(select(JudgmentRun))} == {third}
    assert session.scalar(select(ReportPaperArtifactRecord.report_id)) == third
    assert current_report_id(session, first) == current_report_id(session, second) == third
    assert keep_latest_reports(session, _backend())["older_versions_deleted"] == 0


def test_nothing_goes_while_a_judgment_runs_or_when_the_user_keeps_every_version(session):
    job = _paper(session, "p")
    newest = _version(session, job, 2, judged=False)
    _judged(session)
    session.add(JudgmentRun(report_id=newest, scope_type="personal_owner", scope_id="owner-1", requested_by="t",
                            notice_version="n", panel_version="p", status="running"))
    session.commit()
    assert keep_latest_reports(session, _backend())["older_versions_deleted"] == 0
    session.scalars(select(JudgmentRun).where(JudgmentRun.report_id == newest)).one().status = "completed"
    set_policy(session, "personal_owner", "owner-1", {"reports": {"keep_latest_only": False}})
    session.commit()
    assert keep_latest_reports(session, _backend())["older_versions_deleted"] == 0
    assert len(_reports(session, job)) == 2


def test_a_finished_check_again_replaces_the_earlier_run(session):
    earlier = _paper(session, "first")
    later = _paper(session, "again", age_days=1)
    old_report = next(iter(_reports(session, earlier)))
    new_report = next(iter(_reports(session, later)))
    first_job, second_job = session.get(Job, earlier), session.get(Job, later)
    first_job.upload_evidence = {"recheck_job_id": str(later)}
    second_job.status = JobStatus.RUNNING
    session.commit()
    _judged(session)
    assert keep_latest_reports(session, _backend())["replaced_runs_deleted"] == 0   # still checking
    second_job.status = JobStatus.COMPLETED
    session.commit()
    assert keep_latest_reports(session, _backend())["replaced_runs_deleted"] == 1
    assert session.get(Job, earlier) is None and session.get(Job, later) is not None
    assert current_report_id(session, old_report) == new_report
    assert _count(session, VerificationReportRecord) == 1


def test_an_address_already_moved_follows_the_next_replacement(session):
    job = _paper(session, "p")
    first = next(iter(_reports(session, job)))
    _version(session, job, 2)
    _judged(session)
    keep_latest_reports(session, _backend())
    later = _paper(session, "again", age_days=1)
    _judged(session)
    session.get(Job, job).upload_evidence = {"recheck_job_id": str(later)}
    session.commit()
    keep_latest_reports(session, _backend())
    assert current_report_id(session, first) == next(iter(_reports(session, later)))
    assert _count(session, ReportRedirect) == 2
