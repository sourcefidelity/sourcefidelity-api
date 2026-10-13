"""Delete a paper's whole record once its scope's report retention ends.

Owner decision 2026-10-09 (`storage_policy`, `reports.report_days`): a scope
may keep reports for a number of days after the report was last built or
returned again; 0, the default, keeps them until deleted. When the period
ends, everything the check produced for that paper goes: its judgments,
annotations, paper copy, report versions, per-source checks and the job with
its extracted text. Kept sources are not part of a paper's record; they
follow their own settings. Scopes without a report period are never touched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus
from app.models.judgment import JudgmentCandidateResult, JudgmentRun, JudgmentSourceReserve
from app.models.report import PaperAnnotationRecord, Report, ReportPaperArtifactRecord, VerificationReportRecord
from app.models.storage_policy import StoragePolicyRecord
from app.services.storage_policy import policy_for

FINISHED = (JobStatus.COMPLETED, JobStatus.FAILED)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def last_activity(session: Session, job: Job) -> datetime:
    """When the paper's report was last built or returned again."""
    times = [_as_utc(job.updated_at), _as_utc(job.created_at)]
    times += [_as_utc(at) for at in session.scalars(select(Report.created_at).where(Report.job_id == job.id))]
    for reuse in (job.upload_evidence or {}).get("paper_reuse") or []:
        try:
            times.append(_as_utc(datetime.fromisoformat(reuse["at"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(t for t in times if t is not None)


def expired_jobs(session: Session, *, now: datetime | None = None, limit: int = 100) -> list[Job]:
    """Finished jobs in scopes with a report period whose period has ended."""
    current = _as_utc(now or datetime.now(timezone.utc))
    expired = []
    for record in session.scalars(select(StoragePolicyRecord)):
        days = policy_for(session, record.scope_type, record.scope_id).reports.report_days
        if days <= 0:
            continue
        cutoff = current - timedelta(days=days)
        for job in session.scalars(select(Job).where(
                Job.scope_type == record.scope_type, Job.scope_id == record.scope_id,
                Job.status.in_(FINISHED), Job.created_at <= cutoff).order_by(Job.created_at)):
            if last_activity(session, job) <= cutoff:
                expired.append(job)
                if len(expired) >= limit:
                    return expired
    return expired


def delete_paper_record(session: Session, backend, job: Job, *, now: datetime | None = None) -> bool:
    """Remove everything the check produced for this paper; False leaves it
    untouched when the stored paper copy cannot be deleted first."""
    from app.services.report_paper_artifact import cleanup_report_paper_artifact
    report_ids = list(session.scalars(select(Report.id).where(Report.job_id == job.id)))
    # A submitted file still kept (for checking again) goes with the record.
    if job.input_storage_key and job.input_deleted_at is None:
        try:
            backend.delete(job.input_storage_key)
            gone = not backend.exists(job.input_storage_key)
        except Exception:
            gone = False
        if not gone:
            session.rollback()
            return False
    for artifact in session.scalars(select(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id)):
        if not cleanup_report_paper_artifact(session, backend, artifact, now=now):
            session.rollback()
            return False
    if report_ids:
        session.execute(delete(PaperAnnotationRecord).where(PaperAnnotationRecord.report_id.in_(report_ids)))
        runs = select(JudgmentRun.id).where(JudgmentRun.report_id.in_(report_ids))
        session.execute(delete(JudgmentCandidateResult).where(JudgmentCandidateResult.run_id.in_(runs)))
        session.execute(delete(JudgmentRun).where(JudgmentRun.report_id.in_(report_ids)))
    session.execute(delete(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id))
    if report_ids:
        session.execute(update(Report).where(Report.id.in_(report_ids)).values(previous_report_id=None))
        session.execute(delete(Report).where(Report.id.in_(report_ids)))
    if job.paper_version_id:
        versions = VerificationReportRecord.paper_version_id == job.paper_version_id
        checks = select(VerificationReportRecord.id).where(versions)
        session.execute(delete(JudgmentCandidateResult).where(JudgmentCandidateResult.verification_report_id.in_(checks)))
        session.execute(delete(JudgmentSourceReserve).where(JudgmentSourceReserve.verification_report_id.in_(checks)))
        session.execute(update(VerificationReportRecord).where(versions).values(previous_report_id=None))
        session.execute(delete(VerificationReportRecord).where(versions))
    session.execute(delete(Job).where(Job.id == job.id))
    session.commit()
    return True


def expire_paper_records(session: Session, backend, *, now: datetime | None = None, dry_run: bool = False,
                         limit: int = 100) -> dict:
    """Delete (or, dry run, count) the paper records whose period has ended."""
    jobs = expired_jobs(session, now=now, limit=limit)
    if dry_run:
        return {"expired": len(jobs), "deleted": 0}
    deleted = sum(delete_paper_record(session, backend, job, now=now) for job in jobs)
    return {"expired": len(jobs), "deleted": deleted}
