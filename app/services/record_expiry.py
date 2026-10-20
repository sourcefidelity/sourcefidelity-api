"""Delete a paper's whole record once its scope's report retention ends.

Owner decision 2026-10-09 (`storage_policy`, `reports.report_days`): a scope
may keep reports for a number of days after the report was last built or
returned again; 0, the default, keeps them until deleted. When the period
ends, everything the check produced for that paper goes: its judgments,
annotations, paper copy, report versions, per-source checks and the job with
its extracted text. Kept sources are not part of a paper's record; they
follow their own settings. Scopes without a report period are never touched.

Only the latest report (owner decision 2026-10-20, `reports.keep_latest_only`,
on by default): once a newer report of a paper is finished, its older versions
and the runs a Check Again replaced are deleted with their judgments, notes and
checks; each deleted report's address opens the latest report.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus
from app.models.judgment import JudgmentCandidateResult, JudgmentRun, JudgmentSourceReserve
from app.models.report import (PaperAnnotationRecord, Report, ReportPaperArtifactRecord, ReportRedirect,
                               VerificationReportRecord)
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


def latest_report(session: Session, job_id) -> Report | None:
    return session.scalar(select(Report).where(Report.job_id == job_id)
                          .order_by(Report.report_version.desc(), Report.created_at.desc()).limit(1))


def current_report_id(session: Session, report_id):
    """The report a deleted older report's address now opens, or None."""
    return session.scalar(select(ReportRedirect.report_id).where(ReportRedirect.old_report_id == report_id))


def _busy(session: Session, job: Job) -> bool:
    """Work that may still read this job's reports: an unfinished run, a
    targeted refresh or queued uploads, or a Judgment run in progress."""
    from app.services.judgment_runs import ACTIVE
    evidence = job.upload_evidence or {}
    if job.status not in FINISHED or evidence.get("targeted_source_refresh") or evidence.get("queued_source_uploads"):
        return True
    reports = select(Report.id).where(Report.job_id == job.id)
    return session.scalar(select(JudgmentRun.id).where(JudgmentRun.report_id.in_(reports),
                                                       JudgmentRun.status.in_(ACTIVE)).limit(1)) is not None


def _redirect(session: Session, old_ids: list, target_id) -> None:
    """Point the given reports' addresses, and any already pointing at them, to the target."""
    if not old_ids:
        return
    session.execute(update(ReportRedirect).where(ReportRedirect.report_id.in_(old_ids)).values(report_id=target_id))
    existing = set(session.scalars(select(ReportRedirect.old_report_id).where(ReportRedirect.old_report_id.in_(old_ids))))
    for old in old_ids:
        if old not in existing:
            session.add(ReportRedirect(old_report_id=old, report_id=target_id))
    session.flush()


def _older_versions(session: Session, job: Job, latest: Report) -> list:
    return [rid for rid in session.scalars(select(Report.id).where(Report.job_id == job.id)) if rid != latest.id]


def delete_older_versions(session: Session, job: Job) -> int:
    """Delete every report version of this job but the latest; returns how many."""
    latest = latest_report(session, job.id)
    if latest is None:
        return 0
    older = _older_versions(session, job, latest)
    if not older:
        return 0
    _redirect(session, older, latest.id)
    session.execute(update(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.report_id.in_(older))
                    .values(report_id=latest.id))
    session.execute(delete(PaperAnnotationRecord).where(PaperAnnotationRecord.report_id.in_(older)))
    runs = select(JudgmentRun.id).where(JudgmentRun.report_id.in_(older))
    session.execute(delete(JudgmentCandidateResult).where(JudgmentCandidateResult.run_id.in_(runs)))
    session.execute(delete(JudgmentRun).where(JudgmentRun.report_id.in_(older)))
    session.execute(update(Report).where(Report.job_id == job.id).values(previous_report_id=None))
    session.execute(delete(Report).where(Report.id.in_(older)))
    session.commit()
    return len(older)


def _replacing_check(session: Session, job: Job) -> tuple[Job, Report] | None:
    """The finished Check Again run, with its report, that replaces this job;
    None while a later run is still working or none finished with a report."""
    from app.services.recheck_copy import lineage
    replacement = None
    for later in lineage(session, job)[1:]:
        if later.status not in FINISHED:
            return None
        report = latest_report(session, later.id) if later.status == JobStatus.COMPLETED else None
        if report is not None:
            replacement = (later, report)
    return replacement


def keep_latest_reports(session: Session, backend, *, now: datetime | None = None, dry_run: bool = False) -> dict:
    """Delete (or, dry run, count) older report versions and replaced runs in
    scopes that keep only the latest report."""
    policies: dict = {}
    versions = jobs = 0
    for job_id in list(session.scalars(select(Job.id).where(Job.status.in_(FINISHED)).order_by(Job.created_at))):
        job = session.get(Job, job_id)
        if job is None:
            continue
        scope = (job.scope_type, job.scope_id)
        if scope not in policies:
            policies[scope] = policy_for(session, *scope).reports.keep_latest_only
        if not policies[scope] or _busy(session, job):
            continue
        replacement = _replacing_check(session, job)
        if replacement is not None:
            if _busy(session, replacement[0]):
                continue
            if dry_run:
                jobs += 1
                continue
            _redirect(session, list(session.scalars(select(Report.id).where(Report.job_id == job.id))),
                      replacement[1].id)
            jobs += delete_paper_record(session, backend, job, now=now)
            continue
        latest = latest_report(session, job.id)
        if latest is None:
            continue
        if dry_run:
            versions += len(_older_versions(session, job, latest))
        else:
            versions += delete_older_versions(session, job)
    return {"older_versions_deleted": versions, "replaced_runs_deleted": jobs}
