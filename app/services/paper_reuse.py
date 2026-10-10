"""Return an unchanged paper's existing report instead of checking it again.

Owner decision 2026-10-09 (efficiency): the same file, submitted again in the
same scope with the same assessment settings, under the same code and
behaviour settings, gets the report it already has. Anything different is
checked as usual. The fingerprint covers every application source file and
every setting except credentials and addresses, so any code or configuration
change makes a fresh check.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, settings
from app.models.job import Job, JobStatus
from app.models.report import Report

FINGERPRINT_KEY = "pipeline_fingerprint"
REUSE_KEY = "paper_reuse"
# Credentials and addresses never enter the fingerprint, not even hashed.
_EXCLUDED_SETTING = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|URL|DSN|ACCESS|PRIVATE|COOKIE|SESSION|SALT|SIGN|EMAIL",
    re.IGNORECASE)


@lru_cache(maxsize=1)
def pipeline_fingerprint() -> str:
    """The code and behaviour settings this process checks papers with."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in sorted(Settings.model_fields):
        if _EXCLUDED_SETTING.search(name):
            continue
        value = getattr(settings, name, None)
        if isinstance(value, SecretStr):
            continue
        digest.update(f"{name}={value!r}\n".encode())
    return digest.hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def find_reusable_report(
    session: Session,
    *,
    content: bytes,
    filename: str,
    scope_type: str,
    scope_id: str,
    assessment_configuration: dict,
    assessment_id: str | None,
    now: datetime | None = None,
) -> tuple[Job, Report] | None:
    """The completed job and latest report this exact submission already has, or None."""
    current = _as_utc(now or datetime.now(timezone.utc))
    window = int(settings.SEARCH_RERUN_REUSE_DAYS or 0)
    if window <= 0:
        return None
    digest = hashlib.sha256(content).hexdigest()
    fingerprint = pipeline_fingerprint()
    jobs = session.scalars(
        select(Job).where(Job.input_sha256 == digest, Job.status == JobStatus.COMPLETED,
                          Job.scope_type == scope_type, Job.scope_id == scope_id,
                          Job.filename == filename)
        .order_by(Job.created_at.desc()).limit(5))
    for job in jobs:
        evidence = job.upload_evidence or {}
        if (job.store_only or (job.assessment_id or None) != (assessment_id or None)
                or evidence.get("assessment_configuration") != assessment_configuration):
            continue
        if job.updated_at is None or _as_utc(job.updated_at) < current - timedelta(days=window):
            continue
        report = session.scalar(select(Report).where(Report.job_id == job.id)
                                .order_by(Report.report_version.desc(), Report.created_at.desc()).limit(1))
        if report is None or (report.report_json or {}).get(FINGERPRINT_KEY) != fingerprint:
            continue
        if not _paper_copy_available(session, job, current):
            continue
        return job, report
    return None


def _paper_copy_available(session: Session, job: Job, current: datetime) -> bool:
    if not settings.REPORT_PAPER_COPY_ENABLED:
        return True
    from app.models.report import ReportPaperArtifactRecord
    record = session.scalar(select(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id))
    return bool(record is not None and record.deleted_at is None and record.storage_key
                and _as_utc(record.expires_at) > current)


def record_reuse(session: Session, job: Job, report: Report, now: datetime | None = None) -> None:
    """Note the reuse and keep the report's paper copy for another retention period."""
    current = _as_utc(now or datetime.now(timezone.utc))
    evidence = dict(job.upload_evidence or {})
    evidence[REUSE_KEY] = [*(evidence.get(REUSE_KEY) or [])[-19:],
                           {"at": current.isoformat(), "report_id": str(report.id)}]
    job.upload_evidence = evidence
    if settings.REPORT_PAPER_COPY_ENABLED:
        from app.models.report import ReportPaperArtifactRecord
        record = session.scalar(select(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id))
        if record is not None:
            from app.services.storage_policy import policy_for
            record.expires_at = current + timedelta(
                days=policy_for(session, job.scope_type, job.scope_id).papers.paper_copy_days)
    session.commit()
