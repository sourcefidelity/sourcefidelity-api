"""Keep the submitted file so a paper can be checked again (`recheck-copy-v1`).

Owner decision 2026-10-13 (S1 step 3c). The scope's storage settings at
upload time decide whether the submitted file outlives its check, and for how
many days after the check finishes (0: until the paper's record is deleted).
The copy is operational and stays inside temporary paper retention
(`paper_retention.py`): it is never read by any comparison or corpus, and it
is deleted when its period ends, when the run fails, or with the record.

"Check Again" starts a new check of the kept file, exactly as a new upload
would (whole-paper, reference and model-answer reuse all apply), linked to
the submission it re-checks (`recheck_of_job_id`, `submission_root_job_id`)
so the report page follows the newest version and a future comparison
history can count one submission once.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus

POLICY = "recheck-copy-v1"
EVIDENCE_KEY = "recheck_copy"
# A flat marker the scheduled clean-up can filter on before its batch limit.
KEPT_MARKER = "recheck_copy_kept"
# "Until deleted": the record's own expiry removes the file with the record.
UNTIL_DELETED = datetime(9999, 12, 31, tzinfo=timezone.utc)


def retention_at_upload(session: Session | None, scope_type: str, scope_id: str) -> dict | None:
    """What a new upload records: keep the file for N days after its check, or None."""
    from app.services.storage_policy import policy_for
    papers = policy_for(session, scope_type, scope_id).papers
    if not papers.keep_submitted_file:
        return None
    return {"policy": POLICY, "days": int(papers.submitted_file_days)}


def kept_by_policy(job: Job) -> bool:
    return ((job.upload_evidence or {}).get(EVIDENCE_KEY) or {}).get("policy") == POLICY


def mark_kept(job: Job, now: datetime | None = None) -> None:
    """At the end of a successful check: keep the file and set when it goes."""
    current = now or datetime.now(timezone.utc)
    record = dict((job.upload_evidence or {}).get(EVIDENCE_KEY) or {})
    days = int(record.get("days") or 0)
    until = current + timedelta(days=days) if days else None
    record["kept_until"] = until.isoformat() if until else None
    job.upload_evidence = {**(job.upload_evidence or {}), EVIDENCE_KEY: record, KEPT_MARKER: "true"}
    job.input_expires_at = until or UNTIL_DELETED


def available(job: Job | None, now: datetime | None = None) -> bool:
    """Whether this job's submitted file is still kept and can be checked again."""
    if job is None or job.status != JobStatus.COMPLETED or not job.input_storage_key:
        return False
    evidence = job.upload_evidence or {}
    if job.input_deleted_at is not None or evidence.get(KEPT_MARKER) != "true" or evidence.get("input_cleanup_started"):
        return False
    expires = job.input_expires_at
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires is None or expires > (now or datetime.now(timezone.utc))


def lineage(session: Session, job: Job) -> list[Job]:
    """This job and every re-check after it, oldest first."""
    chain, seen = [job], {job.id}
    while True:
        following = (chain[-1].upload_evidence or {}).get("recheck_job_id")
        if not following:
            return chain
        try:
            import uuid
            later = session.get(Job, uuid.UUID(str(following)))
        except ValueError:
            return chain
        if later is None or later.id in seen:
            return chain
        chain.append(later)
        seen.add(later.id)


class RecheckUnavailable(Exception):
    """The submitted file is no longer kept."""


class RecheckInProgress(Exception):
    """A re-check of this submission is already running."""


def start_recheck(session: Session, backend, job: Job, *, scope_type: str, scope_id: str):
    """Check the kept file again. Returns ("reused", report) when the existing
    report still holds, or ("started", new_job) after queueing a new check."""
    from app.services.paper_reuse import find_reusable_report, record_reuse
    from app.services.paper_upload import create_paper_job, load_paper_job_input
    chain = lineage(session, job)
    if chain[-1].status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
        raise RecheckInProgress()
    latest = next((item for item in reversed(chain) if available(item)), None)
    if latest is None:
        raise RecheckUnavailable()
    _, content = load_paper_job_input(session, backend, latest.id)
    evidence = latest.upload_evidence or {}
    configuration = evidence.get("assessment_configuration") or {}
    found = find_reusable_report(session, content=content, filename=latest.filename, scope_type=scope_type,
                                 scope_id=scope_id, assessment_configuration=configuration,
                                 assessment_id=latest.assessment_id)
    if found is not None:
        record_reuse(session, *found)
        session.commit()
        return "reused", found[1]
    new_job = create_paper_job(
        session, backend, content=content, filename=latest.filename, media_type=latest.input_media_type,
        title=latest.title, require_reference_links=bool(configuration.get("require_reference_links")),
        assessment_id=latest.assessment_id, scope_type=scope_type, scope_id=scope_id,
        recheck_copy=retention_at_upload(session, scope_type, scope_id))
    root = evidence.get("submission_root_job_id") or str(chain[0].id)
    new_job.upload_evidence = {**(new_job.upload_evidence or {}), "recheck_of_job_id": str(latest.id),
                               "submission_root_job_id": root}
    latest.upload_evidence = {**evidence, "recheck_job_id": str(new_job.id)}
    session.commit()
    return "started", new_job
