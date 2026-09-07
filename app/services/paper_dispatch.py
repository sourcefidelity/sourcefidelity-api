"""Durable scheduling intent and execution fences for checkpointed paper jobs."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import uuid

from sqlalchemy import select, text

from app.models.job import Job, JobStage, JobStatus


DISPATCH_KEY = "workflow_dispatch_v1"
DISPATCH_INTERVAL_SECONDS = 60
MAX_STAGE_STARTS = 4
ACTIVE_STATUSES = {JobStatus.PENDING, JobStatus.RUNNING}
STAGES = ("extract", "retrieve", "verify", "finalize")


class WorkflowBusy(RuntimeError):
    code = "job_stage_busy"


def attempt_id_for(job):
    return (job.upload_evidence or {}).get(DISPATCH_KEY, {}).get("attempt_id")


def prepare_dispatch(job, *, attempt_id=None):
    """Persist with the enclosing state transition, never before input upload."""
    attempt_id = attempt_id or str(uuid.uuid4())
    job.upload_evidence = {**(job.upload_evidence or {}), DISPATCH_KEY: {
        "attempt_id": attempt_id,
        "dispatch_after": datetime.now(timezone.utc).isoformat(),
        "dispatch_count": 0,
        "stage_starts": {},
        "stage_retry_after": {},
    }}
    return attempt_id


def attempt_matches(job, attempt_id):
    # Untokened legacy deliveries may never operate a managed attempt. Recovery
    # deliberately does not adopt historical jobs that lack a dispatch record.
    return bool(job and job.status in ACTIVE_STATUSES
                and attempt_id and attempt_id_for(job) == attempt_id)


def pending_stage(job):
    if job.status not in ACTIVE_STATUSES:
        return None
    if job.stage in {JobStage.UPLOADED, JobStage.EXTRACTING}:
        return "extract"
    if job.stage in {JobStage.EXTRACTED, JobStage.RETRIEVING}:
        return "finalize" if job.store_only else "retrieve"
    if job.stage in {JobStage.RETRIEVED, JobStage.VERIFYING}:
        return "verify"
    if job.stage in {JobStage.VERIFIED, JobStage.FINALIZING}:
        return "finalize"
    return None


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError):
        # Invalid scheduling provenance is not permission to run a job.
        return datetime.max.replace(tzinfo=timezone.utc)


@contextmanager
def job_execution_lock(session, job_id):
    """One physical PostgreSQL connection owns every stage of this job."""
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        yield
        return
    seed = hashlib.sha256(f"paper-workflow:{job_id}".encode()).digest()
    lock_id = int.from_bytes(seed[:8], "big", signed=True)
    with bind.connect() as connection:
        if not connection.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": lock_id}).scalar_one():
            raise WorkflowBusy("Another worker owns this paper workflow")
        try:
            yield connection
        finally:
            # A disconnected backend has already released its locks. Never
            # reconnect here or mask the stage's ownership-loss exception.
            if not connection.invalidated and not connection.closed:
                try:
                    connection.rollback()
                    connection.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": lock_id})
                except Exception:
                    connection.invalidate()


def claim_dispatch(session_factory, job_id, *, expected_attempt=None, now=None):
    """Lease one publication before sending; interruption leaves it recoverable."""
    current = now or datetime.now(timezone.utc)
    with session_factory() as session:
        try:
            with job_execution_lock(session, job_id):
                job = session.scalar(select(Job).where(Job.id == uuid.UUID(str(job_id))).with_for_update())
                if job is None or job.status not in ACTIVE_STATUSES or not attempt_id_for(job):
                    return None
                if expected_attempt is not None and attempt_id_for(job) != expected_attempt:
                    return None
                intent = dict(job.upload_evidence[DISPATCH_KEY])
                stage = pending_stage(job)
                if stage is None or timestamp(intent.get("dispatch_after")) > current:
                    return None
                intent["dispatch_after"] = (current + timedelta(seconds=DISPATCH_INTERVAL_SECONDS)).isoformat()
                intent["dispatch_count"] = int(intent.get("dispatch_count", 0)) + 1
                job.upload_evidence = {**job.upload_evidence, DISPATCH_KEY: intent}
                session.commit()
                return {"job_id": str(job.id), "attempt_id": intent["attempt_id"],
                        "stage": stage, "store_only": job.store_only}
        except WorkflowBusy:
            return None


def record_publication(session_factory, job_id, attempt_id, task_id):
    with session_factory() as session:
        job = session.scalar(select(Job).where(Job.id == uuid.UUID(str(job_id))).with_for_update())
        if not attempt_matches(job, attempt_id):
            return
        job.task_id = str(task_id)
        session.commit()


def recovery_candidates(session, *, now=None, limit=25):
    current = now or datetime.now(timezone.utc)
    return list(session.scalars(select(Job.id).where(
        Job.status.in_(ACTIVE_STATUSES),
        Job.upload_evidence[DISPATCH_KEY]["attempt_id"].as_string().is_not(None),
        Job.upload_evidence[DISPATCH_KEY]["dispatch_after"].as_string() <= current.isoformat(),
    ).order_by(Job.updated_at).limit(max(1, min(limit, 100)))))
