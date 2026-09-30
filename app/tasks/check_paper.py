"""Checkpointed Celery workflow for one uploaded paper."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import logging
import uuid

from celery import chain
from celery.exceptions import Ignore, Retry
from sqlalchemy import select

from app.database import SessionLocal
from app.log_safety import safe_exception_code
from app.models.job import Job, JobStatus
from app.services.paper_workflow import (
    PaperWorkflowError,
    extract_paper_job,
    fail_paper_job,
    finalize_paper_job,
    retrieve_paper_sources,
    verify_paper_sources,
)
from app.services.storage.backend import get_storage_backend
from app.services.workflow_retry import is_retryable as _is_retryable
from app.services.workflow_execution import WorkflowOwnershipLost, stage_execution
from app.tasks.celery_app import celery_app
from app.services.processing_metrics import measure_paper_stage
from app.services.paper_dispatch import (
    DISPATCH_KEY, MAX_STAGE_STARTS, STAGES, WorkflowBusy, attempt_id_for,
    attempt_matches, claim_dispatch, job_execution_lock, pending_stage,
    record_publication, recovery_candidates, timestamp,
)


logger = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_BACKOFF = 60


@contextmanager
def _job_execution_lock(session, job_id: str, stage: str):
    """Serialize all stages, not just identical stages, for one paper."""
    try:
        with job_execution_lock(session, job_id) as connection:
            yield connection
    except WorkflowBusy:
        raise PaperWorkflowError("job_stage_busy", "Another worker owns this paper workflow") from None


def _fail(job_id: str, exc: BaseException, attempt_id=None, *, session_factory=None) -> None:
    try:
        backend = get_storage_backend()
        with (session_factory or SessionLocal)() as session:
            job = session.scalar(select(Job).where(Job.id == uuid.UUID(job_id)).with_for_update())
            if not attempt_matches(job, attempt_id):
                return
            fail_paper_job(session, backend, job_id, exc)
    except Exception as failure_error:
        logger.error(
            "Could not persist terminal paper-job failure: %s",
            type(failure_error).__name__,
            extra={"job_id": job_id},
        )


def _retry_or_fail(task, job_id: str, exc: BaseException, attempt_id=None, stage=None, *, session_factory=None):
    if getattr(exc, "code", None) == "job_stage_busy":
        raise Ignore() from None
    starts = getattr(getattr(task, "request", None), "retries", 0) + 1
    if attempt_id is not None:
        with (session_factory or SessionLocal)() as session:
            job = session.scalar(select(Job).where(Job.id == uuid.UUID(job_id)).with_for_update())
            if not attempt_matches(job, attempt_id):
                raise Ignore() from None
            intent = dict(job.upload_evidence[DISPATCH_KEY])
            starts = intent.get("stage_starts", {}).get(stage, 0)
            if _is_retryable(exc) and starts < MAX_STAGE_STARTS:
                delay = min(600, RETRY_BACKOFF * (2 ** max(0, starts - 1)))
                due = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
                intent["stage_retry_after"] = {**intent.get("stage_retry_after", {}), stage: due}
                intent["dispatch_after"] = due
                job.upload_evidence = {**job.upload_evidence, DISPATCH_KEY: intent}
                session.commit()
    if _is_retryable(exc) and starts < MAX_STAGE_STARTS:
        try:
            raise task.retry(
                exc=RuntimeError(f"Transient {type(exc).__name__}"),
                countdown=min(600, RETRY_BACKOFF * (2 ** max(0, starts - 1))),
            )
        except Retry as retry:
            raise retry from None
        except Exception:
            # Durable intent remains due even if publishing the retry fails.
            raise RuntimeError("Paper retry publication unavailable") from None
    if session_factory is not None:
        _fail(job_id, exc, attempt_id, session_factory=session_factory)
    elif attempt_id is None:
        _fail(job_id, exc)
    else:
        _fail(job_id, exc, attempt_id)
    # Where it failed, as application file:line frames only: no message, no
    # values, so no student or source text reaches the log.
    import traceback
    frames = [f"{frame.filename.rsplit('/app/', 1)[-1]}:{frame.lineno}"
              for frame in traceback.extract_tb(exc.__traceback__) if '/app/app/' in frame.filename][-6:]
    logger.error("Paper workflow failure location: %s %s", type(exc).__name__, " < ".join(reversed(frames)))
    # Celery logs and serializes terminal exception messages. Validation errors
    # can embed source/student input, so persist the typed failure locally but
    # never hand the original exception or its context to the task backend.
    raise RuntimeError(f"Paper workflow failed: {safe_exception_code(exc)}") from None


def dispatch_paper_workflow(job_id, attempt_id):
    """Publish a leased intent; ambiguous publication never rolls it back."""
    if not attempt_id:
        return None
    claimed = claim_dispatch(SessionLocal, job_id, expected_attempt=attempt_id)
    if claimed is None:
        return None
    tasks = {"extract": extract_paper_task, "retrieve": retrieve_paper_sources_task,
             "verify": verify_paper_sources_task, "finalize": finalize_paper_job_task}
    stages = ("extract", "finalize") if claimed["store_only"] else STAGES
    steps = [tasks[stage].si(job_id, attempt_id)
             for stage in stages[stages.index(claimed["stage"]):]]
    workflow = chain(*steps).apply_async()
    record_publication(SessionLocal, job_id, attempt_id, workflow.id)
    return str(workflow.id)


@celery_app.task(bind=True, name="check_paper", max_retries=MAX_RETRIES)
def check_paper_task(self, job_id: str, attempt_id=None):
    try:
        task_id = dispatch_paper_workflow(job_id, attempt_id)
    except Exception:
        # A scheduler will resume the committed intent after its publication lease.
        raise RuntimeError("Paper workflow publication unavailable") from None
    return {"job_id": job_id, "workflow_task_id": task_id}


@celery_app.task(name="recover_pending_paper_workflows", soft_time_limit=300, time_limit=360)
def recover_pending_paper_workflows():
    with SessionLocal() as session:
        candidates = [(str(job_id), attempt_id_for(session.get(Job, job_id)))
                      for job_id in recovery_candidates(session)]
    published = unavailable = 0
    for job_id, attempt_id in candidates:
        try:
            published += bool(dispatch_paper_workflow(job_id, attempt_id))
        except Exception:
            unavailable += 1
    return {"candidates": len(candidates), "published": published, "publication_unavailable": unavailable}


def _start_stage(session, job_id, attempt_id, stage):
    job = session.scalar(select(Job).where(Job.id == uuid.UUID(job_id)).with_for_update())
    if not attempt_matches(job, attempt_id):
        raise Ignore()
    pending = pending_stage(job)
    if pending is None or STAGES.index(pending) < STAGES.index(stage):
        raise Ignore()
    if pending != stage:
        return False
    intent = dict(job.upload_evidence[DISPATCH_KEY])
    due = intent.get("stage_retry_after", {}).get(stage)
    if due and timestamp(due) > datetime.now(timezone.utc):
        raise Ignore()
    starts = dict(intent.get("stage_starts", {}))
    if starts.get(stage, 0) >= MAX_STAGE_STARTS:
        raise PaperWorkflowError("workflow_stage_attempts_exhausted", "Paper stage retry limit reached")
    starts[stage] = starts.get(stage, 0) + 1
    intent["stage_starts"] = starts
    job.upload_evidence = {**job.upload_evidence, DISPATCH_KEY: intent}
    job.status = JobStatus.RUNNING
    session.commit()
    return True


def _run_stage(task, job_id, attempt_id, stage):
    with SessionLocal() as lock_session:
        try:
            with _job_execution_lock(lock_session, job_id, stage) as connection:
                with stage_execution(connection, SessionLocal) as owner:
                    factory = owner.session_factory
                    with factory() as session:
                        try:
                            if not _start_stage(session, job_id, attempt_id, stage):
                                return {"already_completed": True}
                            with measure_paper_stage(factory, job_id, stage, workflow_attempt=attempt_id):
                                try:
                                    backend = get_storage_backend()
                                    if stage == "verify":
                                        return verify_paper_sources(factory, backend, job_id)
                                    operation = {"extract": extract_paper_job, "retrieve": retrieve_paper_sources,
                                                 "finalize": finalize_paper_job}[stage]
                                    return operation(session, backend, job_id)
                                finally:
                                    session.rollback()
                        except (Ignore, Retry):
                            raise
                        except Exception as exc:
                            session.rollback()
                            owner.check()
                            return _retry_or_fail(task, job_id, exc, attempt_id, stage, session_factory=factory)
        except WorkflowOwnershipLost:
            # The committed scheduling intent owns recovery. This execution
            # may neither reconnect nor fail/update a same-attempt successor.
            raise Ignore() from None
        except PaperWorkflowError as exc:
            if exc.code == "job_stage_busy":
                raise Ignore() from None
            raise


@celery_app.task(bind=True, name="extract_paper", max_retries=MAX_RETRIES)
def extract_paper_task(self, job_id: str, attempt_id=None):
    return _run_stage(self, job_id, attempt_id, "extract")


@celery_app.task(bind=True, name="retrieve_paper_sources", max_retries=MAX_RETRIES)
def retrieve_paper_sources_task(self, job_id: str, attempt_id=None):
    return _run_stage(self, job_id, attempt_id, "retrieve")


@celery_app.task(bind=True, name="verify_paper_sources", max_retries=MAX_RETRIES)
def verify_paper_sources_task(self, job_id: str, attempt_id=None):
    return _run_stage(self, job_id, attempt_id, "verify")


@celery_app.task(bind=True, name="finalize_paper_job", max_retries=MAX_RETRIES)
def finalize_paper_job_task(self, job_id: str, attempt_id=None):
    return _run_stage(self, job_id, attempt_id, "finalize")
