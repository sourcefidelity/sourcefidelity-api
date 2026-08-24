"""Checkpointed Celery workflow for one uploaded paper."""

import logging
import uuid

from celery import chain

from app.database import SessionLocal
from app.models.job import Job
from app.services.paper_workflow import (
    extract_paper_job,
    fail_paper_job,
    finalize_paper_job,
    retrieve_paper_sources,
    verify_paper_sources,
)
from app.services.storage.backend import get_storage_backend
from app.tasks.celery_app import celery_app


logger = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_BACKOFF = 60


def _fail(job_id: str, exc: BaseException) -> None:
    try:
        backend = get_storage_backend()
        with SessionLocal() as session:
            fail_paper_job(session, backend, job_id, exc)
    except Exception as failure_error:
        logger.error(
            "Could not persist terminal paper-job failure: %s",
            type(failure_error).__name__,
            extra={"job_id": job_id},
        )


def _retry_or_fail(task, job_id: str, exc: BaseException):
    if isinstance(exc, (ConnectionError, TimeoutError)) and task.request.retries < MAX_RETRIES:
        raise task.retry(
            exc=exc,
            countdown=min(600, RETRY_BACKOFF * (2 ** task.request.retries)),
        )
    _fail(job_id, exc)
    raise exc


@celery_app.task(bind=True, name="check_paper")
def check_paper_task(self, job_id: str):
    """Schedule durable stage checkpoints; never process the paper monolithically."""
    with SessionLocal() as session:
        try:
            parsed_job_id = uuid.UUID(job_id)
        except ValueError as exc:
            raise ValueError("Invalid paper job ID") from exc
        job = session.get(Job, parsed_job_id)
        if job is None:
            raise ValueError("Paper job does not exist")
        steps = [extract_paper_task.si(job_id)]
        if not job.store_only:
            steps.extend(
                [
                    retrieve_paper_sources_task.si(job_id),
                    verify_paper_sources_task.si(job_id),
                ]
            )
        steps.append(finalize_paper_job_task.si(job_id))
        workflow = chain(*steps).apply_async()
        job.task_id = workflow.id
        session.commit()
    return {"job_id": job_id, "workflow_task_id": workflow.id}


@celery_app.task(bind=True, name="extract_paper", max_retries=MAX_RETRIES)
def extract_paper_task(self, job_id: str):
    try:
        backend = get_storage_backend()
        with SessionLocal() as session:
            return extract_paper_job(session, backend, job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, exc)


@celery_app.task(bind=True, name="retrieve_paper_sources", max_retries=MAX_RETRIES)
def retrieve_paper_sources_task(self, job_id: str):
    try:
        backend = get_storage_backend()
        with SessionLocal() as session:
            return retrieve_paper_sources(session, backend, job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, exc)


@celery_app.task(bind=True, name="verify_paper_sources", max_retries=MAX_RETRIES)
def verify_paper_sources_task(self, job_id: str):
    try:
        return verify_paper_sources(SessionLocal, get_storage_backend(), job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, exc)


@celery_app.task(bind=True, name="finalize_paper_job", max_retries=MAX_RETRIES)
def finalize_paper_job_task(self, job_id: str):
    try:
        backend = get_storage_backend()
        with SessionLocal() as session:
            return finalize_paper_job(session, backend, job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, exc)
