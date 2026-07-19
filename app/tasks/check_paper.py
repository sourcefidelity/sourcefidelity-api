"""Celery task for citation checking a paper."""

import logging

from celery.exceptions import SoftTimeLimitExceeded

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Constants for retry behavior
MAX_RETRIES = 3
RETRY_BACKOFF = 60  # seconds


@celery_app.task(
    bind=True,
    name="check_paper",
    autoretry_for=(SoftTimeLimitExceeded, ConnectionError, TimeoutError),
    max_retries=MAX_RETRIES,
    default_retry_delay=RETRY_BACKOFF,
    retry_backoff=True,       # Exponential backoff: 60s, 120s, 240s
    retry_backoff_max=600,    # Max 10 minutes between retries
    retry_jitter=True,        # Add randomness to avoid thundering herd
    acks_late=True,           # Only ack after successful completion
    reject_on_worker_lost=True,  # Requeue if worker crashes
)
def check_paper_task(self, job_id: str, file_path: str):
    """
    Run the full citation checking workflow.

    This task orchestrates: text extraction → reference parsing →
    OpenAlex lookup → full-text retrieval → verification → report generation.

    Args:
        job_id: UUID of the job in PostgreSQL.
        file_path: Path to the uploaded paper file.

    Raises:
        NotImplementedError: Until Phase 5 implementation.
        Retry: On transient failures, with exponential backoff.
    """
    logger.info(
        "Starting check_paper_task",
        extra={"job_id": job_id, "file_path": file_path, "attempt": self.request.retries},
    )

    try:
        # Phase 5 placeholder
        raise NotImplementedError("Phase 5 – Celery task wiring")

    except NotImplementedError:
        # Don't retry implementation errors
        logger.error("Task not yet implemented", extra={"job_id": job_id})
        raise

    except SoftTimeLimitExceeded:
        logger.warning("Task timed out", extra={"job_id": job_id, "attempt": self.request.retries})
        if self.request.retries < MAX_RETRIES:
            raise self.retry(exc=SoftTimeLimitExceeded("Task timed out"))
        raise

    except Exception as exc:
        logger.exception(
            "Task failed unexpectedly",
            extra={"job_id": job_id, "attempt": self.request.retries},
        )
        if self.request.retries < MAX_RETRIES:
            raise self.retry(exc=exc)
        # Max retries exhausted – task goes to dead letter queue
        logger.critical(
            "Task failed after max retries – sending to dead letter queue",
            extra={"job_id": job_id, "error": str(exc)},
        )
        raise
