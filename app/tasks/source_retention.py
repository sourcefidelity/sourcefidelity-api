"""Periodic enforcement of durable source-representation expiry."""

from app.config import settings
from app.database import SessionLocal
from app.services.source_repository import (
    expire_representations,
    finalize_pending_object_deletions,
)
from app.services.storage import get_storage_backend
from app.tasks.celery_app import celery_app
from app.services.source_upload_recovery import cleanup_source_upload_intents


_CLEANUP_BATCH_SIZE = 500


@celery_app.task(
    name="cleanup_expired_source_representations",
    soft_time_limit=300,
    time_limit=360,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=900,
    retry_jitter=True,
    max_retries=3,
)
def cleanup_expired_source_representations() -> dict[str, int | str]:
    """Remove expired references, then retry unreferenced object deletion."""
    if not settings.SOURCE_REPOSITORY_ENABLED:
        return {
            "status": "disabled",
            "representations_expired": 0,
            "objects_deleted": 0,
        }

    backend = get_storage_backend()
    with SessionLocal() as session:
        # One bounded batch per scheduled run prevents a large backlog from
        # monopolizing a worker or running past the task's hard time limit.
        expired_total = expire_representations(
            session,
            batch_size=_CLEANUP_BATCH_SIZE,
        )
        session.commit()
        upload_cleanup = cleanup_source_upload_intents(session, backend, batch_size=100)
        objects_deleted = finalize_pending_object_deletions(
            session, backend, batch_size=_CLEANUP_BATCH_SIZE
        )
        session.commit()

    return {
        "status": "ok",
        "representations_expired": expired_total,
        "objects_deleted": objects_deleted,
        **{f"upload_intents_{key}": value for key, value in upload_cleanup.items()},
    }
