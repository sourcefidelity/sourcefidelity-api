"""Periodic enforcement of durable source-representation expiry."""

from app.config import settings
from app.database import SessionLocal
from app.services.source_repository import (
    expire_representations,
    finalize_pending_object_deletions,
)
from app.services.storage import get_storage_backend
from app.tasks.celery_app import celery_app


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
    expired_total = 0
    with SessionLocal() as session:
        while True:
            expired = expire_representations(
                session,
                batch_size=_CLEANUP_BATCH_SIZE,
            )
            session.commit()
            expired_total += expired
            if expired < _CLEANUP_BATCH_SIZE:
                break

        objects_deleted = finalize_pending_object_deletions(session, backend)
        session.commit()

    return {
        "status": "ok",
        "representations_expired": expired_total,
        "objects_deleted": objects_deleted,
    }
