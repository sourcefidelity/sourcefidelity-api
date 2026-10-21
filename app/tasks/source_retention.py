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


@celery_app.task(name="expire_paper_records", soft_time_limit=300, time_limit=360)
def expire_paper_records_task() -> dict[str, int]:
    """Delete paper records whose scope's report period has ended (owner
    decision 2026-10-09); scopes without a report period are never touched.
    Remembered model answers older than the reuse window go too, and so do
    older report versions where only the latest is kept (2026-10-20)."""
    from app.services import llm_cache
    from app.services.record_expiry import expire_paper_records, keep_latest_reports
    with SessionLocal() as session:
        result = expire_paper_records(session, get_storage_backend())
        result.update(keep_latest_reports(session, get_storage_backend()))
        result["model_answers_expired"] = llm_cache.expire(session)
        return result
