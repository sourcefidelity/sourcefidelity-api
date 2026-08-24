"""Retry cleanup for expired or interrupted transient verification runs."""

from app.database import SessionLocal
from app.services.storage import get_storage_backend
from app.services.verification_run import cleanup_stale_verification_runs
from app.tasks.celery_app import celery_app


@celery_app.task(
    name="cleanup_stale_verification_runs",
    soft_time_limit=300,
    time_limit=360,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=900,
    retry_jitter=True,
    max_retries=3,
)
def cleanup_stale_verification_run_objects() -> dict[str, int | str]:
    backend = get_storage_backend()
    with SessionLocal() as session:
        result = cleanup_stale_verification_runs(session, backend, batch_size=500)
    return {"status": "ok", **result}
