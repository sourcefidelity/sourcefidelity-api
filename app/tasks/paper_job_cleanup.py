"""Scheduled cleanup for terminal or expired temporary paper inputs."""

from app.database import SessionLocal
from app.services.paper_upload import cleanup_stale_paper_job_inputs
from app.services.report_paper_artifact import cleanup_expired_report_paper_artifacts
from app.services.storage.backend import get_storage_backend
from app.tasks.celery_app import celery_app


@celery_app.task(
    name="cleanup_stale_paper_job_inputs",
    soft_time_limit=300,
    time_limit=360,
)
def cleanup_stale_paper_job_input_objects() -> dict[str, int]:
    with SessionLocal() as session:
        inputs = cleanup_stale_paper_job_inputs(
            session,
            get_storage_backend(),
            batch_size=100,
        )
        artifacts = cleanup_expired_report_paper_artifacts(
            session,
            get_storage_backend(),
            batch_size=100,
        )
        return {**inputs, **artifacts}
