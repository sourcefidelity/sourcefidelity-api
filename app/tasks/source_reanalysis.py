"""Targeted report reanalysis after an admitted user source upload."""

import logging

from app.tasks.celery_app import celery_app
from app.tasks.check_paper import dispatch_paper_workflow

logger = logging.getLogger(__name__)
# A busy report is usually finishing another upload's refresh within minutes.
BUSY_RETRY_SECONDS = 60
BUSY_MAX_RETRIES = 30


def schedule_uploaded_source_reanalysis(job_id: str, attempt_id: str) -> str | None:
    """Queue only verification and finalization for the affected source member."""
    return dispatch_paper_workflow(job_id, attempt_id)


def schedule_busy_upload_refresh(report_id: str, reference_id: str, representation_id: str) -> str | None:
    """Retry an accepted upload's refresh once the report's running refresh ends."""
    result = retry_uploaded_source_refresh.apply_async(
        args=[report_id, reference_id, representation_id], countdown=BUSY_RETRY_SECONDS)
    return getattr(result, "id", None)


@celery_app.task(bind=True, name="retry_uploaded_source_refresh", max_retries=BUSY_MAX_RETRIES)
def retry_uploaded_source_refresh(self, report_id: str, reference_id: str, representation_id: str) -> dict:
    """An accepted upload arrived while another refresh ran (2026-10-01: Langford's
    file was stored but the report never updated). Prepare its refresh against the
    report's latest version once the job is free, then dispatch it."""
    from app.database import SessionLocal
    from app.services.paper_workflow import PaperWorkflowError, prepare_uploaded_source_refresh
    from app.services.storage.backend import get_storage_backend
    with SessionLocal() as session:
        try:
            prepared = prepare_uploaded_source_refresh(
                session, get_storage_backend(), report_id=report_id,
                reference_id=reference_id, representation_id=representation_id)
        except PaperWorkflowError as exc:
            if getattr(exc, "code", None) == "report_reanalysis_busy":
                raise self.retry(countdown=BUSY_RETRY_SECONDS)
            logger.warning("Uploaded source refresh not prepared (code=%s)", getattr(exc, "code", None))
            return {"status": "not_prepared", "code": getattr(exc, "code", None)}
    if not prepared.get("scheduled"):
        return {"status": str(prepared.get("status") or "already_current")}
    task_id = schedule_uploaded_source_reanalysis(str(prepared["job_id"]), prepared["attempt_id"])
    return {"status": "scheduled", "task_id": task_id}
