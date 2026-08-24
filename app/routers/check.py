"""Submit a temporary paper for checkpointed citation checking."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services.paper_upload import (
    PaperUploadError,
    cleanup_paper_job_input,
    create_paper_job,
)
from app.services.storage.backend import StorageBackend, get_storage_backend
from app.tasks.check_paper import check_paper_task


router = APIRouter()


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def check_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    store_only: bool = Form(False),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Validate, temporarily store, and enqueue one personal-profile paper."""
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported paper type; use PDF or DOCX")
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read(maximum + 1)
    try:
        job = create_paper_job(
            session,
            backend,
            content=content,
            filename=file.filename or "paper",
            media_type=file.content_type,
            title=title,
            store_only=store_only,
            scope_type="personal_owner",
            scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
        )
    except PaperUploadError as exc:
        code = 503 if exc.code in {"storage_unavailable", "malware_scan_unavailable"} else 422
        raise HTTPException(
            status_code=code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    try:
        queued = check_paper_task.delay(str(job.id))
        job.task_id = queued.id
        session.commit()
    except Exception as exc:
        job.status = "failed"
        job.stage = "failed"
        job.error_message = "workflow_enqueue_failed"
        session.commit()
        cleanup_paper_job_input(session, backend, job.id)
        raise HTTPException(status_code=503, detail="Paper workflow queue is unavailable") from exc
    return {
        "job_id": str(job.id),
        "paper_version_id": job.paper_version_id,
        "status": job.status,
        "stage": job.stage,
        "store_only": job.store_only,
    }
