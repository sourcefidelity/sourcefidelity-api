"""Submit a temporary paper for checkpointed citation checking."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.security import (
    AuthenticatedPrincipal,
    PAPER_CHECK_CAPABILITY,
    get_authenticated_principal,
    require_same_origin_request,
)
from app.services.paper_upload import (
    PaperUploadError,
    create_paper_job,
)
from app.services.storage.backend import StorageBackend, get_storage_backend
from app.tasks.check_paper import check_paper_task
from app.services.paper_dispatch import attempt_id_for


router = APIRouter()


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def check_paper(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    store_only: bool = Form(False),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
):
    """Validate, temporarily store, and enqueue one personal-profile paper."""
    principal.require(PAPER_CHECK_CAPABILITY)
    require_same_origin_request(request)
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported paper type; use PDF or DOCX")
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content = file.file.read(maximum + 1)
    try:
        job = create_paper_job(
            session,
            backend,
            content=content,
            filename=file.filename or "paper",
            media_type=file.content_type,
            title=title,
            store_only=store_only,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except PaperUploadError as exc:
        code = 503 if exc.code in {"storage_unavailable", "malware_scan_unavailable"} else 422
        raise HTTPException(
            status_code=code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    publication_pending = False
    try:
        queued = check_paper_task.delay(str(job.id), attempt_id_for(job))
        job.task_id = queued.id
        session.commit()
    except Exception:
        session.rollback()
        publication_pending = True
    return {
        "job_id": str(job.id),
        "paper_version_id": job.paper_version_id,
        "status": job.status,
        "stage": job.stage,
        "store_only": job.store_only,
        "publication_pending": publication_pending,
    }
