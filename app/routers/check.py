"""Submit a temporary paper for checkpointed citation checking."""

from typing import Optional
import secrets
from fastapi.responses import HTMLResponse

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


@router.get('/', response_class=HTMLResponse)
def paper_check_form(principal: AuthenticatedPrincipal = Depends(get_authenticated_principal)):
    principal.require(PAPER_CHECK_CAPABILITY)
    nonce = secrets.token_urlsafe(24)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Check a paper</title>
<style nonce="{nonce}">body{{max-width:42rem;margin:3rem auto;padding:1rem;font:17px/1.6 system-ui;color:#18212b}}fieldset{{margin:1.5rem 0;padding:1rem}}button{{padding:.6rem 1rem}}label{{display:block;margin:.75rem 0}}</style></head>
<body><h1>Check a paper</h1><form method="post" action="/check/" enctype="multipart/form-data">
<label>Paper (PDF or Word)<input type="file" name="file" accept=".pdf,.docx" required></label>
<label>Title (optional)<input name="title" maxlength="500"></label>
<fieldset><legend>Assessment requirements</legend>
<label><input type="checkbox" name="require_reference_links" value="true"> Require a DOI, URL, or library link for every reference</label>
<p>Off by default. Includes uncited references. Checks whether a link or DOI was supplied, not whether it works or provides public full text. Uncertain extraction is not flagged as a missing link.</p>
<p>This choice is saved with this paper's report. It does not change previous reports.</p></fieldset>
<button type="submit">Submit paper</button></form></body></html>''', headers={
        'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff',
        'Content-Security-Policy':f"default-src 'none'; style-src 'nonce-{nonce}'; form-action 'self'; base-uri 'none'; frame-ancestors 'self'"})


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def check_paper(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    store_only: bool = Form(False),
    require_reference_links: bool = Form(False, description='Assessment requirement: every reference must contain a DOI, URL, or library link. Checks presence, not reachability. Defaults to off.'),
    assessment_id: Optional[str] = Form(None, max_length=255),
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
            require_reference_links=require_reference_links,
            assessment_id=assessment_id,
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
        "assessment_configuration": job.upload_evidence.get('assessment_configuration'),
        "publication_pending": publication_pending,
    }
