"""Authenticated delivery of an immutable evidence report and paper surface."""

from __future__ import annotations

import secrets
from urllib.parse import urlsplit
import re
import uuid
from io import BytesIO
from html import escape

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session
import fitz

from app.database import get_db
from app.security import (
    AuthenticatedPrincipal,
    REPORT_ANNOTATION_CAPABILITY,
    REPORT_PAPER_CAPABILITY,
    REPORT_SOURCE_CAPABILITY,
    REPORT_SESSION_COOKIE,
    REPORT_VIEW_CAPABILITY,
    authenticate_personal_bearer,
    create_report_session,
    ensure_report_auth_configured,
    get_report_browser_principal,
    get_report_principal,
    require_same_origin_request,
)
from app.models.job import Job
from app.models.report import Report, VerificationReportRecord
from app.config import settings
from app.services.evidence_report import (
    EvidenceReportAuthorizationError,
    EvidenceReportError,
    enable_authenticated_paper_actions,
    load_authorized_evidence_report_bundle,
    render_evidence_report_html,
)
from app.services.storage.backend import StorageBackend, get_storage_backend
from app.services.source_navigation import (
    SourceNavigationDescriptor,
    authorize_source_navigation_document,
)
from app.services.verification_evidence import EvidenceAuthorizationError
from app.routers.sources import upload_source as admit_uploaded_source
from app.services.pdf_verifier import verify_instructor_upload
from app.services.paper_annotations import (
    PaperAnnotationError,
    create_paper_annotation,
    list_current_paper_annotations,
    revise_paper_annotation,
)
from app.services.paper_workflow import (
    PaperWorkflowError,
    prepare_uploaded_source_refresh,
)
from app.services.report_export import ReportExportError, build_released_report_export
from app.tasks.source_reanalysis import schedule_uploaded_source_reanalysis

router = APIRouter()


class AnnotationCreateRequest(BaseModel):
    annotation_type: str
    anchor_id: str | None = Field(default=None, min_length=64, max_length=64)
    anchor: dict | None = None
    content: str | None = Field(default=None, max_length=4000)
    user_label: str | None = Field(default=None, max_length=100)
    visibility: str = "private"

    @model_validator(mode="after")
    def require_one_anchor(self):
        if (self.anchor_id is None) == (self.anchor is None):
            raise ValueError("Provide exactly one annotation anchor")
        return self


class AnnotationRevisionRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    content: str | None = Field(default=None, max_length=4000)
    user_label: str | None = Field(default=None, max_length=100)
    visibility: str | None = None
    state: str = "active"


@router.get("/login", response_class=HTMLResponse)
async def report_login(next: str = "/"):
    """Show a minimal same-origin Personal login form containing no paper data."""
    ensure_report_auth_configured()
    destination = _safe_report_destination(next)
    if settings.REPORT_AUTH_MODE == "personal_local":
        return RedirectResponse(destination, status_code=303)
    nonce = secrets.token_urlsafe(24)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open SourceFidelity report</title><style nonce="{nonce}">
body{{max-width:32rem;margin:10vh auto;padding:1.25rem;font:16px/1.5 system-ui,sans-serif;color:#18212b}}
label,input,button{{display:block;width:100%}} input,button{{margin-top:.45rem;padding:.7rem;font:inherit}}
button{{margin-top:1rem}} .muted{{color:#5b6672}}
</style></head><body><h1>Open report</h1>
<p class="muted">Enter the Personal report access token configured by this deployment.</p>
<form method="post" action="/report/session" autocomplete="off">
<input type="hidden" name="next" value="{escape(destination, quote=True)}">
<label for="token">Access token</label>
<input id="token" name="token" type="password" required minlength="32" autocomplete="current-password">
<button type="submit">Open report</button></form></body></html>"""
    return HTMLResponse(body, headers=_html_security_headers(nonce))


@router.post("/session")
async def create_browser_session(
    request: Request,
    token: str = Form(..., min_length=32, max_length=4096),
    next: str = Form("/"),
):
    """Exchange the Personal credential for a bounded HttpOnly session."""
    require_same_origin_request(request)
    destination = _safe_report_destination(next)
    principal = authenticate_personal_bearer(token)
    session_token = create_report_session(principal)
    response = RedirectResponse(destination, status_code=303)
    response.set_cookie(
        REPORT_SESSION_COOKIE,
        session_token,
        max_age=settings.REPORT_SESSION_TTL_SECONDS,
        path="/report",
        secure=settings.REPORT_SESSION_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.headers.update(
        {
            "Cache-Control": "no-store, private",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        }
    )
    return response


@router.get("/session", include_in_schema=False)
async def recover_browser_session_navigation():
    """Do not let a refreshed login exchange URL masquerade as a report ID."""
    return RedirectResponse("/report/login", status_code=303)


@router.post("/logout")
async def delete_browser_session(request: Request):
    require_same_origin_request(request)
    response = Response(status_code=204)
    response.delete_cookie(
        REPORT_SESSION_COOKIE,
        path="/report",
        secure=settings.REPORT_SESSION_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store, private"
    return response


@router.get("/{report_id}", response_class=HTMLResponse)
def get_report(
    report_id: str,
    audience: str = Query("student", pattern="^(student|instructor)$"),
    principal: AuthenticatedPrincipal = Depends(get_report_browser_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return the evidence-led HTML projection in the resolved viewer scope."""
    principal.require(REPORT_VIEW_CAPABILITY)
    view, artifact, paper_content = _load_bundle(session, backend, report_id, principal)
    nonce = secrets.token_urlsafe(24)
    view = enable_authenticated_paper_actions(view, report_id=report_id)
    with fitz.open(stream=paper_content, filetype="pdf") as document:
        view["paper_surface"]["selectable_words"] = {
            page.number: page.get_text("words", sort=True) for page in document
        }
    view["audience"] = audience
    try:
        view["annotations"] = list_current_paper_annotations(
            session,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
            visibility=("released" if audience == "student" else None),
        )
    except PaperAnnotationError:
        # The authorized bundle loader rejects malformed real report IDs. This
        # fallback keeps isolated renderer tests with synthetic IDs annotation-free.
        view["annotations"] = []
    view["annotation_action"] = (
        {
            "create_href": f"/report/{report_id}/annotations",
            "revision_href_template": f"/report/{report_id}/annotations/{{annotation_id}}",
            "paper_artifact_id": str(artifact.id),
        }
        if audience == "instructor"
        else {}
    )
    content = render_evidence_report_html(view, csp_nonce=nonce)
    return HTMLResponse(
        content,
        headers=_html_security_headers(nonce),
    )


@router.get("/{report_id}/annotations")
def get_report_annotations(
    report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return the current exact-scope annotation overlay for one report."""
    principal.require(REPORT_VIEW_CAPABILITY)
    _load_bundle(session, backend, report_id, principal)
    return {
        "annotations": list_current_paper_annotations(
            session,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    }


@router.get("/{report_id}/export.pdf")
def get_released_report_pdf(
    report_id: str,
    inline: bool = Query(False),
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return a deterministic released-only derivative of the retained PDF."""
    principal.require(REPORT_PAPER_CAPABILITY)
    try:
        exported = build_released_report_export(
            session,
            backend,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except EvidenceReportAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except (EvidenceReportError, PaperAnnotationError, ReportExportError) as exc:
        raise HTTPException(status_code=409, detail="Report export is unavailable") from exc
    return Response(
        content=exported.content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": ('inline' if inline else 'attachment') + '; filename="sourcefidelity-released-report.pdf"',
            "Content-Security-Policy": "sandbox",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-SourceFidelity-Export-SHA256": exported.manifest["export_sha256"],
            "X-SourceFidelity-Manifest-SHA256": exported.manifest_sha256,
        },
    )


@router.get("/{report_id}/export/manifest")
def get_released_report_manifest(
    report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return the text-free binding manifest for the current released export."""
    principal.require(REPORT_PAPER_CAPABILITY)
    try:
        exported = build_released_report_export(
            session,
            backend,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except EvidenceReportAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except (EvidenceReportError, PaperAnnotationError, ReportExportError) as exc:
        raise HTTPException(status_code=409, detail="Report export is unavailable") from exc
    return JSONResponse(
        {**exported.manifest, "manifest_sha256": exported.manifest_sha256},
        headers={
            "Cache-Control": "no-store, private",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{report_id}/export/print", response_class=HTMLResponse)
def get_released_report_print_view(
    report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_browser_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Keep old print bookmarks pointed at the same released PDF as download."""
    principal.require(REPORT_PAPER_CAPABILITY)
    _load_bundle(session, backend, report_id, principal)
    return RedirectResponse(f"/report/{report_id}/export.pdf?inline=true", status_code=303)


@router.get("/{report_id}/successor")
def get_report_successor_status(
    report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return text-free lineage status for a report awaiting reanalysis."""
    principal.require(REPORT_VIEW_CAPABILITY)
    _load_bundle(session, backend, report_id, principal)
    try:
        requested_id = uuid.UUID(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    requested = session.get(Report, requested_id)
    if requested is None:
        raise HTTPException(status_code=404, detail="Report not found")
    latest = session.scalar(
        select(Report)
        .where(Report.job_id == requested.job_id)
        .order_by(Report.report_version.desc(), Report.created_at.desc())
        .limit(1)
    )
    job = session.get(Job, requested.job_id)
    if latest is None or job is None:
        raise HTTPException(status_code=409, detail="Report lineage is unavailable")
    failure = dict(
        (job.upload_evidence or {}).get("last_targeted_source_refresh_failure") or {}
    )
    return {
        "requested_report_id": str(requested.id),
        "latest_report_id": str(latest.id),
        "latest_report_version": latest.report_version,
        "successor_available": latest.id != requested.id,
        "processing": bool(
            (job.upload_evidence or {}).get("targeted_source_refresh")
        ),
        "last_reanalysis_failure": failure or None,
    }


@router.post("/{report_id}/annotations", status_code=201)
def create_report_annotation(
    report_id: str,
    payload: AnnotationCreateRequest,
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Create one authored comment/highlight revision on a stable citation anchor."""
    principal.require(REPORT_ANNOTATION_CAPABILITY)
    require_same_origin_request(request)
    view, artifact, paper_content = _load_bundle(session, backend, report_id, principal)
    page_dimensions = None
    if payload.anchor_id is not None:
        anchor = _authorized_anchor(view, payload.anchor_id)
    elif (payload.anchor or {}).get("anchor_kind") == "text_selection":
        from app.services.paper_annotations import text_selection_anchor
        try:
            anchor, page_dimensions = text_selection_anchor(paper_content, payload.anchor)
        except PaperAnnotationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    else:
        anchor, page_dimensions = _authorized_page_region_anchor(view, payload.anchor)
    try:
        annotation = create_paper_annotation(
            session,
            artifact=artifact,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
            author_provider=principal.provider,
            author_subject=principal.subject,
            annotation_type=payload.annotation_type,
            anchor=anchor,
            content=payload.content,
            user_label=payload.user_label,
            visibility=payload.visibility,
            page_dimensions=page_dimensions,
        )
    except PaperAnnotationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return annotation


@router.patch("/{report_id}/annotations/{annotation_id}")
def revise_report_annotation(
    report_id: str,
    annotation_id: str,
    payload: AnnotationRevisionRequest,
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Append an edit, visibility transition or deletion revision."""
    principal.require(REPORT_ANNOTATION_CAPABILITY)
    require_same_origin_request(request)
    _load_bundle(session, backend, report_id, principal)
    try:
        annotation = revise_paper_annotation(
            session,
            report_id=report_id,
            annotation_id=annotation_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
            author_provider=principal.provider,
            author_subject=principal.subject,
            expected_revision=payload.expected_revision,
            content=payload.content,
            user_label=payload.user_label,
            visibility=payload.visibility,
            state=payload.state,
        )
    except PaperAnnotationError as exc:
        message = str(exc)
        status_code = 404 if message == "Annotation not found" else 409
        raise HTTPException(status_code=status_code, detail=message) from exc
    return annotation


@router.get("/{report_id}/paper")
def get_report_paper(
    report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return the exact immutable PDF surface after a fresh authorization check."""
    principal.require(REPORT_PAPER_CAPABILITY)
    _, _, content = _load_bundle(session, backend, report_id, principal)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": 'inline; filename="sourcefidelity-paper.pdf"',
            "Content-Security-Policy": "sandbox",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{report_id}/source/{verification_report_id}")
def get_report_source(
    report_id: str,
    verification_report_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return one exact-scope authorized source representation."""
    principal.require(REPORT_SOURCE_CAPABILITY)
    view, _, _ = _load_bundle(session, backend, report_id, principal)
    allowed_ids = {
        str(member.get("verification_report_id"))
        for citation in view.get("citations", [])
        for member in citation.get("members", [])
        if member.get("verification_report_id")
    }
    if verification_report_id not in allowed_ids:
        raise HTTPException(status_code=404, detail="Source not found")
    try:
        record_id = uuid.UUID(verification_report_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Source not found") from None
    record = session.get(VerificationReportRecord, record_id)
    if (
        record is None
        or record.scope_type != principal.scope_type
        or record.scope_id != principal.scope_id
        or record.paper_version_id != view.get("paper_version_id")
    ):
        raise HTTPException(status_code=404, detail="Source not found")
    try:
        descriptor = SourceNavigationDescriptor.model_validate(
            (record.report_payload or {}).get("source_navigation") or {}
        )
        authorized = authorize_source_navigation_document(
            session,
            backend,
            descriptor,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except (ValueError, EvidenceAuthorizationError):
        raise HTTPException(status_code=404, detail="Source not found") from None
    source_media_type = authorized.representation.media_type
    response_media_type = (
        "application/pdf" if source_media_type == "application/pdf" else "text/plain"
    )
    return Response(
        content=authorized.representation.content,
        media_type=response_media_type,
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": (
                'inline; filename="sourcefidelity-source.pdf"'
                if response_media_type == "application/pdf"
                else 'inline; filename="sourcefidelity-source.txt"'
            ),
            "Content-Security-Policy": "sandbox",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{report_id}/source/{reference_id}/upload")
def upload_report_source(
    report_id: str,
    reference_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Admit one PDF using the exact reference selected in a report."""
    principal.require(REPORT_SOURCE_CAPABILITY)
    require_same_origin_request(request)
    if (
        principal.scope_type != "personal_owner"
        or principal.scope_id != settings.SOURCE_REPOSITORY_SCOPE_ID
    ):
        raise HTTPException(
            status_code=409,
            detail="Report-scoped source upload is not configured for this deployment scope",
        )
    view, _, _ = _load_bundle(session, backend, report_id, principal)
    matches = [
        member
        for citation in view.get("citations", [])
        for member in citation.get("members", [])
        if member.get("reference_id") == reference_id
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="Reference not found")
    source = matches[0].get("source") or {}
    identity = {
        (
            str(item.get("title") or ""),
            str(item.get("author") or ""),
            str(item.get("year") or ""),
            str(item.get("doi") or ""),
            str(item.get("source_kind") or "unknown"),
        )
        for item in (member.get("source") or {} for member in matches)
    }
    if len(identity) != 1:
        raise HTTPException(status_code=409, detail="Reference identity is inconsistent")
    result = admit_uploaded_source(
        request=request,
        file=file,
        doi=str(source.get("doi") or "") or None,
        isbn=None,
        title=str(source.get("title") or "") or None,
        author=str(source.get("author") or "") or None,
        year=str(source.get("year") or "") or None,
        expected_pages=None,
        expected_first_page=None,
        expected_last_page=None,
        document_kind=None,
        source_kind=(
            str(source.get("source_kind") or "")
            if str(source.get("source_kind") or "") not in {"", "unknown"}
            else None
        ),
        edition_or_version=None,
        description=f"Report-scoped correction for reference {reference_id}",
        db=session,
        backend=backend,
        principal=principal,
    )
    refresh = _start_uploaded_source_reanalysis(
        session,
        backend,
        report_id=report_id,
        reference_id=reference_id,
        admission=result,
    )
    return {
        "status": result.get("status", "ok"),
        "review_status": result.get("review_status"),
        **refresh,
    }


@router.post("/{report_id}/citation/{claim_id}/source/upload")
def upload_citation_source(
    report_id: str,
    claim_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Identify and admit one missing member source for an exact citation."""
    principal.require(REPORT_SOURCE_CAPABILITY)
    require_same_origin_request(request)
    if (
        principal.scope_type != "personal_owner"
        or principal.scope_id != settings.SOURCE_REPOSITORY_SCOPE_ID
    ):
        raise HTTPException(
            status_code=409,
            detail="Citation-scoped source upload is not configured for this deployment scope",
        )
    view, _, _ = _load_bundle(session, backend, report_id, principal)
    citation = next(
        (
            item
            for item in view.get("citations", [])
            if str(item.get("claim_id") or "") == claim_id
        ),
        None,
    )
    if citation is None:
        raise HTTPException(status_code=404, detail="Citation not found")
    candidates = [
        member
        for member in citation.get("members", [])
        if member.get("coverage_level") != "full_text"
    ]
    if not candidates:
        raise HTTPException(status_code=409, detail="This citation has no missing source")
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    file_bytes = file.file.read(maximum + 1)
    if len(file_bytes) > maximum:
        raise HTTPException(
            status_code=413,
            detail=f"Source exceeds the {settings.MAX_FILE_SIZE_MB} MB upload limit",
        )
    matches = []
    for member in candidates:
        source = member.get("source") or {}
        verified, _messages = verify_instructor_upload(
            file_bytes,
            provided_doi=str(source.get("doi") or "") or None,
            provided_title=str(source.get("title") or "") or None,
            provided_author=str(source.get("author") or "") or None,
        )
        if verified:
            matches.append(member)
    if len(matches) != 1:
        detail = (
            "The uploaded PDF does not match any missing source in this citation."
            if not matches
            else "The uploaded PDF matches more than one citation member; select a source with clearer identity evidence."
        )
        raise HTTPException(status_code=422, detail=detail)
    member = matches[0]
    source = member.get("source") or {}
    checked_file = UploadFile(
        file=BytesIO(file_bytes),
        filename=file.filename or "source.pdf",
        headers=file.headers,
    )
    result = admit_uploaded_source(
        request=request,
        file=checked_file,
        doi=str(source.get("doi") or "") or None,
        isbn=None,
        title=str(source.get("title") or "") or None,
        author=str(source.get("author") or "") or None,
        year=str(source.get("year") or "") or None,
        expected_pages=None,
        expected_first_page=None,
        expected_last_page=None,
        document_kind=None,
        source_kind=(
            str(source.get("source_kind") or "")
            if str(source.get("source_kind") or "") not in {"", "unknown"}
            else None
        ),
        edition_or_version=None,
        description=(
            "Citation-scoped correction for reference "
            f"{member.get('reference_id')}"
        ),
        db=session,
        backend=backend,
        principal=principal,
    )
    reference_id = str(member.get("reference_id") or "")
    refresh = _start_uploaded_source_reanalysis(
        session,
        backend,
        report_id=report_id,
        reference_id=reference_id,
        admission=result,
    )
    return {
        "status": result.get("status", "ok"),
        "review_status": result.get("review_status"),
        "reference_id": reference_id,
        **refresh,
    }


def _start_uploaded_source_reanalysis(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str,
    reference_id: str,
    admission: dict,
) -> dict:
    """Schedule a source-scoped successor report after durable admission."""
    if admission.get("review_status") != "accepted":
        return {
            "reanalysis_status": "awaiting_source_review",
            "message": (
                "Source uploaded, but it must be reviewed before it can be used "
                "to update this report."
            ),
        }
    documents = list(admission.get("documents") or [])
    accepted = [
        item for item in documents if item.get("admission_state") == "accepted"
    ]
    if len(accepted) != 1:
        raise HTTPException(
            status_code=409,
            detail="The upload did not produce one accepted source representation.",
        )
    try:
        prepared = prepare_uploaded_source_refresh(
            session,
            backend,
            report_id=report_id,
            reference_id=reference_id,
            representation_id=str(accepted[0].get("id") or ""),
        )
    except (PaperWorkflowError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not prepared.get("scheduled"):
        status = str(prepared.get("status") or "already_current")
        return {
            "reanalysis_status": status,
            "report_id": prepared.get("report_id"),
            "message": (
                "This source is already being checked for the report."
                if status == "already_scheduled"
                else "This source is already included in the current report."
            ),
        }
    publication_pending = False
    try:
        task_id = schedule_uploaded_source_reanalysis(str(prepared["job_id"]), prepared["attempt_id"])
    except Exception:
        # Publication may have succeeded despite a lost acknowledgement. Keep
        # the committed attempt for bounded recovery, never roll it back here.
        task_id = None
        publication_pending = True
    return {
        "reanalysis_status": "scheduled",
        "base_report_id": prepared.get("report_id"),
        "task_id": task_id,
        "publication_pending": publication_pending,
        "status_url": f"/report/{report_id}/successor",
        "message": (
            "Source stored safely. Reanalysis is waiting for the queue and will be retried automatically."
            if publication_pending else
            "Source uploaded and checked. The affected citation is being reanalyzed; "
            "other citations will not be rerun."
        ),
    }


@router.get("/{report_id}/paper/anchor/{anchor_id}", response_class=HTMLResponse)
def get_report_anchor_view(
    report_id: str,
    anchor_id: str,
    principal: AuthenticatedPrincipal = Depends(get_report_browser_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Show only pages warranted by one hash-bound citation anchor."""
    principal.require(REPORT_PAPER_CAPABILITY)
    view, _, content = _load_bundle(session, backend, report_id, principal)
    anchor = _authorized_anchor(view, anchor_id)
    nonce = secrets.token_urlsafe(24)
    document = fitz.open(stream=content, filetype="pdf")
    try:
        pages = []
        for page_index in anchor.get("page_indexes", []):
            if not 0 <= page_index < document.page_count:
                raise HTTPException(status_code=409, detail="Paper location is unavailable")
            page = document[page_index]
            rectangles = [
                item
                for item in anchor.get("rectangles", [])
                if item.get("page_index") == page_index
            ]
            _validate_overlay_rectangles(rectangles, page.rect)
            overlays = "".join(
                f'<rect x="{float(item["x0"]):.3f}" y="{float(item["y0"]):.3f}" '
                f'width="{float(item["x1"])-float(item["x0"]):.3f}" '
                f'height="{float(item["y1"])-float(item["y0"]):.3f}" />'
                for item in rectangles
            )
            pages.append(
                f'<section><h2>Page {page_index + 1}</h2><svg class="page" '
                f'viewBox="0 0 {page.rect.width:.3f} {page.rect.height:.3f}" '
                f'role="img" aria-label="Paper page {page_index + 1}">'
                f'<image href="/report/{report_id}/paper/anchor/{anchor_id}/page/{page_index}" '
                f'width="{page.rect.width:.3f}" height="{page.rect.height:.3f}" />'
                f'{overlays}</svg></section>'
            )
    finally:
        document.close()
    label = (
        "The exact citation span is highlighted."
        if anchor.get("localization_level") == "exact_rectangle"
        else "The page is known, but exact highlight geometry is unavailable."
    )
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>View in paper</title>
<style nonce="{nonce}">body{{max-width:70rem;margin:auto;padding:1rem;font:16px/1.5 system-ui,sans-serif;background:#eef1f4;color:#18212b}}
.page{{display:block;width:100%;height:auto;background:#fff;box-shadow:0 2px 12px #0003;margin-bottom:2rem}} svg rect{{fill:#2563a744;stroke:#15558f;stroke-width:1.5}}</style>
</head><body><h1>View in paper</h1><p>{escape(label)}</p>{''.join(pages)}</body></html>"""
    return HTMLResponse(body, headers=_html_security_headers(nonce))


@router.get("/{report_id}/paper/page/{page_index}")
def get_report_paper_page(
    report_id: str,
    page_index: int,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Render one authorized page for the continuously flowing report surface."""
    principal.require(REPORT_PAPER_CAPABILITY)
    _, _, content = _load_bundle(session, backend, report_id, principal)
    return _paper_page_response(content, page_index)


@router.get("/{report_id}/paper/anchor/{anchor_id}/page/{page_index}")
def get_report_anchor_page(
    report_id: str,
    anchor_id: str,
    page_index: int,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Render one authenticated immutable PDF page for the overlay viewer."""
    principal.require(REPORT_PAPER_CAPABILITY)
    view, _, content = _load_bundle(session, backend, report_id, principal)
    anchor = _authorized_anchor(view, anchor_id)
    if page_index not in anchor.get("page_indexes", []):
        raise HTTPException(status_code=404, detail="Paper location not found")
    return _paper_page_response(content, page_index)


def _paper_page_response(content: bytes, page_index: int) -> Response:
    document = fitz.open(stream=content, filetype="pdf")
    try:
        if not 0 <= page_index < document.page_count:
            raise HTTPException(status_code=404, detail="Paper location not found")
        page = document[page_index]
        if page.rect.width * 2 * page.rect.height * 2 > 25_000_000:
            raise HTTPException(status_code=409, detail="Paper page is too large to render")
        image = page.get_pixmap(
            matrix=fitz.Matrix(2, 2), alpha=False
        ).tobytes("png")
    finally:
        document.close()
    return Response(
        content=image,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, private",
            "Content-Security-Policy": "sandbox",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _load_bundle(
    session: Session,
    backend: StorageBackend,
    report_id: str,
    principal: AuthenticatedPrincipal,
):
    try:
        return load_authorized_evidence_report_bundle(
            session,
            backend,
            report_id=report_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except EvidenceReportAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except EvidenceReportError as exc:
        raise HTTPException(status_code=409, detail="Report is unavailable") from exc


def _authorized_anchor(view: dict, anchor_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", anchor_id):
        raise HTTPException(status_code=404, detail="Paper location not found")
    matches = [
        item
        for item in (view.get("paper_surface") or {}).get("citation_anchors", [])
        if item.get("anchor_id") == anchor_id
        and item.get("localization_level") in {"exact_rectangle", "page_only"}
    ]
    if len(matches) != 1 or not matches[0].get("page_indexes"):
        raise HTTPException(status_code=404, detail="Paper location not found")
    return matches[0]


def _authorized_page_region_anchor(
    view: dict,
    supplied: dict | None,
) -> tuple[dict, dict[int, tuple[float, float]]]:
    if not isinstance(supplied, dict):
        raise HTTPException(status_code=409, detail="Paper region is unavailable")
    if (
        supplied.get("anchor_version") != "page-region-anchor-v1"
        or supplied.get("anchor_kind") not in {None, "page_region"}
        or supplied.get("localization_level") != "exact_rectangle"
    ):
        raise HTTPException(status_code=409, detail="Paper region is invalid")
    dimensions: dict[int, tuple[float, float]] = {}
    for item in (view.get("paper_surface") or {}).get("page_dimensions") or []:
        try:
            page_index = int(item["page_index"])
            width = float(item["width"])
            height = float(item["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if page_index >= 0 and width > 0 and height > 0:
            dimensions[page_index] = (width, height)
    rectangles = supplied.get("rectangles") or []
    if len(rectangles) != 1:
        raise HTTPException(
            status_code=409,
            detail="Select one paper region for each annotation",
        )
    try:
        page_index = int(rectangles[0]["page_index"])
        x0, y0, x1, y1 = (
            float(rectangles[0][key]) for key in ("x0", "y0", "x1", "y1")
        )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=409, detail="Paper region is invalid") from None
    if page_index not in dimensions:
        raise HTTPException(status_code=409, detail="Paper page is unavailable")
    width, height = dimensions[page_index]
    if not (
        0 <= x0 < x1 <= width
        and 0 <= y0 < y1 <= height
        and x1 - x0 >= 2
        and y1 - y0 >= 2
    ):
        raise HTTPException(status_code=409, detail="Paper region is invalid")
    anchor = {
        "anchor_version": "page-region-anchor-v1",
        "anchor_kind": "page_region",
        "localization_level": "exact_rectangle",
        "page_indexes": [page_index],
        "rectangles": [
            {
                "page_index": page_index,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        ],
    }
    return anchor, dimensions


def _validate_overlay_rectangles(rectangles: list[dict], page_rect) -> None:
    for item in rectangles:
        try:
            x0, y0, x1, y1 = (
                float(item["x0"]),
                float(item["y0"]),
                float(item["x1"]),
                float(item["y1"]),
            )
        except (KeyError, TypeError, ValueError):
            raise HTTPException(
                status_code=409, detail="Paper location is unavailable"
            ) from None
        if not (
            0 <= x0 < x1 <= page_rect.width
            and 0 <= y0 < y1 <= page_rect.height
        ):
            raise HTTPException(status_code=409, detail="Paper location is unavailable")


def _safe_report_destination(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme
        or parsed.netloc
        or not parsed.path.startswith("/report/")
        or parsed.path.startswith("//")
    ):
        return "/"
    return parsed.path


def _html_security_headers(nonce: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-store, private",
        "Content-Security-Policy": (
            "default-src 'none'; "
            f"style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'self'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
    }
