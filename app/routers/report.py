"""Authenticated delivery of an immutable evidence report and paper surface."""

from __future__ import annotations

import secrets
import hashlib
from urllib.parse import urlsplit
import re
import uuid
from io import BytesIO
from html import escape

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from sqlalchemy import select
from sqlalchemy.orm import Session
import fitz

from app.database import get_db
from app.security import (
    AuthenticatedPrincipal,
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
    member_search_incomplete,
    render_evidence_report_html,
    project_reference_flags,
)
from app.services.storage.backend import StorageBackend, get_storage_backend
from app.services.source_navigation import (
    SourceNavigationDescriptor,
    authorize_source_navigation_document,
)
from app.services.verification_evidence import EvidenceAuthorizationError
from app.routers.sources import upload_source as admit_uploaded_source
from app.services.pdf_verifier import verify_instructor_upload
from app.services.paper_workflow import (
    PaperWorkflowError,
    prepare_uploaded_source_refresh,
)
from app.services.report_export import ReportExportError, build_released_report_export
from app.tasks.source_reanalysis import schedule_uploaded_source_reanalysis
from app.services.search_retry import SearchAgainRateLimited, prepare_search_again_refresh
from app.tasks.check_paper import dispatch_paper_workflow

router = APIRouter()


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
    audience: str | None = Query(None, include_in_schema=False),
    principal: AuthenticatedPrincipal = Depends(get_report_browser_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Return the evidence-led HTML projection in the resolved viewer scope."""
    principal.require(REPORT_VIEW_CAPABILITY)
    view, artifact, paper_content = _load_bundle(session, backend, report_id, principal)
    nonce = secrets.token_urlsafe(24)
    view = enable_authenticated_paper_actions(
        view, report_id=report_id,
        search_again_enabled=not _report_marks_released(session, report_id, principal))
    with fitz.open(stream=paper_content, filetype="pdf") as document:
        view = project_reference_flags(view, document, hashlib.sha256(paper_content).hexdigest())
        view["paper_surface"]["selectable_words"] = {
            page.number: page.get_text("words", sort=True) for page in document
        }
        from app.services.report_member_navigation import marker_words
        view['paper_surface']['marker_words'] = marker_words(document)
    # Judgment is part of every report (owner decision 2026-09-28): claim
    # underline geometry here; results are polled by the page.
    from app.services.judgment_layer import build_judgment_layer
    layer = build_judgment_layer(session, view, view["paper_surface"]["selectable_words"],
                                 scope_type=principal.scope_type, scope_id=principal.scope_id)
    layer.update(fake_panel=settings.JUDGMENT_FAKE_PANEL)
    view["judgment_layer"] = layer
    view.update(_run_details(session, report_id, principal))
    # One report for everyone: an old ?audience= link is accepted and ignored.
    content = render_evidence_report_html(view, csp_nonce=nonce)
    return HTMLResponse(
        content,
        headers=_html_security_headers(nonce),
    )


def _report_marks_released(session: Session, report_id: str, principal) -> bool:
    """Whether the report's paper belongs to an assessment with marks released."""
    from app.services.assessment_marks import job_marks_released
    if not isinstance(session, Session):
        return False  # only a database session can hold the record
    try:
        report = session.get(Report, uuid.UUID(str(report_id)))
    except ValueError:
        return False
    if not isinstance(report, Report):
        return False
    job = session.get(Job, report.job_id)
    if (not isinstance(job, Job) or job.scope_type != principal.scope_type
            or job.scope_id != principal.scope_id):
        return False
    return job_marks_released(session, job)


def _static_judgment_layer(session: Session, view: dict, paper: bytes, report_id: str, principal) -> dict | None:
    from app.services import judgment_runs as runs
    from app.services.judgment_layer import build_judgment_layer
    from app.services.judgment_results import all_result_items
    try:
        report_uuid = uuid.UUID(str(report_id))
    except ValueError:
        return None
    with fitz.open(stream=paper, filetype="pdf") as document:
        words = {page.number: page.get_text("words", sort=True) for page in document}
    layer = build_judgment_layer(session, view, words, scope_type=principal.scope_type, scope_id=principal.scope_id)
    run = runs.latest_run(session, report_uuid, principal)
    layer.update(fake_panel=settings.JUDGMENT_FAKE_PANEL, static=True,
                 static_results=all_result_items(session, run, principal))
    return layer


def _export_judgment_summary(session: Session, report_id: str, principal) -> dict | None:
    """The page's Judgment tally for the PDF's Sources to Upload, or None."""
    try:
        details = _run_details(session, report_id, principal)
    except Exception:  # noqa: BLE001 - the PDF is still built without the count
        return None
    summary = details.get("judgment_summary") if isinstance(details, dict) else None
    return summary if isinstance(summary, dict) else None


def _run_details(session: Session, report_id: str, principal) -> dict:
    """Judged citations and single-run technical details (owner requests 2026-09-28)."""
    from sqlalchemy import select
    from app.models.job import Job
    from app.models.judgment import JudgmentArmResult, JudgmentCandidateResult
    from app.models.report import Report
    from app.services import judgment_runs as runs
    from app.services.report_run_metrics import judged_citations, judgment_states, single_run_metrics
    try:
        report = session.get(Report, uuid.UUID(str(report_id)))
    except ValueError:
        return {}
    if not isinstance(report, Report):
        return {}
    job = session.get(Job, report.job_id)
    if not isinstance(job, Job):
        job = None
    run = runs.latest_run(session, report.id, principal)
    results = list(session.scalars(select(JudgmentCandidateResult).where(
        JudgmentCandidateResult.run_id == run.id))) if run is not None else []
    arm_ids = {uuid.UUID(str(i)) for r in results for i in (r.arm_result_ids or [])}
    arms = list(session.scalars(select(JudgmentArmResult).where(JudgmentArmResult.id.in_(arm_ids)))) if arm_ids else []
    details = {"judgment_summary": {"judged_citations": judged_citations(results),
                                    "states": judgment_states(results)}}
    if job is not None:
        details["processing_metrics"] = single_run_metrics(job, results=results, arms=arms, settings=settings)
    return details


@router.get("/{report_id}/export.html")
def get_interactive_report_download(
    report_id: str, audience: str | None = Query(None, include_in_schema=False),
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db), backend: StorageBackend = Depends(get_storage_backend),
):
    principal.require(REPORT_PAPER_CAPABILITY)
    from app.services.interactive_report_export import build_interactive_report_html
    view, artifact, paper = _load_bundle(session, backend, report_id, principal)
    view = {**view, 'paper_surface': {**view.get('paper_surface', {}),
                                     'presentation_sha256': artifact.presentation_sha256}}
    report_record=session.get(Report,uuid.UUID(report_id))
    if report_record is not None:view['report_version']=report_record.report_version
    view.update(_run_details(session, report_id, principal))
    # The export is the same report as the connected one (owner request
    # 2026-09-29): Judgment's underlines, windows and summary are carried in
    # the file as finished results, since it cannot contact the server.
    view['judgment_layer'] = _static_judgment_layer(session, view, paper, report_id, principal)
    try:
        content = build_interactive_report_html(view, paper)
    except (ValueError, ReportExportError) as exc:
        raise HTTPException(status_code=409, detail='Interactive export is unavailable') from exc
    return Response(content, media_type='text/html', headers={
        'Content-Disposition': 'attachment; filename="interactive-report.html"',
        'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff'})


@router.get("/{report_id}/export.pdf")
def get_released_report_pdf(
    report_id: str,
    audience: str | None = Query(None, include_in_schema=False),
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
            judgment_summary=_export_judgment_summary(session, report_id, principal),
        )
    except EvidenceReportAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except (EvidenceReportError, ReportExportError) as exc:
        raise HTTPException(status_code=409, detail="Report export is unavailable") from exc
    return Response(
        content=exported.content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": 'attachment; filename="sourcefidelity-report.pdf"',
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
    audience: str | None = Query(None, include_in_schema=False),
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
            judgment_summary=_export_judgment_summary(session, report_id, principal),
        )
    except EvidenceReportAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except (EvidenceReportError, ReportExportError) as exc:
        raise HTTPException(status_code=409, detail="Report export is unavailable") from exc
    return JSONResponse(
        {**exported.manifest, "manifest_sha256": exported.manifest_sha256},
        headers={
            "Cache-Control": "no-store, private",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


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


def _stated_page_range(source: dict) -> tuple[int | None, int | None]:
    """The page range the reference itself states ("(pp. 64-86)", "53(3),
    152-177"), so an uploaded chapter or article is checked against it
    (2026-10-02: report uploads passed no range, so a complete chapter stayed
    "uncertain")."""
    import re
    raw = str(source.get("raw_reference") or "")
    match = (re.search(r"\bpp?\.\s*(\d{1,5})\s*[-–—]\s*(\d{1,5})", raw)
             or re.search(r"\)\s*,\s*(\d{1,5})\s*[-–—]\s*(\d{1,5})\b", raw))
    if match:
        first, last = int(match.group(1)), int(match.group(2))
        if 0 < first < last <= first + 2000:
            return first, last
    return None, None


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
        expected_first_page=_stated_page_range(source)[0],
        expected_last_page=_stated_page_range(source)[1],
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


@router.post("/{report_id}/source/upload")
def upload_unassigned_report_source(
    report_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Resolve an uploaded PDF uniquely inside this authorized report."""
    return upload_citation_source(report_id, None, request, file, principal, session, backend)


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
    if claim_id is not None and citation is None:
        raise HTTPException(status_code=404, detail="Citation not found")
    from app.services.evidence_report import _member_accepts_upload
    selected = [citation] if citation is not None else view.get("citations", [])
    candidates = [
        member
        for item in selected
        for member in item.get("members", [])
        if member.get("coverage_level") != "full_text"
        and _member_accepts_upload(member)
    ]
    unique = {}
    for member in candidates:
        reference_id = member.get('reference_id')
        if not reference_id:
            continue
        if reference_id in unique and unique[reference_id].get('source') != member.get('source'):
            raise HTTPException(status_code=409, detail="Reference identity is inconsistent")
        unique[reference_id] = member
    candidates = list(unique.values())
    if not candidates:
        raise HTTPException(status_code=409, detail="This citation has no missing source")
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    file_bytes = file.file.read(maximum + 1)
    if len(file_bytes) > maximum:
        raise HTTPException(
            status_code=413,
            detail=f"Source exceeds the {settings.MAX_FILE_SIZE_MB} MB upload limit",
        )
    from app.services.supplied_html_source import (
        SuppliedHtmlRejected,
        looks_like_html_document,
        qualify_supplied_html,
    )
    from app.services.source_type import SourceKindAssessment

    supplied_page = looks_like_html_document(file_bytes, file.content_type)
    if supplied_page and not settings.SUPPLIED_HTML_SOURCE_INTAKE_ENABLED:
        raise HTTPException(
            status_code=415,
            detail="Supplied HTML source intake is disabled; upload a PDF",
        )
    matches = []
    for member in candidates:
        source = member.get("source") or {}
        if supplied_page:
            # A supplied page is matched by the same qualification that will
            # admit it, so a page that cannot be admitted never selects a
            # member. No weaker matching rule exists for supplied files.
            kind = str(source.get("source_kind") or "")
            try:
                qualify_supplied_html(
                    file_bytes,
                    expected_title=str(source.get("title") or "") or None,
                    expected_author=str(source.get("author") or "") or None,
                    expected_year=str(source.get("year") or "") or None,
                    expected_doi=str(source.get("doi") or "") or None,
                    expected_source_kind=(
                        SourceKindAssessment(kind=kind, confidence="high")
                        if kind and kind != "unknown"
                        else SourceKindAssessment()
                    ),
                )
            except SuppliedHtmlRejected:
                continue
            matches.append(member)
            continue
        verified, _messages = verify_instructor_upload(
            file_bytes,
            provided_doi=str(source.get("doi") or "") or None,
            provided_title=str(source.get("title") or "") or None,
            provided_author=str(source.get("author") or "") or None,
            web_page=str(source.get("source_kind") or "") == "webpage",
        )
        if verified:
            matches.append(member)
    if len(matches) != 1:
        label = "page" if supplied_page else "PDF"
        detail = (
            f"The uploaded {label} does not match an eligible reference in this report."
            if not matches
            else f"The uploaded {label} cannot be uniquely matched to one reference in this report."
        )
        raise HTTPException(status_code=422, detail=detail)
    member = matches[0]
    source = member.get("source") or {}
    checked_file = UploadFile(
        file=BytesIO(file_bytes),
        filename=file.filename or ("source.html" if supplied_page else "source.pdf"),
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
        expected_first_page=_stated_page_range(source)[0],
        expected_last_page=_stated_page_range(source)[1],
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


@router.post("/{report_id}/reference/{reference_id}/search-again")
def search_reference_again(
    report_id: str,
    reference_id: str,
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_report_principal),
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Search one reference with an incomplete search again (owner decision 2026-09-29).

    Starts a forced targeted refresh of that reference only; the page then
    follows the successor report exactly as after a source upload.
    """
    principal.require(REPORT_SOURCE_CAPABILITY)
    require_same_origin_request(request)
    view, _, _ = _load_bundle(session, backend, report_id, principal)
    members = [
        member
        for citation in view.get("citations", [])
        for member in citation.get("members", [])
        if member.get("reference_id") == reference_id
    ]
    if not members:
        raise HTTPException(status_code=404, detail="Reference not found")
    if not any(member_search_incomplete(member) for member in members):
        raise HTTPException(status_code=409, detail="This reference has no incomplete search")
    try:
        prepared = prepare_search_again_refresh(
            session,
            report_id=report_id,
            reference_id=reference_id,
            scope_type=principal.scope_type,
            scope_id=principal.scope_id,
        )
    except SearchAgainRateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except PaperWorkflowError as exc:
        if exc.code == "assessment_marks_released":
            # Marks were released after the page was opened: the page removes
            # the button (owner decision 2026-09-29).
            raise HTTPException(status_code=410, detail=exc.code) from exc
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    publication_pending = False
    try:
        task_id = dispatch_paper_workflow(prepared["job_id"], prepared["attempt_id"])
    except Exception:
        # The committed attempt is resumed by bounded workflow recovery.
        task_id = None
        publication_pending = True
    return {
        "reanalysis_status": "scheduled",
        "base_report_id": prepared.get("report_id"),
        "attempt_id": prepared["attempt_id"],
        "task_id": task_id,
        "publication_pending": publication_pending,
        "status_url": f"/report/{report_id}/successor",
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
    except PaperWorkflowError as exc:
        if exc.code != "report_reanalysis_busy":
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # The source is stored and accepted; another upload's refresh is running.
        # Queue this refresh to start when that one ends instead of refusing it
        # (2026-10-01: Langford was stored but reported as refused).
        from app.tasks.source_reanalysis import schedule_busy_upload_refresh
        task_id = schedule_busy_upload_refresh(report_id, reference_id, str(accepted[0].get("id") or ""))
        return {
            "reanalysis_status": "scheduled",
            "base_report_id": report_id,
            "task_id": task_id,
            "publication_pending": False,
            "status_url": f"/report/{report_id}/successor",
            "message": (
                "Source uploaded and checked. The affected citation is being reanalyzed; "
                "other citations will not be rerun."
            ),
        }
    except ValueError as exc:
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
