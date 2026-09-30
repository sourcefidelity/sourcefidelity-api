"""Authenticated Personal review surfaces, never general source navigation."""
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database import get_db
from app.models.edition_review import EditionReviewDecision
from app.security import get_report_browser_principal, get_report_principal, require_same_origin_request
from app.services.storage.backend import get_storage_backend
from app.services.file_safety import FileSafetyUnavailable
from app.services.edition_review_entry import (
    EditionReviewError, ReviewDecisionInput, payload_hash, prepare_review,
    require_personal_review, resolve_review, save_review,
)
from app.services.edition_review_html import INTERFACE_JS, new_review_html, review_html

router = APIRouter(prefix="/edition-reviews", tags=["Personal edition review"])
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; img-src 'self'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"}


def error(exc):
    if isinstance(exc, FileSafetyUnavailable):
        return HTTPException(status_code=503, detail="Source safety inspection is unavailable", headers=HEADERS)
    return HTTPException(status_code=409, detail="Review unavailable, changed, already saved, or input incomplete", headers=HEADERS)


def saved_decision(session, snapshot):
    decision = session.scalar(select(EditionReviewDecision).where(EditionReviewDecision.snapshot_id == snapshot.id))
    if decision is not None and (decision.payload_sha256 != payload_hash(decision.payload)
            or decision.payload.get("snapshot_sha256") != snapshot.snapshot_sha256):
        raise EditionReviewError("Review decision changed")
    return decision


@router.get("/interface.js")
def interface_script():
    return Response(INTERFACE_JS, media_type="text/javascript", headers=HEADERS)


@router.get("/new", response_class=HTMLResponse)
def new_review(representation_id: str = Query(default="", max_length=36), principal=Depends(get_report_browser_principal)):
    require_personal_review(principal)
    return HTMLResponse(new_review_html(representation_id), headers=HEADERS)


@router.post("")
def create_review(request: Request, representation_id: str = Form(..., max_length=36),
                  reference_text: str = Form(..., max_length=4000),
                  principal=Depends(get_report_principal), session=Depends(get_db), backend=Depends(get_storage_backend)):
    require_same_origin_request(request)
    try:
        snapshot = prepare_review(session, backend, principal, representation_id, reference_text)
        session.commit()
        return RedirectResponse(f"/edition-reviews/{snapshot.id}", status_code=303, headers=HEADERS)
    except (EditionReviewError, FileSafetyUnavailable, ValidationError, IntegrityError, FileNotFoundError) as exc:
        session.rollback()
        raise error(exc) from None


@router.get("/{review_id}", response_class=HTMLResponse)
def show_review(review_id: str, principal=Depends(get_report_browser_principal), session=Depends(get_db), backend=Depends(get_storage_backend)):
    try:
        snapshot, _, _ = resolve_review(session, backend, principal, review_id)
        return HTMLResponse(review_html(snapshot, saved_decision(session, snapshot)), headers=HEADERS)
    except (EditionReviewError, FileSafetyUnavailable, FileNotFoundError) as exc:
        raise error(exc) from None


@router.get("/{review_id}/pages/{page_number}")
def show_page(review_id: str, page_number: int, principal=Depends(get_report_principal), session=Depends(get_db), backend=Depends(get_storage_backend)):
    try:
        _, _, pages = resolve_review(session, backend, principal, review_id, page_number=page_number)
        return Response(pages.images[0], media_type="image/png", headers=HEADERS)
    except (EditionReviewError, FileSafetyUnavailable, FileNotFoundError) as exc:
        raise error(exc) from None


@router.post("/{review_id}/decision")
def submit_review(request: Request, review_id: str, snapshot_sha256: str = Form(..., max_length=64),
                  decision: str = Form(..., max_length=32), acknowledged: bool = Form(False),
                  notes: str = Form(..., max_length=4000), work_page: int | None = Form(None),
                  edition_page: int | None = Form(None), principal=Depends(get_report_principal),
                  session=Depends(get_db), backend=Depends(get_storage_backend)):
    require_same_origin_request(request)
    try:
        submission = ReviewDecisionInput(snapshot_sha256=snapshot_sha256, decision=decision,
            acknowledged=acknowledged, notes=notes, work_page=work_page, edition_page=edition_page)
        save_review(session, backend, principal, review_id, submission)
        session.commit()
        return RedirectResponse(f"/edition-reviews/{review_id}", status_code=303, headers=HEADERS)
    except (EditionReviewError, FileSafetyUnavailable, ValidationError, IntegrityError, FileNotFoundError) as exc:
        session.rollback()
        raise error(exc) from None


@router.get("/{review_id}/export")
def export_review(review_id: str, principal=Depends(get_report_principal), session=Depends(get_db), backend=Depends(get_storage_backend)):
    try:
        snapshot, _, _ = resolve_review(session, backend, principal, review_id)
        decision = saved_decision(session, snapshot)
        payload = {"status": "complete" if decision else "pending", "snapshot_id": str(snapshot.id),
            "snapshot_sha256": snapshot.snapshot_sha256, "snapshot": snapshot.payload,
            "decision_id": str(decision.id) if decision else None,
            "decision_sha256": decision.payload_sha256 if decision else None,
            "reviewer_provider": decision.reviewer_provider if decision else None,
            "decision": decision.payload if decision else None}
        return JSONResponse(payload, headers={**HEADERS, "Content-Disposition": 'attachment; filename="edition-review.json"'})
    except (EditionReviewError, FileSafetyUnavailable, FileNotFoundError) as exc:
        raise error(exc) from None


@router.post("/{review_id}/answer")
def submit_answer(request: Request, review_id: str,
                  snapshot_sha256: str = Form(..., max_length=64), answer: str = Form(..., max_length=3),
                  principal=Depends(get_report_principal), session=Depends(get_db), backend=Depends(get_storage_backend)):
    from app.services.personal_edition_answer import PersonalEditionAnswer, save_personal_answer
    require_same_origin_request(request)
    try:
        save_personal_answer(session, backend, principal, review_id,
            PersonalEditionAnswer(snapshot_sha256=snapshot_sha256, answer=answer))
        session.commit()
        return RedirectResponse(f"/edition-reviews/{review_id}", status_code=303, headers=HEADERS)
    except (EditionReviewError, FileSafetyUnavailable, ValidationError, IntegrityError, FileNotFoundError) as exc:
        session.rollback()
        raise error(exc) from None
