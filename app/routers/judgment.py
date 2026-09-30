"""Experimental Judgment layout endpoints (ARCHITECTURE §7).

Papers are judged when they are checked (owner decision 2026-09-27); there is
no notice (owner decision 2026-09-28). `start` begins a run for a report
checked before judging at check time existed. Everything is scoped to the viewer's authorization scope,
state changes require same-origin requests, and responses are never cached.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.security import (
    REPORT_JUDGMENT_CAPABILITY,
    REPORT_VIEW_CAPABILITY,
    AuthenticatedPrincipal,
    get_report_principal,
    require_same_origin_request,
)
from app.services.evidence_report import (
    EvidenceReportAuthorizationError,
    EvidenceReportError,
    get_authorized_evidence_report_view,
)
from app.services import judgment_runs as runs

router = APIRouter()
_NO_STORE = {"Cache-Control": "no-store, private"}


def _view(session: Session, report_id: str, principal: AuthenticatedPrincipal) -> tuple[uuid.UUID, dict]:
    try:
        parsed = uuid.UUID(report_id)
        return parsed, get_authorized_evidence_report_view(
            session, parsed, scope_type=principal.scope_type, scope_id=principal.scope_id)
    except (ValueError, EvidenceReportAuthorizationError):
        raise HTTPException(status_code=404, detail="Report not found") from None
    except EvidenceReportError:
        raise HTTPException(status_code=409, detail="Report is unavailable") from None


def _run_json(run) -> dict | None:
    if run is None:
        return None
    return {"run_id": str(run.id), "status": run.status, "reason_code": run.reason_code,
            "candidates_done": run.candidate_done, "candidates_total": run.candidate_total,
            "spend_usd": round(run.spend_usd, 6), "unpriced_calls": run.unpriced_calls}


def _status(session, report_uuid, view, principal) -> dict:
    return {
        "eligible_sources": len(runs.eligible_members(view)),
        "run": _run_json(runs.latest_run(session, report_uuid, principal)),
    }


def _start(session, report_uuid, principal, *, force_new: bool):
    from app.tasks.judgment import dispatch_report_judgment
    run = runs.create_run(session, report_uuid, principal, force_new=force_new)
    created = run.created_now
    session.commit()
    if created:
        dispatch_report_judgment(run.id)
    return run


@router.get("/{report_id}/judgment/status")
def judgment_status(report_id: str, principal: AuthenticatedPrincipal = Depends(get_report_principal),
                    session: Session = Depends(get_db)):
    principal.require(REPORT_VIEW_CAPABILITY)
    report_uuid, view = _view(session, report_id, principal)
    return JSONResponse(_status(session, report_uuid, view, principal), headers=_NO_STORE)


@router.post("/{report_id}/judgment/start")
def judgment_start(report_id: str, request: Request,
                   principal: AuthenticatedPrincipal = Depends(get_report_principal),
                   session: Session = Depends(get_db)):
    require_same_origin_request(request)
    principal.require(REPORT_JUDGMENT_CAPABILITY)
    report_uuid, view = _view(session, report_id, principal)
    _start(session, report_uuid, principal, force_new=False)
    return JSONResponse(_status(session, report_uuid, view, principal), headers=_NO_STORE)


@router.post("/{report_id}/judgment/retry")
def judgment_retry(report_id: str, request: Request,
                   principal: AuthenticatedPrincipal = Depends(get_report_principal),
                   session: Session = Depends(get_db)):
    """A new run: cached valid results are reused, so only failed arms are called again."""
    require_same_origin_request(request)
    principal.require(REPORT_JUDGMENT_CAPABILITY)
    report_uuid, view = _view(session, report_id, principal)
    current = runs.latest_run(session, report_uuid, principal)
    if current is not None and current.status in runs.ACTIVE:
        return JSONResponse(_status(session, report_uuid, view, principal), headers=_NO_STORE)
    _start(session, report_uuid, principal, force_new=True)
    return JSONResponse(_status(session, report_uuid, view, principal), headers=_NO_STORE)


@router.get("/{report_id}/judgment/results")
def judgment_results(report_id: str, after: int = Query(0, ge=0),
                     principal: AuthenticatedPrincipal = Depends(get_report_principal),
                     session: Session = Depends(get_db)):
    principal.require(REPORT_VIEW_CAPABILITY)
    report_uuid, _ = _view(session, report_id, principal)
    run = runs.latest_run(session, report_uuid, principal)
    from app.services.judgment_results import result_items
    items = result_items(session, run, principal, after=after)
    return JSONResponse({"run": _run_json(run), "items": items}, headers=_NO_STORE)
