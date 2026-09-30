"""Assessment-level "marks released" setting (owner decision 2026-09-29).

An instructor or administrator records that marks are released for an
assessment in the caller's authorization scope; the planned LMS grade-release
signal calls ``set_marks_released(..., source="lms")`` directly. JSON only.
"""
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import (
    ASSESSMENT_MARKS_RELEASE_CAPABILITY,
    AuthenticatedPrincipal,
    get_authenticated_principal,
    require_same_origin_request,
)
from app.services.assessment_marks import (
    AssessmentMarksError,
    clear_marks_released,
    marks_release_record,
    set_marks_released,
)

router = APIRouter(prefix="/assessments", tags=["assessments"])
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_ASSESSMENT_ID = Path(..., min_length=1, max_length=255)


def _payload(assessment_id: str, record, **extra) -> dict:
    released = record is not None and record.marks_released_at is not None
    return {
        "assessment_id": assessment_id,
        "marks_released": released,
        "marks_released_at": record.marks_released_at.isoformat() if released else None,
        "released_by": record.released_by if released else None,
        "release_source": record.release_source if released else None,
        **extra,
    }


def _invalid(exc: AssessmentMarksError) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)},
                         headers=_HEADERS)


@router.get("/{assessment_id}/marks-released")
def get_marks_released(
    assessment_id: str = _ASSESSMENT_ID,
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
    session: Session = Depends(get_db),
):
    principal.require(ASSESSMENT_MARKS_RELEASE_CAPABILITY)
    try:
        record = marks_release_record(session, scope_type=principal.scope_type,
                                      scope_id=principal.scope_id, assessment_id=assessment_id)
    except AssessmentMarksError as exc:
        raise _invalid(exc) from exc
    return _payload(assessment_id.strip(), record)


@router.post("/{assessment_id}/marks-released")
def release_marks(
    request: Request,
    assessment_id: str = _ASSESSMENT_ID,
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
    session: Session = Depends(get_db),
):
    principal.require(ASSESSMENT_MARKS_RELEASE_CAPABILITY)
    require_same_origin_request(request)
    try:
        record, purged = set_marks_released(
            session, scope_type=principal.scope_type, scope_id=principal.scope_id,
            assessment_id=assessment_id, released_by=principal.subject, source="manual")
    except AssessmentMarksError as exc:
        session.rollback()
        raise _invalid(exc) from exc
    return _payload(record.assessment_id, record, judgment_reserves_purged=purged)


@router.delete("/{assessment_id}/marks-released")
def clear_released_marks(
    request: Request,
    assessment_id: str = _ASSESSMENT_ID,
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
    session: Session = Depends(get_db),
):
    principal.require(ASSESSMENT_MARKS_RELEASE_CAPABILITY)
    require_same_origin_request(request)
    try:
        record = clear_marks_released(
            session, scope_type=principal.scope_type, scope_id=principal.scope_id,
            assessment_id=assessment_id, cleared_by=principal.subject, source="manual")
    except AssessmentMarksError as exc:
        session.rollback()
        raise _invalid(exc) from exc
    return _payload(assessment_id.strip(), record)
