"""Read the bounded operational status of a personal paper job."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.job import Job


router = APIRouter()


@router.get("/{job_id}")
async def get_job_status(job_id: str, session: Session = Depends(get_db)):
    try:
        parsed = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Paper job not found") from exc
    job = session.get(Job, parsed)
    if (
        job is None
        or job.scope_type != "personal_owner"
        or job.scope_id != settings.SOURCE_REPOSITORY_SCOPE_ID
    ):
        raise HTTPException(status_code=404, detail="Paper job not found")
    summary = job.verification_summary or {}
    return {
        "job_id": str(job.id),
        "paper_version_id": job.paper_version_id,
        "status": job.status,
        "stage": job.stage,
        "reports_persisted": summary.get("reports_persisted", 0),
        "decision_applied": False,
        "error_code": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
