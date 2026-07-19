"""Check the status of a citation checking job."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a citation checking job.

    Returns: pending, running, completed, or failed
    """
    # Placeholder – will query Celery/PostgreSQL in Phase 5
    raise HTTPException(status_code=501, detail="Not yet implemented (Phase 5)")
