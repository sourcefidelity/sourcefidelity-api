"""Retrieve the citation checking report for a job."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/{job_id}")
async def get_report(job_id: str, format: str = "json"):
    """
    Get the citation checking report.

    - **format**: "json" or "markdown"
    """
    # Placeholder – will query PostgreSQL in Phase 5
    raise HTTPException(status_code=501, detail="Not yet implemented (Phase 5)")
