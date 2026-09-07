"""Targeted report reanalysis after an admitted user source upload."""

from app.tasks.check_paper import dispatch_paper_workflow


def schedule_uploaded_source_reanalysis(job_id: str, attempt_id: str) -> str | None:
    """Queue only verification and finalization for the affected source member."""
    return dispatch_paper_workflow(job_id, attempt_id)
