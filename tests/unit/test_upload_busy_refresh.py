"""An accepted upload during another refresh is queued, not refused (2026-10-01, paper 5 Langford)."""
import pytest
from fastapi import HTTPException

from app.routers import report as report_router
from app.services.paper_workflow import PaperWorkflowError

ADMISSION = {"review_status": "accepted", "documents": [{"admission_state": "accepted", "id": "rep-1"}]}


def test_a_busy_report_queues_the_refresh(monkeypatch):
    def busy(*a, **k):
        raise PaperWorkflowError("report_reanalysis_busy", "Another report reanalysis is already in progress")
    queued = []
    monkeypatch.setattr(report_router, "prepare_uploaded_source_refresh", busy)
    monkeypatch.setattr("app.tasks.source_reanalysis.schedule_busy_upload_refresh",
                        lambda *args: queued.append(args) or "task-1")
    result = report_router._start_uploaded_source_reanalysis(
        None, None, report_id="r-1", reference_id="ref-9", admission=ADMISSION)
    assert queued == [("r-1", "ref-9", "rep-1")]
    assert result["reanalysis_status"] == "scheduled" and result["task_id"] == "task-1"


def test_other_preparation_failures_still_refuse(monkeypatch):
    def mismatch(*a, **k):
        raise PaperWorkflowError("uploaded_source_identity_mismatch", "Uploaded source identity does not match")
    monkeypatch.setattr(report_router, "prepare_uploaded_source_refresh", mismatch)
    with pytest.raises(HTTPException) as caught:
        report_router._start_uploaded_source_reanalysis(
            None, None, report_id="r-1", reference_id="ref-9", admission=ADMISSION)
    assert caught.value.status_code == 409
