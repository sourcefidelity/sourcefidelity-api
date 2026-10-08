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
    waiting = []
    monkeypatch.setattr("app.services.paper_workflow.set_queued_source_upload",
                        lambda _s, *args, queued: waiting.append((*args, queued)))
    session = type("Session", (), {"rollback": lambda self: None})()
    result = report_router._start_uploaded_source_reanalysis(
        session, None, report_id="r-1", reference_id="ref-9", admission=ADMISSION)
    assert queued == [("r-1", "ref-9", "rep-1")]
    # The page keeps waiting for it (2026-10-07: uploads made one after another).
    assert waiting == [("r-1", "ref-9", "rep-1", True)]
    assert result["reanalysis_status"] == "scheduled" and result["task_id"] == "task-1"


def test_other_preparation_failures_still_refuse(monkeypatch):
    def mismatch(*a, **k):
        raise PaperWorkflowError("uploaded_source_identity_mismatch", "Uploaded source identity does not match")
    monkeypatch.setattr(report_router, "prepare_uploaded_source_refresh", mismatch)
    with pytest.raises(HTTPException) as caught:
        report_router._start_uploaded_source_reanalysis(
            None, None, report_id="r-1", reference_id="ref-9", admission=ADMISSION)
    assert caught.value.status_code == 409


def test_a_queued_upload_is_cleared_once_its_refresh_is_prepared(monkeypatch):
    from app.tasks import source_reanalysis
    calls = []
    monkeypatch.setattr("app.services.paper_workflow.prepare_uploaded_source_refresh",
                        lambda *a, **k: {"scheduled": True, "job_id": "j-1", "attempt_id": "a-1"})
    monkeypatch.setattr("app.services.paper_workflow.set_queued_source_upload",
                        lambda _s, *args, queued: calls.append((*args, queued)))
    monkeypatch.setattr(source_reanalysis, "schedule_uploaded_source_reanalysis", lambda *a: "task-2")
    monkeypatch.setattr("app.services.storage.backend.get_storage_backend", lambda: None)

    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def rollback(self): pass
    monkeypatch.setattr("app.database.SessionLocal", FakeSession)
    result = source_reanalysis.retry_uploaded_source_refresh.run("r-1", "ref-9", "rep-1")
    assert result == {"status": "scheduled", "task_id": "task-2"}
    assert calls == [("r-1", "ref-9", "rep-1", False)]
