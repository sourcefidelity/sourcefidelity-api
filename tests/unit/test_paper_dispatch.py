"""Isolated interruption, duplicate-delivery and attempt-fencing checks."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

from celery.exceptions import Ignore, Retry
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.services.paper_dispatch import (
    DISPATCH_KEY, MAX_STAGE_STARTS, attempt_id_for, claim_dispatch,
    prepare_dispatch, record_publication, recovery_candidates,
)
from app.services.paper_workflow import PaperWorkflowError
from app.services.processing_metrics import measure_paper_stage
from app.tasks import check_paper as tasks


@pytest.fixture
def workflow(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        job = Job(filename="synthetic.docx", paper_version_id=str(uuid.uuid4()),
                  scope_id="test-owner", input_sha256="a" * 64,
                  input_media_type="application/docx", input_byte_size=100,
                  input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                  upload_evidence={}, stage=JobStage.RETRIEVED, status=JobStatus.RUNNING)
        attempt = prepare_dispatch(job)
        session.add(job)
        session.commit()
        job_id = str(job.id)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "get_storage_backend", lambda: object())
    yield factory, job_id, attempt
    engine.dispose()


def edit(factory, job_id, change):
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        change(job)
        session.commit()


def expire_lease(factory, job_id):
    def change(job):
        intent = dict(job.upload_evidence[DISPATCH_KEY])
        intent["dispatch_after"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        job.upload_evidence = {**job.upload_evidence, DISPATCH_KEY: intent}
    edit(factory, job_id, change)


def test_interrupted_publication_recovers_same_attempt_from_checkpoint(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    published = []

    def publish(*steps):
        published.append(steps)
        return SimpleNamespace(apply_async=Mock(side_effect=ConnectionError("lost acknowledgement")))

    monkeypatch.setattr(tasks, "chain", publish)
    with pytest.raises(ConnectionError):
        tasks.dispatch_paper_workflow(job_id, attempt)
    assert tasks.dispatch_paper_workflow(job_id, attempt) is None
    expire_lease(factory, job_id)
    recovered = tasks.recover_pending_paper_workflows.run()
    assert recovered == {"candidates": 1, "published": 0, "publication_unavailable": 1}
    assert len(published) == 2
    for steps in published:
        assert [step.task for step in steps] == ["verify_paper_sources", "finalize_paper_job"]
        assert all(step.args == (job_id, attempt) for step in steps)
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert attempt_id_for(job) == attempt
        assert job.upload_evidence[DISPATCH_KEY]["dispatch_count"] == 2
        assert job.stage == JobStage.RETRIEVED


@pytest.mark.parametrize("status,managed", [(JobStatus.COMPLETED, True), (JobStatus.FAILED, True), (JobStatus.RUNNING, False)])
def test_recovery_never_adopts_terminal_or_legacy_jobs(workflow, status, managed):
    factory, job_id, attempt = workflow
    def change(job):
        job.status = status
        if not managed:
            job.upload_evidence = {}
    edit(factory, job_id, change)
    with factory() as session:
        assert recovery_candidates(session) == []
    assert claim_dispatch(factory, job_id, expected_attempt=attempt) is None


@pytest.mark.parametrize("token", [None, "obsolete-attempt"])
def test_stale_or_untokened_delivery_cannot_run_fail_or_record_metrics(workflow, monkeypatch, token):
    factory, job_id, _attempt = workflow
    operation = Mock()
    fail = Mock()
    monkeypatch.setattr(tasks, "verify_paper_sources", operation)
    monkeypatch.setattr(tasks, "fail_paper_job", fail)
    with pytest.raises(Ignore):
        tasks.verify_paper_sources_task.run(job_id, token)
    tasks._fail(job_id, RuntimeError("late failure"), token)
    record_publication(factory, job_id, token, "late-task")
    operation.assert_not_called()
    fail.assert_not_called()
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert "processing_stages" not in job.upload_evidence
        assert job.task_id is None


def test_completed_stage_duplicate_does_not_execute_or_spend_budget(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    operation = Mock()
    monkeypatch.setattr(tasks, "retrieve_paper_sources", operation)
    assert tasks.retrieve_paper_sources_task.run(job_id, attempt) == {"already_completed": True}
    operation.assert_not_called()
    with factory() as session:
        assert session.get(Job, uuid.UUID(job_id)).upload_evidence[DISPATCH_KEY]["stage_starts"] == {}


def test_premature_chain_stage_is_stopped(workflow, monkeypatch):
    _factory, job_id, attempt = workflow
    operation = Mock()
    monkeypatch.setattr(tasks, "finalize_paper_job", operation)
    with pytest.raises(Ignore):
        tasks.finalize_paper_job_task.run(job_id, attempt)
    operation.assert_not_called()


def test_busy_duplicate_never_fails_current_worker_even_after_retry_limit(workflow, monkeypatch):
    _factory, job_id, attempt = workflow
    @contextmanager
    def busy(*args):
        raise PaperWorkflowError("job_stage_busy", "owned")
        yield
    fail = Mock()
    monkeypatch.setattr(tasks, "_job_execution_lock", busy)
    monkeypatch.setattr(tasks, "_fail", fail)
    with pytest.raises(Ignore):
        tasks.verify_paper_sources_task.run(job_id, attempt)
    with pytest.raises(Ignore):
        tasks._retry_or_fail(SimpleNamespace(request=SimpleNamespace(retries=999)), job_id,
                             PaperWorkflowError("job_stage_busy", "owned"), attempt, "verify")
    fail.assert_not_called()


def test_persisted_start_budget_survives_fresh_deliveries(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    # Simulate process exits after the committed start, with no Celery retry record.
    for _ in range(MAX_STAGE_STARTS):
        with factory() as session:
            assert tasks._start_stage(session, job_id, attempt, "verify")
    operation = Mock()
    fail = Mock()
    monkeypatch.setattr(tasks, "verify_paper_sources", operation)
    monkeypatch.setattr(tasks, "fail_paper_job", fail)
    with pytest.raises(RuntimeError, match="paper_workflow_error"):
        tasks.verify_paper_sources_task.run(job_id, attempt)
    operation.assert_not_called()
    fail.assert_called_once()
    assert fail.call_args.args[-1].code == "workflow_stage_attempts_exhausted"


def test_retry_backoff_is_durable_and_recovery_cannot_bypass_it(workflow):
    factory, job_id, attempt = workflow
    with factory() as session:
        tasks._start_stage(session, job_id, attempt, "verify")
    task = SimpleNamespace(request=SimpleNamespace(retries=0), retry=Mock(side_effect=Retry()))
    with pytest.raises(Retry):
        tasks._retry_or_fail(task, job_id, ConnectionError("synthetic"), attempt, "verify")
    assert task.retry.call_args.kwargs["countdown"] == 60
    assert claim_dispatch(factory, job_id, expected_attempt=attempt) is None
    with factory() as session:
        with pytest.raises(Ignore):
            tasks._start_stage(session, job_id, attempt, "verify")


def test_metrics_finalizer_cannot_overwrite_new_attempt(workflow):
    factory, job_id, attempt = workflow
    with measure_paper_stage(factory, job_id, "finalize", workflow_attempt=attempt):
        edit(factory, job_id, lambda job: prepare_dispatch(job, attempt_id="new-attempt"))
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert attempt_id_for(job) == "new-attempt"
        assert "processing_stages" not in job.upload_evidence


def test_normal_stage_advances_and_records_exact_attempt(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    def verify(_factory, _backend, value):
        edit(factory, value, lambda job: setattr(job, "stage", JobStage.VERIFIED))
        return {"synthetic": True}
    monkeypatch.setattr(tasks, "verify_paper_sources", verify)
    assert tasks.verify_paper_sources_task.run(job_id, attempt) == {"synthetic": True}
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert job.stage == JobStage.VERIFIED
        assert job.upload_evidence[DISPATCH_KEY]["stage_starts"] == {"verify": 1}
        assert len(job.upload_evidence["processing_stages"]) == 1


def test_new_refresh_cannot_be_dispatched_by_old_control_message(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    edit(factory, job_id, lambda job: prepare_dispatch(job, attempt_id="successor-attempt"))
    publisher = Mock()
    monkeypatch.setattr(tasks, "chain", publisher)
    assert tasks.check_paper_task.run(job_id, attempt)["workflow_task_id"] is None
    publisher.assert_not_called()


def test_successful_publication_binds_task_id_and_store_only_stages(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    def change(job):
        job.store_only = True
        job.stage = JobStage.UPLOADED
    edit(factory, job_id, change)
    publisher = Mock(return_value=SimpleNamespace(apply_async=lambda: SimpleNamespace(id="workflow-task")))
    monkeypatch.setattr(tasks, "chain", publisher)
    assert tasks.dispatch_paper_workflow(job_id, attempt) == "workflow-task"
    assert [step.task for step in publisher.call_args.args] == ["extract_paper", "finalize_paper_job"]
    with factory() as session:
        assert session.get(Job, uuid.UUID(job_id)).task_id == "workflow-task"


def test_retry_publication_failure_preserves_recovery_intent_and_sanitizes_error(workflow):
    factory, job_id, attempt = workflow
    with factory() as session:
        tasks._start_stage(session, job_id, attempt, "verify")
    task = SimpleNamespace(request=SimpleNamespace(retries=0),
                           retry=Mock(side_effect=ConnectionError("private transport detail")))
    with pytest.raises(RuntimeError, match="Paper retry publication unavailable") as error:
        tasks._retry_or_fail(task, job_id, ConnectionError("private source text"), attempt, "verify")
    assert error.value.__suppress_context__
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert job.status == JobStatus.RUNNING
        assert job.upload_evidence[DISPATCH_KEY]["stage_retry_after"]["verify"]


def test_nonretryable_failure_terminates_only_matching_attempt(workflow, monkeypatch):
    factory, job_id, attempt = workflow
    operation = Mock(side_effect=ValueError("synthetic invalid input"))
    monkeypatch.setattr(tasks, "verify_paper_sources", operation)
    with pytest.raises(RuntimeError, match="Paper workflow failed"):
        tasks.verify_paper_sources_task.run(job_id, attempt)
    with factory() as session:
        job = session.get(Job, uuid.UUID(job_id))
        assert job.status == JobStatus.FAILED
        assert recovery_candidates(session) == []
    with pytest.raises(Ignore):
        tasks.verify_paper_sources_task.run(job_id, attempt)
    operation.assert_called_once()
