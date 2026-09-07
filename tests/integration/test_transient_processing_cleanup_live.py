"""Opt-in processor/expiry interleavings on isolated PostgreSQL and MinIO."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.job import Job
from app.models.report import VerificationReportRecord
from app.models.verification_run import VerificationRunRecord
from app.services import paper_workflow, verification_run as runs
from app.services.paper_dispatch import attempt_id_for
from app.services.workflow_execution import WorkflowOwnershipLost
from app.tasks import check_paper as tasks
from source_upload_recovery_live_worker import NamespacedStorage
from test_paper_dispatch_broker_live import _public_snapshot
from test_transient_verification_retry_live import _load


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TRANSIENT_PROCESSING") != "1",
    reason="requires isolated PostgreSQL schema and MinIO prefix for active-source cleanup")


def _expiry_scan(factory, backend, run_id, *, expected):
    with factory() as session:
        run = session.get(VerificationRunRecord, run_id)
        expiry, key = run.lease_expires_at, run.transient_objects[0]["key"]
        result = runs.cleanup_stale_verification_runs(session, backend,
            now=expiry + timedelta(seconds=1))
        assert result["runs_cleaned"] == expected
        assert backend.exists(key) is (expected == 0)
        if expected == 0:
            session.refresh(run)
            assert run.status == "active" and run.cleanup_attempts == 0
            assert run.lease_expires_at == expiry
            assert all(backend.exists(item["key"]) for item in run.transient_objects)


@pytest.mark.parametrize("case", ["paper_retry", "standalone_success", "abandoned",
    "connection_lost", "renewed_after_scan", "hard_exit"])
def test_processing_and_expiry_are_serialized(monkeypatch, tmp_path, case):
    namespace = "sf_source_upload_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    engine = create_engine(make_url(settings.DATABASE_URL).update_query_dict(
        {"options": "-csearch_path=" + namespace}))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    backend = NamespacedStorage(namespace)
    fixture = _load("active_source_fixture", Path(__file__).parents[1] / "unit/test_verification_run.py")
    receipt = {"namespace": namespace, "case": case, "synthetic_inputs": True,
        "model_calls": 0, "expiry_advanced_explicitly": True}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        Base.metadata.create_all(engine)
        if case == "paper_retry":
            paper = _load("active_paper_fixture", Path(__file__).parents[1] / "unit/test_paper_workflow.py")
            _, _, job_id = paper._retry_test_job(monkeypatch, factory=factory, storage=backend)
            monkeypatch.setattr(tasks, "SessionLocal", factory)
            monkeypatch.setattr(tasks, "get_storage_backend", lambda: backend)
            monkeypatch.setattr(paper_workflow.settings, "PAPER_LLM_PROCESSING_ENABLED", False)
            with factory() as session:
                attempt = attempt_id_for(session.get(Job, job_id))
            from celery.exceptions import Retry
            def retry(**kwargs):
                raise Retry()
            task = SimpleNamespace(request=SimpleNamespace(retries=0), retry=retry)
            def interrupt(*args, **kwargs):
                with factory() as session:
                    run_id = session.scalar(select(VerificationRunRecord.id))
                _expiry_scan(factory, backend, run_id, expected=0)
                raise ConnectionError("Synthetic interrupted processor")
            with monkeypatch.context() as patch:
                patch.setattr(paper_workflow, "_shadow_artifact", interrupt)
                with pytest.raises(Retry):
                    tasks._run_stage(task, str(job_id), attempt, "verify")
            # Directly retry the source workflow; task delay policy is covered
            # independently by the persisted-stage-budget regression.
            assert paper_workflow.verify_paper_sources(factory, backend, job_id,
                llm_enabled=False)["reports_persisted"] == 2
        elif case in {"standalone_success", "connection_lost"}:
            def processor(session, source, run_id):
                artifact = fixture._processor(backend)(session, source, run_id)
                _expiry_scan(factory, backend, run_id, expected=0)
                # A busy first candidate must not block cleanup of another
                # expired run in the same scheduled batch.
                with factory() as abandoned_session:
                    runs.begin_verification_run(abandoned_session, backend,
                        fixture._request(), lease_seconds=1)
                with factory() as cleaning_session:
                    active = cleaning_session.get(VerificationRunRecord, run_id)
                    counts = runs.cleanup_stale_verification_runs(cleaning_session, backend,
                        now=active.lease_expires_at + timedelta(seconds=1))
                    assert counts == {"runs_cleaned": 1, "runs_pending": 1}
                with factory() as competitor:
                    assert not runs.cleanup_verification_run(competitor, backend, run_id)
                with pytest.raises(runs.VerificationRunBusy):
                    runs.complete_active_verification_run(factory, backend, run_id,
                        scope_type="personal_owner", scope_id="owner-1",
                        processor=lambda *args: pytest.fail("Duplicate processor entered"))
                if case == "connection_lost":
                    session.get_bind().invalidate()
                    _expiry_scan(factory, backend, run_id, expected=1)
                return artifact
            if case == "connection_lost":
                with pytest.raises(WorkflowOwnershipLost):
                    runs.execute_verification_run(factory, backend, fixture._request(), processor)
                with factory() as session:
                    assert not session.scalars(select(VerificationReportRecord)).all()
            else:
                result = runs.execute_verification_run(factory, backend, fixture._request(), processor)
                with factory() as session:
                    assert session.get(VerificationReportRecord, result.report_id) is not None
                    assert session.get(VerificationRunRecord, result.run_id).status == "cleaned"
        else:
            with factory() as session:
                run = runs.begin_verification_run(session, backend, fixture._request())
                run_id, original_expiry = run.id, run.lease_expires_at
            if case == "renewed_after_scan":
                original_cleanup = runs.cleanup_verification_run
                def renew_then_cleanup(session, storage, selected_id, **kwargs):
                    with factory() as renewing:
                        runs.renew_verification_run_lease(renewing, selected_id,
                            scope_type="personal_owner", scope_id="owner-1", lease_seconds=100_000)
                    return original_cleanup(session, storage, selected_id, **kwargs)
                with monkeypatch.context() as patch:
                    patch.setattr(runs, "cleanup_verification_run", renew_then_cleanup)
                    with factory() as session:
                        result = runs.cleanup_stale_verification_runs(session, backend,
                            now=original_expiry + timedelta(seconds=1))
                        assert result["runs_cleaned"] == 0
                with factory() as session:
                    assert session.get(VerificationRunRecord, run_id).status == "active"
                _expiry_scan(factory, backend, run_id, expected=1)
            elif case == "hard_exit":
                ready = tmp_path / "processor-ready"
                env = dict(os.environ, DATABASE_URL=engine.url.render_as_string(hide_password=False),
                    SOURCEFIDELITY_TEST_NAMESPACE=namespace, PYTHONPATH=str(Path(__file__).parents[2]))
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    str(run_id), str(ready)], env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    deadline = time.monotonic() + 15
                    while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    assert ready.exists(), "Synthetic helper did not reach processing"
                    _expiry_scan(factory, backend, run_id, expected=0)
                    child.communicate(b"exit\n", timeout=15)
                    assert child.returncode == 73
                    _expiry_scan(factory, backend, run_id, expected=1)
                finally:
                    if child.poll() is None:
                        child.kill()  # Only this newly created isolated helper.
                        child.wait(timeout=10)
            else:
                def fail(session, source, selected_id):
                    _expiry_scan(factory, backend, selected_id, expected=0)
                    raise ConnectionError("Synthetic abandoned processor")
                with pytest.raises(ConnectionError):
                    runs.complete_active_verification_run(factory, backend, run_id,
                        scope_type="personal_owner", scope_id="owner-1", processor=fail,
                        retain_on_retryable_failure=True)
                _expiry_scan(factory, backend, run_id, expected=1)
        receipt["public_data_unchanged"] = _public_snapshot() == before
        assert receipt["public_data_unchanged"]
        receipt["status"] = "passed"
    finally:
        engine.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        for key in backend.storage.list_keys(backend.prefix):
            assert key.startswith(namespace + "/")
            assert backend.storage.delete(key)
        assert not backend.storage.list_keys(backend.prefix)
        receipt["test_data_removed"] = True
        (tmp_path / "active-source-receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    namespace = os.environ["SOURCEFIDELITY_TEST_NAMESPACE"]
    assert namespace.startswith("sf_source_upload_test_")
    assert len(namespace.removeprefix("sf_source_upload_test_")) == 32
    int(namespace.removeprefix("sf_source_upload_test_"), 16)
    assert make_url(settings.DATABASE_URL).query["options"] == "-csearch_path=" + namespace
    helper_factory = sessionmaker(bind=create_engine(settings.DATABASE_URL), expire_on_commit=False)
    def hard_exit(session, source, run_id):
        assert source is not None
        Path(sys.argv[2]).write_text("ready")
        assert sys.stdin.readline().strip() == "exit"
        os._exit(73)
    runs.complete_active_verification_run(helper_factory, NamespacedStorage(namespace), sys.argv[1],
        scope_type="personal_owner", scope_id="owner-1", processor=hard_exit)
