"""Opt-in real database/storage execution-loss regression, synthetic inputs only."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from celery.exceptions import Ignore
import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.job import Job
from app.models.report import VerificationReportRecord
from app.services import paper_workflow
from app.services.paper_dispatch import attempt_id_for, prepare_dispatch
from app.tasks import check_paper as tasks
from source_upload_recovery_live_worker import NamespacedStorage
from test_paper_dispatch_broker_live import _public_snapshot
from test_transient_verification_retry_live import _load


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_WORKFLOW_OWNERSHIP") != "1",
    reason="requires isolated PostgreSQL schema and MinIO prefix for execution ownership")


def _normal_pipeline(monkeypatch, factory, backend, fixture):
    from app.models.report import Report
    from app.services import paper_upload
    from app.services.file_safety import SafetyVerdict
    monkeypatch.setattr(paper_upload, "scan_with_clamd", lambda _content: (SafetyVerdict.CLEAN, "synthetic fixture"))
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "get_storage_backend", lambda: backend)
    monkeypatch.setattr(paper_workflow.settings, "PAPER_LLM_PROCESSING_ENABLED", False)
    monkeypatch.setattr(tasks, "retrieve_paper_sources", lambda session, storage, job_id:
        paper_workflow.retrieve_paper_sources(session, storage, job_id, resolver=fixture.Resolver()))
    with factory() as session:
        job = paper_upload.create_paper_job(session, backend, content=fixture._paper_bytes(),
            filename="synthetic.docx", media_type=paper_upload.DOCX_MEDIA_TYPE, scope_id="personal-default")
        job_id, attempt = job.id, attempt_id_for(job)
    task = SimpleNamespace(request=SimpleNamespace(retries=0),
        retry=lambda **kw: pytest.fail("Normal synthetic stages must not retry"))
    for stage in ("extract", "retrieve", "verify", "finalize"):
        tasks._run_stage(task, str(job_id), attempt, stage)
    with factory() as session:
        job = session.get(Job, job_id)
        assert job.status == "completed" and job.stage == "completed"
        assert job.verification_summary["reports_persisted"] == 2
        assert job.upload_evidence["workflow_dispatch_v1"]["stage_starts"] == {
            "extract": 1, "retrieve": 1, "verify": 1, "finalize": 1}
        assert len(session.scalars(select(Report)).all()) == 1
        assert len(session.scalars(select(VerificationReportRecord)).all()) == 2


@pytest.mark.parametrize("successor", ["same_attempt", "new_attempt", "no_peer", "normal_pipeline"])
def test_lost_lock_cannot_fail_or_write_after_successor(monkeypatch, tmp_path, successor):
    namespace = "sf_source_upload_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    engine = create_engine(make_url(settings.DATABASE_URL).update_query_dict({"options": "-csearch_path=" + namespace}))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    backend = NamespacedStorage(namespace)
    fixture = _load("ownership_synthetic_fixture", Path(__file__).parents[1] / "unit/test_paper_workflow.py")
    receipt = {"namespace": namespace, "case": successor, "synthetic_inputs": True, "model_calls": 0}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        Base.metadata.create_all(engine)
        if successor == "normal_pipeline":
            _normal_pipeline(monkeypatch, factory, backend, fixture)
            receipt["public_data_unchanged"] = _public_snapshot() == before
            assert receipt["public_data_unchanged"]
            receipt["status"] = "passed"
            return
        _, _, job_id = fixture._retry_test_job(monkeypatch, factory=factory, storage=backend)
        monkeypatch.setattr(tasks, "SessionLocal", factory)
        monkeypatch.setattr(tasks, "get_storage_backend", lambda: backend)
        monkeypatch.setattr(paper_workflow.settings, "PAPER_LLM_PROCESSING_ENABLED", False)
        with factory() as session:
            attempt = attempt_id_for(session.get(Job, job_id))
        task = SimpleNamespace(request=SimpleNamespace(retries=0),
            retry=lambda **kw: pytest.fail("An old execution must not publish a retry"))
        owned = []
        peer_reports = {}
        peer_job = {}
        def observe(connection, cursor, statement, parameters, context, executemany):
            if "SELECT pg_try_advisory_lock" in statement:
                owned.append(connection)
        event.listen(engine, "after_cursor_execute", observe)
        original = paper_workflow._shadow_artifact
        intervened = False
        def interleave(*args, **kwargs):
            nonlocal intervened
            if not intervened:
                intervened = True
                # Close only this synthetic job's physical lock connection.
                owned[0].invalidate()
                if successor != "no_peer":
                    peer_attempt = attempt
                    if successor == "new_attempt":
                        with factory() as session:
                            peer_attempt = prepare_dispatch(session.get(Job, job_id))
                            session.commit()
                    result = tasks._run_stage(task, str(job_id), peer_attempt, "verify")
                    assert result["reports_persisted"] == 2
                    with factory() as session:
                        peer_reports.update({str(row.id): row.evidence_sha256 for row in session.scalars(select(VerificationReportRecord))})
                        job = session.get(Job, job_id)
                        peer_job.update({"stage": job.stage, "status": job.status,
                            "summary": job.verification_summary, "upload_evidence": job.upload_evidence})
            return original(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(paper_workflow, "_shadow_artifact", interleave)
            with pytest.raises(Ignore):
                tasks._run_stage(task, str(job_id), attempt, "verify")
        event.remove(engine, "after_cursor_execute", observe)
        with factory() as session:
            job = session.get(Job, job_id)
            rows = {str(row.id): row.evidence_sha256 for row in session.scalars(select(VerificationReportRecord))}
            assert rows == peer_reports
            if successor != "no_peer":
                assert {"stage": job.stage, "status": job.status,
                    "summary": job.verification_summary, "upload_evidence": job.upload_evidence} == peer_job
                assert job.stage == "verified"
            else:
                assert job.stage == "verifying" and job.status == "running"
                assert job.verification_summary is None
        if successor == "no_peer":
            assert tasks._run_stage(task, str(job_id), attempt, "verify")["reports_persisted"] == 2
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
        (tmp_path / "ownership-receipt.json").write_text(json.dumps(receipt, indent=2))
