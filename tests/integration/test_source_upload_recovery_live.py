"""Opt-in real PostgreSQL/object-store interruption and cleanup acceptance."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.job import Job, JobStatus
from app.models.source_repository import ContentObjectRecord
from app.services.source_repository import (
    admit_representation, commit_source_admissions, delete_representation,
    finalize_pending_object_deletions, _advisory_transaction_lock,
)
from app.services.source_upload_recovery import INTENT_PREFIX, CURSOR_KEY, cleanup_source_upload_intents
from app.services.paper_upload import cleanup_stale_paper_job_inputs
from app.services.paper_dispatch import job_execution_lock
from test_paper_dispatch_broker_live import _public_snapshot


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_SOURCE_UPLOAD_RECOVERY") != "1",
    reason="requires isolated PostgreSQL schema and MinIO prefix for upload recovery")


def test_live_recorded_upload_recovery_and_concurrent_admission(monkeypatch, tmp_path):
    namespace = "sf_source_upload_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    url = make_url(settings.DATABASE_URL).update_query_dict({"options": "-csearch_path=" + namespace})
    isolated = create_engine(url)
    factory = sessionmaker(bind=isolated, expire_on_commit=False)
    script = Path(__file__).with_name("source_upload_recovery_live_worker.py")
    spec = importlib.util.spec_from_file_location("isolated_source_recovery", script)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    backend = support.NamespacedStorage(namespace)
    env = dict(os.environ, DATABASE_URL=url.render_as_string(hide_password=False),
               SOURCEFIDELITY_TEST_NAMESPACE=namespace)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    receipt = {"namespace": namespace, "real_postgresql": True, "real_object_storage": True,
               "model_calls": 0, "cases": {}}
    created = False

    def interrupt(mode, label):
        with (tmp_path / "isolated-exit.log").open("ab") as log:
            result = subprocess.run([sys.executable, str(script), mode, label], env=env,
                                    stdout=log, stderr=log, timeout=30)
        assert result.returncode == 73

    def cleanup():
        with factory() as session:
            return cleanup_source_upload_intents(session, backend, now=future)

    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        with isolated.connect() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == namespace
        Base.metadata.create_all(isolated)

        for mode in ("before-source", "after-source", "rollback-delete-failed"):
            interrupt(mode, mode)
            assert len(backend.list_keys(INTENT_PREFIX)) == 1
            result = cleanup()
            if mode == 'rollback-delete-failed':
                assert result['removed'] == 1
                assert backend.list_keys('') == []
                receipt['cases'][mode] = 'recovered'
            else:
                assert result['pending'] == 1
                watches = backend.list_keys(INTENT_PREFIX)
                assert backend.list_keys('') == watches and len(watches) == 1
                receipt['cases'][mode] = 'bytes_removed_uncertain_watch_retained'
                # Dispose only this completed synthetic case's watch so the
                # following independent scenarios retain exact denominators.
                assert backend.delete(watches[0])

        interrupt("after-commit", "committed")
        assert cleanup()["protected"] == 1
        with factory() as session:
            committed = session.scalar(select(ContentObjectRecord))
            assert backend.exists(committed.storage_key)
        receipt["cases"]["after-commit"] = "source_preserved"

        # Expired intent must not defeat an admission's transaction lock, even
        # while its content record is uncommitted and invisible to cleanup.
        with factory() as session:
            record = admit_representation(session, backend, support.fixture_request("busy"))
            assert cleanup()["busy"] == 1
            assert backend.exists(record.content_object.storage_key)
            commit_source_admissions(session)
        receipt["cases"]["active-admission"] = "source_preserved"

        with factory() as session:
            record = admit_representation(session, backend, support.fixture_request("retention"))
            commit_source_admissions(session)
            obj = record.content_object
            key, digest, license_class = obj.storage_key, obj.content_sha256, obj.license_class
            assert delete_representation(session, record.id)
            session.commit()
        with factory() as owner:
            _advisory_transaction_lock(owner, "content-object", f"{license_class}:{digest}")
            with factory() as session:
                assert finalize_pending_object_deletions(session, backend) == 0
                session.commit()
            assert backend.exists(key)
        with factory() as session:
            assert finalize_pending_object_deletions(session, backend) == 1
            session.commit()
        assert not backend.exists(key)
        receipt["cases"]["retention-content-lock"] = "preserved_while_busy_then_removed"

        interrupt("after-confirmation", "adopted")
        with factory() as session:
            record = admit_representation(session, backend, support.fixture_request("adopted"))
            commit_source_admissions(session)
            source = record.content_object.storage_key
        assert cleanup()["removed"] == 1  # Old generation, not the new admission.
        assert backend.exists(source)
        receipt["cases"]["new-admission"] = "source_preserved"

        interrupt("after-confirmation", "journal-delete-failure")
        delete = backend.delete
        with monkeypatch.context() as local:
            local.setattr(backend, "delete", lambda key: False if key.startswith(INTENT_PREFIX) else delete(key))
            assert cleanup()["pending"] == 1
        assert len(backend.list_keys(INTENT_PREFIX)) == 1
        assert cleanup()["removed"] == 1
        receipt["cases"]["journal-delete-failure"] = "retried"

        backend.upload(b"Synthetic object without cleanup authority", "unmanaged-sentinel.txt")
        assert cleanup()["inspected"] == 0
        assert backend.exists("unmanaged-sentinel.txt")
        receipt["cases"]["unrecorded-object"] = "preserved"

        invalid = INTENT_PREFIX + "0" * 32 + ".json"
        backend.upload(b"invalid test recovery record", invalid)
        interrupt("after-confirmation", "bounded-page")
        with factory() as session:
            assert cleanup_source_upload_intents(session, backend, now=future, batch_size=1)["invalid"] == 1
        with factory() as session:
            assert cleanup_source_upload_intents(session, backend, now=future, batch_size=1)["removed"] == 1
        assert backend.exists(invalid)
        backend.delete(invalid)  # Explicitly dispose of our invalid test fixture.
        assert cleanup()["inspected"] == 0
        assert not backend.exists(CURSOR_KEY)
        receipt["cases"]["bounded-page-progress"] = "invalid_record_did_not_block_later_cleanup"

        interrupt("paper-input", "input")
        with factory() as session:
            job = session.scalar(select(Job))
            assert job.status == JobStatus.PENDING
            assert job.upload_evidence["paper_input_upload_state"] == "pending"
            assert "workflow_dispatch_v1" not in job.upload_evidence
            expiry = job.input_expires_at + timedelta(seconds=1)
            with job_execution_lock(session, str(job.id)):
                with factory() as cleanup_session:
                    assert cleanup_stale_paper_job_inputs(cleanup_session, backend, now=expiry)["jobs_pending"] == 1
                assert backend.exists(job.input_storage_key)
        with factory() as session:
            assert cleanup_stale_paper_job_inputs(session, backend, now=expiry)["jobs_cleaned"] == 1
            job = session.scalar(select(Job))
            assert job.status == JobStatus.FAILED
            assert job.error_message == "paper_upload_interrupted"
            assert job.input_storage_key is None
        receipt["cases"]["paper-input-exit"] = "failed_and_cleaned_after_expiry"
        receipt["cases"]["active-paper-upload"] = "cleanup_skipped"
        receipt["public_data_unchanged"] = _public_snapshot() == before
        assert receipt["public_data_unchanged"]
        receipt["status"] = "passed"
    finally:
        isolated.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        # Only objects under this fresh, validated test namespace are removed.
        keys = backend.storage.list_keys(backend.prefix)
        assert all(key.startswith(namespace + "/") for key in keys)
        for key in keys:
            assert backend.storage.delete(key)
        assert backend.storage.list_keys(backend.prefix) == []
        receipt["test_schema_removed"] = created
        receipt["remaining_test_objects_removed"] = len(keys)
        (tmp_path / "source-upload-recovery-receipt.json").write_text(json.dumps(receipt, indent=2))
