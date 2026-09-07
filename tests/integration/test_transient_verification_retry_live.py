"""Opt-in transient-batch retry checks against isolated PostgreSQL and MinIO.

Reuse the synthetic DOCX/source regression cases with real transactions and
object storage. No production jobs, broker queues, sources or models are used.
"""

import importlib.util
import json
import os
from pathlib import Path
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from test_paper_dispatch_broker_live import _public_snapshot


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TRANSIENT_RETRY") != "1",
    reason="requires isolated PostgreSQL schema and MinIO prefix for transient retry")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", [
    "processing", "partial_batch", "lost_commit_ack", "task_budget",
    "abandoned_lease", "cleaned", "tampered", "expired", "missing",
])
def test_transient_retry_real_storage(monkeypatch, tmp_path, case):
    namespace = "sf_source_upload_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    url = make_url(settings.DATABASE_URL).update_query_dict({"options": "-csearch_path=" + namespace})
    isolated = create_engine(url)
    factory = sessionmaker(bind=isolated, expire_on_commit=False)
    support = _load("transient_retry_storage_fixture",
        Path(__file__).with_name("source_upload_recovery_live_worker.py"))
    regressions = _load("transient_retry_regression_fixture",
        Path(__file__).parents[1] / "unit" / "test_paper_workflow.py")
    backend = support.NamespacedStorage(namespace)
    create_job = regressions._retry_test_job
    monkeypatch.setattr(regressions, "_retry_test_job",
        lambda patch, **kwargs: create_job(patch, factory=factory, storage=backend, **kwargs))
    receipt = {"namespace": namespace, "case": case, "real_postgresql": True,
               "real_object_storage": True, "synthetic_inputs": True, "model_calls": 0}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        with isolated.connect() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == namespace
        Base.metadata.create_all(isolated)
        if case in {"processing", "partial_batch"}:
            regressions.test_verification_retry_keeps_source_before_batch_commit(monkeypatch, case)
        elif case == "lost_commit_ack":
            regressions.test_verification_retry_recovers_a_lost_batch_commit_acknowledgement(monkeypatch)
        elif case == "task_budget":
            regressions.test_transient_retry_respects_persisted_task_budget(monkeypatch)
        elif case == "abandoned_lease":
            regressions.test_transient_retry_abandoned_bytes_expire(monkeypatch)
        else:
            regressions.test_interrupted_transient_source_never_becomes_successful_missing_evidence(monkeypatch, case)
        receipt["public_data_unchanged"] = _public_snapshot() == before
        assert receipt["public_data_unchanged"]
        receipt["status"] = "passed"
    finally:
        isolated.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        # Dispose only of objects created inside this exact validated namespace.
        # Read through the original backend: individual regressions may replace
        # the wrapper's download method to prohibit evidence reprocessing.
        keys = backend.storage.list_keys(backend.prefix)
        assert all(key.startswith(namespace + "/") for key in keys)
        for key in keys:
            assert backend.storage.delete(key)
        assert backend.storage.list_keys(backend.prefix) == []
        receipt["test_schema_removed"] = created
        receipt["remaining_test_objects_removed"] = len(keys)
        (tmp_path / "transient-retry-receipt.json").write_text(json.dumps(receipt, indent=2))
