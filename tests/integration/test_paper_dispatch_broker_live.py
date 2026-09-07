"""Opt-in real PostgreSQL/Redis recovery with an isolated worker and schema.

No existing job is dispatched. No source storage or model is accessed. Only
the schema/Redis namespace and subprocesses created by this test are cleaned.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest
from redis import Redis
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report
from app.services.paper_dispatch import DISPATCH_KEY, prepare_dispatch
from app.services.paper_extraction import PaperExtractionArtifact
from app.tasks import check_paper as tasks


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PAPER_DISPATCH_BROKER") != "1",
    reason="requires isolated PostgreSQL schema, Redis namespace and test worker",
)


def _public_snapshot():
    # Only digests leave this function; no source/student contents are logged.
    result = {}
    with public_engine.connect() as connection:
        for table in Base.metadata.sorted_tables:
            rows = connection.execute(select(table).order_by(*table.primary_key)).mappings()
            digest = hashlib.sha256()
            count = 0
            for row in rows:
                digest.update(json.dumps(dict(row), default=str, sort_keys=True).encode())
                count += 1
            result[table.name] = (count, digest.hexdigest())
    return result


def _wait(predicate, *, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.15)
    raise AssertionError("Isolated scheduling acceptance deadline exceeded")


def test_real_queue_interruption_replay_and_attempt_fences(monkeypatch, tmp_path):
    prefix = "sf_dispatch_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    url = make_url(settings.DATABASE_URL).update_query_dict({"options": "-csearch_path=" + prefix})
    isolated = create_engine(url, pool_pre_ping=True)
    factory = sessionmaker(bind=isolated, expire_on_commit=False)
    bootstrap = Path(__file__).with_name("paper_dispatch_live_worker.py")
    env = dict(os.environ, DATABASE_URL=url.render_as_string(hide_password=False),
               SOURCEFIDELITY_TEST_NAMESPACE=prefix, PAPER_LLM_PROCESSING_ENABLED="false")
    spec = importlib.util.spec_from_file_location("isolated_dispatch_worker", bootstrap)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    app = support.configure(prefix)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    redis = Redis.from_url(app.conf.broker_url)
    worker = None
    created = False
    log = (tmp_path / "isolated-worker.log").open("wb")
    receipt = {"namespace": prefix, "real_broker": True, "source_access": False,
               "model_calls": 0, "cases": {}}

    def create_case(label):
        job_id = uuid.uuid4()
        extraction = PaperExtractionArtifact(paper_version_id=str(job_id), citation_format="apa")
        job = Job(id=job_id, filename="synthetic-queue-fixture.docx", scope_id=prefix,
                  paper_version_id=str(job_id), input_sha256="a" * 64,
                  input_media_type="application/docx", input_byte_size=1,
                  input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                  stage=JobStage.EXTRACTED, status=JobStatus.RUNNING, store_only=True,
                  extraction_payload=extraction.model_dump(mode="json"), upload_evidence={})
        attempt = prepare_dispatch(job)
        with factory() as session:
            session.add(job)
            session.commit()
        return str(job_id), attempt

    def job_state(job_id):
        with factory() as session:
            job = session.get(Job, uuid.UUID(job_id))
            return job.status, job.upload_evidence[DISPATCH_KEY], session.scalar(
                select(func.count()).select_from(Report).where(Report.job_id == job.id))

    def barrier():
        result = app.send_task("sf_dispatch_test_barrier", queue=prefix)
        assert result.get(timeout=20) is True

    def recover():
        return tasks.recover_pending_paper_workflows.apply_async(queue=prefix).get(timeout=20)

    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{prefix}"'))
        created = True
        with isolated.connect() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == prefix
        Base.metadata.create_all(isolated)

        unpublished = create_case("committed_without_publication")
        publication_exit = create_case("exit_after_publication_lease")
        stage_exit = create_case("exit_after_stage_start")
        ambiguous = create_case("accepted_publication_without_ack")
        duplicate = create_case("duplicate_delivery")
        busy = create_case("busy_duplicate")
        stale = create_case("stale_attempt")

        for mode, case in (("publication-exit", publication_exit), ("stage-exit", stage_exit)):
            result = subprocess.run([sys.executable, str(bootstrap), mode, *case], env=env,
                                    stdout=log, stderr=log, timeout=20)
            assert result.returncode == 73
        # Accept a real chain, then inject an unavailable acknowledgement to its publisher.
        original_chain = tasks.chain
        class LostAck:
            def __init__(self, *steps):
                self.workflow = original_chain(*steps)
            def apply_async(self):
                self.workflow.apply_async()
                raise ConnectionError("Injected lost publication acknowledgement")
        with monkeypatch.context() as local:
            local.setattr(tasks, "chain", LostAck)
            with pytest.raises(ConnectionError):
                tasks.dispatch_paper_workflow(*ambiguous)
        assert tasks.dispatch_paper_workflow(*ambiguous) is None
        # A fresh publication after the genuine 60-second lease yields a duplicate
        # chain in the same namespace, with no worker yet consuming either copy.
        remaining = max(
            datetime.fromisoformat(job_state(case[0])[1]["dispatch_after"])
            for case in (ambiguous, publication_exit)
        )
        while datetime.now(timezone.utc) <= remaining:
            time.sleep(0.25)
        assert tasks.dispatch_paper_workflow(*ambiguous)
        for _ in range(2):
            tasks.finalize_paper_job_task.apply_async(args=duplicate, queue=prefix)

        # Both old messages must be ignored after a new committed refresh epoch.
        old_attempt = stale[1]
        with factory() as session:
            job = session.get(Job, uuid.UUID(stale[0]))
            new_attempt = prepare_dispatch(job)
            session.commit()
        tasks.check_paper_task.apply_async(args=(stale[0], old_attempt), queue=prefix)
        tasks.finalize_paper_job_task.apply_async(args=(stale[0], old_attempt), queue=prefix)

        worker = subprocess.Popen([sys.executable, str(bootstrap), "worker"], env=env,
                                  stdout=log, stderr=log)
        _wait(lambda: redis.get(prefix + ":ready") == b"1")
        barrier()
        assert job_state(stale[0])[1]["stage_starts"] == {}
        assert job_state(stale[0])[2] == 0

        with factory() as lock_session:
            with tasks._job_execution_lock(lock_session, busy[0], "verify"):
                tasks.finalize_paper_job_task.apply_async(args=busy, queue=prefix)
                barrier()
                assert job_state(busy[0])[0] == JobStatus.RUNNING
                assert job_state(busy[0])[1]["stage_starts"] == {}
                assert job_state(busy[0])[2] == 0

        recovered = recover()
        assert recovered["published"] == 5
        cases = [unpublished, publication_exit, stage_exit, ambiguous, duplicate, busy, stale]
        _wait(lambda: all(job_state(case[0])[0] == JobStatus.COMPLETED for case in cases))
        barrier()
        for name, case in zip(("unpublished", "publication_exit", "stage_exit", "lost_ack",
                               "duplicate", "busy", "stale"), cases):
            status, intent, reports = job_state(case[0])
            assert reports == 1
            expected_starts = 2 if case == stage_exit else 1
            assert intent["stage_starts"] == {"finalize": expected_starts}
            receipt["cases"][name] = {"status": status, "reports": reports,
                                      "stage_starts": expected_starts}
        assert job_state(stale[0])[1]["attempt_id"] == new_attempt
        assert recover()["candidates"] == 0
        receipt["public_data_unchanged"] = _public_snapshot() == before
        assert receipt["public_data_unchanged"]
        receipt["status"] = "passed"
    finally:
        # Only the process we launched is signalled; production workers continue.
        if worker is not None:
            worker.terminate()
            worker.wait(timeout=20)
        log.close()
        isolated.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{prefix}" CASCADE'))
        keys = list(redis.scan_iter(match=prefix + ":*"))
        assert all(key.startswith((prefix + ":").encode()) for key in keys)
        if keys:
            redis.unlink(*keys)
        receipt["test_schema_removed"] = created
        receipt["redis_test_keys_removed"] = len(keys)
        (tmp_path / "scheduling-recovery-receipt.json").write_text(json.dumps(receipt, indent=2))
