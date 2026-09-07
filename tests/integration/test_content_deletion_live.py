"""Opt-in shared-content deletion races; isolated PostgreSQL and MinIO only."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
import uuid

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services import source_repository as repository
from source_upload_recovery_live_worker import NamespacedStorage, fixture_request
from test_paper_dispatch_broker_live import _public_snapshot


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_CONTENT_DELETION") != "1",
    reason="requires isolated PostgreSQL schema and MinIO prefix for concurrent content deletion")


def _wait_for_lock(engine, pid, future):
    """Observe actual PostgreSQL lock waiting, not just a slow Python thread."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if future.done():
            future.result()  # Preserve an unexpected worker error.
            pytest.fail("Competing mutation did not wait for shared-content ownership")
        with engine.connect() as connection:
            waiting = connection.scalar(text("SELECT wait_event_type FROM pg_stat_activity WHERE pid=:pid"),
                {"pid": pid})
        if waiting == "Lock":
            return
        time.sleep(0.02)
    pytest.fail("Did not observe the isolated contender waiting for its lock")


@pytest.mark.parametrize("case", ["commit_deletes", "rollback_delete", "duplicate_delete",
    "delete_then_admit", "admit_then_delete", "cleanup_then_admit", "expiry_busy", "expiry_renewed"])
def test_shared_content_mutations_are_serialized(monkeypatch, tmp_path, case):
    namespace = "sf_source_upload_test_" + uuid.uuid4().hex
    before = _public_snapshot()
    isolated = create_engine(make_url(settings.DATABASE_URL).update_query_dict(
        {"options": "-csearch_path=" + namespace}))
    factory = sessionmaker(bind=isolated, expire_on_commit=False)
    backend = NamespacedStorage(namespace)
    request = fixture_request("shared-content-" + case)
    receipt = {"namespace": namespace, "case": case, "synthetic_inputs": True, "model_calls": 0}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        Base.metadata.create_all(isolated)
        now = datetime.now(timezone.utc)
        if case.startswith("expiry_"):
            request = replace(request, expires_at=now - timedelta(seconds=1))
        with factory() as session:
            first_record = repository.admit_representation(session, backend, request)
            repository.commit_source_admissions(session)
            first_id, object_id = first_record.id, first_record.content_object_id
            key = session.get(ContentObjectRecord, object_id).storage_key
            second_id = None
            if case in {"commit_deletes", "rollback_delete"}:
                second_record = repository.admit_representation(session, backend, replace(request, scope_id="second-owner"))
                repository.commit_source_admissions(session)
                second_id = second_record.id
                assert second_record.content_object_id == object_id

        if case == "expiry_busy":
            other_request = replace(fixture_request("independently-expired"), expires_at=now - timedelta(seconds=1))
            with factory() as session:
                other = repository.admit_representation(session, backend, other_request)
                repository.commit_source_admissions(session)
                other_id = other.id
            with factory() as owner:
                obj = owner.get(ContentObjectRecord, object_id)
                repository._advisory_transaction_lock(owner, "content-object", f"{obj.license_class}:{obj.content_sha256}")
                with factory() as cleaner:
                    assert repository.expire_representations(cleaner, now=now) == 1
                    cleaner.commit()
                    assert cleaner.get(SourceRepresentationRecord, first_id) is not None
                    assert cleaner.get(SourceRepresentationRecord, other_id) is None
                    assert not cleaner.get(ContentObjectRecord, object_id).deletion_pending
            with factory() as cleaner:
                assert repository.expire_representations(cleaner, now=now) == 1
                cleaner.commit()
                assert repository.finalize_pending_object_deletions(cleaner, backend) == 2
                cleaner.commit()
            assert not backend.exists(key)
        elif case == "expiry_renewed":
            original = repository.delete_representation
            def renew_after_scan(session, selected_id, **kwargs):
                # A real independent transaction must be able to renew the
                # scanned row; expiry must not hold row-before-content locks.
                with factory() as renewing:
                    renewing.execute(text("SET LOCAL statement_timeout = '5s'"))
                    renewed = repository.admit_representation(renewing, backend,
                        replace(request, expires_at=now + timedelta(days=1)))
                    assert renewed.id == selected_id
                    repository.commit_source_admissions(renewing)
                return original(session, selected_id, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(repository, "delete_representation", renew_after_scan)
                with factory() as cleaner:
                    # Retain a stale identity-map instance as the API may do.
                    stale = cleaner.get(SourceRepresentationRecord, first_id)
                    assert repository.expire_representations(cleaner, now=now) == 0
                    cleaner.commit()
                    cleaner.refresh(stale)
                    assert stale.expires_at > now
                    assert not cleaner.get(ContentObjectRecord, object_id).deletion_pending
            assert backend.download(key) == request.representation.content
        else:
            ready = threading.Event()
            contender = {}
            operation = "admit" if case in {"delete_then_admit", "cleanup_then_admit"} else "delete"
            target_id = second_id if second_id is not None else first_id
            def competing():
                with factory() as session:
                    session.execute(text("SET LOCAL statement_timeout = '8s'"))
                    contender["pid"] = session.scalar(text("SELECT pg_backend_pid()"))
                    # Prime the identity map before waiting for the first
                    # deletion; the duplicate case must refresh this state.
                    stale = session.get(SourceRepresentationRecord, target_id)
                    ready.set()
                    if operation == "admit":
                        record = repository.admit_representation(session, backend,
                            replace(request, scope_id="new-owner"))
                        repository.commit_source_admissions(session)
                        return record.id
                    result = repository.delete_representation(session, target_id)
                    session.commit()
                    assert stale is not None
                    return result
            # Close the owning session before joining the executor if any
            # assertion fails, so test teardown never leaves a waiting lock.
            with ThreadPoolExecutor(max_workers=1) as executor, factory() as owner:
                owner.execute(text("SET LOCAL statement_timeout = '8s'"))
                if case == "admit_then_delete":
                    new_record = repository.admit_representation(owner, backend,
                        replace(request, scope_id="new-owner"))
                    new_id = new_record.id
                else:
                    assert repository.delete_representation(owner, first_id)
                    if case == "cleanup_then_admit":
                        owner.commit()
                        assert repository.finalize_pending_object_deletions(owner, backend) == 1
                future = executor.submit(competing)
                assert ready.wait(3), "Contender did not start"
                _wait_for_lock(isolated, contender["pid"], future)
                if case == "rollback_delete":
                    owner.rollback()
                elif case == "admit_then_delete":
                    repository.commit_source_admissions(owner)
                else:
                    owner.commit()
                result = future.result(timeout=10)
                if operation == "delete":
                    assert result is (case != "duplicate_delete")
                else:
                    new_id = result

            with factory() as session:
                count = session.scalar(select(func.count(SourceRepresentationRecord.id)))
                if case in {"commit_deletes", "duplicate_delete"}:
                    assert count == 0
                    assert session.get(ContentObjectRecord, object_id).deletion_pending
                    assert backend.exists(key)
                    # Failed physical deletion retains its committed recovery
                    # record, then a later scheduled attempt can remove it.
                    with monkeypatch.context() as patch:
                        patch.setattr(backend, "delete", lambda key: False)
                        assert repository.finalize_pending_object_deletions(session, backend) == 0
                        session.commit()
                    assert session.get(ContentObjectRecord, object_id).deletion_pending
                    assert repository.finalize_pending_object_deletions(session, backend) == 1
                    session.commit()
                    assert session.get(ContentObjectRecord, object_id) is None
                    assert not backend.exists(key)
                elif case == "rollback_delete":
                    assert count == 1
                    assert session.get(SourceRepresentationRecord, first_id) is not None
                    assert not session.get(ContentObjectRecord, object_id).deletion_pending
                    assert repository.finalize_pending_object_deletions(session, backend) == 0
                    assert backend.download(key) == request.representation.content
                else:
                    assert count == 1
                    live = session.get(SourceRepresentationRecord, new_id)
                    obj = session.get(ContentObjectRecord, live.content_object_id)
                    assert not obj.deletion_pending
                    assert repository.finalize_pending_object_deletions(session, backend) == 0
                    assert backend.download(obj.storage_key) == request.representation.content
        receipt["public_data_unchanged"] = _public_snapshot() == before
        assert receipt["public_data_unchanged"]
        receipt["status"] = "passed"
    finally:
        isolated.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        for key in backend.storage.list_keys(backend.prefix):
            assert key.startswith(namespace + "/")
            assert backend.storage.delete(key)
        assert not backend.storage.list_keys(backend.prefix)
        receipt["test_data_removed"] = True
        (tmp_path / "content-deletion-receipt.json").write_text(json.dumps(receipt, indent=2))
