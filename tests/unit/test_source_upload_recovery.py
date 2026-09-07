"""Recorded orphan cleanup must outlive sessions and fail closed on uncertainty."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

from botocore.exceptions import ClientError
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.models.source_repository import ContentObjectRecord
from app.services import source_upload_recovery as recovery
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.source_repository import (
    AdmissionRequest, WorkIdentity, admit_representation, commit_source_admissions,
    rollback_source_admissions,
)
from app.services.storage.backend import S3Backend


class Storage:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.deletes = []
        self.delete_allowed = True
    def upload(self, content, key):
        self.writes.append(key)
        self.objects[key] = content
        return key
    def download(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]
    def exists(self, key):
        return key in self.objects
    def delete(self, key):
        self.deletes.append(key)
        if not self.delete_allowed:
            return False
        self.objects.pop(key, None)
        return True
    def list_keys(self, prefix):
        assert prefix == recovery.INTENT_PREFIX
        return sorted(key for key in self.objects if key.startswith(prefix))


def request():
    return AdmissionRequest(
        work=WorkIdentity(title="Synthetic lifecycle fixture", work_type="journal_article"),
        representation=SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT,
                                             media_type="text/plain", content=b"synthetic source bytes"),
        provenance="instructor_upload", license_class="commercial_user_upload",
        scope_type="personal_owner", scope_id="test-owner", identity_verdict="match",
        completeness_verdict="complete", cleanliness_verdict="clean",
    )


@pytest.fixture
def state():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory, Storage()
    engine.dispose()


def orphan(factory, storage):
    with factory() as session:
        admit_representation(session, storage, request())
        session.rollback()
        session.info.clear()  # No in-memory provenance remains.
    return storage.writes[0], storage.writes[1]


def cleanup(factory, storage, *, expired=True):
    with factory() as session:
        return recovery.cleanup_source_upload_intents(
            session, storage, now=datetime.now(timezone.utc) + timedelta(hours=2 if expired else 0))


def test_intent_precedes_source_bytes_and_contains_no_paper_or_source_text(state):
    factory, storage = state
    intent, source = orphan(factory, storage)
    assert intent.startswith(recovery.INTENT_PREFIX)
    payload = json.loads(storage.objects[intent])
    assert payload["storage_key"] == source
    assert set(payload) == {"version", "id", "storage_key", "license_class", "content_sha256", "kind", "created_at", "expires_at", "upload_confirmed", "generation", "purpose"}
    assert payload["version"] == 2 and payload["purpose"] == "upload"
    assert payload["upload_confirmed"] is True
    assert b"synthetic source bytes" not in storage.objects[intent]


def test_rolled_back_source_recovers_without_original_session(state):
    factory, storage = state
    orphan(factory, storage)
    assert cleanup(factory, storage)["removed"] == 1
    assert storage.objects == {}


def test_failed_rollback_delete_keeps_persistent_retry(state):
    factory, storage = state
    with factory() as session:
        admit_representation(session, storage, request())
        storage.delete_allowed = False
        assert rollback_source_admissions(session, storage) == 0
        assert session.scalar(select(ContentObjectRecord)) is None
    assert len(storage.objects) == 2
    storage.delete_allowed = True
    assert cleanup(factory, storage)["removed"] == 1
    assert storage.objects == {}


def test_lease_prevents_premature_cleanup(state):
    factory, storage = state
    orphan(factory, storage)
    assert cleanup(factory, storage, expired=False)["pending"] == 1
    assert storage.deletes == []


def test_raw_database_commit_survives_exit_before_journal_cleanup(state):
    factory, storage = state
    with factory() as session:
        record = admit_representation(session, storage, request())
        session.commit()  # Simulates exit before post-commit compensation release.
        source = record.content_object.storage_key
    assert cleanup(factory, storage)["protected"] == 1
    assert list(storage.objects) == [source]


def test_normal_commit_removes_only_recovery_record(state):
    factory, storage = state
    with factory() as session:
        admit_representation(session, storage, request())
        commit_source_admissions(session)
    assert len(storage.objects) == 1
    assert storage.writes[1] in storage.objects
    assert storage.deletes == [storage.writes[0]]


def test_new_admission_protects_bytes_from_old_rollback_intent(state):
    factory, storage = state
    old_intent, source = orphan(factory, storage)
    with factory() as session:
        record = admit_representation(session, storage, request())
        new_key = record.content_object.storage_key
        commit_source_admissions(session)
    assert old_intent in storage.objects
    assert new_key != source
    assert cleanup(factory, storage)["removed"] == 1
    assert source not in storage.objects and new_key in storage.objects


def test_no_intent_means_no_cleanup_authority(state):
    factory, storage = state
    storage.objects["commercial_user_upload/historical-unreferenced.txt"] = b"preserve"
    assert cleanup(factory, storage)["inspected"] == 0
    assert storage.deletes == []


def test_bounded_pages_advance_past_invalid_intents_across_sessions(state):
    factory, storage = state
    invalid = recovery.INTENT_PREFIX + "0" * 32 + ".json"
    storage.objects[invalid] = b"invalid"
    _intent, source = orphan(factory, storage)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    with factory() as session:
        assert recovery.cleanup_source_upload_intents(session, storage, now=future, batch_size=1)["invalid"] == 1
    with factory() as session:
        assert recovery.cleanup_source_upload_intents(session, storage, now=future, batch_size=1)["removed"] == 1
    assert source not in storage.objects
    assert invalid in storage.objects


def test_s3_cleanup_listing_is_prefix_bound_and_paginated():
    storage = object.__new__(S3Backend)
    storage._bucket = "test"
    storage._client = Mock()
    storage._client.list_objects_v2.return_value = {"Contents": [{"Key": "prefix/next"}]}
    assert storage.list_keys_page("prefix/", start_after="prefix/last", limit=25) == ["prefix/next"]
    storage._client.list_objects_v2.assert_called_once_with(
        Bucket="test", Prefix="prefix/", StartAfter="prefix/last", MaxKeys=25)


@pytest.mark.parametrize("field,value", [("storage_key", "../other-source"), ("content_sha256", "bad"),
                                         ("version", 3), ("expires_at", "bad"), ("id", "wrong"),
                                         ("generation", "../bad"), ("generation", None),
                                         ("purpose", "unknown")])
def test_invalid_intent_cannot_authorize_deletion(state, field, value):
    factory, storage = state
    intent, _source = orphan(factory, storage)
    payload = json.loads(storage.objects[intent])
    payload[field] = value
    storage.objects[intent] = json.dumps(payload).encode()
    assert cleanup(factory, storage)["invalid"] == 1
    assert storage.deletes == []


def test_false_success_or_uncertain_absence_keeps_intent(state, monkeypatch):
    factory, storage = state
    intent, source = orphan(factory, storage)
    monkeypatch.setattr(storage, "delete", lambda key: True)
    assert cleanup(factory, storage)["pending"] == 1
    assert intent in storage.objects and source in storage.objects
    monkeypatch.setattr(storage, "exists", Mock(side_effect=ConnectionError("storage unavailable")))
    assert cleanup(factory, storage)["unavailable"] == 1
    assert intent in storage.objects


def test_failed_journal_deletion_after_source_delete_is_retryable(state, monkeypatch):
    factory, storage = state
    intent, source = orphan(factory, storage)
    delete = storage.delete
    with monkeypatch.context() as local:
        local.setattr(storage, "delete", lambda key: False if key == intent else delete(key))
        assert cleanup(factory, storage)["pending"] == 1
    assert source not in storage.objects and intent in storage.objects
    assert cleanup(factory, storage)["removed"] == 1
    assert storage.objects == {}


@pytest.mark.parametrize("fail_after_intent", [False, True])
def test_ambiguous_put_failure_never_leaves_unrecorded_bytes(state, monkeypatch, fail_after_intent):
    factory, storage = state
    upload = storage.upload
    def uncertain(content, key):
        upload(content, key)
        if key.startswith(recovery.INTENT_PREFIX) != fail_after_intent:
            raise ConnectionError("PUT acknowledgement unavailable")
        return key
    monkeypatch.setattr(storage, "upload", uncertain)
    with factory() as session:
        with pytest.raises(ConnectionError):
            admit_representation(session, storage, request())
        session.rollback()
    assert len(storage.writes) == (2 if fail_after_intent else 1)
    result = cleanup(factory, storage)
    assert result["pending"] + result["unavailable"] == 1
    assert all(key.startswith(recovery.INTENT_PREFIX) for key in storage.objects)


@pytest.mark.parametrize("code,absent", [("404", True), ("NoSuchKey", True), ("403", False), ("500", False)])
def test_s3_absence_is_not_inferred_from_permissions_or_operational_failure(code, absent):
    storage = object.__new__(S3Backend)
    storage._bucket = "test"
    storage._client = Mock()
    storage._client.head_object.side_effect = ClientError({"Error": {"Code": code}}, "HeadObject")
    if absent:
        assert storage.exists("test-key") is False
    else:
        with pytest.raises(ClientError):
            storage.exists("test-key")
