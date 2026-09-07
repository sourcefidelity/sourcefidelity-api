"""Generation changes retain old cleanup authority without reusing targets."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlalchemy import select

from app.models.source_repository import ContentObjectRecord
from app.services import source_repository as sources, source_upload_recovery as recovery
from app.services.upload_completion import UploadReceipt
from test_uncertain_upload_cleanup import state, request


@pytest.mark.parametrize('commit', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_rotation_preserves_recovery_across_transaction_exit(state, commit, legacy):
    factory, backend = state
    with factory() as session:
        record = sources.admit_representation(session, backend, request())
        sources.commit_source_admissions(session)
        obj = session.get(ContentObjectRecord, record.content_object_id)
        if legacy:
            previous = obj.storage_key
            obj.storage_key = sources._object_key(obj.license_class, obj.content_sha256,
                request().representation.kind)
            backend.upload(request().representation.content, obj.storage_key)
            assert backend.delete(previous)
            session.commit()
        old_key, object_id = obj.storage_key, obj.id
        assert sources.delete_representation(session, record.id)
        session.commit()
        record = sources.admit_representation(session, backend, replace(request(), scope_id='new-owner'))
        new_key = session.get(ContentObjectRecord, record.content_object_id).storage_key
        assert new_key != old_key
        assert backend.exists(old_key) and backend.exists(new_key)
        intents = [json.loads(backend.download(key)) for key in backend.list_keys(recovery.INTENT_PREFIX)]
        assert {item['purpose'] for item in intents} == {'upload', 'retirement'}
        if commit:
            session.commit()  # Exit before immediate journal settlement.
        else:
            session.rollback()
    with factory() as session:
        result = recovery.cleanup_source_upload_intents(session, backend,
            now=datetime.now(timezone.utc) + timedelta(hours=2))
        assert result['removed'] == result['protected'] == 1
        obj = session.get(ContentObjectRecord, object_id)
        assert obj.storage_key == (new_key if commit else old_key)
        assert backend.exists(obj.storage_key)
        assert not backend.exists(old_key if commit else new_key)
        assert not backend.list_keys(recovery.INTENT_PREFIX)


def test_new_generation_can_replace_already_absent_tombstone(state):
    factory, backend = state
    with factory() as session:
        record = sources.admit_representation(session, backend, request())
        sources.commit_source_admissions(session)
        old = record.content_object.storage_key
        assert sources.delete_representation(session, record.id)
        session.commit()
        assert backend.delete(old)
        record = sources.admit_representation(session, backend, request())
        sources.commit_source_admissions(session)
        assert record.content_object.storage_key != old
        assert backend.download(record.content_object.storage_key) == request().representation.content


@pytest.mark.parametrize('confirmed', [False, True])
def test_legacy_upload_intents_keep_original_completion_rules(state, confirmed):
    factory, backend = state
    req = request()
    import hashlib
    digest = hashlib.sha256(req.representation.content).hexdigest()
    key = sources._object_key(req.license_class, digest, req.representation.kind)
    with factory() as session:
        intent = recovery.persist_upload_intent(session, backend, storage_key=key,
            license_class=req.license_class, digest=digest, kind=req.representation.kind)
        receipt = backend.upload(req.representation.content, key)
        if confirmed:
            recovery.confirm_upload_intent(backend, intent, receipt)
        result = recovery.cleanup_source_upload_intents(session, backend,
            now=datetime.now(timezone.utc) + timedelta(hours=2))
        assert result['removed' if confirmed else 'pending'] == 1
        assert not backend.exists(key)
        assert backend.exists(intent) is not confirmed


def test_retirement_does_not_settle_an_earlier_uncertain_upload(state, monkeypatch):
    factory, backend = state
    upload = backend.upload
    def uncertain(content, key):
        result = upload(content, key)
        return result if key.startswith(recovery.INTENT_PREFIX) else UploadReceipt(key, confirmed=False)
    with factory() as session:
        with monkeypatch.context() as patch:
            patch.setattr(backend, 'upload', uncertain)
            record = sources.admit_representation(session, backend, request())
            sources.commit_source_admissions(session)
        old_key = record.content_object.storage_key
        old_watch = backend.list_keys(recovery.INTENT_PREFIX)[0]
        assert sources.delete_representation(session, record.id)
        session.commit()
        record = sources.admit_representation(session, backend, request())
        sources.commit_source_admissions(session)
        new_key = record.content_object.storage_key
        assert new_key != old_key
        assert backend.list_keys(recovery.INTENT_PREFIX) == [old_watch]
        assert not backend.exists(old_key)
        # A late original PUT is still owned by its separate unresolved watch.
        upload(request().representation.content, old_key)
        assert recovery.cleanup_source_upload_intents(session, backend,
            now=datetime.now(timezone.utc) + timedelta(days=2))['pending'] == 1
        assert not backend.exists(old_key)
        assert backend.download(new_key) == request().representation.content
        assert backend.exists(old_watch)
