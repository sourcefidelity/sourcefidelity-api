"""Cleanup must retire the current generation, even with a cached ORM row."""
from dataclasses import replace

import pytest

from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services import source_repository as sources, source_upload_recovery as recovery
from test_uncertain_upload_cleanup import state, request


CASES = ('cached', 'fresh', 'delete_failed', 'false_ack', 'rollback', 'protected')


def exercise_generation_cleanup(factory, backend, monkeypatch, case):
    with factory() as original:
        first = sources.admit_representation(original, backend, request())
        sources.commit_source_admissions(original)
        cached = original.get(ContentObjectRecord, first.content_object_id)
        object_id, old_key = cached.id, cached.storage_key
        original.commit()
        with factory() as other:
            assert sources.delete_representation(other, first.id)
            other.commit()
            new = sources.admit_representation(other, backend,
                replace(request(), scope_id='successor-test-owner'))
            sources.commit_source_admissions(other)
            new_id = new.id
            current_key = other.get(ContentObjectRecord, object_id).storage_key
            assert current_key != old_key
            if case != 'protected':
                assert sources.delete_representation(other, new.id)
                other.commit()
        assert cached.storage_key == old_key
        assert backend.exists(current_key) and not backend.exists(old_key)
        assert not backend.list_keys(recovery.INTENT_PREFIX)
        if case == 'fresh':
            original.expire(cached)

        deleted_keys = []
        delete = backend.delete
        def observed_delete(key):
            deleted_keys.append(key)
            if case == 'delete_failed':
                return False
            if case == 'false_ack':
                return True  # Independent existence check must prevent success.
            return delete(key)
        with monkeypatch.context() as patch:
            patch.setattr(backend, 'delete', observed_delete)
            cleaned = sources.finalize_pending_object_deletions(original, backend)
        if case == 'protected':
            assert cleaned == 0 and deleted_keys == []
            original.commit()
            with factory() as verify:
                assert verify.get(SourceRepresentationRecord, new_id) is not None
                assert not verify.get(ContentObjectRecord, object_id).deletion_pending
            assert backend.download(current_key) == request().representation.content
            return

        assert deleted_keys == [current_key], 'Cleanup used a stale physical target'
        if case in {'delete_failed', 'false_ack'}:
            assert cleaned == 0 and backend.exists(current_key)
            original.commit()
            with factory() as verify:
                tracked = verify.get(ContentObjectRecord, object_id)
                assert tracked.deletion_pending and tracked.storage_key == current_key
            assert sources.finalize_pending_object_deletions(original, backend) == 1
            original.commit()
        elif case == 'rollback':
            assert cleaned == 1 and not backend.exists(current_key)
            original.rollback()
            with factory() as verify:
                tracked = verify.get(ContentObjectRecord, object_id)
                assert tracked.deletion_pending and tracked.storage_key == current_key
            # Storage cannot roll back, but the tombstone still owns recovery.
            assert sources.finalize_pending_object_deletions(original, backend) == 1
            original.commit()
        else:
            assert cleaned == 1
            original.commit()
    with factory() as verify:
        assert verify.get(ContentObjectRecord, object_id) is None
    assert not backend.exists(current_key)
    assert not backend.list_keys(recovery.INTENT_PREFIX)


@pytest.mark.parametrize('case', CASES)
def test_cleanup_refreshes_current_generation(state, monkeypatch, case):
    exercise_generation_cleanup(*state, monkeypatch, case)
