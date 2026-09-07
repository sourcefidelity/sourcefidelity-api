"""Locked mutations must use committed state, not earlier ORM snapshots."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services import source_repository as sources, source_upload_recovery as recovery
from test_source_upload_recovery import Storage, request


DETACH_CASES = ('cached', 'fresh', 'rollback', 'protected', 'expiry', 'delete_failed')
RENEWAL_CASES = ('cached', 'fresh', 'perpetual', 'expired_snapshot', 'longer', 'rollback')


@pytest.fixture(params=[False, True], ids=['retained_session', 'expiring_session'])
def state(request):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(bind=engine, expire_on_commit=request.param, autoflush=False), Storage()
    finally:
        engine.dispose()


def exercise_detach(factory, backend, monkeypatch, case):
    req = request()
    with factory() as setup:
        first = sources.admit_representation(setup, backend, req)
        first_id, object_id = first.id, first.content_object_id
        sources.commit_source_admissions(setup)
        assert sources.delete_representation(setup, first_id)
        setup.commit()
    with factory() as original:
        cached = original.get(ContentObjectRecord, object_id)
        old_key = cached.storage_key
        assert cached.deletion_pending
        with factory() as peer:
            replacement = sources.admit_representation(peer, backend,
                replace(req, scope_id='successor-owner',
                    expires_at=datetime.now(timezone.utc) - timedelta(days=1)))
            replacement_id = replacement.id
            current_key = peer.get(ContentObjectRecord, object_id).storage_key
            if case == 'protected':
                survivor = sources.admit_representation(peer, backend,
                    replace(req, scope_id='surviving-owner'))
                survivor_id = survivor.id
            sources.commit_source_admissions(peer)
        assert current_key != old_key and not backend.exists(old_key)
        assert backend.exists(current_key) and not backend.list_keys(recovery.INTENT_PREFIX)
        assert cached.deletion_pending and cached.storage_key == old_key
        if case == 'fresh':
            original.refresh(cached)
        if case == 'expiry':
            assert sources.expire_representations(original) == 1
        else:
            assert sources.delete_representation(original, replacement_id)
        if case == 'rollback':
            original.rollback()
        else:
            original.commit()
    with factory() as observer:
        tracked = observer.get(ContentObjectRecord, object_id)
        assert tracked.storage_key == current_key
        protected = case in {'rollback', 'protected'}
        assert tracked.deletion_pending is not protected
        assert observer.scalar(select(func.count(SourceRepresentationRecord.id))) == int(protected)
        if protected:
            assert observer.get(SourceRepresentationRecord,
                replacement_id if case == 'rollback' else survivor_id) is not None
            assert sources.finalize_pending_object_deletions(observer, backend) == 0
            observer.commit()
            assert backend.download(current_key) == req.representation.content
            return
        if case == 'delete_failed':
            with monkeypatch.context() as patch:
                patch.setattr(backend, 'delete', lambda key: False)
                assert sources.finalize_pending_object_deletions(observer, backend) == 0
            observer.commit()
            assert observer.get(ContentObjectRecord, object_id).deletion_pending
            assert backend.exists(current_key)
        assert sources.finalize_pending_object_deletions(observer, backend) == 1
        observer.commit()
        assert observer.get(ContentObjectRecord, object_id) is None
        assert not backend.exists(current_key)
        assert not backend.list_keys(recovery.INTENT_PREFIX)


def exercise_renewal(factory, backend, case):
    now = datetime.now(timezone.utc)
    initial = now + timedelta(days=-1 if case == 'expired_snapshot' else 10)
    later = None if case == 'perpetual' else now + timedelta(days=30)
    requested = now + timedelta(days=40 if case == 'longer' else 20)
    req = replace(request(), expires_at=initial, identity_confidence=0.5)
    with factory() as setup:
        first = sources.admit_representation(setup, backend, req)
        record_id, key = first.id, first.content_object.storage_key
        sources.commit_source_admissions(setup)
    with factory() as original:
        cached = original.get(SourceRepresentationRecord, record_id)
        assert sources._as_utc(cached.expires_at) == initial
        with factory() as peer:
            sources.admit_representation(peer, backend,
                replace(req, expires_at=later, identity_confidence=0.9))
            sources.commit_source_admissions(peer)
        with factory() as observer:
            observed = observer.get(SourceRepresentationRecord, record_id).expires_at
            assert (sources._as_utc(observed) if observed else None) == later
        assert sources._as_utc(cached.expires_at) == initial
        if case == 'fresh':
            original.refresh(cached)
        renewed = sources.admit_representation(original, backend,
            replace(req, expires_at=requested))
        assert renewed.id == record_id
        if case == 'rollback':
            original.rollback()
        else:
            sources.commit_source_admissions(original)
    expected = requested if case == 'longer' else later
    with factory() as observer:
        current = observer.get(SourceRepresentationRecord, record_id)
        assert (sources._as_utc(current.expires_at) if current.expires_at else None) == expected
        if case == 'expired_snapshot':
            assert current.identity_confidence == 0.9, 'Stale expiry overwrote current validation'
        assert sources.expire_representations(observer, now=now + timedelta(days=21)) == 0
        observer.commit()
        assert sources.finalize_pending_object_deletions(observer, backend) == 0
        observer.commit()
        assert backend.download(key) == req.representation.content
        # Finite retention still expires normally; perpetual retention does not.
        expired = sources.expire_representations(observer, now=now + timedelta(days=41))
        observer.commit()
        assert expired == int(expected is not None)
        assert sources.finalize_pending_object_deletions(observer, backend) == expired
        observer.commit()
        assert backend.exists(key) is (expected is None)


@pytest.mark.parametrize('case', DETACH_CASES)
def test_last_detach_refreshes_content_state(state, monkeypatch, case):
    exercise_detach(*state, monkeypatch, case)


@pytest.mark.parametrize('case', RENEWAL_CASES)
def test_renewal_refreshes_representation_state(state, case):
    exercise_renewal(*state, case)
