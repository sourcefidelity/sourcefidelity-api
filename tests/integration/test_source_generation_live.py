"""Opt-in ownership and source-generation acceptance; isolated resources only."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/integration'))

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import engine as public_engine
from app.models import Base
from app.models.job import Job
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services import paper_upload, source_repository as sources, source_upload_recovery as recovery
from app.services.paper_dispatch import attempt_id_for
from source_upload_recovery_live_worker import NamespacedStorage, fixture_request
from test_paper_dispatch_broker_live import _public_snapshot


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_SOURCE_GENERATION") != "1",
    reason="requires isolated PostgreSQL/MinIO source-generation acceptance")


@pytest.fixture
def isolated(tmp_path):
    namespace = 'sf_source_upload_test_' + uuid.uuid4().hex
    before = _public_snapshot()
    engine = create_engine(make_url(settings.DATABASE_URL).update_query_dict(
        {'options': '-csearch_path=' + namespace}))
    factory = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    backend = NamespacedStorage(namespace)
    receipt = {'namespace': namespace, 'synthetic_inputs': True, 'model_calls': 0,
               'purpose': 'repair_acceptance'}
    created = False
    try:
        with public_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{namespace}"'))
        created = True
        Base.metadata.create_all(engine)
        yield factory, backend, engine, receipt
    finally:
        engine.dispose()
        if created:
            with public_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{namespace}" CASCADE'))
        for key in backend.storage.list_keys(backend.prefix):
            assert key.startswith(namespace + '/')
            assert backend.storage.delete(key)
        assert not backend.storage.list_keys(backend.prefix)
        receipt['test_data_removed'] = True
        receipt['public_data_unchanged'] = _public_snapshot() == before
        (tmp_path / 'followup-review-receipt.json').write_text(json.dumps(receipt, indent=2))
        assert receipt['public_data_unchanged']


@pytest.mark.parametrize('boundary', ['tombstone', 'upload_intent'])
def test_lost_cleanup_connection_cannot_delete_newly_owned_source(isolated, monkeypatch, boundary):
    factory, backend, engine, receipt = isolated
    request = fixture_request('lost-cleanup-owner-' + boundary)
    with factory() as session:
        first = sources.admit_representation(session, backend, request)
        key = session.get(ContentObjectRecord, first.content_object_id).storage_key
        if boundary == 'tombstone':
            sources.commit_source_admissions(session)
            assert sources.delete_representation(session, first.id)
            session.commit()
        else:
            session.rollback()  # Retain the confirmed journal, as after process exit.
    original_delete = backend.delete
    new_owner = {}
    with factory() as cleaner:
        def delayed_delete(selected_key):
            if selected_key == key and not new_owner:
                new_owner['intervened'] = True
                # Invalidate only the explicitly isolated cleanup connection.
                cleaner.connection().invalidate()
                with factory() as successor:
                    successor.execute(text("SET LOCAL statement_timeout = '5s'"))
                    record = sources.admit_representation(successor, backend,
                        replace(request, scope_id='new-isolated-owner'))
                    sources.commit_source_admissions(successor)
                    new_owner['id'] = record.id
            return original_delete(selected_key)
        with monkeypatch.context() as patch:
            patch.setattr(backend, 'delete', delayed_delete)
            if boundary == 'tombstone':
                with pytest.raises(PendingRollbackError):
                    sources.finalize_pending_object_deletions(cleaner, backend)
                cleaner.rollback()
            else:
                outcome = recovery.cleanup_source_upload_intents(cleaner, backend,
                    now=datetime.now(timezone.utc) + timedelta(hours=2))
                assert outcome['unavailable'] == 1
    with factory() as session:
        record = session.get(SourceRepresentationRecord, new_owner['id'])
        assert record is not None and record.admission_state == 'accepted'
        obj = session.get(ContentObjectRecord, record.content_object_id)
        assert obj is not None and not obj.deletion_pending
        assert obj.storage_key != key
        assert backend.download(obj.storage_key) == request.representation.content
    receipt.update(case=boundary, regression_passed=True,
        accepted_new_owner=True, accepted_source_bytes_preserved=True,
        fault='isolated_cleanup_connection_invalidated_before_delayed_delete')


@pytest.mark.parametrize('upload_error', [False, True])
def test_lost_paper_intake_owner_cannot_erase_cleanup_barrier(isolated, monkeypatch, upload_error):
    factory, backend, engine, receipt = isolated
    monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *a, **kw: {})
    owners = []
    def observe(connection, cursor, statement, parameters, context, executemany):
        if 'SELECT pg_try_advisory_lock' in statement:
            owners.append(connection)
    event.listen(engine, 'after_cursor_execute', observe)
    original_upload = backend.upload
    observed = {}
    def delayed_upload(content, key):
        owners[0].invalidate()  # Only this isolated intake lock.
        with factory() as cleaner:
            result = paper_upload.cleanup_stale_paper_job_inputs(cleaner, backend,
                now=datetime.now(timezone.utc) + timedelta(days=2))
            assert result == {'jobs_cleaned': 0, 'jobs_pending': 1}
            job = cleaner.scalar(select(Job))
            assert job.status == 'failed' and job.upload_evidence['input_cleanup_started']
            observed['cleaner_disabled_input'] = True
        if upload_error:
            raise TimeoutError('Synthetic upload timeout after ownership loss')
        return original_upload(content, key)
    try:
        with monkeypatch.context() as patch, factory() as session:
            patch.setattr(backend, 'upload', delayed_upload)
            with pytest.raises(paper_upload.PaperUploadError) as error:
                paper_upload.create_paper_job(session, backend, content=b'synthetic input bytes',
                    filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='isolated-owner')
            assert error.value.code == 'storage_unavailable'
            job_id = session.scalar(select(Job.id))
    finally:
        event.remove(engine, 'after_cursor_execute', observe)
    with factory() as session:
        job = session.get(Job, job_id)
        assert job.status == 'failed'
        assert job.upload_evidence['paper_input_upload_state'] == 'pending'
        assert job.upload_evidence['input_cleanup_started'] is True
        assert not attempt_id_for(job)
        with pytest.raises(paper_upload.PaperUploadError):
            paper_upload.load_paper_job_input(session, backend, job_id)
        assert paper_upload.cleanup_stale_paper_job_inputs(session, backend,
            now=datetime.now(timezone.utc) + timedelta(days=4))['jobs_pending'] == 1
        assert not backend.exists(job.input_storage_key)
    receipt.update(case='paper_intake', upload_error=upload_error, regression_passed=True,
        cleanup_barrier_preserved=True, failed_input_readable=False,
        stale_intake_returned_pending=False, **observed)


def test_lost_scheduled_cleanup_cannot_overwrite_successor(isolated, monkeypatch):
    factory, backend, engine, receipt = isolated
    monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *a, **kw: {})
    with factory() as session:
        job = paper_upload.create_paper_job(session, backend, content=b'synthetic cleanup bytes',
            filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='isolated-owner')
        job_id, key = job.id, job.input_storage_key
        job.status = 'failed'
        session.commit()
    owners = []
    def observe(connection, cursor, statement, parameters, context, executemany):
        if 'SELECT pg_try_advisory_lock' in statement:
            owners.append(connection)
    event.listen(engine, 'after_cursor_execute', observe)
    original_delete = backend.delete
    peer = {}
    def delayed_delete(selected):
        if selected == key and not peer:
            peer['started'] = True
            owners[0].invalidate()
            with factory() as successor:
                assert paper_upload.cleanup_stale_paper_job_inputs(successor, backend)['jobs_cleaned'] == 1
                job = successor.get(Job, job_id)
                peer['deleted_at'] = job.input_deleted_at
                peer['evidence'] = dict(job.upload_evidence)
        return original_delete(selected)
    try:
        with monkeypatch.context() as patch, factory() as session:
            patch.setattr(backend, 'delete', delayed_delete)
            assert paper_upload.cleanup_stale_paper_job_inputs(session, backend) == {
                'jobs_cleaned': 0, 'jobs_pending': 1}
    finally:
        event.remove(engine, 'after_cursor_execute', observe)
    with factory() as session:
        job = session.get(Job, job_id)
        assert job.input_storage_key is None and job.input_deleted_at == peer['deleted_at']
        assert job.upload_evidence == peer['evidence']
        assert not backend.exists(key)
    receipt.update(case='paper_cleanup', regression_passed=True, successor_preserved=True)
