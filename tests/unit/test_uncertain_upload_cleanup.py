"""Late source/paper/derivative writes never outlive their cleanup authority."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.models.job import Job
from app.models.source_repository import ContentObjectRecord
from app.models.verification_run import VerificationRunRecord
from app.services import paper_upload, source_repository as sources, source_upload_recovery as recovery
from app.services import verification_run as runs
from app.services.storage.backend import S3Backend
from app.services.upload_completion import UploadReceipt, upload_confirmed, next_check, check_due
from test_source_upload_recovery import Storage, request
from test_verification_run import _request as run_request


class DelayedUpload:
    def __init__(self, backend, boundary):
        self.backend, self.boundary = backend, boundary
        self.pending = None
        self.finished = False
    def __getattr__(self, name):
        return getattr(self.backend, name)
    def upload(self, content, key):
        selected = (key.endswith('/chunks.json') if self.boundary == 'derivative' else
            not key.startswith(('source-upload-intents/', 'source-upload-recovery/')))
        if selected and self.pending is None:
            self.pending = (key, content)
            raise TimeoutError('Synthetic delayed upload acknowledgement')
        return self.backend.upload(content, key)
    def finish(self):
        self.backend.upload(self.pending[1], self.pending[0])
        self.finished = True


@pytest.fixture
def state():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False), Storage()
    engine.dispose()


def exercise_late_write(factory, backend, delayed, boundary, monkeypatch):
    """Shared functional assertions; live tests supply actual delayed HTTP PUT."""
    from botocore.exceptions import ReadTimeoutError
    errors = (TimeoutError, ReadTimeoutError)
    future = datetime.now(timezone.utc) + timedelta(days=2)
    if boundary == 'source':
        with factory() as session:
            with pytest.raises(errors):
                sources.admit_representation(session, delayed, request())
            assert sources.rollback_source_admissions(session, delayed) == 0
        def cleanup(now):
            with factory() as session:
                result = recovery.cleanup_source_upload_intents(session, delayed, now=now)
                assert result['removed'] == 0 and result['pending'] == 1
                assert session.scalar(select(ContentObjectRecord)) is None
            keys = backend.list_keys(recovery.INTENT_PREFIX)
            assert len(keys) == 1
            payload = json.loads(backend.download(keys[0]))
            assert payload['upload_confirmed'] is False
            assert payload['cleanup_status'] == 'upload_completion_unresolved'
    elif boundary == 'paper':
        monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *a, **kw: {})
        with factory() as session:
            with pytest.raises(paper_upload.PaperUploadError):
                paper_upload.create_paper_job(session, delayed, content=b'synthetic paper input',
                    filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='test-owner')
            identifier = session.scalar(select(Job.id))
        def cleanup(now):
            with factory() as session:
                result = paper_upload.cleanup_stale_paper_job_inputs(session, delayed, now=now)
                assert result == {'jobs_cleaned': 0, 'jobs_pending': 1}
                job = session.get(Job, identifier)
                assert job.input_storage_key and job.input_deleted_at is None
                assert job.upload_evidence['input_cleanup_status'] == 'upload_completion_unresolved'
                with pytest.raises(paper_upload.PaperUploadError, match='unavailable'):
                    paper_upload.load_paper_job_input(session, delayed, identifier)
    else:
        with factory() as session:
            if boundary == 'transient':
                with pytest.raises(errors):
                    runs.begin_verification_run(session, delayed, run_request())
                identifier = session.scalar(select(VerificationRunRecord.id))
            else:
                run = runs.begin_verification_run(session, delayed, run_request())
                identifier = run.id
                with pytest.raises(errors):
                    runs.add_transient_run_artifact(session, delayed, identifier,
                        scope_type='personal_owner', scope_id='owner-1', role='chunks', content=b'[]')
                assert not runs.cleanup_verification_run(session, delayed, identifier, outcome='failed')
        def cleanup(now):
            with factory() as session:
                result = runs.cleanup_stale_verification_runs(session, delayed, now=now)
                assert result == {'runs_cleaned': 0, 'runs_pending': 1}
                run = session.get(VerificationRunRecord, identifier)
                assert run.status == 'cleanup_pending'
                assert run.last_cleanup_error_code == 'upload_completion_unresolved'
                assert len(run.transient_objects) == 1
                assert run.transient_objects[0]['upload_confirmed'] is False
                with pytest.raises(runs.VerificationRunAuthorizationError):
                    runs.load_verification_run_source(session, delayed, identifier,
                        scope_type='personal_owner', scope_id='owner-1')
    key = delayed.pending[0]
    cleanup(future)
    assert not backend.exists(key)
    delayed.finish()
    assert backend.exists(key)
    cleanup(future + timedelta(days=2))
    assert not backend.exists(key)
    # Absence and old age still cannot resolve an unacknowledged request.
    cleanup(future + timedelta(days=40))
    assert not backend.exists(key)


@pytest.mark.parametrize('boundary', ['source', 'paper', 'transient', 'derivative'])
def test_late_write_keeps_recovery_owner(state, monkeypatch, boundary):
    factory, backend = state
    exercise_late_write(factory, backend, DelayedUpload(backend, boundary), boundary, monkeypatch)


def test_new_owner_protects_bytes_without_settling_old_uncertain_upload(state):
    factory, backend = state
    delayed = DelayedUpload(backend, 'source')
    future = datetime.now(timezone.utc) + timedelta(days=2)
    with factory() as session:
        with pytest.raises(TimeoutError):
            sources.admit_representation(session, delayed, request())
        sources.rollback_source_admissions(session, delayed)
    with factory() as session:
        record = sources.admit_representation(session, backend, replace(request(), scope_id='new-owner'))
        sources.commit_source_admissions(session)
        identifier = record.id
        new_key = record.content_object.storage_key
    key = delayed.pending[0]
    assert new_key != key
    with factory() as session:
        assert recovery.cleanup_source_upload_intents(session, backend, now=future)['pending'] == 1
        assert backend.exists(new_key) and not backend.exists(key)
        assert sources.delete_representation(session, identifier)
        session.commit()
        assert sources.finalize_pending_object_deletions(session, backend) == 1
        session.commit()
    delayed.finish()
    with factory() as session:
        assert recovery.cleanup_source_upload_intents(session, backend,
            now=future + timedelta(days=2))['pending'] == 1
    assert not backend.exists(key)
    assert len(backend.list_keys(recovery.INTENT_PREFIX)) == 1


@pytest.mark.parametrize('retries,confirmed', [(0, True), (1, False), (3, False), (None, False), (False, False)])
def test_s3_receipt_retains_retry_uncertainty(retries, confirmed):
    storage = object.__new__(S3Backend)
    storage._bucket = 'synthetic-bucket'
    storage._client = Mock()
    storage._client.put_object.return_value = {'ResponseMetadata': {'RetryAttempts': retries}}
    receipt = storage.upload(b'synthetic', 'synthetic-key')
    assert receipt == 'synthetic-key' and upload_confirmed(receipt) is confirmed


def test_success_after_retry_retains_source_watch(state, monkeypatch):
    factory, backend = state
    original = backend.upload
    def retried(content, key):
        result = original(content, key)
        return result if key.startswith(recovery.INTENT_PREFIX) else UploadReceipt(key, confirmed=False)
    monkeypatch.setattr(backend, 'upload', retried)
    with factory() as session:
        record = sources.admit_representation(session, backend, request())
        sources.commit_source_admissions(session)
        assert len(backend.list_keys(recovery.INTENT_PREFIX)) == 1
        assert record.admission_state == 'accepted'


@pytest.mark.parametrize('age,delay', [(0, 300), (2, 3600), (40, 86400)])
def test_watch_checks_slow_down_but_remain_bounded(age, delay):
    now = datetime.now(timezone.utc)
    due = next_check(now - timedelta(days=age), now)
    assert (datetime.fromisoformat(due) - now).total_seconds() == delay
    assert not check_due(due, now)
    assert check_due(due, now + timedelta(seconds=delay))
    assert check_due('malformed', now)
    assert check_due((now + timedelta(days=500)).isoformat(), now)


def test_waiting_paper_watch_does_not_starve_next_expired_input(state, monkeypatch):
    factory, backend = state
    monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *a, **kw: {})
    delayed = DelayedUpload(backend, 'paper')
    now = datetime.now(timezone.utc)
    with factory() as session:
        with pytest.raises(paper_upload.PaperUploadError):
            paper_upload.create_paper_job(session, delayed, content=b'synthetic uncertain input',
                filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='test-owner')
        assert paper_upload.cleanup_stale_paper_job_inputs(session, backend, now=now)['jobs_pending'] == 1
        valid = paper_upload.create_paper_job(session, backend, content=b'synthetic confirmed input',
            filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='test-owner',
            now=now - timedelta(days=2))
        result = paper_upload.cleanup_stale_paper_job_inputs(session, backend, now=now, batch_size=1)
        assert result == {'jobs_cleaned': 1, 'jobs_pending': 0}
        assert valid.input_storage_key is None
