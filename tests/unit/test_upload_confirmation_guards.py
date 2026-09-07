"""Missing, retried, and interrupted completion records are not proof of absence."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from app.models.job import JobStatus
from app.services import paper_upload, source_repository as sources, source_upload_recovery as recovery
from app.services import verification_run as runs
from app.services.upload_completion import UploadReceipt
from test_uncertain_upload_cleanup import state, request, run_request


@pytest.mark.parametrize('after_write', [False, True])
def test_completion_record_interruption_preserves_only_proven_state(state, monkeypatch, after_write):
    factory, backend = state
    original = backend.upload
    metadata_writes = 0
    def interrupted(content, key):
        nonlocal metadata_writes
        if key.startswith(recovery.INTENT_PREFIX):
            metadata_writes += 1
            if metadata_writes == 2:
                if after_write:
                    original(content, key)
                raise ConnectionError('Synthetic completion-record interruption')
        return original(content, key)
    monkeypatch.setattr(backend, 'upload', interrupted)
    with factory() as session:
        with pytest.raises(ConnectionError):
            sources.admit_representation(session, backend, request())
        assert sources.rollback_source_admissions(session, backend) == int(after_write)
    watches = backend.list_keys(recovery.INTENT_PREFIX)
    assert len(watches) == (0 if after_write else 1)
    assert all(key.startswith(recovery.INTENT_PREFIX) for key in backend.objects)


@pytest.mark.parametrize('boundary', ['source', 'paper', 'transient'])
def test_legacy_missing_confirmation_keeps_cleanup_authority(state, monkeypatch, boundary):
    factory, backend = state
    future = datetime.now(timezone.utc) + timedelta(days=2)
    with factory() as session:
        if boundary == 'source':
            sources.admit_representation(session, backend, request())
            session.rollback()
            intent = backend.list_keys(recovery.INTENT_PREFIX)[0]
            payload = json.loads(backend.download(intent))
            payload.pop('upload_confirmed')
            backend.upload(json.dumps(payload).encode(), intent)
            assert recovery.cleanup_source_upload_intents(session, backend, now=future)['pending'] == 1
            assert backend.exists(intent) and not backend.exists(payload['storage_key'])
        elif boundary == 'paper':
            monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *args, **kw: {})
            job = paper_upload.create_paper_job(session, backend, content=b'synthetic legacy input',
                filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='test-owner')
            key = job.input_storage_key
            job.upload_evidence = {name: value for name, value in job.upload_evidence.items()
                if name != 'input_upload_confirmed'}
            session.commit()
            assert paper_upload.cleanup_stale_paper_job_inputs(session, backend, now=future)['jobs_pending'] == 1
            assert job.input_storage_key == key and not backend.exists(key)
        else:
            run = runs.begin_verification_run(session, backend, run_request())
            key = run.transient_objects[0]['key']
            run.transient_objects = [{name: value for name, value in item.items() if name != 'upload_confirmed'}
                for item in run.transient_objects]
            session.commit()
            assert not runs.cleanup_verification_run(session, backend, run.id)
            assert run.transient_objects and not backend.exists(key)


@pytest.mark.parametrize('boundary', ['paper', 'transient'])
def test_success_after_retry_is_usable_but_cleanup_remains_uncertain(state, monkeypatch, boundary):
    factory, backend = state
    original = backend.upload
    def retried(content, key):
        original(content, key)
        return UploadReceipt(key, confirmed=False)
    monkeypatch.setattr(backend, 'upload', retried)
    with factory() as session:
        if boundary == 'paper':
            monkeypatch.setattr(paper_upload, 'inspect_paper_upload', lambda *args, **kw: {})
            job = paper_upload.create_paper_job(session, backend, content=b'synthetic retried input',
                filename='synthetic.pdf', media_type=paper_upload.PDF_MEDIA_TYPE, scope_id='test-owner')
            assert paper_upload.load_paper_job_input(session, backend, job.id)[1] == b'synthetic retried input'
            job.status = JobStatus.FAILED
            session.commit()
            assert not paper_upload.cleanup_paper_job_input(session, backend, job.id)
            assert job.input_storage_key and job.input_deleted_at is None
        else:
            run = runs.begin_verification_run(session, backend, run_request())
            assert runs.load_verification_run_source(session, backend, run.id,
                scope_type='personal_owner', scope_id='owner-1')
            assert not runs.cleanup_verification_run(session, backend, run.id)
            assert run.transient_objects and run.status == 'cleanup_pending'
