"""Transient verification-run lifecycle, cleanup, and audit regressions."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base
from app.models.report import VerificationReportRecord
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.models.verification_run import VerificationRunRecord
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import ClaimEvidence, build_passage_evidence
from app.services.verification_run import (
    VerificationRunAuthorizationError,
    VerificationRunCleanupPending,
    VerificationRunError,
    VerificationRunRequest,
    add_transient_run_artifact,
    begin_verification_run,
    cleanup_stale_verification_runs,
    cleanup_verification_run,
    execute_verification_run,
    load_verification_run_source,
    renew_verification_run_lease,
)
from app.tasks import verification_run_cleanup
from app.tasks.celery_app import celery_app


class MemoryStorage(StorageBackend):
    def __init__(self):
        self.objects = {}
        self.delete_allowed = True

    def upload(self, file_bytes, key):
        self.objects[key] = file_bytes
        return key

    def download(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key):
        if not self.delete_allowed:
            return False
        self.objects.pop(key, None)
        return True

    def exists(self, key):
        return key in self.objects

    def list_keys(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def database():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _request(**changes):
    values = {
        "paper_version_id": "paper-v1",
        "scope_type": "personal_owner",
        "scope_id": "owner-1",
        "canonical_work_id": "work-1",
        "representation": SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT,
            media_type="text/plain",
            content=b"The evidence states that careful verification improves accuracy.",
        ),
        "acquisition_route": "authorized_upload",
        "identity_verdict": "verified",
        "identity_confidence": 0.99,
        "completeness_verdict": "complete",
        "cleanliness_verdict": "clean",
        "text_quality": "digital",
        "edition_or_version": "edition-1",
    }
    values.update(changes)
    return VerificationRunRequest(**values)


def test_possible_match_obeys_scope_hash_and_cleanup_boundaries(database):
    storage = MemoryStorage()
    with database() as session:
        run = begin_verification_run(session, storage, _request(identity_verdict='possible_match'))
        source = load_verification_run_source(session, storage, run.id,
            scope_type='personal_owner', scope_id='owner-1')
        assert source.identity_verdict == 'possible_match'
        with pytest.raises(VerificationRunAuthorizationError):
            load_verification_run_source(session, storage, run.id,
                scope_type='personal_owner', scope_id='other-owner')
        key = next(iter(storage.objects))
        storage.objects[key] = b'changed bytes'
        with pytest.raises(VerificationRunError):
            load_verification_run_source(session, storage, run.id,
                scope_type='personal_owner', scope_id='owner-1')
        cleanup_verification_run(session, storage, run.id,
            scope_type='personal_owner', scope_id='owner-1', outcome='failed')
        assert not storage.objects
        assert session.scalar(select(SourceRepresentationRecord)) is None


def _processor(storage, *, add_derivatives=True):
    def process(session, source, run_id):
        if add_derivatives:
            add_transient_run_artifact(
                session,
                storage,
                run_id,
                scope_type="personal_owner",
                scope_id="owner-1",
                role="extracted_text",
                content=b"normalized transient text",
            )
            add_transient_run_artifact(
                session,
                storage,
                run_id,
                scope_type="personal_owner",
                scope_id="owner-1",
                role="chunks",
                content=b"[]",
            )
            add_transient_run_artifact(
                session,
                storage,
                run_id,
                scope_type="personal_owner",
                scope_id="owner-1",
                role="embeddings",
                content=b"vector-bytes",
            )
        claim_text = "Careful verification improves accuracy (Smith, 2020)."
        claim = ClaimEvidence(
            claim_id="claim-1",
            paper_version_id="paper-v1",
            text=claim_text,
            granularity="atomic_claim",
            atomization_method="test",
            reference_ids=["reference-1"],
            citation_marker="(Smith, 2020)",
            citation_marker_type="parenthetical",
            passage_start=0,
            passage_end=len(claim_text),
        )
        return build_passage_evidence(source, claim=claim)

    return process


def test_success_deletes_all_transient_objects_but_keeps_report_and_audit(database):
    storage = MemoryStorage()
    result = execute_verification_run(
        database, storage, _request(), _processor(storage)
    )

    assert storage.objects == {}
    with database() as session:
        run = session.get(VerificationRunRecord, result.run_id)
        report = session.get(VerificationReportRecord, result.report_id)
        assert run.status == "cleaned"
        assert run.terminal_outcome == "success"
        assert run.transient_objects == []
        assert run.transient_object_count == 4
        assert run.transient_byte_size > run.source_byte_size
        assert run.cleaned_at is not None
        assert report.verification_run_id == run.id
        assert report.report_payload["verification_run_id"] == str(run.id)
        assert report.report_payload["passages"]
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 0


def test_processor_failure_cleans_source_and_derivatives_without_report(database):
    storage = MemoryStorage()

    def fail(session, source, run_id):
        add_transient_run_artifact(
            session,
            storage,
            run_id,
            scope_type="personal_owner",
            scope_id="owner-1",
            role="chunks",
            content=b"temporary chunks",
        )
        raise RuntimeError("synthetic processor failure")

    with pytest.raises(RuntimeError, match="synthetic processor failure"):
        execute_verification_run(database, storage, _request(), fail)

    assert storage.objects == {}
    with database() as session:
        run = session.scalar(select(VerificationRunRecord))
        assert run.status == "failed_cleaned"
        assert run.terminal_outcome == "failed"
        assert run.transient_objects == []
        assert session.scalar(select(func.count(VerificationReportRecord.id))) == 0


def test_initial_upload_failure_keeps_discoverable_uncertain_cleanup(database):
    storage = MemoryStorage()

    def fail_upload(file_bytes, key):
        raise RuntimeError("synthetic upload failure")

    storage.upload = fail_upload
    with database() as session:
        with pytest.raises(RuntimeError, match="synthetic upload failure"):
            begin_verification_run(session, storage, _request())
    with database() as session:
        run = session.scalar(select(VerificationRunRecord))
        assert run.status == "cleanup_pending"
        assert len(run.transient_objects) == 1
        assert run.last_cleanup_error_code == "upload_completion_unresolved"
        assert run.terminal_outcome == "failed"


def test_delete_failure_is_retryable_and_idempotent_after_report(database):
    storage = MemoryStorage()
    storage.delete_allowed = False
    with pytest.raises(VerificationRunCleanupPending) as captured:
        execute_verification_run(
            database, storage, _request(), _processor(storage, add_derivatives=False)
        )
    run_id = captured.value.run_id
    report_id = captured.value.report_id
    assert storage.objects

    with database() as session:
        run = session.get(VerificationRunRecord, run_id)
        assert run.status == "cleanup_pending"
        assert run.last_cleanup_error_code == "object_delete_failed"
        assert session.get(VerificationReportRecord, report_id) is not None

    storage.delete_allowed = True
    with database() as session:
        assert cleanup_verification_run(
            session,
            storage,
            run_id,
            scope_type="personal_owner",
            scope_id="owner-1",
            outcome="success",
        )
        assert cleanup_verification_run(
            session,
            storage,
            run_id,
            scope_type="personal_owner",
            scope_id="owner-1",
            outcome="success",
        )
    assert storage.objects == {}


def test_expired_lease_is_recovered_as_abandoned(database, monkeypatch):
    storage = MemoryStorage()
    monkeypatch.setattr(
        "app.services.verification_run.settings.VERIFICATION_RUN_LEASE_SECONDS", 10
    )
    started = datetime(2026, 8, 16, tzinfo=timezone.utc)
    with database() as session:
        run = begin_verification_run(session, storage, _request(), now=started)
        run_id = run.id
    assert storage.objects

    with database() as session:
        result = cleanup_stale_verification_runs(
            session,
            storage,
            now=started + timedelta(seconds=11),
        )
        run = session.get(VerificationRunRecord, run_id)
        assert result == {"runs_cleaned": 1, "runs_pending": 0}
        assert run.status == "abandoned_cleaned"
        assert run.terminal_outcome == "abandoned"
    assert storage.objects == {}


def test_scheduled_cleanup_rechecks_a_renewed_lease(database):
    storage = MemoryStorage()
    with database() as session:
        run = begin_verification_run(session, storage, _request())
        old_expiry = run.lease_expires_at
        renew_verification_run_lease(session, run.id, scope_type="personal_owner",
            scope_id="owner-1", lease_seconds=100_000)
        assert not cleanup_verification_run(session, storage, run.id,
            outcome="abandoned", now=old_expiry, expired_before=old_expiry)
        assert run.status == "active" and run.cleanup_attempts == 0
        assert storage.objects


def test_busy_cleanup_does_not_change_run_or_delete_bytes(database, monkeypatch):
    from contextlib import contextmanager
    from app.services import verification_run as service
    storage = MemoryStorage()
    with database() as session:
        run = begin_verification_run(session, storage, _request())
        @contextmanager
        def busy(*args):
            raise service.VerificationRunBusy("Synthetic active processor")
            yield  # pragma: no cover
        monkeypatch.setattr(service, "_locked_run_session", busy)
        assert not cleanup_verification_run(session, storage, run.id)
        assert run.status == "active" and run.cleanup_attempts == 0
        assert storage.objects


def test_lost_standalone_owner_does_not_issue_failure_cleanup(database):
    from app.services.workflow_execution import WorkflowOwnershipLost
    storage = MemoryStorage()
    def lost(*args):
        raise WorkflowOwnershipLost("Synthetic owner loss")
    with pytest.raises(WorkflowOwnershipLost):
        execute_verification_run(database, storage, _request(), lost)
    with database() as session:
        run = session.scalar(select(VerificationRunRecord))
        assert run.status == "active" and run.cleanup_attempts == 0
        assert cleanup_stale_verification_runs(session, storage,
            now=run.lease_expires_at + timedelta(seconds=1))["runs_cleaned"] == 1
    assert storage.objects == {}


def test_operation_specific_lease_can_cover_a_bounded_paper_workflow(database):
    storage = MemoryStorage()
    started = datetime(2026, 8, 16, tzinfo=timezone.utc)
    with database() as session:
        run = begin_verification_run(
            session,
            storage,
            _request(),
            now=started,
            lease_seconds=86_400,
        )
        assert run.lease_expires_at == started + timedelta(days=1)

        renewed = renew_verification_run_lease(
            session,
            run.id,
            scope_type="personal_owner",
            scope_id="owner-1",
            now=started + timedelta(hours=12),
            lease_seconds=86_400,
        )
        assert renewed == started + timedelta(days=1, hours=12)


def test_exact_scope_and_active_lease_are_required(database):
    storage = MemoryStorage()
    with database() as session:
        run = begin_verification_run(session, storage, _request())
        with pytest.raises(VerificationRunAuthorizationError):
            load_verification_run_source(
                session,
                storage,
                run.id,
                scope_type="personal_owner",
                scope_id="owner-2",
            )
        assert cleanup_verification_run(
            session,
            storage,
            run.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )
        with pytest.raises(VerificationRunAuthorizationError):
            load_verification_run_source(
                session,
                storage,
                run.id,
                scope_type="personal_owner",
                scope_id="owner-1",
            )


def test_cleanup_never_follows_a_tampered_object_locator(database):
    storage = MemoryStorage()
    storage.objects["durable/do-not-delete.pdf"] = b"durable"
    with database() as session:
        run = begin_verification_run(session, storage, _request())
        run.transient_objects = [
            {
                **run.transient_objects[0],
                "key": f"verification-runs/{run.id}/../durable/do-not-delete.pdf",
            }
        ]
        session.commit()
        assert not cleanup_verification_run(
            session,
            storage,
            run.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )
        assert run.last_cleanup_error_code == "invalid_transient_locator"
    assert storage.objects["durable/do-not-delete.pdf"] == b"durable"


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"identity_verdict": "mismatch"}, "identity is not verified"),
        ({"cleanliness_verdict": "unsafe"}, "cleanliness gate"),
    ],
)
def test_unvalidated_content_never_enters_transient_storage(database, changes, error):
    storage = MemoryStorage()
    with database() as session:
        with pytest.raises(VerificationRunError, match=error):
            begin_verification_run(session, storage, _request(**changes))
        assert session.scalar(select(func.count(VerificationRunRecord.id))) == 0
    assert storage.objects == {}


def test_cleanup_worker_has_schedule_and_recovers_expired_run(database, monkeypatch):
    storage = MemoryStorage()
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    monkeypatch.setattr(
        "app.services.verification_run.settings.VERIFICATION_RUN_LEASE_SECONDS", 10
    )
    with database() as session:
        run = begin_verification_run(session, storage, _request(), now=old)
        run_id = run.id

    schedule = celery_app.conf.beat_schedule["cleanup-stale-verification-runs"]
    assert schedule["task"] == "cleanup_stale_verification_runs"
    assert schedule["schedule"] >= 60
    monkeypatch.setattr(verification_run_cleanup, "SessionLocal", database)
    monkeypatch.setattr(
        verification_run_cleanup, "get_storage_backend", lambda: storage
    )

    result = verification_run_cleanup.cleanup_stale_verification_run_objects.run()

    assert result == {"status": "ok", "runs_cleaned": 1, "runs_pending": 0}
    with database() as session:
        assert session.get(VerificationRunRecord, run_id).status == "abandoned_cleaned"
    assert storage.objects == {}
