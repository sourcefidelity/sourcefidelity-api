"""Durable source-representation admission and deletion tests."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models import Base
from app.models.source_repository import (
    CanonicalWorkRecord,
    ContentObjectRecord,
    SourceRepresentationRecord,
)
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.source_repository import (
    AdmissionError,
    AdmissionRequest,
    WorkIdentity,
    admit_derived_representation_pair,
    admit_representation,
    delete_representation,
    expire_representations,
    finalize_pending_object_deletions,
    find_accepted_representation,
    retention_mode_for_scope,
    representation_is_expired,
)
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import (
    EvidenceAuthorizationError,
    authorize_representation,
)


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_count = 0

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.upload_count += 1
        self.objects[key] = file_bytes
        return key

    def download(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


class TemporarilyFailingDeleteStorage(MemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.delete_allowed = False

    def delete(self, key: str) -> bool:
        if not self.delete_allowed:
            return False
        return super().delete(key)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _request(
    *,
    scope_type: str = "personal_owner",
    scope_id: str = "owner-1",
    identity_verdict: str = "match",
    completeness_verdict: str = "complete",
    cleanliness_verdict: str = "clean",
    expires_at: datetime | None = None,
) -> AdmissionRequest:
    return AdmissionRequest(
        work=WorkIdentity(
            title="A Durable Academic Source",
            work_type="journal_article",
            doi="https://doi.org/10.1234/EXAMPLE",
            author="A. Scholar",
            year="2025",
        ),
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=b"%PDF-1.7 durable source bytes",
            source_url="https://repository.example/source.pdf",
        ),
        provenance="instructor_upload",
        license_class="commercial_user_upload",
        scope_type=scope_type,
        scope_id=scope_id,
        identity_verdict=identity_verdict,
        identity_confidence=0.99,
        completeness_verdict=completeness_verdict,
        cleanliness_verdict=cleanliness_verdict,
        text_quality="digital",
        admitted_by="owner-1",
        expires_at=expires_at,
        validation_evidence={"pdf_identity": "doi_match"},
    )


def test_admission_uses_immutable_license_hash_key(session: Session) -> None:
    storage = MemoryStorage()
    record = admit_representation(session, storage, _request())
    session.commit()

    digest = hashlib.sha256(b"%PDF-1.7 durable source bytes").hexdigest()
    content = session.get(ContentObjectRecord, record.content_object_id)
    work = session.get(CanonicalWorkRecord, record.canonical_work_id)

    assert record.admission_state == "accepted"
    assert record.admitted_at is not None
    assert content is not None
    prefix = f"commercial_user_upload/{digest}/"
    assert content.storage_key.startswith(prefix) and content.storage_key.endswith(".pdf")
    assert len(content.storage_key.removeprefix(prefix).removesuffix(".pdf")) == 32
    assert storage.objects[content.storage_key].startswith(b"%PDF-")
    assert work is not None
    assert work.doi == "10.1234/example"
    assert record.validation_evidence["retention_mode"] == "durable"
    assert record.validation_evidence["authorization_scope"] == {
        "type": "personal_owner",
        "id": "owner-1",
    }


def test_ocr_pair_retains_parent_but_accepts_only_derivative(session: Session) -> None:
    storage = MemoryStorage()
    base = _request()
    parent_content = b"%PDF-1.7 immutable pure scan parent"
    derivative_content = b"OCR derivative source text"
    parent_sha256 = hashlib.sha256(parent_content).hexdigest()
    derivative_sha256 = hashlib.sha256(derivative_content).hexdigest()
    parent_request = AdmissionRequest(
        **{
            **base.__dict__,
            "representation": SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=parent_content,
            ),
            "provenance": "local_ocr_parent",
            "text_quality": "pure_scan",
            "request_acceptance": False,
        }
    )
    derivative_request = AdmissionRequest(
        **{
            **base.__dict__,
            "representation": SourceRepresentation(
                kind=RepresentationKind.PLAIN_TEXT,
                media_type="text/plain",
                content=derivative_content,
                original_kind=RepresentationKind.PDF,
                completeness="complete",
            ),
            "provenance": "local_ocr_derivative",
            "identity_confidence": 0.8,
            "text_quality": "scan_ocr",
            "validation_evidence": {
                "ocr_derivative": {
                    "derivation_method": "local-pdf-ocr-derivative-v1",
                    "parent_content_sha256": parent_sha256,
                    "derivative_content_sha256": derivative_sha256,
                    "derivation_manifest_sha256": "b" * 64,
                    "page_labels": ["1"],
                }
            },
        }
    )

    parent, derivative = admit_derived_representation_pair(
        session,
        storage,
        parent_request=parent_request,
        derivative_request=derivative_request,
    )
    session.commit()

    assert parent.admission_state == "needs_review"
    assert derivative.admission_state == "accepted"
    assert derivative.original_kind == "pdf"
    assert (
        derivative.validation_evidence["ocr_derivative"][
            "parent_representation_id"
        ]
        == str(parent.id)
    )
    found = find_accepted_representation(
        session,
        scope_type="personal_owner",
        scope_id="owner-1",
        doi="10.1234/example",
    )
    assert found is not None and found.id == derivative.id
    authorized = authorize_representation(
        session,
        storage,
        representation_id=derivative.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    assert authorized.parent_content_sha256 == parent_sha256
    assert authorized.page_labels == ("1",)

    storage.delete(parent.content_object.storage_key)
    with pytest.raises(EvidenceAuthorizationError, match="immutable parent"):
        authorize_representation(
            session,
            storage,
            representation_id=derivative.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )


def test_durable_lookup_does_not_reuse_incompatible_work_type(session: Session) -> None:
    storage = MemoryStorage()
    admitted = admit_representation(session, storage, _request())
    session.commit()

    assert find_accepted_representation(
        session,
        scope_type="personal_owner",
        scope_id="owner-1",
        doi="10.1234/example",
        work_type="journal_article",
    ).id == admitted.id
    assert find_accepted_representation(
        session,
        scope_type="personal_owner",
        scope_id="owner-1",
        doi="10.1234/example",
        work_type="monograph",
    ) is None


def test_admission_rejects_identifier_collision_across_incompatible_types(
    session: Session,
) -> None:
    storage = MemoryStorage()
    admit_representation(session, storage, _request())
    session.commit()
    conflicting = _request()
    conflicting = AdmissionRequest(
        **{
            **conflicting.__dict__,
            "work": WorkIdentity(
                title="A Durable Academic Source",
                work_type="monograph",
                doi="10.1234/example",
                author="A. Scholar",
                year="2025",
            ),
            "representation": SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=b"%PDF-conflicting book bytes",
            ),
        }
    )

    with pytest.raises(AdmissionError, match="incompatible work type"):
        admit_representation(session, storage, conflicting)

    assert storage.upload_count == 3  # Intent, source, and persisted completion.


@pytest.mark.parametrize("scope_type", ["assessment", "course_offering"])
def test_bounded_retention_requires_expiry(
    session: Session,
    scope_type: str,
) -> None:
    with pytest.raises(AdmissionError, match="requires an explicit expiry"):
        admit_representation(
            session,
            MemoryStorage(),
            _request(scope_type=scope_type, scope_id="bounded-1"),
        )


def test_verification_run_cannot_enter_durable_admission(session: Session) -> None:
    with pytest.raises(AdmissionError, match="must remain ephemeral"):
        admit_representation(
            session,
            MemoryStorage(),
            _request(scope_type="verification_run", scope_id="run-1"),
        )

    assert find_accepted_representation(
        session,
        scope_type="verification_run",
        scope_id="run-1",
        doi="10.1234/example",
    ) is None


def test_unknown_scope_fails_closed(session: Session) -> None:
    with pytest.raises(AdmissionError, match="Unsupported source authorization scope"):
        admit_representation(
            session,
            MemoryStorage(),
            _request(scope_type="all_courses", scope_id="unsafe-global-scope"),
        )

    assert find_accepted_representation(
        session,
        scope_type="all_courses",
        scope_id="unsafe-global-scope",
        doi="10.1234/example",
    ) is None


def test_assessment_and_course_scopes_are_isolated(session: Session) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    assessment = admit_representation(
        session,
        storage,
        _request(
            scope_type="assessment",
            scope_id="assessment-1",
            expires_at=now + timedelta(days=30),
        ),
    )
    course = admit_representation(
        session,
        storage,
        _request(
            scope_type="course_offering",
            scope_id="course-2026-fall",
            expires_at=now + timedelta(days=120),
        ),
    )
    session.commit()

    assert assessment.id != course.id
    assert assessment.content_object_id == course.content_object_id
    assert storage.upload_count == 3  # Shared source, intent, and completion.
    assert retention_mode_for_scope(assessment.scope_type).value == "assessment"
    assert retention_mode_for_scope(course.scope_type).value == "course"
    assert assessment.validation_evidence["retention_mode"] == "assessment"
    assert course.validation_evidence["retention_mode"] == "course"

    assert find_accepted_representation(
        session,
        scope_type="assessment",
        scope_id="assessment-1",
        doi="10.1234/example",
        now=now,
    ).id == assessment.id
    assert find_accepted_representation(
        session,
        scope_type="course_offering",
        scope_id="course-2026-fall",
        doi="10.1234/example",
        now=now,
    ).id == course.id
    assert find_accepted_representation(
        session,
        scope_type="course_offering",
        scope_id="course-2027-spring",
        doi="10.1234/example",
        now=now,
    ) is None
    assert find_accepted_representation(
        session,
        scope_type="assessment",
        scope_id="course-2026-fall",
        doi="10.1234/example",
        now=now,
    ) is None


def test_expired_course_link_does_not_remove_live_assessment_object(
    session: Session,
) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    course = admit_representation(
        session,
        storage,
        _request(
            scope_type="course_offering",
            scope_id="course-2026-fall",
            expires_at=now - timedelta(seconds=1),
        ),
    )
    assessment = admit_representation(
        session,
        storage,
        _request(
            scope_type="assessment",
            scope_id="assessment-2",
            expires_at=now + timedelta(days=1),
        ),
    )
    object_id = course.content_object_id
    storage_key = course.content_object.storage_key
    session.commit()

    assert find_accepted_representation(
        session,
        scope_type="course_offering",
        scope_id="course-2026-fall",
        doi="10.1234/example",
        now=now,
    ) is None
    assert find_accepted_representation(
        session,
        scope_type="assessment",
        scope_id="assessment-2",
        doi="10.1234/example",
        now=now,
    ).id == assessment.id

    assert expire_representations(session, now=now) == 1
    session.commit()
    assert session.get(SourceRepresentationRecord, course.id) is None
    assert finalize_pending_object_deletions(session, storage) == 0
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is not None
    assert storage.exists(storage_key)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"identity_verdict": "unknown"}, "identity_verdict"),
        ({"completeness_verdict": "incomplete"}, "completeness_verdict"),
        ({"cleanliness_verdict": "unknown"}, "cleanliness_verdict"),
    ],
)
def test_acceptance_fails_closed_to_needs_review(
    session: Session,
    overrides: dict,
    field: str,
) -> None:
    storage = MemoryStorage()
    record = admit_representation(session, storage, _request(**overrides))

    assert getattr(record, field) == next(iter(overrides.values()))
    assert record.admission_state == "needs_review"
    assert record.admitted_at is None
    assert record.admitted_by is None


def test_repeated_admission_deduplicates_object_and_record(session: Session) -> None:
    storage = MemoryStorage()
    first = admit_representation(session, storage, _request())
    second = admit_representation(session, storage, _request())
    session.commit()

    assert first.id == second.id
    assert storage.upload_count == 3  # Deduplication does not create another intent.
    assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
    assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 1


def test_retention_keeps_tombstone_when_delete_ack_does_not_remove_bytes(session, monkeypatch):
    storage = MemoryStorage()
    record = admit_representation(session, storage, _request())
    session.commit()
    object_id = record.content_object_id
    assert delete_representation(session, record.id)
    session.commit()
    monkeypatch.setattr(storage, "delete", lambda _key: True)
    assert finalize_pending_object_deletions(session, storage) == 0
    session.commit()
    pending = session.get(ContentObjectRecord, object_id)
    assert pending is not None and pending.deletion_pending
    assert storage.exists(pending.storage_key)
    assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
    assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 0


def test_accepted_lookup_excludes_expired_representation(session: Session) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    expired = admit_representation(
        session,
        storage,
        _request(expires_at=now - timedelta(seconds=1)),
    )
    session.commit()

    assert expired.admission_state == "accepted"
    assert find_accepted_representation(
        session,
        scope_type="personal_owner",
        scope_id="owner-1",
        doi="10.1234/example",
        now=now,
    ) is None


def test_reacquisition_can_renew_an_expired_representation(session: Session) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    original = admit_representation(
        session,
        storage,
        _request(expires_at=now - timedelta(days=1)),
    )
    session.commit()

    renewed = admit_representation(
        session,
        storage,
        _request(expires_at=now + timedelta(days=30)),
    )
    session.commit()

    assert renewed.id == original.id
    assert renewed.expires_at is not None
    assert not representation_is_expired(renewed, now=now)
    assert renewed.admission_state == "accepted"
    assert find_accepted_representation(
        session,
        scope_type="personal_owner",
        scope_id="owner-1",
        doi="10.1234/example",
        now=now,
    ).id == original.id


def test_expiry_cleanup_preserves_shared_live_object(session: Session) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    expired = admit_representation(
        session,
        storage,
        _request(
            scope_id="course-expired",
            expires_at=now - timedelta(seconds=1),
        ),
    )
    live = admit_representation(
        session,
        storage,
        _request(
            scope_id="course-live",
            expires_at=now + timedelta(days=1),
        ),
    )
    object_id = expired.content_object_id
    storage_key = expired.content_object.storage_key
    session.commit()

    assert expire_representations(session, now=now) == 1
    session.commit()
    assert session.get(SourceRepresentationRecord, expired.id) is None
    assert session.get(SourceRepresentationRecord, live.id) is not None
    assert finalize_pending_object_deletions(session, storage) == 0
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is not None
    assert storage.exists(storage_key)


def test_expiry_cleanup_deletes_last_unreferenced_object(session: Session) -> None:
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    expired = admit_representation(
        session,
        storage,
        _request(expires_at=now - timedelta(seconds=1)),
    )
    object_id = expired.content_object_id
    storage_key = expired.content_object.storage_key
    session.commit()

    assert expire_representations(session, now=now) == 1
    session.commit()
    assert storage.exists(storage_key)
    assert finalize_pending_object_deletions(session, storage) == 1
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is None
    assert not storage.exists(storage_key)


def test_failed_expiry_object_deletion_remains_retryable(session: Session) -> None:
    storage = TemporarilyFailingDeleteStorage()
    now = datetime.now(timezone.utc)
    expired = admit_representation(
        session,
        storage,
        _request(expires_at=now - timedelta(seconds=1)),
    )
    object_id = expired.content_object_id
    storage_key = expired.content_object.storage_key
    session.commit()

    assert expire_representations(session, now=now) == 1
    session.commit()
    assert finalize_pending_object_deletions(session, storage) == 0
    session.commit()

    tombstone = session.get(ContentObjectRecord, object_id)
    assert tombstone is not None
    assert tombstone.deletion_pending is True
    assert storage.exists(storage_key)

    storage.delete_allowed = True
    assert finalize_pending_object_deletions(session, storage) == 1
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is None
    assert not storage.exists(storage_key)


def test_delete_is_reference_safe_for_shared_content(session: Session) -> None:
    storage = MemoryStorage()
    first = admit_representation(session, storage, _request(scope_id="course-1"))
    second = admit_representation(session, storage, _request(scope_id="course-2"))
    object_id = first.content_object_id
    storage_key = first.content_object.storage_key
    session.commit()

    assert first.content_object_id == second.content_object_id
    assert delete_representation(session, first.id) is True
    session.commit()
    assert finalize_pending_object_deletions(session, storage) == 0
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is not None
    assert storage.exists(storage_key)

    assert delete_representation(session, second.id) is True
    session.commit()
    tombstone = session.get(ContentObjectRecord, object_id)
    assert tombstone is not None
    assert tombstone.deletion_pending is True
    assert storage.exists(storage_key)
    assert finalize_pending_object_deletions(session, storage) == 1
    session.commit()
    assert session.get(ContentObjectRecord, object_id) is None
    assert not storage.exists(storage_key)


def test_existing_metadata_with_missing_object_fails_closed(session: Session) -> None:
    storage = MemoryStorage()
    first = admit_representation(session, storage, _request())
    storage.objects.clear()
    session.commit()

    with pytest.raises(AdmissionError, match="immutable object is missing"):
        admit_representation(session, storage, _request(scope_id="owner-2"))
    assert first.admission_state == "accepted"


def test_detach_and_tombstone_rollback_together(session: Session) -> None:
    storage = MemoryStorage()
    record = admit_representation(session, storage, _request())
    session.commit()
    record_id, object_id = record.id, record.content_object_id
    assert delete_representation(session, record_id)
    assert session.get(ContentObjectRecord, object_id).deletion_pending
    session.rollback()
    assert session.get(SourceRepresentationRecord, record_id) is not None
    assert not session.get(ContentObjectRecord, object_id).deletion_pending
    assert finalize_pending_object_deletions(session, storage) == 0


def test_expiry_rechecks_renewal_after_candidate_scan(session: Session, monkeypatch) -> None:
    from dataclasses import replace
    from app.services import source_repository as service
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    request = _request(expires_at=now - timedelta(seconds=1))
    record = admit_representation(session, storage, request)
    session.commit()
    original = service.delete_representation
    def renew_before_delete(db, record_id, **kwargs):
        admit_representation(db, storage, replace(request, expires_at=now + timedelta(days=1)))
        db.commit()
        return original(db, record_id, **kwargs)
    monkeypatch.setattr(service, "delete_representation", renew_before_delete)
    assert expire_representations(session, now=now) == 0
    assert session.get(SourceRepresentationRecord, record.id) is not None
    assert not record.content_object.deletion_pending


def test_expiry_defers_busy_content_without_mutation(session: Session, monkeypatch) -> None:
    from app.services import source_repository as service
    storage = MemoryStorage()
    now = datetime.now(timezone.utc)
    record = admit_representation(session, storage, _request(expires_at=now - timedelta(seconds=1)))
    session.commit()
    monkeypatch.setattr(service, "_try_content_lock", lambda *args: False)
    assert expire_representations(session, now=now) == 0
    assert session.get(SourceRepresentationRecord, record.id) is not None
    assert not record.content_object.deletion_pending
    assert storage.exists(record.content_object.storage_key)


def test_abstract_cannot_enter_durable_content_store(session: Session) -> None:
    storage = MemoryStorage()
    request = _request()
    request = AdmissionRequest(
        **{
            **request.__dict__,
            "representation": SourceRepresentation(
                kind=RepresentationKind.ABSTRACT,
                media_type="text/plain",
                content=b"An abstract is limited evidence, not a source object.",
            ),
        }
    )

    with pytest.raises(AdmissionError, match="not durable source content"):
        admit_representation(session, storage, request)
