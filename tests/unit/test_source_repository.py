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
    admit_representation,
    delete_representation,
    expire_representations,
    finalize_pending_object_deletions,
    find_accepted_representation,
    retention_mode_for_scope,
    representation_is_expired,
)
from app.services.storage.backend import StorageBackend


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_count = 0

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.upload_count += 1
        self.objects.setdefault(key, file_bytes)
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
    assert content.storage_key == f"commercial_user_upload/{digest}.pdf"
    assert storage.objects[content.storage_key].startswith(b"%PDF-")
    assert work is not None
    assert work.doi == "10.1234/example"
    assert record.validation_evidence["retention_mode"] == "durable"
    assert record.validation_evidence["authorization_scope"] == {
        "type": "personal_owner",
        "id": "owner-1",
    }


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

    assert storage.upload_count == 1


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
    assert storage.upload_count == 1
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
    assert storage.upload_count == 1
    assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
    assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 1


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
