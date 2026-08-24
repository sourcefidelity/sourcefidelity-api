"""Durable admission service for academic source representations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.source_repository import (
    CanonicalWorkRecord,
    ContentObjectRecord,
    SourceRepresentationRecord,
)
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.storage.backend import StorageBackend
from app.services.source_type import (
    SourceKindAssessment,
    compare_source_kinds,
    normalize_source_kind,
)


LICENSE_CLASSES = frozenset(
    {
        "open_access",
        "public_domain",
        "paywalled_db_retrieved",
        "commercial_user_upload",
        "other_restricted",
        "rights_unclassified",
    }
)
ACCEPTABLE_IDENTITY = frozenset({"match", "verified"})
ACCEPTABLE_COMPLETENESS = frozenset({"complete", "not_applicable"})
ACCEPTABLE_CLEANLINESS = frozenset({"clean"})
STORABLE_KINDS = frozenset(
    {
        RepresentationKind.PDF,
        RepresentationKind.HTML,
        RepresentationKind.XML,
        RepresentationKind.EPUB,
        RepresentationKind.PLAIN_TEXT,
    }
)
EXTENSIONS = {
    RepresentationKind.PDF: "pdf",
    RepresentationKind.HTML: "html",
    RepresentationKind.XML: "xml",
    RepresentationKind.EPUB: "epub",
    RepresentationKind.PLAIN_TEXT: "txt",
}


class RetentionMode(str, Enum):
    """Policy modes for the lifecycle of a source representation."""

    VERIFICATION_RUN = "verification_run"
    ASSESSMENT = "assessment"
    COURSE = "course"
    DURABLE = "durable"


SCOPE_RETENTION_MODES = {
    "verification_run": RetentionMode.VERIFICATION_RUN,
    "assessment": RetentionMode.ASSESSMENT,
    "course_offering": RetentionMode.COURSE,
    "personal_owner": RetentionMode.DURABLE,
    "institution": RetentionMode.DURABLE,
}
EXPIRING_RETENTION_MODES = frozenset(
    {RetentionMode.ASSESSMENT, RetentionMode.COURSE}
)


class AdmissionError(ValueError):
    """The representation cannot enter the durable repository as requested."""


def retention_mode_for_scope(scope_type: str) -> RetentionMode:
    """Return the canonical retention mode for an authorization scope."""
    normalized = scope_type.strip().casefold()
    try:
        return SCOPE_RETENTION_MODES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SCOPE_RETENTION_MODES))
        raise AdmissionError(
            f"Unsupported source authorization scope: {scope_type!r}; "
            f"expected one of {supported}"
        ) from exc


@dataclass(frozen=True)
class WorkIdentity:
    title: str
    work_type: str
    doi: str | None = None
    isbn: str | None = None
    author: str | None = None
    year: str | None = None


@dataclass(frozen=True)
class AdmissionRequest:
    work: WorkIdentity
    representation: SourceRepresentation
    provenance: str
    license_class: str
    scope_type: str
    scope_id: str
    identity_verdict: str
    completeness_verdict: str
    cleanliness_verdict: str
    identity_confidence: float | None = None
    text_quality: str | None = None
    edition_or_version: str | None = None
    expires_at: datetime | None = None
    admitted_by: str | None = None
    validation_evidence: dict = field(default_factory=dict)
    request_acceptance: bool = True


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    return normalized or None


def _normalize_isbn(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^0-9Xx]", "", value).upper()
    return normalized or None


def _normalize_title(value: str) -> str:
    return " ".join(value.casefold().split())


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def representation_is_expired(
    record: SourceRepresentationRecord,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether a representation's authorization period has ended."""
    if record.expires_at is None:
        return False
    current = _as_utc(now or datetime.now(timezone.utc))
    return _as_utc(record.expires_at) <= current


def active_representation_clause(*, now: datetime | None = None):
    """SQL clause excluding representations whose retention period ended."""
    current = _as_utc(now or datetime.now(timezone.utc))
    return or_(
        SourceRepresentationRecord.expires_at.is_(None),
        SourceRepresentationRecord.expires_at > current,
    )


def find_accepted_representation(
    session: Session,
    *,
    scope_type: str,
    scope_id: str,
    doi: str | None = None,
    isbn: str | None = None,
    title: str | None = None,
    work_type: str | None = None,
    now: datetime | None = None,
) -> SourceRepresentationRecord | None:
    """Find one unexpired accepted representation in an authorized scope."""
    normalized_scope_type = scope_type.strip().casefold()
    try:
        retention_mode = retention_mode_for_scope(normalized_scope_type)
    except AdmissionError:
        return None
    if retention_mode is RetentionMode.VERIFICATION_RUN:
        return None
    query = (
        select(SourceRepresentationRecord)
        .join(SourceRepresentationRecord.canonical_work)
        .where(
            SourceRepresentationRecord.scope_type == normalized_scope_type,
            SourceRepresentationRecord.scope_id == scope_id.strip(),
            SourceRepresentationRecord.admission_state == "accepted",
            active_representation_clause(now=now),
        )
    )
    if doi:
        query = query.where(CanonicalWorkRecord.doi == _normalize_doi(doi))
    elif isbn:
        query = query.where(CanonicalWorkRecord.isbn == _normalize_isbn(isbn))
    elif title:
        query = query.where(
            CanonicalWorkRecord.normalized_title == _normalize_title(title)
        )
    else:
        return None
    records = session.scalars(
        query.order_by(SourceRepresentationRecord.created_at.desc()).limit(1)
    ).all()
    if not records:
        return None
    record = records[0]
    expected_kind = normalize_source_kind(work_type)
    if expected_kind != "unknown":
        compatibility = compare_source_kinds(
            SourceKindAssessment(expected_kind, "high", ("cache lookup contract",)),
            SourceKindAssessment(
                normalize_source_kind(record.canonical_work.work_type),
                "high",
                ("durable canonical work",),
            ),
        )
        if compatibility.verdict == "incompatible":
            return None
    return record


def _admission_state(request: AdmissionRequest) -> str:
    if not request.request_acceptance:
        return "needs_review"
    if (
        request.identity_verdict.casefold() not in ACCEPTABLE_IDENTITY
        or request.completeness_verdict.casefold() not in ACCEPTABLE_COMPLETENESS
        or request.cleanliness_verdict.casefold() not in ACCEPTABLE_CLEANLINESS
    ):
        return "needs_review"
    return "accepted"


def _object_key(
    license_class: str,
    digest: str,
    kind: RepresentationKind,
) -> str:
    return f"{license_class}/{digest}.{EXTENSIONS[kind]}"


def _canonical_work(session: Session, identity: WorkIdentity) -> CanonicalWorkRecord:
    doi = _normalize_doi(identity.doi)
    isbn = _normalize_isbn(identity.isbn)
    normalized_title = _normalize_title(identity.title)
    existing = None
    if doi:
        existing = session.scalar(
            select(CanonicalWorkRecord).where(CanonicalWorkRecord.doi == doi)
        )
    if existing is None and isbn:
        existing = session.scalar(
            select(CanonicalWorkRecord).where(CanonicalWorkRecord.isbn == isbn)
        )
    if existing is None and not doi and not isbn:
        existing = session.scalar(
            select(CanonicalWorkRecord).where(
                CanonicalWorkRecord.normalized_title == normalized_title,
                CanonicalWorkRecord.author == identity.author,
                CanonicalWorkRecord.year == identity.year,
            )
        )
    if existing is not None:
        compatibility = compare_source_kinds(
            SourceKindAssessment(
                normalize_source_kind(identity.work_type),
                "high",
                ("admission request",),
            ),
            SourceKindAssessment(
                normalize_source_kind(existing.work_type),
                "high",
                ("existing canonical work",),
            ),
        )
        if compatibility.verdict == "incompatible":
            raise AdmissionError(
                "Canonical identifier already belongs to an incompatible work "
                f"type: {compatibility.reason}"
            )
        return existing
    work = CanonicalWorkRecord(
        doi=doi,
        isbn=isbn,
        normalized_title=normalized_title,
        display_title=identity.title.strip(),
        author=identity.author,
        year=identity.year,
        work_type=identity.work_type,
    )
    session.add(work)
    session.flush()
    return work


def admit_representation(
    session: Session,
    backend: StorageBackend,
    request: AdmissionRequest,
) -> SourceRepresentationRecord:
    """Store immutable content and a durable, fail-closed admission record."""
    if request.license_class not in LICENSE_CLASSES:
        raise AdmissionError(f"Unsupported license class: {request.license_class}")
    if request.representation.kind not in STORABLE_KINDS:
        raise AdmissionError(
            f"Representation kind is not durable source content: "
            f"{request.representation.kind.value}"
        )
    if not request.representation.content:
        raise AdmissionError("Cannot admit an empty representation")
    if not request.work.title.strip() or not request.work.work_type.strip():
        raise AdmissionError("Canonical work title and work type are required")
    if not request.scope_type.strip() or not request.scope_id.strip():
        raise AdmissionError("A durable authorization scope is required")
    scope_type = request.scope_type.strip().casefold()
    scope_id = request.scope_id.strip()
    retention_mode = retention_mode_for_scope(scope_type)
    if retention_mode is RetentionMode.VERIFICATION_RUN:
        raise AdmissionError(
            "Verification-run representations must remain ephemeral and cannot "
            "enter durable admission"
        )
    if retention_mode in EXPIRING_RETENTION_MODES and request.expires_at is None:
        raise AdmissionError(
            f"{retention_mode.value} retention requires an explicit expiry"
        )

    validation_evidence = dict(request.validation_evidence)
    validation_evidence["retention_mode"] = retention_mode.value
    validation_evidence["authorization_scope"] = {
        "type": scope_type,
        "id": scope_id,
    }

    # Resolve canonical identity before any object-store write so a type
    # conflict cannot leave an unreferenced immutable object behind.
    work = _canonical_work(session, request.work)
    digest = hashlib.sha256(request.representation.content).hexdigest()
    content_object = session.scalar(
        select(ContentObjectRecord).where(
            ContentObjectRecord.license_class == request.license_class,
            ContentObjectRecord.content_sha256 == digest,
        )
    )
    if content_object is None:
        key = _object_key(request.license_class, digest, request.representation.kind)
        backend.upload(request.representation.content, key)
        content_object = ContentObjectRecord(
            content_sha256=digest,
            license_class=request.license_class,
            storage_key=key,
            media_type=request.representation.media_type,
            byte_size=len(request.representation.content),
        )
        session.add(content_object)
        session.flush()
    elif not backend.exists(content_object.storage_key):
        raise AdmissionError(
            "Content metadata exists but the immutable object is missing; "
            "admission failed closed"
        )
    elif content_object.deletion_pending:
        # A new authorized reference revives the object before cleanup.
        content_object.deletion_pending = False

    state = _admission_state(request)
    now = datetime.now(timezone.utc)
    existing = session.scalar(
        select(SourceRepresentationRecord).where(
            SourceRepresentationRecord.canonical_work_id == work.id,
            SourceRepresentationRecord.content_object_id == content_object.id,
            SourceRepresentationRecord.provenance == request.provenance,
            SourceRepresentationRecord.scope_type == scope_type,
            SourceRepresentationRecord.scope_id == scope_id,
        )
    )
    if existing is not None:
        was_expired = representation_is_expired(existing, now=now)
        if existing.expires_at is not None:
            if request.expires_at is None:
                existing.expires_at = None
            elif _as_utc(request.expires_at) > _as_utc(existing.expires_at):
                existing.expires_at = request.expires_at
        if was_expired:
            existing.admission_state = state
            existing.identity_verdict = request.identity_verdict.casefold()
            existing.identity_confidence = request.identity_confidence
            existing.completeness_verdict = request.completeness_verdict.casefold()
            existing.cleanliness_verdict = request.cleanliness_verdict.casefold()
            existing.text_quality = request.text_quality
            existing.edition_or_version = request.edition_or_version
            existing.source_url = request.representation.source_url
            existing.validation_evidence = validation_evidence
            existing.admitted_at = now if state == "accepted" else None
            existing.admitted_by = request.admitted_by if state == "accepted" else None
        session.flush()
        return existing

    record = SourceRepresentationRecord(
        canonical_work_id=work.id,
        content_object_id=content_object.id,
        representation_kind=request.representation.kind.value,
        original_kind=(
            request.representation.original_kind.value
            if request.representation.original_kind
            else None
        ),
        provenance=request.provenance,
        admission_state=state,
        scope_type=scope_type,
        scope_id=scope_id,
        source_url=request.representation.source_url,
        edition_or_version=request.edition_or_version,
        identity_verdict=request.identity_verdict.casefold(),
        identity_confidence=request.identity_confidence,
        completeness_verdict=request.completeness_verdict.casefold(),
        cleanliness_verdict=request.cleanliness_verdict.casefold(),
        text_quality=request.text_quality,
        validation_evidence=validation_evidence,
        expires_at=request.expires_at,
        admitted_at=now if state == "accepted" else None,
        admitted_by=request.admitted_by if state == "accepted" else None,
    )
    session.add(record)
    session.flush()
    return record


def expire_representations(
    session: Session,
    *,
    now: datetime | None = None,
    batch_size: int = 500,
) -> int:
    """Delete one bounded batch of expired representation references.

    Content objects are only tombstoned here. The caller commits these record
    deletions before invoking ``finalize_pending_object_deletions`` so a failed
    object-store operation remains retryable and cannot strand a live record.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    current = _as_utc(now or datetime.now(timezone.utc))
    representation_ids = session.scalars(
        select(SourceRepresentationRecord.id)
        .where(
            SourceRepresentationRecord.expires_at.is_not(None),
            SourceRepresentationRecord.expires_at <= current,
        )
        .order_by(SourceRepresentationRecord.expires_at, SourceRepresentationRecord.id)
        .limit(batch_size)
    ).all()
    expired = 0
    for representation_id in representation_ids:
        if delete_representation(session, representation_id):
            expired += 1
    return expired


def delete_representation(
    session: Session,
    representation_id: uuid.UUID,
) -> bool:
    """Detach one representation and tombstone an unreferenced object.

    The caller commits this database change before finalizing object deletion,
    so a failed commit can never leave a live representation pointing at a
    deleted object.
    """
    record = session.get(SourceRepresentationRecord, representation_id)
    if record is None:
        return False
    object_id = record.content_object_id
    session.delete(record)
    session.flush()
    remaining = session.scalar(
        select(func.count(SourceRepresentationRecord.id)).where(
            SourceRepresentationRecord.content_object_id == object_id
        )
    )
    if remaining == 0:
        content_object = session.get(ContentObjectRecord, object_id)
        if content_object is not None:
            content_object.deletion_pending = True
            session.flush()
    return True


def finalize_pending_object_deletions(
    session: Session,
    backend: StorageBackend,
) -> int:
    """Delete tombstoned unreferenced objects; retain failures for retry."""
    pending = session.scalars(
        select(ContentObjectRecord).where(ContentObjectRecord.deletion_pending.is_(True))
    ).all()
    deleted = 0
    for content_object in pending:
        remaining = session.scalar(
            select(func.count(SourceRepresentationRecord.id)).where(
                SourceRepresentationRecord.content_object_id == content_object.id
            )
        )
        if remaining:
            content_object.deletion_pending = False
            continue
        if not backend.delete(content_object.storage_key):
            continue
        session.delete(content_object)
        deleted += 1
    session.flush()
    return deleted
