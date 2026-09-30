"""Durable admission service for academic source representations."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
import uuid

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.models.source_repository import (
    CanonicalWorkRecord,
    ContentObjectRecord,
    SourceRepresentationRecord,
)
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.storage.backend import StorageBackend
from app.services.source_upload_recovery import (
    persist_upload_intent, persist_retirement_intent, confirm_upload_intent,
    settle_session_upload_intents, _try_content_lock,
)
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


def _advisory_transaction_lock(session: Session, namespace: str, value: str) -> None:
    """Serialize a content/identity admission key for this DB transaction."""
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    seed = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
    lock_id = int.from_bytes(seed[:8], byteorder="big", signed=True)
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id}
    )


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
            SourceRepresentationRecord.completeness_verdict.in_((*ACCEPTABLE_COMPLETENESS, 'incomplete')),
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
        query.order_by(SourceRepresentationRecord.created_at.desc()).limit(20)
    ).all()
    if not records:
        return None
    from app.services.publisher_preview import valid_preview_receipt
    eligible = [r for r in records if r.completeness_verdict in ACCEPTABLE_COMPLETENESS or (
        r.identity_verdict in ACCEPTABLE_IDENTITY and r.cleanliness_verdict == 'clean'
        and r.representation_kind == 'pdf'
        and valid_preview_receipt((r.validation_evidence or {}).get('publisher_preview'),
                                  r.content_object.content_sha256, r.source_url))]
    if not eligible:
        return None
    # A complete copy remains preferable to a newer limited preview.
    record = min(eligible, key=lambda r: r.completeness_verdict not in ACCEPTABLE_COMPLETENESS)
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


def recheck_accepted_pdf_completeness(
    session: Session, backend: StorageBackend, *, representation_id: str | uuid.UUID,
    scope_type: str, scope_id: str, expected_content_sha256: str, document_kind: str,
) -> dict:
    """Explicit, downgrade-only correction of an already authorized copy.

    This is not an admission route for new partial files. Existing scope/bytes
    remain authorized for limited report evidence, while incomplete copies no
    longer satisfy complete-source reuse. Keep the previous observation and
    checker signals; immutable packages are corrected only by a successor.
    """
    from dataclasses import asdict
    from app.services.verification_evidence import authorize_representation
    from app.services.completeness_checker import check_completeness

    record = session.scalar(select(SourceRepresentationRecord).where(
        SourceRepresentationRecord.id == uuid.UUID(str(representation_id))
    ).with_for_update().execution_options(populate_existing=True))
    if record is None:
        raise AdmissionError('Representation unavailable for completeness correction')
    source = authorize_representation(session, backend, representation_id=record.id,
                                      scope_type=scope_type, scope_id=scope_id)
    if source.content_sha256 != expected_content_sha256 or source.media_type != 'application/pdf':
        raise AdmissionError('Completeness correction source binding differs')
    if document_kind not in {'book', 'article', 'chapter', 'unknown'}:
        raise AdmissionError('Unsupported completeness document kind')
    observation = check_completeness(source.content, document_kind=document_kind, external_lookup=False)
    receipt = dict(version='accepted-pdf-completeness-recheck-v1',
                   content_sha256=source.content_sha256,
                   previous_verdict=record.completeness_verdict,
                   observation=asdict(observation), changed=False)
    if record.completeness_verdict in ACCEPTABLE_COMPLETENESS and observation.verdict == 'INCOMPLETE':
        receipt['changed'] = True
        receipt['observed_at'] = datetime.now(timezone.utc).isoformat()
        evidence = dict(record.validation_evidence or {})
        evidence['completeness_corrections'] = [*evidence.get('completeness_corrections', []), receipt]
        record.validation_evidence = evidence
        record.completeness_verdict = 'incomplete'
        session.flush()
    return receipt


def _admission_state(request: AdmissionRequest) -> str:
    if not request.request_acceptance:
        return "needs_review"
    from app.services.publisher_preview import valid_preview_receipt
    preview = (request.representation.kind is RepresentationKind.PDF
               and normalize_source_kind(request.work.work_type) == 'monograph'
               and request.completeness_verdict.casefold() == 'incomplete'
               and valid_preview_receipt(request.validation_evidence.get('publisher_preview'),
                    hashlib.sha256(request.representation.content).hexdigest(), request.representation.source_url))
    if (
        request.identity_verdict.casefold() not in ACCEPTABLE_IDENTITY
        or (request.completeness_verdict.casefold() not in ACCEPTABLE_COMPLETENESS and not preview)
        or request.cleanliness_verdict.casefold() not in ACCEPTABLE_CLEANLINESS
    ):
        return "needs_review"
    return "accepted"


def _object_key(
    license_class: str,
    digest: str,
    kind: RepresentationKind,
    generation: str | None = None,
) -> str:
    suffix = f"/{generation}" if generation is not None else ""
    return f"{license_class}/{digest}{suffix}.{EXTENSIONS[kind]}"


def _canonical_work(session: Session, identity: WorkIdentity, *, separate_preview_kind: bool = False) -> CanonicalWorkRecord:
    doi = _normalize_doi(identity.doi)
    isbn = _normalize_isbn(identity.isbn)
    normalized_title = _normalize_title(identity.title)
    identity_key = doi or isbn or f"{normalized_title}\0{identity.author}\0{identity.year}"
    _advisory_transaction_lock(session, "canonical-work", identity_key)
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
        query = select(CanonicalWorkRecord).where(
                CanonicalWorkRecord.normalized_title == normalized_title,
                CanonicalWorkRecord.author == identity.author,
                CanonicalWorkRecord.year == identity.year,
            )
        if separate_preview_kind:
            # Title/author/year is not a globally unique identifier. Keep old
            # webpage classifications intact rather than mutating their history.
            query = query.where(CanonicalWorkRecord.work_type.in_({'book', 'monograph'}))
        existing = session.scalar(query)
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
    from app.services.publisher_preview import valid_preview_receipt
    separate_preview_kind = (request.completeness_verdict == 'incomplete'
        and normalize_source_kind(request.work.work_type) == 'monograph'
        and _admission_state(request) == 'accepted'
        and valid_preview_receipt(request.validation_evidence.get('publisher_preview'),
            hashlib.sha256(request.representation.content).hexdigest(), request.representation.source_url))
    work = _canonical_work(session, request.work, separate_preview_kind=separate_preview_kind)
    digest = hashlib.sha256(request.representation.content).hexdigest()
    _advisory_transaction_lock(
        session, "content-object", f"{request.license_class}:{digest}"
    )
    content_object = session.scalar(
        select(ContentObjectRecord).where(
            ContentObjectRecord.license_class == request.license_class,
            ContentObjectRecord.content_sha256 == digest,
        ).execution_options(populate_existing=True)
    )
    if content_object is None or content_object.deletion_pending:
        if content_object is not None:
            if session.scalar(select(func.count(SourceRepresentationRecord.id)).where(
                SourceRepresentationRecord.content_object_id == content_object.id
            )):
                raise AdmissionError("Deletion-pending content still has references")
            # Retain the old target before changing its database locator. An
            # old DELETE may still complete after this transaction commits.
            old_kind = next(kind for kind, extension in EXTENSIONS.items()
                            if content_object.storage_key.endswith("." + extension))
            persist_retirement_intent(session, backend,
                storage_key=content_object.storage_key,
                license_class=request.license_class, digest=digest, kind=old_kind)
        generation = uuid.uuid4().hex
        key = _object_key(request.license_class, digest, request.representation.kind, generation)
        intent = persist_upload_intent(session, backend, storage_key=key,
                              license_class=request.license_class, digest=digest,
                              kind=request.representation.kind, generation=generation)
        receipt = backend.upload(request.representation.content, key)
        confirm_upload_intent(backend, intent, receipt)
        if content_object is None:
            content_object = ContentObjectRecord(content_sha256=digest,
                license_class=request.license_class)
            session.add(content_object)
        content_object.storage_key = key
        content_object.media_type = request.representation.media_type
        content_object.byte_size = len(request.representation.content)
        content_object.deletion_pending = False
        session.flush()
    elif not backend.exists(content_object.storage_key):
        raise AdmissionError(
            "Content metadata exists but the immutable object is missing; "
            "admission failed closed"
        )

    state = _admission_state(request)
    now = datetime.now(timezone.utc)
    # The content lock serializes renewal, but does not refresh a caller's
    # cached expiry or validation state after a peer's committed renewal.
    existing = session.scalar(
        select(SourceRepresentationRecord).where(
            SourceRepresentationRecord.canonical_work_id == work.id,
            SourceRepresentationRecord.content_object_id == content_object.id,
            SourceRepresentationRecord.provenance == request.provenance,
            SourceRepresentationRecord.scope_type == scope_type,
            SourceRepresentationRecord.scope_id == scope_id,
        ).execution_options(populate_existing=True)
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


def admit_derived_representation_pair(
    session: Session,
    backend: StorageBackend,
    *,
    parent_request: AdmissionRequest,
    derivative_request: AdmissionRequest,
) -> tuple[SourceRepresentationRecord, SourceRepresentationRecord]:
    """Atomically stage one immutable PDF parent and accepted OCR derivative."""
    parent = parent_request.representation
    derivative = derivative_request.representation
    if parent.kind is not RepresentationKind.PDF:
        raise AdmissionError("OCR derivative parent must be a PDF")
    if (
        derivative.kind is not RepresentationKind.PLAIN_TEXT
        or derivative.original_kind is not RepresentationKind.PDF
    ):
        raise AdmissionError("OCR derivative must be plain text derived from a PDF")
    if parent_request.request_acceptance:
        raise AdmissionError("OCR parent must remain a non-evidence representation")
    if not derivative_request.request_acceptance:
        raise AdmissionError("OCR derivative admission must request acceptance")
    comparable_parent = replace(
        parent_request,
        representation=derivative_request.representation,
        provenance=derivative_request.provenance,
        request_acceptance=derivative_request.request_acceptance,
        validation_evidence=derivative_request.validation_evidence,
    )
    for field_name in (
        "work",
        "license_class",
        "scope_type",
        "scope_id",
        "expires_at",
    ):
        if getattr(comparable_parent, field_name) != getattr(
            derivative_request, field_name
        ):
            raise AdmissionError("OCR parent and derivative policies do not match")
    parent_sha256 = hashlib.sha256(parent.content).hexdigest()
    derivative_sha256 = hashlib.sha256(derivative.content).hexdigest()
    provenance = derivative_request.validation_evidence.get("ocr_derivative", {})
    if (
        provenance.get("parent_content_sha256") != parent_sha256
        or provenance.get("derivative_content_sha256") != derivative_sha256
        or not re.fullmatch(
            r"[0-9a-f]{64}", provenance.get("derivation_manifest_sha256", "")
        )
    ):
        raise AdmissionError("OCR derivative provenance hashes do not match")

    parent_record = admit_representation(session, backend, parent_request)
    derivative_evidence = dict(derivative_request.validation_evidence)
    derivative_provenance = dict(derivative_evidence["ocr_derivative"])
    derivative_provenance["parent_representation_id"] = str(parent_record.id)
    derivative_evidence["ocr_derivative"] = derivative_provenance
    derivative_record = admit_representation(
        session,
        backend,
        replace(
            derivative_request,
            validation_evidence=derivative_evidence,
        ),
    )
    return parent_record, derivative_record


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
        if delete_representation(session, representation_id, expired_before=current):
            expired += 1
    return expired


def delete_representation(
    session: Session,
    representation_id: uuid.UUID,
    *,
    expired_before: datetime | None = None,
) -> bool:
    """Detach one representation and tombstone an unreferenced object.

    The caller commits this database change before finalizing object deletion,
    so a failed commit can never leave a live representation pointing at a
    deleted object. Serialize the detach/count/tombstone transaction with
    admission and cleanup using their shared immutable content key. Scheduled
    expiry skips busy content and rechecks expiry under the lock; it must not
    take representation row locks before content locks.
    """
    target = session.execute(
        select(ContentObjectRecord.id, ContentObjectRecord.license_class,
               ContentObjectRecord.content_sha256)
        .join(SourceRepresentationRecord,
              SourceRepresentationRecord.content_object_id == ContentObjectRecord.id)
        .where(SourceRepresentationRecord.id == representation_id)
    ).first()
    if target is None:
        return False
    object_id, license_class, digest = target
    if expired_before is not None:
        if not _try_content_lock(session, {
            "license_class": license_class, "content_sha256": digest,
        }):
            return False
    else:
        _advisory_transaction_lock(session, "content-object", f"{license_class}:{digest}")
    query = select(SourceRepresentationRecord).where(
        SourceRepresentationRecord.id == representation_id,
        SourceRepresentationRecord.content_object_id == object_id,
    )
    if expired_before is not None:
        query = query.where(SourceRepresentationRecord.expires_at.is_not(None),
            SourceRepresentationRecord.expires_at <= _as_utc(expired_before))
    # API authorization may already have loaded this ORM instance. Refresh it
    # after waiting: another transaction may have deleted or renewed the row.
    record = session.scalar(query.execution_options(populate_existing=True))
    if record is None:
        return False
    session.delete(record)
    session.flush()
    remaining = session.scalar(
        select(func.count(SourceRepresentationRecord.id)).where(
            SourceRepresentationRecord.content_object_id == object_id
        )
    )
    if remaining == 0:
        # A peer may have readmitted this logical object at a fresh generation.
        # Refresh under the content lock so a cached True cannot suppress the
        # False -> True update that persists current-generation cleanup.
        content_object = session.get(ContentObjectRecord, object_id, populate_existing=True)
        if content_object is not None:
            content_object.deletion_pending = True
            session.flush()
    return True


def finalize_pending_object_deletions(
    session: Session,
    backend: StorageBackend,
    *,
    batch_size: int = 100,
) -> int:
    """Delete tombstoned unreferenced objects; retain failures for retry."""
    pending = session.scalars(
        select(ContentObjectRecord)
        .where(ContentObjectRecord.deletion_pending.is_(True))
        .order_by(ContentObjectRecord.created_at, ContentObjectRecord.id)
        .limit(max(1, min(batch_size, 1_000)))
        .with_for_update(skip_locked=True)
        # A logical content row may now name a newer physical generation.
        # The locked SELECT must replace any caller-cached locator before
        # deleting bytes and retiring that row's cleanup authority.
        .execution_options(populate_existing=True)
    ).all()
    deleted = 0
    for content_object in pending:
        if not _try_content_lock(session, {
            "license_class": content_object.license_class,
            "content_sha256": content_object.content_sha256,
        }):
            continue
        remaining = session.scalar(
            select(func.count(SourceRepresentationRecord.id)).where(
                SourceRepresentationRecord.content_object_id == content_object.id
            )
        )
        if remaining:
            content_object.deletion_pending = False
            continue
        if not backend.delete(content_object.storage_key) or backend.exists(content_object.storage_key):
            continue
        session.delete(content_object)
        deleted += 1
    session.flush()
    return deleted


def commit_source_admissions(session: Session) -> None:
    """Commit admission records and release their object-store compensations."""
    session.commit()
    settle_session_upload_intents(session)


def rollback_source_admissions(
    session: Session,
    backend: StorageBackend,
) -> int:
    """Rollback admissions and remove only objects left without durable records."""
    session.rollback()
    return settle_session_upload_intents(session)
