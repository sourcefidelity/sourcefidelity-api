"""Lease-backed execution and guaranteed cleanup for one-run source use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.report import VerificationReportRecord
from app.models.verification_run import VerificationRunRecord
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    VerificationEvidenceArtifact,
)
from app.services.verification_report import persist_verification_report


RUN_AUDIT_VERSION = "verification-run-audit-v1"
logger = logging.getLogger(__name__)
ACTIVE_STATUS = "active"
PLANNING_STATUS = "allocating"
REPORT_PERSISTED_STATUS = "report_persisted"
CLEANUP_PENDING_STATUS = "cleanup_pending"
CLEANED_STATUSES = frozenset({"cleaned", "failed_cleaned", "abandoned_cleaned"})
_CLEANABLE_STALE_STATUSES = frozenset(
    {PLANNING_STATUS, ACTIVE_STATUS, REPORT_PERSISTED_STATUS, CLEANUP_PENDING_STATUS}
)
_EXTENSIONS = {
    RepresentationKind.PDF: "pdf",
    RepresentationKind.HTML: "html",
    RepresentationKind.XML: "xml",
    RepresentationKind.EPUB: "epub",
    RepresentationKind.PLAIN_TEXT: "txt",
}
_DERIVED_EXTENSIONS = {"extracted_text": "txt", "chunks": "json", "embeddings": "bin"}


class VerificationRunError(RuntimeError):
    """The transient execution boundary could not proceed safely."""


class VerificationRunAuthorizationError(VerificationRunError):
    """The run is absent, expired, inactive, or outside the requested scope."""


class VerificationRunCleanupPending(VerificationRunError):
    """A report exists, but transient object deletion requires retry."""

    def __init__(self, run_id: uuid.UUID, report_id: uuid.UUID) -> None:
        super().__init__("Verification report persisted but transient cleanup is pending")
        self.run_id = run_id
        self.report_id = report_id


@dataclass(frozen=True)
class VerificationRunRequest:
    paper_version_id: str
    scope_type: str
    scope_id: str
    canonical_work_id: str
    representation: SourceRepresentation
    acquisition_route: str
    identity_verdict: str
    completeness_verdict: str
    cleanliness_verdict: str
    identity_confidence: float | None = None
    text_quality: str = "unknown"
    edition_or_version: str | None = None


@dataclass(frozen=True)
class VerificationRunExecutionResult:
    run_id: uuid.UUID
    report_id: uuid.UUID
    report_version: int


@dataclass(frozen=True)
class VerificationRunBatchExecutionResult:
    run_id: uuid.UUID
    reports: tuple[tuple[uuid.UUID, int], ...]


def begin_verification_run(
    session: Session,
    backend: StorageBackend,
    request: VerificationRunRequest,
    *,
    now: datetime | None = None,
) -> VerificationRunRecord:
    """Create a committed cleanup lease, then upload the transient source.

    This function intentionally owns commits: the planned key must be durable
    before upload so a hard exit cannot create an undiscoverable orphan.
    """
    current = _as_utc(now or datetime.now(timezone.utc))
    scope_type, scope_id = _scope(request.scope_type, request.scope_id)
    paper_version_id = request.paper_version_id.strip()
    canonical_work_id = request.canonical_work_id.strip()
    route = request.acquisition_route.strip()
    representation = request.representation
    if not paper_version_id or not canonical_work_id or not route:
        raise VerificationRunError(
            "Paper version, canonical work, and acquisition route are required"
        )
    if representation.kind not in _EXTENSIONS:
        raise VerificationRunError("Representation kind is not transient source content")
    if not representation.content:
        raise VerificationRunError("Transient verification source cannot be empty")
    identity_verdict = request.identity_verdict.strip().casefold()
    cleanliness_verdict = request.cleanliness_verdict.strip().casefold()
    if identity_verdict not in {"match", "verified"}:
        raise VerificationRunError("Transient source identity is not verified")
    if cleanliness_verdict != "clean":
        raise VerificationRunError("Transient source did not pass the cleanliness gate")
    max_bytes = settings.VERIFICATION_RUN_MAX_TRANSIENT_MB * 1024 * 1024
    if len(representation.content) > max_bytes:
        raise VerificationRunError("Transient source exceeds the configured byte limit")

    run_id = uuid.uuid4()
    digest = hashlib.sha256(representation.content).hexdigest()
    key = f"verification-runs/{run_id}/source.{_EXTENSIONS[representation.kind]}"
    object_record = _transient_object(
        role="source",
        key=key,
        content=representation.content,
    )
    run = VerificationRunRecord(
        id=run_id,
        audit_version=RUN_AUDIT_VERSION,
        paper_version_id=paper_version_id,
        scope_type=scope_type,
        scope_id=scope_id,
        status=PLANNING_STATUS,
        canonical_work_id=canonical_work_id,
        representation_kind=representation.kind.value,
        media_type=representation.media_type,
        acquisition_route=route,
        content_sha256=digest,
        source_byte_size=len(representation.content),
        transient_object_count=1,
        transient_byte_size=len(representation.content),
        transient_objects=[object_record],
        identity_verdict=identity_verdict,
        identity_confidence=request.identity_confidence,
        completeness_verdict=request.completeness_verdict.strip().casefold(),
        cleanliness_verdict=cleanliness_verdict,
        text_quality=request.text_quality.strip().casefold() or "unknown",
        edition_or_version=request.edition_or_version,
        cleanup_attempts=0,
        started_at=current,
        lease_expires_at=current
        + timedelta(seconds=max(1, settings.VERIFICATION_RUN_LEASE_SECONDS)),
        updated_at=current,
    )
    session.add(run)
    session.commit()
    try:
        backend.upload(representation.content, key)
    except Exception:
        try:
            cleanup_verification_run(
                session,
                backend,
                run.id,
                scope_type=scope_type,
                scope_id=scope_id,
                outcome="failed",
            )
        except Exception as cleanup_error:
            logger.error(
                "Initial transient upload failed and cleanup needs scheduled retry: %s",
                type(cleanup_error).__name__,
            )
        raise
    run.status = ACTIVE_STATUS
    run.updated_at = datetime.now(timezone.utc)
    session.commit()
    return run


def load_verification_run_source(
    session: Session,
    backend: StorageBackend,
    run_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
) -> AuthorizedRepresentation:
    """Load the transient source only while its exact-scope lease is active."""
    run = _authorized_run(session, run_id, scope_type=scope_type, scope_id=scope_id)
    current = _as_utc(now or datetime.now(timezone.utc))
    if run.status != ACTIVE_STATUS or _as_utc(run.lease_expires_at) <= current:
        raise VerificationRunAuthorizationError("Verification run is not active")
    sources = [item for item in run.transient_objects if item.get("role") == "source"]
    if len(sources) != 1 or not _safe_run_key(run.id, sources[0].get("key")):
        raise VerificationRunError("Transient source locator is invalid")
    source = sources[0]
    content = backend.download(source["key"])
    digest = hashlib.sha256(content).hexdigest()
    if digest != run.content_sha256 or len(content) != run.source_byte_size:
        raise VerificationRunError("Transient source failed immutable-byte verification")
    return AuthorizedRepresentation(
        representation_id=f"verification-run:{run.id}",
        canonical_work_id=run.canonical_work_id,
        content_object_id=f"verification-run:{run.id}:source",
        content_sha256=run.content_sha256,
        content=content,
        representation_kind=run.representation_kind,
        media_type=run.media_type,
        provenance=run.acquisition_route,
        scope_type=run.scope_type,
        scope_id=run.scope_id,
        identity_verdict=run.identity_verdict,
        identity_confidence=run.identity_confidence,
        completeness_verdict=run.completeness_verdict,
        text_quality=run.text_quality,
        edition_or_version=run.edition_or_version,
        created_at=run.started_at,
        admitted_at=None,
        verification_run_id=str(run.id),
    )


def add_transient_run_artifact(
    session: Session,
    backend: StorageBackend,
    run_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
    role: str,
    content: bytes,
) -> str:
    """Add one planned derivative so stale cleanup can always discover it."""
    if role not in _DERIVED_EXTENSIONS:
        raise VerificationRunError("Unsupported transient artifact role")
    if not content:
        raise VerificationRunError("Transient artifact cannot be empty")
    run = _authorized_run(session, run_id, scope_type=scope_type, scope_id=scope_id)
    if run.status != ACTIVE_STATUS:
        raise VerificationRunError("Transient artifacts require an active run")
    if any(item.get("role") == role for item in run.transient_objects):
        raise VerificationRunError(f"Transient artifact role already exists: {role}")
    total = run.transient_byte_size + len(content)
    if total > settings.VERIFICATION_RUN_MAX_TRANSIENT_MB * 1024 * 1024:
        raise VerificationRunError("Transient run exceeds the configured byte limit")
    digest = hashlib.sha256(content).hexdigest()
    key = f"verification-runs/{run.id}/{role}.{_DERIVED_EXTENSIONS[role]}"
    item = _transient_object(role=role, key=key, content=content)
    run.transient_objects = [*run.transient_objects, item]
    run.transient_object_count += 1
    run.transient_byte_size = total
    run.updated_at = datetime.now(timezone.utc)
    session.commit()
    backend.upload(content, key)
    return key


def renew_verification_run_lease(
    session: Session,
    run_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
) -> datetime:
    run = _authorized_run(session, run_id, scope_type=scope_type, scope_id=scope_id)
    if run.status != ACTIVE_STATUS:
        raise VerificationRunError("Only an active run lease can be renewed")
    current = _as_utc(now or datetime.now(timezone.utc))
    run.lease_expires_at = current + timedelta(
        seconds=max(1, settings.VERIFICATION_RUN_LEASE_SECONDS)
    )
    run.updated_at = current
    session.commit()
    return run.lease_expires_at


def cleanup_verification_run(
    session: Session,
    backend: StorageBackend,
    run_id: str | uuid.UUID,
    *,
    scope_type: str | None = None,
    scope_id: str | None = None,
    outcome: str = "success",
    now: datetime | None = None,
) -> bool:
    """Delete every planned transient object; safe to call repeatedly."""
    parsed_id = _run_id(run_id)
    run = session.get(VerificationRunRecord, parsed_id)
    if run is None:
        return True
    if scope_type is not None or scope_id is not None:
        requested_type, requested_id = _scope(scope_type or "", scope_id or "")
        if run.scope_type != requested_type or run.scope_id != requested_id:
            raise VerificationRunAuthorizationError(
                "Verification run does not exist in the requested scope"
            )
    if run.status in CLEANED_STATUSES:
        return True
    terminal_outcome = run.terminal_outcome or outcome.strip().casefold() or "failed"
    run.terminal_outcome = terminal_outcome
    run.status = CLEANUP_PENDING_STATUS
    run.cleanup_attempts += 1
    run.updated_at = _as_utc(now or datetime.now(timezone.utc))
    session.commit()

    remaining = []
    error_code = None
    for item in run.transient_objects:
        key = item.get("key")
        if not _safe_run_key(run.id, key):
            remaining.append(item)
            error_code = "invalid_transient_locator"
            continue
        try:
            backend.delete(key)
            if backend.exists(key):
                remaining.append(item)
                error_code = "object_delete_failed"
        except Exception:
            remaining.append(item)
            error_code = "object_delete_error"

    if remaining:
        run.transient_objects = remaining
        run.last_cleanup_error_code = error_code or "object_delete_failed"
        run.status = CLEANUP_PENDING_STATUS
        run.updated_at = datetime.now(timezone.utc)
        session.commit()
        return False

    run.transient_objects = []
    run.last_cleanup_error_code = None
    run.cleaned_at = datetime.now(timezone.utc)
    run.updated_at = run.cleaned_at
    run.status = {
        "success": "cleaned",
        "failed": "failed_cleaned",
        "abandoned": "abandoned_cleaned",
    }.get(terminal_outcome, "failed_cleaned")
    session.commit()
    return True


def cleanup_stale_verification_runs(
    session: Session,
    backend: StorageBackend,
    *,
    now: datetime | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    """Recover expired leases, report-persisted runs, and prior delete failures."""
    current = _as_utc(now or datetime.now(timezone.utc))
    records = session.scalars(
        select(VerificationRunRecord)
        .where(
            VerificationRunRecord.status.in_(_CLEANABLE_STALE_STATUSES),
            or_(
                VerificationRunRecord.status.in_(
                    {REPORT_PERSISTED_STATUS, CLEANUP_PENDING_STATUS}
                ),
                VerificationRunRecord.lease_expires_at <= current,
            ),
        )
        .order_by(VerificationRunRecord.started_at)
        .limit(max(1, min(batch_size, 1_000)))
    ).all()
    cleaned = 0
    pending = 0
    for run in records:
        outcome = run.terminal_outcome or "abandoned"
        if cleanup_verification_run(
            session, backend, run.id, outcome=outcome, now=current
        ):
            cleaned += 1
        else:
            pending += 1
    return {"runs_cleaned": cleaned, "runs_pending": pending}


def execute_verification_run(
    session_factory,
    backend: StorageBackend,
    request: VerificationRunRequest,
    processor: Callable[
        [Session, AuthorizedRepresentation, uuid.UUID],
        VerificationEvidenceArtifact,
    ],
) -> VerificationRunExecutionResult:
    """Execute, persist the report, and clean on both success and failure."""
    with session_factory() as session:
        run = begin_verification_run(session, backend, request)
        run_id = run.id
    try:
        with session_factory() as session:
            source = load_verification_run_source(
                session,
                backend,
                run_id,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
            )
            artifact = processor(session, source, run_id)
            report = persist_verification_report(
                session,
                artifact,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                verification_run_id=run_id,
            )
            session.commit()
            report_id = report.id
            report_version = report.report_version
    except BaseException:
        try:
            with session_factory() as cleanup_session:
                cleanup_verification_run(
                    cleanup_session,
                    backend,
                    run_id,
                    scope_type=request.scope_type,
                    scope_id=request.scope_id,
                    outcome="failed",
                )
        except Exception as cleanup_error:
            logger.error(
                "Failed verification run requires scheduled cleanup retry: %s",
                type(cleanup_error).__name__,
            )
        raise

    with session_factory() as cleanup_session:
        cleaned = cleanup_verification_run(
            cleanup_session,
            backend,
            run_id,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            outcome="success",
        )
    if not cleaned:
        raise VerificationRunCleanupPending(run_id, report_id)
    return VerificationRunExecutionResult(
        run_id=run_id,
        report_id=report_id,
        report_version=report_version,
    )


def complete_active_verification_run(
    session_factory,
    backend: StorageBackend,
    run_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
    processor: Callable[
        [Session, AuthorizedRepresentation, uuid.UUID],
        list[VerificationEvidenceArtifact],
    ],
) -> VerificationRunBatchExecutionResult:
    """Persist all claim reports for one already-acquired source, then clean it.

    Retrieval and verification may be separate durable job stages. This is the
    only supported completion boundary for a transient run created by the
    retrieval stage: all reports are committed before the shared idempotent
    cleanup path removes the source and derivatives.
    """
    parsed_id = _run_id(run_id)
    report_refs: list[tuple[uuid.UUID, int]] = []
    try:
        with session_factory() as session:
            source = load_verification_run_source(
                session,
                backend,
                parsed_id,
                scope_type=scope_type,
                scope_id=scope_id,
            )
            artifacts = processor(session, source, parsed_id)
            if not artifacts:
                raise VerificationRunError(
                    "Transient verification completion requires at least one report"
                )
            for artifact in artifacts:
                report = persist_verification_report(
                    session,
                    artifact,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    verification_run_id=parsed_id,
                    mark_run_persisted=False,
                )
                report_refs.append((report.id, report.report_version))
            run = session.get(VerificationRunRecord, parsed_id)
            run.status = REPORT_PERSISTED_STATUS
            run.terminal_outcome = "success"
            session.flush()
            session.commit()
    except BaseException:
        try:
            with session_factory() as cleanup_session:
                cleanup_verification_run(
                    cleanup_session,
                    backend,
                    parsed_id,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    outcome="failed",
                )
        except Exception as cleanup_error:
            logger.error(
                "Failed verification run requires scheduled cleanup retry: %s",
                type(cleanup_error).__name__,
            )
        raise

    with session_factory() as cleanup_session:
        cleaned = cleanup_verification_run(
            cleanup_session,
            backend,
            parsed_id,
            scope_type=scope_type,
            scope_id=scope_id,
            outcome="success",
        )
    if not cleaned:
        raise VerificationRunCleanupPending(parsed_id, report_refs[-1][0])
    return VerificationRunBatchExecutionResult(
        run_id=parsed_id,
        reports=tuple(report_refs),
    )


def _authorized_run(session, run_id, *, scope_type, scope_id):
    parsed_id = _run_id(run_id)
    requested_type, requested_id = _scope(scope_type, scope_id)
    run = session.get(VerificationRunRecord, parsed_id)
    if (
        run is None
        or run.scope_type != requested_type
        or run.scope_id != requested_id
    ):
        raise VerificationRunAuthorizationError(
            "Verification run does not exist in the requested scope"
        )
    return run


def _run_id(value) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise VerificationRunAuthorizationError("Invalid verification run ID") from exc


def _scope(scope_type: str, scope_id: str) -> tuple[str, str]:
    normalized_type = scope_type.strip().casefold()
    normalized_id = scope_id.strip()
    if not normalized_type or not normalized_id:
        raise VerificationRunAuthorizationError("Verification run scope is required")
    return normalized_type, normalized_id


def _transient_object(*, role: str, key: str, content: bytes) -> dict:
    return {
        "role": role,
        "key": key,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }


def _safe_run_key(run_id: uuid.UUID, key) -> bool:
    if not isinstance(key, str) or "\\" in key:
        return False
    parts = key.split("/")
    return (
        len(parts) == 3
        and parts[0] == "verification-runs"
        and parts[1] == str(run_id)
        and bool(parts[2])
        and parts[2] not in {".", ".."}
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
