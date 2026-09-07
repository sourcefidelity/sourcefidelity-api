"""Lease-backed execution and guaranteed cleanup for one-run source use."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import uuid

from sqlalchemy import or_, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

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
from app.services.workflow_retry import is_retryable
from app.services.workflow_execution import WorkflowOwnershipLost, stage_execution
from app.services.upload_completion import upload_confirmed, next_check, due_clause


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


class VerificationRunBusy(VerificationRunError):
    """Another connection is processing or cleaning this transient run."""


@contextmanager
def _locked_run_session(session: Session, run_id):
    """Serialize processing/cleanup across commits on one owned connection.

    Reuse a paper stage's already-owned connection; standalone calls pin their
    own. SQLite preserves sequential test behavior, not concurrency guarantees.
    Callers must have committed their writes before entering this boundary.
    """
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        yield session
        return
    if session.new or session.dirty or session.deleted:
        raise VerificationRunError("Commit pending changes before acquiring a run")
    session.rollback()
    lock_id = int.from_bytes(hashlib.sha256(
        f"verification-run:{_run_id(run_id)}".encode()).digest()[:8], "big", signed=True)
    connection_context = nullcontext(bind) if isinstance(bind, Connection) else bind.connect()
    try:
        with connection_context as connection:
            acquired = connection.execute(text("SELECT pg_try_advisory_lock(:id)"),
                {"id": lock_id}).scalar_one()
            if not acquired:
                connection.rollback()
                raise VerificationRunBusy("Transient source is currently in use")
            try:
                factory = sessionmaker(bind=connection, class_=type(session),
                    expire_on_commit=session.expire_on_commit)
                with stage_execution(connection, factory) as owner:
                    with owner.session_factory() as owned_session:
                        yield owned_session
            finally:
                if not connection.invalidated and not connection.closed:
                    try:
                        connection.rollback()
                        connection.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": lock_id})
                        connection.commit()
                    except Exception:
                        connection.invalidate()
    finally:
        session.expire_all()


@contextmanager
def _processing_session(session_factory, run_id):
    with session_factory() as session:
        with _locked_run_session(session, run_id) as owned_session:
            yield owned_session


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
    lease_seconds: int | None = None,
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
        + timedelta(
            seconds=max(
                1,
                settings.VERIFICATION_RUN_LEASE_SECONDS
                if lease_seconds is None
                else lease_seconds,
            )
        ),
        updated_at=current,
    )
    session.add(run)
    session.commit()
    try:
        with _locked_run_session(session, run_id) as owned:
            current_run = owned.get(VerificationRunRecord, run_id)
            if current_run.status != PLANNING_STATUS:
                raise VerificationRunError("Transient upload no longer owns an allocating run")
            receipt = backend.upload(representation.content, key)
            current_run.transient_objects = [
                {**item, "upload_confirmed": upload_confirmed(receipt)}
                for item in current_run.transient_objects]
            current_run.status = ACTIVE_STATUS
            current_run.updated_at = datetime.now(timezone.utc)
            owned.commit()
    except WorkflowOwnershipLost:
        raise
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
    if session.get_bind().dialect.name == "postgresql":
        session.refresh(run)
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
    with _locked_run_session(session, run_id) as owned:
        return _add_transient_run_artifact(owned, backend, run_id,
            scope_type=scope_type, scope_id=scope_id, role=role, content=content)


def _add_transient_run_artifact(session, backend, run_id, *, scope_type, scope_id, role, content):
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
    receipt = backend.upload(content, key)
    run.transient_objects = [
        {**item, "upload_confirmed": upload_confirmed(receipt)} if item["key"] == key else item
        for item in run.transient_objects]
    session.commit()
    return key


def renew_verification_run_lease(
    session: Session,
    run_id: str | uuid.UUID,
    *,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
    lease_seconds: int | None = None,
) -> datetime:
    with _locked_run_session(session, run_id) as owned_session:
        return _renew_verification_run_lease(owned_session, run_id,
            scope_type=scope_type, scope_id=scope_id, now=now, lease_seconds=lease_seconds)


def _renew_verification_run_lease(session, run_id, *, scope_type, scope_id, now, lease_seconds):
    run = _authorized_run(session, run_id, scope_type=scope_type, scope_id=scope_id)
    current = _as_utc(now or datetime.now(timezone.utc))
    if run.status != ACTIVE_STATUS or _as_utc(run.lease_expires_at) <= current:
        raise VerificationRunError("Only an unexpired active run lease can be renewed")
    run.lease_expires_at = current + timedelta(
        seconds=max(
            1,
            settings.VERIFICATION_RUN_LEASE_SECONDS
            if lease_seconds is None
            else lease_seconds,
        )
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
    expired_before: datetime | None = None,
) -> bool:
    """Delete planned objects under the processor lock; busy runs stay pending.

    Scheduled callers also pass expired_before to recheck eligibility after
    acquiring ownership, rather than trusting a potentially stale scan.
    """
    try:
        with _locked_run_session(session, run_id) as owned_session:
            return _cleanup_verification_run(owned_session, backend, run_id,
                scope_type=scope_type, scope_id=scope_id, outcome=outcome,
                now=now, expired_before=expired_before)
    except VerificationRunBusy:
        return False


def _cleanup_verification_run(session, backend, run_id, *, scope_type, scope_id,
                              outcome, now, expired_before):
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
    if expired_before is not None and (
        run.status not in _CLEANABLE_STALE_STATUSES or (
            run.status not in {REPORT_PERSISTED_STATUS, CLEANUP_PENDING_STATUS}
            and _as_utc(run.lease_expires_at) > _as_utc(expired_before)
        )
    ):
        return False
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
            if not backend.delete(key) or backend.exists(key):
                remaining.append(item)
                error_code = "object_delete_failed"
            elif item.get("upload_confirmed") is not True:
                remaining.append({**item, "cleanup_next_check_at": next_check(
                    run.started_at, now or datetime.now(timezone.utc))})
                error_code = error_code or "upload_completion_unresolved"
        except Exception:
            remaining.append(item)
            error_code = "object_delete_error"

    if remaining:
        if error_code != "upload_completion_unresolved":
            remaining = [{key: value for key, value in item.items() if key != "cleanup_next_check_at"}
                         for item in remaining]
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
        select(VerificationRunRecord.id)
        .where(
            VerificationRunRecord.status.in_(_CLEANABLE_STALE_STATUSES),
            due_clause(VerificationRunRecord.transient_objects[0]["cleanup_next_check_at"], current),
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
    for run_id in records:
        if cleanup_verification_run(
            session, backend, run_id, outcome="abandoned", now=current,
            expired_before=current,
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
        with _processing_session(session_factory, run_id) as session:
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
    except (WorkflowOwnershipLost, VerificationRunBusy):
        # A competing/lost owner cannot clean another processor's source.
        raise
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
    retain_on_retryable_failure: bool = False,
) -> VerificationRunBatchExecutionResult:
    """Persist all claim reports for one already-acquired source, then clean it.

    Retrieval and verification may be separate durable job stages. This is the
    only supported completion boundary for a transient run created by the
    retrieval stage: all reports are committed before the shared idempotent
    cleanup path removes the source and derivatives.

    A bounded paper-workflow retry may preserve the existing lease and bytes
    on operational failure. The failed report transaction still rolls back;
    this never renews a lease or makes partial reports reusable. Standalone
    callers retain immediate failure cleanup unless they explicitly opt in.
    """
    parsed_id = _run_id(run_id)
    report_refs: list[tuple[uuid.UUID, int]] = []
    try:
        with _processing_session(session_factory, parsed_id) as session:
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
    except (WorkflowOwnershipLost, VerificationRunBusy):
        raise
    except BaseException as exc:
        if retain_on_retryable_failure and isinstance(exc, Exception) and is_retryable(exc):
            raise
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
        "upload_confirmed": False,
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
