"""Zero-trust temporary intake and cleanup for student paper jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job, JobStage, JobStatus
from app.services.file_safety import (
    FileSafetyUnavailable,
    SafetyVerdict,
    inspect_uploaded_pdf,
    scan_with_clamd,
)
from app.services.paper_retention import (
    PaperRetentionPolicyError,
    resolve_paper_retention_policy,
)
from app.services.storage.backend import StorageBackend


PAPER_UPLOAD_POLICY_VERSION = "paper-upload-v1"
PDF_MEDIA_TYPE = "application/pdf"
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ALLOWED_MEDIA_TYPES = {PDF_MEDIA_TYPE: ".pdf", DOCX_MEDIA_TYPE: ".docx"}
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 200
_EXECUTABLE_SUFFIXES = {
    ".bat", ".class", ".cmd", ".com", ".dll", ".exe", ".hta", ".jar",
    ".js", ".lnk", ".msi", ".ps1", ".scr", ".vbs",
}


class PaperUploadError(ValueError):
    """The uploaded paper failed intake or immutable-object checks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def inspect_paper_upload(content: bytes, *, filename: str, media_type: str) -> dict:
    """Return bounded safety evidence or reject before object storage."""
    expected_suffix = ALLOWED_MEDIA_TYPES.get(media_type)
    suffix = Path(filename or "").suffix.casefold()
    if expected_suffix is None or suffix != expected_suffix:
        raise PaperUploadError("type_mismatch", "Paper type and filename extension do not match")
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if not content:
        raise PaperUploadError("empty_upload", "Paper upload is empty")
    if len(content) > maximum:
        raise PaperUploadError("file_too_large", "Paper exceeds the configured upload limit")

    if expected_suffix == ".pdf":
        try:
            report = inspect_uploaded_pdf(content)
        except FileSafetyUnavailable as exc:
            raise PaperUploadError("malware_scan_unavailable", str(exc)) from exc
        if report.verdict is SafetyVerdict.REJECTED:
            raise PaperUploadError(
                "unsafe_pdf",
                "; ".join(report.findings) or "PDF safety inspection rejected the paper",
            )
        return {
            "policy_version": PAPER_UPLOAD_POLICY_VERSION,
            "structural_verdict": report.structural_verdict.value,
            "malware_verdict": report.malware_verdict.value,
            "findings": list(report.findings),
        }

    archive_evidence = _inspect_docx_archive(content)
    try:
        malware_verdict, scanner_detail = scan_with_clamd(content)
    except FileSafetyUnavailable as exc:
        if settings.MALWARE_SCAN_REQUIRED:
            raise PaperUploadError("malware_scan_unavailable", str(exc)) from exc
        malware_verdict, scanner_detail = SafetyVerdict.NOT_ASSESSED, str(exc)
    if malware_verdict is SafetyVerdict.REJECTED:
        raise PaperUploadError("malware_rejected", "Malware scanner rejected the paper")
    if malware_verdict is SafetyVerdict.UNAVAILABLE and settings.MALWARE_SCAN_REQUIRED:
        raise PaperUploadError("malware_scan_unavailable", scanner_detail)
    return {
        "policy_version": PAPER_UPLOAD_POLICY_VERSION,
        "structural_verdict": "clean",
        "malware_verdict": malware_verdict.value,
        "scanner_detail": scanner_detail,
        **archive_evidence,
    }


def create_paper_job(
    session: Session,
    backend: StorageBackend,
    *,
    content: bytes,
    filename: str,
    media_type: str,
    title: str | None = None,
    store_only: bool = False,
    paper_retention_mode: str | None = None,
    scope_type: str = "personal_owner",
    scope_id: str,
    now: datetime | None = None,
) -> Job:
    """Commit a recoverable locator before uploading immutable paper bytes."""
    safe_name = _safe_filename(filename)
    evidence = inspect_paper_upload(content, filename=safe_name, media_type=media_type)
    requested_retention = paper_retention_mode or settings.PAPER_RETENTION_MODE
    try:
        retention_policy = resolve_paper_retention_policy(requested_retention)
    except PaperRetentionPolicyError as exc:
        raise PaperUploadError(exc.code, str(exc)) from exc
    evidence["paper_retention_mode"] = retention_policy.mode.value
    evidence["paper_retention_policy_version"] = "paper-retention-v1"
    normalized_scope = scope_type.strip().casefold()
    normalized_id = scope_id.strip()
    if normalized_scope not in {
        "personal_owner", "assessment", "course_offering", "institution"
    } or not normalized_id:
        raise PaperUploadError("scope_invalid", "Paper upload authorization scope is invalid")
    current = _as_utc(now or datetime.now(timezone.utc))
    job_id = uuid.uuid4()
    suffix = ALLOWED_MEDIA_TYPES[media_type]
    key = f"paper-jobs/{job_id}/input{suffix}"
    digest = hashlib.sha256(content).hexdigest()
    job = Job(
        id=job_id,
        filename=safe_name,
        title=(title or "").strip()[:500] or None,
        status=JobStatus.PENDING,
        stage=JobStage.UPLOADED,
        paper_version_id=f"paper:{job_id}:{digest[:16]}",
        scope_type=normalized_scope,
        scope_id=normalized_id,
        input_storage_key=key,
        input_sha256=digest,
        input_media_type=media_type,
        input_byte_size=len(content),
        input_expires_at=current + timedelta(seconds=max(60, settings.PAPER_UPLOAD_LEASE_SECONDS)),
        store_only=store_only,
        upload_evidence=evidence,
        created_at=current,
        updated_at=current,
    )
    session.add(job)
    session.commit()
    try:
        backend.upload(content, key)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.stage = JobStage.FAILED
        job.error_message = "paper_object_upload_failed"
        job.updated_at = datetime.now(timezone.utc)
        session.commit()
        raise PaperUploadError("storage_unavailable", "Paper storage was unavailable") from exc
    return job


def load_paper_job_input(session: Session, backend: StorageBackend, job_id) -> tuple[Job, bytes]:
    job = _job(session, job_id)
    if not job.input_storage_key or job.input_deleted_at is not None:
        raise PaperUploadError("paper_input_unavailable", "Temporary paper input is unavailable")
    try:
        content = backend.download(job.input_storage_key)
    except FileNotFoundError as exc:
        raise PaperUploadError("paper_input_missing", "Temporary paper input is missing") from exc
    if len(content) != job.input_byte_size or hashlib.sha256(content).hexdigest() != job.input_sha256:
        raise PaperUploadError("paper_input_tampered", "Temporary paper input failed immutable-byte verification")
    return job, content


def cleanup_paper_job_input(
    session: Session,
    backend: StorageBackend,
    job_id,
    *,
    now: datetime | None = None,
) -> bool:
    job = _job(session, job_id)
    if job.input_deleted_at is not None or not job.input_storage_key:
        return True
    key = job.input_storage_key
    try:
        deleted = backend.delete(key)
        absent = not backend.exists(key)
    except Exception:
        deleted = absent = False
    if not deleted or not absent:
        job.error_message = job.error_message or "paper_input_cleanup_pending"
        job.updated_at = _as_utc(now or datetime.now(timezone.utc))
        session.commit()
        return False
    job.input_storage_key = None
    job.input_deleted_at = _as_utc(now or datetime.now(timezone.utc))
    job.updated_at = job.input_deleted_at
    session.commit()
    return True


def cleanup_stale_paper_job_inputs(
    session: Session,
    backend: StorageBackend,
    *,
    now: datetime | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    current = _as_utc(now or datetime.now(timezone.utc))
    jobs = session.scalars(
        select(Job)
        .where(
            Job.input_storage_key.is_not(None),
            Job.input_deleted_at.is_(None),
            or_(
                Job.status.in_({JobStatus.COMPLETED, JobStatus.FAILED}),
                Job.input_expires_at <= current,
            ),
        )
        .order_by(Job.created_at)
        .limit(max(1, min(batch_size, 1_000)))
    ).all()
    cleaned = sum(cleanup_paper_job_input(session, backend, job.id, now=current) for job in jobs)
    return {"jobs_cleaned": cleaned, "jobs_pending": len(jobs) - cleaned}


def _inspect_docx_archive(content: bytes) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, ValueError) as exc:
        raise PaperUploadError("invalid_docx", "DOCX is not a valid ZIP package") from exc
    with archive:
        entries = archive.infolist()
        if not entries or len(entries) > MAX_DOCX_ENTRIES:
            raise PaperUploadError("unsafe_docx_archive", "DOCX has an unsafe archive entry count")
        names = set()
        total = 0
        for entry in entries:
            name = entry.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not name:
                raise PaperUploadError("unsafe_docx_path", "DOCX contains an unsafe archive path")
            if entry.flag_bits & 0x1:
                raise PaperUploadError("encrypted_docx", "Encrypted DOCX entries are unsupported")
            lower = name.casefold()
            if (
                lower.endswith(tuple(_EXECUTABLE_SUFFIXES))
                or lower == "word/vbaproject.bin"
                or lower.startswith("word/embeddings/")
                or lower.startswith("word/activex/")
            ):
                raise PaperUploadError("active_docx_content", "DOCX contains active or embedded content")
            total += entry.file_size
            if total > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise PaperUploadError("docx_expansion_limit", "DOCX exceeds the uncompressed-size limit")
            compressed = max(1, entry.compress_size)
            if entry.file_size > 1_000_000 and entry.file_size / compressed > MAX_DOCX_COMPRESSION_RATIO:
                raise PaperUploadError("docx_compression_ratio", "DOCX has an unsafe compression ratio")
            names.add(lower)
        if "[content_types].xml" not in names or "word/document.xml" not in names:
            raise PaperUploadError("invalid_docx", "DOCX is missing its main document parts")
        if archive.testzip() is not None:
            raise PaperUploadError("corrupt_docx", "DOCX failed its archive integrity check")
    return {"archive_entry_count": len(entries), "uncompressed_byte_size": total}


def _safe_filename(value: str) -> str:
    name = Path(value or "paper").name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if not name:
        raise PaperUploadError("filename_invalid", "Paper filename is invalid")
    return name[:255]


def _job(session: Session, value) -> Job:
    try:
        job_id = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise PaperUploadError("job_id_invalid", "Invalid paper job ID") from exc
    job = session.get(Job, job_id)
    if job is None:
        raise PaperUploadError("job_not_found", "Paper job does not exist")
    return job


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
