"""Sanitized, expiring paper artifacts for the authenticated report surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, replace
from collections import OrderedDict
import hashlib
import io
import re
from threading import Lock
import uuid
import zipfile

import fitz
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job
from app.models.report import Report, ReportPaperArtifactRecord
from app.services.docx_presentation import (
    DocxPresentationError,
    render_docx_to_pdf,
)
from app.services.paper_upload import DOCX_MEDIA_TYPE, PDF_MEDIA_TYPE
from app.services.presentation_anchors import bind_citations_to_pdf
from app.services.schemas import InTextCitation, ParsedReference
from app.services.reference_layout import extract_reference_layout_from_bytes
from app.services.storage.backend import StorageBackend
from app.services.text_extractor import extract_qualified_text_from_bytes


REPORT_PAPER_ARTIFACT_VERSION = "report-paper-artifact-v2"
_ARTIFACT_CACHE_MAX_BYTES = 128 * 1024 * 1024
_artifact_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_artifact_cache_bytes = 0
_artifact_cache_lock = Lock()


class ReportPaperArtifactError(ValueError):
    """The report marking copy failed sanitization or authorization."""


@dataclass(frozen=True)
class _PreparedArtifact:
    source_bytes: bytes
    source_evidence: dict
    source_suffix: str
    artifact_kind: str
    presentation_status: str
    presentation_bytes: bytes | None
    presentation_evidence: dict | None


def ensure_report_paper_artifact(
    session: Session,
    backend: StorageBackend,
    *,
    job: Job,
    content: bytes,
    citations: list[InTextCitation] | None = None,
    references: list[ParsedReference] | None = None,
    citation_format: str = "apa",
    now: datetime | None = None,
) -> ReportPaperArtifactRecord | None:
    """Create one separately hashed pending marking copy before input cleanup."""
    if not settings.REPORT_PAPER_COPY_ENABLED:
        return None
    existing = session.scalar(
        select(ReportPaperArtifactRecord).where(
            ReportPaperArtifactRecord.job_id == job.id
        )
    )
    if existing is not None:
        try:
            _verify_stored(backend, existing)
        except FileNotFoundError:
            prepared = _with_anchors(
                _prepare_artifact(job, content), citations, job.input_media_type
            )
            if hashlib.sha256(prepared.source_bytes).hexdigest() != existing.content_sha256:
                raise ReportPaperArtifactError(
                    "Pending marking-copy recovery did not reproduce its immutable bytes"
                )
            if not existing.storage_key:
                raise ReportPaperArtifactError("Pending marking-copy locator is unavailable")
            backend.upload(prepared.source_bytes, existing.storage_key)
            _verify_stored(backend, existing)
        if existing.presentation_storage_key:
            try:
                _verify_presentation_stored(backend, existing)
            except FileNotFoundError:
                prepared = _with_anchors(
                    _prepare_artifact(job, content), citations, job.input_media_type
                )
                if prepared.presentation_bytes is None:
                    raise ReportPaperArtifactError(
                        "Pending presentation recovery could not reproduce its derivative"
                    )
                new_digest = hashlib.sha256(prepared.presentation_bytes).hexdigest()
                if new_digest != existing.presentation_sha256:
                    if existing.report_id is not None or not _same_presentation_surface(
                        existing.presentation_evidence, prepared.presentation_evidence
                    ):
                        raise ReportPaperArtifactError(
                            "Pending presentation recovery did not reproduce its accepted surface"
                        )
                    # LibreOffice can vary non-visual PDF serialization between
                    # runs. Before report attachment only, a missing object may
                    # be recovered only when its validated page/text surface agrees.
                    existing.presentation_sha256 = new_digest
                    existing.presentation_byte_size = len(prepared.presentation_bytes)
                    existing.presentation_evidence = prepared.presentation_evidence
                    session.commit()
                backend.upload(prepared.presentation_bytes, existing.presentation_storage_key)
                _verify_presentation_stored(backend, existing)
        return existing
    prepared = _with_anchors(
        _prepare_artifact(job, content), citations, job.input_media_type
    )
    if references and prepared.presentation_bytes is not None and job.input_media_type == DOCX_MEDIA_TYPE:
        # Reuse the exact retained rendering, not a second conversion. This is
        # navigation only; original-submission counts never consume this PDF.
        try:
            layout = extract_reference_layout_from_bytes(
                prepared.presentation_bytes, 'presentation.pdf', references=references,
                citation_format=citation_format)
            prepared.presentation_evidence['submitted_reference_navigation'] = {
                'input_sha256': hashlib.sha256(content).hexdigest(),
                'layout': layout.model_dump(mode='json'),
            }
        except ValueError:
            pass  # No speculative location if the optional layout is unavailable.
    current = _as_utc(now or datetime.now(timezone.utc))
    artifact_id = uuid.uuid4()
    key = f"report-paper-artifacts/{artifact_id}/semantic-source{prepared.source_suffix}"
    digest = hashlib.sha256(prepared.source_bytes).hexdigest()
    presentation_key = (
        key
        if prepared.presentation_bytes is prepared.source_bytes
        else (
            f"report-paper-artifacts/{artifact_id}/canonical-presentation.pdf"
            if prepared.presentation_bytes is not None
            else None
        )
    )
    presentation_digest = (
        hashlib.sha256(prepared.presentation_bytes).hexdigest()
        if prepared.presentation_bytes is not None
        else None
    )
    record = ReportPaperArtifactRecord(
        id=artifact_id,
        job_id=job.id,
        paper_version_id=job.paper_version_id,
        scope_type=job.scope_type,
        scope_id=job.scope_id,
        storage_key=key,
        content_sha256=digest,
        media_type=job.input_media_type,
        byte_size=len(prepared.source_bytes),
        artifact_kind=prepared.artifact_kind,
        presentation_status=prepared.presentation_status,
        presentation_storage_key=presentation_key,
        presentation_sha256=presentation_digest,
        presentation_media_type=(PDF_MEDIA_TYPE if presentation_key else None),
        presentation_byte_size=(len(prepared.presentation_bytes) if prepared.presentation_bytes is not None else None),
        presentation_evidence=prepared.presentation_evidence,
        sanitization_evidence={
            "artifact_version": REPORT_PAPER_ARTIFACT_VERSION,
            "input_sha256": job.input_sha256,
            "output_sha256": digest,
            **prepared.source_evidence,
        },
        created_at=current,
        expires_at=current
        + timedelta(seconds=max(60, settings.PAPER_UPLOAD_LEASE_SECONDS)),
    )
    session.add(record)
    session.commit()
    try:
        backend.upload(prepared.source_bytes, key)
        if presentation_key != key and prepared.presentation_bytes is not None:
            backend.upload(prepared.presentation_bytes, presentation_key)
        _verify_stored(backend, record)
        if presentation_key:
            _verify_presentation_stored(backend, record)
    except Exception as exc:
        raise ReportPaperArtifactError(
            "The report marking copy could not be stored and verified"
        ) from exc
    return record


def attach_report_paper_artifact(
    session: Session,
    *,
    job: Job,
    report: Report,
    now: datetime | None = None,
) -> ReportPaperArtifactRecord | None:
    record = session.scalar(
        select(ReportPaperArtifactRecord).where(
            ReportPaperArtifactRecord.job_id == job.id
        )
    )
    if record is None:
        return None
    if (
        record.paper_version_id != job.paper_version_id
        or record.scope_type != job.scope_type
        or record.scope_id != job.scope_id
        or report.job_id != job.id
    ):
        raise ReportPaperArtifactError("The marking copy does not match its report scope")
    # One immutable paper surface is shared by every immutable report version
    # for this job. Preserve the first attachment instead of moving the artifact
    # away from an older report when a successor is created.
    if record.report_id is None:
        record.report_id = report.id
    current = _as_utc(now or datetime.now(timezone.utc))
    record.expires_at = current + timedelta(
        days=max(1, settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS)
    )
    session.flush()
    return record


def paper_surface_descriptor(record: ReportPaperArtifactRecord | None) -> dict:
    if record is None:
        return {
            "status": "bounded_citation_spans_only",
            "reason_code": "report_marking_copy_unavailable",
            "message": (
                "The report has exact citation-dependent spans, but no retained "
                "marking copy is available."
            ),
        }
    anchor_artifact = (record.presentation_evidence or {}).get("citation_anchors") or {}
    anchor_hash_matches = bool(
        record.presentation_sha256
        and anchor_artifact.get("presentation_sha256") == record.presentation_sha256
    )
    anchors = anchor_artifact.get("anchors", []) if anchor_hash_matches else []
    return {
        "status": "page_faithful_ready"
        if record.presentation_status == "page_faithful_ready"
        else "presentation_source_retained",
        "reason_code": record.presentation_status,
        "artifact_id": str(record.id),
        "media_type": record.presentation_media_type or record.media_type,
        "source_media_type": record.media_type,
        "expires_at": _as_utc(record.expires_at).isoformat(),
        "font_substitution_risk": bool(
            (record.presentation_evidence or {}).get("font_substitution_risk", False)
        ),
        "page_dimensions": list(
            (record.presentation_evidence or {}).get("page_dimensions", [])
        ),
        "anchor_status": (
            anchor_artifact.get("status", "not_assessed")
            if anchor_hash_matches
            else "not_assessed"
        ),
        "anchor_reason_code": (
            "presentation_hash_bound"
            if anchor_hash_matches
            else "presentation_hash_binding_unavailable"
        ),
        "citation_anchor_count": (
            anchor_artifact.get("citation_count", 0) if anchor_hash_matches else 0
        ),
        "matched_citation_anchor_count": (
            anchor_artifact.get("matched_citation_count", 0)
            if anchor_hash_matches
            else 0
        ),
        "page_localized_citation_count": (
            anchor_artifact.get("page_localized_citation_count", 0)
            if anchor_hash_matches
            else 0
        ),
        "structurally_localized_citation_count": (
            anchor_artifact.get("structurally_localized_citation_count", 0)
            if anchor_hash_matches
            else 0
        ),
        "citation_anchors": anchors,
        "presentation_sha256": record.presentation_sha256,
        "submitted_reference_navigation": (record.presentation_evidence or {}).get('submitted_reference_navigation'),
        "message": (
            "A sanitized page-faithful marking copy is retained with this report."
            if record.presentation_status == "page_faithful_ready"
            else "A sanitized word-processing marking source is retained; a deterministic page-faithful PDF rendering is still required."
        ),
    }


def load_authorized_report_paper_artifact(
    session: Session,
    backend: StorageBackend,
    *,
    report_id: str | uuid.UUID,
    artifact_id: str | uuid.UUID,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
) -> tuple[ReportPaperArtifactRecord, bytes]:
    """Load exact-scope bytes for a future authenticated viewer; no route is added."""
    try:
        parsed_report = uuid.UUID(str(report_id))
        parsed_artifact = uuid.UUID(str(artifact_id))
    except (TypeError, ValueError) as exc:
        raise ReportPaperArtifactError("Marking copy is unavailable") from exc
    record = session.get(ReportPaperArtifactRecord, parsed_artifact)
    report = session.get(Report, parsed_report)
    current = _as_utc(now or datetime.now(timezone.utc))
    if (
        record is None
        or report is None
        or report.job_id != record.job_id
        or record.scope_type != scope_type.strip()
        or record.scope_id != scope_id.strip()
        or record.deleted_at is not None
        or _as_utc(record.expires_at) <= current
        or not record.storage_key
    ):
        raise ReportPaperArtifactError("Marking copy is unavailable")
    if record.presentation_status == "page_faithful_ready":
        return record, _verify_presentation_stored(backend, record)
    return record, _verify_stored(backend, record)


def cleanup_report_paper_artifact(
    session: Session,
    backend: StorageBackend,
    record: ReportPaperArtifactRecord,
    *,
    now: datetime | None = None,
) -> bool:
    if record.deleted_at is not None or (
        not record.storage_key and not record.presentation_storage_key
    ):
        return True
    keys = list(dict.fromkeys(filter(None, [record.presentation_storage_key, record.storage_key])))
    try:
        deleted = all([backend.delete(key) for key in keys])
        absent = all(not backend.exists(key) for key in keys)
    except Exception:
        return False
    if not deleted or not absent:
        return False
    record.storage_key = None
    record.presentation_storage_key = None
    record.deleted_at = _as_utc(now or datetime.now(timezone.utc))
    _cache_evict_record(str(record.id))
    session.commit()
    return True


def cleanup_expired_report_paper_artifacts(
    session: Session,
    backend: StorageBackend,
    *,
    now: datetime | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    current = _as_utc(now or datetime.now(timezone.utc))
    records = session.scalars(
        select(ReportPaperArtifactRecord)
        .where(
            (
                ReportPaperArtifactRecord.storage_key.is_not(None)
                | ReportPaperArtifactRecord.presentation_storage_key.is_not(None)
            ),
            ReportPaperArtifactRecord.deleted_at.is_(None),
            ReportPaperArtifactRecord.expires_at <= current,
        )
        .order_by(ReportPaperArtifactRecord.created_at)
        .limit(max(1, min(batch_size, 1_000)))
        .with_for_update(skip_locked=True)
    ).all()
    cleaned = sum(
        cleanup_report_paper_artifact(session, backend, record, now=current)
        for record in records
    )
    return {"artifacts_cleaned": cleaned, "artifacts_pending": len(records) - cleaned}


def _verify_stored(backend: StorageBackend, record: ReportPaperArtifactRecord) -> bytes:
    if not record.storage_key:
        raise ReportPaperArtifactError("Marking copy is unavailable")
    if not backend.exists(record.storage_key):
        raise FileNotFoundError("Marking-copy object is unavailable")
    cache_key = (str(record.id), record.content_sha256)
    content = _cache_get(cache_key)
    if content is None:
        content = backend.download(record.storage_key)
    if len(content) != record.byte_size or hashlib.sha256(content).hexdigest() != record.content_sha256:
        raise ReportPaperArtifactError("Marking copy failed immutable-byte verification")
    _cache_put(cache_key, content)
    return content


def _verify_presentation_stored(
    backend: StorageBackend, record: ReportPaperArtifactRecord
) -> bytes:
    if (
        not record.presentation_storage_key
        or record.presentation_byte_size is None
        or not record.presentation_sha256
    ):
        raise ReportPaperArtifactError("Page-faithful presentation is unavailable")
    if not backend.exists(record.presentation_storage_key):
        raise FileNotFoundError("Presentation object is unavailable")
    cache_key = (str(record.id), record.presentation_sha256)
    content = _cache_get(cache_key)
    if content is None:
        content = backend.download(record.presentation_storage_key)
    if (
        len(content) != record.presentation_byte_size
        or hashlib.sha256(content).hexdigest() != record.presentation_sha256
    ):
        raise ReportPaperArtifactError("Presentation PDF failed immutable-byte verification")
    _cache_put(cache_key, content)
    return content


def _cache_get(key: tuple[str, str]) -> bytes | None:
    with _artifact_cache_lock:
        content = _artifact_cache.get(key)
        if content is not None:
            _artifact_cache.move_to_end(key)
        return content


def _cache_put(key: tuple[str, str], content: bytes) -> None:
    global _artifact_cache_bytes
    if len(content) > _ARTIFACT_CACHE_MAX_BYTES:
        return
    with _artifact_cache_lock:
        previous = _artifact_cache.pop(key, None)
        if previous is not None:
            _artifact_cache_bytes -= len(previous)
        _artifact_cache[key] = content
        _artifact_cache_bytes += len(content)
        while _artifact_cache and _artifact_cache_bytes > _ARTIFACT_CACHE_MAX_BYTES:
            _, removed = _artifact_cache.popitem(last=False)
            _artifact_cache_bytes -= len(removed)


def _cache_evict_record(record_id: str) -> None:
    global _artifact_cache_bytes
    with _artifact_cache_lock:
        keys = [key for key in _artifact_cache if key[0] == record_id]
        for key in keys:
            _artifact_cache_bytes -= len(_artifact_cache.pop(key))


def _prepare_artifact(job: Job, content: bytes):
    if hashlib.sha256(content).hexdigest() != job.input_sha256:
        raise ReportPaperArtifactError("Marking-copy input does not match the paper job")
    if job.input_media_type == PDF_MEDIA_TYPE:
        sanitized, evidence = _sanitize_pdf(content)
        return _PreparedArtifact(
            source_bytes=sanitized,
            source_evidence=evidence,
            source_suffix=".pdf",
            artifact_kind="sanitized_submitted_pdf",
            presentation_status="page_faithful_ready",
            presentation_bytes=sanitized,
            presentation_evidence={"presentation_version": "canonical-presentation-v1", **evidence},
        )
    if job.input_media_type == DOCX_MEDIA_TYPE:
        sanitized, evidence = _sanitize_docx(content)
        if not settings.DOCX_PRESENTATION_RENDERING_ENABLED:
            return _PreparedArtifact(
                source_bytes=sanitized,
                source_evidence=evidence,
                source_suffix=".docx",
                artifact_kind="sanitized_submitted_docx",
                presentation_status="deterministic_pdf_render_required",
                presentation_bytes=None,
                presentation_evidence=None,
            )
        presentation, presentation_evidence = _render_validated_docx(sanitized)
        return _PreparedArtifact(
            source_bytes=sanitized,
            source_evidence=evidence,
            source_suffix=".docx",
            artifact_kind="sanitized_submitted_docx",
            presentation_status="page_faithful_ready",
            presentation_bytes=presentation,
            presentation_evidence=presentation_evidence,
        )
    raise ReportPaperArtifactError("Unsupported report paper media type")


def _render_validated_docx(content: bytes) -> tuple[bytes, dict]:
    """Production path: render once, validate, hash, and retain that exact PDF."""
    try:
        rendered = render_docx_to_pdf(content)
    except DocxPresentationError as exc:
        raise ReportPaperArtifactError(
            "DOCX presentation rendering failed"
        ) from exc
    presentation, evidence = _sanitize_pdf(rendered.pdf_bytes)
    return presentation, {
        "presentation_version": "canonical-presentation-v1",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "derivative_sha256": hashlib.sha256(presentation).hexdigest(),
        **rendered.provenance,
        **evidence,
        "per_document_render_count": 1,
        "renderer_qualification": "real_corpus_repeat_surface_acceptance_2026_08_29",
    }


def _render_repeat_verified_docx(content: bytes) -> tuple[bytes, dict]:
    """Acceptance-only qualification helper; not used by production ingestion."""
    first_pdf, first_evidence = _render_validated_docx(content)
    second_pdf, second_evidence = _render_validated_docx(content)
    comparable = ("page_count", "page_render_sha256", "page_text_sha256")
    if any(first_evidence[key] != second_evidence[key] for key in comparable):
        raise ReportPaperArtifactError(
            "Controlled DOCX rendering did not reproduce the same page/text surface"
        )
    return first_pdf, {
        "presentation_version": "canonical-presentation-v1",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "derivative_sha256": hashlib.sha256(first_pdf).hexdigest(),
        **first_evidence,
        "repeat_render_surface_verified": True,
        "repeat_render_pdf_bytes_equal": first_pdf == second_pdf,
    }


def _same_presentation_surface(left: dict | None, right: dict | None) -> bool:
    if not left or not right:
        return False
    return all(
        left.get(key) == right.get(key)
        for key in ("page_count", "page_render_sha256", "page_text_sha256")
    )


def _with_anchors(
    prepared: _PreparedArtifact,
    citations: list[InTextCitation] | None,
    source_media_type: str,
) -> _PreparedArtifact:
    if prepared.presentation_bytes is None or citations is None:
        return prepared
    suffix = ".docx" if source_media_type == DOCX_MEDIA_TYPE else ".pdf"
    qualified_text = extract_qualified_text_from_bytes(
        prepared.source_bytes,
        f"paper{suffix}",
    )
    semantic_text = qualified_text.selected.text
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", semantic_text) if part.strip()
    ]
    artifact = bind_citations_to_pdf(
        prepared.presentation_bytes,
        citations=citations,
        paragraphs=paragraphs,
    )
    evidence = {
        **(prepared.presentation_evidence or {}),
        "anchor_text_extraction": qualified_text.evidence(),
        "citation_anchors": artifact.model_dump(mode="json"),
    }
    return replace(prepared, presentation_evidence=evidence)


def _sanitize_pdf(content: bytes) -> tuple[bytes, dict]:
    try:
        original = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ReportPaperArtifactError("PDF marking copy could not be opened") from exc
    try:
        original_hashes = [_page_render_sha256(page) for page in original]
        editable_keys = (
            "title", "author", "subject", "keywords", "creator", "producer",
            "creationDate", "modDate", "trapped",
        )
        metadata_present = any(bool(original.metadata.get(key)) for key in editable_keys)
        xml_present = original.xref_xml_metadata() > 0
        original.set_metadata({key: "" for key in editable_keys})
        if xml_present:
            original.del_xml_metadata()
        output = io.BytesIO()
        # Metadata removal must not rewrite content streams or font resources:
        # aggressive PDF cleaning changed pixels in real LibreOffice outputs.
        # Minimal garbage collection preserves the fixed presentation surface.
        original.save(output, garbage=1, clean=False, deflate=False, no_new_id=True)
    finally:
        original.close()
    sanitized = output.getvalue()
    checked = fitz.open(stream=sanitized, filetype="pdf")
    try:
        sanitized_hashes = [_page_render_sha256(page) for page in checked]
        if sanitized_hashes != original_hashes:
            raise ReportPaperArtifactError("PDF sanitization changed rendered pages")
        if (
            any(bool(checked.metadata.get(key)) for key in editable_keys)
            or checked.xref_xml_metadata() > 0
        ):
            raise ReportPaperArtifactError("PDF document metadata was not removed")
        page_count = checked.page_count
        if page_count > settings.MAX_PDF_PAGES:
            raise ReportPaperArtifactError("Presentation PDF exceeds the page limit")
        page_text_hashes = [
            hashlib.sha256(page.get_text("text", sort=True).encode("utf-8")).hexdigest()
            for page in checked
        ]
        page_dimensions = [
            {
                "page_index": index,
                "width": round(float(page.rect.width), 3),
                "height": round(float(page.rect.height), 3),
            }
            for index, page in enumerate(checked)
        ]
    finally:
        checked.close()
    return sanitized, {
        "method": "pdf_metadata_removed_render_hash_verified",
        "page_count": page_count,
        "metadata_removed": metadata_present or xml_present,
        "render_hashes_verified": True,
        "page_render_sha256": sanitized_hashes,
        "page_text_sha256": page_text_hashes,
        "page_dimensions": page_dimensions,
    }


def _page_render_sha256(page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csRGB, alpha=False)
    return hashlib.sha256(pixmap.samples).hexdigest()


def _sanitize_docx(content: bytes) -> tuple[bytes, dict]:
    try:
        source = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ReportPaperArtifactError("DOCX marking copy could not be opened") from exc
    output = io.BytesIO()
    changed = []
    with source, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        original_non_metadata = {}
        sanitized_non_metadata = {}
        for entry in source.infolist():
            name = entry.filename
            value = source.read(name)
            if name.casefold().startswith("docprops/") and name.casefold().endswith(".xml"):
                replacement = _sanitized_docx_property_part(name)
                if replacement != value:
                    changed.append(name.casefold())
                value = replacement
            else:
                original_non_metadata[name] = hashlib.sha256(value).hexdigest()
                sanitized_non_metadata[name] = hashlib.sha256(value).hexdigest()
            clone = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            clone.compress_type = zipfile.ZIP_DEFLATED
            clone.external_attr = entry.external_attr
            target.writestr(clone, value)
    if original_non_metadata != sanitized_non_metadata:
        raise ReportPaperArtifactError("DOCX sanitization changed document content")
    sanitized = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(sanitized)) as checked:
        if checked.testzip() is not None or "word/document.xml" not in checked.namelist():
            raise ReportPaperArtifactError("Sanitized DOCX package failed validation")
    return sanitized, {
        "method": "docx_package_properties_blanked_content_hash_verified",
        "metadata_parts_changed": len(changed),
        "content_parts_verified": len(original_non_metadata),
        "page_render_verified": False,
    }


def _sanitized_docx_property_part(name: str) -> bytes:
    lower = name.casefold()
    if lower == "docprops/core.xml":
        return (
            b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            b'<cp:coreProperties '
            b'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            b'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            b'xmlns:dcterms="http://purl.org/dc/terms/" '
            b'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>'
        )
    if lower == "docprops/custom.xml":
        return (
            b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            b'<Properties '
            b'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
            b'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"/>'
        )
    return (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<Properties '
        b'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        b'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        b'<Application>SourceFidelity</Application></Properties>'
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
