"""Source repository API endpoints backed by durable admission records."""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.security import (
    AuthenticatedPrincipal,
    SOURCE_REPOSITORY_READ_CAPABILITY,
    SOURCE_REPOSITORY_WRITE_CAPABILITY,
    get_authenticated_principal,
    require_same_origin_request,
)
from app.models.source_repository import (
    CanonicalWorkRecord,
    SourceRepresentationRecord,
)
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.source_repository import (
    AdmissionError,
    AdmissionRequest,
    WorkIdentity,
    active_representation_clause,
    admit_representation,
    delete_representation,
    finalize_pending_object_deletions,
    commit_source_admissions,
    rollback_source_admissions,
    retention_mode_for_scope,
    representation_is_expired,
)
from app.services.storage.backend import StorageBackend, get_storage_backend
from app.services.pdf_verifier import verify_instructor_upload
from app.services.chapter_splitter import is_edited_collection, split_into_chapters
from app.services.book_metadata import (
    document_kind_for_source_kind,
    normalize_source_kind,
)
from app.services.completeness_checker import check_completeness, INCOMPLETE, UNCERTAIN
from app.services.file_safety import (
    FileSafetyUnavailable,
    SafetyVerdict,
    inspect_uploaded_pdf,
)
from app.services.page_layout import classify_text_quality, PURE_SCAN, SCAN_OCR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


def _storage_backend() -> StorageBackend:
    """Resolve storage at request time so deployment overrides remain effective."""
    return get_storage_backend()


@router.post("/upload")
def upload_source(
    request: Request,
    file: UploadFile = File(...),
    doi: str | None = Form(None),
    isbn: str | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    year: str | None = Form(None),
    expected_pages: int | None = Form(None),
    expected_first_page: int | None = Form(None),
    expected_last_page: int | None = Form(None),
    document_kind: str | None = Form(None),
    source_kind: str | None = Form(None),
    edition_or_version: str | None = Form(None),
    description: str | None = Form(None),  # noqa: ARG001 (reserved for future use)
    db: Session = Depends(get_db),
    backend: StorageBackend = Depends(_storage_backend),
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
):
    """Upload an academic source document.

    The instructor provides a PDF and optionally a DOI or ISBN. The system
    verifies the PDF matches the provided metadata, then stores it. Edited
    collections (detected via TOC + editor markers) are split into chapters.
    """
    principal.require(SOURCE_REPOSITORY_WRITE_CAPABILITY)
    require_same_origin_request(request)
    if not settings.SOURCE_REPOSITORY_ENABLED:
        raise HTTPException(status_code=404, detail="Source repository is disabled")

    if not doi and not isbn and not title:
        raise HTTPException(
            status_code=400,
            detail="Provide at least a DOI, ISBN, or title for the source",
        )

    normalized_source_kind = None
    if source_kind is not None:
        try:
            normalized_source_kind = normalize_source_kind(source_kind)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if document_kind is not None:
        document_kind = document_kind.strip().lower()
        if document_kind not in {"article", "book", "chapter", "unknown"}:
            raise HTTPException(
                status_code=400,
                detail="document_kind must be article, book, chapter, or unknown",
            )
    source_document_kind = (
        document_kind_for_source_kind(normalized_source_kind)
        if normalized_source_kind
        else None
    )
    if document_kind and source_document_kind and document_kind != source_document_kind:
        raise HTTPException(
            status_code=400,
            detail="document_kind conflicts with source_kind",
        )
    resolved_document_kind = document_kind or source_document_kind or (
        "article" if doi is not None and isbn is None
        else "book" if isbn is not None
        else "unknown"
    )
    resolved_source_kind = normalized_source_kind or {
        "article": "journal_article",
        "book": "monograph",
        "chapter": "book_section",
        "unknown": "unknown",
    }[resolved_document_kind]

    if (expected_first_page is None) != (expected_last_page is None):
        raise HTTPException(
            status_code=400,
            detail="Provide both expected_first_page and expected_last_page",
        )
    expected_page_range = None
    if expected_first_page is not None and expected_last_page is not None:
        if expected_first_page < 1 or expected_last_page < expected_first_page:
            raise HTTPException(
                status_code=400,
                detail="Expected page range is invalid",
            )
        expected_page_range = (expected_first_page, expected_last_page)

    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    file_bytes = file.file.read(maximum + 1)
    if len(file_bytes) > maximum:
        raise HTTPException(
            status_code=413,
            detail=f"Source exceeds the {settings.MAX_FILE_SIZE_MB} MB upload limit",
        )

    # Safety precedes every metadata, text-quality, and completeness parser.
    try:
        safety_report = inspect_uploaded_pdf(file_bytes)
    except FileSafetyUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Required malware inspection is unavailable",
                "retryable": True,
            },
        ) from exc
    if safety_report.verdict is SafetyVerdict.REJECTED:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Upload rejected by hostile-file inspection",
                "findings": list(safety_report.findings),
            },
        )

    # Verify the PDF matches the provided metadata.
    verified, messages = verify_instructor_upload(
        file_bytes,
        provided_doi=doi,
        provided_title=title,
        provided_author=author,
        provided_year=year,
        provided_isbn=isbn,
    )
    if not verified:
        raise HTTPException(
            status_code=422,
            detail={"error": "PDF verification failed", "messages": messages},
        )

    # ── Text-quality classification ───────────────────────────────────
    # Classify the PDF's text layer: digital / scan_ocr / pure_scan.
    # Pure scans have no extractable text and are unusable without OCR.
    text_quality = classify_text_quality(file_bytes).verdict

    warnings: list[str] = []
    if text_quality == PURE_SCAN:
        scan_msg = (
            "This PDF has no extractable text (it appears to be a scanned image with "
            "no OCR layer). It cannot be used for citation verification until OCR'd. "
            "Please OCR it (Adobe Acrobat / ABBYY / ocrmypdf) and re-upload."
        )
        mode = settings.STRICTNESS_MODE.lower()
        if mode == "strict":
            raise HTTPException(
                status_code=422,
                detail={"error": "Upload rejected: PDF has no text layer (pure scan, needs OCR)", "hint": scan_msg},
            )
        warnings.append(scan_msg)
    elif text_quality == SCAN_OCR:
        warnings.append(
            "Note: this PDF is a scan with an OCR text layer. OCR may contain "
            "recognition errors — exact-match citation results will be treated with "
            "lower confidence."
        )

    # ── Completeness check ────────────────────────────────────────────
    completeness_verdict: str | None = None
    review_status = "accepted"
    logical_pages: int | None = None

    if settings.COMPLETENESS_CHECK_ENABLED:
        report = check_completeness(
            file_bytes,
            isbn=isbn,
            title=title,
            author=author,
            expected_pages=expected_pages,
            expected_page_range=expected_page_range,
            # Strong caller metadata selects source-specific rules. Title-only
            # uploads remain unknown unless the operator supplies a bounded kind.
            document_kind=resolved_document_kind,
        )
        completeness_verdict = report.verdict
        logical_pages = report.n_up_layout.logical_pages if report.n_up_layout else None

        flagged = report.verdict in (INCOMPLETE, UNCERTAIN)
        if flagged:
            warnings.extend(report.messages)
            warnings.extend(f"signal: {s}" for s in report.signals)
        elif report.verdict == "COMPLETE" and report.n_up_layout and report.n_up_layout.is_n_up:
            # Not flagged, but worth noting the N-up layout.
            warnings.append(
                f"Note: detected {report.n_up_layout.pages_per_sheet}-up layout "
                f"({logical_pages} logical pages)."
            )

        # Apply strictness policy.
        mode = settings.STRICTNESS_MODE.lower()
        if flagged and mode == "strict":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Upload rejected (strict mode): completeness check flagged this PDF",
                    "completeness_verdict": report.verdict,
                    "messages": warnings,
                },
            )
        if flagged and mode == "standard":
            review_status = "pending_review"

    # Pure scans in standard mode are also held for review (they're unusable
    # until OCR'd). In lenient they're accepted with a warning; in strict
    # they were already rejected above.
    if text_quality == PURE_SCAN and settings.STRICTNESS_MODE.lower() == "standard":
        review_status = "pending_review"

    # ── Durable storage/admission ─────────────────────────────────────
    results: list[dict] = []

    # A supplied source kind guides detection but cannot prove that usable
    # chapter boundaries exist. Split only after structural confirmation.
    should_inspect_for_chapters = (
        resolved_source_kind == "edited_collection" or bool(isbn)
    )
    should_split = should_inspect_for_chapters and is_edited_collection(file_bytes)

    if should_split:
        chapters = split_into_chapters(file_bytes)
        admission_items = [
            (
                WorkIdentity(
                    title=info.title,
                    author=info.author or author,
                    year=year,
                    work_type="book_section",
                ),
                chapter_bytes,
                {
                    "parent_isbn": isbn,
                    "parent_title": title,
                    "chapter_number": i + 1,
                    "page_range_start": info.page_start,
                    "page_range_end": info.page_end,
                },
            )
            for i, (info, chapter_bytes) in enumerate(chapters)
        ]
    else:
        admission_items = [
            (
                WorkIdentity(
                    work_type=resolved_source_kind,
                    title=title or doi or isbn or "",
                    author=author,
                    year=year,
                    doi=doi,
                    isbn=isbn,
                ),
                file_bytes,
                {},
            )
        ]

    try:
        for work, content, item_evidence in admission_items:
            record = admit_representation(
                db,
                backend,
                AdmissionRequest(
                    work=work,
                    representation=SourceRepresentation(
                        kind=RepresentationKind.PDF,
                        media_type="application/pdf",
                        content=content,
                    ),
                    provenance="instructor_upload",
                    license_class="commercial_user_upload",
                    scope_type=principal.scope_type,
                    scope_id=principal.scope_id,
                    identity_verdict="verified",
                    identity_confidence=_upload_identity_confidence(messages),
                    completeness_verdict=(
                        (completeness_verdict or "not_assessed").casefold()
                    ),
                    cleanliness_verdict=safety_report.verdict.value,
                    text_quality=text_quality,
                    edition_or_version=(edition_or_version or None),
                    admitted_by=None,
                    validation_evidence={
                        "upload_verification": messages,
                        "file_safety": {
                            "structural_verdict": safety_report.structural_verdict.value,
                            "malware_verdict": safety_report.malware_verdict.value,
                        },
                        "logical_pages": logical_pages,
                        "source_kind": resolved_source_kind,
                        **item_evidence,
                    },
                    request_acceptance=(review_status == "accepted"),
                ),
            )
            results.append(_representation_dict(record))
        commit_source_admissions(db)
    except AdmissionError as exc:
        rollback_source_admissions(db, backend)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        rollback_source_admissions(db, backend)
        raise

    durable_review_status = (
        "accepted"
        if results and all(item["admission_state"] == "accepted" for item in results)
        else "needs_review"
    )

    return {
        "status": "ok",
        "verification": messages,
        "text_quality": text_quality,
        "completeness_verdict": completeness_verdict,
        "review_status": durable_review_status,
        "warnings": warnings,
        "split_detected": should_split,
        "source_kind": resolved_source_kind,
        "documents": results,
    }


@router.get("/search")
def search_sources(
    doi: str | None = Query(None),
    isbn: str | None = Query(None),
    title: str | None = Query(None),
    author: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10_000),
    db: Session = Depends(get_db),
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
):
    """Search for stored sources by DOI, ISBN, title, or author."""
    principal.require(SOURCE_REPOSITORY_READ_CAPABILITY)
    query = (
        select(SourceRepresentationRecord)
        .join(SourceRepresentationRecord.canonical_work)
        .where(
            SourceRepresentationRecord.scope_type == principal.scope_type,
            SourceRepresentationRecord.scope_id == principal.scope_id,
            active_representation_clause(),
        )
    )
    if doi:
        normalized = doi.strip().lower().removeprefix("https://doi.org/")
        query = query.where(CanonicalWorkRecord.doi == normalized)
    if isbn:
        normalized = "".join(char for char in isbn.upper() if char.isdigit() or char == "X")
        query = query.where(CanonicalWorkRecord.isbn == normalized)
    if title:
        query = query.where(CanonicalWorkRecord.display_title.ilike(f"%{title}%"))
    if author:
        query = query.where(CanonicalWorkRecord.author.ilike(f"%{author}%"))

    query = query.order_by(SourceRepresentationRecord.created_at.desc()).offset(offset).limit(limit)
    results = [_representation_dict(record) for record in db.scalars(query).all()]
    return {"count": len(results), "documents": results}


@router.delete("/{doc_id}")
def delete_source(
    doc_id: str,
    request: Request,
    db: Session = Depends(get_db),
    backend: StorageBackend = Depends(_storage_backend),
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
):
    """Delete a stored source by its ID."""
    principal.require(SOURCE_REPOSITORY_WRITE_CAPABILITY)
    require_same_origin_request(request)
    try:
        parsed_id = uuid.UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    try:
        record = db.get(SourceRepresentationRecord, parsed_id)
        if (
            record is None
            or record.scope_type != principal.scope_type
            or record.scope_id != principal.scope_id
        ):
            raise HTTPException(status_code=404, detail="Document not found")
        if not delete_representation(db, parsed_id):
            raise HTTPException(status_code=404, detail="Document not found")
        db.commit()
        finalize_pending_object_deletions(db, backend)
        db.commit()
        return {"status": "deleted", "id": doc_id}
    except HTTPException:
        db.rollback()
        raise
    except AdmissionError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{doc_id}/review")
def review_source(
    doc_id: str,
    request: Request,
    decision: str = Form(...),
    db: Session = Depends(get_db),
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
):
    """Approve or reject a source held for review (Standard strictness mode).

    Args:
        decision: "accept" sets review_status to accepted (usable for verification).
                  "reject" sets review_status to rejected (excluded from lookups).
    """
    principal.require(SOURCE_REPOSITORY_WRITE_CAPABILITY)
    require_same_origin_request(request)
    decision_lower = decision.strip().lower()
    if decision_lower not in ("accept", "reject"):
        raise HTTPException(
            status_code=400,
            detail="decision must be 'accept' or 'reject'",
        )

    new_status = "accepted" if decision_lower == "accept" else "rejected"

    try:
        parsed_id = uuid.UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    record = db.get(SourceRepresentationRecord, parsed_id)
    if (
        record is None
        or record.scope_type != principal.scope_type
        or record.scope_id != principal.scope_id
    ):
        raise HTTPException(status_code=404, detail="Document not found")
    if representation_is_expired(record):
        raise HTTPException(status_code=410, detail="Source representation has expired")
    record.admission_state = new_status
    if new_status == "accepted":
        record.admitted_at = datetime.now(timezone.utc)
        record.admitted_by = "manual_review"
    else:
        record.admitted_at = None
        record.admitted_by = None
    db.commit()
    return {"status": "ok", "id": doc_id, "review_status": new_status}


# ── Helpers ──────────────────────────────────────────────


def _representation_dict(record: SourceRepresentationRecord) -> dict:
    work = record.canonical_work
    content = record.content_object
    return {
        "id": str(record.id),
        "canonical_work_id": str(record.canonical_work_id),
        "content_type": work.work_type,
        "title": work.display_title,
        "author": work.author or "",
        "year": work.year,
        "doi": work.doi,
        "isbn": work.isbn,
        "file_size_bytes": content.byte_size,
        "representation_kind": record.representation_kind,
        "media_type": content.media_type,
        "content_sha256": content.content_sha256,
        "license_class": content.license_class,
        "provenance": record.provenance,
        "edition_or_version": record.edition_or_version,
        "identity_verdict": record.identity_verdict,
        "completeness_verdict": record.completeness_verdict,
        "cleanliness_verdict": record.cleanliness_verdict,
        "text_quality": record.text_quality,
        "admission_state": record.admission_state,
        "review_status": record.admission_state,
        "retention_mode": retention_mode_for_scope(record.scope_type).value,
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "created_at": record.created_at.isoformat(),
    }


def _upload_identity_confidence(messages: list[str]) -> float:
    evidence = set(messages)
    if evidence & {"doi_match", "isbn_match"}:
        return 1.0
    if {"title_match", "author_match"} <= evidence:
        return 0.9
    if {"title_match", "year_match"} <= evidence:
        return 0.85
    if "distinctive_title_match" in evidence:
        return 0.8
    return 0.0
