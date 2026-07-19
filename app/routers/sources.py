"""Source repository API endpoints.

Instructor-facing endpoints for uploading, searching, and deleting source
documents. Uses an in-memory document registry for now; DB persistence
(via StoredDocument + an Alembic migration) is deferred.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.config import settings
from app.services.storage import get_storage_backend
from app.services.pdf_verifier import verify_instructor_upload
from app.services.chapter_splitter import is_edited_collection, split_into_chapters
from app.services.completeness_checker import check_completeness, INCOMPLETE, UNCERTAIN
from app.services.page_layout import classify_text_quality, PURE_SCAN, SCAN_OCR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])

# In-memory registry until DB integration (spec section 7 defers this).
_stored_documents: list[dict] = []


@router.post("/upload")
async def upload_source(
    file: UploadFile = File(...),
    doi: str | None = Form(None),
    isbn: str | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    expected_pages: int | None = Form(None),
    description: str | None = Form(None),  # noqa: ARG001 (reserved for future use)
):
    """Upload an academic source document.

    The instructor provides a PDF and optionally a DOI or ISBN. The system
    verifies the PDF matches the provided metadata, then stores it. Edited
    collections (detected via TOC + editor markers) are split into chapters.
    """
    if not settings.SOURCE_REPOSITORY_ENABLED:
        raise HTTPException(status_code=404, detail="Source repository is disabled")

    if not doi and not isbn and not title:
        raise HTTPException(
            status_code=400,
            detail="Provide at least a DOI, ISBN, or title for the source",
        )

    file_bytes = await file.read()

    # Verify the PDF matches the provided metadata.
    verified, messages = verify_instructor_upload(
        file_bytes,
        provided_doi=doi,
        provided_title=title,
        provided_author=author,
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
            # An ISBN indicates a book; a DOI without ISBN indicates a journal
            # article. The completeness heuristics differ: books are checked for
            # short-length/no-back-matter, articles are not (they're legitimately short).
            is_article=(doi is not None and isbn is None),
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

    # ── Storage ───────────────────────────────────────────────────────
    backend = get_storage_backend()
    results: list[dict] = []

    # Detect edited collection (single pass) and split if applicable.
    should_split = bool(isbn) and is_edited_collection(file_bytes)

    if should_split:
        chapters = split_into_chapters(file_bytes)
        for i, (info, chapter_bytes) in enumerate(chapters):
            key = f"by-isbn/{isbn}/chapter_{i + 1:02d}.pdf"
            backend.upload(chapter_bytes, key)
            results.append(
                _build_document_entry(
                    content_type="book_chapter",
                    title=info.title,
                    author=info.author or author or "",
                    isbn=isbn,
                    key=key,
                    size=len(chapter_bytes),
                    chapter_number=i + 1,
                    page_start=info.page_start,
                    page_end=info.page_end,
                    logical_pages=logical_pages,
                    completeness_verdict=completeness_verdict,
                    text_quality=text_quality,
                    review_status=review_status,
                )
            )
    else:
        key = _storage_key(doi, isbn, title)
        backend.upload(file_bytes, key)
        results.append(
            _build_document_entry(
                content_type="book" if isbn else "journal_article",
                title=title or "",
                author=author or "",
                doi=doi,
                isbn=isbn,
                key=key,
                size=len(file_bytes),
                logical_pages=logical_pages,
                completeness_verdict=completeness_verdict,
                text_quality=text_quality,
                review_status=review_status,
            )
        )

    # Register in the in-memory store so search/delete can find them.
    _stored_documents.extend(results)

    return {
        "status": "ok",
        "verification": messages,
        "text_quality": text_quality,
        "completeness_verdict": completeness_verdict,
        "review_status": review_status,
        "warnings": warnings,
        "split_detected": should_split,
        "documents": results,
    }


@router.get("/search")
async def search_sources(
    doi: str | None = Query(None),
    isbn: str | None = Query(None),
    title: str | None = Query(None),
    author: str | None = Query(None),
):
    """Search for stored sources by DOI, ISBN, title, or author."""
    results = _stored_documents

    if doi:
        results = [d for d in results if d.get("doi") == doi]
    if isbn:
        results = [d for d in results if d.get("isbn") == isbn]
    if title:
        t = title.lower()
        results = [d for d in results if t in d.get("title", "").lower()]
    if author:
        a = author.lower()
        results = [d for d in results if a in d.get("author", "").lower()]

    return {"count": len(results), "documents": results}


@router.delete("/{doc_id}")
async def delete_source(doc_id: str):
    """Delete a stored source by its ID."""
    global _stored_documents
    backend = get_storage_backend()

    for doc in _stored_documents:
        if doc["id"] == doc_id:
            backend.delete(doc["s3_key"])
            _stored_documents = [d for d in _stored_documents if d["id"] != doc_id]
            return {"status": "deleted", "id": doc_id}

    raise HTTPException(status_code=404, detail="Document not found")


@router.post("/{doc_id}/review")
async def review_source(doc_id: str, decision: str = Form(...)):
    """Approve or reject a source held for review (Standard strictness mode).

    Args:
        decision: "accept" sets review_status to accepted (usable for verification).
                  "reject" sets review_status to rejected (excluded from lookups).
    """
    decision_lower = decision.strip().lower()
    if decision_lower not in ("accept", "reject"):
        raise HTTPException(
            status_code=400,
            detail="decision must be 'accept' or 'reject'",
        )

    new_status = "accepted" if decision_lower == "accept" else "rejected"

    for doc in _stored_documents:
        if doc["id"] == doc_id:
            doc["review_status"] = new_status
            return {"status": "ok", "id": doc_id, "review_status": new_status}

    raise HTTPException(status_code=404, detail="Document not found")


# ── Helpers ──────────────────────────────────────────────


def _storage_key(doi: str | None, isbn: str | None, title: str | None) -> str:
    if doi:
        return f"by-doi/{doi}.pdf"
    if isbn:
        return f"by-isbn/{isbn}.pdf"
    if title:
        h = hashlib.sha256(title.strip().lower().encode()).hexdigest()[:12]
        return f"by-title-hash/{h}.pdf"
    raise ValueError("Need DOI, ISBN, or title for storage key")


def _build_document_entry(
    content_type: str,
    title: str,
    author: str,
    key: str,
    size: int,
    doi: str | None = None,
    isbn: str | None = None,
    chapter_number: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    logical_pages: int | None = None,
    completeness_verdict: str | None = None,
    text_quality: str | None = None,
    review_status: str = "accepted",
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "content_type": content_type,
        "title": title,
        "author": author,
        "doi": doi,
        "isbn": isbn,
        "s3_key": key,
        "file_size_bytes": size,
        "license_class": "instructor_upload",
        "provenance": "instructor_upload",
        "verified": True,
        "verification_method": "instructor_attestation",
        "chapter_number": chapter_number,
        "page_range_start": page_start,
        "page_range_end": page_end,
        "logical_pages": logical_pages,
        "completeness_verdict": completeness_verdict,
        "text_quality": text_quality,
        "review_status": review_status,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
