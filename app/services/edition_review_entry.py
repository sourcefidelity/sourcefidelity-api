"""Personal-only, safety-cleared, review-only source access and receipts.

No source is admitted and no historical report/package is modified. Reference
text is explicitly owner-supplied, not silently attributed to a student paper.
"""
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Literal

import fitz
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from app.config import settings
from app.models.edition_review import EditionReviewSnapshot, EditionReviewDecision
from app.models.source_repository import SourceRepresentationRecord
from app.security import EDITION_REVIEW_CAPABILITY, REPORT_SOURCE_CAPABILITY
from app.services.alternate_edition import AlternateEditionRecord
from app.services.edition_review_workflow import EditionReviewSubmission, record_edition_review
from app.services.file_safety import inspect_uploaded_pdf, SafetyVerdict
from app.services.source_repository import representation_is_expired


class EditionReviewError(ValueError):
    pass


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_hash(value: dict) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def require_personal_review(principal):
    principal.require(EDITION_REVIEW_CAPABILITY)
    principal.require(REPORT_SOURCE_CAPABILITY)
    if settings.REPORT_AUTH_MODE not in {"personal_local", "personal_bearer"} or principal.scope_type != "personal_owner":
        raise HTTPException(status_code=403, detail="Personal edition review only")


def load_review_source(session, backend, principal, representation_id):
    """Review-only loader; never use this to authorize citation evidence."""
    require_personal_review(principal)
    try:
        source_id = uuid.UUID(str(representation_id))
    except ValueError as exc:
        raise EditionReviewError("Source unavailable for review") from exc
    record = session.get(SourceRepresentationRecord, source_id)
    if (record is None or (record.scope_type, record.scope_id) != (principal.scope_type, principal.scope_id)
            or record.admission_state not in {"accepted", "needs_review"}
            or representation_is_expired(record)
            or record.cleanliness_verdict != "clean"
            or record.representation_kind != "pdf"
            or record.canonical_work.work_type not in {"book", "monograph"}):
        raise EditionReviewError("Source unavailable for review")
    obj = record.content_object
    if (obj.deletion_pending or obj.media_type != "application/pdf"
            or not 0 < obj.byte_size <= settings.MAX_FILE_SIZE_MB * 1024 * 1024
            or not backend.exists(obj.storage_key)):
        raise EditionReviewError("Source unavailable for review")
    # No paths, URLs, bytes or scanner details from failed sources in responses.
    content = backend.download(obj.storage_key)
    if len(content) != obj.byte_size or sha(content) != obj.content_sha256:
        raise EditionReviewError("Source bytes changed")
    audit_scope = (record.validation_evidence or {}).get("authorization_scope")
    if audit_scope and audit_scope != {"type": record.scope_type, "id": record.scope_id}:
        raise EditionReviewError("Source authorization changed")
    # Derivatives require their parent lifecycle; this first entry path is
    # deliberately original-PDF-only rather than bypassing parent checks.
    if (record.validation_evidence or {}).get("ocr_derivative"):
        raise EditionReviewError("Derived copies are not supported by this review path")
    if inspect_uploaded_pdf(content).verdict is not SafetyVerdict.CLEAN:
        raise EditionReviewError("Source safety check did not pass")
    return record, content


@dataclass(frozen=True)
class ReviewPages:
    manifest: dict
    images: tuple[bytes, ...]


def render_review_pages(content: bytes, *, page_number: int | None = None) -> ReviewPages:
    """Render at most six pages; no OCR, remote processing or active PDF UI."""
    images, pages = [], []
    with fitz.open(stream=content, filetype="pdf") as document:
        if not document.page_count:
            raise EditionReviewError("No review pages")
        count = min(6, document.page_count)
        if page_number is not None and not 1 <= page_number <= count:
            raise EditionReviewError("Page outside review")
        indexes = range(count) if page_number is None else [page_number - 1]
        for index in indexes:
            page = document[index]
            scale = 120 / 72
            pixels = (math.ceil(page.rect.width * scale) + 2) * (math.ceil(page.rect.height * scale) + 2)
            if pixels > 8_000_000:
                raise EditionReviewError("Review page exceeds render budget")
            image = page.get_pixmap(dpi=120, colorspace=fitz.csRGB, alpha=False).tobytes("png")
            images.append(image)
            pages.append({"page_index": index, "render_sha256": sha(image)})
        return ReviewPages({"renderer": f"pymupdf-{fitz.VersionBind}-rgb-120-v1",
                            "total_pages": document.page_count, "pages": pages}, tuple(images))


def prepare_review(session, backend, principal, representation_id, reference_text):
    if not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 4000:
        raise EditionReviewError("Provide the reference to compare (at most 4000 characters)")
    record, content = load_review_source(session, backend, principal, representation_id)
    pages = render_review_pages(content)
    from app.services.personal_edition_answer import separate_work_signal, QUESTION, OPTIONS
    with fitz.open(stream=content, filetype="pdf") as document:
        front_text = "\n".join(document[i].get_text() for i in range(min(6, len(document))))
    payload = {"version": "personal-edition-review-v3", "reference_origin": "owner_supplied",
        "question": QUESTION, "options": OPTIONS,
        "separate_retrieval_required": separate_work_signal(reference_text + "\n" + front_text),
        "reference_text": reference_text, "reference_sha256": sha(reference_text.encode()),
        "source_sha256": sha(content), "representation_id": str(record.id),
        "scope_type": record.scope_type, "scope_id": record.scope_id,
        "source_status": {"identity": record.identity_verdict,
                          "completeness": record.completeness_verdict,
                          "admission": record.admission_state},
        "page_manifest": pages.manifest}
    snapshot = EditionReviewSnapshot(representation_id=record.id, scope_type=record.scope_type,
        scope_id=record.scope_id, snapshot_sha256=payload_hash(payload), payload=payload)
    session.add(snapshot)
    session.flush()
    return snapshot


def resolve_review(session, backend, principal, review_id, *, lock=False, page_number=None):
    require_personal_review(principal)
    try:
        parsed = uuid.UUID(str(review_id))
    except ValueError as exc:
        raise EditionReviewError("Review unavailable") from exc
    query = select(EditionReviewSnapshot).where(EditionReviewSnapshot.id == parsed,
        EditionReviewSnapshot.scope_type == principal.scope_type,
        EditionReviewSnapshot.scope_id == principal.scope_id)
    if lock:
        query = query.with_for_update()
    snapshot = session.scalar(query)
    if snapshot is None or payload_hash(snapshot.payload) != snapshot.snapshot_sha256:
        raise EditionReviewError("Review unavailable or changed")
    record, content = load_review_source(session, backend, principal, snapshot.representation_id)
    payload = snapshot.payload
    if (payload.get("representation_id") != str(record.id)
            or payload.get("scope_type") != principal.scope_type or payload.get("scope_id") != principal.scope_id
            or payload.get("source_sha256") != sha(content)
            or payload.get("reference_sha256") != sha(payload["reference_text"].encode())
            or payload.get("source_status") != {"identity": record.identity_verdict,
                "completeness": record.completeness_verdict, "admission": record.admission_state}):
        raise EditionReviewError("Review inputs changed; prepare a new review")
    pages = render_review_pages(content, page_number=page_number)
    expected_manifest = payload["page_manifest"]
    if page_number is not None:
        expected_manifest = dict(expected_manifest, pages=[p for p in expected_manifest["pages"]
                                                          if p["page_index"] == page_number - 1])
    if pages.manifest != expected_manifest:
        raise EditionReviewError("Review pages changed; prepare a new review")
    return snapshot, content, pages


class ReviewDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["confirmed", "same_edition_later_printing", "uncertain", "rejected"]
    acknowledged: Literal[True]
    notes: str = Field(min_length=1, max_length=4000)
    work_page: int | None = Field(default=None, ge=1, le=6)
    edition_page: int | None = Field(default=None, ge=1, le=6)

    @model_validator(mode="after")
    def check_decision(self):
        if not self.notes.strip():
            raise ValueError("Explain the review decision")
        positive = self.decision in {"confirmed", "same_edition_later_printing"}
        if positive and (self.work_page is None or self.edition_page is None):
            raise ValueError("Confirmation requires work and edition evidence pages")
        if not positive and (self.work_page is not None or self.edition_page is not None):
            raise ValueError("Evidence page selections apply only to confirmation")
        return self


def save_review(session, backend, principal, review_id, submission: ReviewDecisionInput):
    submission = ReviewDecisionInput.model_validate(submission.model_dump())
    snapshot, content, pages = resolve_review(session, backend, principal, review_id, lock=True)
    if snapshot.payload.get("version") == "personal-edition-review-v3":
        raise EditionReviewError("Use the simple review answer")
    if submission.snapshot_sha256 != snapshot.snapshot_sha256:
        raise EditionReviewError("Review inputs changed")
    existing = session.scalar(select(EditionReviewDecision).where(EditionReviewDecision.snapshot_id == snapshot.id))
    if existing is not None:
        raise EditionReviewError("This review already has a decision; start a new review to reconsider")
    same_printing = submission.decision == "same_edition_later_printing"
    if same_printing and snapshot.payload.get("version") != "personal-edition-review-v2":
        raise EditionReviewError("New decision option requires a new review snapshot")
    confirmed = submission.decision in {"confirmed", "same_edition_later_printing"}
    if confirmed:
        from app.services.personal_edition_answer import separate_work_signal
        with fitz.open(stream=content, filetype="pdf") as document:
            front_text = "\n".join(document[i].get_text() for i in range(min(6, len(document))))
        if separate_work_signal(snapshot.payload['reference_text'] + '\n' + front_text):
            raise EditionReviewError("Translation or abridgment requires separate retrieval")
    selections = ({"work_identity": submission.work_page, "edition_relationship": submission.edition_page}
                  if confirmed else {f"observed_page_{i + 1}": i + 1 for i in range(len(pages.images))})
    if any(number > len(pages.images) for number in selections.values()):
        raise EditionReviewError("Evidence page is outside this review")
    evidence, passages = [], {}
    for purpose, number in selections.items():
        evidence_id = f"{purpose}:pdf-page-{number}"
        image = pages.images[number - 1]
        passages[evidence_id] = image
        evidence.append({"evidence_id": evidence_id, "representation_sha256": sha(content),
            "passage_sha256": sha(image), "purpose": purpose if confirmed else "edition_relationship"})
    record = AlternateEditionRecord(submitted_reference_sha256=snapshot.payload["reference_sha256"],
        contract_version="alternate-edition-v2" if same_printing else "alternate-edition-v1",
        retrieved_representation_sha256=sha(content),
        work_identity="verified" if confirmed else "unverified",
        relationship="same_edition_later_printing" if same_printing else "verified_alternate_edition" if confirmed else "unverified", evidence=tuple(evidence),
        limitations=("Owner-supplied reference, not linked to a student report.",
            "Bounded first-six-page visual review; evidence hashes identify rendered pages, not text excerpts.",
            "Relationship review grants no source admission, unchanged-text equivalence, task usability or locator correspondence."))
    reviewed = record_edition_review(principal=principal, scope_type=snapshot.scope_type,
        scope_id=snapshot.scope_id, record=record, submitted_reference=snapshot.payload["reference_text"].encode(),
        representations={sha(content): content}, passages=passages,
        submission=EditionReviewSubmission(record_sha256=record.review_input_sha256(),
            decision=submission.decision, reviewed_evidence_ids=tuple(passages), notes=submission.notes))
    payload = {"snapshot_sha256": snapshot.snapshot_sha256, "submission": submission.model_dump(mode="json"),
               "alternate_edition": reviewed.model_dump(mode="json")}
    decision = EditionReviewDecision(snapshot_id=snapshot.id, reviewer_provider=principal.provider,
        payload=payload, payload_sha256=payload_hash(payload))
    session.add(decision)
    session.flush()
    return decision
