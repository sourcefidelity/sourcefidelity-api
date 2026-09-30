"""Trusted review processor; deliberately separate from source admission.

The adapter must resolve a current, authorized snapshot before calling this
processor. Neither a client-supplied record nor matching hashes authenticate
the reviewer or authorize source access.
"""
from datetime import datetime, timezone
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.security import AuthenticatedPrincipal, REPORT_SOURCE_CAPABILITY, EDITION_REVIEW_CAPABILITY
from app.services.alternate_edition import AlternateEditionRecord


class EditionReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["confirmed", "same_edition_later_printing", "uncertain", "rejected"]
    reviewed_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    notes: str = Field(min_length=1, max_length=4000)


def record_edition_review(
    *,
    principal: AuthenticatedPrincipal,
    scope_type: str,
    scope_id: str,
    record: AlternateEditionRecord,
    submitted_reference: bytes,
    representations: dict[str, bytes],
    passages: dict[str, bytes],
    submission: EditionReviewSubmission,
) -> AlternateEditionRecord:
    """Create a new decision from a trusted, independently resolved snapshot.

    ``representations`` and ``passages`` must be resolved by the server under
    current source authorization, not deserialized from a review request.
    Their byte hashes are checked again here. The persistence adapter must
    append the result, preserving all prior decisions and source/report state.
    This function does not authenticate transport, resolve files, or persist.
    """
    principal.require(EDITION_REVIEW_CAPABILITY)
    principal.require(REPORT_SOURCE_CAPABILITY)
    if (principal.scope_type, principal.scope_id) != (scope_type, scope_id):
        raise ValueError("edition_review_scope_mismatch")
    # Revalidate even if a caller used Pydantic's unchecked model_copy API.
    record = AlternateEditionRecord.model_validate(record.model_dump())
    submission = EditionReviewSubmission.model_validate(submission.model_dump())
    if not record.evidence or len(record.evidence) > 32:
        raise ValueError("edition_review_evidence_required")
    if submission.record_sha256 != record.review_input_sha256():
        raise ValueError("edition_review_stale_input")
    if hashlib.sha256(submitted_reference).hexdigest() != record.submitted_reference_sha256:
        raise ValueError("edition_review_reference_mismatch")
    expected = {item.evidence_id for item in record.evidence}
    if set(passages) != expected:
        raise ValueError("edition_review_unresolved_evidence")
    required_sources = {item.representation_sha256 for item in record.evidence}
    required_sources.add(record.retrieved_representation_sha256)
    if set(representations) != required_sources:
        raise ValueError("edition_review_unresolved_source")
    if any(hashlib.sha256(value).hexdigest() != key
           for key, value in representations.items()):
        raise ValueError("edition_review_source_mismatch")
    for item in record.evidence:
        if hashlib.sha256(passages[item.evidence_id]).hexdigest() != item.passage_sha256:
            raise ValueError("edition_review_passage_mismatch")
    payload = record.model_dump(mode="json")
    payload["human_review"] = {
        "reviewer_id": principal.subject,
        "reviewed_at": datetime.now(timezone.utc),
        "record_sha256": submission.record_sha256,
        "decision": submission.decision,
        "reviewed_evidence_ids": submission.reviewed_evidence_ids,
        "notes": submission.notes,
    }
    return AlternateEditionRecord.model_validate(payload)
