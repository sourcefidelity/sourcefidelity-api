"""Simple owner confirmation, deliberately not an alternate-edition grant."""
from datetime import datetime, timezone
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.models.edition_review import EditionReviewDecision

QUESTION = "Does this copy match the work and edition in the reference?"
OPTIONS = (("yes", "Yes", "The work and edition match; a later printing of that edition counts."),
           ("no", "No", "They do not match, or you cannot confirm a match from these pages."))


def separate_work_signal(text):
    # Conservative routing signal, not a determination that two texts differ.
    # In particular, 'unabridged' must not match 'abridged'.
    return bool(re.search(r"\b(?:abridged|abridgment|abridgement|translated\s+by|translation\s+(?:by|of)|this\s+translation)\b", text, re.I))


class PersonalEditionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer: Literal["yes", "no"]


def save_personal_answer(session, backend, principal, review_id, answer):
    from app.services.edition_review_entry import resolve_review, payload_hash, EditionReviewError
    answer = PersonalEditionAnswer.model_validate(answer.model_dump())
    snapshot, _, _ = resolve_review(session, backend, principal, review_id, lock=True)
    if snapshot.payload.get("version") != "personal-edition-review-v3" or answer.snapshot_sha256 != snapshot.snapshot_sha256:
        raise EditionReviewError("Prepare a new simple review")
    if session.scalar(select(EditionReviewDecision).where(EditionReviewDecision.snapshot_id == snapshot.id)):
        raise EditionReviewError("This review already has a decision")
    if answer.answer == "yes" and snapshot.payload["separate_retrieval_required"]:
        raise EditionReviewError("Translation or abridgment requires separate retrieval")
    payload = {"version": "personal-edition-answer-v1", "snapshot_sha256": snapshot.snapshot_sha256,
        "answer": answer.answer, "outcome": "owner_confirmed_cited_edition" if answer.answer == "yes" else "unverified",
        "reviewer_id": principal.subject, "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "page_manifest": snapshot.payload["page_manifest"],
        "source_sha256": snapshot.payload["source_sha256"], "reference_sha256": snapshot.payload["reference_sha256"],
        "source_admission_granted": False, "task_usability_granted": False,
        "unchanged_text_established": False, "locator_correspondence_established": False}
    decision = EditionReviewDecision(snapshot_id=snapshot.id, reviewer_provider=principal.provider,
        payload=payload, payload_sha256=payload_hash(payload))
    session.add(decision)
    session.flush()
    return decision
