import hashlib

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.security import AuthenticatedPrincipal, REPORT_SOURCE_CAPABILITY
from app.services.alternate_edition import AlternateEditionRecord
from app.services.edition_review_workflow import (
    EDITION_REVIEW_CAPABILITY, EditionReviewSubmission, record_edition_review,
)


def fixture():
    digest = lambda value: hashlib.sha256(value).hexdigest()
    source = b"Synthetic source; not real edition evidence."
    passage = b"Synthetic publication statement."
    record = AlternateEditionRecord(
        submitted_reference_sha256=digest(b"Reference"),
        retrieved_representation_sha256=digest(source),
        work_identity="verified", relationship="verified_alternate_edition",
        evidence=tuple(dict(evidence_id=purpose, purpose=purpose,
                            representation_sha256=digest(source),
                            passage_sha256=digest(passage))
                       for purpose in ("work_identity", "edition_relationship")),
    )
    return dict(
        principal=AuthenticatedPrincipal(provider="test", subject="reviewer",
            scope_type="personal_owner", scope_id="one",
            capabilities={EDITION_REVIEW_CAPABILITY, REPORT_SOURCE_CAPABILITY}),
        scope_type="personal_owner", scope_id="one", record=record,
        submitted_reference=b"Reference", representations={digest(source): source},
        passages={e.evidence_id: passage for e in record.evidence},
        submission=EditionReviewSubmission(record_sha256=record.review_input_sha256(),
            decision="confirmed", reviewed_evidence_ids=tuple(e.evidence_id for e in record.evidence),
            notes="Reviewer observation."),
    )


@pytest.mark.parametrize("decision", ["confirmed", "uncertain", "rejected"])
def test_decision_is_new_record_not_admission_or_task_grant(decision):
    args = fixture()
    original = args["record"].model_dump_json()
    args["submission"] = args["submission"].model_copy(update={"decision": decision})
    result = record_edition_review(**args)
    assert result.human_verified == (decision == "confirmed")
    assert result.human_review.reviewer_id == "reviewer"
    assert result.human_review.reviewed_at.tzinfo is not None
    assert not result.quotation_usable and not result.paraphrase_usable
    assert args["record"].model_dump_json() == original
    assert AlternateEditionRecord.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("field,value", [
    ("scope_id", "another"), ("submitted_reference", b"Changed"),
    ("passages", {}), ("representations", {}),
])
def test_unresolved_or_cross_scope_snapshot_rejected(field, value):
    args = fixture(); args[field] = value
    with pytest.raises(ValueError):
        record_edition_review(**args)


@pytest.mark.parametrize("capability", [EDITION_REVIEW_CAPABILITY, REPORT_SOURCE_CAPABILITY])
def test_both_capabilities_required(capability):
    args = fixture()
    args["principal"] = args["principal"].model_copy(update={"capabilities": frozenset({capability})})
    with pytest.raises(HTTPException) as error:
        record_edition_review(**args)
    assert error.value.status_code == 403


@pytest.mark.parametrize("field,value", [
    ("record_sha256", "0" * 64), ("reviewed_evidence_ids", ("work_identity",)),
    ("reviewed_evidence_ids", ("work_identity", "work_identity", "edition_relationship")),
    ("notes", " "),
])
def test_stale_or_partial_decision_rejected(field, value):
    args = fixture()
    args["submission"] = args["submission"].model_copy(update={field: value})
    with pytest.raises(ValueError):
        record_edition_review(**args)


def test_client_cannot_choose_reviewer_or_time():
    value = fixture()["submission"].model_dump()
    with pytest.raises(ValidationError):
        EditionReviewSubmission(**value, reviewer_id="someone-else")


@pytest.mark.parametrize("field", ["passages", "representations"])
def test_changed_bytes_rejected(field):
    args = fixture(); key = next(iter(args[field])); args[field][key] = b"changed"
    with pytest.raises(ValueError):
        record_edition_review(**args)
