"""Bounded exact-span student-claim clarity gate regressions."""

from datetime import datetime, timezone
import hashlib
import json

import pytest

from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.schemas import CitationMarkerMember
from app.services.student_claim_clarity import (
    apply_student_claim_clarity_gate,
    attach_student_claim_clarity_preflight,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    build_passage_evidence,
)
from app.services.verification_report import (
    ReportAuthorizationError,
    _validate_student_claim_clarity,
    build_inspectable_report_payload,
)


def _artifact(text="Licensing supports competition (Smith, 2020)."):
    marker = "(Smith, 2020)"
    content = b"Licensing can support competition when multiple providers remain."
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="clarity-representation-1",
        canonical_work_id="clarity-work-1",
        content_object_id="clarity-object-1",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        representation_kind="plain_text",
        media_type="text/plain",
        provenance="authorized_upload",
        scope_type="personal_owner",
        scope_id="owner-1",
        identity_verdict="verified",
        identity_confidence=0.99,
        completeness_verdict="complete",
        text_quality="digital",
        edition_or_version=None,
        created_at=now,
        admitted_at=now,
    )
    marker_start = text.index(marker)
    claim = ClaimEvidence(
        claim_id="clarity-claim-1",
        paper_version_id="paper-v1",
        text=text,
        granularity="citation_unit",
        reference_ids=["reference-1"],
        citation_marker=marker,
        citation_markers=[
            CitationMarkerMember(
                text=marker,
                local_start=marker_start,
                local_end=marker_start + len(marker),
                reference_ids=["reference-1"],
                marker_type="parenthetical",
            )
        ],
        citation_marker_type="parenthetical",
        extraction_confidence="high",
        passage_start=100,
        passage_end=100 + len(text),
    )
    artifact = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="reference-1",
        cited_author_label="Smith",
    )
    return attach_citation_use_routes(attach_verification_candidates(artifact))


def _candidate_id(prompt):
    return json.loads(prompt)["candidate_id"]


def test_preflight_does_not_turn_a_vague_word_into_a_verdict_rule():
    artifact = attach_student_claim_clarity_preflight(
        _artifact("Licensing has an important effect (Smith, 2020).")
    )

    finding = artifact.student_claim_clarity.findings[0]
    assert artifact.student_claim_clarity.status == "not_assessed"
    assert finding.status == "uncertain"
    assert finding.reason_code == "clarity_assessment_unavailable"
    assert artifact.student_claim_clarity.blocked_candidate_ids == []


def test_fixed_clear_response_cannot_rewrite_or_return_text():
    artifact = apply_student_claim_clarity_gate(
        _artifact(),
        response_provider=lambda _system, prompt: {
            "candidate_id": _candidate_id(prompt),
            "status": "clear",
            "reason_code": "interpretable_relationship",
            "confidence": "high",
            "problem_ranges": [],
        },
    )

    finding = artifact.student_claim_clarity.findings[0]
    assert artifact.student_claim_clarity.status == "complete"
    assert finding.status == "clear"
    assert finding.problem_segments == []
    assert artifact.student_claim_clarity.decision_applied is False


def test_typed_abstention_resolves_problem_ids_to_exact_student_text():
    def response(_system, prompt):
        payload = json.loads(prompt)
        return {
            "candidate_id": payload["candidate_id"],
            "status": "not_assessed",
            "reason_code": "internally_underspecified_relationship",
            "confidence": "high",
            "problem_ranges": [{"start_id": "t002", "end_id": "t003"}],
        }

    artifact = apply_student_claim_clarity_gate(
        _artifact("The policy changes something important (Smith, 2020)."),
        response_provider=response,
    )

    finding = artifact.student_claim_clarity.findings[0]
    segment = finding.problem_segments[0]
    assert finding.status == "not_assessed"
    assert segment.text == artifact.claim.text[segment.local_start:segment.local_end]
    assert artifact.student_claim_clarity.blocked_candidate_ids == [finding.candidate_id]
    assert "something important" not in finding.explanation


def test_invented_or_cross_segment_token_range_fails_closed():
    artifact = apply_student_claim_clarity_gate(
        _artifact(),
        response_provider=lambda _system, prompt: {
            "candidate_id": _candidate_id(prompt),
            "status": "not_assessed",
            "reason_code": "semantically_uninterpretable_wording",
            "confidence": "high",
            "problem_ranges": [{"start_id": "t000", "end_id": "t999"}],
        },
    )

    finding = artifact.student_claim_clarity.findings[0]
    assert artifact.student_claim_clarity.status == "incomplete"
    assert finding.status == "uncertain"
    assert finding.reason_code == "clarity_uncertain"
    assert finding.problem_segments == []


def test_invalid_status_reason_pair_fails_closed():
    artifact = apply_student_claim_clarity_gate(
        _artifact(),
        response_provider=lambda _system, prompt: {
            "candidate_id": _candidate_id(prompt),
            "status": "clear",
            "reason_code": "conflicting_internal_scope",
            "confidence": "high",
            "problem_ranges": [],
        },
    )

    assert artifact.student_claim_clarity.status == "incomplete"
    assert artifact.student_claim_clarity.findings[0].status == "uncertain"


def test_unresolved_application_context_abstains_without_model_call():
    artifact = _artifact()
    candidate_set = artifact.verification_candidates
    candidates = [
        candidate.model_copy(update={"requires_antecedent_context": True})
        for candidate in candidate_set.candidates
    ]
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={"context_dependency_status": "unresolved"}
            ),
            "verification_candidates": candidate_set.model_copy(
                update={"candidates": candidates}
            ),
        }
    )
    called = False

    def response(_system, _prompt):
        nonlocal called
        called = True
        return {}

    assessed = apply_student_claim_clarity_gate(
        artifact, response_provider=response
    )

    assert called is False
    finding = assessed.student_claim_clarity.findings[0]
    assert finding.status == "not_assessed"
    assert finding.reason_code == "unresolved_local_reference"


def test_report_payload_persists_gate_and_rejects_tampered_explanation():
    artifact = apply_student_claim_clarity_gate(
        _artifact(),
        response_provider=lambda _system, prompt: {
            "candidate_id": _candidate_id(prompt),
            "status": "clear",
            "reason_code": "interpretable_relationship",
            "confidence": "high",
            "problem_ranges": [],
        },
    )
    payload = build_inspectable_report_payload(artifact)
    assert payload["student_claim_clarity"]["findings"][0]["status"] == "clear"
    _validate_student_claim_clarity(artifact)

    finding = artifact.student_claim_clarity.findings[0]
    tampered = artifact.model_copy(
        update={
            "student_claim_clarity": artifact.student_claim_clarity.model_copy(
                update={
                    "findings": [
                        finding.model_copy(update={"explanation": "Rewritten student claim."})
                    ]
                }
            )
        }
    )
    with pytest.raises(ReportAuthorizationError):
        _validate_student_claim_clarity(tampered)
