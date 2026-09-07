"""Source-blind repair and blue Coverage regressions."""

from datetime import datetime, timezone
import hashlib
import json

import pytest
from pydantic import ValidationError

from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.factual_facet_composition import (
    FactualFacetOutcome,
    compose_factual_facets,
    factual_outcome_presentation,
    summarize_factual_source_use,
)
from app.services.schemas import CitationMarkerMember
from app.services.student_statement_interpretation import (
    attach_source_blind_interpretations,
    interpret_student_statement,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    build_passage_evidence,
)


_SOURCE_TEXT = "PRIVATE SOURCE CONTENT MUST NOT ENTER INTERPRETATION"


def _artifact(text: str):
    marker = "(Smith, 2020)"
    now = datetime.now(timezone.utc)
    content = _SOURCE_TEXT.encode()
    source = AuthorizedRepresentation(
        representation_id="interpretation-representation-1",
        canonical_work_id="interpretation-work-1",
        content_object_id="interpretation-object-1",
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
        claim_id="interpretation-claim-1",
        paper_version_id="paper-v1",
        text=text,
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
        passage_start=50,
        passage_end=50 + len(text),
    )
    artifact = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="reference-1",
        cited_author_label="Smith",
    )
    return attach_citation_use_routes(attach_verification_candidates(artifact))


def _candidate_id(artifact):
    return next(
        item.candidate_id
        for item in artifact.citation_use_routing.routes
        if item.relationship_judgment_allowed
    )


def test_clear_mla_style_wording_remains_accuracy_eligible_and_source_blind():
    artifact = _artifact("Brand awareness engages audiences (Smith, 2020).")
    candidate_id = _candidate_id(artifact)

    def response(_system, prompt):
        assert _SOURCE_TEXT not in prompt
        payload = json.loads(prompt)
        assert payload["source_evidence"] == []
        return {
            "candidate_id": payload["candidate_id"],
            "status": "as_written",
            "interpreted_statement": None,
            "reason_code": "stable_as_written",
            "confidence": "high",
            "repair_operations": [],
        }

    result = interpret_student_statement(
        artifact, candidate_id, response_provider=response
    )

    assert result.status == "as_written"
    assert result.accuracy_judgment_allowed is True
    assert result.coverage_judgment_allowed is False
    assert result.source_evidence_received is False


def test_source_blind_interpretation_is_persisted_on_the_artifact():
    artifact = _artifact("Brand awareness engages audiences (Smith, 2020).")

    def response(_system, prompt):
        assert _SOURCE_TEXT not in prompt
        payload = json.loads(prompt)
        return {
            "candidate_id": payload["candidate_id"],
            "status": "as_written",
            "interpreted_statement": None,
            "reason_code": "stable_as_written",
            "confidence": "high",
            "repair_operations": [],
        }

    result = attach_source_blind_interpretations(
        artifact, response_provider=response
    )

    assert len(result.student_statement_interpretations) == 1
    assert result.student_statement_interpretations[0].status == "as_written"
    assert result.student_statement_interpretations[0].source_evidence_received is False


def test_unresolved_antecedent_abstains_before_interpretation_model_call():
    artifact = _artifact(
        "This is especially true in industries with complex market structures "
        "(Smith, 2020)."
    )
    candidate_id = _candidate_id(artifact)
    candidate_set = artifact.verification_candidates
    artifact = artifact.model_copy(
        update={
            "verification_candidates": candidate_set.model_copy(
                update={
                    "candidates": [
                        candidate.model_copy(
                            update={"requires_antecedent_context": True}
                        )
                        if candidate.candidate_id == candidate_id
                        else candidate
                        for candidate in candidate_set.candidates
                    ]
                }
            )
        }
    )
    called = False

    def response(_system, _prompt):
        nonlocal called
        called = True
        return {}

    result = interpret_student_statement(
        artifact, candidate_id, response_provider=response
    )

    assert called is False
    assert result.status == "not_assessed"
    assert result.failure_code == "context_unresolved"
    assert result.accuracy_judgment_allowed is False
    assert result.coverage_judgment_allowed is False
    assert result.prompt_sha256 is None


def test_candidate_one_semantic_repair_allows_coverage_but_not_accuracy():
    artifact = _artifact(
        "Companies need to promote and engage audiences through brand awareness "
        "(Smith, 2020)."
    )
    candidate_id = _candidate_id(artifact)

    def response(_system, prompt):
        payload = json.loads(prompt)
        tokens = payload["candidate_tokens"]
        promote = next(item["token_id"] for item in tokens if item["text"] == "promote")
        engage = next(item["token_id"] for item in tokens if item["text"] == "engage")
        return {
            "candidate_id": payload["candidate_id"],
            "status": "semantic_repair",
            "interpreted_statement": (
                "Companies need to promote products to and engage with audiences "
                "through brand awareness."
            ),
            "reason_code": "single_plausible_semantic_repair",
            "confidence": "medium",
            "repair_operations": [
                {
                    "kind": "supplied_object",
                    "ranges": [{"start_id": promote, "end_id": promote}],
                },
                {
                    "kind": "supplied_preposition",
                    "ranges": [{"start_id": engage, "end_id": engage}],
                },
            ],
        }

    result = interpret_student_statement(
        artifact, candidate_id, response_provider=response
    )

    assert result.status == "semantic_repair"
    assert result.accuracy_judgment_allowed is False
    assert result.coverage_judgment_allowed is True
    assert [item.kind for item in result.repair_operations] == [
        "supplied_object",
        "supplied_preposition",
    ]

    def proposal(_system, prompt):
        payload = json.loads(prompt)
        assert payload["source_evidence"] == []
        assert payload["student_interpretation"]["status"] == "semantic_repair"
        tokens = payload["candidate_tokens"]
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": result.interpreted_statement,
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [
                        {
                            "start_id": tokens[0]["token_id"],
                            "end_id": tokens[-1]["token_id"],
                        }
                    ],
                    "constraint_keys": [],
                }
            ],
            "shared_constraints": [],
            "structural_edges": [],
            "uncovered_ranges": [],
        }

    def preservation(_system, prompt):
        payload = json.loads(prompt)
        assert payload["source_evidence"] == []
        assert payload["student_interpretation"]["status"] == "semantic_repair"
        return {
            "candidate_id": payload["candidate_id"],
            "reviews": [
                {
                    "facet_id": item["facet_id"],
                    "status": "faithful",
                    "confidence": "high",
                }
                for item in payload["proposed_facets"]
            ],
            "constraint_reviews": [],
            "uncovered_status": "no_material_factual_wording",
            "material_uncovered_ids": [],
        }

    composition = compose_factual_facets(
        artifact,
        candidate_id,
        proposal_provider=proposal,
        preservation_provider=preservation,
        interpretation=result,
    )
    assert composition.status == "complete"
    assert composition.contract_version == "complete-proposition-scope-composition-v3"
    assert composition.interpretation_id == result.interpretation_id
    assert composition.interpretation_status == "semantic_repair"
    assert composition.accuracy_judgment_allowed is False
    assert composition.coverage_judgment_allowed is True


def test_multiple_plausible_interpretations_abstain():
    artifact = _artifact("Companies promote and engage audiences (Smith, 2020).")
    candidate_id = _candidate_id(artifact)
    result = interpret_student_statement(
        artifact,
        candidate_id,
        response_provider=lambda _system, prompt: {
            "candidate_id": json.loads(prompt)["candidate_id"],
            "status": "not_assessed",
            "interpreted_statement": None,
            "reason_code": "multiple_plausible_interpretations",
            "confidence": "medium",
            "repair_operations": [],
        },
    )

    assert result.status == "not_assessed"
    assert result.accuracy_judgment_allowed is False
    assert result.coverage_judgment_allowed is False


def test_invented_repair_range_fails_closed():
    artifact = _artifact("Companies promote audiences (Smith, 2020).")
    candidate_id = _candidate_id(artifact)
    result = interpret_student_statement(
        artifact,
        candidate_id,
        response_provider=lambda _system, prompt: {
            "candidate_id": json.loads(prompt)["candidate_id"],
            "status": "semantic_repair",
            "interpreted_statement": "Companies promote products to audiences.",
            "reason_code": "single_plausible_semantic_repair",
            "confidence": "medium",
            "repair_operations": [
                {
                    "kind": "supplied_object",
                    "ranges": [{"start_id": "t000", "end_id": "t999"}],
                }
            ],
        },
    )

    assert result.status == "uncertain"
    assert result.failure_code == "provider_or_contract_failure"


def test_blue_coverage_requires_semantic_repair_and_does_not_claim_accuracy():
    with pytest.raises(ValidationError):
        FactualFacetOutcome(
            facet_id="f1",
            direction="source_content_coverage",
            interpretation_status="as_written",
        )
    with pytest.raises(ValidationError):
        FactualFacetOutcome(
            facet_id="f1",
            direction="supports",
            interpretation_status="semantic_repair",
        )

    summary = summarize_factual_source_use(
        [
            FactualFacetOutcome(
                facet_id="f1",
                direction="source_content_coverage",
                interpretation_status="semantic_repair",
            )
        ]
    )
    assert summary.status == "source_content_coverage_found"
    assert summary.source_content_coverage_count == 1
    assert "accuracy was not assessed" in summary.limitation
    presentation = factual_outcome_presentation("source_content_coverage")
    assert presentation.display_label == "Coverage"
    assert presentation.color_token == "blue"
    assert presentation.non_color_marker == "coverage"
    assert presentation.accuracy_assessed is False
