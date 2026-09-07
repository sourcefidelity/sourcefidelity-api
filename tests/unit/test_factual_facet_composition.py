"""Narrow factual facet proposal/preservation and mixed-summary regressions."""

from datetime import datetime, timezone
import hashlib
import json
import re

from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.factual_facet_composition import (
    FactualFacetOutcome,
    compose_factual_facets,
    summarize_factual_source_use,
)
from app.services.schemas import CitationMarkerMember
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    build_passage_evidence,
)


_SOURCE_ONLY_TEXT = "PRIVATE SOURCE SENTENCE MUST NEVER ENTER COMPOSITION"


def _artifact(text: str, marker: str = "(Smith, 2020)"):
    content = _SOURCE_ONLY_TEXT.encode()
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="facet-composition-source-1",
        canonical_work_id="facet-composition-work-1",
        content_object_id="facet-composition-object-1",
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
        claim_id="facet-composition-claim-1",
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
        top_k=3,
        active_reference_id="reference-1",
        cited_author_label="Smith",
    )
    return attach_citation_use_routes(attach_verification_candidates(artifact))


def _eligible_candidate_id(artifact) -> str:
    return next(
        route.candidate_id
        for route in artifact.citation_use_routing.routes
        if route.relationship_judgment_allowed
    )


def _token_id(payload: dict, word: str) -> str:
    for token in payload["candidate_tokens"]:
        normalized = re.sub(r"\W+", "", token["text"]).casefold()
        if normalized == word.casefold():
            return token["token_id"]
    raise AssertionError(f"token not found: {word}")


def _composite_proposal(_system: str, prompt: str) -> dict:
    assert _SOURCE_ONLY_TEXT not in prompt
    payload = json.loads(prompt)
    conservative = _token_id(payload, "conservative")
    values = _token_id(payload, "values")
    conjunction = _token_id(payload, "and")
    censorship = _token_id(payload, "censorship")
    exists = _token_id(payload, "exist")
    return {
        "candidate_id": payload["candidate_id"],
        "status": "complete",
        "facets": [
            {
                "proposal_key": "p001",
                "proposition_form": "complete_factual_proposition",
                "checking_gloss": "Conservative values exist.",
                "gloss_inherits_from_complete_unit": False,
                "ranges": [
                    {"start_id": conservative, "end_id": values},
                    {"start_id": exists, "end_id": exists},
                ],
                "constraint_keys": ["c001"],
            },
            {
                "proposal_key": "p002",
                "proposition_form": "complete_factual_proposition",
                "checking_gloss": "Censorship exists.",
                "gloss_inherits_from_complete_unit": False,
                "ranges": [
                    {"start_id": censorship, "end_id": censorship},
                    {"start_id": exists, "end_id": exists},
                ],
                "constraint_keys": ["c001"],
            },
        ],
        "shared_constraints": [
            {
                "constraint_key": "c001",
                "kind": "shared_predicate",
                "ranges": [{"start_id": exists, "end_id": exists}],
                "applies_to": ["p001", "p002"],
                "scope": "shared_exact",
            }
        ],
        "structural_edges": [
            {
                "edge_key": "e001",
                "kind": "coordination",
                "ranges": [{"start_id": conjunction, "end_id": conjunction}],
                "connects": ["p001", "p002"],
            }
        ],
        "uncovered_ranges": [],
    }


def _faithful_preservation(_system: str, prompt: str) -> dict:
    assert _SOURCE_ONLY_TEXT not in prompt
    payload = json.loads(prompt)
    return {
        "candidate_id": payload["candidate_id"],
        "reviews": [
            {"facet_id": item["facet_id"], "status": "faithful", "confidence": "high"}
            for item in payload["proposed_facets"]
        ],
        "constraint_reviews": [
            {
                "constraint_id": item["constraint_id"],
                "status": "faithful",
                "confidence": "high",
            }
            for item in payload["shared_constraints"]
        ],
        "uncovered_status": "no_material_factual_wording",
        "material_uncovered_ids": [],
    }


def test_composite_exact_span_facets_may_reuse_shared_predicate() -> None:
    artifact = _artifact(
        "Conservative values and censorship exist (Smith, 2020)."
    )

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=_composite_proposal,
        preservation_provider=_faithful_preservation,
    )

    assert result.status == "complete"
    assert result.coverage_status == "complete"
    assert len(result.proposed_facets) == 2
    assert len(result.scope_constraints) == 1
    assert len(result.structural_edges) == 1
    assert len(result.accepted_facet_ids) == 2
    assert len(result.accepted_constraint_ids) == 1
    assert {facet.checking_gloss for facet in result.proposed_facets} == {
        "Conservative values exist.",
        "Censorship exists.",
    }
    assert result.decision_applied is False
    assert result.processing_boundary == "local"


def test_proposal_must_account_for_every_candidate_token() -> None:
    artifact = _artifact("Licensing preserves competition (Smith, 2020).")

    def incomplete_proposal(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        first = payload["candidate_tokens"][0]["token_id"]
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": "Licensing is asserted.",
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [{"start_id": first, "end_id": first}],
                    "constraint_keys": [],
                }
            ],
            "shared_constraints": [],
            "structural_edges": [],
            "uncovered_ranges": [],
        }

    called = False

    def preservation(_system: str, _prompt: str) -> dict:
        nonlocal called
        called = True
        return {}

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=incomplete_proposal,
        preservation_provider=preservation,
    )

    assert result.status == "incomplete"
    assert result.failure_code == "proposal_contract_invalid"
    assert called is False


def test_antecedent_dependent_proposition_requires_explicit_resolved_inheritance() -> None:
    artifact = _artifact(
        "These individuals are usually viewed as emotionless (Smith, 2020)."
    )
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={"context_dependency_status": "resolved"}
            )
        }
    )
    candidate_id = _eligible_candidate_id(artifact)
    candidate = next(
        item
        for item in artifact.verification_candidates.candidates
        if item.candidate_id == candidate_id
    )
    assert candidate.requires_antecedent_context is True

    def unresolved_gloss(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        first = payload["candidate_tokens"][0]["token_id"]
        last = payload["candidate_tokens"][-1]["token_id"]
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": "These individuals are usually viewed as emotionless.",
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [{"start_id": first, "end_id": last}],
                    "constraint_keys": [],
                }
            ],
            "shared_constraints": [],
            "structural_edges": [],
            "uncovered_ranges": [],
        }

    preservation_called = False

    def preservation(_system: str, _prompt: str) -> dict:
        nonlocal preservation_called
        preservation_called = True
        return {}

    result = compose_factual_facets(
        artifact,
        candidate_id,
        proposal_provider=unresolved_gloss,
        preservation_provider=preservation,
    )

    assert result.failure_code == "proposal_contract_invalid"
    assert preservation_called is False


def test_unknown_token_id_fails_closed_before_preservation() -> None:
    artifact = _artifact("Licensing preserves competition (Smith, 2020).")

    def invented_range(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": "Licensing is asserted.",
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [{"start_id": "t999", "end_id": "t999"}],
                    "constraint_keys": [],
                }
            ],
            "shared_constraints": [],
            "structural_edges": [],
            "uncovered_ranges": [],
        }

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=invented_range,
        preservation_provider=lambda _system, _prompt: {},
    )

    assert result.status == "incomplete"
    assert result.failure_code == "proposal_contract_invalid"
    assert result.accepted_facet_ids == []


def test_proposal_provider_failure_is_distinct_from_invalid_contract() -> None:
    artifact = _artifact("Licensing preserves competition (Smith, 2020).")

    def unavailable(_system: str, _prompt: str) -> dict:
        raise RuntimeError("provider unavailable")

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=unavailable,
        preservation_provider=lambda _system, _prompt: {},
    )

    assert result.status == "incomplete"
    assert result.failure_code == "provider_or_runtime_failure"
    assert result.proposal_prompt_sha256 is not None


def test_preservation_rejection_retains_mixed_coverage_without_accepting_facet() -> None:
    artifact = _artifact(
        "Conservative values and censorship exist (Smith, 2020)."
    )

    def preservation(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        facet_ids = [item["facet_id"] for item in payload["proposed_facets"]]
        return {
            "candidate_id": payload["candidate_id"],
            "reviews": [
                {"facet_id": facet_ids[0], "status": "faithful", "confidence": "high"},
                {
                    "facet_id": facet_ids[1],
                    "status": "material_detail_omitted",
                    "confidence": "high",
                },
            ],
            "constraint_reviews": [
                {
                    "constraint_id": item["constraint_id"],
                    "status": "faithful",
                    "confidence": "high",
                }
                for item in payload["shared_constraints"]
            ],
            "uncovered_status": "no_material_factual_wording",
            "material_uncovered_ids": [],
        }

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=_composite_proposal,
        preservation_provider=preservation,
    )

    assert result.status == "complete"
    assert result.coverage_status == "partial"
    assert len(result.accepted_facet_ids) == 1
    assert {item.status for item in result.preservation_findings} == {
        "faithful",
        "material_detail_omitted",
    }


def test_nonfactual_proposal_is_rejected_by_preservation() -> None:
    artifact = _artifact(
        "Conservative values and censorship exist (Smith, 2020)."
    )

    def preservation(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        facet_ids = [item["facet_id"] for item in payload["proposed_facets"]]
        return {
            "candidate_id": payload["candidate_id"],
            "reviews": [
                {
                    "facet_id": facet_id,
                    "status": "not_factual_source_representation",
                    "confidence": "high",
                }
                for facet_id in facet_ids
            ],
            "constraint_reviews": [
                {
                    "constraint_id": item["constraint_id"],
                    "status": "faithful",
                    "confidence": "high",
                }
                for item in payload["shared_constraints"]
            ],
            "uncovered_status": "no_material_factual_wording",
            "material_uncovered_ids": [],
        }

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=_composite_proposal,
        preservation_provider=preservation,
    )

    assert result.status == "complete"
    assert result.coverage_status == "partial"
    assert result.accepted_facet_ids == []
    assert {item.status for item in result.preservation_findings} == {
        "not_factual_source_representation"
    }


def test_legacy_fragment_only_proposal_contract_fails_before_preservation() -> None:
    artifact = _artifact("Licensing preserves competition (Smith, 2020).")

    def legacy_fragment(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        first = payload["candidate_tokens"][0]["token_id"]
        last = payload["candidate_tokens"][-1]["token_id"]
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "ranges": [{"start_id": first, "end_id": last}],
                }
            ],
            "uncovered_ranges": [],
        }

    called = False

    def preservation(_system: str, _prompt: str) -> dict:
        nonlocal called
        called = True
        return {}

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=legacy_fragment,
        preservation_provider=preservation,
    )

    assert result.status == "incomplete"
    assert result.failure_code == "proposal_contract_invalid"
    assert called is False


def test_collective_quantity_and_unresolved_modifier_remain_explicit() -> None:
    artifact = _artifact(
        "Many vampire-related gore and sex scenes are shown (Smith, 2020)."
    )

    def proposal(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        many = _token_id(payload, "many")
        vampire = _token_id(payload, "vampirerelated")
        gore = _token_id(payload, "gore")
        conjunction = _token_id(payload, "and")
        sex = _token_id(payload, "sex")
        scenes = _token_id(payload, "scenes")
        shown = _token_id(payload, "shown")
        return {
            "candidate_id": payload["candidate_id"],
            "status": "complete",
            "facets": [
                {
                    "proposal_key": "p001",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": "Vampire-related gore scenes are shown.",
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [
                        {"start_id": vampire, "end_id": gore},
                        {"start_id": scenes, "end_id": shown},
                    ],
                    "constraint_keys": ["c001", "c002", "c003"],
                },
                {
                    "proposal_key": "p002",
                    "proposition_form": "complete_factual_proposition",
                    "checking_gloss": "Sex scenes are shown.",
                    "gloss_inherits_from_complete_unit": False,
                    "ranges": [
                        {"start_id": sex, "end_id": sex},
                        {"start_id": scenes, "end_id": shown},
                    ],
                    "constraint_keys": ["c001", "c002", "c003"],
                },
            ],
            "shared_constraints": [
                {
                    "constraint_key": "c001",
                    "kind": "quantity",
                    "ranges": [{"start_id": many, "end_id": many}],
                    "applies_to": ["p001", "p002"],
                    "scope": "collective",
                },
                {
                    "constraint_key": "c002",
                    "kind": "modifier",
                    "ranges": [{"start_id": vampire, "end_id": vampire}],
                    "applies_to": ["p001", "p002"],
                    "scope": "unresolved",
                },
                {
                    "constraint_key": "c003",
                    "kind": "shared_predicate",
                    "ranges": [{"start_id": scenes, "end_id": shown}],
                    "applies_to": ["p001", "p002"],
                    "scope": "shared_exact",
                },
            ],
            "structural_edges": [
                {
                    "edge_key": "e001",
                    "kind": "coordination",
                    "ranges": [{"start_id": conjunction, "end_id": conjunction}],
                    "connects": ["p001", "p002"],
                }
            ],
            "uncovered_ranges": [],
        }

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=proposal,
        preservation_provider=_faithful_preservation,
    )

    assert result.status == "complete"
    assert result.coverage_status == "complete"
    assert {(item.kind, item.scope) for item in result.scope_constraints} == {
        ("quantity", "collective"),
        ("modifier", "unresolved"),
        ("shared_predicate", "shared_exact"),
    }
    assert len(result.accepted_constraint_ids) == 3


def test_constraint_linkage_must_be_bidirectional() -> None:
    artifact = _artifact("Conservative values and censorship exist (Smith, 2020).")

    def one_sided_link(system: str, prompt: str) -> dict:
        proposal = _composite_proposal(system, prompt)
        proposal["facets"][1]["constraint_keys"] = []
        return proposal

    called = False

    def preservation(_system: str, _prompt: str) -> dict:
        nonlocal called
        called = True
        return {}

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=one_sided_link,
        preservation_provider=preservation,
    )

    assert result.status == "incomplete"
    assert result.failure_code == "proposal_contract_invalid"
    assert called is False


def test_preservation_must_review_every_shared_constraint() -> None:
    artifact = _artifact("Conservative values and censorship exist (Smith, 2020).")

    def missing_constraint_review(_system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
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

    result = compose_factual_facets(
        artifact,
        _eligible_candidate_id(artifact),
        proposal_provider=_composite_proposal,
        preservation_provider=missing_constraint_review,
    )

    assert result.status == "incomplete"
    assert result.failure_code == "preservation_contract_invalid"
    assert result.accepted_facet_ids == []


def test_student_application_abstains_without_calling_either_provider() -> None:
    artifact = _artifact(
        "Applying Smith's framework to Dracula shows that the film resists "
        "convention (Smith, 2020)."
    )
    candidate_id = artifact.citation_use_routing.routes[0].candidate_id
    called = False

    def provider(_system: str, _prompt: str) -> dict:
        nonlocal called
        called = True
        return {}

    result = compose_factual_facets(
        artifact,
        candidate_id,
        proposal_provider=provider,
        preservation_provider=provider,
    )

    assert result.status == "not_assessed"
    assert result.failure_code == "candidate_not_eligible"
    assert called is False


def test_mixed_facet_summary_does_not_hide_supported_or_contradicted_parts() -> None:
    summary = summarize_factual_source_use(
        [
            FactualFacetOutcome(facet_id="f1", direction="supports"),
            FactualFacetOutcome(facet_id="f2", direction="contradicts"),
            FactualFacetOutcome(facet_id="f3", direction="not_assessed"),
        ]
    )

    assert summary.status == "mixed_source_content_use"
    assert summary.supported_or_qualified_count == 1
    assert summary.contradicted_count == 1
    assert summary.not_assessed_count == 1


def test_contradiction_only_does_not_infer_engagement() -> None:
    summary = summarize_factual_source_use(
        [FactualFacetOutcome(facet_id="f1", direction="contradicts")]
    )

    assert summary.status == "factual_inconsistency_found"
    assert "does not determine" in summary.limitation


def test_unassessed_facets_do_not_become_no_engagement_claim() -> None:
    summary = summarize_factual_source_use(
        [FactualFacetOutcome(facet_id="f1", direction="not_assessed")]
    )

    assert summary.status == "not_assessed"
    assert summary.supported_or_qualified_count == 0
