"""Regressions for the bounded passage-relevance gate."""

from datetime import datetime, timezone
import hashlib
import json

import pytest

from app.services.evidence_conditioned_judgment import (
    apply_evidence_conditioned_unit_judgment,
)
from app.services.passage_relevance import (
    apply_passage_relevance_gate,
    assess_abstract_relevance,
)
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.evidence_obligations import attach_evidence_obligations
from app.services.student_statement_interpretation import StudentStatementInterpretation
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    VerificationVerdict,
    build_passage_evidence,
)


def _artifact(*, top_k=3, page_count=4):
    source_pages = [
            "Licensing supports competition by preserving diverse providers.",
            "Licensing can reduce barriers to entry for new market participants.",
            "Licensing supports competition through filing procedures.",
            "Licensing reduces market barriers in historical settings.",
        ]
    while len(source_pages) < page_count:
        source_pages.append(
            f"Page {len(source_pages) + 1}: Licensing supports competition and "
            f"reduces market barriers in condition {len(source_pages) + 1}."
        )
    source_text = "\f".join(source_pages)
    content = source_text.encode()
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="representation-1",
        canonical_work_id="work-1",
        content_object_id="object-1",
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
        edition_or_version="edition-1",
        created_at=now,
        admitted_at=now,
    )
    text = (
        "Licensing supports competition and reduces barriers "
        "(Smith, 2020). Contact reviewer@example.edu"
    )
    claim = ClaimEvidence(
        claim_id="citation-unit-1",
        paper_version_id="paper-v1",
        text=text,
        granularity="citation_unit",
        reference_ids=["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        extraction_confidence="high",
        passage_start=100,
        passage_end=100 + len(text),
    )
    return build_passage_evidence(source, claim=claim, top_k=top_k)


def test_relevant_and_partial_candidates_are_selected_without_changing_verdict(
    monkeypatch,
):
    artifact = _artifact()
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        captured["call_kwargs"] = kwargs
        ids = [passage["passage_id"] for passage in payload["passages"]]
        return {
            "assessments": [
                {
                    "passage_id": ids[0],
                    "relevance": "relevant",
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "Addresses licensing and competition.",
                },
                {
                    "passage_id": ids[1],
                    "relevance": "partially_relevant",
                    "evidence_role": "source_synthesis_or_conclusion",
                    "confidence": "medium",
                    "rationale": "Addresses barriers only.",
                },
                {
                    "passage_id": ids[2],
                    "relevance": "not_relevant",
                    "evidence_role": "methods_or_background",
                    "confidence": "high",
                    "rationale": "Procedural material.",
                },
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert assessed.passage_relevance.status == "complete"
    assert assessed.passage_relevance.outcome == "relevant_candidates_found"
    assert assessed.passage_relevance.relevant_passage_ids == [
        passage["passage_id"] for passage in captured["passages"][:2]
    ]
    assert len(captured["passages"]) == 3
    assert "reviewer@example.edu" not in captured["complete_citation_unit"]
    assert "reviewer@example.edu" not in captured["source_attributed_text"]
    assert assessed.passage_relevance.decision_applied is False
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE
    assert captured["call_kwargs"]["disable_thinking"] is True


def test_relevance_prompt_keeps_same_relationship_scope_mismatches_displayable(
    monkeypatch,
):
    artifact = _artifact()
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        passages = json.loads(user_prompt)["passages"]
        return {
            "assessments": [
                {
                    "passage_id": item["passage_id"],
                    "relevance": "partially_relevant",
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "Same relationship with a different outcome scope.",
                }
                for item in passages
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert assessed.passage_relevance.outcome == "relevant_candidates_found"
    normalized_prompt = " ".join(captured["system_prompt"].lower().split())
    assert "national versus global" in normalized_prompt
    assert "do not treat a scope mismatch as support" in normalized_prompt
    assert "reporting-intensity verbs" in normalized_prompt
    assert "states, argues, believes, or emphasizes" in normalized_prompt
    assert "sharing actors or a broad topic" in normalized_prompt
    assert "strategically or critically" in normalized_prompt
    assert assessed.passage_relevance.decision_applied is False
    assert assessed.verdict is VerificationVerdict.INCONCLUSIVE


def test_abstract_relevance_uses_the_same_bounded_contract(monkeypatch):
    claim = _artifact().claim
    captured = {}

    def fake_call(_system_prompt, user_prompt, **kwargs):
        captured.update(json.loads(user_prompt))
        captured["kwargs"] = kwargs
        return {
            "assessments": [
                {
                    "passage_id": "abstract",
                    "relevance": "partially_relevant",
                    "evidence_role": "source_synthesis_or_conclusion",
                    "confidence": "medium",
                    "rationale": "The abstract addresses only one material part.",
                }
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    result = assess_abstract_relevance(
        claim,
        "The abstract discusses competition but not every asserted condition.",
    )

    assert result["status"] == "complete"
    assert result["relevance"] == "partially_relevant"
    assert result["decision_applied"] is False
    assert captured["coverage"] == "abstract_only"
    assert captured["passages"][0]["passage_id"] == "abstract"
    assert captured["kwargs"]["disable_thinking"] is True


@pytest.mark.parametrize('excerpt', ['The abstract discusses competition.', 'discusses competition', 'Invented text.'])
def test_abstract_scope_is_separate_from_specific_excerpt(monkeypatch, excerpt):
    text = 'The abstract discusses competition. Other findings are also summarized.'
    def response(*args, **kwargs):
        return {'assessments':[{'passage_id':'abstract','relevance':'relevant',
            'evidence_role':'source_own_claim_or_finding','confidence':'medium','rationale':'Related content.'}],
            'scope':{'relevance':'generally_relevant','confidence':'medium'},'related_excerpt':excerpt}
    monkeypatch.setattr('app.services.passage_relevance.chat_completion_json',response)
    result = assess_abstract_relevance(_artifact().claim,text)
    assert result['scope_assessment']['relevance'] == 'generally_relevant'
    assert result['scope_assessment']['attention'] is False
    assert result['related_excerpt'] == (excerpt if excerpt == 'The abstract discusses competition.' else '')


def test_document_level_member_evidence_is_typed_and_retained(monkeypatch):
    artifact = _artifact(top_k=1)
    passage = artifact.passages[0].model_copy(
        update={"passage_role": "document_metadata"}
    )
    artifact = artifact.model_copy(update={"passages": [passage]})
    captured = {}

    def fake_call(_system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        return {
            "assessments": [
                {
                    "passage_id": payload["passages"][0]["passage_id"],
                    "relevance": "partially_relevant",
                    "evidence_role": "document_level_member_evidence",
                    "confidence": "high",
                    "rationale": "Identifies the current member's research topic.",
                }
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert captured["passages"][0]["passage_role"] == "document_metadata"
    assert assessed.passage_relevance.outcome == "relevant_candidates_found"
    assert (
        assessed.passage_relevance.assessments[0].evidence_role
        == "document_level_member_evidence"
    )


def test_coverage_repair_is_assessed_separately_from_original_accuracy(monkeypatch):
    artifact = attach_citation_use_routes(
        attach_verification_candidates(_artifact(top_k=1))
    )
    candidate = next(
        item
        for item in artifact.verification_candidates.candidates
        if item.role == "relationship_candidate"
    )
    repaired = "Licensing can preserve competition under some conditions."
    interpretation = StudentStatementInterpretation(
        interpretation_id="interpretation:coverage-repair",
        candidate_id=candidate.candidate_id,
        candidate_text_sha256=hashlib.sha256(candidate.text.encode()).hexdigest(),
        status="semantic_repair",
        interpreted_statement=repaired,
        reason_code="single_plausible_semantic_repair",
        confidence="high",
        accuracy_judgment_allowed=False,
        coverage_judgment_allowed=True,
    )
    artifact = attach_evidence_obligations(
        artifact, interpretations=[interpretation]
    )
    modes = []

    def fake_call(_system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        modes.append(payload["relevance_mode"])
        relevance = (
            "relevant"
            if payload["relevance_mode"] == "coverage_only_semantic_repair"
            else "not_relevant"
        )
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": relevance,
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "Bounded test response.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert modes == ["exact_factual_assertion", "coverage_only_semantic_repair"]
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"
    assert assessed.passage_relevance.relevant_passage_ids == []
    assert len(assessed.passage_relevance.obligation_findings) == 2
    coverage = assessed.passage_relevance.obligation_findings[1]
    assert coverage.obligation_type == "coverage_only_semantic_repair"
    assert coverage.outcome == "relevant_candidates_found"
    assert coverage.relevant_passage_ids


def test_all_not_relevant_uses_all_three_in_shadow_judgment_fallback(
    monkeypatch,
):
    artifact = _artifact()

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "evidence_role": "unclear",
                    "confidence": "high",
                    "rationale": "Does not address the citation assertion.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"

    captured = {}

    def fallback_judgment(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        return {
            "findings": [{
                "segments": [{
                    "start_id": payload["unit_tokens"][0][0],
                    "end_id": payload["unit_tokens"][4][0],
                }],
                "attribution": "cited_source",
                "relationship": "insufficient_evidence",
                "confidence": "low",
                "passage_ids": [payload["passages"][0]["passage_id"]],
                "rationale": "Recall-preserving fallback found no adequate support.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fallback_judgment,
    )
    judged = apply_evidence_conditioned_unit_judgment(assessed)
    assert judged.unit_judgment.status == "incomplete"
    assert judged.unit_judgment.outcome == "segmented_findings"
    assert len(captured["passages"]) == 3
    assert judged.unit_judgment.findings[0].relationship.value == (
        "insufficient_evidence"
    )
    assert any(
        "recall-preserving shadow fallback" in limitation
        for limitation in judged.unit_judgment.limitations
    )
    assert judged.verdict is VerificationVerdict.INCONCLUSIVE


def test_uncertain_relevance_stops_segmented_judgment(monkeypatch):
    artifact = _artifact()

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "uncertain",
                    "evidence_role": "unclear",
                    "confidence": "low",
                    "rationale": "Insufficient context.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    assessed = apply_passage_relevance_gate(artifact)
    judged = apply_evidence_conditioned_unit_judgment(assessed)
    assert assessed.passage_relevance.outcome == "uncertain"
    assert judged.unit_judgment.status == "not_assessed"
    assert judged.unit_judgment.outcome == "uncertain_relevance"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "invented"])
def test_invalid_application_owned_passage_ids_fail_closed(monkeypatch, failure):
    artifact = _artifact()

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        ids = [passage["passage_id"] for passage in payload["passages"]]
        if failure == "missing":
            ids = ids[:-1]
        elif failure == "duplicate":
            ids[-1] = ids[0]
        else:
            ids[-1] = "invented-passage-id"
        return {
            "assessments": [
                {
                    "passage_id": passage_id,
                    "relevance": "relevant",
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "medium",
                    "rationale": "Test.",
                }
                for passage_id in ids
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_call
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.status == "not_assessed"
    assert assessed.passage_relevance.outcome == "not_assessed"
    assert assessed.passage_relevance.relevant_passage_ids == []


def test_mismatched_authorization_never_reaches_relevance_model(monkeypatch):
    artifact = _artifact()
    artifact = artifact.model_copy(
        update={
            "passages": [
                passage.model_copy(update={"authorization_scope_id": "other-owner"})
                for passage in artifact.passages
            ]
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("relevance model must not run")

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fail_if_called
    )
    assessed = apply_passage_relevance_gate(artifact)
    assert assessed.passage_relevance.status == "not_assessed"
    assert assessed.passage_relevance.method == "no_authorized_candidate_passage"


def test_retrieval_can_preserve_ten_candidates_for_calibration():
    artifact = _artifact(top_k=10)
    assert len(artifact.passages) == 4


def test_relevance_gate_uses_bounded_diverse_pool_from_larger_calibration_pool(
    monkeypatch,
):
    artifact = _artifact(top_k=10)
    assert len(artifact.passages) == 4
    captured_ids = []

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured_ids.extend(
            passage["passage_id"] for passage in payload["passages"]
        )
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "evidence_role": "representation_of_other_work",
                    "confidence": "high",
                    "rationale": "Bounded cutoff regression.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert set(captured_ids) == {
        passage.passage_id for passage in artifact.passages
    }
    assert len(captured_ids) == 4
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"


def test_relevance_gate_batches_and_assesses_the_complete_bounded_union(
    monkeypatch,
):
    artifact = _artifact(top_k=10, page_count=8)
    calls = []

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        calls.append([passage["passage_id"] for passage in payload["passages"]])
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "evidence_role": "representation_of_other_work",
                    "confidence": "high",
                    "rationale": "The passage does not address the citation unit.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    monkeypatch.setattr(
        "app.services.passage_relevance.get_provider_config",
        lambda _model: type("Provider", (), {"input_batch_tokens": 1_500})(),
    )
    assessed = apply_passage_relevance_gate(artifact)

    assessed_ids = [value for batch in calls for value in batch]
    assert len(calls) == 3
    assert [len(batch) for batch in calls] == [3, 3, 2]
    assert set(assessed_ids) == {
        passage.passage_id for passage in artifact.passages
    }
    assert assessed.passage_relevance.candidate_count_assessed == 8
    assert assessed.passage_relevance.batch_count == 3
    assert len(assessed.passage_relevance.assessments) == 8
    assert assessed.passage_relevance.outcome == "no_relevant_candidate_passage"


def test_relevance_gate_assesses_a_hash_bound_query_window_not_only_prefix(
    monkeypatch,
):
    artifact = _artifact(top_k=1)
    passage = artifact.passages[0]
    filler = "Unrelated historical background. " * 70
    target = "Licensing supports competition and reduces barriers."
    long_text = filler + target
    artifact = artifact.model_copy(
        update={
            "passages": [
                passage.model_copy(
                    update={
                        "text": long_text,
                        "character_end": passage.character_start + len(long_text),
                    }
                )
            ]
        }
    )
    captured = {}

    def fake_relevance(_system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload["passages"][0])
        return {
            "assessments": [
                {
                    "passage_id": payload["passages"][0]["passage_id"],
                    "relevance": "relevant",
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "The bounded window addresses the claim.",
                }
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    assessed = apply_passage_relevance_gate(artifact)
    assessment = assessed.passage_relevance.assessments[0]

    assert target in captured["text"]
    assert len(captured["text"]) == 1_400
    assert assessment.assessment_input_truncated is True
    assert assessment.assessed_text_offset_start > 0
    assert assessment.assessed_text_offset_end == (
        assessment.assessed_text_offset_start + len(captured["text"])
    )
    expected_hash = hashlib.sha256(captured["text"].encode("utf-8")).hexdigest()
    assert assessment.assessed_text_sha256 == expected_hash
    assert assessment.model_input_text_sha256 == expected_hash


def test_relevance_window_uses_bounded_concepts_to_reach_late_scope_evidence(
    monkeypatch,
):
    artifact = _artifact(top_k=1)
    passage = artifact.passages[0]
    target = "Protectionism does not create national prosperity."
    prefix = ("Protectionism appears in historical background. " * 45)
    evidence = "Reducing protectionism increases global well-being for trading participants."
    long_text = prefix + evidence
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={"text": target, "citation_marker": ""}
            ),
            "passages": [
                passage.model_copy(
                    update={
                        "text": long_text,
                        "character_end": passage.character_start + len(long_text),
                    }
                )
            ],
        }
    )
    captured = {}

    def fake_relevance(_system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload["passages"][0])
        return {
            "assessments": [
                {
                    "passage_id": payload["passages"][0]["passage_id"],
                    "relevance": "partially_relevant",
                    "evidence_role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "Same relationship with different geographic scope.",
                }
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert evidence in captured["text"]
    assert assessed.passage_relevance.assessments[0].assessed_text_offset_start > 0


def test_relevance_gate_excludes_unranked_extraction_extras(monkeypatch):
    artifact = _artifact(top_k=3)
    protected_id = artifact.passages[0].passage_id
    artifact = artifact.model_copy(
        update={
            "relationship": artifact.relationship.model_copy(
                update={"passage_ids": [protected_id]}
            )
        }
    )
    captured_ids = []

    def fake_relevance(_system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        captured_ids.extend(item["passage_id"] for item in payload["passages"])
        return {
            "assessments": [
                {
                    "passage_id": item["passage_id"],
                    "relevance": "not_relevant",
                    "evidence_role": "unclear",
                    "confidence": "high",
                    "rationale": "Does not address the assertion.",
                }
                for item in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    apply_passage_relevance_gate(artifact)

    assert captured_ids == [protected_id]


def test_relevance_gate_accepts_bounded_role_alias_from_provider(monkeypatch):
    artifact = _artifact()

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "relevant",
                    "role": "source_own_claim_or_finding",
                    "confidence": "high",
                    "rationale": "Addresses the citation unit.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json",
        fake_relevance,
    )
    assessed = apply_passage_relevance_gate(artifact)

    assert assessed.passage_relevance.status == "complete"
    assert {
        item.evidence_role for item in assessed.passage_relevance.assessments
    } == {"source_own_claim_or_finding"}


def test_mid_sentence_parenthetical_limits_relevance_to_pre_marker_assertion(
    monkeypatch,
):
    artifact = _artifact()
    text = (
        "Humans and MT systems make different kinds of errors (Smith, 2020), "
        "so instructors should receive additional training."
    )
    artifact = artifact.model_copy(
        update={
            "claim": artifact.claim.model_copy(
                update={
                    "text": text,
                    "citation_marker": "(Smith, 2020)",
                    "citation_marker_type": "parenthetical",
                    "passage_end": artifact.claim.passage_start + len(text),
                }
            )
        }
    )
    captured = {}

    def fake_relevance(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        return {
            "assessments": [
                {
                    "passage_id": passage["passage_id"],
                    "relevance": "not_relevant",
                    "evidence_role": "unclear",
                    "confidence": "high",
                    "rationale": "Does not address the attributed assertion.",
                }
                for passage in payload["passages"]
            ]
        }

    monkeypatch.setattr(
        "app.services.passage_relevance.chat_completion_json", fake_relevance
    )
    apply_passage_relevance_gate(artifact)

    assert captured["source_attributed_text"].endswith("(Smith, 2020)")
    assert "instructors" not in captured["source_attributed_text"]
    assert "instructors" in captured["complete_citation_unit"]
