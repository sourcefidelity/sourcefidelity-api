"""Retrieval-first, evidence-conditioned citation-unit judgment regressions."""

from datetime import datetime, timezone
import hashlib
import json

from app.services.evidence_conditioned_judgment import (
    apply_evidence_conditioned_unit_judgment,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    CandidatePassageRelevanceEvidence,
    ClaimEvidence,
    PassageRelevanceGateEvidence,
    VerificationVerdict,
    build_passage_evidence,
)


def _artifact(
    text: str = "Licensing supports competition and reduces barriers (Smith, 2020).",
    *,
    granularity: str = "citation_unit",
):
    source_text = (
        "The study concludes that licensing supports competition and reduces "
        "barriers to entry."
    )
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
    claim = ClaimEvidence(
        claim_id="citation-unit-1",
        paper_version_id="paper-v1",
        text=text,
        granularity=granularity,
        atomization_method="test" if granularity == "atomic_claim" else "not_run",
        reference_ids=["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        extraction_confidence="high",
        passage_start=100,
        passage_end=100 + len(text),
    )
    artifact = build_passage_evidence(source, claim=claim, top_k=3)
    passage_id = artifact.passages[0].passage_id
    gate = PassageRelevanceGateEvidence(
        status="complete",
        method="test_relevance_gate",
        model_id="test-model",
        gate_version="test-v1",
        outcome="relevant_candidates_found",
        assessments=[
            CandidatePassageRelevanceEvidence(
                passage_id=passage_id,
                relevance="relevant",
                confidence="high",
                rationale="Test passage addresses the citation unit.",
            )
        ],
        relevant_passage_ids=[passage_id],
        decision_applied=False,
        processing_boundary="local",
    )
    return artifact.model_copy(update={"passage_relevance": gate})


def _token_id(payload, text, occurrence=0):
    matches = [row[0] for row in payload["unit_tokens"] if row[1] == text]
    return matches[occurrence]


def test_exact_shared_segments_create_inspectable_shadow_findings(monkeypatch):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id
    captured = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        captured.update(payload)
        captured["call_kwargs"] = kwargs
        licensing = _token_id(payload, "Licensing")
        supports = _token_id(payload, "supports")
        competition = _token_id(payload, "competition")
        reduces = _token_id(payload, "reduces")
        barriers = _token_id(payload, "barriers")
        return {
            "findings": [
                {
                    "segments": [{"start_id": licensing, "end_id": competition}],
                    "attribution": "cited_source",
                    "relationship": "supports",
                    "confidence": "high",
                    "passage_ids": [passage_id],
                    "rationale": "The passage states this relationship.",
                    "limitations": [],
                },
                {
                    "segments": [
                        {"start_id": licensing, "end_id": licensing},
                        {"start_id": reduces, "end_id": barriers},
                    ],
                    "attribution": "cited_source",
                    "relationship": "supports",
                    "confidence": "high",
                    "passage_ids": [passage_id],
                    "rationale": "The passage states this relationship.",
                    "limitations": [],
                },
            ],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_evidence_conditioned_unit_judgment(artifact)

    assert judged.unit_judgment.status == "complete"
    assert [finding.text for finding in judged.unit_judgment.findings] == [
        "Licensing supports competition",
        "Licensing reduces barriers",
    ]
    assert judged.unit_judgment.findings[1].segments[0].local_start == 0
    assert judged.unit_judgment.findings[1].segments[1].text == "reduces barriers"
    assert judged.unit_judgment.decision_applied is False
    assert judged.verdict is VerificationVerdict.INCONCLUSIVE
    assert captured["call_kwargs"]["disable_thinking"] is True
    assert all(
        row[2] is False
        for row in captured["unit_tokens"]
        if row[1] in {"Smith", "2020"}
    )


def test_uncovered_substantive_wording_is_persisted_as_unresolved(monkeypatch):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "Licensing"),
                    "end_id": _token_id(payload, "competition"),
                }],
                "attribution": "cited_source",
                "relationship": "supports",
                "confidence": "high",
                "passage_ids": [passage_id],
                "rationale": "Bounded passage support.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_evidence_conditioned_unit_judgment(artifact)

    assert judged.unit_judgment.status == "incomplete"
    assert [segment.text for segment in judged.unit_judgment.unresolved_segments] == [
        "and reduces barriers"
    ]
    assert "Smith" not in judged.unit_judgment.unresolved_segments[0].text


def test_single_limitation_string_is_normalized_without_relaxing_other_fields(
    monkeypatch,
):
    artifact = _artifact()
    passage_id = artifact.passages[0].passage_id

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "Licensing"),
                    "end_id": _token_id(payload, "barriers"),
                }],
                "attribution": "cited_source",
                "relationship": "supports",
                "confidence": "medium",
                "passage_ids": [passage_id],
                "rationale": "Bounded passage support.",
                "limitations": "The passage is less specific than the citation unit.",
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_evidence_conditioned_unit_judgment(artifact)

    assert judged.unit_judgment.findings[0].limitations == [
        "The passage is less specific than the citation unit."
    ]


def test_fabricated_token_or_passage_id_fails_the_whole_shadow_response(monkeypatch):
    artifact = _artifact()

    def fabricated_token(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "Licensing"),
                    "end_id": "model-invented-token",
                }],
                "attribution": "cited_source",
                "relationship": "supports",
                "confidence": "high",
                "passage_ids": [artifact.passages[0].passage_id],
                "rationale": "Invalid.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fabricated_token,
    )
    token_failure = apply_evidence_conditioned_unit_judgment(artifact)
    assert token_failure.unit_judgment.status == "not_assessed"
    assert token_failure.unit_judgment.method == (
        "unit_judgment_invalid_token_or_evidence_id"
    )

    def fabricated_passage(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "Licensing"),
                    "end_id": _token_id(payload, "competition"),
                }],
                "attribution": "cited_source",
                "relationship": "supports",
                "confidence": "high",
                "passage_ids": ["model-invented-passage"],
                "rationale": "Invalid.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fabricated_passage,
    )
    passage_failure = apply_evidence_conditioned_unit_judgment(artifact)
    assert passage_failure.unit_judgment.status == "not_assessed"


def test_student_analysis_must_be_not_assessed_without_source_passages(monkeypatch):
    text = "I consider the licensing argument weak (Smith, 2020)."
    artifact = _artifact(text)

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "I"),
                    "end_id": _token_id(payload, "weak"),
                }],
                "attribution": "student",
                "relationship": "not_assessed",
                "confidence": "high",
                "passage_ids": [],
                "rationale": "This is the student's evaluation.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_evidence_conditioned_unit_judgment(artifact)

    assert judged.unit_judgment.status == "complete"
    finding = judged.unit_judgment.findings[0]
    assert finding.attribution == "student"
    assert finding.relationship.value == "not_assessed"
    assert finding.passage_ids == []


def test_citation_marker_tokens_cannot_be_selected(monkeypatch):
    artifact = _artifact()

    def fake_call(system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        return {
            "findings": [{
                "segments": [{
                    "start_id": _token_id(payload, "Licensing"),
                    "end_id": _token_id(payload, "Smith"),
                }],
                "attribution": "cited_source",
                "relationship": "supports",
                "confidence": "high",
                "passage_ids": [artifact.passages[0].passage_id],
                "rationale": "Invalid marker selection.",
                "limitations": [],
            }],
            "unresolved_ranges": [],
        }

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fake_call,
    )
    judged = apply_evidence_conditioned_unit_judgment(artifact)

    assert judged.unit_judgment.status == "not_assessed"
    assert judged.unit_judgment.findings == []


def test_non_citation_unit_and_mismatched_passage_authorization_never_call_model(
    monkeypatch,
):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model should not run")

    monkeypatch.setattr(
        "app.services.evidence_conditioned_judgment.chat_completion_json",
        fail_if_called,
    )
    atomic = apply_evidence_conditioned_unit_judgment(
        _artifact(granularity="atomic_claim")
    )
    assert atomic.unit_judgment.method == "citation_unit_required"

    artifact = _artifact()
    bad_passage = artifact.passages[0].model_copy(
        update={"authorization_scope_id": "different-owner"}
    )
    artifact = artifact.model_copy(update={"passages": [bad_passage]})
    unauthorized = apply_evidence_conditioned_unit_judgment(artifact)
    assert unauthorized.unit_judgment.method == "no_authorized_candidate_passage"
    assert called is False
