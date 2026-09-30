"""Joint selection contract checks; every model boundary is mocked."""
import hashlib
import json

import pytest

from app.services import joint_evidence_selection as joint
from app.services.verification_evidence import (
    CandidatePassageRelevanceEvidence, PassageRelevanceGateEvidence,
    JointEvidenceSelection,
)
from tests.unit.test_passage_relevance import _artifact


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_source_sentence_labels_survive_identifier_redaction():
    text = 'Contact person@example.edu for records. The bridge closed after inspection.'
    labelled = joint._labelled_source(text)
    assert 'person@example.edu' not in labelled
    assert '[s001] The bridge closed after inspection.' in labelled


def test_source_labels_do_not_bypass_cover_name_redaction():
    labelled = joint._labelled_source('Name: Example Student\nThe bridge closed.')
    assert 'Example Student' not in labelled
    assert '[s000]' in labelled


def test_exact_text_cluster_preserves_all_ids_and_roles():
    from app.services.verification_evidence import JointEvidenceCandidate
    text = 'The bridge closed after inspection.'
    candidates = [JointEvidenceCandidate(passage_id=pid, source_span=text,
                  source_span_sha256=_sha(text)) for pid in ('first', 'second')]
    assessments = {'first': {'relevance': 'relevant', 'evidence_role': 'source_own_claim_or_finding'},
                   'second': {'relevance': 'partially_relevant', 'evidence_role': 'unclear'}}
    rows = joint._comparison_rows(candidates, assessments)
    assert [r['passage_id'] for r in rows] == ['first', 'second']
    assert rows[1]['identical_text_of'] == 'first' and 'text' not in rows[1]
    assert rows[1]['evidence_role'] == 'unclear'
    assert rows[0]['text'] == '[s000] ' + text


def test_clustering_never_merges_redaction_collisions():
    from app.services.verification_evidence import JointEvidenceCandidate
    texts = ['Contact a@example.org.', 'Contact b@example.org.']
    assert joint._labelled_source(texts[0]) == joint._labelled_source(texts[1])
    candidates = [JointEvidenceCandidate(passage_id=str(i), source_span=t,
                  source_span_sha256=_sha(t)) for i, t in enumerate(texts)]
    rows = joint._comparison_rows(candidates, {str(i): {'relevance': 'relevant'} for i in range(2)})
    assert all('text' in row and 'identical_text_of' not in row for row in rows)


def test_duplicate_alias_selects_original_id_and_round_trips(case, monkeypatch):
    artifact, _ = case
    artifact.passages[1].text = artifact.passages[0].text
    a = artifact.passage_relevance.assessments[1]
    a.assessed_text_sha256 = _sha(artifact.passages[1].text)
    a.assessed_text_offset_end = len(artifact.passages[1].text)
    pid = artifact.passages[1].passage_id
    def response(system, prompt, **kwargs):
        assert json.loads(prompt)['candidates'][1]['identical_text_of'] == artifact.passages[0].passage_id
        return {'selected': [{'passage_id': pid, 'reason': 'primary',
                'claim_token_ranges': [[0, 2]], 'source_sentence_ids': ['s000']}]}
    monkeypatch.setattr(joint, 'chat_completion_json', response)
    result = joint.select_joint_evidence(artifact)
    assert result.status == 'complete' and result.selected[0].passage_id == pid
    assert joint.validate_joint_selection(artifact, result)


def test_selected_extract_cannot_expand_into_a_long_context():
    from app.services.verification_evidence import JointEvidenceCandidate
    text = 'The evidence explains ' + 'important details ' * 100 + '.'
    candidate = JointEvidenceCandidate(passage_id='p', source_span=text, source_span_sha256=_sha(text))
    with pytest.raises(ValueError):
        joint._selected_span(candidate, ['s000'])


def test_distinct_aspect_cannot_be_contained_in_primary_claim(case):
    artifact, _ = case
    result = joint.select_joint_evidence(artifact)
    target = joint._source_attributed_relevance_text(artifact.claim)
    from app.services.verification_evidence import JointEvidenceSelectedPassage
    result.selected[0].claim_spans = [target]
    other = result.candidates[1]
    span = joint._selected_span(other, ['s000'])
    result.selected.append(JointEvidenceSelectedPassage(passage_id=other.passage_id,
        reason='distinct_aspect', claim_spans=['supports competition'], source_sentence_ids=['s000'],
        source_span=span, source_span_sha256=_sha(span)))
    assert not joint._valid_selected(result, target, set())


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(joint.settings, "VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS", 4000)
    monkeypatch.setattr(joint.settings, "VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS", 1600)
    artifact = _artifact()
    assessments = [CandidatePassageRelevanceEvidence(
        passage_id=p.passage_id, relevance="partially_relevant", confidence="high",
        assessed_text_sha256=_sha(p.text), assessed_text_offset_end=len(p.text),
        evidence_role="source_own_claim_or_finding") for p in artifact.passages]
    artifact = artifact.model_copy(update={"passage_relevance": PassageRelevanceGateEvidence(
        status="complete", assessments=assessments)})
    calls = []

    def reply(system, prompt, **kwargs):
        calls.append((system, json.loads(prompt), kwargs))
        pid = json.loads(prompt)["candidates"][0]["passage_id"]
        return {"selected": [{"passage_id": pid, "reason": "primary",
                              "claim_token_ranges": [[0, 2]], "source_sentence_ids": ["s000"]}]}

    monkeypatch.setattr(joint, "chat_completion_json", reply)
    return artifact, calls


def test_all_candidates_single_call_and_round_trip(case):
    artifact, calls = case
    original = artifact.model_dump_json()
    result = joint.select_joint_evidence(artifact, source_title="Submitted title")
    assert result.status == "complete" and len(calls) == result.call_count == 1
    assert len(result.candidates) == len(artifact.passages)
    assert calls[0][2] == dict(model=joint.settings.LLM_MODEL, temperature=0.0,
                              max_tokens=1600, max_retries=0, disable_thinking=True)
    assert calls[0][1]["submitted_source_title_orientation_only"] == "Submitted title"
    assert joint.validate_joint_selection(artifact, JointEvidenceSelection.model_validate_json(result.model_dump_json()))
    assert artifact.model_dump_json() == original
    assert joint.validate_joint_projection(result.model_dump(), **joint._inputs(artifact))


def test_not_run_and_incomplete_do_not_dispatch(case):
    artifact, calls = case
    for status in ("not_run", "not_assessed"):
        artifact.passage_relevance.status = status
        result = joint.select_joint_evidence(artifact)
        assert result.status == status and result.call_count == 0
    assert not calls


def test_overflow_abstains_without_subset(case, monkeypatch):
    artifact, calls = case
    monkeypatch.setattr(joint.settings, "VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS", 1)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "not_assessed" and result.limitation_codes == ["prompt_budget_exceeded"]
    assert len(result.candidates) == len(artifact.passages) and not calls


@pytest.mark.parametrize("field", ["source_title", "input_fingerprint", "model_input_sha256", "target_text_sha256"])
def test_receipt_hash_tamper(case, field):
    artifact, _ = case
    result = joint.select_joint_evidence(artifact)
    setattr(result, field, "tampered")
    assert not joint.validate_joint_selection(artifact, result)


@pytest.mark.parametrize("part", ["claim", "source", "coverage", "assessment", "passage", "member"])
def test_current_input_tamper(case, part):
    artifact, _ = case
    result = joint.select_joint_evidence(artifact)
    if part == "claim": artifact.claim.text += " Changed."
    if part == "source": artifact.source_identity.content_sha256 = "a" * 64
    if part == "coverage": artifact.coverage.extracted_text_sha256 = "b" * 64
    if part == "assessment": artifact.passage_relevance.assessments[0].rationale = "Changed"
    if part == "passage": artifact.passages[-1].text += " Changed."
    if part == "member": artifact.claim.reference_ids = ["different-member"]
    assert not joint.validate_joint_selection(artifact, result)


@pytest.mark.parametrize("change", ["unknown_id", "fragment", "hash", "duplicate", "invalid_sentence", "support", "invalid_range"])
def test_invalid_input_or_output_abstains(case, monkeypatch, change):
    artifact, calls = case
    if change == "fragment":
        p = artifact.passages[0]
        p.text = "an incomplete fragment"
        a = artifact.passage_relevance.assessments[0]
        a.assessed_text_offset_end = len(p.text)
        a.assessed_text_sha256 = _sha(p.text)
    elif change == "hash": artifact.passage_relevance.assessments[0].assessed_text_sha256 = "f" * 64
    else:
        def response(*args, **kwargs):
            choice = dict(passage_id=artifact.passages[0].passage_id, reason="primary",
                          claim_token_ranges=[], source_sentence_ids=["s000"])
            if change == "unknown_id": choice["passage_id"] = "invented"
            if change == "invalid_sentence": choice["source_sentence_ids"] = ["s999"]
            if change == "invalid_range": choice["claim_token_ranges"] = [[0, 999]]
            if change == "support": choice["support"] = True
            return {"selected": [choice, choice] if change == "duplicate" else [choice]}
        monkeypatch.setattr(joint, "chat_completion_json", response)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "not_assessed" and not result.selected


def test_provider_failure_sanitized(case, monkeypatch):
    artifact, _ = case
    def fail(*args, **kwargs): raise RuntimeError("PRIVATE provider body")
    monkeypatch.setattr(joint, "chat_completion_json", fail)
    result = joint.select_joint_evidence(artifact)
    assert result.call_count == 1 and "PRIVATE" not in result.model_dump_json()


def test_protected_priority_and_unassessed_protected_abstain(case, monkeypatch):
    artifact, _ = case
    artifact.quotation_check.evidence_passage_ids = [artifact.passages[-1].passage_id]
    assert joint.select_joint_evidence(artifact).status == "not_assessed"
    artifact.quotation_check.evidence_passage_ids = ["missing-protected"]
    result = joint.select_joint_evidence(artifact)
    assert result.status == "not_assessed" and result.call_count == 0


def test_whole_window_not_local_observation_and_sentence_selection(case, monkeypatch):
    artifact, _ = case
    p = artifact.passages[0]
    p.text = "The framework defines licensing. Its application changes market access."
    a = artifact.passage_relevance.assessments[0]
    a.assessed_text_offset_end, a.assessed_text_sha256 = len(p.text), _sha(p.text)
    from app.services.verification_evidence import PassageDisplayObservation
    a.display_observation = PassageDisplayObservation(basis="general_framework",
        claim_spans=["Licensing"], source_span="The framework defines licensing.")
    def response(system, prompt, **kwargs):
        row = json.loads(prompt)["candidates"][0]
        assert row["text"] == "[s000] The framework defines licensing. [s001] Its application changes market access."
        assert row["nonselectable_sentence_ids"] == []
        return {"selected": [dict(passage_id=p.passage_id, reason="primary",
                                  source_sentence_ids=["s001"], claim_token_ranges=[[0, 2]])]}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete"
    assert result.selected[0].source_span == "Its application changes market access."
    assert joint.validate_joint_selection(artifact, result)


def test_empty_selection_and_no_eligible_candidates(case, monkeypatch):
    artifact, calls = case
    monkeypatch.setattr(joint, "chat_completion_json", lambda *a, **kw: {"selected": []})
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete" and result.selected == []
    for a in artifact.passage_relevance.assessments: a.relevance = "not_relevant"
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete" and result.call_count == 0
    assert joint.validate_joint_selection(artifact, result)


def test_fragment_window_retained_complete_inner_sentence_selectable(case, monkeypatch):
    artifact, _ = case
    p = artifact.passages[0]
    p.text = "a cut opening. Licensing changes access. An unfinished ending"
    a = artifact.passage_relevance.assessments[0]
    a.assessed_text_offset_end, a.assessed_text_sha256 = len(p.text), _sha(p.text)
    def response(system, prompt, **kwargs):
        row = json.loads(prompt)["candidates"][0]
        assert "a cut opening." in row["text"] and "An unfinished ending" in row["text"]
        assert row["nonselectable_sentence_ids"] == ["s000", "s002"]
        return {"selected": [dict(passage_id=p.passage_id, reason="primary",
                                  source_sentence_ids=["s001"], claim_token_ranges=[[0, 2]])]}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete" and result.selected[0].source_span == "Licensing changes access."


@pytest.mark.parametrize("duplicate", ["claim", "source"])
def test_duplicate_additional_extracts_abstain(case, monkeypatch, duplicate):
    artifact, _ = case
    if duplicate == "source":
        artifact.passages[1].text = artifact.passages[0].text
        a = artifact.passage_relevance.assessments[1]
        a.assessed_text_sha256 = _sha(artifact.passages[1].text)
        a.assessed_text_offset_end = len(artifact.passages[1].text)
    def response(*args, **kwargs):
        return {"selected": [dict(passage_id=p.passage_id,
            reason="primary" if i == 0 else "distinct_aspect",
            source_sentence_ids=["s000"], claim_token_ranges=[[0, 2]] if i == 0 or duplicate == "claim" else [[3, 4]])
            for i, p in enumerate(artifact.passages[:2])]}
    monkeypatch.setattr(joint, "chat_completion_json", response)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "not_assessed" and not result.selected


def test_redundant_protected_supplement_cannot_be_dropped(case, monkeypatch):
    artifact, _ = case
    ids = [p.passage_id for p in artifact.passages[:2]]
    artifact.quotation_check.evidence_passage_ids = ids
    monkeypatch.setattr(joint, 'chat_completion_json', lambda *a, **kw: {'selected': [
        dict(passage_id=pid, reason='primary' if i == 0 else 'distinct_aspect',
             source_sentence_ids=['s000'], claim_token_ranges=[[0, 2]]) for i, pid in enumerate(ids)]})
    result = joint.select_joint_evidence(artifact)
    assert result.status == 'not_assessed' and not result.selected


def test_malformed_redundant_supplement_rejects_whole_advice(case, monkeypatch):
    artifact, _ = case
    monkeypatch.setattr(joint, 'chat_completion_json', lambda *a, **kw: {'selected': [
        dict(passage_id=p.passage_id, reason='primary' if i == 0 else 'distinct_aspect',
             source_sentence_ids=['s000' if i == 0 else 's999'], claim_token_ranges=[[0, 2]])
        for i, p in enumerate(artifact.passages[:2])]})
    result = joint.select_joint_evidence(artifact)
    assert result.status == 'not_assessed' and not result.selected


def test_redacted_claim_span_cannot_be_invented(case, monkeypatch):
    artifact, _ = case
    artifact.claim.text = "reviewer@example.edu describes licensing."
    result = joint.select_joint_evidence(artifact)
    assert result.status == "not_assessed"
    assert result.direct_identifier_redactions["email"] == 2


def test_exact_primary_obligation_target_not_whole_citation(case, monkeypatch):
    artifact, _ = case
    from app.services.verification_evidence import (
        CitationSourceBinding, EvidenceObligation, EvidenceObligationSet,
        ObligationPassageRelevanceEvidence,
    )
    marker = artifact.claim.citation_marker
    start = artifact.claim.text.index(marker)
    artifact.source_binding = CitationSourceBinding(reference_id="reference-1", cited_author_label="Smith",
        marker_text=marker, marker_local_start=start, marker_local_end=start + len(marker))
    target = "Licensing supports competition"
    obligation = EvidenceObligation(obligation_id="primary", obligation_type="exact_factual_assertion",
        reference_id="reference-1", target_text=target, target_text_sha256=_sha(target),
        original_text_sha256=_sha(target), derivation_method="exact_source_attributed_text",
        aggregate_scope="not_aggregate", accuracy_judgment_allowed=True, coverage_judgment_allowed=False)
    artifact.evidence_obligations = EvidenceObligationSet(status="complete", obligations=[obligation])
    artifact.passage_relevance.obligation_findings = [ObligationPassageRelevanceEvidence(
        obligation_id="primary", obligation_type="exact_factual_assertion", status="complete",
        method="fixture", gate_version="fixture", outcome="relevant_candidates_found",
        assessments=artifact.passage_relevance.assessments)]
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete" and result.target_text_sha256 == _sha(target)
    assert joint.validate_joint_selection(artifact, result)
    obligation.reference_id = "other-member"
    assert joint.select_joint_evidence(artifact).call_count == 0
    obligation.reference_id = "reference-1"
    obligation.accuracy_judgment_allowed = False
    obligation.obligation_type = "coverage_only_semantic_repair"
    assert joint.select_joint_evidence(artifact).status == "not_assessed"


def test_context_is_orientation_only_bound_and_capped(case):
    artifact, calls = case
    from app.services.verification_evidence import ClaimContextSegment
    artifact.claim.antecedent_context = [ClaimContextSegment(context_index=i % 2,
        distance_before=1, text=f"Orientation number {i}.", paper_start=0, paper_end=21) for i in range(3)]
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete"
    assert len(calls[0][1]["antecedent_orientation_only"]) == 2
    assert "Orientation" not in calls[0][1]["source_attributed_text"]
    artifact.claim.antecedent_context[-1].text = "Changed omitted context."
    assert not joint.validate_joint_selection(artifact, result)


def test_projection_rejects_missing_and_extra_original_passages(case):
    artifact, _ = case
    result = joint.select_joint_evidence(artifact)
    inputs = joint._inputs(artifact)
    inputs["passages"]["synthetic-continuation"] = "An extra text."
    assert not joint.validate_joint_projection(result, **inputs)
    del inputs["passages"]["synthetic-continuation"]
    inputs["passages"].pop(next(iter(inputs["passages"])))
    assert not joint.validate_joint_projection(result, **inputs)


def test_budget_substitutes_bound_observations_without_dropping_ids(case, monkeypatch):
    artifact, calls = case
    from app.services.verification_evidence import PassageDisplayObservation
    for i, (p, a) in enumerate(zip(artifact.passages, artifact.passage_relevance.assessments)):
        short = f"Licensing changes access in region {i}."
        p.text = short + " " + "Background " + "detail " * 800 + "ends."
        a.assessed_text_offset_end, a.assessed_text_sha256 = len(p.text), _sha(p.text)
        a.display_observation = PassageDisplayObservation(basis="direct_attribution",
            claim_spans=["Licensing"], source_span=short)
    result = joint.select_joint_evidence(artifact)
    assert result.status == "complete" and len(calls) == 1
    assert [c.passage_id for c in result.candidates] == [p.passage_id for p in artifact.passages]
    assert any(len(c.source_span) < len(p.text) for c, p in zip(result.candidates, artifact.passages))
    assert joint.validate_joint_selection(artifact, result)
    # Historical plan validity is independent of today's configured ceiling.
    monkeypatch.setattr(joint.settings, "VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS", 1)
    assert joint.validate_joint_selection(artifact, result)
    assert joint.select_joint_evidence(artifact).limitation_codes == ["prompt_budget_exceeded"]


def test_fitting_whole_input_does_not_use_observation(case):
    artifact, _ = case
    from app.services.verification_evidence import PassageDisplayObservation
    p, a = artifact.passages[0], artifact.passage_relevance.assessments[0]
    original = p.text
    p.text += " This additional sentence supplies context."
    a.assessed_text_offset_end, a.assessed_text_sha256 = len(p.text), _sha(p.text)
    a.display_observation = PassageDisplayObservation(basis="direct_attribution",
        claim_spans=["Licensing"], source_span=original)
    result = joint.select_joint_evidence(artifact)
    assert result.candidates[0].source_span == p.text
