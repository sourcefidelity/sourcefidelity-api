from dataclasses import replace
from datetime import datetime, timezone
import hashlib

import pytest
from pydantic import ValidationError

from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.evidence_package import build_evidence_package
from app.services.source_informed_interpretation import (
    SourceInformedProposal, SourceInformedRetrieval, build_source_informed_retrieval,
    project_source_informed_explanation, validate_source_informed_retrieval,
    ReadingExposure, prepare_source_informed_request, bind_source_informed_response,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation, ClaimEvidence, build_passage_evidence,
)


def fixture():
    now = datetime.now(timezone.utc)
    content = ("Messages about regional identity were edited for overseas audiences. "
               "The cultural identity of the shows was obscured by changing names. "
               "Other shows retained their original cultural identity.").encode()
    source = AuthorizedRepresentation(
        representation_id="r1", canonical_work_id="w1", content_object_id="o1",
        content_sha256=hashlib.sha256(content).hexdigest(), content=content,
        representation_kind="plain_text", media_type="text/plain", provenance="authorized_upload",
        scope_type="personal_owner", scope_id="owner1", identity_verdict="verified",
        identity_confidence=0.99, completeness_verdict="complete", text_quality="digital",
        edition_or_version=None, created_at=now, admitted_at=now)
    text = "Messages about regional identity were removed (Smith, 2020)."
    claim = ClaimEvidence(claim_id="c1", paper_version_id="p1", text=text,
                          reference_ids=["ref1"], citation_marker="(Smith, 2020)",
                          citation_marker_type="parenthetical", passage_start=10,
                          passage_end=10 + len(text))
    artifact = attach_verification_candidates(build_passage_evidence(source, claim=claim))
    candidate = next(c for c in artifact.verification_candidates.candidates
                     if c.attribution == "cited_source")
    proposal = SourceInformedProposal(
        candidate_id=candidate.candidate_id,
        interpreted_statement="The cultural identity of the shows was obscured.",
        problem_spans=[dict(local_start=0, local_end=32, text=text[:32])],
        motivating_passage_ids=[artifact.passages[0].passage_id],
        wording_difference="Messages about identity and cultural identity are different concepts.",
        unresolved_scope="The generalization to all shows remains unresolved.",
        origin="owner_after_source_exposure", provenance_sha256="a" * 64)
    return source, artifact, proposal


def test_opt_in_and_preservation_and_roundtrip():
    source, artifact, proposal = fixture()
    before = artifact.model_dump_json()
    package_before = build_evidence_package(artifact).model_dump_json()
    with pytest.raises(ValueError, match="not_enabled"):
        build_source_informed_retrieval(source, artifact, [proposal])
    record = build_source_informed_retrieval(source, artifact, [proposal], enabled=True)
    loaded = SourceInformedRetrieval.model_validate_json(record.model_dump_json())
    validate_source_informed_retrieval(loaded, source, artifact)
    assert loaded.searches[0].passages
    assert record.provider_calls == 0 and record.search_rounds == 1
    assert artifact.model_dump_json() == before
    assert build_evidence_package(artifact).model_dump_json() == package_before
    view = project_source_informed_explanation(loaded, source, artifact)
    assert view['accuracy_judgment'] is None
    assert view['original_statement'] == artifact.claim.text
    assert view['original_passage_ids'] == [p.passage_id for p in artifact.passages]
    assert 'retrieval only' in view['possible_readings'][0]['label']


def test_competing_readings_not_selected_and_bounded():
    source, artifact, proposal = fixture()
    other = proposal.model_copy(update={"interpreted_statement": "Political messages were removed."})
    result = build_source_informed_retrieval(source, artifact, [proposal, other], enabled=True)
    assert result.competing_interpretations and result.selected_interpretation_id is None
    assert sum(len(s.passages) for s in result.searches) <= 6
    with pytest.raises(ValueError, match="budget"):
        build_source_informed_retrieval(source, artifact, [proposal] * 3, enabled=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_source_informed_retrieval(source, artifact, [proposal] * 2, enabled=True)


@pytest.mark.parametrize('field,value', [('scope_id', 'other'), ('verification_run_id', 'other'),
                                      ('representation_id', 'other'), ('content', b'changed')])
def test_source_boundaries(field, value):
    source, artifact, proposal = fixture()
    with pytest.raises(ValueError, match="source_mismatch"):
        build_source_informed_retrieval(replace(source, **{field: value}), artifact, [proposal], enabled=True)


@pytest.mark.parametrize('field,value', [('candidate_id', 'missing'),
    ('motivating_passage_ids', ['missing']), ('problem_spans', [dict(local_start=0, local_end=3, text='bad')])])
def test_exact_proposal_bindings(field, value):
    source, artifact, proposal = fixture()
    bad = SourceInformedProposal.model_validate(proposal.model_dump() | {field: value})
    with pytest.raises(ValueError):
        build_source_informed_retrieval(source, artifact, [bad], enabled=True)


def test_no_hits_is_not_source_absence():
    source, artifact, proposal = fixture()
    proposal = proposal.model_copy(update={'interpreted_statement': 'Quasar nebula astrophysics.'})
    result = build_source_informed_retrieval(source, artifact, [proposal], enabled=True)
    assert not result.searches[0].passages
    assert not result.searches[0].source_absence_claim_permitted
    assert not result.searches[0].accuracy_judgment_allowed


def test_tampered_saved_output_and_extraction_fail_closed():
    source, artifact, proposal = fixture()
    result = build_source_informed_retrieval(source, artifact, [proposal], enabled=True)
    with pytest.raises(ValueError, match='tampered'):
        validate_source_informed_retrieval(result.model_copy(update={'original_student_text': 'edited'}), source, artifact)
    with pytest.raises(ValidationError):
        SourceInformedRetrieval.model_validate(result.model_dump() | {'original_accuracy_judgment': 'supports'})
    stale = artifact.model_copy(deep=True)
    stale.coverage.extracted_text_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='extraction_mismatch'):
        build_source_informed_retrieval(source, stale, [proposal], enabled=True)


def test_source_evidence_cannot_be_forged_or_repaired():
    source, artifact, proposal = fixture()
    changed = artifact.model_copy(deep=True)
    changed.passages[0].text = 'Invented source evidence.'
    with pytest.raises(ValueError):
        build_source_informed_retrieval(source, changed, [proposal], enabled=True)


def reading_inputs():
    source, artifact, proposal = fixture()
    p = artifact.passages[0]
    exposure = ReadingExposure(unit_id='u0', passage_id=p.passage_id,
                               start=0, end=len(p.text), text=p.text)
    request = prepare_source_informed_request(source, artifact, proposal.candidate_id,
                                             [exposure], enabled=True)
    response = dict(readings=[dict(statement=proposal.interpreted_statement,
        problem_token_ranges=[[0, 2]], evidence_ids=['u0'],
        wording_difference=proposal.wording_difference, unresolved_scope=proposal.unresolved_scope)])
    return source, artifact, exposure, request, response


def test_reading_generation_boundary_does_not_replace_original_or_judge():
    source, artifact, exposure, request, raw = reading_inputs()
    before = artifact.model_dump_json()
    with pytest.raises(ValueError, match='not_enabled'):
        prepare_source_informed_request(source, artifact, request['candidate_id'], [exposure])
    bound = bind_source_informed_response(source, artifact, request, raw, provenance_sha256='a'*64)
    assert bound['original_accuracy_judgment'] is None
    assert bound['selected_interpretation_id'] is None
    assert not bound['semantic_acceptance']
    proposals = [SourceInformedProposal.model_validate(p) for p in bound['proposals']]
    supplement = build_source_informed_retrieval(source, artifact, proposals, enabled=True)
    validate_source_informed_retrieval(supplement, source, artifact)
    assert artifact.model_dump_json() == before
    empty = bind_source_informed_response(source, artifact, request, {'readings':[]}, provenance_sha256='a'*64)
    assert empty['proposals'] == empty['selections'] == []


@pytest.mark.parametrize('defect', ['id', 'range', 'duplicate', 'judgment', 'scope', 'prompt', 'exposure'])
def test_reading_proposals_reject_unbound_or_stale_results(defect):
    source, artifact, exposure, request, raw = reading_inputs()
    if defect == 'id': raw['readings'][0]['evidence_ids'] = ['invented']
    if defect == 'range': raw['readings'][0]['problem_token_ranges'] = [[0,9999]]
    if defect == 'duplicate': raw['readings'][0]['evidence_ids'] = ['u0','u0']
    if defect == 'judgment': raw['support'] = True
    if defect == 'scope': source = replace(source, scope_id='other')
    if defect == 'prompt': request['prompt'] += 'changed'
    if defect == 'exposure': request['exposures'][0]['text'] = 'invented'
    with pytest.raises(ValueError):
        bind_source_informed_response(source, artifact, request, raw, provenance_sha256='a'*64)


def test_reading_input_budget_abstains_without_pruning():
    from app.services.llm_input_boundary import LLMInputBudgetExceeded
    source, artifact, exposure, request, raw = reading_inputs()
    with pytest.raises(LLMInputBudgetExceeded):
        prepare_source_informed_request(source, artifact, request['candidate_id'], [exposure],
                                       enabled=True, max_input_tokens=1)


def test_reading_alternatives_both_survive_without_becoming_facets():
    source, artifact, exposure, request, raw = reading_inputs()
    raw['readings'].append(raw['readings'][0] | {'statement':'Political messages were removed.'})
    result = bind_source_informed_response(source, artifact, request, raw, provenance_sha256='a'*64)
    assert len(result['proposals']) == len(result['selections']) == 2
    assert result['selected_interpretation_id'] is None


def test_reading_prompt_identifies_allowed_proposition_not_citation_tokens():
    import json,re
    source, artifact, exposure, request, raw = reading_inputs()
    data = json.loads(request['prompt'])
    tokens = list(re.finditer(r'\S+', artifact.claim.text))
    candidate = next(c for c in artifact.verification_candidates.candidates if c.candidate_id == request['candidate_id'])
    for lo,hi in data['allowed_problem_token_ranges']:
        assert any(s.local_start <= tokens[lo].start() < tokens[hi].end() <= s.local_end for s in candidate.segments)
    raw['readings'][0]['problem_token_ranges'] = [[0,len(tokens)-1]]
    with pytest.raises(ValueError, match='problem_span_not_bound'):
        bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)


def part_inputs():
    import json
    source, artifact, exposure, old, raw = reading_inputs()
    request = prepare_source_informed_request(source,artifact,old['candidate_id'],[exposure],
                                             enabled=True,part_linked=True)
    hi = json.loads(request['prompt'])['allowed_problem_token_ranges'][0][1]
    raw['readings'][0].update(problem_token_ranges=[[2,3]], inherited_token_ranges=[[0,1]],
                             unresolved_token_ranges=[[4,hi]], motivating_evidence_ids=['u0'],
                             selection_status='material_evidence_selected',causal_dependency='not_applicable')
    return source,artifact,request,raw


def test_part_readings_preserve_original_remainder_separately():
    source,artifact,request,raw=part_inputs()
    before=artifact.model_dump_json()
    result=bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)
    assert set(result['scope_bindings'][0]) == {'focal','inherited','unresolved','causal_dependency'}
    assert result['scope_bindings'][0]['unresolved']
    assert result['original_accuracy_judgment'] is None and not result['semantic_acceptance']
    assert artifact.model_dump_json()==before


@pytest.mark.parametrize('defect',['omission','overlap','marker','whole','bool','huge'])
def test_part_readings_cannot_hide_or_invent_original_scope(defect):
    source,artifact,request,raw=part_inputs()
    reading=raw['readings'][0]
    if defect=='omission':reading['unresolved_token_ranges']=[]
    if defect=='overlap':reading['inherited_token_ranges']=[[0,2]]
    if defect=='marker':reading['unresolved_token_ranges']=[[4,999]]
    if defect=='whole':
        import json
        reading.update(problem_token_ranges=json.loads(request['prompt'])['allowed_problem_token_ranges'],
                       inherited_token_ranges=[],unresolved_token_ranges=[])
    if defect=='bool':reading['inherited_token_ranges']=[[False,True]]
    if defect=='huge':reading['unresolved_token_ranges']=[[4,10**12]]
    with pytest.raises(ValueError):bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)


def test_part_mode_does_not_accept_whole_reading_response_shape():
    source,artifact,request,raw=part_inputs()
    del raw['readings'][0]['unresolved_token_ranges']
    with pytest.raises(ValueError):bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)


def test_part_reading_can_have_motivation_but_no_useful_input_evidence():
    source,artifact,request,raw=part_inputs()
    raw['readings'][0].update(evidence_ids=[],selection_status='no_material_evidence_in_supplied_units')
    bound=bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)
    assert not bound['selections'][0]['unit_ids']
    assert bound['proposals'][0]['motivating_passage_ids']
    assert bound['original_accuracy_judgment'] is None
    supplement=build_source_informed_retrieval(source,artifact,
        [SourceInformedProposal.model_validate(p) for p in bound['proposals']],enabled=True)
    assert not supplement.searches[0].source_absence_claim_permitted


def test_part_reading_selection_status_must_match_selected_evidence():
    source,artifact,request,raw=part_inputs()
    raw['readings'][0]['selection_status']='no_material_evidence_in_supplied_units'
    with pytest.raises(ValueError,match='inconsistent_reading_selection'):
        bind_source_informed_response(source,artifact,request,raw,provenance_sha256='a'*64)
