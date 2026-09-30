import pytest
from app.services import facet_passage_selector as selector
from app.services.facet_evidence_plan import OperationFacetSelection, project_facet_evidence
from app.services.verification_evidence import ConfidenceLevel


def inputs():
    sentences = [selector.SelectorSentence(sentence_id=s, text=s + '.') for s in ('a','b','c')]
    facets = [selector.SelectorFacet(facet_id=f,text=f + '.',allowed_sentence_ids=['a','b','c']) for f in ('audience','development')]
    labels = {'audience': ['sufficient','sufficient','irrelevant'],
              'development': ['irrelevant','irrelevant','sufficient']}
    result = selector.FacetPassageSelectorResult(status='complete',processing_boundary='local',assessments=[
        selector.FacetSentenceUsefulness(facet_id=f.facet_id,sentence_id=s.sentence_id,usefulness=labels[f.facet_id][i],confidence=ConfidenceLevel.NONE)
        for f in facets for i,s in enumerate(sentences)])
    result.facet_sentence_sha256=selector.facet_sentence_fingerprint(facets,sentences)
    return facets,sentences,result


def test_new_facet_precedes_repeat_without_pruning_judgment():
    f,s,r=inputs()
    p=project_facet_evidence(f,s,r,display_limit=2)
    assert p.selected_sentence_ids==('a','c')
    assert p.additional_sentence_ids==('b',)
    assert p.judgment_sentence_ids==('a','b','c')
    assert not p.unresolved_facet_ids
    assert not r.decision_applied


def test_complete_coverage_does_not_fill_remaining_slot():
    f,s,r=inputs()
    p=project_facet_evidence(f,s,r,display_limit=3)
    assert p.selected_sentence_ids==('a','c')
    assert p.judgment_sentence_ids==('a','b','c')


def test_partial_combination_does_not_manufacture_sufficiency():
    f,s,r=inputs()
    for a in r.assessments:
        if a.facet_id=='audience' and a.sentence_id in ('a','b'):
            a.usefulness='partially_useful'
        if a.facet_id=='development':a.usefulness='uncertain'
    p=project_facet_evidence(f,s,r,display_limit=2)
    assert p.selected_sentence_ids==('a','b')
    assert p.unresolved_facet_ids==('audience','development')
    assert p.partially_addressed_facet_ids==('audience',)


def test_complementary_policy_does_not_fill_with_repeated_partial():
    f,s,r=inputs()
    for a in r.assessments:
        if a.facet_id=='audience' and a.sentence_id in ('a','b'):
            a.usefulness='partially_useful'
    p=project_facet_evidence(f,s,r,complementary_only=True)
    assert p.selected_sentence_ids==('c','a')
    assert p.unresolved_facet_ids==('audience',)
    assert p.judgment_sentence_ids==('a','b','c')
    assert p.projection_version.endswith('v2-complementary')


def test_complementary_policy_preserves_explicit_protected_context():
    f,s,r=inputs()
    for a in r.assessments:
        a.usefulness='partially_useful'
    p=project_facet_evidence(f,s,r,complementary_only=True,protected_sentence_ids=('a','b'))
    assert p.selected_sentence_ids==('a','b')
    assert len(p.unresolved_facet_ids)==2


def test_compact_proposition_map_exact_grid_and_no_retry(monkeypatch):
    f,s,_=inputs();calls=[]
    def provider(system,prompt,**kwargs):
        calls.append(kwargs)
        return {'rows':[{'facet_id':'f1','labels':['S','T','I']},
                        {'facet_id':'f2','labels':['I','I','P']}]}
    monkeypatch.setattr(selector,'chat_completion_json',provider)
    r=selector.classify_facet_sentence_usefulness('exact claim','complete unit',f,s,complete_propositions=True)
    assert r.status=='complete' and len(r.assessments)==6
    assert r.selector_version=='bounded-proposition-usefulness-v5'
    assert calls[0]['max_retries']==0
    assert project_facet_evidence(f,s,r,complementary_only=True).selected_sentence_ids==('a','c')


@pytest.mark.parametrize('rows',[
    [{'facet_id':'f1','labels':['P','P','P']}],
    [{'facet_id':'f1','labels':['P']},{'facet_id':'f2','labels':['I','I','I']}],
    [{'facet_id':'f1','labels':['P','P','P']},{'facet_id':'f1','labels':['I','I','I']}],
    [{'facet_id':'f1','labels':['S','invented','P']},{'facet_id':'f2','labels':['I','I','I']}],
])
def test_compact_map_rejects_invalid_or_incomplete_rows(monkeypatch,rows):
    f,s,_=inputs()
    monkeypatch.setattr(selector,'chat_completion_json',lambda *a,**k:{'rows':rows})
    r=selector.classify_facet_sentence_usefulness('claim','unit',f,s,complete_propositions=True)
    assert r.status=='not_assessed' and not r.assessments


def test_proposition_operation_reuses_mapping_without_another_model_call(monkeypatch):
    f,s,r=inputs();seen=[]
    def classify(*args,**kwargs):
        seen.append(kwargs)
        return r.model_copy(deep=True)
    monkeypatch.setattr(selector,'classify_facet_sentence_usefulness',classify)
    with OperationFacetSelection() as operation:
        kw=dict(permission_context='p',source_binding='s',enabled=True,complete_propositions=True)
        operation.select('claim','unit',f,s,**kw)
        operation.select('claim','unit',f,s,**kw)
        assert operation.reuse_hits==1 and len(seen)==1
        operation.select('claim','unit',f,s,**dict(kw,complete_propositions=False))
        assert len(seen)==2


def composition_inputs():
    from tests.unit.test_factual_facet_composition import (
        _artifact, _eligible_candidate_id, _composite_proposal, _faithful_preservation,
    )
    from app.services.factual_facet_composition import compose_factual_facets
    artifact=_artifact('Conservative values and censorship exist (Smith, 2020).')
    composition=compose_factual_facets(artifact,_eligible_candidate_id(artifact),
        proposal_provider=_composite_proposal,preservation_provider=_faithful_preservation)
    return artifact,composition,inputs()[1]


def test_source_blind_composition_adapter_preserves_scope_and_context():
    import json
    from app.services.facet_evidence_plan import composed_selector_facets
    artifact,composition,sentences=composition_inputs()
    facets=composed_selector_facets(artifact,composition,sentences)
    assert len(facets)==2
    for facet in facets:
        payload=json.loads(facet.text)
        assert payload['complete_citation_unit']==artifact.claim.text
        assert payload['scope_constraints'] and payload['structural_dependencies']
        assert facet.allowed_sentence_ids==['a','b','c']


@pytest.mark.parametrize('defect',['hash','partial','repair','finding','span','constraint','duplicate','prompt'])
def test_composition_adapter_abstains_on_unusable_foundation(defect):
    from app.services.facet_evidence_plan import composed_selector_facets
    artifact,composition,sentences=composition_inputs()
    if defect=='hash':composition.candidate_text_sha256='0'*64
    if defect=='partial':composition.coverage_status='partial'
    if defect=='repair':composition.interpretation_status='semantic_repair'
    if defect=='finding':composition.preservation_findings[0].accepted=False
    if defect=='span':composition.proposed_facets[0].segments[0].text='invented'
    if defect=='constraint':composition.proposed_facets[0].constraint_ids=[]
    if defect=='duplicate':composition.proposed_facets[1].facet_id=composition.proposed_facets[0].facet_id
    if defect=='prompt':composition.proposal_prompt_sha256=None
    with pytest.raises(ValueError):composed_selector_facets(artifact,composition,sentences)


def test_invalid_pair_grid_falls_back_without_evidence_loss():
    f,s,r=inputs();r.assessments.pop()
    p=project_facet_evidence(f,s,r,display_limit=2)
    assert p.selection_status=='not_assessed'
    assert p.selected_sentence_ids==('a','b')
    assert p.judgment_sentence_ids==('a','b','c')


def test_historical_v3_remains_readable_without_applying_unbound_map():
    facets, sentences, result = inputs()
    historical = result.model_dump(mode='json')
    historical['selector_version'] = 'bounded-facet-sentence-usefulness-v3'
    historical.pop('facet_sentence_sha256')
    restored = selector.FacetPassageSelectorResult.model_validate(historical)
    plan = project_facet_evidence(facets, sentences, restored, display_limit=2)
    assert plan.selection_status == 'not_assessed'
    assert plan.judgment_sentence_ids == ('a', 'b', 'c')


def test_protected_context_can_exceed_compact_limit():
    f,s,r=inputs()
    assert project_facet_evidence(f,s,r,display_limit=1,protected_sentence_ids=('b','c')).selected_sentence_ids==('b','c')
    with pytest.raises(ValueError):project_facet_evidence(f,s,r,protected_sentence_ids=('invented',))


def test_identical_operation_reuses_call_and_copies_result(monkeypatch):
    f,s,r=inputs();calls=[]
    monkeypatch.setattr(selector,'classify_facet_sentence_usefulness',lambda *args:(calls.append(1) or r.model_copy(deep=True)))
    memo=OperationFacetSelection()
    kw=dict(permission_context='current-permissions',source_binding='source-extraction-spans',enabled=True)
    a=memo.select('candidate','citation',f,s,**kw)
    a.assessments.clear()
    b=memo.select('candidate','citation',f,s,**kw)
    assert b.assessments and len(calls)==1 and memo.reuse_hits==1
    memo.clear();memo.select('candidate','citation',f,s,**kw)
    assert len(calls)==2


@pytest.mark.parametrize('change',['candidate','source','permission','sentence','facet','model','budget','endpoint'])
def test_changed_inputs_never_reuse(monkeypatch,change):
    f,s,r=inputs();calls=[]
    monkeypatch.setattr(selector,'classify_facet_sentence_usefulness',lambda *args:(calls.append(1) or r.model_copy(deep=True)))
    memo=OperationFacetSelection();kw=dict(permission_context='p',source_binding='s',enabled=True)
    memo.select('candidate','citation',f,s,**kw)
    candidate='candidate'
    if change=='candidate':candidate='changed'
    if change=='source':kw['source_binding']='new'
    if change=='permission':kw['permission_context']='new'
    if change=='sentence':s[0].text='changed'
    if change=='facet':f[0].text='changed'
    if change=='model':monkeypatch.setattr(selector.settings,'LLM_MODEL','changed')
    if change=='endpoint':monkeypatch.setattr(selector.settings,'LLM_BASE_URL','http://localhost:9999')
    if change=='budget':monkeypatch.setattr(selector.settings,'VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS',1)
    memo.select(candidate,'citation',f,s,**kw)
    assert len(calls)==2


def test_disabled_never_calls_and_failure_is_not_cached(monkeypatch):
    f,s,r=inputs();r.status='not_assessed';r.failure_code='provider_or_schema_failure'
    calls=[]
    monkeypatch.setattr(selector,'classify_facet_sentence_usefulness',lambda *args:(calls.append(1) or r))
    memo=OperationFacetSelection();kw=dict(permission_context='p',source_binding='s')
    with pytest.raises(ValueError):memo.select('c','c',f,s,**kw)
    assert not calls
    for _ in range(2):memo.select('c','c',f,s,enabled=True,**kw)
    assert len(calls)==2 and memo.reuse_hits==0


def test_operation_exit_clears_cached_state_on_error(monkeypatch):
    f,s,r=inputs()
    monkeypatch.setattr(selector,'classify_facet_sentence_usefulness',lambda *args:r)
    memo=OperationFacetSelection()
    with pytest.raises(RuntimeError):
        with memo:
            memo.select('c','c',f,s,permission_context='p',source_binding='s',enabled=True)
            assert memo._result is not None
            raise RuntimeError('operation failed')
    assert memo._key is None and memo._result is None


def test_same_ids_with_changed_text_or_missing_binding_fail_closed():
    f,s,r=inputs()
    s[0].text='Different source sentence.'
    assert project_facet_evidence(f,s,r).selection_status=='not_assessed'
    f,s,r=inputs();r.facet_sentence_sha256=''
    assert project_facet_evidence(f,s,r).selection_status=='not_assessed'


@pytest.mark.parametrize('defect', ['binding', 'missing_pair', 'duplicate_pair'])
def test_complete_but_invalid_result_is_not_reused(monkeypatch, defect):
    facets, sentences, result = inputs()
    if defect == 'binding':
        result.facet_sentence_sha256 = ''
    elif defect == 'missing_pair':
        result.assessments.pop()
    else:
        result.assessments[-1] = result.assessments[0].model_copy()
    calls = []
    monkeypatch.setattr(selector, 'classify_facet_sentence_usefulness',
                        lambda *args: (calls.append(1) or result))
    memo = OperationFacetSelection()
    for _ in range(2):
        memo.select('candidate', 'citation', facets, sentences,
                    permission_context='p', source_binding='s', enabled=True)
    assert len(calls) == 2
    assert memo.reuse_hits == 0
    assert memo._result is None


def test_missing_permission_boundary_clears_previous_result(monkeypatch):
    facets, sentences, result = inputs()
    monkeypatch.setattr(selector, 'classify_facet_sentence_usefulness', lambda *args: result)
    memo = OperationFacetSelection()
    memo.select('candidate', 'citation', facets, sentences,
                permission_context='p', source_binding='s', enabled=True)
    with pytest.raises(ValueError, match='missing_reuse_boundary'):
        memo.select('candidate', 'citation', facets, sentences,
                    permission_context='', source_binding='s', enabled=True)
    assert memo._result is None and memo._key is None
