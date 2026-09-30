import hashlib
import json

import pytest

from app.services import facet_passage_selector as selector
from app.services.facet_evidence_plan import project_inspection_evidence
from app.services.factual_facet_composition import compose_factual_facets
from tests.unit.test_factual_facet_composition import _artifact
from tests.unit.test_facet_evidence_plan import composition_inputs
from app.services.facet_evidence_plan import group_provisional_compositions


def mapped(monkeypatch, response=None):
    claim='Growth increased audiences and created new markets.'
    facets=[selector.SelectorFacet(facet_id='f',text=claim,allowed_sentence_ids=['a','b','c','d'],claim_spans=[(0,len(claim))])]
    units=[selector.SelectorSentence(sentence_id=s,text=s+'.') for s in ('a','b','c','d')]
    answer=response or {'rows':[{'facet_id':'f1','labels':['M','M','C','M']}], 'details':[
        {'facet_id':'f1','sentence_id':'s1','ranges':[[0,2]],'context_for':[]},
        {'facet_id':'f1','sentence_id':'s2','ranges':[[4,6]],'context_for':[]},
        {'facet_id':'f1','sentence_id':'s3','ranges':[],'context_for':['s1']},
        {'facet_id':'f1','sentence_id':'s4','ranges':[[0,2]],'context_for':[]}]}
    def provider(system,prompt,**kwargs):
        assert 'JSON' in system and '[t0]' in json.loads(prompt)['candidate_as_written']
        assert kwargs['max_retries']==0
        return answer
    monkeypatch.setattr(selector,'chat_completion_json',provider)
    result=selector.classify_facet_sentence_usefulness(claim,claim,facets,units,inspection_parts=True)
    return claim,facets,units,result


def test_complementary_parts_and_needed_context_do_not_become_sufficient(monkeypatch):
    c,f,u,r=mapped(monkeypatch)
    assert r.status=='complete'
    plan=project_inspection_evidence(c,f,u,r)
    assert plan.selected_sentence_ids==('a','c','b')
    assert plan.unresolved_facet_ids==('f',)
    assert plan.judgment_sentence_ids==('a','b','c','d')
    assert not r.decision_applied


@pytest.mark.parametrize('defect',['hash','span','context','detail','role'])
def test_stale_or_tampered_inspection_cannot_drive_selection(monkeypatch,defect):
    c,f,u,r=mapped(monkeypatch)
    if defect=='hash':r.inspection_claim_sha256='0'*64
    if defect=='span':r.inspection_details[0].claim_spans=[(0,10000)]
    if defect=='context':r.inspection_details[2].context_for=['unknown']
    if defect=='detail':r.inspection_details.pop()
    if defect=='role':r.inspection_details[0].role='W'
    p=project_inspection_evidence(c,f,u,r)
    assert p.selection_status=='not_assessed' and len(p.judgment_sentence_ids)==4


def test_protected_inspection_kept_even_beyond_compact_cap(monkeypatch):
    c,f,u,r=mapped(monkeypatch)
    p=project_inspection_evidence(c,f,u,r,display_limit=1,protected_sentence_ids=('c','d'))
    assert p.selected_sentence_ids==('c','d')


def test_context_cycles_or_missing_material_are_invalid(monkeypatch):
    response={'rows':[{'facet_id':'f1','labels':['C','C','N','N']}],'details':[
        {'facet_id':'f1','sentence_id':'s1','ranges':[],'context_for':['s2']},
        {'facet_id':'f1','sentence_id':'s2','ranges':[],'context_for':['s1']}]}
    assert mapped(monkeypatch,response)[3].status=='not_assessed'


def test_whole_unit_composition_is_opt_in_source_blind_and_nonmutating():
    a=_artifact('Growth increased audiences and created new markets (Smith, 2020).')
    guard=next(c for c in a.verification_candidates.candidates if c.role=='whole_unit_guard')
    before=a.model_dump(mode='json');seen=[]
    def proposal(system,prompt):
        payload=json.loads(prompt)
        assert 'PRIVATE SOURCE SENTENCE' not in prompt
        seen.append(payload)
        return {'candidate_id':guard.candidate_id,'status':'not_assessed','facets':[],
                'shared_constraints':[],'structural_edges':[],
                'uncovered_ranges':[{'start_id':payload['candidate_tokens'][0]['token_id'],
                    'end_id':payload['candidate_tokens'][-1]['token_id'],'reason':'unresolved_scope'}]}
    kw=dict(proposal_provider=proposal,preservation_provider=lambda *_:pytest.fail('unexpected preservation'))
    assert compose_factual_facets(a,guard.candidate_id,**kw).failure_code=='candidate_not_eligible'
    assert not seen
    r=compose_factual_facets(a,guard.candidate_id,whole_unit_development=True,**kw)
    assert seen and r.candidate_text_sha256==hashlib.sha256(guard.text.encode()).hexdigest()
    assert a.model_dump(mode='json')==before


def test_grouping_preserves_minority_scope_and_rejected_propositions():
    _,r,_=composition_inputs();minor=r.model_copy(deep=True)
    minor.scope_constraints[0].scope='unresolved'
    minor.accepted_facet_ids=[]
    g=group_provisional_compositions([('a',r),('b',r),('c',minor)])
    assert len(g['groups'])==4 and g['proposed_count']==g['retained_count']==6
    assert g['canonical_reading'] is None
    assert sum(not m['preserved'] for x in g['groups'] for m in x['members'])==2
    minor.candidate_text_sha256='0'*64
    with pytest.raises(ValueError):group_provisional_compositions([('a',r),('c',minor)])


def test_internal_dependency_is_not_a_new_fragment_proposition():
    from app.services.factual_facet_composition import _ProposalResponse
    raw={'candidate_id':'c','status':'complete','facets':[{
        'proposal_key':'p001','proposition_form':'complete_factual_proposition',
        'checking_gloss':'Growth increased audiences and markets.',
        'gloss_inherits_from_complete_unit':False,'ranges':[{'start_id':'t000','end_id':'t004'}],
        'constraint_keys':[]}], 'shared_constraints':[], 'uncovered_ranges':[],
        'structural_edges':[{'edge_key':'e001','kind':'coordination',
            'ranges':[{'start_id':'t003','end_id':'t003'}],'connects':['p001']}]}
    result=_ProposalResponse.model_validate(raw)
    assert len(result.facets)==1 and result.structural_edges[0].connects==['p001']
    raw['structural_edges'][0]['connects']=['unknown']
    with pytest.raises(ValueError):_ProposalResponse.model_validate(raw)


def test_inspection_compaction_is_lossless_and_keeps_masked_collisions_separate():
    fs=[selector.SelectorFacet(facet_id=str(i),text=json.dumps({'context':'same','part':str(i)}),allowed_sentence_ids=['s']) for i in range(2)]
    rows=[{'facet_id':f.facet_id,'text':f.text,'sentence_ids':['s']} for f in fs]
    compact,shared=selector._compact_inspection_facets(rows,fs)
    assert shared=={'context':'same'}
    assert [{**shared,**r['data']} for r in compact]==[json.loads(f.text) for f in fs]
    rows[1]['text']=rows[0]['text']
    compact,shared=selector._compact_inspection_facets(rows,fs)
    assert 'part' not in shared and len(compact)==2


def test_inspection_rejects_neighboring_proposition_range(monkeypatch):
    c,f,u,_=mapped(monkeypatch)
    f[0].claim_spans=[(0,c.index(' and'))]
    result=selector.classify_facet_sentence_usefulness(c,c,f,u,inspection_parts=True)
    assert result.status=='not_assessed' and not result.assessments


def test_unbound_proposition_never_dispatches_inspection(monkeypatch):
    c,f,u,_=mapped(monkeypatch)
    f[0].claim_spans=[]
    monkeypatch.setattr(selector,'chat_completion_json',lambda *a,**k:pytest.fail('unbound call'))
    assert selector.classify_facet_sentence_usefulness(c,c,f,u,inspection_parts=True).failure_code=='invalid_input'
