import json
import pytest
from app.services import evidence_window_experiment as windows, passage_relevance as relevance

def inputs():
    return relevance._system_prompt(None)+relevance._DISPLAY_PROMPT,json.dumps({
        'source_attributed_text':'[t0] Films [t1] changed.',
        'passages':[{'passage_id':'p','source_sentences':['s000'], 'text':'[s000] Some films changed.'}]})

def answer():
    return {'assessments':[{'passage_id':'p','relevance':'partially_relevant','confidence':'high',
        'evidence_role':'source_own_claim_or_finding','rationale':'Inspect partial coverage.',
        'basis':'direct_attribution','source_window_id':'a0'}]}

def test_no_mapping_is_manufactured_and_original_wording_retained():
    system,prompt=inputs(); p=windows.prepare_unmapped_request(system,prompt)
    assert json.loads(p['prompt'])['source_attributed_text']=='Films changed.'
    assert 'claim_token' not in p['system'] and 'claim_token_ids' not in json.loads(p['prompt'])
    result=windows.bind_unmapped_response(answer(),system,prompt,expected_fingerprint=p['fingerprint'])
    assert result.display_observations['p'].claim_token_ranges==[]
    assert result.display_observations['p'].claim_spans==[]

@pytest.mark.parametrize('kind',['mapping','support','window','missing','duplicate','stale'])
def test_invalid_output_not_salvaged(kind):
    system,prompt=inputs(); p=windows.prepare_unmapped_request(system,prompt); raw=answer(); fingerprint=p['fingerprint']
    if kind=='mapping':raw['assessments'][0]['claim_token_labels']=['t0']
    elif kind=='support':raw['support']=True
    elif kind=='window':raw['assessments'][0]['source_window_id']='b9'
    elif kind=='missing':raw['assessments']=[]
    elif kind=='duplicate':raw['assessments']*=2
    else:fingerprint='wrong'
    with pytest.raises(ValueError):windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=fingerprint)

def test_null_window_preserves_partial_assessment():
    system,prompt=inputs(); p=windows.prepare_unmapped_request(system,prompt); raw=answer()
    raw['assessments'][0]['source_window_id']=None
    result=windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=p['fingerprint'])
    assert result.assessments[0].relevance=='partially_relevant' and not result.display_observations

def test_explicit_tagged_diagnostic_preserves_required_findings_and_strict_history():
    system,prompt=inputs(); p=windows.prepare_unmapped_request(system,prompt); raw=answer()
    expected=windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=p['fingerprint'])
    raw['type']='json_object'
    with pytest.raises(ValueError):windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=p['fingerprint'])
    assert windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=p['fingerprint'],allow_json_object_tag=True)==expected

@pytest.mark.parametrize('defect',['wrong_tag','null_tag','unknown_key','missing_assessment','claim_mapping'])
def test_tagged_diagnostic_does_not_relax_required_findings(defect):
    system,prompt=inputs(); p=windows.prepare_unmapped_request(system,prompt); raw=answer(); raw['type']='json_object'
    if defect=='wrong_tag':raw['type']='other'
    elif defect=='null_tag':raw['type']=None
    elif defect=='unknown_key':raw['support']=True
    elif defect=='missing_assessment':raw['assessments'][0].pop('relevance')
    else:raw['assessments'][0]['claim_token_labels']=['t0']
    with pytest.raises(ValueError):windows.bind_unmapped_response(raw,system,prompt,expected_fingerprint=p['fingerprint'],allow_json_object_tag=True)

def comparison_fixture():
    from types import SimpleNamespace
    from tests.unit.test_evidence_display_policy import prepared
    artifact=prepared()
    pages=[SimpleNamespace(index=p.page_index,text=' '*p.character_start+p.text) for p in artifact.passages]
    return artifact,pages

def test_empty_comparison_is_bound_and_leaves_reservoir_unchanged():
    from copy import deepcopy
    artifact,pages=comparison_fixture(); before=deepcopy(artifact)
    request=windows.prepare_unmapped_comparison(artifact,pages,source_title='Study')
    result=windows.bind_unmapped_comparison(request,{'selected':[]},artifact,pages,source_title='Study')
    assert result['outcome']=='no_selection_from_supplied_inputs'
    assert not result['source_support_assessed'] and artifact==before

@pytest.mark.parametrize('full',[False,True])
def test_comparison_request_survives_json_transport(full):
    artifact,pages=comparison_fixture()
    request=windows.prepare_unmapped_comparison(artifact,pages,source_title='Study',full_inspected_context=full)
    saved=json.loads(json.dumps(request))
    result=windows.bind_unmapped_comparison(saved,{'selected':[]},artifact,pages,source_title='Study',full_inspected_context=full)
    assert result['outcome']=='no_selection_from_supplied_inputs'

@pytest.mark.parametrize('change',['source','claim','title','protected'])
def test_comparison_fails_closed_on_stale_or_protected_inputs(change):
    artifact,pages=comparison_fixture()
    request=windows.prepare_unmapped_comparison(artifact,pages,source_title='Study');title='Study'
    if change=='source':pages[0].text+=' Changed.'
    elif change=='claim':artifact.claim.text+=' Changed.'
    elif change=='title':title='Other'
    else:artifact.quotation_check.evidence_passage_ids=[artifact.passages[0].passage_id]
    with pytest.raises(ValueError):windows.bind_unmapped_comparison(request,{'selected':[]},artifact,pages,source_title=title)
