import copy
import json
from dataclasses import replace

import pytest

from app.services import evidence_context_experiment as c
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from tests.unit.test_evidence_context_experiment import fixture


def prepared():
    pages, claim, regions, _ = fixture()
    kw = dict(pages=pages, claim=claim, regions=regions, source_title='Study', source_binding={'source':'fixed'})
    return c.prepare_primary_comparison(**kw), kw


def test_json_mode_instruction_is_explicit():
    assert 'json' in c.PRIMARY_SYSTEM.lower()


@pytest.mark.parametrize('pid', ['p000', None])
def test_whole_passage_or_empty_preserves_inputs(pid):
    request, kw = prepared(); before = copy.deepcopy(kw)
    result = c.bind_primary_comparison(json.loads(json.dumps(request)), {'primary_id':pid}, **kw)
    assert (result['primary'] is None) == (pid is None)
    if pid: assert result['primary']['text'] == kw['regions'][0].text
    assert not result['source_support_assessed'] and not result['semantic_acceptance']
    assert kw == before


@pytest.mark.parametrize('raw', [{}, {'primary_id':'unknown'}, {'primary_id':[]}, {'primary_id':1},
    {'primary_id':None,'support':False}, {'primary_id':'p000','sentence_ids':['s000']}])
def test_strict_small_response(raw):
    request, kw = prepared()
    with pytest.raises(ValueError): c.bind_primary_comparison(request, raw, **kw)


@pytest.mark.parametrize('defect', ['claim','source','page','title','prompt','unit'])
def test_fresh_binding(defect):
    request, kw = prepared()
    if defect == 'claim': kw['claim'] = kw['claim'].model_copy(update={'text':'Changed.'})
    elif defect == 'source': kw['source_binding'] = {'source':'changed'}
    elif defect == 'page': kw['pages'][0] = replace(kw['pages'][0], text=kw['pages'][0].text+' Changed.')
    elif defect == 'title': kw['source_title'] = 'Changed'
    elif defect == 'prompt': request['prompt'] += 'Changed'
    else: request['regions'][0]['text'] = 'Changed.'
    with pytest.raises(ValueError): c.bind_primary_comparison(request, {'primary_id':'p000'}, **kw)


@pytest.mark.parametrize('defect', ['overlap','fragment','role','long','budget','count'])
def test_requires_prepared_bounded_inputs(defect):
    request, kw = prepared()
    if defect == 'overlap': kw['regions'] *= 2
    elif defect == 'fragment':
        r = kw['regions'][0]; kw['regions'] = [replace(r, start=1, text=r.text[1:])]
    elif defect == 'role': kw['regions'] = [replace(kw['regions'][0],role='publication_metadata')]
    elif defect == 'long':
        text = 'Words follow. '*110
        kw['pages'][0] = replace(kw['pages'][0],text=text)
        kw['regions'] = [replace(kw['regions'][0],start=0,end=len(text),text=text)]
    elif defect == 'count': kw['regions'] *= 11
    else: kw['max_input_tokens'] = 10
    with pytest.raises((ValueError, LLMInputBudgetExceeded)): c.prepare_primary_comparison(**kw)
