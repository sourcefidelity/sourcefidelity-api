import copy
import json
from dataclasses import replace

import pytest

from app.services import evidence_context_experiment as c
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from tests.unit.test_linked_comparison_experiment import fixture


def prepared():
    linked, kw = fixture()
    kw['multipage_extracts'] = linked['multipage_extracts']
    return c.prepare_unit_comparison(**kw), kw


def answer(uid='m000:0-0'):
    return {'selected': [dict(unit_id=uid, purpose='primary', context_for=None, why_useful='Useful context.')]}


def test_exact_menu_keeps_context_and_closed_seam_without_mutation():
    request, kw = prepared()
    before = copy.deepcopy(kw)
    payload = json.loads(request['prompt'])
    assert [{k: v for k, v in r.items() if k != 'selectable_units'} for r in payload['regions']] == json.loads(request['baseline']['prompt'])['regions']
    assert all(not (u['region_id'] == 'r001' and u['sentence_ids'][0] == 's000') for u in request['units'].values())
    bound = c.bind_unit_comparison(json.loads(json.dumps(request)), answer(), **kw)
    assert len(bound['selected'][0]['parts']) == 2
    assert not bound['source_support_assessed'] and not bound['semantic_acceptance']
    assert kw == before


@pytest.mark.parametrize('defect', ['fragment', 'extra', 'duplicate', 'overlap', 'context', 'missing', 'support'])
def test_invalid_supplement_rejects_whole_advice(defect):
    request, kw = prepared()
    raw = answer()
    if defect == 'fragment': raw['selected'].append({**raw['selected'][0], 'unit_id': 'r001:0-0', 'purpose': 'additional_material'})
    elif defect == 'extra': raw['source_absence'] = True
    elif defect == 'duplicate': raw['selected'] *= 2
    elif defect == 'overlap':
        # Both ordinary pieces of the seam are fragments; duplicate overlapping
        # multipage input must also fail before any selection is consumed.
        kw['multipage_extracts'] *= 2
    elif defect == 'context': raw['selected'][0]['context_for'] = 'm000:0-0'
    elif defect == 'missing': raw['selected'][0].pop('purpose')
    else: raw['selected'][0]['purpose'] = 'supports'
    with pytest.raises(ValueError): c.bind_unit_comparison(request, raw, **kw)


@pytest.mark.parametrize('defect', ['menu', 'prompt', 'claim', 'source', 'page', 'title'])
def test_fresh_inputs_required(defect):
    request, kw = prepared()
    if defect == 'menu': request['units'].clear()
    elif defect == 'prompt': request['prompt'] += 'x'
    elif defect == 'claim': kw['claim'] = kw['claim'].model_copy(update={'text': 'Changed.'})
    elif defect == 'source': kw['source_binding'] = {'source': 'changed'}
    elif defect == 'page': kw['pages'][0] = replace(kw['pages'][0], text='Changed.')
    else: kw['source_title'] = 'Changed'
    with pytest.raises(ValueError): c.bind_unit_comparison(request, answer(), **kw)


def test_budget_and_empty_boundaries():
    request, kw = prepared()
    result = c.bind_unit_comparison(request, {'selected': []}, **kw)
    assert result['outcome'] == 'no_selection_from_supplied_inputs'
    with pytest.raises(LLMInputBudgetExceeded): c.prepare_unit_comparison(**kw, max_input_tokens=100)


def test_menu_complete_under_declared_boundary_and_overlapping_output_rejected():
    from types import SimpleNamespace
    from app.services.verification_evidence import ClaimEvidence
    text = 'The market grew. The policy changed. Investment increased.'
    kw = dict(claim=ClaimEvidence(claim_id='c', paper_version_id='p', text='Markets grew.', claim_type='paraphrase'),
              source_title='Study', pages=[SimpleNamespace(index=0, text=text)],
              regions=[c.Region(0, 0, len(text), text, 'body_prose')],
              source_binding={'source': 'fixed'}, layout_by_page={}, multipage_extracts=[])
    request = c.prepare_unit_comparison(**kw)
    assert set(request['units']) == {f'r000:{lo}-{hi}' for lo in range(3) for hi in range(lo, 3)}
    raw = answer('r000:0-1')
    raw['selected'].append({**raw['selected'][0], 'unit_id': 'r000:1-2', 'purpose': 'additional_material'})
    with pytest.raises(ValueError, match='overlapping'): c.bind_unit_comparison(request, raw, **kw)
    raw['selected'][1]['unit_id'] = 'r000:2-2'
    assert len(c.bind_unit_comparison(request, raw, **kw)['selected']) == 2
