import copy
import json

import pytest

from app.services import evidence_context_experiment as context
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from app.services.verification_evidence import ClaimEvidence
from tests.unit.test_multipage_extract_experiment import fixture as source_fixture


def fixture():
    parts, source = source_fixture()
    m = context.prepare_multipage_transition(*parts, **source)
    kw = dict(claim=ClaimEvidence(claim_id='c', paper_version_id='p',
               text='The account describes a journey and its consequences.', claim_type='paraphrase'),
              source_title='Study', regions=source['inspected_regions'], pages=source['pages'],
              layout_by_page=source['layout_by_page'], source_binding=source['source_binding'])
    request = context.prepare_linked_comparison(**kw, multipage_extracts=[m])
    return request, kw


def answer():
    return {'selected': [{'region_id': 'm000', 'sentence_ids': ['s000'], 'purpose': 'primary',
                          'context_for': None, 'why_useful': 'The joined sentence describes the journey.'}]}


def test_linked_projection_retains_two_pages_and_baseline_inputs():
    request, kw = fixture()
    before = copy.deepcopy(kw)
    payload, base = json.loads(request['prompt']), json.loads(request['baseline']['prompt'])
    assert payload['regions'][:-1] == base['regions']
    assert {k: v for k, v in payload.items() if k != 'regions'} == {k: v for k, v in base.items() if k != 'regions'}
    bound = context.bind_linked_comparison(json.loads(json.dumps(request)), answer(), **kw)
    result = bound['selected'][0]
    assert 'page_index' not in result and 'start' not in result
    assert [p['page_index'] for p in result['parts']] == [0, 1]
    assert not bound['source_support_assessed'] and not bound['semantic_acceptance']
    assert kw == before


def test_empty_is_not_source_wide_absence():
    request, kw = fixture()
    bound = context.bind_linked_comparison(request, {'selected': []}, **kw)
    assert bound['outcome'] == 'no_selection_from_supplied_inputs'
    assert not bound['selected']


@pytest.mark.parametrize('change', ['window', 'extra', 'duplicate', 'context', 'support', 'fragment'])
def test_invalid_response_rejected_whole(change):
    request, kw = fixture()
    raw = answer()
    if change == 'window': raw['selected'][0]['sentence_ids'] = ['s001']
    elif change == 'extra': raw['selected'][0]['invented'] = True
    elif change == 'duplicate': raw['selected'] *= 2
    elif change == 'context': raw['selected'][0]['context_for'] = 'r000'
    elif change == 'support': raw['selected'][0]['purpose'] = 'supports'
    else:
        raw['selected'].append({**raw['selected'][0], 'region_id': 'r001', 'purpose': 'additional_material'})
    with pytest.raises(ValueError): context.bind_linked_comparison(request, raw, **kw)


@pytest.mark.parametrize('change', ['claim', 'source', 'prompt', 'part', 'missing'])
def test_changed_inputs_fail_binding(change):
    request, kw = fixture()
    if change == 'claim': kw['claim'] = kw['claim'].model_copy(update={'text': 'Different.'})
    elif change == 'source': kw['source_binding'] = {'content': 'different'}
    elif change == 'prompt': request['prompt'] += 'modified'
    elif change == 'part': request['multipage_extracts'][0]['parts'][0]['start'] += 1
    else: request['multipage_extracts'] = []
    with pytest.raises(ValueError): context.bind_linked_comparison(request, answer(), **kw)


def test_complete_prompt_budget_still_applies():
    request, kw = fixture()
    with pytest.raises(LLMInputBudgetExceeded):
        context.prepare_linked_comparison(**kw, multipage_extracts=request['multipage_extracts'], max_input_tokens=100)


def test_duplicate_seam_not_a_new_candidate():
    request, kw = fixture()
    with pytest.raises(ValueError, match='duplicate_linked_extract'):
        context.prepare_linked_comparison(**kw, multipage_extracts=request['multipage_extracts'] * 2)


def test_single_page_selection_still_uses_existing_coordinates():
    from tests.unit.test_evidence_context_experiment import fixture as ordinary_fixture
    pages, claim, regions, _ = ordinary_fixture()
    kw = dict(claim=claim, source_title='Study', regions=regions, pages=pages,
              layout_by_page={}, source_binding={'content': 'bound'})
    request = context.prepare_linked_comparison(**kw, multipage_extracts=[])
    raw = answer()
    raw['selected'][0]['region_id'] = 'r000'
    result = context.bind_linked_comparison(request, raw, **kw)['selected'][0]
    assert result['page_index'] == 0 and result['start'] == 0
    assert 'parts' not in result


def test_overlapping_multi_page_choices_are_not_two_distinct_extracts():
    from dataclasses import replace
    parts, source = source_fixture()
    suffix = ' Another consequence followed.'
    page = source['pages'][1]
    source['pages'][1] = replace(page, text=page.text + suffix)
    parts[1] = replace(parts[1], text=parts[1].text + suffix, end=parts[1].end + len(suffix))
    source['inspected_regions'][1] = parts[1]
    box = source['layout_by_page'][1][1]
    source['layout_by_page'][1][1] = replace(box, text=box.text + suffix, end=box.end + len(suffix))
    shortest = context.prepare_multipage_transition(*parts, **source)
    longer = context.prepare_multipage_extract(parts, **source)
    kw = dict(claim=ClaimEvidence(claim_id='c', paper_version_id='p', text='A journey.', claim_type='paraphrase'),
              source_title='Study', regions=source['inspected_regions'], pages=source['pages'],
              layout_by_page=source['layout_by_page'], source_binding=source['source_binding'])
    request = context.prepare_linked_comparison(**kw, multipage_extracts=[shortest, longer])
    raw = answer()
    raw['selected'].append({**raw['selected'][0], 'region_id': 'm001', 'purpose': 'additional_material'})
    with pytest.raises(ValueError, match='overlapping_linked_selection'):
        context.bind_linked_comparison(request, raw, **kw)


def disjoint_fixture():
    from tests.unit.test_evidence_context_experiment import fixture as ordinary_fixture
    pages, claim, regions, _ = ordinary_fixture()
    kw = dict(claim=claim, source_title='Study', regions=regions, pages=pages,
              layout_by_page={}, source_binding={'content': 'bound'})
    first = {**answer()['selected'][0], 'region_id': 'r000'}
    second = {**first, 'sentence_ids': ['s001'], 'purpose': 'additional_material'}
    return kw, {'selected': [first, second]}


def test_disjoint_opt_in_does_not_relabel_historical_rejection():
    kw, raw = disjoint_fixture()
    original = context.prepare_linked_comparison(**kw, multipage_extracts=[])
    with pytest.raises(ValueError, match='selection_order'):
        context.bind_linked_comparison(original, raw, **kw)
    new = context.prepare_linked_comparison(**kw, multipage_extracts=[], allow_disjoint_same_region=True)
    assert new['prompt'] == original['prompt'] and new['system'] == original['system']
    assert new['request_sha256'] != original['request_sha256']
    result = context.bind_linked_comparison(new, raw, **kw, allow_disjoint_same_region=True)
    assert len(result['selected']) == 2
    assert result['selected'][0]['end'] <= result['selected'][1]['start']
    assert [v['selection_index'] for v in result['selected']] == [0, 1]
    with pytest.raises(ValueError, match='stale_linked_comparison'):
        context.bind_linked_comparison(new, raw, **kw)


def test_disjoint_rejects_overlapping_coordinates():
    kw, raw = disjoint_fixture()
    req = context.prepare_linked_comparison(**kw, multipage_extracts=[], allow_disjoint_same_region=True)
    raw['selected'][1]['sentence_ids'] = ['s000', 's001']
    with pytest.raises(ValueError, match='overlapping_linked_selection'):
        context.bind_linked_comparison(req, raw, **kw, allow_disjoint_same_region=True)


def test_disjoint_context_target_is_exact_and_ambiguous_target_rejected():
    kw, raw = disjoint_fixture()
    req = context.prepare_linked_comparison(**kw, multipage_extracts=[], allow_disjoint_same_region=True)
    raw['selected'][1].update(purpose='necessary_context', context_for='r000')
    result = context.bind_linked_comparison(req, raw, **kw, allow_disjoint_same_region=True)
    assert result['selected'][1]['context_for_selection_index'] == 0
    raw['selected'].append({**raw['selected'][1], 'sentence_ids': ['s002']})
    with pytest.raises(ValueError, match='context_binding'):
        context.bind_linked_comparison(req, raw, **kw, allow_disjoint_same_region=True)
