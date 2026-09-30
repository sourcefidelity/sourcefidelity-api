from copy import deepcopy
import hashlib
import json

import pytest

from app.services.evidence_display_selection import bound_observation, select_display_passages
from app.services.evidence_report import _display_terms, _responsive_display_excerpt, _eligible_display_passages, _passage_view


def passage(pid, text):
    return dict(passage_id=pid, excerpt=text, boundary_status='sentence_complete')


def assessment(p, claim_spans, basis='direct_attribution'):
    return dict(relevance='partially_relevant', confidence='high', evidence_role='source_own_claim_or_finding',
                assessed_text_offset_start=0, assessed_text_offset_end=len(p['excerpt']),
                assessed_text_sha256=hashlib.sha256(p['excerpt'].encode()).hexdigest(),
                display_observation=dict(basis=basis, claim_spans=claim_spans, source_span=p['excerpt']))


def select(ps, claim, assessments=None, protected=()):
    return select_display_passages(ps, claim, assessments or {}, protected,
                                  terms=_display_terms, excerpt=_responsive_display_excerpt)


def test_primary_is_diagnostic_not_merely_supportive_and_context_adds_mechanism():
    claim='Studio publicity made Jordan a mysterious foreign star.'
    theory=passage('theory', 'Studio publicity created images through magazine advertising.')
    specific=passage('specific', 'Jordan rejected the mysterious foreign star persona.')
    duplicate=passage('duplicate', 'Jordan refused the mysterious foreign star image.')
    ps=[theory,specific,duplicate]
    before=deepcopy(ps)
    a={'theory':assessment(theory,['Studio publicity'],'general_framework'),
       'specific':assessment(specific,['Jordan a mysterious foreign star']),
       'duplicate':assessment(duplicate,['Jordan a mysterious foreign star'])}
    selected, trace=select(ps,claim,a)
    assert [p['passage_id'] for p in selected]==['specific','theory']
    assert trace['support_assessed'] is False
    assert set(trace['eligible_passage_ids'])=={'theory','specific','duplicate'}
    assert ps==before


def test_explicit_framework_application_does_not_require_named_entity_match():
    claim='Theory explains Jordan through deliberate image construction.'
    theory=passage('theory','Deliberate image construction coordinates promotion.')
    actor=passage('actor','Jordan was born in a coastal city.')
    selected,_=select([actor,theory],claim,{
        'actor':assessment(actor,[], 'unclear'),
        'theory':assessment(theory,['deliberate image construction'])})
    assert selected[0]==theory


def test_no_quota_filling_or_example_when_no_incremental_value():
    claim='Studios controlled promotion and publicity.'
    direct=passage('direct','Studios controlled promotion and publicity.')
    example=passage('example','A famous performer attracted magazine publicity.')
    selected,_=select([direct,example],claim,{
        'direct':assessment(direct,['Studios controlled promotion and publicity']),
        'example':assessment(example,['promotion and publicity'],'illustrative_example')})
    assert selected==[direct]
    assert select([direct,example],claim)[0]==[direct]


def test_longer_primary_context_counts_as_already_present():
    p=passage('p','The primary sentence describes control. Longer context explains promotion and publicity.')
    q=passage('q','Promotion and publicity were coordinated.')
    result,_=select_display_passages([p,q],'Control of promotion and publicity.',{},[],
        terms=_display_terms,excerpt=lambda text,claim:text.split('.')[0]+'.')
    assert result==[p]


def test_equal_advisory_labels_preserve_prior_rank_not_claim_span_length():
    claim='The bridge collapsed and the village was evacuated.'
    setup=passage('setup','The village was evacuated.')
    ending=passage('ending','The bridge collapsed and the village was evacuated.')
    observations={'setup':assessment(setup,['the village was evacuated']),
                  'ending':assessment(ending,['The bridge collapsed','the village was evacuated'])}
    selected,trace=select([setup,ending],claim,observations)
    assert selected[0] == setup
    assert trace['support_assessed'] is False
    # A protected quotation remains first even when it covers fewer aspects.
    assert select([setup,ending],claim,observations,protected=['setup'])[0][0] == setup


def test_partial_word_count_does_not_beat_relevant_same_basis():
    claim='The industry and audiences shape star images through publicity.'
    weaker=passage('weak','One star had a constructed image.')
    direct=passage('direct','Audiences contribute to the making of star images.')
    observations={'weak':assessment(weaker,[claim]),
                  'direct':{**assessment(direct,['audiences']), 'relevance':'relevant'}}
    assert select([weaker,direct],claim,observations)[0][0] == direct


def test_unclosed_or_heading_hint_cannot_promote_primary():
    claim='Actors faced limited work.'
    for text in ('She recalled, “There were few parts .',
                 'Popular Culture\nActors faced limited work.'):
        p=passage('p',text)
        assert not bound_observation(p,assessment(p,[claim]),claim)


def test_confidence_does_not_displace_connected_subject_evidence():
    from app.services.evidence_report import _prioritize_display_passages
    ps=[passage('generic','Publicity and studio marketing created images.'),
        passage('subject','Jordan rejected the foreign persona.')]
    gate={'assessments':[dict(passage_id=p['passage_id'],relevance='partially_relevant',
        confidence='medium' if p['passage_id']=='subject' else 'high',evidence_role='source_own_claim_or_finding') for p in ps]}
    assert _prioritize_display_passages(ps,'Publicity and studio marketing created Jordan’s image.',gate)[0]['passage_id']=='subject'


@pytest.mark.parametrize('change',['hash','span','claim','offset','fragment'])
def test_invalid_observation_falls_back_without_mutating_source(change):
    p=passage('p','The source describes a general process.')
    a=assessment(p,['a general process'])
    if change=='hash': a['assessed_text_sha256']='0'*64
    if change=='span': a['display_observation']['source_span']='Invented source text.'
    if change=='claim': a['display_observation']['claim_spans']=['not supplied']
    if change=='offset': a['assessed_text_offset_end']+=1
    if change=='fragment': a['display_observation']['source_span']='general process'
    assert not bound_observation(p,a,'This is a general process.')


def test_protected_quote_keeps_priority_and_partial_can_add_context():
    ps=[passage('a','A direct general account is here.'),passage('quote','The quoted wording is here.')]
    selected,_=select(ps,'The quoted wording is here.',protected=['quote'])
    assert selected[0]['passage_id']=='quote'
    gate=dict(status='complete',assessments=[dict(passage_id='a',relevance='relevant'),
        dict(passage_id='quote',relevance='partially_relevant',evidence_role='methods_or_background')])
    assert len(_eligible_display_passages(ps,gate))==2


def test_bound_sentence_is_used_and_full_context_preserved():
    p=passage('p','The generic introduction comes first. Jordan disputed the claim.')
    a=assessment(p,['Jordan disputed the claim'])
    a['display_observation']['source_span']='Jordan disputed the claim.'
    view=_passage_view(p,a,claim_text='Jordan disputed the claim.')
    assert view['display_text']=='Jordan disputed the claim.'
    assert 'generic introduction' in view['context_text']


def test_persistence_rejects_unbound_selection_observation():
    from types import SimpleNamespace
    from app.services.verification_report import _validate_display_observation, ReportAuthorizationError
    from app.services.verification_evidence import PassageDisplayObservation
    a=SimpleNamespace(display_observation=PassageDisplayObservation(
        basis='direct_attribution',claim_spans=['Real claim'],source_span='Invented source.'))
    with pytest.raises(ReportAuthorizationError):
        _validate_display_observation(a,'Real source.','Real claim')
    a.display_observation.source_span='Real source.'
    with pytest.raises(ReportAuthorizationError):
        _validate_display_observation(a,'Real source.','Different claim')
