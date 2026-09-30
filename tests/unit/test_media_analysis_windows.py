import copy
import pytest
from app.services.media_analysis_windows import (
    plan_media_analysis_windows, summarize_media_analysis_windows, media_window_prompt_tokens)
from app.services.subject_identifier import bind_media_analysis_candidates


def test_every_nonwhitespace_character_is_accounted_for_without_title_selection():
    body='First paragraph.\n\nSecond paragraph with no media.\n\nThird paragraph.'
    plan=plan_media_analysis_windows(body,max_input_tokens=media_window_prompt_tokens('First paragraph.')+4,max_windows=1)
    covered=set()
    for r in plan['regions']:covered.update(range(r['start'],r['end']))
    assert all(i in covered for i,c in enumerate(body) if not c.isspace())
    assert any(r['status']=='window_allowance_exhausted' for r in plan['regions'])
    assert summarize_media_analysis_windows(body,plan,{})['coverage_status']=='incomplete'


def test_oversize_paragraph_is_retained_as_gap_and_plan_tampering_rejected():
    body='A'*5000+'\n\nShort paragraph.'
    plan=plan_media_analysis_windows(body,max_input_tokens=700,max_windows=4)
    assert plan['regions'][0]['status']=='paragraph_over_budget'
    changed=copy.deepcopy(plan);changed['regions'][0]['end']-=1
    with pytest.raises(ValueError):summarize_media_analysis_windows(body,changed,{})
    with pytest.raises(ValueError):summarize_media_analysis_windows(body,plan,{0:{}})


def test_results_rebind_to_exact_global_spans_without_absence_conclusion():
    text='Work begins quietly. Work ends in silence.'
    body='Prelude.\n\n'+text
    plan=plan_media_analysis_windows(body,max_input_tokens=media_window_prompt_tokens(text),max_windows=3)
    results={}
    for i,r in enumerate(plan['regions']):
        window=body[r['start']:r['end']]
        raw=[dict(title='Work',passage=text,proposed_role='substantive_analysis',media_type='song')] if text in window else []
        results[i]=bind_media_analysis_candidates(raw,window,[]).model_dump(mode='json')
    summary=summarize_media_analysis_windows(body,plan,results)
    assert summary['coverage_status']=='processed_all_windows'
    assert body[summary['candidates'][0]['passage_start']:summary['candidates'][0]['passage_end']]==text
    assert not summary['reference_absence_assessed'] and not summary['automatic_findings_enabled']
    corrupt=copy.deepcopy(results)
    last=max(corrupt);corrupt[last]['body_sha256']='x'*64
    assert summarize_media_analysis_windows(body,plan,corrupt)['coverage_status']=='incomplete'


def test_empty_input_is_not_a_completed_negative():
    body=' \n\n '
    plan=plan_media_analysis_windows(body,max_input_tokens=1500,max_windows=4)
    assert summarize_media_analysis_windows(body,plan,{})['coverage_status']=='incomplete'


def test_output_cap_and_failed_window_cannot_claim_processed_coverage():
    parts=[f'Work{i:02d} changes the ending.' for i in range(12)]
    body=' '.join(parts)
    plan=plan_media_analysis_windows(body,max_input_tokens=1500,max_windows=1)
    raw=[dict(title=f'Work{i:02d}',passage=p,proposed_role='uncertain',media_type='book')
         for i,p in enumerate(parts)]
    result=bind_media_analysis_candidates(raw,body,[]).model_dump(mode='json')
    summary=summarize_media_analysis_windows(body,plan,{0:result})
    assert summary['regions'][0]['processing_status']=='candidate_cap_reached'
    assert summary['coverage_status']=='incomplete'
    assert len(summary['candidates'])==12
    result['status']='unavailable'
    assert summarize_media_analysis_windows(body,plan,{0:result})['coverage_status']=='incomplete'
