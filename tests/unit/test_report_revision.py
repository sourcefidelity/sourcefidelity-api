"""Report usability fixes with source and paper boundaries preserved."""

import hashlib
from types import SimpleNamespace

import fitz
import pytest

from app.services.evidence_report import (
    _unavailable_member, _normalize_display_text, _render_panel_template,
    citation_tones, attach_quotation_difference_geometry,
    _member_quotation_targets, _bind_inline_styles,
)
from app.services.paper_annotations import text_selection_anchor, _validated_anchor, PaperAnnotationError
from app.services.processing_metrics import record_llm_attempt, record_llm_usage, report_processing_metrics, _current
from app.services.text_extractor import _clean_text


def reference():
    return SimpleNamespace(reference_id='r1',raw_ref='Researcher (2020). Study. https://example.org/study',author='Researcher',year='2020',title='Study',doi=None,url='https://example.org/study',parsed_fields={})


def abstract_member(text, quote='The trials used three independent groups.'):
    claim=SimpleNamespace(text=f'The authors wrote “{quote}” (Researcher, 2020).',claim_type='quotation',page_locator='')
    return _unavailable_member({'abstract_available':True,'abstract_evidence':{'text':text},'abstract_relevance':{'relevance':'partially_relevant'}},reference(),claim)


def test_complete_abstract_quote_bypasses_semantic_partial_relevance():
    member=abstract_member('Abstract. The trials used three independent groups. Further context.')
    assert member['quotation_check']['status']=='complete'
    assert member['quotation_check']['attention'] is False
    assert member['show_quotation_check']
    assert 'three independent groups' in member['best_evidence']['display_text']
    assert member['best_evidence']['context_text'].startswith('Abstract.')


def test_abstract_fragment_and_missing_member_cannot_establish_quote_failure():
    member=abstract_member('The trials used three independent')
    assert member['quotation_check']['status']=='not_assessed'
    assert member['quotation_check']['attention'] is False
    assert 'full source' in member['quotation_check']['label']


def test_every_quoted_span_must_match_abstract():
    member=abstract_member('First complete quotation.', 'First complete quotation.” and “Another complete quotation.')
    assert member['quotation_check']['status']=='not_assessed'


def test_hyphen_is_preserved_in_paper_and_display_text():
    from app.services.verification_evidence import _quotation_match
    assert _clean_text('Spanish-\nspeaking writers')=='Spanish-speaking writers'
    assert _normalize_display_text('Spanish-\nspeaking writers')=='Spanish-speaking writers'
    assert _clean_text('Spanish\u00adspeaking writers')=='Spanish-speaking writers'
    assert _quotation_match('Spanish-\nspeaking writers', 'Spanish-speaking writers') is not None
    assert _quotation_match('Spanishspeaking writers', 'Spanish-speaking writers') is None


def test_mixed_citation_segments_follow_availability_order():
    members=[{'coverage_level':'unavailable'},{'coverage_level':'full_text','relevance_status':'connected'},{'coverage_level':'abstract_only','relevance_status':'connected'}]
    assert citation_tones({'members':members,'tone':'limited_evidence'})==['evidence_available','limited_evidence','not_assessed']


def pdf():
    doc=fitz.open(); page=doc.new_page()
    page.insert_text((72,72),'Spanish-speaking writers')
    page.insert_text((72,92),'Second line for selection')
    content=doc.tobytes(no_new_id=True);doc.close();return content


def test_native_word_selection_derives_two_lines_and_rejects_forged_range():
    content=pdf()
    anchor, dimensions=text_selection_anchor(content,{'selected_words':[{'page_index':0,'start':0,'end':5}],'rectangles':[{'x0':0,'x1':999999}]})
    assert len(anchor['rectangles'])==2
    validated=_validated_anchor(anchor,paper_content_sha256=hashlib.sha256(content).hexdigest(),page_dimensions=dimensions)
    assert validated['anchor_kind']=='text_selection'
    assert validated['rectangles'][0]['x0']==72
    with pytest.raises(PaperAnnotationError):
        text_selection_anchor(content,{'selected_words':[{'page_index':0,'start':0,'end':99999}]})


def test_historical_display_restores_only_pdf_proven_hyphen():
    citation={'student_text':'Spanishspeaking writers','display_student_text':'Spanishspeaking writers','members':[], 'paper_location':{'localization_level':'exact_rectangle','rectangles':[{'page_index':0,'x0':65,'y0':50,'x1':400,'y1':80}]}}
    view=attach_quotation_difference_geometry({'citations':[citation]},pdf())
    assert view['citations'][0]['student_text']=='Spanishspeaking writers'
    assert view['citations'][0]['display_student_text']=='Spanish-speaking writers'
    assert view['citations'][0]['paper_wording_provenance']['paper_sha256']==hashlib.sha256(pdf()).hexdigest()


def test_missing_usage_is_not_reported_as_zero_tokens():
    import time
    token=_current.set({'stage':'verify','llm_calls':0,'total_tokens':0,'missing_usage_calls':0,'_wall':time.perf_counter(),'_cpu':time.process_time()})
    try:
        record_llm_attempt();record_llm_usage(None)
        metrics=report_processing_metrics(SimpleNamespace(upload_evidence={}))
        assert metrics['llm_calls']==1 and metrics['total_tokens'] is None
        assert metrics['partial']
    finally:
        _current.reset(token)
    assert report_processing_metrics(SimpleNamespace(upload_evidence={})).get('total_tokens') is None


def test_quote_is_not_borrowed_by_another_marker_in_the_same_sentence():
    text='A finding (Alpha, 2020) and “the quoted words” (Beta, 2021).'
    markers=[]
    for label,ref in [('(Alpha, 2020)','r1'),('(Beta, 2021)','r2')]:
        start=text.index(label)
        markers.append(SimpleNamespace(text=label,local_start=start,local_end=start+len(label),reference_ids=[ref],marker_type='parenthetical'))
    claim=SimpleNamespace(text=text,reference_ids=['r1','r2'],citation_markers=markers)
    assert _member_quotation_targets(claim,'r1')==[]
    assert _member_quotation_targets(claim,'r2')==['the quoted words']


def test_geometry_styles_use_nonce_stylesheet_without_inline_style_permission():
    rendered=_bind_inline_styles('<style nonce="safe-nonce"></style><span style="left:12.5%;width:4%">text</span>', 'safe-nonce')
    assert ' style=' not in rendered
    assert '[data-report-style="s0"]{left:12.5%;width:4%}' in rendered
    assert '<style nonce="safe-nonce">' in rendered


def test_grouped_upload_is_one_button_per_unavailable_member():
    member=abstract_member('No quoted wording is here.')
    citation={'members':[member,{**member,'reference_id':'r2'}],'student_text':'Example','upload_action':{'enabled':True,'href':'/report/id/citation/id/source/upload'}}
    rendered=_render_panel_template(citation,1)
    assert rendered.count('Abstract Retrieved</span></span></h3>')==2
    assert 'data-source-member="1" hidden' in rendered
    assert 'aria-label="Next source"' in rendered
    assert rendered.count('data-choose-source')==2
    assert rendered.count('name="file"')==2
    assert 'type="submit"' not in rendered
    assert 'class="reference-url"' in rendered
