"""Field-only reference differences bind to one complete paper entry."""
import hashlib
from types import SimpleNamespace

import fitz
import pytest

from app.services.evidence_report import (
    attach_quotation_difference_geometry, _reference_identity_conflict_view,
    _render_reference_panel_template, _unavailable_member, member_tone,
)


def test_reference_year_highlight_excludes_author_title_and_parentheses():
    raw = 'Writer, A. (2020). A study of learning. Academic Press.'
    view = {'reference_practice': [{
        'finding_type': 'bibliographic_conflict', 'source': {'raw_reference': raw},
        'field_difference': {'field_name': 'year', 'submitted_value': '2020'},
    }]}
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), 'Earlier discussion (Writer, 2020).')
        page.insert_text((50, 120), raw)
        data = doc.tobytes()
        result = attach_quotation_difference_geometry(view, data)
        finding = result['reference_practice'][0]
        assert finding['localization_status'] == 'exact_field'
        boxes = finding['rectangles']
        assert len(boxes) == 1
        assert page.get_textbox(fitz.Rect(*(boxes[0][k] for k in ('x0','y0','x1','y1')))) == '2020'
        assert finding['geometry_provenance']['presentation_sha256'] == hashlib.sha256(data).hexdigest()
        assert 'rectangles' not in view['reference_practice'][0]


def test_wrapped_title_highlight_covers_only_the_differing_field():
    title = 'A study of learning across different classrooms'
    raw = f'Writer, A. (2020). {title}. Academic Press.'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50,100),'Writer, A. (2020). A study of learning')
        page.insert_text((50,120),'across different classrooms. Academic Press.')
        view = {'reference_practice':[{'source':{'raw_reference':raw,'title':title},
            'field_difference':{'field_name':'title','submitted_value':title}}]}
        finding = attach_quotation_difference_geometry(view,doc.tobytes())['reference_practice'][0]
        assert finding['localization_status'] == 'exact_field'
        assert len(finding['rectangles']) == 2
        marked = ' '.join(page.get_textbox(fitz.Rect(*(b[k] for k in ('x0','y0','x1','y1')))) for b in finding['rectangles'])
        assert marked == title


def test_printed_locator_cannot_bridge_discontinuous_article_pages():
    from app.services.citation_extractor import _reference_contains_locator
    reference = SimpleNamespace(raw_ref='Writer (1946). Article. Magazine, pp. 42, 86-87.')
    assert _reference_contains_locator(reference,'86-87')
    assert not _reference_contains_locator(reference,'42-86')


@pytest.mark.parametrize('repeat', [False, True])
def test_missing_or_ambiguous_complete_entry_never_marks_a_year_elsewhere(repeat):
    raw = 'Writer, A. (2020). A study. Academic Press.'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), raw if repeat else 'Another author (2020). Another study.')
        if repeat:
            page.insert_text((50, 100), raw)
        view = {'reference_practice': [{'source': {'raw_reference':raw},
            'rectangles':[{'page_index':0,'x0':1,'y0':1,'x1':500,'y1':500}],
            'field_difference':{'field_name':'year','submitted_value':'2020'}}]}
        result = attach_quotation_difference_geometry(view, doc.tobytes())
        assert result['reference_practice'][0]['rectangles'] == []


def test_identity_panel_uses_plausible_candidate_and_exact_field_values():
    comparison = {'field_name':'year','outcome':'material_conflict','reason_code':'year_identity_conflict'}
    result = _reference_identity_conflict_view({'expected':{'year':'2020'}, 'candidates':[
        {'plausible_identity_match':False,'observed':{'year':'1800'},'comparisons':[comparison]},
        {'plausible_identity_match':True,'provider':'catalog','candidate_id':'candidate-1',
         'observed':{'year':'2021'},'comparisons':[comparison]},
    ]})
    assert result['located_record']['year'] == '2021'
    difference = result['field_differences'][0]
    html = _render_reference_panel_template({'finding_type':'bibliographic_conflict',
        'finding':'Compare the year.', 'conflicting_fields':['year'], 'field_difference':difference,
        'located_record':result['located_record'],
        'source':{'raw_reference':'Writer (2020). Study.', 'author':'Writer','year':'2020','title':'Study'}},1)
    assert 'In your reference:</strong> 2020' in html
    assert 'In the located record:</strong> 2021' in html
    assert 'Record provider: catalog' in html


@pytest.mark.parametrize('fault', ['', 'hash', 'span', 'confidence', 'missing_detail', 'truncated'])
def test_abstract_scope_warning_requires_bound_affirmative_evidence(fault):
    text = 'The study concerns marine ecosystems. It examines coral growth.'
    statement = 'The source discusses classroom language learning (Writer, 2020).'
    abstract_hash, claim_hash = (hashlib.sha256(v.encode()).hexdigest() for v in (text, statement))
    scope = {'status':'complete','relevance':'apparent_mismatch','confidence':'high',
        'discrepancy':'different_subject','abstract_span':'marine ecosystems',
        'claim_span':'classroom language learning','rationale':'The described subjects differ.',
        'attention':True,'abstract_sha256':abstract_hash,'claim_sha256':claim_hash}
    if fault == 'hash': scope['abstract_sha256'] = '0'*64
    if fault == 'span': scope['abstract_span'] = 'invented wording'
    if fault == 'confidence': scope['confidence'] = 'low'
    if fault == 'missing_detail': scope['discrepancy'] = 'detail_not_mentioned'
    relevance = {'status':'complete','relevance':'not_relevant','scope_assessment':scope,
        'abstract_sha256':abstract_hash,'claim_sha256':claim_hash}
    ref = SimpleNamespace(reference_id='a',raw_ref='Writer (2020). Study.',author='Writer',year='2020',title='Study',doi=None,url=None)
    member = _unavailable_member({'abstract_available':True,'abstract_evidence':{'text':text,'truncated':fault=='truncated'},
        'abstract_relevance':relevance},ref,SimpleNamespace(text=statement,reference_ids=['a']))
    assert member_tone(member) == ('attention' if not fault else 'limited_evidence')
    assert member['best_evidence']['display_text'] == text
