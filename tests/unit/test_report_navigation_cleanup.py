import hashlib
from copy import deepcopy

import fitz
from bs4 import BeautifulSoup

from app.services.citation_extractor import _extract_attributed_text_and_index
from app.services.sentence_splitter import split_sentences
from app.services.evidence_report import project_reference_flags, _render_reference_panel_template


def test_secondary_question_marker_keeps_exact_sentence_without_following_claim():
    first = 'An earlier statement.'
    sentence = 'The speaker asked, Why this portrayal? (as cited in Writer, 2020).'
    text = first + ' ' + sentence + ' A later claim.'
    marker = '(as cited in Writer, 2020)'
    start = text.index(marker)
    found, _, a, b = _extract_attributed_text_and_index(text, start, start+len(marker), split_sentences(text), 'parenthetical')
    assert found == sentence == text[a:b]
    assert len(split_sentences('Why?\n\n(as cited in Writer, 2020).')) == 2
    assert len(split_sentences('Why? (An independent aside.)')) == 2


def test_saved_marker_navigation_recovers_context_without_rewriting_claim():
    marker = '(as cited in Writer, 2020).'
    sentence = 'The speaker asked, Why this portrayal? ' + marker
    view = {'citations':[{'student_text':marker, 'display_student_text':marker, 'members':[],
                         'paper_location':{'rectangles':[{'page_index':0,'x0':400,'y0':50,'x1':500,'y1':70}]}}],
            'paper_surface':{}}
    before = deepcopy(view)
    with fitz.open() as doc:
        doc.new_page().insert_text((50,70), sentence, fontsize=9)
        result = project_reference_flags(view, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    citation = result['citations'][0]
    assert citation['student_text'] == marker
    assert citation['display_student_text'] == sentence
    assert citation['paper_location']['rectangles'][0]['x0'] == 50
    assert citation['display_context_provenance']['presentation_sha256']
    assert view == before


def test_locator_orange_marks_marker_not_quotation_and_priorities_locate_reference():
    text = 'Writer (2020) states: "The first sentence. The second sentence."'
    source = {'author':'Writer','year':'2020','title':'A book','raw_reference':'Writer, W. (2020). A book. Publisher.'}
    finding = {'finding_type':'required_quotation_locator_missing','source':source,'reference_id':'r',
               'citation_text':text,'quote_text':'The first sentence. The second sentence.',
               'finding':'Quotation lacks a locator.', 'rule_source':'https://apastyle.apa.org/style',
               'field_difference':{'submitted_value':'The first sentence. The second sentence.'}}
    view = {'paper_surface':{},'reference_practice':[finding], 'citations':[{'student_text':text,
            'members':[{'reference_id':'r','source':source,'coverage_level':'unavailable'}]}]}
    with fitz.open() as doc:
        page=doc.new_page();page.insert_text((50,70),text,fontsize=9)
        page.insert_text((50,170),source['raw_reference'],fontsize=9)
        result=project_reference_flags(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
    box=result['reference_practice'][0]['rectangles'][0]
    assert box['x1']-box['x0'] < 30
    assert result['paper_surface']['reference_locations']['r']['rectangles'][0]['y0'] > 150
    full=BeautifulSoup(_render_reference_panel_template(finding,1),'html.parser')
    combined=BeautifulSoup(_render_reference_panel_template(finding,1,combined=True),'html.parser')
    assert not full.select('a') and not combined.select('blockquote,.full-reference,a')
    assert 'This quotation lacks a page or paragraph locator.' in combined.select_one('template').get_text()
    assert 'does not verify' not in full.get_text()
