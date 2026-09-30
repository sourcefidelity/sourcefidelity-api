from app.services.evidence_report import summary_text
"""User-facing report controls and exact marker geometry regressions."""
from copy import deepcopy
import fitz

from app.services.report_member_navigation import marker_words, member_targets
from app.services.evidence_report import render_evidence_report_html, _render_citation_information, _panel_statement, _build_report_summary


def test_joined_quote_marker_uses_exact_pdf_glyphs_without_changing_native_words():
    with fitz.open() as doc:
        page=doc.new_page()
        page.insert_text((72,72), 'A "quoted passage."(Kobal, 1982, p. 56).')
        native=page.get_text('words', sort=True)
        citation={'citation_marker':'(Kobal, 1982, p. 56)', 'members':[{'coverage_level':'unavailable', 'source':{'author':'Kobal, J','year':'1982'}}],
                  'paper_location':{'rectangles':[{'page_index':0,'x0':70,'y0':50,'x1':550,'y1':90}]}}
        assert not member_targets(citation, {0:native})
        projected=marker_words(doc)
        targets=member_targets(citation, projected)
        assert len(targets)==1
        assert page.get_textbox(fitz.Rect(*(targets[0][k] for k in ('x0','y0','x1','y1')))).startswith('(Kobal,')
        assert native==page.get_text('words',sort=True)


def test_film_title_disambiguates_shared_year_without_linking_other_sources():
    with fitz.open() as doc:
        page=doc.new_page(); page.insert_text((72,72),'Gilda (1946) is discussed (Modern Screen, 1946).')
        citation={'citation_marker':'Gilda (1946); (Modern Screen, 1946)', 'members':[
            {'coverage_level':'unavailable','source':{'author':'Vidor, C. (Director)','year':'1946','title':'Gilda [Film]','source_kind':'traditional_media'}},
            {'coverage_level':'unavailable','source':{'author':'Modern Screen','year':'1946'}}],
            'paper_location':{'rectangles':[{'page_index':0,'x0':70,'y0':50,'x1':550,'y1':90}]}}
        original=deepcopy(citation)
        targets=member_targets(citation,marker_words(doc))
        assert {t['member_index'] for t in targets}=={0,1}
        assert targets[0]['tone']=='not_assessed'
        assert page.get_textbox(fitz.Rect(*(targets[0][k] for k in ('x0','y0','x1','y1'))))=='(1946)'
        assert citation==original


def test_zoom_and_select_text_are_the_only_paper_tools():
    view={'title':'Synthetic report','citation_format':'APA','citations':[],'paper_surface':{}}
    html=render_evidence_report_html(view,csp_nonce='controls-test-nonce')
    assert 'id="zoom-in"' in html and 'id="zoom-out"' in html and 'id="select-text"' in html
    for removed in ('select-region','redo-annotation','undo-annotation','add-comment','add-highlight','pen-tool'):
        assert f'id="{removed}"' not in html
    assert 'Select text or a citation, then choose' not in html


def test_empty_information_is_absent_and_nonempty_information_does_not_repeat_number():
    assert _render_citation_information({'members':[]},21)==''
    result=_render_citation_information({'members':[{'limitations':['Pagination is unavailable.']}]},21)
    assert 'Citation Information' in result and 'Citation 21' not in result


def test_panel_instructions_removed_but_issue_and_evidence_limitation_retained():
    assert _panel_statement("Two references share this author and year. Add distinguishing labels.")=="Two references share this author and year."
    assert _panel_statement('Source completeness is uncertain. Check manually.')==''


def test_one_summary_uses_one_neutral_sentence_without_coaching():
    summaries=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=True)
    assert summary_text(summaries['reference_formatting'][0]).strip().count('.')==1
    assert 'Apply the required' not in summary_text(summaries['reference_formatting'][0])
