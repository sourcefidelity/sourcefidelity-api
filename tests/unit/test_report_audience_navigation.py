from bs4 import BeautifulSoup
import fitz
import hashlib

from app.services.citation_extractor import extract_citations
from app.services.schemas import ParsedReference
from app.services.sentence_splitter import split_sentences
from app.services.evidence_report import (
    _build_report_summary, _render_reference_panel_template, _render_panel_template,
    _render_formatted_reference, project_reference_flags, render_evidence_report_html,
)


def test_one_report_keeps_every_issue_type_without_a_cap():
    findings = [{'finding_type':kind, 'reference_id':str(i), 'claim_id':str(i),
                 'observation_id':str(i), 'finding':'Incorrect formatting.', 'rectangles':[{'x0':0}]}
                for i,kind in enumerate(('reference_title_style','body_title_style','reference_order','required_quotation_locator_missing'))]
    summary=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=True,reference_practice=findings)
    assert len(summary['reference_formatting']) == 5


def test_combined_headings_are_explicit_without_repeated_source():
    issue={'source':{},'finding':'Title lacks italics.','finding_type':'reference_title_style'}
    panel=BeautifulSoup(_render_reference_panel_template(issue,1,combined=True),'html.parser')
    assert str(panel.select_one('strong').string) == 'Citation and Reference Formatting'
    assert not panel.select('.full-reference')
    citation={'student_text':'A claim (Writer, 2020).','members':[],
              'missing_reference_members':['Writer, 2020'],'boundary_reason':'No reference-list entry was found.'}
    panel=BeautifulSoup(_render_panel_template(citation,1),'html.parser')
    assert str(panel.select_one('strong').string) == 'Academic Practice'


def test_title_question_and_quoted_question_continue_without_changing_text():
    title='Later, A Film? (Studio Bros., 1938) also portrays a star.'
    quote='In an interview she asked, “Why this portrayal?” and stated, “I am weary” (Writer, 2020).'
    assert split_sentences(title)==[title]
    assert split_sentences(quote)==[quote]
    assert len(split_sentences('She asked, “Why?” Another sentence.'))==2


def test_explicit_corporate_credit_and_fullwidth_film_year_keep_identity_boundaries():
    refs=[ParsedReference(reference_id='film',author='Director, A.',year='1931',title='A Film [Film]',
                          raw_ref='Director, A. (1931). A Film [Film].',source_kind='traditional_media'),
          ParsedReference(reference_id='press',author='A Film? (Studio Bros.)',year='1938',title='Pressbook',
                          raw_ref='A Film? (Studio Bros.). (1938). Pressbook.')]
    cs=extract_citations('In A Film（1931） the actor appears. Later, A Film? (Studio Bros., 1938) also portrays a star.',refs,'apa')
    assert {rid for c in cs for rid in c.reference_ids} == {'film','press'}
    assert next(c for c in cs if c.reference_ids==['film']).citation_marker=='A Film（1931）'
    assert next(c for c in cs if c.reference_ids==['press']).text.startswith('Later,')
    duplicate=refs[1].model_copy(update={'reference_id':'other'})
    cs=extract_citations('A claim (Studio Bros., 1938).',refs+[duplicate],'apa')
    assert cs[0].link_status=='ambiguous' and not cs[0].reference_ids


def test_reference_pdf_hyperlink_is_bound_to_its_entry_and_rendered_as_url():
    raw='Writer, W. (2020). A book. Read online.'
    source={'raw_reference':raw,'author':'Writer','year':'2020','title':'A book'}
    view={'citations':[{'student_text':'A claim (Writer, 2020).','members':[
        {'reference_id':'r','source':source,'coverage_level':'unavailable'}]}],'paper_surface':{}}
    with fitz.open() as doc:
        p=doc.new_page();p.insert_text((50,100),raw)
        p.insert_link({'kind':fitz.LINK_URI,'from':p.search_for('Read online.')[0],'uri':'https://example.org/book'})
        p.insert_text((50,200),'Another source.');p.insert_link({'kind':fitz.LINK_URI,'from':p.search_for('Another source.')[0],'uri':'https://example.org/other'})
        doc=fitz.open(stream=doc.tobytes(),filetype='pdf')
        projected=project_reference_flags(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
    actual=projected['citations'][0]['members'][0]['source']
    html=_render_formatted_reference(actual,raw)
    assert 'href="https://example.org/book"' in html and '>https://example.org/book</a>' in html
    assert 'example.org/other' not in html
    assert 'submitted_hyperlinks' not in source
    bad=_render_formatted_reference({**source,'submitted_hyperlinks':['javascript:alert(1)','http://127.0.0.1/private']},raw)
    assert '<a' not in bad


def test_paper_controls_do_not_overlay_viewport_and_citation_heading_returns_to_anchor():
    box={'page_index':0,'x0':50,'y0':50,'x1':200,'y1':70}
    view={'title':'Report','citation_format':'APA','citations':[{'student_text':'A claim.', 'members':[],
          'paper_location':{'rectangles':[box]}}], 'paper_surface':{'page_dimensions':[
          {'page_index':0,'width':600,'height':800}],'page_href_template':'page-{page_index}'}}
    html=render_evidence_report_html(view,csp_nonce='audience-navigation-nonce')
    soup=BeautifulSoup(html,'html.parser')
    assert soup.select_one('#citation-panel-1 h2 a')['href']=='#citation-location-1'
    assert soup.select_one('#citation-location-1') is not None
    assert not soup.select_one('.paper-toolbar').find_parent(class_='paper-viewport')
    # One control bar spans the workspace above the paper and the window.
    bar = soup.select_one('#report-layout > .workspace-bar')
    assert bar.select_one('.paper-toolbar') and bar.select_one('.evidence-controls [data-report-step]')
    assert not bar.find_parent(class_='paper-viewport')
    assert '--controls-height' not in html and 'position:sticky;top:0' not in html
    assert soup.select_one('.paper>h2') is None
    # Two layouts (owner decision 2026-09-27); Judgment is off unless enabled.
    assert [b.text for b in soup.select('.evidence-controls > button')] == []   # one layout (2026-09-28)
    assert soup.select_one('#judgment-layout') is None
    assert 'Availability is not a correctness judgment' not in soup.select_one('#active-key').text
    assert not soup.select('a[href="?audience=student"], a[href="?audience=instructor"]')
    assert 'Student report' not in html and 'Instructor report' not in html
