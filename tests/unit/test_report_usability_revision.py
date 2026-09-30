from app.services.evidence_report import summary_text
import base64
import hashlib
from copy import deepcopy

import fitz
import pytest
from bs4 import BeautifulSoup

from test_report_export import export_store
from app.services.evidence_report import _render_panel_template
from app.services.interactive_report_export import build_interactive_report_html
from app.services.report_member_navigation import missing_reference_targets, marker_words
from app.services.report_export import build_released_report_export


def test_multiline_field_has_a_highlight_on_each_affected_line():
    from app.services.evidence_report import _render_continuous_paper
    rectangles=[dict(page_index=0,x0=50,y0=80,x1=300,y1=95),
                dict(page_index=0,x0=50,y0=100,x1=180,y1=115)]
    surface={'page_dimensions':[dict(page_index=0,width=600,height=800)],'page_href_template':'page-{page_index}'}
    html,_=_render_continuous_paper(surface,[],[dict(finding_type='reference_title_style',rectangles=rectangles)])
    soup=BeautifulSoup(html,'html.parser');markers=soup.select('.reference-formatting-hit')
    assert len(markers)==2 and all(m.name=='rect' for m in markers)
    assert [(float(m['x']),float(m['width']),float(m['y'])) for m in markers]==[(50,250,80),(50,130,100)]
    assert not soup.select('line.reference-field-marker')


def test_missing_source_summary_counts_sources_not_occurrences():
    from app.services.evidence_report import _build_report_summary
    cs=[{'members':[],'missing_reference_members':[name]} for name in ['Alpha, 2020','Alpha, 2020','Beta, 2021']]
    text=' '.join(map(summary_text, _build_report_summary(citations=cs,overview={},pervasive_hanging_indent=False)['academic_practice']))
    assert '2 in-text sources have' in text and text.count('Alpha, 2020')==1 and 'citation 1' not in text


def test_character_geometry_recovers_joined_words_but_not_duplicate_text():
    from app.services.evidence_report import project_reference_flags
    text='This exact sentence names the source (Smith, 2020).'
    doc=fitz.open();p=doc.new_page();p.insert_text((40,80),text+'Another sentence.')
    view={'citations':[{'student_text':text,'members':[],'paper_location':{}}]}
    result=project_reference_flags(view,doc,'a'*64)
    assert result['citations'][0]['paper_location']['rectangles']
    assert not view['citations'][0]['paper_location']
    p.insert_text((40,110),text)
    assert not project_reference_flags(view,doc,'a'*64)['citations'][0]['paper_location'].get('rectangles')


def test_missing_reference_yellow_panel_has_no_revision_instruction():
    citation={'student_text':'Words (Parsons, 1946).','members':[],
        'missing_reference_members':['Parsons, 1946'],
        'boundary_reason':'No reference-list entry was found for Parsons, 1946. Add the missing reference or correct the citation.'}
    html=_render_panel_template(citation,5)
    assert 'practice-notice' in html and 'No reference-list entry' in html
    assert 'Add the missing reference' not in html


def test_missing_reference_exact_marker_not_full_sentence():
    doc=fitz.open();p=doc.new_page();p.insert_text((72,72),'The account differs (Parsons, 1946).')
    c={'citation_marker':'(Parsons, 1946)','missing_reference_members':['Parsons, 1946'],
       'paper_location':{'rectangles':[dict(page_index=0,x0=70,y0=50,x1=500,y1=90)]}}
    targets=missing_reference_targets(c,marker_words(doc))
    assert targets
    for t in targets:
        text=p.get_textbox(fitz.Rect(*(t[k] for k in ['x0','y0','x1','y1'])))
        assert 'account' not in text
    assert missing_reference_targets({**c,'missing_reference_members':[]},marker_words(doc))==[]


def test_offline_report_embeds_exact_pdf_and_no_server_dependencies(export_store):
    _,_,_,_,view,paper=export_store
    view=deepcopy(view);view['reference_practice']=[]
    view['citations'][0]['members'][0]['reference_id']='ref-0001-000000000001'
    view['paper_surface']['presentation_sha256']=hashlib.sha256(paper).hexdigest()
    original=deepcopy(view)
    html=build_interactive_report_html(view,paper)
    soup=BeautifulSoup(html,'html.parser')
    assert 'Student report' not in soup.get_text() and 'Instructor report' not in soup.get_text()
    assert 'StudentInstructor' not in soup.get_text()
    assert view==original
    images=soup.select('image');assert len(images)==1
    assert images[0]['href'].startswith('data:image/png;base64,')
    assert not soup.find('a',download='student-paper.pdf')
    assert soup.find('meta',attrs={'name':'paper-sha256'})['content']==hashlib.sha256(paper).hexdigest()
    assert 'Report Counts and Evidence Breakdown' not in soup.get_text()
    assert not soup.select('.counts-disclosure,.locator-count')
    assert not soup.select('form, #pen-tool, #add-comment, #add-highlight, #judgment-layout')
    assert not any(a.get('href','').startswith(('/', '?')) for a in soup.find_all('a'))
    assert "connect-src 'none'" in soup.find('meta',attrs={'http-equiv':'Content-Security-Policy'})['content']
    assert soup.select('#evidence-panel, #zoom-in, #select-text')
    # The export shows How to read first, every time it is opened (owner request 2026-09-28).
    assert soup.select_one('dialog#how-to-read-dialog [data-close-guide]')
    assert 'const remember=false' in html.decode()
    # The redesigned workspace, reference windows and summaries survive export.
    assert soup.select_one('#report-layout > .workspace-bar .paper-toolbar #select-text')
    assert soup.select_one('#report-layout > .side-pane #evidence-panel')
    assert not soup.select('a.paper-reference-link')
    assert all(a['href'].startswith('#') and a.get('data-go-to') for a in soup.select('a.summary-instance'))
    assert soup.select('template[id^="reference-entry-panel-"]')
    for template in soup.select('template[id^="reference-entry-panel-"]'):
        inner = BeautifulSoup(template.decode_contents(), 'html.parser')
        assert not inner.select('form')                     # uploads need the server
        assert all(a.get('target') == '_blank' for a in inner.select('a[href^="http"]'))
    # Processing and cost details are instructor-only.
    # Technical details are in the one report for now (owner decision 2026-09-25).
    assert soup.select('details.technical-details')


def test_offline_export_rejects_wrong_pdf(export_store):
    _,_,_,_,view,paper=export_store
    with pytest.raises(ValueError,match='hash mismatch'):
        build_interactive_report_html(view,paper)


@pytest.mark.parametrize('href,expected',[
    ('https://example.org/article.pdf',True),
    ('/report/private/source/id',False),
    ('https://example.org/article.pdf?token=secret',False),
])
def test_portable_report_preserves_only_public_verified_source_access(export_store,href,expected):
    _,_,_,_,view,paper=export_store
    view=deepcopy(view);view['reference_practice']=[]
    view['paper_surface']['presentation_sha256']=hashlib.sha256(paper).hexdigest()
    member=view['citations'][0]['members'][0]
    member['coverage_level']='partial_text'
    member['source_action']=dict(enabled=True,status='verified_public_source_available',
        href=href,label='Open available text')
    html=build_interactive_report_html(view,paper)
    soup=BeautifulSoup(html,'html.parser')
    assert bool(soup.find('a',href=href)) is expected
    assert not soup.select('form')
