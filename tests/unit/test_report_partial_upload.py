from copy import deepcopy
import pytest
from bs4 import BeautifulSoup
from app.services.report_layers import partial_relevance
from app.services.evidence_report import _render_upload_priorities, _render_panel_template, render_evidence_report_html, _build_report_summary


def member(coverage='full_text'):
    return dict(reference_id='r',coverage_level=coverage,relevance_status='connected',
        source=dict(author='Author',year='2020',title='Title',raw_reference='Author (2020). Title. Publisher.',url='https://example.org/book'),
        best_evidence=dict(text='Retained passage.',relevance='partially_relevant',evidence_note='This passage addresses only part of the citation.'))


@pytest.mark.parametrize('coverage,expected',[('full_text',False),('partial_text',False),('abstract_only',False),('unavailable',False)])
def test_full_text_only_partial_marks_and_counts(coverage,expected):
    m=member(coverage);before=deepcopy(m)
    box=dict(page_index=0,x0=30,y0=50,x1=250,y1=70)
    citation=dict(student_text='Statement.',members=[m],paper_location=dict(rectangles=[box],localization_level='exact_rectangle'))
    view=dict(title='Example',citation_format='APA',citations=[citation],paper_surface=dict(page_href_template='/page/{page_index}',page_dimensions=[dict(page_index=0,width=600,height=800)]))
    html=render_evidence_report_html(view,csp_nonce='partial-relevance-test-nonce')
    assert bool(BeautifulSoup(html,'html.parser').select('rect.partial-relevance-outline')) is expected
    assert partial_relevance(m) is expected
    assert m==before
    summary=_build_report_summary(citations=[citation],overview={},pervasive_hanging_indent=False)
    assert ('only part' in ' '.join(summary['evidence'])) is expected
    m['relevance_status']='not_assessed';assert not partial_relevance(m)
    m['relevance_status']='connected';m['identity_status']='possible_match';assert not partial_relevance(m)


def test_upload_full_reference_link_and_deliberate_target():
    citation=dict(members=[member('abstract_only')],upload_action=dict(enabled=True,href='/report/id/citation/c/source/upload'))
    soup=BeautifulSoup(_render_upload_priorities([citation]),'html.parser')
    assert soup.li.get_text().startswith('1 citation – ') and 'Publisher.' in soup.li.get_text()
    assert soup.li.a['href']=='https://example.org/book'
    assert soup.button.text=='Upload Sources'
    assert soup.select_one('select') is None
    assert soup.form['action']=='/report/id/source/upload'
    citation.pop('upload_action')
    assert '<form' not in _render_upload_priorities([citation])


def test_media_heading_and_no_notice_or_upload():
    m=member('unavailable');m['source']['source_kind']='traditional_media';m['best_evidence']=None
    m['availability']='This is a media reference. Automated source retrieval and checks are not available.'
    html=_render_panel_template(dict(members=[m],upload_action=dict(enabled=True,href='/upload')),1)
    assert 'Media Reference - Cannot Retrieve</h3>' in html
    assert 'Automated source' not in html and '<form' not in html


def test_mixed_sources_only_flag_actual_full_text_partial():
    from app.services.report_layers import citation_partial_relevance
    full=member();full['best_evidence']['relevance']='relevant';full['best_evidence']['evidence_note']=''
    assert not citation_partial_relevance(dict(members=[full,member('abstract_only')]))
    assert not citation_partial_relevance(dict(members=[member(),member('abstract_only')]))


def test_reference_url_with_native_suffix_hyperlink_is_not_duplicated():
    from app.services.evidence_report import _render_formatted_reference
    url='https://example.org/long_book_title.pdf'
    raw='Author (2020). Title. '+url
    start=raw.index('book_title');end=len(raw)
    source=dict(raw_reference=raw,title='Title',year='2020',
        submitted_hyperlink_labels=[dict(start=start,end=end,label=raw[start:end],href=url)])
    soup=BeautifulSoup(_render_formatted_reference(source,raw),'html.parser')
    assert soup.get_text()==raw
    assert len(soup.select('a'))==1 and soup.a['href']==url
