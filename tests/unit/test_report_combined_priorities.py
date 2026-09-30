from copy import deepcopy
from bs4 import BeautifulSoup
from app.services.evidence_report import _upload_priorities,_render_upload_priorities,_formatting_overlaps,_render_continuous_paper,render_evidence_report_html


def member(rid,coverage='unavailable',kind='monograph'):
    return {'reference_id':rid,'coverage_level':coverage,'source':{'author':rid,'year':'2020','title':'A source','raw_reference':rid+'. A source.','source_kind':kind}}


def test_priority_counts_unique_citations_and_excludes_full_text_media():
    citations=[{'members':[member('A'),member('A'),member('B','abstract_only'),member('film',kind='traditional_media')]},
        {'members':[member('A'),member('C','partial_text')]},
        {'members':[member('D'),member('B','full_text')]},
        {'members':[member('E')]}]
    before=deepcopy(citations)
    rows=_upload_priorities(citations)
    assert [r['member']['reference_id'] for r in rows]==['A','C','D']
    assert len(rows[0]['citations'])==2
    assert citations==before
    html=_render_upload_priorities(citations, {'A':{'index':0}})
    # Owner layout 2026-09-28: the citation count first, then the reference.
    assert '<li>2 citations – A. A source.</li>' in html and '<li>1 citation – C. A source.</li>' in html
    assert 'upload-intro' not in html     # no Judgment in this rendering
    judged=_render_upload_priorities(citations, judgments=14)
    # Owner wording 2026-09-29: judgments; A is cited twice, C and D once each.
    assert ('This paper has <span data-judgment-count>14 judgments</span>. If these 3 sources are '
            'uploaded, the paper will have 4 more judgments.') in judged
    assert 'href="#reference-location-0"' not in html
    assert 'data-panel-template' not in html and 'data-member-index' not in html
    assert not _render_upload_priorities([{'members':[member('A','full_text')]}])


def test_overlap_combines_and_reference_only_stays_separate():
    box={'page_index':0,'x0':50,'y0':50,'x1':300,'y1':65}
    citation={'student_text':'A statement (Writer, 2020).','members':[], 'paper_location':{'rectangles':[box]}}
    issue={'finding_type':'reference_title_style','finding':'Title lacks italics.',
           'source':{'author':'Writer','year':'2020','title':'Title','raw_reference':'Writer. Title.'},
           'rule_source':'https://example.org/style','rectangles':[box]}
    surface={'page_dimensions':[{'page_index':0,'width':600,'height':800}],'page_href_template':'p-{page_index}'}
    view={'title':'Report','citation_format':'APA','citations':[citation],'reference_practice':[issue],'paper_surface':surface}
    html=render_evidence_report_html(view,csp_nonce='combined-panel-test-nonce')
    soup=BeautifulSoup(html,'html.parser')
    assert 'Title lacks italics.' in soup.select_one('#citation-panel-1').get_text()
    assert not soup.select_one('#citation-panel-1').select('.full-reference')
    assert not soup.select('template a[href="https://example.org/style"]')
    hit=soup.select_one('.reference-formatting-hit')
    assert hit.find_parent('a')['data-panel-template']=='citation-panel-1'
    assert float(hit['height'])==15
    issue['rectangles']=[{**box,'y0':100,'y1':115}]
    assert not _formatting_overlaps(citation,issue)
    html=render_evidence_report_html(view,csp_nonce='combined-panel-test-nonce')
    soup=BeautifulSoup(html,'html.parser')
    assert 'Title lacks italics.' not in soup.select_one('#citation-panel-1').get_text()
    assert soup.select_one('.reference-formatting-hit').find_parent('a')['data-panel-template']=='reference-panel-1'
    issue['finding_type']='submitted_link_issue';issue['rectangles']=[box]
    assert not _formatting_overlaps(citation,issue)


def test_priority_section_is_immediately_after_summaries():
    view={'title':'Report','citation_format':'APA','citations':[{'student_text':'Claim','members':[member('A')]}],
          'paper_surface':{},'summary':{'evidence':['One issue.']}}
    soup=BeautifulSoup(render_evidence_report_html(view,csp_nonce='priority-section-test-nonce'),'html.parser')
    section=soup.select_one('.upload-priorities')
    assert 'summary' in ' '.join(section.find_previous_sibling().get('class',[]))
    assert section.find_next_sibling().get('class')==['read-guide']
