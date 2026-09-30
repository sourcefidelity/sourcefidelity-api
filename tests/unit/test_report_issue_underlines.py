import hashlib
import fitz
from bs4 import BeautifulSoup
from app.services.evidence_report import project_reference_flags,render_evidence_report_html,_render_continuous_paper


def test_literal_title_is_not_confused_with_url_slug():
    title='A book about cinema'
    raw=f'Writer, W. (2020). {title}. Publisher. https://example.org/A_book_about_cinema.pdf'
    finding={'finding_type':'reference_title_style','finding':'Italicize the book title.',
        'source':{'raw_reference':raw,'author':'Writer','year':'2020','title':title},
        'field_difference':{'field_name':'title','submitted_value':title},'rule_source':'https://example.org/style'}
    view={'title':'Report','citation_format':'APA','citations':[],'paper_surface':{},'reference_practice':[finding]}
    with fitz.open() as doc:
        p=doc.new_page();p.insert_text((50,70),'Writer, W. (2020). '+title+'. Publisher.')
        p.insert_text((50,95),'https://example.org/A_book_about_cinema.pdf')
        result=project_reference_flags(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
        rectangles=result['reference_practice'][0]['rectangles']
        assert len(rectangles)==1 and rectangles[0]['y1']<80
        assert not finding.get('rectangles')
        rendered=render_evidence_report_html(result,csp_nonce='underline-test-nonce')
        assert 'Other reference issues' not in rendered


def test_large_link_diamond_has_separate_hit_area_outside_link_text():
    rect={'page_index':0,'x0':50,'y0':90,'x1':300,'y1':105}
    surface={'page_dimensions':[{'page_index':0,'width':600,'height':800}],'page_href_template':'p-{page_index}'}
    rendered,_=_render_continuous_paper(surface,[],[{'finding_type':'submitted_link_issue','rectangles':[rect]}])
    soup=BeautifulSoup(rendered,'html.parser')
    marker=soup.select_one('path.submitted-link-marker')
    assert marker and marker['d'] == 'M 310.000 91.500 L 316.000 97.500 L 310.000 103.500 L 304.000 97.500 Z'
    assert soup.select_one('rect.submitted-link-hit')['width'] == '16'
    assert float(soup.select_one('rect.submitted-link-hit')['x'])>rect['x1']
    assert not soup.select('line.submitted-link-marker,circle')
