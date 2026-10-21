from bs4 import BeautifulSoup
from app.services.evidence_report import _render_reference_panel_template,_render_formatted_reference,_render_continuous_paper
from app.services.paper_extraction import _build_marker_census
from app.services.citation_extractor import extract_citations
from app.services.schemas import ParsedReference


def test_standalone_and_combined_formatting_categories_match():
    finding={'finding_type':'duplicate_citation_key','finding':'These references share an author and year.',
             'source':{'author':'Writer','year':'2020','title':'Work','raw_reference':'Writer. Work.'}}
    for combined in (False,True):
        h=_render_reference_panel_template(finding,1,combined=combined)
        assert 'class="issue-heading formatting"' in h and 'Citation and Reference Formatting' in h
        assert 'Reference Practice' not in h
    finding.update(finding_type='missing_reference_entry')
    assert 'class="issue-heading academic"' in _render_reference_panel_template(finding,1)


def source(label):
    raw='Writer, W. (2020). A book. '+label
    return {'author':'Writer','year':'2020','title':'A book','raw_reference':raw,
            'submitted_hyperlinks':['https://example.org/book'],
            'submitted_hyperlink_labels':[{'start':raw.index(label),'end':len(raw),'label':label,'href':'https://example.org/book'}]}


def test_trailing_duplicate_label_replaced_by_url_but_metadata_preserved():
    s=source('Read online Read online')
    h=_render_formatted_reference(s,'')
    assert 'Read online' not in h and 'A book.' in h
    assert BeautifulSoup(h,'html.parser').get_text().count('https://example.org/book')==1
    assert s['raw_reference'].endswith('Read online Read online')


def test_linked_bibliographic_title_is_not_removed_and_stale_spans_abstain():
    s=source('Read online')
    start=s['raw_reference'].index('A book')
    s['submitted_hyperlink_labels']=[{'start':start,'end':start+6,'label':'A book','href':'https://example.org/book'}]
    assert 'A book.' in _render_formatted_reference(s,'')
    s['submitted_hyperlink_labels'][0]['label']='stale'
    assert 'Read online' in _render_formatted_reference(s,'')


def test_historical_period_range_is_not_an_extra_source_but_real_ranges_remain():
    refs=[ParsedReference(reference_id='r',author='Writer',year='2020',title='A book',raw_ref='Writer (2020). A book.')]
    text='During the Classical period (1927-1954), production changed (Writer, 2020).'
    census=_build_marker_census(extract_citations(text,refs,'apa'),text,refs)
    assert [c.text for c in census]==['(Writer, 2020)']
    # Owner decision 2026-10-01: a date range in parentheses is an aside, not a citation.
    text='An uncertain mention (1927-1954).'
    assert not _build_marker_census([],text,refs)
    ref=ParsedReference(reference_id='range',author='Period',year='1927-1954',title='A serial',raw_ref='Period (1927-1954). A serial.')
    text='Period (1927-1954) reported events.'
    assert any(c.reference_ids==['range'] for c in _build_marker_census(extract_citations(text,[ref],'apa'),text,[ref]))


def test_diamond_and_entry_open_the_reference_window_without_a_paper_link():
    r={'page_index':0,'x0':50,'y0':50,'x1':250,'y1':65}
    surface={'page_dimensions':[{'page_index':0,'width':600,'height':800}],'page_href_template':'p-{page_index}',
             'reference_locations':{'r':{'index':0,'rectangles':[r],'links':[{'href':'https://example.org/book','rectangle':r}]}}}
    finding={'finding_type':'submitted_link_issue','reference_id':'r','rectangles':[r]}
    catalog=[{'reference_id':'r','number':1,'template_id':'reference-entry-panel-1',
              'location':surface['reference_locations']['r']}]
    html,_=_render_continuous_paper(surface,[],[finding],numbers={'r':1},catalog=catalog)
    soup=BeautifulSoup(html,'html.parser')
    assert len(soup.select('path.submitted-link-marker'))==1 and not soup.select('line.submitted-link-marker')
    # DOIs/URLs on the paper are not links; the Reference window lists them.
    assert not soup.select('a.paper-reference-link') and 'https://example.org/book' not in html
    assert soup.select_one('.submitted-link-marker').find_parent('a')['data-panel-template']=='reference-entry-panel-1'
    entry=soup.select_one('a.reference-entry-overlay')
    assert entry['data-panel-template']=='reference-entry-panel-1'
    assert entry.select_one('rect.reference-entry-hit')['width']=='200.000'


def test_an_uncited_reference_is_marked_yellow_over_its_whole_entry():
    r={'page_index':0,'x0':50,'y0':50,'x1':250,'y1':65}
    surface={'page_dimensions':[{'page_index':0,'width':600,'height':800}],'page_href_template':'p-{page_index}',
             'reference_locations':{'r':{'index':0,'rectangles':[r]}}}
    catalog=[{'reference_id':'r','number':1,'template_id':'reference-entry-panel-1',
              'location':surface['reference_locations']['r']}]
    html,_=_render_continuous_paper(surface,[],[],numbers={'r':1},catalog=catalog,uncited_reference_ids=['r'])
    mark=BeautifulSoup(html,'html.parser').select_one('rect.uncited-highlight')
    assert mark is not None and mark['width']=='200.000'
    assert mark.find_parent('a')['data-panel-template']=='reference-entry-panel-1'
    cited,_=_render_continuous_paper(surface,[],[],numbers={'r':1},catalog=catalog,uncited_reference_ids=[])
    assert 'uncited-highlight' not in cited
    from app.services.highlight_priority import prioritize_svg_highlights
    both = prioritize_svg_highlights('<rect class="uncited-highlight" x="0" y="0" width="100" height="10"/>'
                                     '<rect class="reference-formatting-hit" x="40" y="0" width="20" height="10"/>')
    # The finding's own orange mark stays whole over the yellow.
    assert 'class="reference-formatting-hit" x="40" y="0" width="20" height="10"' in both
