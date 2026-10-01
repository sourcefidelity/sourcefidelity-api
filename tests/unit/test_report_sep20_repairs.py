from bs4 import BeautifulSoup
from app.services.schemas import ParsedReference
from app.services.citation_extractor import extract_citations, _extract_attributed_text_and_index
from app.services.sentence_splitter import split_sentences
from app.services.paper_extraction import _build_marker_census
from app.services.reference_credibility import credibility_records_html
from app.services.evidence_report import _render_panel_template


def test_calendar_aside_is_not_a_missing_reference_marker():
    assert not _build_marker_census([], 'Current information (May 4, 2025).', [])
    assert _build_marker_census([], 'A claim (Writer, 2025).', [])


def test_joined_narrative_with_space_inside_parentheses_links_exact_reference():
    ref=ParsedReference(reference_id='r',author='Boylorn, R. M.',year='2021',title='An article')
    text='Boylorn( 2021)emphasizes the value of storytelling.'
    rows=extract_citations(text,[ref],format_hint='apa',use_llm_boundaries=False)
    assert len(rows)==1 and rows[0].reference_ids==['r']
    assert rows[0].citation_marker=='Boylorn( 2021)'
    assert rows[0].text==text


def test_spelled_out_coauthor_list_links_as_a_unit_not_last_surname():
    ref=ParsedReference(reference_id='r',author='Stein, L., Jenkins, H., Ford, S., & Green, J.',year='2013',title='Spreadable Media')
    text='Henry Jenkins, Sam Ford, and Joshua Green (2013) describe circulation.'
    rows=extract_citations(text,[ref],format_hint='apa',use_llm_boundaries=False)
    assert len(rows)==1 and rows[0].reference_ids==['r']
    assert rows[0].citation_marker.startswith('Henry Jenkins')
    duplicate=ref.model_copy(update={'reference_id':'other'})
    ambiguous=extract_citations(text,[ref,duplicate],format_hint='apa',use_llm_boundaries=False)
    assert not any(c.reference_ids for c in ambiguous)
    assert len(ambiguous)==1 and ambiguous[0].citation_marker.startswith('Henry Jenkins')
    assert set(ambiguous[0].candidate_reference_ids)=={'r','other'}
    different_year=ref.model_copy(update={'year':'2014'})
    unresolved=extract_citations(text,[different_year],format_hint='apa',use_llm_boundaries=False)
    assert len(unresolved)==1 and not unresolved[0].reference_ids
    assert unresolved[0].candidate_reference_ids==['r'] and unresolved[0].link_status=='ambiguous'
    assert unresolved[0].citation_marker=='Henry Jenkins, Sam Ford, and Joshua Green (2013)'


def test_unmatched_coordinated_author_marker_does_not_become_only_last_author():
    ref=ParsedReference(reference_id='r',author='Someone, A.',year='2020',title='Another work')
    text='Morris, Taylor, and Evans (2020) describe circulation.'
    rows=extract_citations(text,[ref],format_hint='apa',use_llm_boundaries=False)
    assert len(rows)==1 and rows[0].citation_marker=='Morris, Taylor, and Evans (2020)'
    assert rows[0].link_status=='missing_reference' and not rows[0].reference_ids


def test_narrative_cannot_extend_back_to_unclosed_earlier_quote():
    text='Earlier discussion contains “an unclosed phrase. Another sentence. According to Vella (2015), characters emerge from stories.'
    marker='Vella (2015)';start=text.index(marker)
    found,_,left,right=_extract_attributed_text_and_index(text,start,start+len(marker),split_sentences(text),'narrative')
    assert found.startswith('According to') and found==text[left:right]


def test_fabrication_search_details_collapsed_without_similar_work_list():
    html=credibility_records_html(dict(finding_type='potentially_fabricated_reference',
        evidence_explanation='Title searches completed.',records=[{'observed':{'title':'Unrelated title'}}]))
    soup=BeautifulSoup(html,'html.parser')
    assert soup.select_one('details summary').text=='Search Details'
    assert 'Unrelated title' not in html


def test_unverified_member_heading_uses_soft_red_without_claiming_retrieval():
    member=dict(reference_id='r',unverified=True,coverage_level='unavailable',
        availability='Source Not Retrieved',source=dict(author='Writer',year='2020',title='Title',raw_reference='Writer. Title.'))
    html=_render_panel_template(dict(student_text='A claim.',members=[member]),1)
    soup=BeautifulSoup(html,'html.parser')
    assert soup.select_one('h3 mark.unverified').string=='Cannot be verified'
    assert 'No Text Retrieved' in str(soup.select_one('h3'))


def test_download_listing_does_not_pass_as_the_listed_book():
    from app.services.source_validator import _detect_nonwork_listing
    text='A book. Author. ISBN 9780000000000.\nDOWNLOAD\nhttps://example.org/1\nDOWNLOAD\nhttps://example.org/2\nhttps://example.org/3\n'
    text+='\n'.join(f'Other book {i}, Author, 2005, Economics, 400 pages.' for i in range(4))
    assert _detect_nonwork_listing(text, 'A book')=='a download/catalog listing'
    assert _detect_nonwork_listing(text.replace('DOWNLOAD','Chapter'), 'A book') is None
    assert _detect_nonwork_listing('This study discusses download catalogs and online piracy.', 'A study') is None
