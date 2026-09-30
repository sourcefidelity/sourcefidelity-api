"""Range identity and source-bound bibliography advisories."""
import hashlib
from copy import deepcopy

import fitz
import pytest

from app.services.citation_extractor import extract_citations
from app.services.ref_field_extractor import extract_fields_apa
from app.services.evidence_report import project_reference_flags, _render_reference_panel_template, _render_continuous_paper
from tests.unit.test_report_layers import mismatch
from app.services.report_member_navigation import marker_words, member_targets
from app.services.parsers.apa_parser import ApaParser
from app.services.report_export import _render_pdf
from app.services.evidence_report import _render_how_to_read


def test_reading_guide_explains_purple_diamond_under_academic_practice():
    from bs4 import BeautifulSoup
    guide=BeautifulSoup(_render_how_to_read({}),'html.parser')
    heading=next(h for h in guide.select('details h2') if h.get_text()=='Poor academic practice')
    assert 'A purple diamond marks an issue with a submitted link or DOI' in heading.find_next_sibling('p').get_text()


def test_how_to_read_is_the_owners_revised_text_in_order():
    from bs4 import BeautifulSoup
    guide=BeautifulSoup(_render_how_to_read({}),'html.parser')
    assert [h.get_text() for h in guide.select('details h2')]==[
        'Source identification, verification and retrieval','Poor academic practice',
        'Citation and reference format checking','Source use judgment']
    # The same text is shown once in the dialog before the report.
    assert [h.get_text() for h in guide.select('dialog h2')]==[h.get_text() for h in guide.select('details h2')]
    assert guide.select_one('dialog [data-close-guide]').get_text()=='Close'
    assert 'Blue marks full text' not in guide.get_text() and 'Judgment layout' not in guide.get_text()


@pytest.mark.parametrize('dash', ['-', '–', '—'])
def test_serial_year_range_is_preserved_and_linked(dash):
    ref = extract_fields_apa('Screen Review. (1930–1931). Annual collection. Sample Press.')
    assert ref.year == '1930–1931'
    citations = extract_citations(f'The photograph depicts the actor (Screen Review, 1930{dash}1931).',
        [ref], format_hint='apa', use_llm_boundaries=False)
    assert citations[0].reference_ids == [ref.reference_id]
    assert citations[0].citation_marker.endswith(f'1930{dash}1931)')


def test_serial_range_does_not_match_single_year_or_different_range():
    refs = [extract_fields_apa(f'Screen Review. ({year}). Annual collection. Sample Press.')
            for year in ['1930', '1931–1932']]
    citation = extract_citations('The photograph depicts the actor (Screen Review, 1930–1931).',
        refs, format_hint='apa', use_llm_boundaries=False)[0]
    assert not citation.reference_ids


def test_range_references_split_and_duplicate_range_remains_ambiguous():
    lines=['Screen Review. (1930–1931). First collection. Sample Press.',
           'Screen Review. (1930–1931). Second collection. Sample Press.']
    assert ApaParser.split_references('\n'.join(lines)) == lines
    refs=[extract_fields_apa(line) for line in lines]
    c=extract_citations('A claim (Screen Review, 1930–1931).',refs,
        format_hint='apa',use_llm_boundaries=False)[0]
    assert not c.reference_ids


def test_joined_next_sentence_does_not_hide_source_marker():
    with fitz.open() as doc:
        page=doc.new_page()
        page.insert_text((72,72),'A claim (Screen Review, 1930-1931).This follows.')
        citation={'citation_marker':'(Screen Review, 1930-1931)',
            'members':[{'source':{'author':'Screen Review','year':'1930-1931'}}],
            'paper_location':{'rectangles':[{'page_index':0,'x0':60,'y0':50,'x1':500,'y1':90}]}}
        targets=member_targets(citation,marker_words(doc))
        assert len(targets)==1
        following=page.search_for('This')[0]
        assert targets[0]['x1']<=following.x0


def test_bibliography_outline_reuses_bound_advisory_and_exact_geometry():
    member, citation = mismatch()
    raw = 'Writer, W. (2020). Fish research. Sample Press.'
    member.update(reference_id='ref',source={'raw_reference':raw,'author':'Writer, W.','year':'2020','title':'Fish research'})
    citation.update(members=[member])
    view = {'citations':[citation], 'reference_practice':[]}
    with fitz.open() as doc:
        doc.new_page().insert_text((72,72),raw)
        result = project_reference_flags(view, doc, hashlib.sha256(doc.tobytes()).hexdigest())
        finding = result['reference_practice'][0]
        assert finding['finding_type'] == 'source_topical_mismatch'
        assert finding['rectangles'] and not view['reference_practice']
        panel = _render_reference_panel_template(finding,1)
        assert 'possible topical mismatch' in panel and 'fish' in panel and 'birds' in panel
        html,_ = _render_continuous_paper({'page_dimensions':[{'page_index':0,'width':612,'height':792}],
            'page_href_template':'page-{page_index}.png'},[],reference_practice=[finding])
        assert 'reference-field-marker layer-mark mark-relevance' in html
        assert '<circle' not in html
        output,_=_render_pdf(doc.tobytes(),citations=[],reference_practice=[finding],export_binding='synthetic')
        with fitz.open(stream=output,filetype='pdf') as rendered:
            assert any(d['fill'] and all(abs(a-b)<.002 for a,b in zip(d['fill'],(.937,.510,.729)))
                       for d in rendered[0].get_drawings())
            assert 'Potential topical mismatch' in ''.join(p.get_text() for p in rendered)
        broken = deepcopy(result)
        broken['citations'][0]['student_text'] += ' Changed.'
        assert not project_reference_flags(broken,doc,'hash')['reference_practice']
