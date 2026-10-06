"""Owner requests 2026-10-03 from the paper 5 review."""
from types import SimpleNamespace

from app.services.evidence_report import (
    _abstract_disclosure,
    _build_report_summary,
    _member_action_html,
    _mismatch_lines,
    _narrative_label,
)


def _citation(details, members=({'reference_id': 'ref-1'},)):
    return {'members': list(members), 'reference_mismatch_details': details,
            'reference_mismatch_members': [f"{d['label']}: {d['difference']}" for d in details]}


def test_the_summary_lists_citation_numbers_and_the_window_names_the_difference():
    sexton = {'label': 'Sexton, 2023', 'difference': 'reference year n.d.', 'reference_ids': ['ref-1']}
    citations = [{}, {}, {}, _citation([sexton]), _citation([{**sexton, 'label': 'Bond, 2020'}])]
    summary = _build_report_summary(citations=citations, overview={}, pervasive_hanging_indent=False)
    [row] = [r for r in summary['academic_practice'] if r['kind'] == 'citation_reference_mismatch']
    assert row['text'] == '2 in-text citations differ from their reference-list entries (citations 4, 5).'
    assert [i['number'] for i in row['instances']] == [4, 5]
    assert _mismatch_lines(citations[3], 0) == ['Citation differs from its reference: Sexton, 2023 vs. n.d.']


def test_a_difference_goes_to_its_own_source_or_else_the_first():
    row = {'label': 'Smith, 2020', 'difference': 'reference year 2019', 'reference_ids': ['ref-2']}
    citation = _citation([row], members=({'reference_id': 'ref-1'}, {'reference_id': 'ref-2'}))
    assert _mismatch_lines(citation, 0) == [] and len(_mismatch_lines(citation, 1)) == 1
    unbound = _citation([{**row, 'reference_ids': ['ref-9']}])
    assert len(_mismatch_lines(unbound, 0)) == 1
    stored = {'members': [{'reference_id': 'ref-1'}], 'reference_mismatch_members': ['Smith, 2020: reference year 2019']}
    assert _mismatch_lines(stored, 0) == ['Citation differs from its reference: Smith, 2020 vs. 2019']


def test_a_narrative_citation_is_named_with_the_words_before_its_year():
    marker = SimpleNamespace(text='(2016)', report_text='As Hartmn (2016) highlights, norms persist.')
    assert _narrative_label(marker) == 'Hartmn (2016)'
    assert _narrative_label(SimpleNamespace(text='(2016)', report_text='')) == '2016'


def test_retrieved_text_from_the_cited_link_still_gets_its_button():
    url = 'https://example.org/page'
    member = {'coverage_level': 'partial_text', 'source_action': {
        'enabled': True, 'status': 'verified_public_source_available', 'href': url, 'label': 'Open available text'}}
    assert 'Open available text' in _member_action_html(member, f'<a href="{url}">{url}</a>')
    landing = {**member, 'source_action': {**member['source_action'], 'status': 'cited_https_route_available'}}
    assert _member_action_html(landing) == ''


def test_an_abstract_is_collapsed_and_only_for_an_abstract_only_source():
    member = {'coverage_level': 'abstract_only', 'best_evidence': {'text': 'An abstract.', 'evidence_kind': 'abstract'}}
    html = _abstract_disclosure(member)
    assert html.startswith('<details class="abstract-disclosure"><summary>Abstract</summary>') and 'An abstract.' in html
    assert _abstract_disclosure({**member, 'coverage_level': 'full_text'}) == ''


def test_only_the_references_differing_text_is_shown():
    # Owner request 2026-10-03: "Hartmn (2016) vs. Hartman, 2010", no descriptors.
    langford = {'label': 'Hartmn (2016)', 'difference': 'reference author Hartman, reference year 2010',
                'differences': ['author_spelling:Hartman', 'year:2010'], 'reference_ids': ['ref-1']}
    assert _mismatch_lines(_citation([langford]), 0) == [
        'Citation differs from its reference: Hartmn (2016) vs. Hartman, 2010']
    bordwell = {'label': 'Bordwell et al., 1985, p. 316', 'difference': 'reference has one author',
                'differences': ['author_count:one author'], 'reference_ids': []}
    member = {'reference_id': 'ref-3', 'source': {'author': 'Bordwell, D'}}
    assert _mismatch_lines(_citation([bordwell], members=(member,)), 0) == [
        'Citation differs from its reference: Bordwell et al., 1985, p. 316 vs. Bordwell']
    stored = {**langford}
    del stored['differences']
    assert _mismatch_lines(_citation([stored]), 0)[0].endswith('vs. Hartman, 2010')
    from app.services.evidence_report import _reference_authors
    assert _reference_authors({'source': {'author': 'Smith, J., & Jones, K.'}}) == 'Smith & Jones'
    assert _reference_authors({'source': {'author': 'Bordwell, D., Staiger, J., & Thompson, K.'}}) == 'Bordwell et al.'



def test_a_heading_naming_a_work_is_not_a_citation():
    from app.services.evidence_report import heading_citation
    heading = {'student_text': 'Classical Hollywood: The Day the Earth Stood Still (Wise, 1951)',
               'citation_marker': '(Wise, 1951)'}
    assert heading_citation(heading)
    assert heading_citation({'student_text': 'Post-Classical Hollywood: Star Trek: The Motion Picture (Wise, 1979)',
                             'citation_marker': '(Wise, 1979)'})
    sentence = {'student_text': 'The film was released across 2,000 theatres (Bond, 2020).', 'citation_marker': '(Bond, 2020)'}
    no_stop = {'student_text': 'the studio controlled every stage of release (Bond, 2020)', 'citation_marker': '(Bond, 2020)'}
    assert not heading_citation(sentence) and not heading_citation(no_stop)
    assert heading_citation({'student_text': "Killer's Kiss (1955): Classical Hollywood Film Analysis",
                             'citation_marker': "Killer's Kiss (1955)"})


def test_a_confirmed_cited_web_page_is_whole_when_not_cut_off():
    # Paper 5 reference 13 (a 569-word event page) read as limited text, 2026-10-03.
    from app.services.web_completeness import confirmed_page_completeness
    page = 'The archive presents a double feature of science fiction films. ' * 40
    assert confirmed_page_completeness(page, 'webpage')['verdict'] == 'complete'
    assert confirmed_page_completeness(page + ' Continue reading', 'webpage')['verdict'] == 'not_assessed'
    assert confirmed_page_completeness('Too short.', 'webpage')['verdict'] == 'not_assessed'
    assert confirmed_page_completeness(page, 'journal_article')['verdict'] == 'not_assessed'



def test_an_entry_repeated_as_a_body_heading_is_located_in_the_reference_list():
    # Paper 7, 2026-10-04: each discussed book heads its section with its entry.
    import hashlib
    import fitz
    from app.services.evidence_report import _attach_reference_field_geometry
    entry = 'Johnson, L. (2021). The Yellow Press Era.'
    doc = fitz.open(); page = doc.new_page()
    page.insert_text((72, 100), entry)
    page.insert_text((72, 300), 'References')
    page.insert_text((72, 330), entry)
    probe = {'source': {'raw_reference': entry}, 'field_difference': {'submitted_value': entry}}
    _attach_reference_field_geometry({'reference_practice': [probe]}, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    assert probe['rectangles'] and all(r['y0'] > 300 for r in probe['rectangles'])
    # Without a reference-list heading the repeated entry stays unplaced.
    doc2 = fitz.open(); page2 = doc2.new_page()
    page2.insert_text((72, 100), entry); page2.insert_text((72, 330), entry)
    probe2 = {'source': {'raw_reference': entry}, 'field_difference': {'submitted_value': entry}}
    _attach_reference_field_geometry({'reference_practice': [probe2]}, doc2, hashlib.sha256(doc2.tobytes()).hexdigest())
    assert probe2['rectangles'] == []


def test_a_topical_mismatch_on_a_retrieved_document_does_not_break_the_report(monkeypatch):
    # Regulation 4, 2026-10-06: the report failed (HTTP 500) because the
    # mismatch finding read best_evidence, which a retrieved document lacks.
    import fitz
    from app.services import evidence_report, report_layers
    monkeypatch.setattr(report_layers, "topical_mismatch", lambda member, citation: True)
    member = {"reference_id": "r1", "coverage_level": "full_text", "best_evidence": None,
              "scope_source": {"text": "The leading excerpt the topic was judged against."},
              "abstract_relevance": {"scope_assessment": {"rationale": "Different topic."}},
              "source": {"raw_reference": "Org. (2014). A page title here. http://example.test"}}
    view = {"citations": [{"student_text": "A claim (Org, 2014).", "members": [member]}],
            "reference_practice": [], "paper_surface": {}}
    document = fitz.open(); document.new_page()
    result = evidence_report.project_reference_flags(view, document, "x" * 64)
    [finding] = [f for f in result["reference_practice"] if f["finding_type"] == "source_topical_mismatch"]
    assert finding["abstract_text"] == "The leading excerpt the topic was judged against."
