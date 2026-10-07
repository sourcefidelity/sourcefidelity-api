"""Report presentation of "Cannot be verified" and the regrouped categories
(owner decisions 2026-09-24)."""
from bs4 import BeautifulSoup

from app.services.evidence_report import (
    _build_report_summary, _formatting_overlaps, normalize_reference_findings, render_evidence_report_html,
)
from test_report_reference_windows import SURFACE, entry_box, rid, source, window


def unverified(n=1, **extra):
    return {'finding_type': 'unverified_reference', 'reference_id': rid(n), 'source': source(n),
            'finding': 'Cannot be verified. The searches suited to this kind of work did not locate it. '
                       'This does not establish that the work does not exist.',
            'evidence_explanation': 'Searched by title and author: Crossref, OpenAlex, Brave Search.',
            'limitations': ['Search coverage is bounded, not an exhaustive catalog of published works.'],
            'field_difference': {'field_name': 'entry', 'submitted_value': source(n)['raw_reference']},
            'rectangles': [entry_box(1, 100 + 30 * n)], **extra}


def formatting(kind, n=1, text='Italicize the title.'):
    return {'finding_type': kind, 'reference_id': rid(n), 'source': source(n), 'finding': text,
            'rectangles': [dict(entry_box(1, 100 + 30 * n), x1=200)]}


def render(findings, members=None, audience='instructor'):
    members = members if members is not None else [{'reference_id': rid(1), 'coverage_level': 'unavailable',
                                                     'source': source(1)}]
    view = {'title': 'Report', 'citation_format': 'APA', 'audience': audience,
            'citations': [{'student_text': 'A claim (Writer, 2020).', 'citation_marker': '(Writer, 2020)', 'members': members,
                           'paper_location': {'localization_level': 'exact_rectangle',
                                              'rectangles': [entry_box(0, 200)]}}],
            'reference_practice': findings,
            'bibliography': [{'reference_id': rid(n), 'source': source(n)} for n in (1, 2)],
            'paper_surface': {**SURFACE, 'selectable_words': {0: [(100, 201, 150, 210, '(Writer,'), (152, 201, 190, 210, '2020).')],
                                                              1: SURFACE['selectable_words'][1]},
                              'reference_locations': {
                rid(1): {'rectangles': [entry_box(1, 130)]}, rid(2): {'rectangles': [entry_box(1, 160)]}}}}
    return BeautifulSoup(render_evidence_report_html(view, csp_nonce='unverified-reference-nonce'), 'html.parser')


def test_cannot_be_verified_is_an_evidence_summary_not_academic_practice():
    summaries = _build_report_summary(citations=[], overview={'reference_count': 7}, pervasive_hanging_indent=False,
                                      reference_practice=[unverified()])
    [item] = summaries['evidence']
    # Owner wording 2026-10-02.
    assert item['kind'] == 'unverified_reference' and item['text'] == '1/7 sources cannot be verified and may not exist.'
    assert not summaries['academic_practice']


def test_legacy_flag_is_relabelled_and_old_text_never_shown():
    legacy = {**unverified(), 'finding_type': 'potentially_fabricated_reference',
              'finding': 'Potentially fabricated reference. Searches did not establish a matching work.'}
    [shown] = normalize_reference_findings([legacy])
    assert shown['finding_type'] == 'unverified_reference' and shown['legacy_finding_type'] == 'potentially_fabricated_reference'
    html = str(render([legacy]))
    assert 'otentially fabricated' not in html


def test_unverified_reference_and_its_citations_are_soft_red():
    soup = render([unverified()])
    entry = [hit for hit in soup.select('.reference-formatting-hit') if 'unverified-highlight' in hit.get('class', [])]
    assert entry, 'the reference entry is marked'
    member = soup.select_one('.citation-overlay .source-highlight')
    assert 'unverified-highlight' in member['class']
    heading = window(soup, 'citation-panel-1').select_one('h3')
    assert heading.get_text() == 'Not Judged - No Text Retrieved – Cannot be verified'
    assert heading.select_one('mark.unverified').get_text() == 'Cannot be verified'
    key = soup.select_one('#active-key')
    assert key.select_one('.key-unverified').get_text() == 'Unverifiable Reference'
    assert key.select_one('.key-record').get_text() == 'Source Record Conflict'


def test_reference_window_groups_findings_under_one_heading_per_category():
    findings = [formatting('reference_title_style'), formatting('reference_order', text='Check alphabetical placement.'),
                formatting('formatting', text='The visible continuation indent differs.'),
                unverified(),
                {**formatting('doi_registers_a_different_title', text='The DOI is registered to a different title.')}]
    win = window(render(findings), 'reference-entry-panel-1')
    headings = [h.get_text() for h in win.select('section.reference-finding > h3 .issue-heading')]
    # No "Evidence" label (2026-09-30); Academic Practice before formatting (2026-10-03).
    assert headings == ['Academic Practice', 'Citation and Reference Formatting']
    formatting_section = win.select('section.reference-finding')[2]
    assert len(formatting_section.select('.finding-item')) == 3 and len(formatting_section.select('h3')) == 1
    assert 'Search Details' in win.select('section.reference-finding')[0].get_text()


def test_wrong_doi_is_a_submitted_link_issue_never_formatting_and_stated_once():
    identifier = {'finding_type': 'reference_identifier_conflict', 'reference_id': rid(1), 'source': source(1),
                  'finding': 'The submitted DOI identifies “Other” by Someone, not the title and authors in this reference.',
                  'records': [{'observed': {'title': 'Other work', 'authors': ['Someone, B.'], 'year': '2021',
                                            'doi': '10.1234/other'}}],
                  'rectangles': [entry_box(1, 130)]}
    registered = formatting('doi_registers_a_different_title', text='The DOI given in this reference is registered to a different title.')
    assert not _formatting_overlaps({'paper_location': {'rectangles': registered['rectangles']}}, registered)
    kept = normalize_reference_findings([identifier, registered])
    assert [f['finding_type'] for f in kept] == ['reference_identifier_conflict']
    win = window(render([identifier, registered]), 'reference-entry-panel-1')
    text = win.get_text(' ')
    assert 'The submitted DOI identifies:' in text and 'not the title and authors' not in text
    assert 'Other work' in text and 'registered to a different title' not in text
    summaries = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False, reference_practice=[registered])
    assert [i['kind'] for i in summaries['academic_practice']] == ['submitted_link_issue']
    assert not summaries['reference_formatting']


def test_differences_in_reference_information_are_evidence():
    difference = {'finding_type': 'bibliographic_field_conflict', 'reference_id': rid(2), 'source': source(2),
                  'finding': 'The year differs from the located record.', 'rectangles': [entry_box(1, 160)],
                  'field_differences': [{'field_name': 'year', 'submitted_value': '2020', 'located_value': '2019'}]}
    summaries = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False, reference_practice=[difference])
    assert [i['kind'] for i in summaries['evidence']] == ['bibliographic_field_conflict']
    assert not summaries['reference_formatting']
    soup = render([difference])
    assert soup.select('.reference-formatting-hit.reference-difference-highlight')


def test_no_located_record_comparison_for_a_reference_that_cannot_be_verified():
    borrowed = {'finding_type': 'bibliographic_conflict', 'reference_id': rid(1), 'source': source(1),
                'finding': 'The title in this reference differs from the located record.', 'rectangles': []}
    kept = normalize_reference_findings([unverified(), borrowed])
    assert [f['finding_type'] for f in kept] == ['unverified_reference']


def test_no_comparison_with_the_work_a_wrong_doi_names():
    registered = formatting('doi_registers_a_different_title', text='The DOI is registered to a different title.')
    borrowed = {'finding_type': 'bibliographic_conflict', 'reference_id': rid(1), 'source': source(1),
                'finding': 'The year in this reference differs from the located record.', 'rectangles': []}
    anchored = {'finding_type': 'bibliographic_field_conflict', 'reference_id': rid(1), 'source': source(1),
                'finding': 'The volume differs.', 'rectangles': []}
    kept = normalize_reference_findings([registered, borrowed, anchored])
    assert [f['finding_type'] for f in kept] == ['doi_registers_a_different_title', 'bibliographic_field_conflict']


def test_a_citation_without_a_space_after_the_comma_is_still_highlighted():
    from app.services.report_member_navigation import member_targets
    citation = {'citation_marker': '(Bennett,2007)', 'members': [{'source': {'author': 'Bennett, T.', 'year': '2007'}}],
                'paper_location': {'rectangles': [{'page_index': 0, 'x0': 0, 'y0': 0, 'x1': 500, 'y1': 20}]}}
    words = {0: [(10, 2, 60, 12, 'cultural'), (62, 2, 140, 12, 'populations'), (142, 2, 230, 12, '(Bennett,2007).')]}
    [target] = member_targets(citation, words)
    assert (target['x0'], target['x1']) == (142, 230)
    spaced = {**citation, 'citation_marker': '(Bennett, 2007)'}
    words = {0: [(10, 2, 60, 12, 'cultural'), (142, 2, 190, 12, '(Bennett,'), (192, 2, 230, 12, '2007).')]}
    [target] = member_targets(spaced, words)
    assert (target['x0'], target['x1']) == (142, 230)


def test_the_identified_record_completes_the_doi_sentence_in_the_pdf():
    import fitz
    from app.services.report_export import _render_pdf
    identifier = {'finding_type': 'reference_identifier_conflict', 'reference_id': rid(1), 'source': source(1),
                  'finding': 'old text', 'rectangles': [entry_box(0, 100)],
                  'records': [{'observed': {'title': 'Identified other work', 'authors': ['Someone, B.']}}]}
    doc = fitz.open(); doc.new_page()
    rendered, _ = _render_pdf(doc.tobytes(), citations=[], reference_practice=[identifier], 
                              export_binding='synthetic-test', view=None)
    with fitz.open(stream=rendered, filetype='pdf') as exported:
        text = ' '.join(' '.join(page.get_text() for page in exported).split())
    start = text.index('The submitted DOI identi')
    assert text.index('other work', start) < text.index('Reference as submitted: Writer', start)


def test_citation_window_heading_links_its_reference_in_the_reference_colour():
    soup = render([])
    template = BeautifulSoup(soup.find('template', id='citation-panel-1').decode_contents(), 'html.parser')
    link = template.find('h2').find('button', class_='reference-heading-link')
    assert link.get_text() == 'Reference 1' and link['data-go-to'] == 'reference-entry-panel-1'
    window = BeautifulSoup(soup.find('template', id='reference-entry-panel-1').decode_contents(), 'html.parser')
    assert window.find('h2').find(class_='reference-heading-link').get_text() == 'Reference 1'


def test_a_wrapped_url_with_its_submitted_link_is_written_once():
    from app.services.evidence_report import _render_formatted_reference
    render_source = lambda source: _render_formatted_reference(source, '')
    full = "https://example.test/publication/1_Theorising_the_Practice_of_Media"
    html = render_source({"raw_reference": "Adams, A. (2020). A study. https://example.test/publication/1_Theorising_the_P ractice_of_Media",
                          "submitted_hyperlinks": [full]})
    assert html.count(f'>{full}</a>') == 1 and "ractice_of_Media" not in html.replace(full, "")


def test_a_refused_submitted_link_explains_the_missing_text():
    from app.services.evidence_report import _submitted_link_refused
    refused = {'coverage_level': 'unavailable', 'submitted_link_observations': [
        {'state': 'observed', 'requests': [{'completed_at': '2026-10-01T00:00:00Z', 'outcome': 'access_refused'}]}]}
    assert _submitted_link_refused(refused)
    assert not _submitted_link_refused({**refused, 'coverage_level': 'full_text'})
    ok = {'coverage_level': 'unavailable', 'submitted_link_observations': [
        {'state': 'observed', 'requests': [{'completed_at': '2026-10-01T00:00:00Z', 'outcome': 'response'}]}]}
    assert not _submitted_link_refused(ok)


def test_reference_findings_are_counted_in_three_broad_lines():
    from app.services.evidence_report import broad_reference_summary, _summary_item
    ref = lambda n: {'type': 'reference', 'number': n, 'target': f'reference-entry-panel-{n}'}
    summary = {'evidence': [], 'academic_practice': [
        _summary_item('submitted_link_issue', 'x', instances=[ref(3)], count=1)],
        'reference_formatting': [
        _summary_item('required_doi_missing', 'x', instances=[ref(5), ref(3)], count=2),
        _summary_item('publication_year_discrepancy', 'x', instances=[ref(2)], count=1),
        _summary_item('chapter_pages_missing', 'x', instances=[ref(6)], count=1)]}
    findings = [{'finding_type': 'reference_title_missing', 'reference_id': 'r7', 'rectangles': [{}]}]
    grouped = broad_reference_summary(summary, findings, {'r7': 7})
    # Incorrect information is Academic Practice (owner decision 2026-10-07).
    assert [(i['kind'], i['lead'], [x['number'] for x in i['instances']]) for i in grouped['academic_practice']] == [
        ('reference_links', '2 references have incorrect or missing links', [3, 5]),
        ('reference_incorrect_information', '1 reference contains incorrect information', [2])]
    assert [(i['kind'], i['lead']) for i in grouped['reference_formatting']] == [
        ('reference_missing_information', '2 references are missing information')]
    assert broad_reference_summary(grouped, findings, {'r7': 7}) == grouped


def test_a_possible_match_copy_of_an_unverifiable_reference_is_not_its_text():
    from app.services.evidence_report import withdraw_unverified_possible_matches
    member = {'reference_id': 'r1', 'identity_status': 'possible_match', 'coverage_level': 'partial',
              'best_evidence': {'text': 'x'}}
    view = {'overview': {'verified_full_text_sources': 0, 'abstract_or_limited_sources': 1, 'unavailable_sources': 6}}
    out = withdraw_unverified_possible_matches(view, [{'members': [member]}], {'r1'})
    assert member['coverage_level'] == 'unavailable' and member['best_evidence'] is None
    assert out['overview']['abstract_or_limited_sources'] == 0 and out['overview']['unavailable_sources'] == 7
    kept = {'reference_id': 'r2', 'identity_status': 'possible_match', 'coverage_level': 'partial'}
    assert withdraw_unverified_possible_matches(view, [{'members': [kept]}], {'r1'}) is view and kept['coverage_level'] == 'partial'
