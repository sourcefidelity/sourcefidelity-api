import fitz
from bs4 import BeautifulSoup

from app.services.evidence_report import (
    _build_report_summary, _render_report_summary, project_reference_flags,
    render_evidence_report_html, summary_text,
)

BOX = {'page_index': 0, 'x0': 72, 'y0': 100, 'x1': 300, 'y1': 115}


def rid(n):
    return f"ref-{n:04d}-{n:012x}"


def finding(kind, n, **extra):
    return {'finding_type': kind, 'reference_id': rid(n), 'rectangles': [dict(BOX, y0=100 + n, y1=110 + n)],
            'source': {'raw_reference': f'Writer {n} (2020). Title {n}.'}, **extra}


def summaries(**kwargs):
    return _build_report_summary(citations=kwargs.pop('citations', []), overview={},
                                 pervasive_hanging_indent=kwargs.pop('pervasive_hanging_indent', False), **kwargs)


def items(summary, category):
    return summary[category]


def rendered(summary):
    return BeautifulSoup(_render_report_summary(summary), 'html.parser')


def test_more_than_five_instances_state_total_then_five_links_and_ellipsis():
    practice = [finding('reference_title_style', n, finding='Title formatting differs.') for n in range(1, 8)]
    summary = summaries(reference_practice=practice)
    item = items(summary, 'reference_formatting')[0]
    assert item['count'] == 7 and len(item['instances']) == 7
    assert summary_text(item) == '7 reference titles use incorrect title formatting.'
    li = rendered(summary).select_one('li[data-summary-kind="reference_title_style"]')
    assert li.get_text() == '7 reference titles use incorrect title formatting (references 1, 2, 3, 4, 5, …).'
    links = li.select('a.summary-instance')
    assert [a['data-go-to'] for a in links] == [f'reference-entry-panel-{n}' for n in range(1, 6)]
    assert links[0]['aria-label'] == 'Reference 1'


def test_five_or_fewer_instances_are_all_listed_without_ellipsis():
    summary = summaries(reference_practice=[finding('required_doi_missing', n) for n in (4, 2, 9)])
    li = rendered(summary).select_one('li')
    assert li.get_text() == '3 references omit DOIs for the cited works (references 2, 4, 9).'
    assert '…' not in li.get_text()


def test_locator_summary_counts_citations_and_truncates_links():
    citations = [{'claim_id': f'c{i}', 'members': []} for i in range(1, 8)]
    practice = [finding('required_quotation_locator_missing', 1, claim_id=f'c{i}') for i in range(1, 8)]
    summary = summaries(citations=citations, reference_practice=practice)
    item = items(summary, 'reference_formatting')[0]
    assert summary_text(item) == '7 quotations lack page or paragraph locators in the parenthetical citation.'
    text = rendered(summary).select_one('li').get_text()
    assert text.startswith('7 quotations lack page or paragraph locators in the parenthetical citation (citations 1, 2, 3, 4, 5, …).')


def test_groups_names_and_distinct_body_titles():
    practice = [finding('duplicate_reference_entry', n, retained_finding_id=group)
                for n, group in ((1, 'g1'), (3, 'g1'), (2, 'g2'), (4, 'g2'))]
    group_html = rendered(summaries(reference_practice=practice)).select_one('li')
    assert 'references 1 and 3; 2 and 4' in group_html.get_text()
    assert group_html.select('a')[0]['data-go-to'] == 'reference-entry-panel-1'

    citations = [{'members': [], 'missing_reference_members': ['Absent, 2020']},
                 {'members': [], 'missing_reference_members': ['as cited in Absent, 2020', 'Other, 2019']}]
    summary = summaries(citations=citations)
    item = items(summary, 'academic_practice')[0]
    assert summary_text(item) == '2 in-text sources have no matching reference-list entry (Absent, 2020; Other, 2019).'
    targets = {a.get_text(): a['data-go-to'] for a in rendered(summary).select('a.summary-instance')}
    assert targets == {'Absent, 2020': 'citation-panel-1', 'Other, 2019': 'citation-panel-2'}

    titles = [finding('body_title_style', 5, field_difference={'submitted_value': 'Get Out'}) for _ in range(3)]
    titles.append(finding('body_title_style', 6, field_difference={'submitted_value': 'Beloved'}))
    item = items(summaries(reference_practice=titles), 'reference_formatting')[0]
    assert item['count'] == 2 and summary_text(item).startswith('2 film or book titles lack required italics')
    assert [row['label'] for row in item['instances']] == ['Get Out', 'Beloved']


def test_hanging_indent_isolated_is_listed_and_pervasive_is_stated_once():
    practice = [finding('formatting', n) for n in (2, 5)]
    isolated = items(summaries(reference_practice=practice), 'reference_formatting')[0]
    assert isolated['kind'] == 'hanging_indent' and [i['number'] for i in isolated['instances']] == [2, 5]
    assert summary_text(isolated) == '2 reference(s) lack the expected hanging indent.'
    pervasive = items(summaries(pervasive_hanging_indent=True, reference_practice=practice), 'reference_formatting')
    assert [i['kind'] for i in pervasive] == ['hanging_indent_pervasive'] and not pervasive[0]['instances']


def test_pervasive_reference_order_has_no_count_list_or_links():
    summary = summaries(pervasive_reference_order=True)
    item = items(summary, 'reference_formatting')[0]
    assert item['kind'] == 'reference_order_pervasive' and item['pervasive'] and not item['instances']
    assert summary_text(item) == 'The reference list is not in alphabetical order by first-author surname.'
    assert not rendered(summary).select('a.summary-instance')


def test_no_priority_cap_in_the_one_report():
    practice = ([finding('unverified_reference', 1), finding('reference_author_conflict', 2)]
                + [finding('required_doi_missing', 3)])
    summary = summaries(reference_practice=practice)
    assert sum(len(values) for values in summary.values()) == 3


def test_same_author_group_moves_to_formatting_by_kind():
    shared = {'finding_type': 'duplicate_citation_key', 'reference_ids': [rid(1), rid(2)], 'finding_id': 'kept'}
    members = [{'reference_id': rid(n), 'reference_findings': [shared],
                'source': {'raw_reference': f'Writer (2020). Study {n}.', 'year': '2020', 'title': f'Study {n}'}}
               for n in (1, 2)]
    view = {'citations': [{'members': members, 'paper_location': {'localization_level': 'exact_rectangle',
            'rectangles': [BOX]}}], 'reference_practice': []}
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), 'Writer (2020). Study 1.')
        page.insert_text((72, 100), 'Writer (2020). Study 2.')
        result = project_reference_flags(view, document, 'bound-paper')
    moved = result['summary']['reference_formatting']
    assert [item['kind'] for item in moved] == ['duplicate_citation_key']
    assert moved[0]['instances'] == [{'type': 'group', 'numbers': [1, 2], 'target': 'reference-entry-panel-1'}]
    assert not result['summary']['academic_practice']


def test_every_summary_link_opens_a_window_that_exists():
    practice = [finding('reference_title_style', n, finding='Title formatting differs.') for n in (1, 2)]
    view = {'title': 'Report', 'citation_format': 'APA', 'audience': 'instructor',
            'citations': [{'student_text': 'A claim (Writer, 2020).', 'members': [
                {'reference_id': rid(1), 'source': {'raw_reference': 'Writer 1 (2020). Title 1.',
                 'author': 'Writer 1', 'year': '2020', 'title': 'Title 1'}}]}],
            'reference_practice': practice,
            'paper_surface': {'page_dimensions': [{'page_index': 0, 'width': 612, 'height': 792}],
                              'page_href_template': 'p-{page_index}'}}
    view['summary'] = summaries(citations=view['citations'], reference_practice=practice)
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce='linked-summary-nonce'), 'html.parser')
    links = soup.select('.summary a.summary-instance[data-go-to]')
    assert links and all(soup.select_one(f'template#{a["data-go-to"]}') for a in links)
    # No located entry: the link cannot scroll, so it points at the window.
    assert {a['href'] for a in links} == {'#evidence-panel'}
