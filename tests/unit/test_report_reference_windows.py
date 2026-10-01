import re

from bs4 import BeautifulSoup

from app.services.evidence_report import render_evidence_report_html
from app.services.report_badges import badge_width, place_badges, text_block_edges

SURFACE = {'page_dimensions': [{'page_index': 0, 'width': 612, 'height': 792},
                               {'page_index': 1, 'width': 612, 'height': 792}],
           'page_href_template': 'p-{page_index}',
           'selectable_words': {0: [(72, 90, 540, 100, 'Body')], 1: [(72, 90, 540, 100, 'Body')]}}


def rid(n):
    return f"ref-{n:04d}-{n:012x}"


def source(n, **extra):
    raw = f'Writer, A. (2020). Title {n}. https://example.org/{n}'
    return {'author': 'Writer, A.', 'year': '2020', 'title': f'Title {n}', 'raw_reference': raw, **extra}


def window(soup, template_id):
    """Template contents as ordinary markup (bs4 hides template strings)."""
    template = soup.select_one(f'template#{template_id}')
    return template and BeautifulSoup(template.decode_contents(), 'html.parser')


def entry_box(page, y):
    return {'page_index': page, 'x0': 72, 'y0': y, 'x1': 540, 'y1': y + 12}


def report(**overrides):
    cited = {'reference_id': rid(1), 'coverage_level': 'abstract_only',
             'source': source(1, submitted_hyperlinks=['https://example.org/1', 'javascript:alert(1)', 'http://127.0.0.1/x'])}
    view = {
        'title': 'Report', 'citation_format': 'APA', 'audience': 'instructor',
        'citations': [{'student_text': 'A claim (Writer, 2020).', 'members': [cited],
                       'paper_location': {'localization_level': 'exact_rectangle', 'rectangles': [entry_box(0, 200)]}}],
        'reference_practice': [
            {'finding_type': 'reference_title_style', 'reference_id': rid(1), 'finding': 'Italicize the title.',
             'source': source(1), 'rectangles': [dict(entry_box(1, 100), x1=200)]},
            {'finding_type': 'submitted_link_issue', 'reference_id': rid(1), 'finding': 'The submitted link returned a missing page.',
             'source': source(1), 'rectangles': [dict(entry_box(1, 100), x0=300)]},
            {'finding_type': 'body_title_style', 'reference_id': rid(1), 'finding': 'Italicize the title.',
             'source': source(1), 'rectangles': [dict(entry_box(0, 200), x1=150)]},
        ],
        'bibliography': [{'reference_id': rid(1), 'source': source(1)}, {'reference_id': rid(2), 'source': source(2)}],
        'paper_surface': {**SURFACE, 'reference_locations': {rid(1): {'rectangles': [entry_box(1, 100), entry_box(1, 112)],
                                                                      'links': [{'href': 'https://example.org/1',
                                                                                 'rectangle': entry_box(1, 112)}]}}},
    }
    view.update(overrides)
    return BeautifulSoup(render_evidence_report_html(view, csp_nonce='reference-window-nonce'), 'html.parser')


def test_reference_window_shows_entry_links_citations_and_grouped_findings():
    soup = report()
    win = window(soup, 'reference-entry-panel-1')
    assert win.select_one('h2 a')['href'] == '#reference-location-1'
    assert win.select_one('.reference-availability').get_text() == 'Abstract Retrieved'  # one wording (2026-09-30)
    links = [a['href'] for a in win.select('.reference-window-entry a')]
    assert 'https://example.org/1' in links
    assert not any('javascript' in href or '127.0.0.1' in href for href in links)
    assert [b['data-go-to'] for b in win.select('.reference-citations button')] == ['citation-panel-1']
    sections = win.select('section.reference-finding')
    assert len(sections) == 2                          # the body-title finding keeps its own window
    # One heading per category; a submitted-link issue is Academic Practice.
    assert [s.select_one('.issue-heading').get_text() for s in sections] == [
        'Citation and Reference Formatting', 'Academic Practice']
    assert 'The submitted link returned a missing page.' in sections[1].get_text()
    # The complete reference appears once; each finding drops its own copy.
    assert len(win.select('.full-reference')) == 1
    # Merged reference-list findings no longer have standalone windows.
    assert soup.select_one('template#reference-panel-3') and not soup.select_one('template#reference-panel-1')


def test_every_reference_list_target_opens_the_same_window_and_paper_links_are_gone():
    soup = report()
    title = next(hit.find_parent('a') for hit in soup.select('.reference-formatting-hit')
                 if hit.find_parent('svg')['data-page-index'] == '1')
    diamond = soup.select_one('.submitted-link-marker').find_parent('a')
    entries = soup.select('a.reference-entry-overlay')
    assert title['data-panel-template'] == diamond['data-panel-template'] == 'reference-entry-panel-1'
    assert {a['data-panel-template'] for a in entries} == {'reference-entry-panel-1'}
    assert not soup.select('a.paper-reference-link')
    # A body-text finding on one citation span still opens that citation.
    body = [a for a in soup.select('a.reference-practice-overlay') if a['data-panel-template'].startswith('citation-panel-')]
    assert body and body[0]['data-panel-template'] == 'citation-panel-1'


def test_uncited_unlocated_reference_has_a_window_but_no_paper_target():
    soup = report()
    win = window(soup, 'reference-entry-panel-2')
    assert win and win.select_one('h2').get_text() == 'Reference 2' and not win.select_one('h2 a')
    assert 'No in-text citation in this report is linked to this reference.' in win.get_text()
    assert not soup.select('[data-panel-template="reference-entry-panel-2"]')


def test_entry_spanning_two_pages_gets_targets_on_both_but_one_badge_and_anchor():
    locations = {rid(1): {'rectangles': [entry_box(0, 700), entry_box(1, 72)], 'links': []}}
    soup = report(paper_surface={**SURFACE, 'reference_locations': locations})
    pages = {a.find_parent('svg')['data-page-index'] for a in soup.select('a.reference-entry-overlay')}
    assert pages == {'0', '1'}
    assert len(soup.select('a.reference-badge')) == 1 and len(soup.select('#reference-location-1')) == 1


def test_badges_label_citations_and_references_in_the_margin():
    soup = report()
    citation = soup.select_one('a.citation-badge')
    reference = soup.select_one('a.reference-badge')
    assert citation['data-panel-template'] == 'citation-panel-1' and citation.select_one('text').get_text() == '1'
    assert reference['data-panel-template'] == 'reference-entry-panel-1'
    for badge in (citation, reference):
        assert float(badge.select_one('rect')['x']) + float(badge.select_one('rect')['width']) <= 72 - 3
        assert badge['aria-hidden'] == 'true' and badge['tabindex'] == '-1'
    assert float(citation.select_one('rect')['rx']) > float(reference.select_one('rect')['rx'])
    assert not re.search(r'<a[^>]*paper-badge[^>]*style=', str(soup))
    html = str(soup)
    assert 'body.layout-paper .paper-badge{display:none}' in html and '@media print{.paper-badge{display:none}}' in html


def test_badge_placement_stacks_outward_and_avoids_occupied_boxes():
    requests = [{'template': f't{n}', 'kind': 'citation', 'number': n, 'line': (200, 100, 300, 112)} for n in (1, 2)]
    placed = place_badges(requests, text_left=72, text_right=540, page_width=612)
    boxes = [row['box'] for row in placed]
    assert all(row['placement'] == 'left' for row in placed)
    assert boxes[0][2] == 69 and boxes[1][2] < boxes[0][0]          # second stacks further out
    diamond = (55, 95, 71, 115)
    placed = place_badges(requests[:1], text_left=72, text_right=540, page_width=612, occupied=[diamond])
    assert placed[0]['box'][2] <= diamond[0]


def test_badge_placement_falls_back_to_right_margin_then_chip():
    request = [{'template': 't', 'kind': 'reference', 'number': 12, 'line': (20, 100, 300, 112)}]
    right = place_badges(request, text_left=10, text_right=500, page_width=612)[0]
    assert right['placement'] == 'right' and right['box'][0] >= 503
    chip = place_badges(request, text_left=10, text_right=600, page_width=612)[0]
    assert chip['placement'] == 'chip' and chip['box'][3] <= 112
    assert badge_width(12) > badge_width(3)
    assert text_block_edges([], [(40, 0, 90, 10)]) == (40, 90) and text_block_edges([], []) is None
