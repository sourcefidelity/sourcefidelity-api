from pathlib import Path

from bs4 import BeautifulSoup

from app.services.evidence_report import render_evidence_report_html

SCRIPT = Path('app/services/report_interactions.js').read_text()


def page(**extra):
    view = {'title': 'Report', 'citation_format': 'APA', 'citations': [],
            'paper_surface': {'page_dimensions': [{'page_index': 0, 'width': 612, 'height': 792}],
                              'page_href_template': 'p-{page_index}'}, **extra}
    html = render_evidence_report_html(view, csp_nonce='workspace-layout-nonce')
    return html, BeautifulSoup(html, 'html.parser')


def test_one_workspace_with_a_single_bar_and_equal_height_panes():
    html, soup = page()
    layout = soup.select_one('main#report-layout')
    assert [child['class'][0] for child in layout.find_all(recursive=False)] == ['workspace-bar', 'paper', 'side-pane']
    assert '--controls-height' not in html and 'alignEvidence' not in html
    assert 'height:100dvh' in html and 'grid-template-rows:auto minmax(0,1fr)' in html
    assert '.side-pane .panel{flex:1;min-height:0;max-height:none' in html


def test_sliding_pane_has_reduced_motion_narrow_and_print_rules():
    html, _ = page()
    assert '.layout.panel-sliding .side-pane{transform:translateX(100%)}' in html
    assert '.layout.panel-collapsed{grid-template-columns:minmax(0,1fr)}' in html
    assert '@media(prefers-reduced-motion:reduce){.side-pane{transition:none}}' in html
    assert '@media(max-width:760px){.layout{grid-template-columns:1fr' in html
    assert '@media print{.workspace-bar,.side-pane{display:none}' in html


def test_key_is_static_and_how_to_read_uses_current_names():
    html, soup = page()
    assert 'data-key="\'+name' not in SCRIPT and "key.querySelectorAll('[data-key=" not in SCRIPT
    guide = soup.select_one('.read-guide').get_text()
    assert 'Show report' not in guide and 'Paper only' not in guide
    # The owner's revised text (2026-09-28), without columns.
    headings = [h.get_text() for h in soup.select('.read-guide h2')]
    assert headings == ['Source identification, verification and retrieval', 'Poor academic practice',
                        'Citation and reference format checking', 'Source use judgment']
    assert not soup.select('.read-guide .guide-grid')
    key = soup.select_one('#active-key').get_text()
    assert 'citation number' not in key and 'Issues:' in key


def test_the_workspace_ends_the_page():
    # Owner request 2026-09-28: scrolled to the bottom, the toolbar is at the top
    # of the window and the paper and window fill the rest.
    html, soup = page()
    assert 'height:100dvh;overflow:clip;margin:0 clamp(.75rem,2vw,1.5rem);' in html
    main = soup.select_one('main#report-layout')
    assert not [el for el in main.find_next_siblings() if el.name not in {'div', 'script', 'dialog'}
                or (el.name == 'div' and not el.has_attr('hidden') and el.get('id') != 'judgment-live')]
    assert not soup.select('.technical-export')


def test_view_state_contract_in_script():
    start, end = SCRIPT.index('// view-state:start'), SCRIPT.index('// view-state:end')
    block = SCRIPT[start:end]
    assert "side.inert = true" in block and "side.hidden = true" in block
    assert "button.disabled = next === 'paper'" in block     # arrows disabled in Paper
    assert "['paper','sources','judgment']" in block          # three layouts
    assert "prefers-reduced-motion: reduce" in block
    assert "restoreAnchor(anchor)" in block
    assert "--paper-share" not in block                       # the splitter share survives
    assert "window.open" not in SCRIPT                        # no paper link opening
