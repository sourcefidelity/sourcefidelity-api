"""One report with three layouts (owner decisions 2026-09-25).

Student and Instructor reports are merged; old ?audience= links still work.
Comment, highlight and pen tools are gone. APA/MLA links sit at the bottom
of the windows that explain the findings, not in the summary.
"""
import re
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import settings
from app.database import get_db
from app.main import app
from app.services.evidence_report import render_evidence_report_html
from app.services.storage.backend import get_storage_backend
from test_report_reference_windows import SURFACE, entry_box, rid, source, window


def _view(**extra):
    view = {'title': 'Report', 'citation_format': 'APA 7',
            'citations': [{'student_text': 'A claim (Writer, 2020).', 'members': [
                {'reference_id': rid(1), 'coverage_level': 'unavailable', 'source': source(1)}],
                'missing_reference_members': ['Absent, 2019'],
                'paper_location': {'localization_level': 'exact_rectangle', 'rectangles': [entry_box(0, 200)]}}],
            'reference_practice': [
                {'finding_type': 'reference_title_style', 'reference_id': rid(1), 'source': source(1),
                 'finding': 'Italicize the title.', 'rectangles': [dict(entry_box(1, 130), x1=200)]},
                {'finding_type': 'bibliographic_field_conflict', 'reference_id': rid(1), 'source': source(1),
                 'finding': 'The year differs from the located record.', 'rectangles': [dict(entry_box(1, 130), x0=300)]}],
            'bibliography': [{'reference_id': rid(1), 'source': source(1)}],
            'paper_surface': {**SURFACE, 'reference_locations': {rid(1): {'rectangles': [entry_box(1, 130)]}}}}
    view.update(extra)
    return view


def _html(**extra):
    return render_evidence_report_html(_view(**extra), csp_nonce='one-report-nonce-0001')


def test_style_links_sit_at_the_bottom_of_the_window_category_not_in_the_summary():
    soup = BeautifulSoup(_html(), 'html.parser')
    assert not soup.select('.summary a.style-guidance')
    win = window(soup, 'reference-entry-panel-1')
    sections = {s['data-category']: s for s in win.select('section.reference-finding')}
    formatting = sections['formatting']
    assert formatting.find_all(recursive=False)[-1]['class'] == ['style-guidance-links']
    assert 'apastyle.apa.org' in formatting.select_one('a.style-guidance')['href']
    # A located-record difference is Academic Practice in the window (owner 2026-10-03), not a style question.
    assert not sections['academic'].select('a.style-guidance')
    assert 'The information in this reference differs from the located record.' in sections['academic'].get_text()
    citation = window(soup, 'citation-panel-1')
    links = citation.select('.style-guidance-links a')
    assert [a.get_text() for a in links] == ['APA: Connecting citations to references']


@pytest.mark.parametrize('style,expected', [('MLA 9', 'style.mla.org'), ('Chicago', None)])
def test_style_links_follow_the_citation_style(style, expected):
    soup = BeautifulSoup(_html(citation_format=style), 'html.parser')
    hrefs = [a['href'] for a in window(soup, 'reference-entry-panel-1').select('a.style-guidance')]
    assert (expected in ' '.join(hrefs)) if expected else not hrefs


def test_one_layout_without_layout_buttons_and_judgment_only_in_the_interactive_report():
    """Owner decision 2026-09-28: one layout; Judgment is part of every interactive report."""
    html = _html()
    soup = BeautifulSoup(html, 'html.parser')
    for button in ('#sources-layout', '#judgment-layout', '#paper-layout'):
        assert soup.select_one(button) is None
    assert '.judgment-part,.judgment-sep' not in html
    portable = _html(portable_export=True)
    assert '.legend [data-key=judgment],.judgment-part,.judgment-sep,.judgment-slot{display:none}' in portable


def test_old_audience_links_open_the_same_one_report(monkeypatch):
    token = 'q' * 48
    import fitz
    document = fitz.open(); document.new_page(width=612, height=792)
    paper = document.tobytes()
    monkeypatch.setattr(settings, 'REPORT_AUTH_MODE', 'personal_bearer')
    monkeypatch.setattr(settings, 'REPORT_PERSONAL_ACCESS_TOKEN', SecretStr(token))
    monkeypatch.setattr(settings, 'SOURCE_REPOSITORY_SCOPE_ID', 'owner-1')
    monkeypatch.setattr('app.routers.report.load_authorized_evidence_report_bundle',
                        lambda *args, **kwargs: (_view(), SimpleNamespace(id='artifact-1'), paper))
    monkeypatch.setattr('app.routers.report._run_details', lambda *a, **k: {})
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {'Authorization': f'Bearer {token}'}
        pages = []
        for suffix in ('', '?audience=student', '?audience=instructor', '?audience=zzz'):
            response = client.get('/report/report-1' + suffix, headers=headers)
            assert response.status_code == 200
            pages.append(re.sub(r'nonce-[\w-]+|nonce="[\w-]+"', 'NONCE', response.text))
        assert len(set(pages)) == 1
        assert 'Priorities for revision' not in pages[0] and 'report-summary' in pages[0]
    finally:
        app.dependency_overrides.clear()


def test_retrieval_counts_open_the_evidence_summary_as_one_sentence():
    from app.services.evidence_report import _retrieval_sentence
    assert _retrieval_sentence({"reference_count": 20, "verified_full_text_sources": 7,
                                "abstract_or_limited_sources": 5, "unavailable_sources": 8}) == (
        "7/20 full-texts retrieved, 5/20 abstract/limited-texts retrieved, and 8/20 unretrieved texts.")
    assert _retrieval_sentence({}) == ""


def test_retrieval_counts_are_the_first_sources_item():
    from app.services.evidence_report import _render_report_summary
    html = _render_report_summary({"evidence": [{"kind": "x", "text": "Other"}]}, retrieval="There are 1/2 x.")
    evidence = html.split('<h2>Sources</h2>', 1)[1]
    assert evidence.startswith('<ul><li class="retrieval-summary">There are 1/2 x.</li><li ')
    empty = _render_report_summary({}, retrieval="There are 0/0 x.").split('<h2>Sources</h2>', 1)[1]
    assert empty.startswith('<ul><li class="retrieval-summary">There are 0/0 x.</li></ul></section>')


def test_summary_columns_in_owner_order_without_empty_wording():
    from app.services.evidence_report import _render_report_summary
    html = _render_report_summary({})
    assert (html.index('<h2>Sources</h2>') < html.index('<h2>Academic Practice</h2>')
            < html.index('<h2>Citation and Reference Formatting</h2>'))
    assert 'No issue was established' not in html
