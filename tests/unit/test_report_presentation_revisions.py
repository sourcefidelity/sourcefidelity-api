"""Presentation-only findings, style links and layout controls."""
from copy import deepcopy

import fitz

from app.services.evidence_report import project_reference_flags, _render_report_summary, _render_panel_template
from app.services.report_style_guidance import render_guidance


def fixture_view():
    finding = {'finding_type': 'duplicate_citation_key', 'reference_ids': ['a', 'b'], 'finding_id': 'retained'}
    members = [{'reference_id': rid, 'source': {'raw_reference': f'Writer (2020). {title}.', 'year': '2020', 'title': title},
                'reference_findings': [finding]} for rid, title in [('a', 'First study'), ('b', 'Second study')]]
    return {'citations': [{'members': members, 'paper_location': {'localization_level': 'exact_rectangle',
             'rectangles': [{'page_index': 0, 'x0': 72, 'y0': 30, 'x1': 220, 'y1': 50}]}}], 'reference_practice': []}


def test_retained_duplicate_findings_get_exact_year_panels_without_mutation():
    view = fixture_view()
    original = deepcopy(view)
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), 'Writer (2020). First study.')
        page.insert_text((72, 100), 'Writer (2020). Second study.')
        result = project_reference_flags(view, document, 'bound-paper')
        assert len(result['reference_practice']) == 2
        for finding in result['reference_practice']:
            assert finding['localization_status'] == 'exact_field'
            rect = finding['rectangles'][0]
            assert page.get_textbox(fitz.Rect(*(rect[k] for k in ('x0', 'y0', 'x1', 'y1')))).strip() == '2020'
        assert result['summary']['reference_formatting']
        assert len(project_reference_flags(result, document, 'bound-paper')['reference_practice']) == 2
    assert view == original


def test_unplaced_duplicate_cannot_survive_as_summary_claim():
    with fitz.open() as document:
        document.new_page()
        result = project_reference_flags(fixture_view(), document, 'bound-paper')
    assert not result['summary']['reference_formatting']
    assert all(not f['rectangles'] for f in result['reference_practice'])


def test_help_is_style_specific_in_windows_not_summaries_and_not_source_evidence():
    from app.services.report_style_guidance import guidance_links
    issue = 'These references share the same author and year.'
    summary = _render_report_summary({'reference_formatting': [issue]})
    assert 'apastyle.apa.org' not in summary and 'Citation and Reference Formatting' in summary
    assert 'same-year-author' in guidance_links(['duplicate_citation_key'], 'APA 7')
    assert 'style.mla.org' in guidance_links(['duplicate_citation_key'], 'MLA 9')
    assert not guidance_links(['duplicate_citation_key'], 'Chicago')
    assert not guidance_links(['unverified_reference', 'bibliographic_field_conflict'], 'APA 7')
    assert 'style.mla.org' in render_guidance(issue, 'MLA 9')
    assert not render_guidance(issue, 'Chicago')
    assert not render_guidance('No source was retrieved.', 'APA')


def test_panel_heading_uses_citation_number():
    html = _render_panel_template({'members': [], 'student_text': 'Text.'}, 17)
    assert '<h2>Citation 17<span data-proposition-suffix></span></h2>' in html
    assert 'Selected Citation</h2>' not in html
