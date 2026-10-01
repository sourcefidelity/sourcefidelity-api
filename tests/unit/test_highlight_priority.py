import xml.etree.ElementTree as ET
from app.services.highlight_priority import subtract_rectangles, prioritize_svg_highlights
from app.services.evidence_report import _display_evidence_note, _render_technical_details


def test_three_levels_do_not_mix_paint():
    html = ('<a><rect class="source-highlight" x="0" y="0" width="30" height="10" /></a>'
            '<a><rect class="reference-formatting-hit" x="10" y="0" width="20" height="10" /></a>'
            '<a><rect class="reference-formatting-hit academic-highlight" x="20" y="0" width="10" height="10" /></a>')
    result = ET.fromstring('<g>'+prioritize_svg_highlights(html)+'</g>')
    rects = list(result.iter('rect'))
    assert [float(r.get('x')) for r in rects] == [0,10,20]
    assert [float(r.get('width')) for r in rects] == [10,10,10]


def test_nonoverlap_preserved_and_partial_overlap_keeps_surrounding_text():
    assert subtract_rectangles((0,0,10,10), [(20,20,30,30)]) == [(0,0,10,10)]
    parts = subtract_rectangles((0,0,10,10), [(2,2,8,8)])
    assert sum((r[2]-r[0])*(r[3]-r[1]) for r in parts) == 64


def test_role_boilerplate_suppressed_in_legacy_projection():
    for note in ["This passage states the source author's synthesis or conclusion.",
                 'This passage appears to provide methods or background rather than a direct finding.',
                 "This passage reports another work rather than this source's own finding. Check whether the original work should also be cited."]:
        assert _display_evidence_note({}, {'evidence_note':note}) == ''


def test_cost_label_has_no_explanatory_notes():
    # Owner request 2026-09-28: the cost figure without the partial-pricing note.
    text = _render_technical_details({'estimated_cost_usd':.12, 'cost_estimate_partial':True})
    assert 'Estimated Cost (Before Credits)' in text and 'US$0.1200' in text
    assert 'Partial: usage with no price on record is excluded' not in text
    assert 'total API cost' not in text


def test_academic_color_has_explicit_csp_safe_specificity():
    from app.services.evidence_report import render_evidence_report_html
    html = render_evidence_report_html({'title':'Fixture','citation_format':'APA','citations':[], 'paper_surface':{}}, csp_nonce='highlight-priority-test-nonce')
    assert '.reference-practice-overlay .reference-formatting-hit.academic-highlight{fill:#ffe45c;fill-opacity:.4;stroke:none}' in html
    assert 'Downloads include report evidence' not in html
