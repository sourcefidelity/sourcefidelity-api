from app.services.evidence_report import summary_text
import hashlib
import fitz
from app.services.evidence_report import (
    _attach_reference_field_geometry, _build_report_summary,
    _render_continuous_paper, _render_reference_panel_template,
)


def test_identical_duplicate_entries_each_yellow_one_academic_group():
    raw='Example Group. (2020). An account of visual history. https://example.org/a'
    peers=[dict(reference_id=rid,raw_reference=raw,author='Example Group',year='2020',title='An account of visual history') for rid in ('a','b')]
    findings=[dict(finding_type='duplicate_reference_entry',reference_id=p['reference_id'],
        retained_finding_id='group-1',source=p,related_references=peers,
        finding='These entries repeat the same source. The author, title and source URL agree.',
        field_difference=dict(field_name='entry',submitted_value=raw),rectangles=[]) for p in peers]
    doc=fitz.open();page=doc.new_page()
    for y in (80,130): page.insert_text((40,y),raw,fontsize=9)
    view={'reference_practice':findings}
    _attach_reference_field_geometry(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
    assert all(f['rectangles'] for f in findings)
    assert findings[0]['rectangles'] != findings[1]['rectangles']
    summaries=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=findings,require_paper_flags=True)
    assert len(summaries['academic_practice']) == 1
    assert summary_text(summaries['academic_practice'][0]).startswith('1 duplicate reference group')
    assert not summaries['reference_formatting']
    surface=dict(page_dimensions=[dict(page_index=0,width=612,height=792)],page_href_template='p-{page_index}')
    html,_=_render_continuous_paper(surface,[],findings)
    assert html.count('fill:#ffe45c') == 2
    panel=_render_reference_panel_template(findings[0],1)
    assert 'Academic Practice' in panel and 'issue-heading academic' in panel
    from bs4 import BeautifulSoup
    assert BeautifulSoup(panel.replace('template','div'),'html.parser').get_text().count(raw) == 2
    doc.close()


def test_unaccounted_duplicate_occurrence_does_not_guess_entry_identity():
    raw='Example Group. (2020). An account of visual history.'
    source=dict(reference_id='a',raw_reference=raw)
    f=dict(finding_type='duplicate_reference_entry',reference_id='a',source=source,
           related_references=[source],field_difference=dict(field_name='entry',submitted_value=raw))
    doc=fitz.open();p=doc.new_page()
    for y in (80,130):p.insert_text((40,y),raw)
    _attach_reference_field_geometry({'reference_practice':[f]},doc,'a'*64)
    assert not f['rectangles']
    doc.close()
