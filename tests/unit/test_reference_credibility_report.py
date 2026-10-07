import hashlib
import fitz
from bs4 import BeautifulSoup
from app.services.evidence_report import (
    _attach_reference_field_geometry, _build_report_summary,
    _render_continuous_paper, _render_reference_panel_template,
)
from app.services.highlight_priority import ACADEMIC_FINDINGS, finding_category


def test_legacy_strong_flag_is_soft_red_evidence_with_one_summary_and_audit_only_records():
    raw='Alvarez, A. (2020). Archival Methods in Coastal Communities. doi:10.1234/coastal'
    f=dict(finding_type='potentially_fabricated_reference',reference_id='ref',
        source=dict(reference_id='ref',raw_reference=raw,author='Alvarez, A.',year='2020',title='Archival Methods in Coastal Communities'),
        finding='Potentially fabricated reference. Two independently registered works have conflicting title and author details.',
        records=[dict(observed=dict(title='Different work',authors=['Bishop, B.'],year='2020',doi='10.1234/coastal'))],
        field_difference=dict(field_name='entry',submitted_value=raw),rectangles=[])
    doc=fitz.open(); p=doc.new_page(); p.insert_text((40,80),raw,fontsize=9)
    _attach_reference_field_geometry({'reference_practice':[f]},doc,hashlib.sha256(doc.tobytes()).hexdigest())
    assert f['rectangles'] and f['localization_status']=='exact_field'
    summaries=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=[f],require_paper_flags=True)
    assert [item['kind'] for item in summaries['evidence']] == ['unverified_reference']
    assert not summaries['academic_practice'] and not summaries['reference_formatting']
    surface=dict(page_dimensions=[dict(page_index=0,width=612,height=792)],page_href_template='p-{page_index}')
    html,_=_render_continuous_paper(surface,[],[f])
    assert 'fill:#f28b82' in html and 'unverified-highlight' in html
    panel=_render_reference_panel_template(f,1)
    assert 'issue-heading evidence' not in panel and 'Cannot be verified' in panel
    soup=BeautifulSoup(panel,'html.parser')
    assert soup.find('a') is None  # Similar candidate records remain audit-only.
    assert f['finding_type'] not in ACADEMIC_FINDINGS and finding_category(f['finding_type']) == 'evidence'
    assert 'https://apastyle' not in panel
    combined=_render_reference_panel_template(f,1,combined=True)
    assert 'issue-heading evidence' not in combined and raw not in combined
    from app.services.report_export import _render_pdf
    rendered, _ = _render_pdf(doc.tobytes(), citations=[], reference_practice=[f],
        export_binding='synthetic-test', view=None)
    with fitz.open(stream=rendered, filetype='pdf') as exported:
        text = '\n'.join(page.get_text() for page in exported)
        assert 'Sources 1' in text and 'Cannot be veri' in text      # the PDF font sets 'fi' as a ligature
        assert 'Potentially fabricated' not in text
        assert not any('10.1234/coastal' in str(link) for page in exported for link in page.get_links())
    # An unplaced report issue is not advertised in the summary.
    f['rectangles']=[]
    summaries=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=[f],require_paper_flags=True)
    assert not summaries['academic_practice']
    doc.close()


def test_factual_identifier_error_is_purple_link_issue_in_academic_summary():
    f=dict(finding_type='reference_identifier_conflict',reference_id='r',
        source=dict(raw_reference='Author. Title. doi:10.1234/a'),finding='The DOI identifies a different work.',
        rectangles=[dict(page_index=0,x0=40,y0=50,x1=100,y1=60)])
    summary=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=[f])
    assert len(summary['academic_practice'])==1
    assert not summary['reference_formatting']
    assert not summary['evidence']
    panel=_render_reference_panel_template(f,1)
    assert 'Submitted-Link Issue' in panel and 'issue-heading formatting' not in panel
    surface=dict(page_dimensions=[dict(page_index=0,width=612,height=792)],page_href_template='p-{page_index}')
    html,_=_render_continuous_paper(surface,[],[f])
    soup=BeautifulSoup(html,'html.parser')
    assert soup.select('path.submitted-link-marker')
    assert not soup.select('rect.reference-formatting-hit')
    from app.services.evidence_report import _formatting_overlaps
    assert not _formatting_overlaps({'paper_location':{'rectangles':f['rectangles']}},f)
    from app.services.report_export import _render_pdf
    with fitz.open() as doc:
        doc.new_page().insert_text((40,60),'Author. Title. doi:10.1234/a')
        rendered,_=_render_pdf(doc.tobytes(),citations=[],reference_practice=[f],export_binding='test',view=None)
    with fitz.open(stream=rendered,filetype='pdf') as doc:
        assert 'Submitted-Link Issue' in '\n'.join(p.get_text() for p in doc)
        strokes=[d['color'] for d in doc[0].get_drawings() if d['color']]
        assert any(all(abs(a-b)<.001 for a,b in zip(color,(.463,.318,.659))) for color in strokes)
        assert doc[0].get_links()


def test_placeholder_remains_orange():
    f=dict(finding_type='reference_identifier_placeholder',reference_id='r',source={'raw_reference':'Author. arXiv:2305.XXXX','author':'Author','year':'','title':''},
        finding='The reference contains an unfinished identifier placeholder.',rectangles=[dict(page_index=0,x0=40,y0=50,x1=100,y1=60)])
    summary=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=[f])
    assert len(summary['reference_formatting'])==1
    assert not summary['evidence']
    assert 'issue-heading formatting' in _render_reference_panel_template(f,1)


def test_duplicate_doi_diagnostic_keeps_distinct_url_and_missing_page():
    from app.services.submitted_link_display import without_repeated_identifier_conflicts
    from app.services.submitted_links import binding
    from copy import deepcopy
    same=dict(finding_type='submitted_link_issue',reference_id='r',link_outcome='destination_conflict',
        submitted_link_observations=[dict(request_sha256=binding('https://doi.org/10.1234/a'))])
    other=deepcopy(same);other['submitted_link_observations'][0]['request_sha256']=binding('https://example.org/other')
    missing={**same,'link_outcome':'missing_page'}
    credibility=[dict(finding_type='reference_identifier_conflict',reference_id='r',source={'doi':'10.1234/a'})]
    assert without_repeated_identifier_conflicts([same,other,missing],credibility)==[other,missing]
