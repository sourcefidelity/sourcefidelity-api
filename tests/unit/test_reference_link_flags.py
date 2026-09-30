from app.services.evidence_report import summary_text
from copy import deepcopy
import hashlib
import fitz
import pytest
from bs4 import BeautifulSoup
from test_submitted_link_display import observed
from app.services.submitted_link_display import reference_link_findings
from app.services.evidence_report import project_reference_flags, _render_continuous_paper, _render_reference_panel_template, _reference_view


def fixture(outcome='not_found',status=404):
    ref,rows,digest=observed()
    ref.raw_ref='Writer, J. (2020). A source. https://example.org/paper'
    rows[0].requests[0].outcome=outcome;rows[0].requests[0].http_status=status
    return ref,[r.model_dump(mode='json') for r in rows],digest


@pytest.mark.parametrize('outcome,status',[('timeout',None),('access_refused',403),('rate_limited',429),('response',200),('not_found',200)])
def test_operational_and_success_states_are_not_bad_links(outcome,status):
    ref,rows,_=fixture(outcome,status)
    assert not reference_link_findings(rows,{'r':_reference_view(ref)})


def test_missing_page_has_purple_reference_flag_and_academic_summary():
    ref,rows,_=fixture();source=_reference_view(ref)
    doc=fitz.open();p=doc.new_page();p.insert_text((40,80),ref.raw_ref)
    view={'citations':[],'submitted_link_observations':rows,'uncited_link_checks':[{'reference':source,'observations':rows}]}
    result=project_reference_flags(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
    f,=result['reference_practice'];assert f['rectangles']
    assert result['summary']['academic_practice']
    assert not result['summary']['evidence']
    html,_=_render_continuous_paper({'page_dimensions':[{'page_index':0,'width':612,'height':792}], 'page_href_template':'p-{page_index}'},[],[f])
    assert 'submitted-link-marker' in html
    panel = _render_reference_panel_template(f,1)
    assert 'missing or removed page' in panel
    assert 'HTTP 404' not in panel  # Operational disclosure was removed by owner decision.
    assert not view.get('reference_practice')


def test_wrong_link_requires_bound_identity_and_preserves_conflict_wording():
    ref,rows,digest=fixture('response',200)
    request=rows[0]['requests'][0]
    request.update(destination_identity='bibliographic_conflict',identity_fields=['title','author'],identity_evidence_sha256=digest)
    assert reference_link_findings(rows,{'r':_reference_view(ref)})[0]['link_outcome']=='destination_conflict'
    broken=deepcopy(rows);broken[0]['requests'][0]['identity_evidence_sha256']='0'*64
    assert not reference_link_findings(broken,{'r':_reference_view(ref)})
    ref.url='https://example.org/other'
    assert not reference_link_findings(rows,{'r':_reference_view(ref)})


def test_legacy_observation_cannot_flag_a_wrapped_url_prefix_as_broken():
    ref,rows,_=fixture()
    ref.raw_ref+=' /rest-of-the-address'
    before=deepcopy(rows)
    assert not reference_link_findings(rows,{'r':_reference_view(ref)})
    assert rows==before


def test_author_only_conflict_is_yellow_at_author_not_purple_at_link():
    ref,rows,digest=fixture('response',200)
    rows[0]['requests'][0].update(destination_identity='bibliographic_conflict',
        identity_fields=['author'], identity_evidence_sha256=digest,
        identity_differences=[dict(field='author',submitted='Writer, J.',destination='Other Writer')])
    source=_reference_view(ref)
    doc=fitz.open();doc.new_page().insert_text((40,80),ref.raw_ref)
    view={'citations':[], 'submitted_link_observations':rows,
          'uncited_link_checks':[{'reference':source,'observations':rows}]}
    result=project_reference_flags(view,doc,hashlib.sha256(doc.tobytes()).hexdigest())
    f,=result['reference_practice']
    assert f['finding_type']=='reference_author_conflict'
    assert f['field_difference']['submitted_value']=='Writer, J.' and f['rectangles']
    assert not result['summary']['evidence']
    assert 'author' in summary_text(result['summary']['academic_practice'][0])
    html,_=_render_continuous_paper({'page_dimensions':[{'page_index':0,'width':612,'height':792}],
                                  'page_href_template':'p-{page_index}'},[],[f])
    assert 'fill:#ffe45c' in html and 'submitted-link-marker' not in html
    assert 'Academic Practice' in _render_reference_panel_template(f,1)
