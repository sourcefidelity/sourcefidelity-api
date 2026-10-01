import pytest
from app.services.assessment_configuration import AssessmentConfiguration, assessment_link_omissions
from app.services.evidence_report import render_evidence_report_html


def test_default_off_strict_and_roundtrip():
    assert not AssessmentConfiguration().require_reference_links
    value=AssessmentConfiguration(require_reference_links=True)
    assert AssessmentConfiguration.model_validate_json(value.model_dump_json())==value
    with pytest.raises(ValueError):AssessmentConfiguration(require_reference_links='false')
    with pytest.raises(ValueError):AssessmentConfiguration(unrecognized=True)


def test_all_references_not_only_cited_and_unknown_not_missing():
    inventory={'entries':[{'reference_id':'uncited','status':'not_observed','rectangles':[]},
                          {'reference_id':'library','status':'supplied','rectangles':[]},
                          {'reference_id':'uncertain','status':'unknown','rectangles':[]}]}
    assert assessment_link_omissions({},inventory)==[]
    results=assessment_link_omissions({'require_reference_links':True},inventory)
    assert [r['reference_id'] for r in results]==['uncited']
    assert 'assessment requires' in results[0]['finding']


def test_assessment_rule_renders_distinct_from_style_rule():
    finding=assessment_link_omissions({'require_reference_links':True},{'entries':[
        {'reference_id':'r','status':'not_observed','rectangles':[{'page_index':0,'x0':72,'y0':100,'x1':250,'y1':115}]}]})[0]
    finding['source']={'raw_reference':'An uncited reference','author':'','title':'','year':''}
    view={'title':'Assessment test','citation_format':'MLA','citations':[],
          'assessment_configuration':{'require_reference_links':True},'reference_practice':[finding],
          'paper_surface':{'page_dimensions':[{'page_index':0,'width':612,'height':792}], 'page_href_template':'/page-{page_index}.png'}}
    html=render_evidence_report_html(view,csp_nonce='assessment-test-nonce')
    assert 'Assessment-Required Link Missing' in html
    assert 'not a universal citation-style rule' in html
    assert 'APA requires' not in html


def test_submission_form_off_by_default_and_authorized():
    from app.routers.check import paper_check_form
    from app.security import AuthenticatedPrincipal, PAPER_CHECK_CAPABILITY
    principal=AuthenticatedPrincipal(provider='test',subject='owner',scope_type='personal_owner',scope_id='owner',capabilities=frozenset({PAPER_CHECK_CAPABILITY}))
    response=paper_check_form(principal)
    html=response.body.decode()
    assert 'name="require_reference_links"' in html
    assert ' checked' not in html
    assert "form-action 'self'" in response.headers['content-security-policy']
    from fastapi import HTTPException
    with pytest.raises(HTTPException):paper_check_form(principal.model_copy(update={'capabilities':frozenset()}))


@pytest.mark.parametrize('value,expected,status', [
    (None, False, 202), ('true', True, 202), ('false', False, 202),
    ('invalid', None, 422),
    ('cross_origin', None, 403), ('unauthorized', None, 403),
])
def test_multipart_assessment_configuration(monkeypatch, value, expected, status):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import check
    from app.security import AuthenticatedPrincipal, PAPER_CHECK_CAPABILITY

    principal = AuthenticatedPrincipal(
        provider='test', subject='owner', scope_type='personal_owner',
        scope_id='owner', capabilities=frozenset() if value == 'unauthorized' else frozenset({PAPER_CHECK_CAPABILITY}))
    app = FastAPI()
    app.include_router(check.router, prefix='/check')
    app.dependency_overrides[check.get_authenticated_principal] = lambda: principal
    app.dependency_overrides[check.get_db] = lambda: Mock()
    app.dependency_overrides[check.get_storage_backend] = lambda: Mock()
    create = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(
        id='job', paper_version_id='paper', status='pending', stage='uploaded',
        store_only=False, upload_evidence={'assessment_configuration':
            AssessmentConfiguration(require_reference_links=kwargs['require_reference_links']).model_dump()}))
    dispatch = Mock(return_value=SimpleNamespace(id='task'))
    monkeypatch.setattr(check, 'create_paper_job', create)
    monkeypatch.setattr(check.check_paper_task, 'delay', dispatch)
    data = {} if value is None else {'require_reference_links':
        'true' if value in {'cross_origin', 'unauthorized'} else value}
    with TestClient(app) as client:
        response = client.post('/check/', data=data,
            files={'file': ('paper.pdf', b'fixture', 'application/pdf')},
            headers={'origin': 'http://other.example' if value == 'cross_origin' else 'http://testserver'})
    assert response.status_code == status
    if status == 202:
        assert create.call_args.kwargs['require_reference_links'] is expected
        assert response.json()['assessment_configuration'] == AssessmentConfiguration(
            require_reference_links=expected).model_dump()
        dispatch.assert_called_once()
    else:
        create.assert_not_called()
        dispatch.assert_not_called()
