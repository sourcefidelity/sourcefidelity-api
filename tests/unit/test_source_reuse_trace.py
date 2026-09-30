from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE


@pytest.mark.parametrize('mode,expected', [
    ('disabled', 'unavailable'), ('missing_object', 'unavailable'), ('empty', 'no_match'),
])
def test_reuse_availability_is_not_a_completed_miss(monkeypatch, mode, expected):
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None if mode == 'disabled' else Mock()
    resolver._repository_session_factory = lambda: nullcontext(object())
    monkeypatch.setattr(settings, 'SOURCE_REPOSITORY_ENABLED', True)
    record = None if mode == 'empty' else SimpleNamespace(
        id='fixture', content_object=SimpleNamespace(storage_key='fixture'))
    lookup = Mock(return_value=record)
    monkeypatch.setattr('app.services.source_resolver.find_accepted_representation', lookup)
    if resolver._backend:
        resolver._backend.download.side_effect = FileNotFoundError
    result = resolver._check_local_cache(None, None, 'Fixture title')
    assert lookup.call_count == (0 if mode == 'disabled' else 1)
    token = _ACTIVE_DISCOVERY_TRACE.set({
        'reference_id':'fixture', 'expected':ExpectedBibliographicFields(title='Fixture title'),
        'required':set(), 'queries':[], 'attempts':[], 'candidates':[], 'limitations':[],
    })
    try:
        SourceResolver._record_discovery_attempt(category='durable_repository',
            provider='local_cache', result=result, required=False)
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        assert trace['attempts'][0].outcome == expected
        assert trace['attempts'][0].required is False
        assert trace['queries'][0].execution_outcome == (
            'no_results' if mode == 'empty' else 'operational_failure')
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


@pytest.mark.parametrize('change,accepted', [('none',True),('author',False),('year',False),('edition',False),('missing',False),('rejected',False)])
def test_edition_qualifier_lookup_keeps_identity_and_scope_boundaries(monkeypatch,change,accepted):
    import fitz
    from app.services.source_type import SourceKindAssessment
    resolver=SourceResolver.__new__(SourceResolver)
    resolver._backend=Mock()
    resolver._repository_session_factory=lambda:nullcontext(object())
    monkeypatch.setattr(settings,'SOURCE_REPOSITORY_ENABLED',True)
    with fitz.open() as doc:
        doc.new_page().insert_text((30,40),'Third edition' if change=='edition' else 'Second edition')
        resolver._backend.download.return_value=doc.tobytes()
    work=SimpleNamespace(display_title='Example book',year='2001' if change=='year' else '2004',
        author='Another author' if change=='author' else 'Reader, R',doi=None,work_type='monograph')
    record=SimpleNamespace(id='accepted-copy',content_object=SimpleNamespace(storage_key='fixture',media_type='application/pdf'),
        canonical_work=work,representation_kind='pdf',original_kind=None,source_url=None,completeness_verdict='complete',
        identity_confidence=.9,admission_state='accepted',provenance='user_upload',edition_or_version=None,validation_evidence={})
    lookup=Mock(side_effect=[None,None if change=='missing' else record])
    monkeypatch.setattr('app.services.source_resolver.find_accepted_representation',lookup)
    monkeypatch.setattr('app.services.source_resolver.validate_retrieved_pdf',lambda *a,**kw:SimpleNamespace(identity_confidence='rejected' if change=='rejected' else 'medium',completeness='complete'))
    result=resolver._check_local_cache(None,None,'Example book (2nd ed.)',author='Reader, R.',year='2004',expected_source_kind=SourceKindAssessment('monograph','high',()))
    assert result.success is accepted
    assert lookup.call_count==2
    assert all(call.kwargs['scope_id']==settings.SOURCE_REPOSITORY_SCOPE_ID for call in lookup.call_args_list)
