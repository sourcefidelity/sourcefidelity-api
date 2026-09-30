from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import hashlib
import fitz
import pytest

from test_source_repository import session, MemoryStorage, _request
from app.services.publisher_preview import preview_receipt, valid_preview_receipt
from app.services.source_repository import admit_representation, find_accepted_representation, WorkIdentity
from app.services.retrieval.base import SourceRepresentation, RepresentationKind, RetrievalResult
from app.services.source_resolver import SourceResolver
from app.services.source_type import SourceKindAssessment
from app.services.file_safety import SafetyVerdict

URL = 'https://hkupress.hku.hk/image/catalog/pdf-preview/9789888139637.pdf'


@pytest.fixture
def content():
    with fitz.open() as doc:
        for _ in range(4):
            page = doc.new_page()
            page.insert_textbox((40,40,560,780), 'A sentence about a historical subject and its development. ' * 40)
        return doc.tobytes()


def receipt(content, **changes):
    args = dict(identity='high', completeness='incomplete', source_kind='monograph', text_quality='digital', cleanliness='clean')
    args.update(changes)
    return preview_receipt(content, URL, **args)


@pytest.mark.parametrize('field,value', [('identity','medium'),('identity','rejected'),('cleanliness','not_assessed'),
    ('cleanliness','rejected'),('text_quality','pure_scan'),('source_kind','journal_article'),('completeness','uncertain')])
def test_exception_requires_every_gate(content, field, value):
    assert receipt(content, **{field:value}) is None


def test_receipt_binds_content_location_and_limited_coverage(content):
    r = receipt(content)
    assert r and r['coverage'] == 'partial_text'
    assert valid_preview_receipt(r, hashlib.sha256(content).hexdigest(), URL)
    assert not valid_preview_receipt(r, '0'*64, URL)
    assert not valid_preview_receipt(r, r['content_sha256'], URL.replace('hkupress.hku.hk', 'example.org'))
    assert receipt(b'broken') is None


def test_scoped_admission_reuse_and_expiry(session, content):
    from app.models.source_repository import CanonicalWorkRecord
    old = CanonicalWorkRecord(normalized_title='a preview book',display_title='A Preview Book',
        work_type='webpage', author='Writer', year='2012')
    session.add(old); session.commit()
    request = replace(_request(completeness_verdict='incomplete'),
        work=WorkIdentity(title='A Preview Book', work_type='monograph', author='Writer', year='2012'),
        representation=SourceRepresentation(kind=RepresentationKind.PDF, media_type='application/pdf',
            content=content, source_url=URL, completeness='incomplete'),
        validation_evidence={'publisher_preview':receipt(content)})
    record = admit_representation(session, MemoryStorage(), request)
    session.commit()
    assert record.admission_state == 'accepted' and record.completeness_verdict == 'incomplete'
    assert record.canonical_work_id != old.id and old.work_type == 'webpage'
    kwargs=dict(scope_type='personal_owner', scope_id='owner-1', title='A Preview Book', work_type='monograph')
    assert find_accepted_representation(session, **kwargs).id == record.id
    assert find_accepted_representation(session, **{**kwargs,'scope_id':'other'}) is None
    record.validation_evidence = {}
    session.flush()
    assert find_accepted_representation(session, **kwargs) is None
    record.validation_evidence = request.validation_evidence
    record.expires_at = datetime.now(timezone.utc)-timedelta(seconds=1)
    session.flush()
    assert find_accepted_representation(session, **kwargs) is None


@pytest.mark.parametrize('identity,clean,expected', [('high',True,True),('medium',True,False),('high',False,False)])
def test_acquisition_preflight_never_promotes_preview_to_full_text(content, monkeypatch, identity, clean, expected):
    import app.services.source_resolver as module
    resolver = object.__new__(SourceResolver)
    monkeypatch.setattr(resolver, '_durable_repository_active', lambda:True)
    monkeypatch.setattr(resolver, '_retain_provisional_candidate', lambda *a,**k:None)
    monkeypatch.setattr(module, 'inspect_uploaded_pdf', lambda _: SimpleNamespace(
        verdict=SafetyVerdict.CLEAN if clean else SafetyVerdict.REJECTED, findings=[]))
    monkeypatch.setattr(module, 'validate_retrieved_pdf', lambda *a,**k:SimpleNamespace(
        identity_confidence=identity, completeness='complete', text_quality='digital',
        reason='fixture', source_kind_verdict='compatible', observed_source_kind='monograph'))
    result = RetrievalResult(source_name='test', success=True, metadata={},
        representation=SourceRepresentation(kind=RepresentationKind.PDF,media_type='application/pdf',content=content,source_url=URL))
    accepted,_,_ = resolver._preflight_acquired_representation(result, expected_doi=None,
        expected_title='A Preview Book', expected_author='Writer', expected_year='2012',
        expected_source_kind=SourceKindAssessment('monograph','high'))
    assert accepted == expected
    if accepted:
        assert result.representation.completeness == 'incomplete'
        assert result.metadata['publisher_preview']['coverage'] == 'partial_text'
