import hashlib
import pytest
from app.services.identity_ocr_observation import IdentityOcrObservation, _digest
from app.services.identity_ocr_fields import locate_identity_fields

def observation(text):
    source=b'bound source'
    item=IdentityOcrObservation(source_sha256=hashlib.sha256(source).hexdigest(),
        render_sha256='a'*64,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        region=(0,0,300,400),engine_version='test',renderer_version='test',text=text,
        observation_sha256='0'*64)
    values=item.model_dump();values.pop('observation_sha256')
    return item.model_copy(update={'observation_sha256':_digest(values)}),source

def test_exact_original_spans_and_non_grants():
    item,source=observation('The Original\nArticle\nJane Smith\n2001')
    result=locate_identity_fields(item,source,{'title':'The Original Article','author':'Jane Smith','year':'2001'})
    assert [f.status for f in result.fields]==['observed_wording']*3+['not_supplied']
    assert item.text[slice(*result.fields[0].spans[0])]=='The Original\nArticle'
    assert result.identity_status=='unverified' and not result.grants_admission
    assert all(f.bibliographic_role=='unverified' for f in result.fields)

def test_mention_in_bibliography_does_not_verify_identity():
    item,source=observation('Another document\nReferences\nSmith, 2001. The Original Article.')
    result=locate_identity_fields(item,source,{'title':'The Original Article'})
    assert result.fields[0].status=='observed_wording'
    assert result.identity_status=='unverified' and not result.grants_evidence_use

def test_missing_and_repeated_wording_are_not_conflicts():
    item,source=observation('Original Article\nOriginal Article\nPerson\n20010')
    result=locate_identity_fields(item,source,{'title':'Original Article','author':'Son','year':'2001'})
    assert [f.status for f in result.fields][:3]==['ambiguous','not_observed','not_observed']

def test_author_order_is_not_silently_repaired():
    item,source=observation('Jane Smith')
    assert locate_identity_fields(item,source,{'author':'Smith, J.'}).fields[1].status=='not_observed'

def test_wrong_source_and_unsupported_fields_fail():
    item,source=observation('Original Article')
    with pytest.raises(ValueError):locate_identity_fields(item,source,{'publisher':'X'})
    with pytest.raises(Exception):locate_identity_fields(item,b'other',{'title':'Original Article'})

def test_size_and_occurrence_bounds():
    item,source=observation('Original Article\n'*30)
    result=locate_identity_fields(item,source,{'title':'Original Article'})
    assert len(result.fields[0].spans)==10 and result.fields[0].status=='ambiguous'
    with pytest.raises(ValueError):locate_identity_fields(item,source,{'title':'x'*2001})


def test_authorized_consumer_rechecks_access_before_comparison(monkeypatch):
    from types import SimpleNamespace
    from app.services import identity_ocr_fields as module
    item,source=observation('Original Article')
    seen=[]
    principal=SimpleNamespace(scope_type='personal',scope_id='owner',require=lambda capability:seen.append(capability))
    def deny(*args,**kwargs):
        assert kwargs['scope_id']=='owner'
        raise PermissionError('held or inaccessible')
    monkeypatch.setattr(module,'authorize_representation',deny)
    with pytest.raises(PermissionError):
        module.compare_authorized_identity_fields(None,None,principal=principal,
            representation_id='source-id',observation=item,expected_fields={'title':'Original Article'})
    assert seen==[module.REPORT_SOURCE_CAPABILITY]
