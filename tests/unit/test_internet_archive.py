import pytest
from app.services.retrieval.internet_archive import item_identifier, locations_from_metadata

URL='https://archive.org/details/example/page/19/mode/2up'
def metadata():
    return {'metadata':{'identifier':'example'}, 'files':[
        {'name':'book.pdf','format':'Text PDF','size':'100'}]}

def test_public_file_and_bounded_url():
    r=locations_from_metadata(metadata(),URL,max_bytes=1000)
    assert len(r)==1 and r[0].url=='https://archive.org/download/example/book.pdf'
    assert r[0].landing_page_url==URL
    assert item_identifier('https://archive.org.evil/details/example') is None
    assert item_identifier('https://user@archive.org/details/example') is None

@pytest.mark.parametrize('key',['access-restricted-item','is_dark','noindex'])
def test_restrictions(key):
    m=metadata();m['metadata'][key]='true'
    with pytest.raises(ValueError,match='restricted'):locations_from_metadata(m,URL,max_bytes=1000)

@pytest.mark.parametrize('change',[{'private':'true'},{'size':'1001'},{'size':'bad'},
                                  {'name':'../book.pdf'},{'name':'x\\book.pdf'},{'format':'Unknown'}])
def test_file_controls(change):
    m=metadata();m['files'][0].update(change)
    assert locations_from_metadata(m,URL,max_bytes=1000)==[]

def test_item_binding_and_budget():
    m=metadata();m['metadata']['identifier']='different'
    with pytest.raises(ValueError):locations_from_metadata(m,URL,max_bytes=1000)
    m=metadata();m['files']=[{'name':f'{i}.pdf','format':'Text PDF','size':100} for i in range(5)]
    assert len(locations_from_metadata(m,URL,max_bytes=1000))==2

@pytest.mark.parametrize('accepted',[True,False])
def test_resolver_keeps_validation_gate(monkeypatch,accepted):
    from unittest.mock import Mock
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval.base import RetrievalResult, AcquisitionLocation
    resolver=object.__new__(SourceResolver)
    monkeypatch.setattr('app.services.retrieval.internet_archive.public_pdf_locations',
        lambda *a,**k:locations_from_metadata(metadata(),URL,max_bytes=1000))
    resolver._safe_download=Mock(return_value=b'%PDF-test')
    resolver._preflight_acquired_representation=Mock(return_value=(accepted,'acquired' if accepted else 'identity_rejected','control'))
    result=RetrievalResult(source_name='control',success=False,locations=[AcquisitionLocation(url=URL,provider='control')])
    assert resolver._acquire_from_locations(result)==accepted
    resolver._preflight_acquired_representation.assert_called_once()
    assert result.metadata['location_attempts'][0]['reason_code']=='archive_files_discovered'
    if not accepted:assert result.representation is None

def test_cited_item_uses_graph_and_preserves_failure(monkeypatch):
    from unittest.mock import Mock
    from app.services.source_resolver import SourceResolver
    resolver=object.__new__(SourceResolver)
    resolver._acquire_from_locations=Mock(return_value=False)
    blocked=Mock(side_effect=AssertionError('catalog page must not be evidence'))
    monkeypatch.setattr('app.services.source_resolver.safe_request',blocked)
    result=resolver._try_web_fetch(URL,'Expected work',expected_author='Author',expected_year='1927')
    assert not result.success and result.error=='archive_source_not_acquired'
    assert result.locations[0].url==URL
    assert resolver._acquire_from_locations.call_args.kwargs['expected_title']=='Expected work'
    blocked.assert_not_called()

def test_replacement_keeps_original_url_failure():
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval.base import RetrievalResult
    resolver=object.__new__(SourceResolver)
    result=RetrievalResult(source_name='internet_archive',success=True,
        title='Example work',authors=['Author'],year='1927',full_text=b'%PDF-control')
    final=resolver._finalize_resolution_result(result,None,'Example work','Author','1927','archive_source_not_acquired')
    assert final.metadata['url_failure_reason']=='archive_source_not_acquired'
    assert 'cited URL was unavailable' in final.metadata['source_note']
    assert final.metadata['source_match_confidence']=='high'

def test_archive_contract_participates_in_lookup_freshness(monkeypatch):
    from app.services.source_resolver import SourceResolver
    resolver=object.__new__(SourceResolver)
    before=resolver._lookup_policy_signature()
    monkeypatch.setattr('app.services.retrieval.internet_archive.VERSION','archive-public-files-test-change')
    assert resolver._lookup_policy_signature()!=before
