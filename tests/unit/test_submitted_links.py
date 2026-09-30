import hashlib
import json
import socket
import ssl
from types import SimpleNamespace

import httpx
import pytest

from app.services import safe_fetch
from app.services.schemas import ParsedReference
from app.services.submitted_links import (
    ACTIVE_LINKS, ACTIVE_REQUEST, SubmittedLink, initial_observations,
    observe_reference, project_observations,
)
from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import SourceResolutionError


URL='https://source.example/article?token=PRIVATE_SENTINEL'


@pytest.fixture
def ref():
    return ParsedReference(reference_id='r1',url=URL,title='Fixture',doi='10.1234/example')


def fixture_transport(monkeypatch,handler):
    client=httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(safe_fetch.httpx,'Client',lambda **kwargs:client)
    monkeypatch.setattr(safe_fetch,'_validate_url',lambda url:None)
    monkeypatch.setattr(safe_fetch,'_validate_connected_peer',lambda *a,**k:None)


@pytest.mark.parametrize('status,outcome',[(200,'response'),(404,'not_found'),(410,'removed'),
    (401,'authentication_required'),(407,'proxy_authentication_required'),(403,'access_refused'),
    (429,'rate_limited'),(500,'server_failure'),(503,'server_failure'),(451,'legal_restriction_reported')])
def test_actual_http_status_and_serialization(monkeypatch,ref,status,outcome):
    fixture_transport(monkeypatch,lambda req:httpx.Response(status,content=b'body',request=req))
    @observe_reference
    def resolve(self,reference):
        try: safe_fetch.safe_request(reference.url)
        except httpx.HTTPStatusError: pass
        return RetrievalResult(source_name='alternative',success=True)
    result=resolve(None,ref)
    payload=json.dumps(result.metadata['submitted_link_observations'])
    assert 'PRIVATE_SENTINEL' not in payload and '/article' not in payload
    rows=[SubmittedLink.model_validate(r) for r in json.loads(payload)]
    request=rows[0].requests[0]
    assert request.http_status==status and request.outcome==outcome
    assert request.completed_at>=request.started_at and request.elapsed_seconds>=0
    assert request.destination_identity==request.admitted_content=='not_assessed'
    assert rows[1].state=='not_checked'  # Metadata/alternative success never checks DOI.
    assert ACTIVE_LINKS.get() is ACTIVE_REQUEST.get() is None
    assert request.response_sha256==hashlib.sha256(b'body').hexdigest()


@pytest.mark.parametrize('kind,outcome',[('timeout','timeout'),('dns','dns_failure'),
    ('tls','tls_failure'),('connect','connection_failure'),('safety','safety_refused')])
def test_typed_failures_without_exception_text(monkeypatch,ref,kind,outcome):
    def handler(request):
        if kind=='timeout': raise httpx.ReadTimeout('PRIVATE_SENTINEL')
        if kind=='dns': raise httpx.ConnectError('PRIVATE_SENTINEL') from socket.gaierror()
        if kind=='tls': raise httpx.ConnectError('PRIVATE_SENTINEL') from ssl.SSLError()
        if kind=='safety': raise safe_fetch.UnsafeUrlError('PRIVATE_SENTINEL')
        raise httpx.ConnectError('PRIVATE_SENTINEL')
    fixture_transport(monkeypatch,handler)
    @observe_reference
    def resolve(self,reference):
        try: safe_fetch.safe_request(reference.url)
        except Exception: raise SourceResolutionError('source unavailable') from None
    with pytest.raises(SourceResolutionError) as caught:
        resolve(None,ref)
    rows=caught.value.submitted_link_observations
    assert rows[0]['requests'][0]['outcome']==outcome
    assert rows[0]['requests'][0]['http_status'] is None
    assert 'PRIVATE_SENTINEL' not in json.dumps(rows)


def test_redirect_status_and_loop(monkeypatch,ref):
    fixture_transport(monkeypatch,lambda req:httpx.Response(302,headers={'location':URL},request=req))
    rows=initial_observations(ref); token=ACTIVE_LINKS.set(rows)
    try:
        with pytest.raises(httpx.TooManyRedirects):
            safe_fetch.safe_request(URL,max_redirects=1)
    finally: ACTIVE_LINKS.reset(token)
    request=rows[0].requests[0]
    assert request.outcome=='redirect_limit' and len(request.hops)==2
    assert all(h.http_status==302 for h in request.hops)


def test_early_reuse_historical_uncited_and_snapshot_mismatch(ref):
    @observe_reference
    def resolve(self,reference): return RetrievalResult(source_name='reuse',success=True)
    rows=resolve(None,ref).metadata['submitted_link_observations']
    assert all(r['state']=='not_checked' for r in rows)
    saved=[{'reference_id':'r1','submitted_link_observations':rows}]
    assert project_observations([ref],json.loads(json.dumps(saved)))==rows
    assert all(r['state']=='historical_unknown' for r in project_observations([ref],[{'reference_id':'r1'}]))
    assert all(r['state']=='not_checked' for r in project_observations([ref],[]))
    changed=ref.model_copy(update={'title':'Changed snapshot'})
    assert all(r['state']=='historical_unknown' for r in project_observations([changed],saved))


def test_equivalent_doi_url_records_both_without_extra_requests(monkeypatch,ref):
    ref=ref.model_copy(update={'url':'https://doi.org/'+ref.doi})
    calls=[]
    def handler(req):
        calls.append(req)
        return httpx.Response(200,content=b'login shell',request=req)
    fixture_transport(monkeypatch,handler)
    rows=initial_observations(ref); token=ACTIVE_LINKS.set(rows)
    try: safe_fetch.safe_request(ref.url)
    finally: ACTIVE_LINKS.reset(token)
    assert len(calls)==1 and all(len(r.requests)==1 for r in rows)
    assert rows[0].request_sha256==rows[1].request_sha256
    assert all(r.requests[0].destination_identity=='not_assessed' for r in rows)


def test_unrelated_discovery_request_not_captured(monkeypatch,ref):
    fixture_transport(monkeypatch,lambda req:httpx.Response(200,content=b'PRIVATE_SENTINEL',request=req))
    rows=initial_observations(ref); token=ACTIVE_LINKS.set(rows)
    try: safe_fetch.safe_request('https://other.example/discovery')
    finally: ACTIVE_LINKS.reset(token)
    assert all(r.state=='not_checked' and not r.requests for r in rows)


def test_observation_does_not_bypass_redirect_safety(monkeypatch,ref):
    calls=[]
    def handler(req):
        calls.append(req)
        return httpx.Response(302,headers={'location':'http://127.0.0.1/private'},request=req)
    fixture_transport(monkeypatch,handler)
    def validate(url):
        if url.startswith('http://127.'):
            raise safe_fetch.UnsafeUrlError('blocked private destination')
    monkeypatch.setattr(safe_fetch,'_validate_url',validate)
    rows=initial_observations(ref); token=ACTIVE_LINKS.set(rows)
    try:
        with pytest.raises(safe_fetch.UnsafeUrlError): safe_fetch.safe_request(ref.url)
    finally: ACTIVE_LINKS.reset(token)
    assert len(calls)==1 and rows[0].requests[0].outcome=='safety_refused'
    assert rows[0].requests[0].http_status==302
    assert '127.0.0.1' not in rows[0].model_dump_json()


def test_invalid_observation_is_not_projected_as_checked(ref):
    rows=[r.model_dump(mode='json') for r in initial_observations(ref)]
    rows[0]['state']='observed'
    rows[0]['private_headers']={'authorization':'PRIVATE_SENTINEL'}
    result=project_observations([ref],[{'reference_id':ref.reference_id,
                                     'submitted_link_observations':rows}])
    assert all(r['state']=='historical_unknown' for r in result)
    assert 'PRIVATE_SENTINEL' not in json.dumps(result)


@pytest.mark.parametrize('mutation', ['entry_request', 'nested_request', 'both_requests', 'infinite_elapsed'])
def test_wrong_request_binding_and_invalid_timing_abstain(monkeypatch, ref, mutation):
    fixture_transport(monkeypatch, lambda req: httpx.Response(200, content=b'body', request=req))
    rows = initial_observations(ref)
    token = ACTIVE_LINKS.set(rows)
    try:
        safe_fetch.safe_request(ref.url)
    finally:
        ACTIVE_LINKS.reset(token)
    payload = [r.model_dump(mode='json') for r in rows]
    other = hashlib.sha256(b'unrelated request').hexdigest()
    if mutation in {'entry_request', 'both_requests'}:
        payload[0]['request_sha256'] = other
    if mutation in {'nested_request', 'both_requests'}:
        payload[0]['requests'][0]['request_sha256'] = other
    if mutation == 'infinite_elapsed':
        payload[0]['requests'][0]['elapsed_seconds'] = float('inf')
    result = project_observations([ref], [{'reference_id': ref.reference_id,
        'submitted_link_observations': payload}])
    assert all(r['state'] == 'historical_unknown' and not r['requests'] for r in result)


def test_connected_peer_failure_preserves_status_not_body(monkeypatch, ref):
    fixture_transport(monkeypatch, lambda req: httpx.Response(200, content=b'private body', request=req))
    def reject(*args, **kwargs):
        raise safe_fetch.UnsafeUrlError('private detail', reason='non_public_connected_peer')
    monkeypatch.setattr(safe_fetch, '_validate_connected_peer', reject)
    rows = initial_observations(ref)
    token = ACTIVE_LINKS.set(rows)
    try:
        with pytest.raises(safe_fetch.UnsafeUrlError):
            safe_fetch.safe_request(ref.url)
    finally:
        ACTIVE_LINKS.reset(token)
    request = rows[0].requests[0]
    assert request.http_status == 200 and request.response_sha256 is None
    assert request.safety_reason == 'non_public_connected_peer'
    assert request.outcome == 'safety_refused'
    assert 'private detail' not in rows[0].model_dump_json()
