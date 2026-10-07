import hashlib
from types import SimpleNamespace

import httpx
import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import SourceResolver
from app.services.web_fetch_diagnostics import WebFetchDiagnostic, read_web_fetch_diagnostic


def fetch(monkeypatch, body, *, title='History of cinema', author=None, content_type='text/html'):
    response = httpx.Response(200, text=body, headers={'content-type': content_type},
        request=httpx.Request('GET', 'https://example.org/private-sentinel'))
    monkeypatch.setattr('app.services.source_resolver.safe_request', lambda *a, **k: response)
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch(
        str(response.url), title, expected_author=author)
    diagnostic = read_web_fetch_diagnostic(result)
    assert diagnostic.observed_content_sha256 == hashlib.sha256(response.content).hexdigest()
    assert 'private-sentinel' not in diagnostic.model_dump_json()
    assert body not in diagnostic.model_dump_json()
    return result, diagnostic


@pytest.mark.parametrize('body,reason', [
    ('<title>Unrelated chemistry methods</title>', 'page_title_mismatch_unconfirmed'),
    ('<title>电影史</title>', 'cross_script_title_unresolved'),
])
def test_wrong_page_is_not_automatically_wrong_work(monkeypatch, body, reason):
    result, diagnostic = fetch(monkeypatch, body)
    assert not result.success
    assert diagnostic.reason == reason
    assert diagnostic.outcome == 'identity_unconfirmed'


@pytest.mark.parametrize('body', ['<title>Sign in</title>', '<title>Example Visitor System</title>'])
def test_a_log_in_or_visitor_wall_is_refused_access(monkeypatch, body):
    # 2026-10-07: a wall standing in for the page refuses automated access.
    result, diagnostic = fetch(monkeypatch, body)
    assert not result.success
    assert (diagnostic.reason, diagnostic.outcome) == ('access_wall', 'access_restricted')


def test_pdf_requires_separate_route(monkeypatch):
    result, diagnostic = fetch(monkeypatch, '%PDF-1.7 fixture', content_type='application/pdf')
    assert diagnostic.reason == 'pdf_route_required'
    assert diagnostic.outcome == 'not_attempted'
    assert not result.success


def test_empty_or_javascript_page_is_unresolved(monkeypatch):
    monkeypatch.setattr('trafilatura.extract', lambda *a, **k: None)
    result, diagnostic = fetch(monkeypatch, '<title>History of cinema</title><script>load()</script>')
    assert diagnostic.reason == 'readable_text_unavailable'
    assert diagnostic.outcome == 'identity_unconfirmed'


@pytest.mark.parametrize('status,expected', [(403, 'access_restricted'), (451, 'access_restricted'), (500, 'fetch_unavailable')])
def test_http_errors_keep_operational_reason(monkeypatch, status, expected):
    response = httpx.Response(status, request=httpx.Request('GET', 'https://example.org/secret'))
    def request(*a, **k): response.raise_for_status()
    monkeypatch.setattr('app.services.source_resolver.safe_request', request)
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch('https://example.org/secret')
    diagnostic = read_web_fetch_diagnostic(result)
    assert diagnostic.reason == expected and diagnostic.observed_content_sha256 is None
    assert 'secret' not in diagnostic.model_dump_json()


def test_timeout_is_not_identity_rejection(monkeypatch):
    def request(*a, **k): raise httpx.ReadTimeout('sensitive response details')
    monkeypatch.setattr('app.services.source_resolver.safe_request', request)
    result = SourceResolver.__new__(SourceResolver)._try_web_fetch('https://example.org/test')
    assert read_web_fetch_diagnostic(result).reason == 'transport_failure'


@pytest.mark.parametrize('expected_author,reason', [
    ('Alex Morgan', 'bibliographic_identity_confirmed'),
    ('Taylor Brown', 'bibliographic_fields_conflict'),
    (None, 'bibliographic_identity_unconfirmed'),
])
def test_identity_outcome_is_content_bound(monkeypatch, expected_author, reason):
    monkeypatch.setattr('trafilatura.extract', lambda *a, **k: 'A history of cinema and film archives. ' * 30)
    result, diagnostic = fetch(monkeypatch,
        '<title>History of cinema</title><meta name="citation_title" content="History of cinema">'
        '<meta name="citation_author" content="Alex Morgan">', author=expected_author)
    assert diagnostic.reason == reason


@pytest.mark.parametrize('author,outcome', [('Alex Morgan', 'confirmed'),
    ('Taylor Brown', 'bibliographic_conflict'), (None, 'unconfirmed')])
@pytest.mark.parametrize('readable', [True, False])
def test_existing_verifier_populates_only_observed_destination(monkeypatch, author, outcome, readable):
    from app.services.schemas import ParsedReference
    from app.services.submitted_links import ACTIVE_LINKS, initial_observations, observe_request, response_observed
    url = 'https://example.org/source'
    ref = ParsedReference(reference_id='r', url=url, title='History of cinema')
    rows = initial_observations(ref)
    body = ('<title>History of cinema</title><meta name="citation_title" content="History of cinema">'
            '<meta name="citation_author" content="Alex Morgan">')
    @observe_request
    def request(url, **kwargs):
        response_observed(url, 200)
        return httpx.Response(200, text=body, request=httpx.Request('GET', url))
    monkeypatch.setattr('app.services.source_resolver.safe_request', request)
    monkeypatch.setattr('trafilatura.extract', lambda *a, **k: 'A history of cinema and film archives. ' * 30 if readable else None)
    token = ACTIVE_LINKS.set(rows)
    try:
        result = SourceResolver.__new__(SourceResolver)._try_web_fetch(url, ref.title, expected_author=author)
    finally:
        ACTIVE_LINKS.reset(token)
    observation = rows[0].requests[0]
    assert observation.destination_identity == outcome
    assert observation.identity_evidence_sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert observation.admitted_content == 'not_assessed'
    if not readable:
        assert not result.success and result.representation is None
        assert 'accepted_representation_sha256' not in result.metadata
        assert observation.page_observation == 'readable_text_unavailable'


def test_source_kind_failure_is_separate(monkeypatch):
    monkeypatch.setattr('trafilatura.extract', lambda *a, **k: 'A history of cinema and film archives. ' * 30)
    monkeypatch.setattr('app.services.source_resolver.compare_source_kinds',
        lambda *a: SimpleNamespace(verdict='incompatible', reason='incompatible kind'))
    result, diagnostic = fetch(monkeypatch, '<title>History of cinema</title>')
    assert diagnostic.outcome == 'type_unconfirmed'


@pytest.mark.parametrize('value', [None, {'reason': 'invented'},
    {'reason': 'bibliographic_fields_conflict'},
    {'reason': 'transport_failure', 'url': 'https://example.org/private'}])
def test_old_or_malformed_diagnostics_fail_closed(value):
    result = RetrievalResult(source_name='fixture', success=False,
        error='Page title does not match', metadata={'web_fetch_diagnostic': value})
    assert read_web_fetch_diagnostic(result).outcome == 'unknown'
