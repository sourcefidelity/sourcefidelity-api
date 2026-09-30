"""Exact-response binding only; no network or source-admission side effects."""
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.schemas import ParsedReference
from app.services.source_resolver import SourceResolver
from app.services.retrieval.base import RetrievalResult, SourceRepresentation, RepresentationKind
from app.services.submitted_links import (
    ACTIVE_LINKS, LinkRequest, initial_observations, binding,
    validated_response_identity, bind_authorized_admission,
)


def observation():
    url, final = 'https://example.org/download', 'https://example.org/paper.pdf'
    data = b'%PDF-1.7 fixture content; validator is mocked'
    digest = hashlib.sha256(data).hexdigest()
    ref = ParsedReference(reference_id='r', url=url)
    rows = initial_observations(ref)
    now = datetime.now(timezone.utc)
    rows[0].requests = [LinkRequest(started_at=now, completed_at=now,
        request_sha256=binding(url), destination_sha256=binding(final),
        response_sha256=digest, http_status=200, outcome='response')]
    rows[0].state = 'observed'
    return rows, final, data


@pytest.mark.parametrize('confidence,completeness,identity,accepted', [
    ('high', 'complete', 'confirmed', True),
    ('high', 'incomplete', 'confirmed', False),
    ('high', 'uncertain', 'confirmed', True),
    ('medium', 'complete', 'unconfirmed', False),
    ('rejected', 'complete', 'bibliographic_conflict', False),
])
def test_pdf_preflight_preserves_independent_identity(monkeypatch, confidence, completeness, identity, accepted):
    rows, final, data = observation()
    resolver = SourceResolver.__new__(SourceResolver)
    monkeypatch.setattr(resolver, '_durable_repository_active', lambda: False)
    monkeypatch.setattr('app.services.source_resolver.validate_retrieved_pdf', lambda *a, **k:
        SimpleNamespace(identity_confidence=confidence, reason='fixture result',
            completeness=completeness, text_quality='digital', source_kind_verdict='unknown'))
    result = RetrievalResult(source_name='fixture', success=True,
        representation=SourceRepresentation(kind=RepresentationKind.PDF,
            media_type='application/pdf', content=data, source_url=final))
    token = ACTIVE_LINKS.set(rows)
    try:
        actual, _, _ = resolver._preflight_acquired_representation(result,
            expected_doi=None, expected_title=None, expected_author=None, expected_year=None)
    finally:
        ACTIVE_LINKS.reset(token)
    assert actual is accepted
    assert rows[0].requests[0].destination_identity == identity
    assert rows[0].requests[0].admitted_content == 'not_assessed'
    assert bool(result.metadata.get('accepted_representation_sha256')) is accepted
    expected = {'incomplete': 'completeness_rejected', 'uncertain': 'completeness_uncertain'}.get(completeness, 'not_assessed')
    assert rows[0].requests[0].source_validation == expected


@pytest.mark.parametrize('change', ['none', 'url', 'content', 'replacement', 'not_accepted'])
def test_only_exact_validated_response_can_receive_authorized_admission(change):
    rows, final, data = observation()
    digest = hashlib.sha256(data).hexdigest()
    token = ACTIVE_LINKS.set(rows)
    try:
        records = validated_response_identity(final if change != 'url' else 'https://other.example/pdf',
            digest if change != 'content' else 'a'*64, 'high')
    finally:
        ACTIVE_LINKS.reset(token)
    metadata = {'submitted_response_identity': records,
        'accepted_representation_sha256': digest if change != 'replacement' else 'b'*64,
        'durable_admission': {'state': 'accepted' if change != 'not_accepted' else 'not_stored',
                             'representation_id': 'rep'}}
    result = bind_authorized_admission([r.model_dump(mode='json') for r in rows], metadata, 'rep')
    assert (result[0]['requests'][0]['admitted_content'] == 'admitted') == (change == 'none')


def test_landing_identity_is_not_admission_of_a_linked_pdf():
    import httpx
    rows, final, _ = observation()
    html = '<meta name="citation_title" content="A specific study"><meta name="citation_author" content="Alice Example">'
    digest = hashlib.sha256(html.encode()).hexdigest()
    rows[0].requests[0].response_sha256 = digest
    response = httpx.Response(200, text=html, request=httpx.Request('GET', final))
    token = ACTIVE_LINKS.set(rows)
    try:
        result = SourceResolver._landing_metadata_identity(response,
            expected_title='A specific study', expected_author='Alice Example', expected_year=None, expected_doi=None)
    finally:
        ACTIVE_LINKS.reset(token)
    assert result and rows[0].requests[0].destination_identity == 'confirmed'
    assert rows[0].requests[0].admitted_content == 'not_assessed'
    metadata = {'accepted_representation_sha256': 'c'*64,
        'durable_admission': {'state': 'accepted', 'representation_id': 'linked-pdf'}}
    projected = bind_authorized_admission([r.model_dump(mode='json') for r in rows], metadata, 'linked-pdf')
    assert projected[0]['requests'][0]['admitted_content'] == 'not_assessed'


@pytest.mark.parametrize('status,outcome', [(404, 'not_found'), (403, 'access_refused'),
    (200, 'safety_refused'), (302, 'redirect_limit')])
def test_error_response_cannot_be_promoted_by_matching_bytes(status, outcome):
    rows, final, data = observation()
    rows[0].requests[0].http_status = status
    rows[0].requests[0].outcome = outcome
    token = ACTIVE_LINKS.set(rows)
    try:
        assert not validated_response_identity(final, hashlib.sha256(data).hexdigest(), 'high')
    finally:
        ACTIVE_LINKS.reset(token)
    assert rows[0].requests[0].destination_identity == 'not_assessed'


@pytest.mark.parametrize('unavailable', [False, True])
def test_pdf_safety_gate_records_bound_reason_without_identity(monkeypatch, unavailable):
    from app.services.source_resolver import FileSafetyUnavailable, SafetyVerdict
    rows, final, data = observation()
    resolver = SourceResolver.__new__(SourceResolver)
    monkeypatch.setattr(resolver, '_durable_repository_active', lambda: True)

    def safety_check(content):
        assert content == data
        if unavailable:
            raise FileSafetyUnavailable('unavailable')
        return SimpleNamespace(verdict=SafetyVerdict.REJECTED, findings=['fixture rejection'])

    monkeypatch.setattr('app.services.source_resolver.inspect_uploaded_pdf', safety_check)
    def unexpected_validator(*args, **kwargs):
        pytest.fail('Identity must not run after a safety rejection')
    monkeypatch.setattr('app.services.source_resolver.validate_retrieved_pdf', unexpected_validator)
    result = RetrievalResult(source_name='fixture', success=True,
        representation=SourceRepresentation(kind=RepresentationKind.PDF,
            media_type='application/pdf', content=data, source_url=final))
    token = ACTIVE_LINKS.set(rows)
    try:
        accepted, outcome, _ = resolver._preflight_acquired_representation(result,
            expected_doi=None, expected_title=None, expected_author=None, expected_year=None)
    finally:
        ACTIVE_LINKS.reset(token)
    assert not accepted
    assert outcome == ('safety_unavailable' if unavailable else 'safety_rejected')
    request = LinkRequest.model_validate(rows[0].requests[0].model_dump())
    assert request.source_validation == outcome
    assert request.destination_identity == request.admitted_content == 'not_assessed'
