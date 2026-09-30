"""Possible matches are reference-local evidence, never admitted library sources."""
from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime, timezone
import hashlib

import pytest

from app.services import source_resolver as resolver_module
from app.services.source_resolver import SourceResolver, SourceResolutionError, _PROVISIONAL_CANDIDATES
from app.services.retrieval.base import RetrievalResult
from app.services.file_safety import SafetyVerdict
from app.services.verification_evidence import AuthorizedRepresentation, ClaimEvidence, build_passage_evidence
from app.services.evidence_package import build_evidence_package


@pytest.mark.parametrize('transient', [True, False])
def test_provisional_copy_scrubs_transient_location_without_losing_source_or_accounting(monkeypatch, transient):
    import json
    from dataclasses import asdict
    from app.services.search.transient import BRAVE_TRANSIENT_POLICY, finalize_transient_brave
    from app.services import pdf_verifier
    monkeypatch.setattr(resolver_module, 'inspect_uploaded_pdf', lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN))
    monkeypatch.setattr(pdf_verifier, '_extract_title_from_first_page', lambda _: None)
    url = 'https://example.org/transient-discovery-control.pdf'
    attempt = {'provider': 'brave', 'provider_calls': 1, 'latency_seconds': .25,
               'outcome': 'results', 'cost_usd': None}
    result = RetrievalResult(source_name='web_search', success=True,
        full_text=b'%PDF-source-control', full_text_url=url,
        error='Earlier request: ' + url, abstract='Discovery snippet control', metadata={
            'search_retention_policy': BRAVE_TRANSIENT_POLICY if transient else None,
            'search_attempts': [attempt],
            'location_attempts': [{'url': url, 'outcome': 'identity_unconfirmed'}]})
    original_representation = result.representation
    original_representation.metadata = {'discovery_location': url}
    result.parent_representation = replace(original_representation)
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        object.__new__(SourceResolver)._retain_provisional_candidate(result,
            confidence='medium', reason='Year uncertain', completeness='uncertain',
            text_quality='digital', kind_verdict='compatible')
        candidate = _PROVISIONAL_CANDIDATES.get()[0]
        # Copies are clean before any exception/cancellation can unwind the caller.
        if transient:
            assert url not in json.dumps(asdict(candidate), default=str)
            assert 'Discovery snippet control' not in json.dumps(asdict(candidate), default=str)
            assert candidate.metadata['search_retention_policy'] == BRAVE_TRANSIENT_POLICY
            assert candidate.metadata['transient_details_discarded'] is True
            assert candidate.locations == []
            assert 'transient_search_audits' not in candidate.metadata
        else:
            assert candidate.representation.source_url == url
            assert candidate.metadata['provisional_source']['source_url'] == url
        assert original_representation.source_url == url  # No premature mutation of active acquisition.
        finalize_transient_brave(result)
        finalize_transient_brave(candidate)
        assert result.metadata['search_attempts'] == [attempt]
        if transient:
            audit = result.metadata['transient_search_audits'][0]
            assert audit['candidate_count'] == audit['unresolved'] == 1
            assert audit['identity_established'] == 0
            assert url not in json.dumps(asdict(candidate), default=str)
        assert candidate.full_text == b'%PDF-source-control'
        assert candidate.metadata['provisional_source']['library_admission'] is False
        assert candidate.metadata['provisional_source']['content_sha256'] == hashlib.sha256(candidate.full_text).hexdigest()
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)


@pytest.mark.parametrize('confidence,quality,kind,reason,clean,expected', [
    ('medium','digital','compatible','Year differs',True,1),
    ('medium','scan_ocr','unknown','Title uncertain',True,1),
    ('low','digital','compatible','No title observed',True,0),
    ('rejected','digital','compatible','Wrong source',True,0),
    ('medium','pure_scan','compatible','Title uncertain',True,0),
    ('medium','digital','incompatible','Wrong component',True,0),
    ('medium','digital','compatible','Different edition',True,0),
    ('medium','digital','compatible','Year differs',False,0),
])
def test_reference_local_candidate_boundary(monkeypatch, confidence, quality, kind, reason, clean, expected):
    monkeypatch.setattr(resolver_module, 'inspect_uploaded_pdf', lambda _: SimpleNamespace(
        verdict=SafetyVerdict.CLEAN if clean else SafetyVerdict.REJECTED))
    resolver = object.__new__(SourceResolver)
    result = RetrievalResult(source_name='test', success=False, full_text=b'%PDF-test')
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        resolver._retain_provisional_candidate(result, confidence=confidence, reason=reason,
            completeness='uncertain', text_quality=quality, kind_verdict=kind)
        assert len(_PROVISIONAL_CANDIDATES.get()) == expected
        if expected:
            candidate = _PROVISIONAL_CANDIDATES.get()[0]
            assert candidate.success
            assert candidate.metadata['provisional_source']['library_admission'] is False
            assert 'accepted_representation_sha256' not in candidate.metadata
            result.set_representation(replace(result.representation, content=b'changed'))
            assert candidate.full_text == b'%PDF-test'
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)


def test_possible_match_retains_passages_not_quotation_or_relationship_findings():
    content = b'Careful checking improves accuracy.'
    source = AuthorizedRepresentation(
        representation_id='verification-run:test', canonical_work_id='test',
        content_object_id='transient', content_sha256=hashlib.sha256(content).hexdigest(),
        content=content, representation_kind='plain_text', media_type='text/plain',
        provenance='test', scope_type='personal_owner', scope_id='test',
        identity_verdict='possible_match', identity_confidence=.5,
        completeness_verdict='uncertain', text_quality='digital', edition_or_version=None,
        created_at=datetime.now(timezone.utc), admitted_at=None, verification_run_id='test')
    text = '“Careful checking improves accuracy.” (Smith, 2020).'
    claim = ClaimEvidence(claim_id='test', paper_version_id='paper', text=text,
        reference_ids=['r1'], citation_marker='(Smith, 2020)',
        citation_marker_type='parenthetical', passage_start=0, passage_end=len(text))
    artifact = build_passage_evidence(source, claim=claim)
    assert artifact.passages
    assert artifact.source_identity.status == 'uncertain'
    assert artifact.quotation_check.outcome == 'source_identity_unconfirmed'
    assert artifact.locator_check.status == 'not_assessable'
    assert artifact.verdict == 'not_assessed'
    package = build_evidence_package(artifact)
    assert package.source_identity.status == 'uncertain'
    assert package.quotation_check.status == 'not_assessable'


@pytest.mark.parametrize('outcome', ['error', 'abstract', 'confirmed'])
def test_fallback_only_after_routes_finish_and_never_replaces_confirmed(monkeypatch, outcome):
    resolver = object.__new__(SourceResolver)
    fallback = RetrievalResult(source_name='possible', success=True, full_text=b'%PDF-candidate',
        metadata={'provisional_source': {'policy_version': 'submission-possible-match-v1'}})
    confirmed = RetrievalResult(source_name='confirmed', success=True, full_text=b'%PDF-confirmed')
    def resolve(**kwargs):
        _PROVISIONAL_CANDIDATES.get().append(fallback)
        if outcome == 'error':
            raise SourceResolutionError('bounded routes finished without an accepted source')
        return confirmed if outcome == 'confirmed' else RetrievalResult(
            source_name='abstract', success=True, abstract='Abstract only')
    monkeypatch.setattr(resolver, 'resolve', resolve)
    monkeypatch.setattr(resolver, '_enrich_book_editions', lambda: None)
    monkeypatch.setattr(resolver, '_discovery_artifacts', lambda: ({'complete':False}, {'outcome':'search_incomplete'}))
    from app.services.schemas import ParsedReference
    result = resolver.resolve_reference(ParsedReference(reference_id='r1', title='A work',
        author='Smith', year='2020', source_kind='journal_article', raw_ref='Smith. A work.'))
    assert result is (confirmed if outcome == 'confirmed' else fallback)
    assert result.metadata['reference_discovery']['outcome'] == 'search_incomplete'
    assert _PROVISIONAL_CANDIDATES.get() is None


@pytest.mark.parametrize('observed', ['History of a Different Work', 'United Kingdom', 'Economics'])
def test_visible_other_title_cannot_be_rescued_by_medium_score(monkeypatch, observed):
    from app.services import pdf_verifier
    monkeypatch.setattr(resolver_module, 'inspect_uploaded_pdf', lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN))
    monkeypatch.setattr(pdf_verifier, '_extract_title_from_first_page', lambda _: observed)
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        object.__new__(SourceResolver)._retain_provisional_candidate(
            RetrievalResult(source_name='test',success=True,full_text=b'%PDF-test'),
            confidence='medium',reason='Title mentioned in body',completeness='complete',
            text_quality='digital',kind_verdict='unknown',expected_title='Storytelling in the New Hollywood')
        assert not _PROVISIONAL_CANDIDATES.get()
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)


@pytest.mark.parametrize('inspection', [
    {'identity':'different_work'},
    {'representation_role':'catalog_or_listing'},
    {'differences':[{'field':'edition'}]},
    {'differences':[{'field':'identifier'}]},
])
def test_known_inspection_conflicts_remain_excluded(inspection):
    token = _PROVISIONAL_CANDIDATES.set([])
    try:
        object.__new__(SourceResolver)._retain_provisional_candidate(
            RetrievalResult(source_name='test',success=True,full_text=b'%PDF-test',
                metadata={'source_inspection':inspection}),
            confidence='medium',reason='Needs review',completeness='complete',
            text_quality='digital',kind_verdict='unknown')
        assert not _PROVISIONAL_CANDIDATES.get()
    finally:
        _PROVISIONAL_CANDIDATES.reset(token)
