import hashlib
from datetime import datetime, timezone

from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryTrace, ReferenceRouteAttempt,
    ReferenceSearchQuery, build_reference_discovery_candidate,
)
from app.services.retrieval.base import RetrievalResult
from app.services.search.contribution import provider_contribution


def trace(outcome='not_attempted', *, qualified=False, prior=False):
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    expected = ExpectedBibliographicFields(title='A specific scholarly work', authors=['Alex Writer'], year='2020')
    queries, attempts, candidates = [], [], []
    for i, provider in enumerate(['crossref', 'exa'] if prior else ['exa']):
        category = 'academic_adapter' if provider == 'crossref' else 'bounded_web'
        query = 'a specific scholarly work'
        queries.append(ReferenceSearchQuery(query_id=f'q{i}', route_category=category,
            provider=provider, execution_provider=provider, execution_outcome='results',
            normalized_query=query, query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            provider_calls=1, latency_seconds=2, cost_usd=.007))
        attempts.append(ReferenceRouteAttempt(attempt_id=f'a{i}', route_category=category,
            provider=provider, required=True, permitted=True, query_ids=[f'q{i}'],
            started_at=now, completed_at=now, outcome='candidate_found'))
        c = build_reference_discovery_candidate(attempt_id=f'a{i}', provider=provider,
            expected=expected, result=RetrievalResult(source_name=provider, success=True,
                title=expected.title, authors=expected.authors, year=expected.year),
            discovery_provider=provider, location_url='https://source.example/article',
            acquisition_outcome=outcome, disposition_reason_code='attempt_limit_reached')
        if qualified:
            c = c.model_copy(update={'validated_identity_content_sha256': 'a'*64,
                'location_provenance': 'independently_acquired_content', 'identity_evidence_kind': 'source_representation'})
        candidates.append(c)
    return ReferenceDiscoveryTrace(reference_id='ref-control', expected=expected,
        required_route_categories=['bounded_web'], queries=queries, attempts=attempts, candidates=candidates)


def test_unattempted_plausible_candidate_is_not_a_confirmed_or_usable_gain():
    result = provider_contribution([trace()])
    assert result['candidate_records'] == result['plausible_match_records'] == 1
    assert result['confirmed_identity_references'] == result['usable_source_references'] == 0
    assert result['unattempted_reasons'] == {'attempt_limit_reached': 1}


def test_confirmed_metadata_does_not_require_fulltext_acquisition():
    result = provider_contribution([trace('unavailable', qualified=True)])
    assert result['additional_confirmed_identity_references'] == 1
    assert result['usable_source_references'] == 0


def test_usable_increment_is_separate_from_repeat_of_earlier_source():
    new = provider_contribution([trace('acquired', qualified=True)])
    repeated = provider_contribution([trace('acquired', qualified=True, prior=True)])
    assert new['additional_usable_source_references'] == 1
    assert repeated['usable_source_references'] == 1
    assert repeated['additional_usable_source_references'] == 0
    assert repeated['records_overlapping_retained_earlier_locations'] == 1


def test_repeated_trace_does_not_double_calls_latency_or_cost():
    original = trace()
    result = provider_contribution([original, original.model_dump(mode='json')])
    assert result['provider_calls'] == 1
    assert result['api_latency_seconds'] == 2
    assert result['observed_cost_usd'] == .007


def test_missing_cost_and_transient_novelty_remain_explicit_unknowns():
    value = trace()
    value.queries[0].cost_usd = None
    value.search_retention_policy = 'brave-operational-transient-v1'
    value.queries.append(value.queries[0].model_copy(update={'query_id':'brave-query',
        'execution_provider':'brave', 'provider':'brave'}))
    result = provider_contribution([value])
    assert result['calls_without_observed_cost'] == 1
    assert result['references_with_unavailable_prior_brave_results'] == 1
    assert result['observed_cost_usd'] == 0  # explicitly incomplete observed subtotal
