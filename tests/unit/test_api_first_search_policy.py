"""Frozen policy controls. Provider doubles are not live quality acceptance."""

from datetime import datetime, timezone
import hashlib
from unittest.mock import Mock

import httpx
import pytest

from app.config import settings
from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryCandidate, ReferenceDiscoveryTrace,
    ReferenceRouteAttempt, ReferenceSearchQuery, assess_reference_discovery_trace,
)
from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.base import SearchResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY, LEGACY_SEARCH_POLICY
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", True)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:4,exa:4,tavily:4")
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None)
                 for name in ("brave", "exa", "searxng", "tavily")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name or "searxng"))
    return WebSearchRetriever(health_store=Mock()), providers


def run(search):
    return search.search_reference(doi=None, title="Frozen source control", author="Writer", year="2020")


def test_provider_order_is_tier_first_and_excludes_tavily(cascade):
    search, providers = cascade
    order = []
    for name in ("brave", "exa"):
        providers[name].search.side_effect = lambda *a, n=name, **kw: order.append(n) or []
    result = run(search)
    # Exa has no filetype operator, so it receives the exact query only.
    assert order == ["brave", "brave", "exa"]
    assert [a["provider"] for a in result.metadata["search_attempts"]] == order
    providers["searxng"].search.assert_not_called()
    providers["tavily"].search.assert_not_called()


def test_full_candidate_queue_prevents_second_paid_query(cascade):
    search, providers = cascade
    providers['brave'].search.return_value = [
        SearchResult(f'https://candidate.example/{i}', 'Frozen source control', '') for i in range(5)]
    result = run(search)
    assert len(result.locations) == 5
    assert providers['brave'].search.call_count == 1
    assert providers['brave'].search.call_args.kwargs['num_results'] == 5
    providers['exa'].search.assert_not_called()  # Resolver decides escalation after inspection.


def test_second_query_requests_only_remaining_candidate_slots(cascade):
    search, providers = cascade
    providers['brave'].search.side_effect = [
        [SearchResult('https://candidate.example/first', 'Frozen source control', '')], []]
    result = run(search)
    assert [c.kwargs['num_results'] for c in providers['brave'].search.call_args_list] == [5, 4]
    assert len(result.locations) == 1


def test_zero_inspection_capacity_does_not_spend_and_stays_incomplete(cascade):
    from app.services.candidate_budget import candidate_budget_scope
    from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE
    search, providers = cascade
    with candidate_budget_scope(0):
        result = run(search)
    for p in providers.values():
        p.search.assert_not_called()
    assert {a['outcome'] for a in result.metadata['search_attempts']} == {'budget_skipped'}
    assert all(a['provider_calls'] == 0 for a in result.metadata['search_attempts'])
    trace = trace_for([(a['provider'], a['outcome']) for a in result.metadata['search_attempts']])
    assert assess_reference_discovery_trace(trace).record.outcome == 'search_incomplete'


def test_shared_budget_limits_collection_without_reset(cascade):
    from app.services.candidate_budget import candidate_budget_scope, require_source_candidate
    search, providers = cascade
    with candidate_budget_scope(2) as budget:
        require_source_candidate('https://already.example/source')
        providers['brave'].search.return_value = [SearchResult('https://candidate.example/one', 'Frozen source control', '')]
        result = run(search)
        assert providers['brave'].search.call_args.kwargs['num_results'] == 1
        assert providers['brave'].search.call_count == 1
        assert len(result.locations) == 1
        assert budget.remaining() == 1  # collection is not an inspection reservation


def test_capacity_exhausted_after_brave_skips_exa_and_preserves_transient_audit(cascade):
    from app.services.candidate_budget import candidate_budget_scope, require_source_candidate
    search, providers = cascade
    providers['brave'].search.return_value = [SearchResult('https://candidate.example/one', 'Frozen source control', '')]
    resolver = SourceResolver.__new__(SourceResolver)
    def inspect(source, result, *args, **kwargs):
        require_source_candidate(result.locations[0].url)
        result.metadata['location_attempts'] = [{'url': result.locations[0].url,
            'discovery_provider': 'brave', 'outcome': 'identity_unconfirmed'}]
        return result
    resolver._download_and_cache = inspect
    with candidate_budget_scope(1):
        result = resolver._try_source(search, None, 'Frozen source control', 'Writer', '2020')
    providers['exa'].search.assert_not_called()
    assert result.metadata['retrieval_trace'][0]['transient_search_audits']
    assert result.metadata['retrieval_trace'][-1]['search_attempts'][0]['outcome'] == 'budget_skipped'


def test_elapsed_budget_skips_required_tiers_without_calls_or_health_reset(cascade):
    from app.services.retrieval_deadline import deadline_scope
    search, providers = cascade
    before = dict(search.search_metrics)
    with deadline_scope(0):
        result = run(search)
    assert not result.success
    assert {a['provider'] for a in result.metadata['search_attempts']} == {'brave', 'exa'}
    # `budget_skipped` since 2026-09-24: the reference's own budget ran out and
    # neither provider was called, so neither may be named as having timed out
    # -- that label made Brave and Exa re-run triggers for our own deadline.
    assert all(a['outcome'] == 'budget_skipped' and a['provider_calls'] == 0
               and a['reason_code'] == 'reference_elapsed_budget_timeout'
               for a in result.metadata['search_attempts'])
    assert dict(search.search_metrics) == before
    for provider in providers.values():
        provider.search.assert_not_called()


def test_deadline_after_brave_preserves_audit_and_does_not_call_exa(cascade, monkeypatch):
    from app.services import retrieval_deadline as clock
    search, providers = cascade
    now = [0.0]
    monkeypatch.setattr(clock.time, 'monotonic', lambda: now[0])
    providers['brave'].search.return_value = [SearchResult('https://brave.example/lead', 'Frozen source control', '')]
    resolver = SourceResolver.__new__(SourceResolver)
    def acquire(source, result, *args, **kwargs):
        now[0] = 11
        result.metadata['location_attempts'] = [{'url': result.locations[0].url,
            'discovery_provider': 'brave', 'outcome': 'transport_failure'}]
        return result
    resolver._download_and_cache = acquire
    with clock.deadline_scope(10):
        result = resolver._try_source(search, None, 'Frozen source control', 'Writer', '2020')
    providers['exa'].search.assert_not_called()
    import json
    assert 'https://brave.example/lead' not in json.dumps(result.metadata)
    phases = result.metadata['retrieval_trace']
    assert phases[0]['transient_search_audits']
    assert phases[-1]['search_attempts'][0]['outcome'] == 'budget_skipped'


def test_rejected_nonempty_tier_escalates_through_shared_resolver(cascade):
    search, providers = cascade
    for name in ("brave", "exa"):
        providers[name].search.return_value = [SearchResult(f"https://{name}.example/source", "Frozen source control", "")]
    resolver = SourceResolver.__new__(SourceResolver)
    seen = []
    def acquire(source, result, *args, **kwargs):
        name = result.locations[0].metadata["search_provider"]
        seen.append(name)
        result.metadata["location_attempts"] = [{"url": result.locations[0].url,
            "discovery_provider": name, "outcome": "identity_rejected" if name == "brave" else "acquired"}]
        if name == "exa":
            result.full_text = b"validated-test-representation"
        return result
    resolver._download_and_cache = acquire
    result = resolver._try_source(search, None, "Frozen source control", "Writer", "2020")
    assert seen == ["brave", "exa"]
    assert result.full_text
    assert len(result.metadata["retrieval_trace"]) == 2


@pytest.mark.parametrize('terminal_error', [False, True])
def test_provisional_brave_handoff_preserves_tier_accounting_not_candidate_url(cascade, monkeypatch, terminal_error):
    """Exercise tier cleanup followed by the reference-local fallback return."""
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from app.services.file_safety import SafetyVerdict
    from app.services.schemas import ParsedReference
    from app.services.source_resolver import SourceResolutionError
    search, providers = cascade
    url = 'https://example.org/provisional-candidate.pdf'
    providers['brave'].search.return_value = [SearchResult(url, 'Candidate title', 'Transient snippet')]
    resolver = SourceResolver.__new__(SourceResolver)
    monkeypatch.setattr('app.services.source_resolver.inspect_uploaded_pdf',
        lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN))
    monkeypatch.setattr('app.services.pdf_verifier._extract_title_from_first_page', lambda _: None)
    captured = {}
    def acquire(source, result, *args, **kwargs):
        from app.services.retrieval.base import SourceRepresentation, RepresentationKind
        result.set_representation(SourceRepresentation(kind=RepresentationKind.PDF,
            media_type='application/pdf', content=b'%PDF-provisional-control', source_url=url))
        resolver._retain_provisional_candidate(result, confidence='medium', reason='Year uncertain',
            completeness='uncertain', text_quality='digital', kind_verdict='compatible')
        result.metadata['location_attempts'] = [{'url': url, 'outcome': 'identity_unconfirmed'}]
        result.full_text = result.representation = result.full_text_url = None
        return result
    resolver._download_and_cache = acquire
    def resolve(**kwargs):
        result = resolver._try_source(search, None, 'Frozen source control', 'Writer', '2020')
        captured.update(result.metadata)
        if terminal_error:
            raise SourceResolutionError('No confirmed source')
        return result
    resolver.resolve = resolve
    resolver._enrich_book_editions = lambda: None
    resolver._discovery_artifacts = lambda: (captured, {'outcome': 'search_incomplete'})
    result = resolver.resolve_reference(ParsedReference(reference_id='r1', title='Frozen source control',
        author='Writer', year='2020', source_kind='journal_article', raw_ref='Writer (2020). Frozen source control.'))
    assert result.full_text == b'%PDF-provisional-control'
    assert url not in json.dumps(asdict(result), default=str)
    assert result.metadata['reference_discovery']['outcome'] == 'search_incomplete'
    trace = result.metadata['reference_discovery_trace']['retrieval_trace']
    attempts = [a for tier in trace for a in tier['search_attempts']]
    for provider in ('brave', 'exa'):
        assert sum(a['provider_calls'] for a in attempts if a['provider'] == provider) == providers[provider].search.call_count
    assert all(a['latency_seconds'] >= 0 for a in attempts)
    audits = [audit for tier in trace for audit in tier['transient_search_audits']]
    assert sum(a['unresolved'] for a in audits) == 1


@pytest.mark.parametrize("failure", ["timeout", "rate_limited", "access_restricted", "response_invalid"])
def test_required_failures_remain_traced_while_next_api_runs(cascade, failure):
    search, providers = cascade
    providers["brave"].last_status = failure
    result = run(search)
    assert result.metadata["search_attempts"][0]["outcome"] == failure
    providers["brave"].search.assert_called_once()
    assert providers["exa"].search.call_count == 1


def test_brave_contract_reason_survives_execution_trace(cascade):
    search, providers = cascade
    providers["brave"].last_reason_code = "brave_web_v2_empty_web"
    result = run(search)
    attempts = [a for a in result.metadata["search_attempts"] if a["provider"] == "brave"]
    assert attempts and all(a["outcome"] == "no_results" and a["reason_code"] == "brave_web_v2_empty_web" for a in attempts)


def test_retention_permission_is_not_inferred_from_api_key(cascade, monkeypatch):
    search, providers = cascade
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", False)
    result = run(search)
    providers["brave"].search.assert_not_called()
    assert result.metadata["search_attempts"][0]["reason_code"] == "search_retention_permission_unconfirmed"


def test_budget_applies_to_primary_and_cache_does_not_spend_twice(cascade):
    search, providers = cascade
    search._escalation_limits = {"brave": 1, "exa": 0}
    first, second = run(search), run(search)
    providers["brave"].search.assert_called_once()
    providers["exa"].search.assert_not_called()
    assert any(a["outcome"] == "budget_skipped" for a in first.metadata["search_attempts"])
    assert all(a["provider_calls"] == 0 for a in second.metadata["search_attempts"])


def test_optional_searxng_uses_existing_cooldowns_without_search_or_reset(cascade):
    from app.services.search.searxng import SearXNGSearch
    search, _ = cascade
    searx = SearXNGSearch("https://search.example")
    searx.search = Mock()
    search._policy_providers["searxng"] = searx
    search._health_store.cooldown_remaining.return_value = 180
    result = run(search)
    attempts = [a for a in result.metadata["search_attempts"] if a["provider"] == "searxng"]
    assert attempts and all(a["outcome"] == "cooldown_skipped" and a["required"] is False for a in attempts)
    searx.search.assert_not_called()
    # SearXNG's health must not be reset while it is cooling down. Brave and
    # Exa are now tracked (2026-09-24), so the store is consulted for them;
    # this test's concern is SearXNG only.
    assert "searxng" not in {c.args[0] for c in search._health_store.record_success.call_args_list}
    search._health_store.claim_recovery_probe.assert_not_called()


def test_ranked_out_candidates_keep_explicit_dispositions(cascade):
    search, providers = cascade
    providers["brave"].search.return_value = [SearchResult(f"https://example.org/{i}", "Frozen source control", "") for i in range(10)]
    result = run(search)
    assert len(result.locations) == 5
    assert result.metadata["candidate_dispositions"] == []
    assert result.metadata["transient_unselected_count"] == 5


def test_transient_brave_runs_without_persistent_retention_permission(cascade, monkeypatch):
    search, providers = cascade
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    run(search)
    assert providers["brave"].search.call_count == 2
    assert not any(key[0] == "brave" for key in search._policy_query_cache)


def test_brave_response_is_not_reused_between_operations(cascade):
    search, providers = cascade
    providers["brave"].search.return_value = [SearchResult("https://discovery.example/secret", "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL")]
    first, second = run(search), run(search)
    assert providers["brave"].search.call_count == 4
    assert not search._policy_query_cache
    assert "BRAVE_TITLE_SENTINEL" not in repr(first)
    assert "BRAVE_SNIPPET_SENTINEL" not in repr(second)


@pytest.mark.parametrize("outcome", ["identity_rejected", "transport_failure", "not_attempted", "identity_unconfirmed", "type_rejected"])
def test_transient_dispositions_and_traces_do_not_retain_failed_results(cascade, outcome):
    import json
    from dataclasses import asdict
    search, providers = cascade
    url = "https://discovery.example/BRAVE_URL_SENTINEL"
    providers["brave"].search.return_value = [SearchResult(url, "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL")]
    resolver = SourceResolver.__new__(SourceResolver)
    def acquisition(source, result, *args, **kwargs):
        result.metadata["location_attempts"] = [{"url": url, "outcome": outcome, "rank": 1,
                                                 "candidate_title": "BRAVE_TITLE_SENTINEL"}]
        return result
    resolver._download_and_cache = acquisition
    result = resolver._try_source(search, None, "Frozen source control", "Writer", "2020")
    seed = trace_for(COMPLETE[:1])
    token = _ACTIVE_DISCOVERY_TRACE.set({"reference_id": seed.reference_id, "expected": seed.expected,
        "search_policy_version": API_FIRST_SEARCH_POLICY, "search_retention_policy": "brave-operational-transient-v1",
        "required": {"academic_adapter", "bounded_web"}, "queries": seed.queries,
        "attempts": seed.attempts, "candidates": []})
    try:
        resolver._record_discovery_attempt(category="bounded_web", provider="web_search", result=result, required=True)
        trace, record = resolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    serialized = json.dumps({"result": asdict(result), "trace": trace, "record": record}, default=str)
    for forbidden in ("BRAVE_URL_SENTINEL", "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL", hashlib.sha256(url.encode()).hexdigest()):
        assert forbidden not in serialized
    restored = ReferenceDiscoveryTrace.model_validate(json.loads(json.dumps(trace)))
    completion = assess_reference_discovery_trace(restored)
    assert completion.ready
    assert completion.record.outcome == ("unlocated_after_search" if outcome == "identity_rejected" else "search_incomplete")
    if outcome == "identity_rejected":
        assert record is None  # Acceptance is still suppressed.


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_transient_cleanup_on_exception_and_cancellation(cascade, error, caplog):
    import logging
    search, providers = cascade
    url = "https://discovery.example/BRAVE_URL_SENTINEL"
    providers["brave"].search.return_value = [SearchResult(url, "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL")]
    captured = []
    resolver = SourceResolver.__new__(SourceResolver)
    def acquisition(source, result, *args, **kwargs):
        captured.append(result)
        logging.getLogger("source-operation-test").warning("candidate %s", url)
        raise error(url)
    resolver._download_and_cache = acquisition
    with pytest.raises(error) as raised:
        resolver._try_source(search, None, "Frozen source control", "Writer", "2020")
    assert "BRAVE_URL_SENTINEL" not in str(raised.value)
    assert captured[0].locations == [] and captured[0].full_text_url is None
    assert "BRAVE_URL_SENTINEL" not in repr(captured)
    assert "BRAVE_URL_SENTINEL" not in caplog.text
    assert not search._policy_query_cache
    logging.getLogger("source-operation-test").warning("ordinary operation unaffected")
    assert "ordinary operation unaffected" in caplog.text


def test_independent_identity_provenance_survives_without_search_content(cascade):
    from app.services.search.transient import finalize_transient_brave
    from app.services.retrieval.base import SourceRepresentation, RepresentationKind
    search, providers = cascade
    url = "https://source.example/independently-fetched"
    failed = "https://discovery.example/FAILED_URL_SENTINEL"
    providers["brave"].search.return_value = [SearchResult(url, "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL"),
                                             SearchResult(failed, "Other", "")]
    result = run(search)
    content = b"Independently fetched and validated source content"
    digest = hashlib.sha256(content).hexdigest()
    result.set_representation(SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain",
                                                   content=content, source_url=url))
    result.metadata["location_attempts"] = [{"url": url, "outcome": "acquired",
        "candidate_title": "BRAVE_TITLE_SENTINEL", "rank": 1, "validated_identity_content_sha256": digest},
        {"url": failed, "outcome": "transport_failure"}]
    finalize_transient_brave(result)
    assert result.full_text_url == url
    assert result.metadata["location_attempts"][0]["validated_identity_content_sha256"] == digest
    assert "FAILED_URL_SENTINEL" not in repr(result)
    assert "BRAVE_TITLE_SENTINEL" not in repr(result)
    before = repr(result)
    finalize_transient_brave(result)
    assert repr(result) == before
    result.metadata["retrieval_trace"] = [{"search_attempts": result.metadata["search_attempts"],
        "location_attempts": result.metadata["location_attempts"],
        "transient_search_audits": result.metadata["transient_search_audits"]}]
    seed = trace_for(COMPLETE[:1])
    token = _ACTIVE_DISCOVERY_TRACE.set({"reference_id": seed.reference_id, "expected": seed.expected,
        "search_policy_version": API_FIRST_SEARCH_POLICY, "search_retention_policy": "brave-operational-transient-v1",
        "required": {"academic_adapter", "bounded_web"}, "queries": seed.queries,
        "attempts": seed.attempts, "candidates": []})
    try:
        SourceResolver._record_discovery_attempt(category="bounded_web", provider="web_search", result=result, required=True)
        trace, record = SourceResolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    assert record["outcome"] == "possible_match"
    trace["candidates"][0]["validated_identity_content_sha256"] = None
    assert "transient_audit_incomplete" in assess_reference_discovery_trace(trace).blocker_codes


def test_durable_admission_entry_discards_discovery_details_before_any_write(cascade):
    search, providers = cascade
    providers["brave"].search.return_value = [SearchResult("https://discovery.example/FAILED_URL_SENTINEL", "BRAVE_TITLE_SENTINEL", "")]
    result = run(search)
    resolver = SourceResolver.__new__(SourceResolver)
    # No representation: the entry still must scrub before any early return.
    resolver._persist_retrieved_representation(result, ref_doi=None, ref_title="Expected source", ref_author=None,
        ref_year=None, identity_confidence="high", identity_reason="test", downloaded_via_publisher=False, safety_report=None)
    assert "FAILED_URL_SENTINEL" not in repr(result)
    assert result.metadata["transient_search_audits"][0]["not_attempted"] == 1


def test_transient_audit_rejects_unbalanced_counts_and_extra_result_fields():
    from app.services.search.transient import TransientSearchAudit
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransientSearchAudit(candidate_count=2, identity_rejected=1)
    with pytest.raises(ValidationError):
        TransientSearchAudit(candidate_count=0, url="https://discovery.example")


def test_durable_admission_request_contains_only_independent_provenance(cascade, monkeypatch):
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    from dataclasses import asdict
    from app.services.retrieval.base import SourceRepresentation, RepresentationKind
    search, providers = cascade
    url = "https://source.example/independent-source"
    providers["brave"].search.return_value = [SearchResult(url, "BRAVE_TITLE_SENTINEL", "BRAVE_SNIPPET_SENTINEL"),
        SearchResult("https://discovery.example/FAILED_URL_SENTINEL", "Other", "")]
    result = run(search)
    content = b"Independently acquired source"
    result.set_representation(SourceRepresentation(kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain", content=content, source_url=url))
    result.metadata["location_attempts"] = [{"url": url, "outcome": "acquired", "rank": 1,
        "candidate_title": "BRAVE_TITLE_SENTINEL", "validated_identity_content_sha256": hashlib.sha256(content).hexdigest()}]
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = Mock()
    resolver._repository_session_factory = MagicMock()
    resolver._retrieval_retention_decision = lambda *a, **kw: ("public_domain", "test")
    admitted = []
    def admit(session, backend, request):
        admitted.append(request)
        return SimpleNamespace(id="test-record", admission_state="accepted")
    monkeypatch.setattr("app.services.source_resolver.admit_representation", admit)
    monkeypatch.setattr("app.services.source_resolver.commit_source_admissions", lambda _: None)
    resolver._persist_retrieved_representation(result, ref_doi=None, ref_title="Expected title", ref_author=None,
        ref_year=None, identity_confidence="high", identity_reason="validated content", downloaded_via_publisher=False, safety_report=None)
    assert len(admitted) == 1
    payload = repr(asdict(admitted[0]))
    assert "SENTINEL" not in payload
    assert url in payload
    assert admitted[0].validation_evidence["transient_search_audits"][0]["not_attempted"] == 1


def test_transient_result_query_without_accounting_cannot_complete():
    trace = trace_for([( "crossref", "no_results"), ("brave", "results"), ("exa", "no_results")])
    trace.search_retention_policy = "brave-operational-transient-v1"
    completion = assess_reference_discovery_trace(trace)
    assert not completion.ready
    assert "transient_audit_incomplete" in completion.blocker_codes


def test_transient_cleanup_preserves_operation_only_cross_tier_deduplication(cascade):
    search, providers = cascade
    failed = "https://discovery.example/same-failed-candidate"
    providers["brave"].search.return_value = [SearchResult(failed, "Source", "")]
    providers["exa"].search.return_value = [SearchResult(failed, "Source", ""),
                                           SearchResult("https://other.example/source", "Source", "")]
    resolver = SourceResolver.__new__(SourceResolver)
    seen = []
    def acquire(source, result, *args, **kwargs):
        seen.extend(location.url for location in result.locations)
        result.metadata["location_attempts"] = [{"url": location.url, "outcome": "transport_failure"} for location in result.locations]
        return result
    resolver._download_and_cache = acquire
    resolver._try_source(search, None, "Frozen source control", "Writer", "2020")
    assert seen.count(failed) == 1
    from app.services.search.transient import _ATTEMPTED_URLS
    assert _ATTEMPTED_URLS.get() is None


def test_disabled_optional_fallback_does_not_generate_background_probes(monkeypatch):
    from app.tasks import provider_recovery
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    health = Mock()
    health.incident_providers.return_value = ["searxng:brave", "duckduckgo"]
    monkeypatch.setattr(provider_recovery, "ProviderHealthStore", lambda: health)
    assert provider_recovery.probe_retrieval_provider_recovery.run()["incidents_probed"] == 0
    health.record_success.assert_not_called()
    health.record_unavailable.assert_not_called()
    health.claim_recovery_probe.assert_not_called()


def trace_for(outcomes, *, version=API_FIRST_SEARCH_POLICY, candidate=None):
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    queries, attempts = [], []
    for i, (provider, outcome) in enumerate(outcomes):
        category = "academic_adapter" if provider == "crossref" else "bounded_web"
        qid, aid = f"q-{i}", f"a-{i}"
        query = "frozen bibliographic control"
        queries.append(ReferenceSearchQuery(query_id=qid, route_category=category, provider=provider,
            execution_provider=provider, execution_outcome=outcome, normalized_query=query,
            query_sha256=hashlib.sha256(query.encode()).hexdigest()))
        attempts.append(ReferenceRouteAttempt(attempt_id=aid, route_category=category, provider=provider,
            required=(provider != "searxng" or version == LEGACY_SEARCH_POLICY), permitted=True,
            query_ids=[qid], outcome="candidate_found" if candidate and i == 0 else "no_match",
            started_at=now, completed_at=now))
    candidates = []
    if candidate:
        candidates = [ReferenceDiscoveryCandidate(candidate_id="c-1", attempt_id="a-0", provider="crossref", **candidate)]
    return ReferenceDiscoveryTrace(reference_id="frozen-control", search_policy_version=version,
        expected=ExpectedBibliographicFields(title="Frozen bibliographic control"),
        required_route_categories=["academic_adapter", "bounded_web"], queries=queries,
        attempts=attempts, candidates=candidates)


COMPLETE = [("crossref", "no_results"), ("brave", "no_results"), ("exa", "no_results")]


def test_completed_routes_cannot_exhaust_cross_script_candidate():
    from app.services.reference_discovery import build_reference_discovery_candidate
    candidate = build_reference_discovery_candidate(
        attempt_id='a-0', provider='google_books',
        expected=ExpectedBibliographicFields(title='Economic history', authors=['Alex Morgan'], year='2020'),
        result=RetrievalResult(source_name='google_books', success=True,
            title='经济史', authors=['Alex Morgan'], year='2020'),
        acquisition_outcome='metadata_only')
    trace = trace_for(COMPLETE)
    trace.candidates = [candidate]
    trace.attempts[0].provider = 'google_books'
    trace.attempts[0].outcome = 'candidate_found'
    trace.queries[0].provider = 'google_books'
    trace.queries[0].execution_provider = 'google_books'
    trace.queries[0].execution_outcome = 'results'
    result = assess_reference_discovery_trace(trace)
    assert result.record.outcome == 'search_incomplete'
    assert not result.record.contributes_to_neutral_pattern


def test_completed_negative_ignores_optional_failures_but_preserves_legacy():
    outcomes = [*COMPLETE, ("searxng", "cooldown_skipped")]
    new = assess_reference_discovery_trace(trace_for(outcomes)).record
    old = assess_reference_discovery_trace(trace_for(outcomes, version=LEGACY_SEARCH_POLICY)).record
    assert new.outcome == "unlocated_after_search"
    assert new.required_web_providers == ["brave", "exa"]
    assert old.outcome == "search_incomplete"


@pytest.mark.parametrize("failure", ["timeout", "budget_skipped", "rate_limited", "access_restricted", "unknown"])
def test_required_failure_controls_never_become_completed_negative(failure):
    result = assess_reference_discovery_trace(trace_for([COMPLETE[0], ("brave", failure), COMPLETE[2]]))
    assert not result.ready or result.record.outcome == "search_incomplete"


def test_unattempted_exa_cannot_be_satisfied_by_searxng():
    result = assess_reference_discovery_trace(trace_for([*COMPLETE[:2], ("searxng", "no_results")]))
    assert result.record.outcome == "search_incomplete"


@pytest.mark.parametrize('provider', ['crossref', 'brave', 'exa'])
@pytest.mark.parametrize('failure', ['timeout', 'rate_limited', 'budget_skipped', 'response_invalid'])
def test_frozen_procedural_control_required_failures(provider, failure):
    outcomes = [(name, failure if name == provider else outcome) for name, outcome in COMPLETE]
    result = assess_reference_discovery_trace(trace_for(outcomes))
    assert not result.ready or result.record.outcome == 'search_incomplete'


@pytest.mark.parametrize('disposition', [
    'not_attempted', 'identity_unconfirmed', 'transport_failure',
    'access_restricted', 'completeness_rejected', 'metadata_only',
])
def test_frozen_procedural_control_unresolved_candidates(disposition):
    result = assess_reference_discovery_trace(trace_for(
        [('crossref', 'results'), *COMPLETE[1:]],
        candidate={'acquisition_outcome': disposition}))
    assert result.record.outcome == 'search_incomplete'
    assert not result.record.contributes_to_neutral_pattern


def test_frozen_procedural_control_exhausted_candidate_is_not_zero_links():
    result = assess_reference_discovery_trace(trace_for(
        [('crossref', 'results'), *COMPLETE[1:]],
        candidate={'acquisition_outcome': 'identity_rejected'}))
    assert result.record.outcome == 'unlocated_after_search'


@pytest.mark.parametrize("outcome", ["transport_failure", "identity_unconfirmed", "not_attempted",
                                    "type_rejected", "completeness_rejected", "metadata_only"])
def test_unresolved_candidate_identity_is_not_a_completed_negative(outcome):
    result = assess_reference_discovery_trace(trace_for(COMPLETE, candidate={"acquisition_outcome": outcome}))
    assert result.record.outcome == "search_incomplete"


def test_known_identity_does_not_require_full_text_or_exhaustive_search():
    result = assess_reference_discovery_trace(trace_for([("crossref", "results")], candidate={
        "acquisition_outcome": "completeness_rejected", "validated_identity_content_sha256": "a" * 64}))
    assert result.record.outcome == "possible_match"
    assert not result.record.contributes_to_neutral_pattern


def test_new_completed_negative_stays_suppressed_at_live_attachment_boundary():
    trace = trace_for(COMPLETE)
    token = _ACTIVE_DISCOVERY_TRACE.set({"reference_id": trace.reference_id, "expected": trace.expected,
        "search_policy_version": trace.search_policy_version, "required": set(trace.required_route_categories),
        "required_web_providers": ["brave", "exa"], "queries": trace.queries,
        "attempts": trace.attempts, "candidates": []})
    try:
        saved, record = SourceResolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    assert record is None and saved["outcome_derived"] is False
    assert saved["search_policy_version"] == API_FIRST_SEARCH_POLICY


@pytest.mark.parametrize("module", ["brave", "exa"])
@pytest.mark.parametrize("payload", [{}, {"results": [None]}, {"web": {"results": [None]}}])
def test_malformed_provider_responses_are_not_empty_success(monkeypatch, module, payload):
    from app.services.search.brave import BraveSearch
    from app.services.search.exa import ExaSearch
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(f"app.services.search.{module}.httpx." + ("get" if module == "brave" else "post"),
                        lambda *a, **kw: response)
    provider = BraveSearch("test") if module == "brave" else ExaSearch("test")
    assert provider.search("frozen control") == []
    assert provider.last_status != "completed"
