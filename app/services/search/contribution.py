"""Read-only provider contribution accounting from retained discovery traces.

Record/location counts are not counts of unique works. Missing transient result
sets prevent a complete retrospective novelty comparison; never reconstruct them.
"""
from collections import Counter

from app.services.reference_discovery import (
    ReferenceDiscoveryTrace, credible_field_candidates, derive_reference_discovery_record,
)


def provider_contribution(traces, provider="exa"):
    counts = Counter()
    outcomes = Counter()
    reasons = Counter()
    references, confirmed, acquired, novel_confirmed, novel_acquired = (set() for _ in range(5))
    locations = set()
    prior_unavailable = set()
    latency = cost = 0.0
    seen_traces = set()
    for raw in traces:
        trace = ReferenceDiscoveryTrace.model_validate(raw)
        # Repeated input must not multiply calls, candidates, or cost.
        key = trace.model_dump_json()
        if key in seen_traces:
            continue
        seen_traces.add(key)
        queries = [q for q in trace.queries if q.execution_provider == provider]
        for q in queries:
            counts["queries_without_call_count"] += int(q.provider_calls is None)
            counts["queries_without_latency"] += int(q.latency_seconds is None)
            calls = q.provider_calls or 0
            counts["provider_calls"] += calls
            latency += q.latency_seconds or 0
            if calls and q.cost_usd is None:
                counts["calls_without_observed_cost"] += calls
            elif calls:
                cost += q.cost_usd
        record = derive_reference_discovery_record(
            reference_id=trace.reference_id, expected=trace.expected,
            required_route_categories=trace.required_route_categories,
            queries=trace.queries, attempts=trace.attempts, candidates=trace.candidates,
            search_policy_version=trace.search_policy_version,
            search_retention_policy=trace.search_retention_policy)
        qualified = {c.candidate_id for c in credible_field_candidates(record, fields=("title", "author"))
                     if c.agreement_count >= 2 and not any(
                         v.reason_code == "book_edition_year_unresolved" for v in c.comparisons)}
        prior_locations = set()
        prior_identity = prior_acquisition = False
        provider_seen = False
        for c in trace.candidates:
            attributed = (c.discovery_provider or c.provider) == provider
            if not attributed:
                if not provider_seen:
                    if c.location_sha256:
                        prior_locations.add(c.location_sha256)
                    prior_identity |= c.candidate_id in qualified
                    prior_acquisition |= (c.candidate_id in qualified and
                                          c.acquisition_outcome in {"acquired", "acquired_fallback"})
                continue
            provider_seen = True
            references.add(trace.reference_id)
            counts["candidate_records"] += 1
            outcomes[c.acquisition_outcome] += 1
            counts["plausible_match_records"] += int(c.plausible_identity_match)
            if c.acquisition_outcome == "not_attempted":
                reasons[c.disposition_reason_code or "unspecified"] += 1
            if c.location_sha256:
                locations.add((trace.reference_id, c.location_sha256))
                counts["records_overlapping_retained_earlier_locations"] += int(c.location_sha256 in prior_locations)
                counts["records_not_seen_in_retained_earlier_locations"] += int(c.location_sha256 not in prior_locations)
            else:
                counts["records_without_location_binding"] += 1
            if c.candidate_id in qualified:
                confirmed.add(trace.reference_id)
                if not prior_identity:
                    novel_confirmed.add(trace.reference_id)
                if c.acquisition_outcome in {"acquired", "acquired_fallback"}:
                    acquired.add(trace.reference_id)
                    if not prior_acquisition:
                        novel_acquired.add(trace.reference_id)
        if queries and trace.search_retention_policy and any(
                q.execution_provider == "brave" and q.execution_outcome == "results" for q in trace.queries):
            prior_unavailable.add(trace.reference_id)
    return {
        "version": "retained-provider-contribution-v1", "provider": provider,
        **{key: counts[key] for key in (
            "provider_calls", "candidate_records", "calls_without_observed_cost", "plausible_match_records",
            "queries_without_call_count", "queries_without_latency",
            "records_overlapping_retained_earlier_locations", "records_not_seen_in_retained_earlier_locations",
            "records_without_location_binding")},
        "references_with_candidates": len(references),
        "distinct_reference_location_bindings": len(locations),
        "confirmed_identity_references": len(confirmed),
        "additional_confirmed_identity_references": len(novel_confirmed),
        "usable_source_references": len(acquired),
        "additional_usable_source_references": len(novel_acquired),
        "references_with_unavailable_prior_brave_results": len(prior_unavailable),
        "api_latency_seconds": latency, "observed_cost_usd": cost,
        "acquisition_outcomes": dict(outcomes), "unattempted_reasons": dict(reasons),
        "limitations": ["Counts describe retained candidate records/location bindings, not unique works.",
            "Retained-order comparison cannot reconstruct discarded Brave results or establish full novelty.",
            "No recorded gain is not proof of provider ineffectiveness; unavailable inspection is separate.",
            "API latency excludes candidate fetching/validation; observed cost is before credits, not an invoice."],
    }
