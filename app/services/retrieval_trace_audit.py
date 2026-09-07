"""Text-free acquisition-funnel audit for persisted reference discovery traces."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def audit_retrieval_funnel(source_results: Iterable[dict]) -> dict:
    """Aggregate provider conversion without exposing queries, URLs, or references."""
    results = list(source_results)
    attempts: list[dict] = []
    queries: list[dict] = []
    candidates: list[dict] = []
    for result in results:
        trace = result.get("reference_discovery_trace") or {}
        attempts.extend(trace.get("attempts") or [])
        queries.extend(trace.get("queries") or [])
        candidates.extend(trace.get("candidates") or [])

    web_queries = [query for query in queries if query.get("provider") == "web_search"]
    web_candidates = [
        candidate for candidate in candidates if candidate.get("provider") == "web_search"
    ]
    web_providers = sorted(
        {
            str(value).lower()
            for value in (
                *[query.get("execution_provider") for query in web_queries],
                *[candidate.get("discovery_provider") for candidate in web_candidates],
            )
            if value
        }
    )
    web_conversion = {}
    for provider in web_providers:
        provider_queries = [
            query
            for query in web_queries
            if str(query.get("execution_provider") or "").lower() == provider
        ]
        provider_candidates = [
            candidate
            for candidate in web_candidates
            if str(candidate.get("discovery_provider") or "").lower() == provider
        ]
        acquisition = Counter(
            str(candidate.get("acquisition_outcome") or "unknown")
            for candidate in provider_candidates
        )
        acquired = acquisition.get("acquired", 0) + acquisition.get(
            "acquired_fallback", 0
        )
        web_conversion[provider] = {
            "queries": len(provider_queries),
            "query_outcomes": dict(
                sorted(
                    Counter(
                        str(query.get("execution_outcome") or "unknown")
                        for query in provider_queries
                    ).items()
                )
            ),
            "candidates": len(provider_candidates),
            "acquisition_outcomes": dict(sorted(acquisition.items())),
            "acquired_documents": acquired,
            "candidate_conversion_rate": (
                round(acquired / len(provider_candidates), 4)
                if provider_candidates
                else None
            ),
        }

    provider_candidates: dict[str, list[dict]] = {}
    for candidate in candidates:
        provider_candidates.setdefault(
            str(candidate.get("provider") or "unknown"), []
        ).append(candidate)
    structured_location_gap = {}
    for provider, items in sorted(provider_candidates.items()):
        if provider == "web_search":
            continue
        located = [item for item in items if item.get("location_available")]
        outcomes = Counter(
            str(item.get("acquisition_outcome") or "unknown") for item in items
        )
        structured_location_gap[provider] = {
            "candidates": len(items),
            "location_available": len(located),
            "acquisition_outcomes": dict(sorted(outcomes.items())),
            "location_conversion_trace_missing": bool(
                located and set(outcomes) <= {"metadata_only"}
            ),
        }

    return {
        "source_results": len(results),
        "source_outcomes": dict(
            sorted(Counter(str(item.get("status") or "unknown") for item in results).items())
        ),
        "route_attempts": len(attempts),
        "route_attempts_by_provider": _nested_counts(
            attempts, "provider", "outcome"
        ),
        "queries": len(queries),
        "candidates": len(candidates),
        "candidate_acquisition_outcomes": dict(
            sorted(
                Counter(
                    str(candidate.get("acquisition_outcome") or "unknown")
                    for candidate in candidates
                ).items()
            )
        ),
        "web_provider_conversion": web_conversion,
        "structured_provider_locations": structured_location_gap,
        "limitations": [
            "Counts measure candidate conversion, not source relevance or retrieval quality.",
            "A provider can contribute a location that is deduplicated with another provider's location.",
            "Older traces recorded structured-provider locations before shared acquisition and therefore cannot attribute their later failure or success; location_conversion_trace_missing marks that gap.",
        ],
    }


def _nested_counts(items: list[dict], key: str, outcome: str) -> dict:
    grouped: dict[str, Counter] = {}
    for item in items:
        grouped.setdefault(str(item.get(key) or "unknown"), Counter())[
            str(item.get(outcome) or "unknown")
        ] += 1
    return {
        name: dict(sorted(counts.items())) for name, counts in sorted(grouped.items())
    }
