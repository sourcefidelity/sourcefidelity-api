"""Build a valid `search_incomplete` discovery record for recovery tests.

The provider-recovery trigger now recomputes blocking providers from the
stored discovery record instead of trusting a stored provider list, so a test
of that path needs a record that is genuinely blocked by the provider under
test. Synthetic throughout: no student reference text.
"""
import hashlib
from datetime import datetime, timezone

_NOW = datetime(2026, 9, 24, tzinfo=timezone.utc).isoformat()


def _query(query_id, category, provider, execution_provider, outcome):
    text = f"synthetic {query_id}"
    return {
        "query_id": query_id, "route_category": category, "provider": provider,
        "normalized_query": text,
        "query_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "execution_provider": execution_provider, "execution_outcome": outcome,
    }


def blocked_record(reference_id, *, adapter=None, web_outcomes=None,
                   policy="configured-search-v1"):
    """A record held incomplete by `adapter` failing, or by web query outcomes.

    ``adapter``: a required academic adapter whose attempt failed operationally.
    ``web_outcomes``: {execution_provider: execution_outcome} for one required
    bounded-web attempt.
    """
    attempts, queries, categories = [], [], []
    if adapter:
        categories.append("academic_adapter")
        queries.append(_query("q-adapter", "academic_adapter", adapter, adapter,
                              "operational_failure"))
        attempts.append({
            "attempt_id": "a-adapter", "route_category": "academic_adapter",
            "provider": adapter, "required": True, "permitted": True,
            "outcome": "operational_failure", "reason_code": "route_operational_failure",
            "query_ids": ["q-adapter"], "started_at": _NOW, "completed_at": _NOW,
        })
    if web_outcomes:
        categories.append("bounded_web")
        ids = []
        for n, (provider, outcome) in enumerate(web_outcomes.items()):
            qid = f"q-web-{n}"
            ids.append(qid)
            queries.append(_query(qid, "bounded_web", "web_search", provider, outcome))
        attempts.append({
            "attempt_id": "a-web", "route_category": "bounded_web",
            "provider": "web_search", "required": True, "permitted": True,
            "outcome": "no_match", "reason_code": "no_candidate_returned",
            "query_ids": ids, "started_at": _NOW, "completed_at": _NOW,
        })
    return {
        "reference_id": reference_id, "created_at": _NOW, "outcome": "search_incomplete",
        "contributes_to_neutral_pattern": False,
        "expected": {"title": "A synthetic work title for recovery tests",
                     "authors": ["Writer, A."], "year": "2020",
                     "source_kind": "journal_article"},
        "search_policy_version": policy,
        "required_route_categories": categories,
        "attempts": attempts, "queries": queries,
    }
