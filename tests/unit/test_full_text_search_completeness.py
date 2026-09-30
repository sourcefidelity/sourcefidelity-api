"""A skipped or failed required full-text search is not "full text unavailable".

Diagnosed 2026-09-29: a reference's identity was confirmed, Brave's candidates
were all rejected, and Exa -- a required provider -- was `budget_skipped` by
the job-wide call ceiling. The reference was recorded `full_text_unavailable`,
which the report reads as a completed search. It is now
`full_text_search_incomplete`, naming the providers a targeted refresh can
wait for. Synthetic throughout: no student reference text.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.models.job import Job, JobStatus
from app.services.reference_discovery import full_text_search_incompleteness
from app.services.retrieval.base import RetrievalResult
from tests.unit.blocked_discovery import blocked_record

API_FIRST = "api-first-search-v2"


def _trace(web_outcomes, *, policy=API_FIRST, reason_codes=None):
    record = blocked_record("ref-1", policy=policy, web_outcomes=web_outcomes)
    record["required_web_providers"] = ["brave", "exa"] if policy == API_FIRST else []
    for query in record["queries"]:
        code = (reason_codes or {}).get(query["execution_provider"])
        if code:
            query["reason_code"] = code
    return record


@pytest.mark.parametrize("outcomes", [
    {"brave": "results", "exa": "budget_skipped"},        # job-wide call ceiling
    {"brave": "results", "exa": "timeout"},               # operational failure
    {"brave": "operational_failure", "exa": "results"},
    {"brave": "results"},                                 # Exa never ran
])
def test_a_skipped_or_failed_required_provider_leaves_the_search_incomplete(outcomes):
    incomplete = full_text_search_incompleteness(_trace(outcomes))
    assert incomplete is not None
    expected = {p for p in ("brave", "exa") if outcomes.get(p) not in {"results", "no_results"}}
    assert set(incomplete["providers"]) == expected


def test_both_required_providers_answering_is_a_completed_search():
    assert full_text_search_incompleteness(_trace({"brave": "results", "exa": "no_results"})) is None


def test_optional_searxng_cooldown_does_not_make_the_search_incomplete():
    trace = _trace({"brave": "results", "exa": "results", "searxng": "cooldown_skipped"})
    assert full_text_search_incompleteness(trace) is None


def test_reaching_the_candidate_inspection_bound_is_not_an_unfinished_search():
    trace = _trace({"brave": "results", "exa": "budget_skipped"},
                   reason_codes={"exa": "candidate_inspection_capacity_exhausted"})
    # Every candidate slot was already inspected; Exa had nothing left to add.
    assert full_text_search_incompleteness(trace) is None
    trace = _trace({"brave": "results", "exa": "results"})
    trace["queries"].append({**trace["queries"][1], "query_id": "q-web-extra",
                             "execution_outcome": "budget_skipped",
                             "reason_code": "candidate_inspection_capacity_exhausted"})
    trace["attempts"][0]["query_ids"].append("q-web-extra")
    assert full_text_search_incompleteness(trace) is None


def test_a_required_web_route_that_never_ran_is_incomplete_without_naming_a_provider():
    trace = _trace({"brave": "results", "exa": "results"})
    trace["attempts"] = []
    assert full_text_search_incompleteness(trace) == {"providers": []}


def test_no_required_web_route_is_not_assessed_here():
    trace = _trace({"brave": "results"})
    trace["attempts"], trace["required_route_categories"] = [], []
    assert full_text_search_incompleteness(trace) is None
    assert full_text_search_incompleteness(None) is None


def test_legacy_policy_reads_query_outcomes_under_the_required_web_attempt():
    assert full_text_search_incompleteness(
        _trace({"searxng": "cooldown_skipped"}, policy="configured-search-v1")) == {"providers": ["searxng"]}
    assert full_text_search_incompleteness(
        _trace({"searxng": "results"}, policy="configured-search-v1")) is None


def _abstract_only_resolver(trace):
    class AbstractOnlyResolver:
        def resolve_reference(self, reference):
            return RetrievalResult(source_name="openalex", success=True, title=reference.title,
                                   abstract="A synthetic abstract long enough to be retained as evidence.",
                                   metadata={"identity_confidence": "high",
                                             "reference_discovery_trace": trace})
    return AbstractOnlyResolver()


@pytest.mark.parametrize("outcomes,reason,providers", [
    ({"brave": "results", "exa": "budget_skipped"}, "full_text_search_incomplete", ["exa"]),
    ({"brave": "results", "exa": "no_results"}, "full_text_unavailable", None),
])
def test_retrieval_records_the_distinct_reason(monkeypatch, outcomes, reason, providers):
    from tests.unit.test_paper_workflow import _retry_test_job
    from app.services.paper_workflow import retrieve_paper_sources

    factory, storage, job_id = _retry_test_job(monkeypatch)
    with factory() as session:
        job = session.get(Job, job_id)
        job.source_results = None
        job.stage = "extracted"
        session.commit()
        retrieve_paper_sources(session, storage, job_id, resolver=_abstract_only_resolver(_trace(outcomes)))
        [saved] = session.get(Job, job_id).source_results
    assert saved["reason_code"] == reason
    assert saved.get("full_text_search_incomplete_providers") == providers


def _job(source_result):
    return SimpleNamespace(status=JobStatus.COMPLETED, store_only=False, extraction_payload={"x": 1},
                           upload_evidence={}, verification_summary={}, stage="completed",
                           source_results=[source_result])


def test_an_incomplete_full_text_search_is_eligible_for_the_targeted_refresh(monkeypatch):
    import app.services.paper_workflow as workflow
    monkeypatch.setattr(workflow, "prepare_dispatch", lambda *_a, **_k: None)
    trace = _trace({"brave": "results", "exa": "budget_skipped"})
    confirmed = {**trace, "outcome": "confirmed"}
    item = {"reference_id": "ref-1", "status": "abstract_only", "reason_code": "full_text_search_incomplete",
            "reference_discovery_trace": trace, "reference_discovery": confirmed,
            "full_text_search_incomplete_providers": ["exa"]}

    monkeypatch.setattr(workflow, "_job", lambda _s, _id: _job(dict(item)))
    assert workflow.prepare_provider_recovery_refresh(Mock(), "j", provider="exa", commit=False) == ["ref-1"]
    monkeypatch.setattr(workflow, "_job", lambda _s, _id: _job(dict(item)))
    assert workflow.prepare_provider_recovery_refresh(Mock(), "j", provider="brave", commit=False) == []
    # A completed search recorded `full_text_unavailable` is never re-run.
    monkeypatch.setattr(workflow, "_job", lambda _s, _id: _job({**item, "reason_code": "full_text_unavailable"}))
    assert workflow.prepare_provider_recovery_refresh(Mock(), "j", provider="exa", commit=False) == []
