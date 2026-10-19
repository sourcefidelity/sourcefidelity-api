"""One answer to "did the required searches finish?" (search-completion-v1, owner decisions 2026-10-19).

Each case was a situation where the former readings disagreed (architecture review 2026-10-18).
"""
import pytest

from app.services.search_completion import assess_search_completion

TITLE = "Harbour cranes and port labour"
NOW = "2026-10-19T10:00:00+00:00"


def _q(qid, provider, outcome="no_results", text=None, category="bounded_web", reason=None):
    return {"query_id": qid, "route_category": category, "provider": provider, "execution_provider": provider,
            "execution_outcome": outcome, "normalized_query": text if text is not None else f'"{TITLE}" Writer 2020',
            "reason_code": reason}


def _a(aid, provider, category, qids, outcome="no_match", finished=True, permitted=True, audits=(), required=True,
       reason=None):
    return {"attempt_id": aid, "provider": provider, "route_category": category, "query_ids": list(qids),
            "outcome": outcome, "permitted": permitted, "completed_at": NOW if finished else None,
            "required": required, "transient_search_audits": list(audits), "reason_code": reason}


def _record(*, web=(("brave", "no_results"), ("exa", "no_results")), adapters=("crossref", "openalex"),
            adapter_outcome="no_match", web_text=None, providers=("brave", "exa"), extra_attempts=(),
            extra_queries=(), web_finished=True, audits=()):
    queries, attempts = [], []
    for name in adapters:
        queries.append(_q(f"q-{name}", name, category="academic_adapter", text=f"title:{TITLE} author:Writer"))
        attempts.append(_a(f"a-{name}", name, "academic_adapter", [f"q-{name}"], outcome=adapter_outcome))
    web_qids = []
    for engine, outcome in web:
        queries.append(_q(f"q-{engine}", engine, outcome=outcome, text=web_text))
        web_qids.append(f"q-{engine}")
    attempts.append(_a("a-web", "web_search", "bounded_web", web_qids, finished=web_finished, audits=audits))
    return {"expected": {"title": TITLE, "source_kind": "journal_article"}, "required_web_providers": list(providers),
            "queries": queries + list(extra_queries), "attempts": attempts + list(extra_attempts)}


def test_every_required_search_finished():
    result = assess_search_completion(_record())
    assert result.complete and result.blocking_providers == ()
    assert result.web_routes == ("brave", "exa")


def test_a_provider_side_failure_of_one_web_provider_leaves_the_other_sufficient():
    result = assess_search_completion(_record(web=(("brave", "no_results"), ("exa", "timeout"))))
    assert result.complete


@pytest.mark.parametrize("skip,blocking", [("cooldown_skipped", ("exa",)), ("budget_skipped", ())])
def test_a_web_provider_we_skipped_leaves_it_incomplete(skip, blocking):
    # Cooldown counts like a budget skip (owner decision 2026-10-19); only a
    # cooldown names the provider for a retry, since our budget is not its fault.
    result = assess_search_completion(_record(web=(("brave", "no_results"), ("exa", skip))))
    assert not result.complete and result.reason_code == "web_search_skipped_by_us"
    assert result.blocking_providers == blocking


def test_a_web_search_that_never_used_the_title_does_not_count():
    result = assess_search_completion(_record(web_text='"10.1234/x"'))
    assert not result.complete and result.reason_code == "title_web_search_incomplete"


def test_a_started_but_unfinished_attempt_does_not_count():
    result = assess_search_completion(_record(web_finished=False))
    assert not result.complete and result.reason_code == "title_web_search_incomplete"


def test_a_failed_required_index_is_incomplete_and_named():
    record = _record(adapters=("crossref",))
    record["queries"].append(_q("q-openalex", "openalex", outcome="timeout", category="academic_adapter",
                                text=f"title:{TITLE}"))
    record["attempts"].append(_a("a-openalex", "openalex", "academic_adapter", ["q-openalex"],
                                 outcome="operational_failure"))
    result = assess_search_completion(record)
    assert not result.complete and result.reason_code == "required_route_incomplete"
    assert result.blocking_providers == ("openalex",)


def test_an_unresolved_web_lead_blocks_only_the_completed_negative():
    # Not a rule for Cannot be verified (owner decision 2026-10-20); the
    # "unlocated after search" label keeps it as a stricter add-on.
    record = _record(audits=({"unresolved": 1},))
    assert assess_search_completion(record).complete
    strict = assess_search_completion(record, completed_negative=True)
    assert not strict.complete and strict.reason_code == "web_search_lead_unresolved"


@pytest.mark.parametrize("outcomes,complete", [
    (("operational_failure",), False),
    (("operational_failure", "no_match"), True),     # another attempt on the link answered
    (("unavailable",), True),                         # the link answered that the page is gone
])
def test_a_student_link_failed_on_our_side(outcomes, complete):
    extra = [_a(f"a-link-{i}", provider, "student_url", [], outcome=outcome, required=False)
             for i, (provider, outcome) in enumerate(zip(("student_url_pdf", "student_url_html"), outcomes))]
    result = assess_search_completion(_record(extra_attempts=extra))
    assert result.complete is complete
    if not complete:
        assert result.reason_code == "student_link_failed"


def test_an_archive_lookup_failure_is_not_the_students_link():
    extra = [_a("a-wayback", "wayback_snapshot", "student_url", [], outcome="operational_failure", required=False)]
    assert assess_search_completion(_record(extra_attempts=extra)).complete


def test_a_stored_record_is_judged_by_the_providers_it_names(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "SEARCH_SECOND_REQUIRED_PROVIDER", "deepseek")
    assert assess_search_completion(_record()).complete


def test_a_web_page_needs_no_index():
    result = assess_search_completion(_record(adapters=()), kind="webpage", web_rule=True)
    assert result.complete and result.required_routes == ()


def test_a_filtered_metadata_answer_blocks_only_the_completed_negative():
    record = _record()
    record["queries"][0]["reason_code"] = "metadata_candidates_filtered"
    assert assess_search_completion(record).complete
    assert not assess_search_completion(record, completed_negative=True).complete


def test_the_own_script_route_meets_the_same_standard():
    native = "港口起重机与港口劳动"
    record = _record()
    record["queries"] += [_q("n-crossref", "crossref", category="academic_adapter", text=f"title:{native}"),
                          _q("n-brave", "brave", text=f'"{native}"')]
    record["attempts"] += [_a("n-a1", "crossref", "academic_adapter", ["n-crossref"]),
                           _a("n-a2", "web_search", "bounded_web", ["n-brave"], finished=False)]
    route = assess_search_completion(record, title=native, attempt_ids={"n-a1", "n-a2"}, any_adapter=True)
    assert not route.complete            # its web attempt never finished
    record["attempts"][-1]["completed_at"] = NOW
    route = assess_search_completion(record, title=native, attempt_ids={"n-a1", "n-a2"}, any_adapter=True)
    assert route.complete
    ordinary = assess_search_completion(record, exclude_attempt_ids={"n-a1", "n-a2"})
    assert ordinary.complete


def test_models_and_dicts_give_the_same_answer():
    from app.services.reference_discovery import ReferenceDiscoveryTrace  # noqa: F401  (import check)
    record = _record(web=(("brave", "no_results"), ("exa", "cooldown_skipped")))
    assert assess_search_completion(record) == assess_search_completion(record)
