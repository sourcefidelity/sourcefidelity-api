"""A completed paid web search that found no source is not repeated for 30 days.

`search-reuse-memo-v1`, owner decision 2026-09-29 (pause set to 30 days).
Provider doubles and an in-memory database only: no live search, no student
text.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.models import Base
from app.models.search_memo import SearchMemoRecord
from app.services.reference_discovery import full_text_search_incompleteness
from app.services.reference_verification import assess_reference_verification
from app.services.retrieval.base import RetrievalResult, RetrievalSource
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.schemas import ParsedReference
from app.services.search import search_memo
from app.services.search.base import SearchResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY
from app.services.search.search_memo import SearchMemoContext, SearchMemoStore, search_memo_scope
from app.services.source_resolver import SourceResolutionError, SourceResolver

TITLE = "Frozen source control"


class _EmptyIndex(RetrievalSource):
    """A required metadata index that answers every title search with nothing."""

    required_for_search_completion = True
    capabilities = frozenset({"title_author", "doi"})

    def __init__(self, name):
        self.name = name

    def search_by_doi(self, doi):
        return RetrievalResult(source_name=self.name, success=False, error="No results")

    def search_by_title_author(self, title, author=None):
        return RetrievalResult(source_name=self.name, success=False, error="No results")


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:40,exa:40,tavily:40")
    monkeypatch.setattr(settings, "SEARCH_REUSE_PAUSE_DAYS", 30)
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", None)
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None)
                 for name in ("brave", "exa", "searxng", "tavily")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name or "searxng"))
    return providers


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as db:
        yield db


def _resolver(indexes=None):
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._repository_session_factory = None
    resolver._acquisition_capabilities = None
    resolver._lookup_cache = None
    resolver._retrieval_sources = [*(indexes or [_EmptyIndex("crossref"), _EmptyIndex("openalex")]),
                                   WebSearchRetriever(health_store=Mock())]
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name="local_cache", success=False))
    resolver._enrich_book_editions = lambda: None
    resolver._enrich_journal_registration = lambda reference: None
    return resolver


def _reference(reference_id="r1"):
    return ParsedReference(reference_id=reference_id, title=TITLE, author="Writer, A.", year="2020",
                           source_kind="journal_article", raw_ref=f"Writer, A. (2020). {TITLE}.")


def _run(session, providers, *, scope=("personal_owner", "owner-a"), force=False, reference_id="r1",
         indexes=None):
    """Resolve one reference in a scope; return its trace and the paid calls it made."""
    before = sum(p.search.call_count for p in providers.values())
    context = SearchMemoContext(scope_type=scope[0], scope_id=scope[1], store=SearchMemoStore(session),
                                force_search=force)
    with search_memo_scope(context), pytest.raises(SourceResolutionError) as raised:
        _resolver(indexes).resolve_reference(_reference(reference_id))
    session.commit()
    calls = sum(p.search.call_count for p in providers.values()) - before
    return raised.value.reference_discovery_trace, calls


def _memos(session):
    return list(session.scalars(select(SearchMemoRecord)))


def test_completed_no_find_is_reused_by_a_second_paper_in_the_same_scope(cascade, session):
    trace, calls = _run(session, cascade)
    assert calls > 0
    assert full_text_search_incompleteness(trace) is None
    [memo] = _memos(session)
    assert memo.scope_type == "personal_owner" and memo.scope_id == "owner-a"
    assert memo.memo_policy_version == "search-reuse-memo-v1"
    assert memo.search_policy_version == API_FIRST_SEARCH_POLICY
    assert memo.outcome_summary["discovery_outcome"] == "unlocated_after_search"
    assert timedelta(days=29, hours=23) < memo.expires_at - memo.created_at <= timedelta(days=30)
    assert "http" not in str(memo.outcome_summary)

    # A second paper in the same scope, a week later.
    reused, calls = _run(session, cascade, reference_id="another-paper-r7")
    assert calls == 0
    memo_attempts = [a for a in reused["attempts"] if a["provider"] == "search_memo"]
    assert len(memo_attempts) == 1 and memo_attempts[0]["route_category"] == "bounded_web"
    assert memo_attempts[0]["search_memo"]["memo_id"] == str(memo.id)
    assert memo_attempts[0]["search_memo"]["original_outcome"] == "unlocated_after_search"
    assert memo_attempts[0]["reason_code"] == "search_memo_reused"
    # The free academic adapters still ran.
    assert {a["provider"] for a in reused["attempts"] if a["route_category"] == "academic_adapter"} \
        == {"crossref", "openalex"}
    # The reused search reads exactly as the completed original.
    assert full_text_search_incompleteness(reused) is None
    assert reused["candidates_complete"] is True
    reused_queries = {q["query_id"]: q for q in reused["queries"]}
    assert {reused_queries[q]["execution_provider"] for q in memo_attempts[0]["query_ids"]} >= {"brave", "exa"}
    assert all(reused_queries[q]["cache_hit"] for q in memo_attempts[0]["query_ids"])
    reference = _reference()
    original = assess_reference_verification(reference, {**trace, "outcome": "unlocated_after_search"})
    replayed = assess_reference_verification(reference, {**reused, "outcome": "unlocated_after_search"})
    assert original["status"] == replayed["status"] == "cannot_be_verified"
    assert original["findings"][0]["searched_routes"] == replayed["findings"][0]["searched_routes"]
    # Reuse never extends the memo.
    [same] = _memos(session)
    assert same.created_at == memo.created_at


def test_a_different_scope_searches(cascade, session):
    _run(session, cascade, scope=("personal_owner", "owner-a"))
    _, calls = _run(session, cascade, scope=("institution", "owner-a"))
    assert calls > 0
    _, calls = _run(session, cascade, scope=("personal_owner", "owner-b"))
    assert calls > 0
    assert len(_memos(session)) == 3


def test_after_thirty_days_the_search_runs_again(cascade, session):
    _run(session, cascade)
    [memo] = _memos(session)
    memo.created_at = datetime.now(timezone.utc) - timedelta(days=31)
    memo.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    session.commit()
    trace, calls = _run(session, cascade)
    assert calls > 0
    assert not any(a["provider"] == "search_memo" for a in trace["attempts"])
    [renewed] = _memos(session)
    assert search_memo._aware(renewed.expires_at) > datetime.now(timezone.utc) + timedelta(days=29)


def test_a_shorter_setting_caps_an_existing_memo(cascade, session, monkeypatch):
    _run(session, cascade)
    [memo] = _memos(session)
    memo.created_at = datetime.now(timezone.utc) - timedelta(days=8)
    session.commit()
    monkeypatch.setattr(settings, "SEARCH_REUSE_PAUSE_DAYS", 7)
    _, calls = _run(session, cascade)
    assert calls > 0


def test_an_incomplete_search_creates_no_memo(cascade, session):
    cascade["exa"].last_status = "timeout"
    trace, calls = _run(session, cascade)
    assert calls > 0
    assert full_text_search_incompleteness(trace) is not None
    assert _memos(session) == []


def test_a_failed_required_index_creates_no_memo(cascade, session):
    class Failing(_EmptyIndex):
        def search_by_title_author(self, title, author=None):
            return RetrievalResult(source_name=self.name, success=False, error="http_500")

    _run(session, cascade, indexes=[_EmptyIndex("crossref"), Failing("openalex")])
    assert _memos(session) == []


def test_force_search_searches_and_renews_the_memo(cascade, session):
    _run(session, cascade)
    [memo] = _memos(session)
    trace, calls = _run(session, cascade, force=True)
    assert calls > 0
    assert not any(a["provider"] == "search_memo" for a in trace["attempts"])
    [renewed] = _memos(session)
    assert renewed.id == memo.id and renewed.created_at >= memo.created_at


def test_a_web_candidate_with_identity_signal_is_never_memoized(cascade, session, monkeypatch):
    """The reused attempt carries no candidates, so it may only replace a search that had none that mattered."""
    cascade["exa"].search.return_value = [SearchResult("https://exa.example/lead.pdf", TITLE, "")]

    def acquisition(self, source, result, *args, **kwargs):
        result.metadata["location_attempts"] = [{
            "url": location.url, "outcome": "identity_rejected", "reason_code": "identity_rejected",
            "discovery_provider": location.metadata.get("search_provider"), "rank": 1,
            "candidate_title": TITLE} for location in result.locations]
        return result

    monkeypatch.setattr(SourceResolver, "_download_and_cache", acquisition)
    trace, _ = _run(session, cascade)
    # The search completed, and a same-title web record makes it a possible
    # match. Replaying it without that record would lose the possible match.
    assert full_text_search_incompleteness(trace) is None
    verdict = assess_reference_verification(_reference(), {**trace, "outcome": "unlocated_after_search"})
    assert verdict["status"] == "possible_match"
    assert _memos(session) == []


def test_an_acquired_source_clears_the_memo(cascade, session, monkeypatch):
    _run(session, cascade)
    assert len(_memos(session)) == 1
    found = RetrievalResult(source_name="student_url", success=True, full_text=b"%PDF-found")
    context = SearchMemoContext("personal_owner", "owner-a", SearchMemoStore(session))
    with search_memo_scope(context):
        resolver = _resolver()
        resolver.resolve = lambda **_kwargs: found
        resolver.resolve_reference(_reference())
    session.commit()
    assert _memos(session) == []


def test_an_uploaded_source_clears_the_memo(cascade, session):
    _run(session, cascade)
    context = SearchMemoContext("personal_owner", "owner-a", SearchMemoStore(session))
    search_memo.clear_memo(context, search_memo.reference_key(None, TITLE, "Writer, A.", "2020"))
    session.commit()
    assert _memos(session) == []


def test_without_a_scope_nothing_is_read_or_written(cascade, session):
    with pytest.raises(SourceResolutionError):
        _resolver().resolve_reference(_reference())
    assert _memos(session) == []


def test_reference_key_normalizes_identity():
    key = search_memo.reference_key
    assert key("https://doi.org/10.1/ABC", "x", "y", "2020") == key("10.1/abc", "other", "z", "1999")
    assert key(None, "Frozen  Source, Control!", "Writer, A.", "2020") \
        == key(None, "frozen source control", "A. Writer", "(2020a)")
    assert key(None, TITLE, "Writer, A.", "2020") != key(None, TITLE, "Other, B.", "2020")
    assert key(None, "", "Writer", "2020") is None


def test_request_force_search_reaches_the_targeted_refresh(monkeypatch):
    from types import SimpleNamespace
    import app.services.paper_workflow as workflow
    from tests.unit.test_full_text_search_completeness import _trace

    monkeypatch.setattr(workflow, "prepare_dispatch", lambda *_a, **_k: None)
    trace = _trace({"brave": "results", "exa": "budget_skipped"})
    item = {"reference_id": "ref-1", "status": "abstract_only", "reason_code": "full_text_search_incomplete",
            "reference_discovery_trace": trace, "reference_discovery": {**trace, "outcome": "confirmed"},
            "full_text_search_incomplete_providers": ["exa"]}
    job = SimpleNamespace(status=workflow.JobStatus.COMPLETED, store_only=False, extraction_payload={"x": 1},
                          upload_evidence={}, verification_summary={}, stage="completed", source_results=[item])
    monkeypatch.setattr(workflow, "_job", lambda _s, _id: job)
    assert workflow.prepare_provider_recovery_refresh(Mock(), "j", provider="exa", commit=False,
                                                      force_search=True) == ["ref-1"]
    assert job.upload_evidence[workflow.TARGETED_SOURCE_REFRESH_KEY]["force_search"] is True


class _ConfirmingIndex(_EmptyIndex):
    """Confirms the work's identity but offers no route to its text."""

    def search_by_title_author(self, title, author=None):
        return RetrievalResult(source_name=self.name, success=True, title=TITLE,
                               authors=["Writer, A."], year="2020")


def test_a_confirmed_work_whose_text_was_not_found_is_not_searched_again(cascade, session):
    """The main spend case: identity settled by free indexes, paid search found no copy."""
    indexes = [_ConfirmingIndex("crossref"), _ConfirmingIndex("openalex")]
    trace, calls = _run(session, cascade, indexes=indexes)
    assert calls > 0
    [memo] = _memos(session)
    assert memo.outcome_summary["discovery_outcome"] in {"confirmed", "confirmed_with_minor_differences"}
    reused, calls = _run(session, cascade, indexes=indexes)
    assert calls == 0
    assert any(a["provider"] == "search_memo" for a in reused["attempts"])
    assert full_text_search_incompleteness(reused) is None


@pytest.mark.parametrize("force", [False, True])
def test_paper_retrieval_resolves_within_the_jobs_own_scope(monkeypatch, force):
    from app.models.job import Job
    from app.services.paper_workflow import TARGETED_SOURCE_REFRESH_KEY, retrieve_paper_sources
    from tests.unit.test_paper_workflow import _retry_test_job

    factory, storage, job_id = _retry_test_job(monkeypatch)
    seen = []

    class ScopeRecordingResolver:
        def resolve_reference(self, reference, *, identity_only=False):
            context = search_memo.active_context()
            seen.append((context.scope_type, context.scope_id, context.force_search))
            raise SourceResolutionError("not found")

    with factory() as db:
        job = db.get(Job, job_id)
        job.source_results = None
        job.stage = "extracted"
        job.upload_evidence = {**(job.upload_evidence or {}),
                               TARGETED_SOURCE_REFRESH_KEY: {"force_search": force, "reference_ids": []}}
        db.commit()
        retrieve_paper_sources(db, storage, job_id, resolver=ScopeRecordingResolver())
        scope = (job.scope_type, job.scope_id)
    assert seen and set(seen) == {(*scope, force)}
    assert search_memo.active_context() is None


def test_a_missing_memo_table_never_breaks_the_run(cascade):
    """Before the migration runs, memo statements fail inside a savepoint only."""
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    tables = [t for t in Base.metadata.sorted_tables if t.name != "search_memos"]
    Base.metadata.create_all(engine, tables=tables)
    with sessionmaker(bind=engine, expire_on_commit=False)() as db:
        trace, calls = _run(db, cascade)
        assert calls > 0 and trace["candidates_complete"] is True
        _, calls = _run(db, cascade)
        assert calls > 0
