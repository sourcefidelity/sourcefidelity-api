"""Reuse of a completed paid web search that found no source (`search-reuse-memo-v1`).

Owner decision 2026-09-29. A spend audit found most web-search spend went to
references whose source was never found, largely the same search repeated in
later runs. When a reference's required web search *completed* (every
required provider answered; the discovery was neither `search_incomplete` nor
`full_text_search_incomplete`) and no acceptable source was acquired, a memo
is recorded for the reference's normalized identity within one authorization
scope. For `SEARCH_REUSE_PAUSE_DAYS` after that search, a later run in the same
scope skips the paid web tier and records the earlier search's dated outcome
on a `search_memo` bounded-web attempt, so the completion rules and the
Cannot-be-verified rule read it exactly as they read the original search.

Boundaries:

* An incomplete search never creates a memo; it stays retryable.
* A memo is never read across scopes.
* A memo is only written when the original web tier contributed no identity
  signal (no credible or title-agreeing web candidate, no established
  transient identity), because the reused attempt carries no candidates:
  replaying it can therefore never remove a possible match or a conflict.
* `force_search`, a source upload, or a later acquisition bypasses or clears it.
* It stores a reference-key hash and operational counts/outcomes only: no
  URLs and no search-result content. Query text is the application's own
  query built from the reference, which discovery traces already hold.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import uuid

from app.config import settings

logger = logging.getLogger(__name__)

POLICY = "search-reuse-memo-v1"
MEMO_PROVIDER = "search_memo"
REUSED_REASON = "search_memo_reused"
# Outcomes of a search that completed. `search_incomplete` and
# `insufficient_metadata` never qualify.
ELIGIBLE_OUTCOMES = frozenset({
    "confirmed", "confirmed_with_minor_differences", "possible_match",
    "bibliographic_conflict", "unlocated_after_search",
})
_COMPLETED_ROUTE_OUTCOMES = frozenset({"candidate_found", "no_match", "candidates_processed"})
_COMPLETED_QUERY_OUTCOMES = frozenset({"results", "no_results"})
# Per-call accounting describes the original request, not this reuse.
_QUERY_FIELDS = (
    "execution_provider", "execution_engine_group", "execution_outcome", "result_count",
    "required", "reason_code", "normalized_query", "query_sha256",
)


@dataclass(frozen=True)
class SearchMemoContext:
    scope_type: str
    scope_id: str
    store: "SearchMemoStore | None" = None
    force_search: bool = False


_CONTEXT: ContextVar[SearchMemoContext | None] = ContextVar("sourcefidelity_search_memo_context", default=None)


@contextmanager
def search_memo_scope(context: SearchMemoContext | None):
    """Make one authorization scope's memos available to reference resolution."""
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def active_context() -> SearchMemoContext | None:
    context = _CONTEXT.get()
    if context is None or context.store is None:
        return None
    if not str(context.scope_type or "").strip() or not str(context.scope_id or "").strip():
        return None
    return context


def reference_key(doi: str | None, title: str | None, author: str | None, year: str | None) -> str | None:
    """Hash of the reference's normalized identity: DOI, else title + first surname + year."""
    from app.services.reference_discovery import _normalize_doi, _normalize_text, _year
    from app.services.relevance import extract_surnames

    normalized_doi = _normalize_doi(doi or "")
    if normalized_doi:
        identity = f"doi:{normalized_doi}"
    else:
        normalized_title = _normalize_text(title or "")
        if not normalized_title:
            return None
        surnames = extract_surnames(author or "")
        identity = "|".join((
            "title", normalized_title, _normalize_text(surnames[0]) if surnames else "",
            _year(year or ""),
        ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class SearchMemoStore:
    """Memo rows in the job's own database session.

    Writes are flushed, not committed: they commit with the reference's
    retrieval checkpoint, so an interrupted run leaves no memo behind. Each
    operation runs inside a savepoint, so a failed memo statement (for example
    before the table's migration has run) cannot abort the job's transaction.
    """

    def __init__(self, session) -> None:
        self._session = session

    def _guarded(self, operation):
        nested = self._session.begin_nested()
        try:
            value = operation()
        except Exception:
            nested.rollback()
            raise
        nested.commit()
        return value

    def _select(self, scope_type: str, scope_id: str, key: str):
        from sqlalchemy import select
        from app.models.search_memo import SearchMemoRecord

        record = self._session.scalar(select(SearchMemoRecord).where(
            SearchMemoRecord.scope_type == scope_type,
            SearchMemoRecord.scope_id == scope_id,
            SearchMemoRecord.reference_key_sha256 == key,
        ))
        return record if isinstance(record, SearchMemoRecord) else None

    def get(self, scope_type: str, scope_id: str, key: str):
        return self._guarded(lambda: self._select(scope_type, scope_id, key))

    def put(self, scope_type: str, scope_id: str, key: str, summary: dict, *,
            created_at: datetime, expires_at: datetime):
        from app.models.search_memo import SearchMemoRecord

        def write():
            record = self._select(scope_type, scope_id, key)
            if record is None:
                record = SearchMemoRecord(id=uuid.uuid4(), scope_type=scope_type, scope_id=scope_id,
                                          reference_key_sha256=key)
                self._session.add(record)
            record.memo_policy_version = POLICY
            record.search_policy_version = str(summary.get("search_policy_version") or "")
            record.outcome_summary = summary
            record.created_at = created_at
            record.expires_at = expires_at
            self._session.flush()
            return record

        return self._guarded(write)

    def delete(self, scope_type: str, scope_id: str, key: str) -> None:
        from sqlalchemy import delete
        from app.models.search_memo import SearchMemoRecord

        def remove():
            self._session.execute(delete(SearchMemoRecord).where(
                SearchMemoRecord.scope_type == scope_type,
                SearchMemoRecord.scope_id == scope_id,
                SearchMemoRecord.reference_key_sha256 == key,
            ))
            self._session.flush()

        self._guarded(remove)


def completed_search_summary(trace_payload: dict | None, outcome: str | None) -> dict | None:
    """The memo summary for a completed web search, or None when it must stay retryable."""
    from app.services.reference_discovery import ReferenceDiscoveryTrace, full_text_search_incompleteness

    if outcome not in ELIGIBLE_OUTCOMES or not isinstance(trace_payload, dict):
        return None
    if not trace_payload.get("candidates_complete"):
        return None
    if full_text_search_incompleteness(trace_payload) is not None:
        return None
    try:
        trace = ReferenceDiscoveryTrace.model_validate(trace_payload)
    except ValueError:
        return None
    if "bounded_web" not in trace.required_route_categories:
        return None
    web = [a for a in trace.attempts if a.route_category == "bounded_web" and a.required]
    if not web or any(a.provider == MEMO_PROVIDER or a.search_memo is not None for a in web):
        return None
    if any(not a.permitted or a.outcome not in _COMPLETED_ROUTE_OUTCOMES or a.completed_at is None for a in web):
        return None
    if any(audit.identity_established for a in web for audit in a.transient_search_audits):
        return None
    web_ids = {a.attempt_id for a in web}
    for candidate in trace.candidates:
        if candidate.attempt_id not in web_ids:
            continue
        if candidate.is_credible or any(
                c.field_name == "title" and c.outcome in {"agreement", "minor_difference"}
                for c in candidate.comparisons):
            return None
    queries = {q.query_id: q for q in trace.queries}
    attempts = []
    for attempt in web:
        bound = [queries[qid] for qid in attempt.query_ids if qid in queries]
        if not bound or any(q.execution_outcome not in _COMPLETED_QUERY_OUTCOMES
                            and q.execution_provider in trace.required_web_providers for q in bound):
            return None
        copied = []
        for query in bound:
            item = query.model_dump(mode="json", include=set(_QUERY_FIELDS))
            screen = query.bounded_review_screen
            if screen is not None and screen.observations is None:
                item["bounded_review_screen"] = screen.model_dump(mode="json")
            copied.append(item)
        attempts.append({
            "provider": attempt.provider,
            "outcome": attempt.outcome,
            "transient_search_audits": [a.model_dump(mode="json") for a in attempt.transient_search_audits],
            "started_at": attempt.started_at.isoformat(),
            "completed_at": attempt.completed_at.isoformat(),
            "queries": copied,
        })
    return {
        "memo_policy_version": POLICY,
        "search_policy_version": trace.search_policy_version,
        "required_web_providers": list(trace.required_web_providers),
        "source_kind": trace.expected.source_kind,
        "discovery_outcome": outcome,
        "source_found": False,
        "searched_at": min(a["started_at"] for a in attempts),
        "web_attempts": attempts,
    }


def record_completed_search(context: SearchMemoContext, key: str, summary: dict,
                            *, now: datetime | None = None):
    days = int(settings.SEARCH_REUSE_PAUSE_DAYS or 0)
    if days <= 0:
        return None
    created = now or datetime.now(timezone.utc)
    try:
        return context.store.put(context.scope_type, context.scope_id, key, summary,
                                 created_at=created, expires_at=created + timedelta(days=days))
    except Exception as exc:  # A memo is an optimization; never fail a paper run.
        logger.warning("Search memo not recorded (type=%s)", type(exc).__name__)
        return None


def clear_memo(context: SearchMemoContext, key: str) -> None:
    try:
        context.store.delete(context.scope_type, context.scope_id, key)
    except Exception as exc:
        logger.warning("Search memo not cleared (type=%s)", type(exc).__name__)


def reusable_memo(context: SearchMemoContext, key: str, *, source_kind: str,
                  now: datetime | None = None):
    """A memo this run may reuse instead of repeating the paid web search."""
    if context.force_search or int(settings.SEARCH_REUSE_PAUSE_DAYS or 0) <= 0:
        return None
    try:
        record = context.store.get(context.scope_type, context.scope_id, key)
    except Exception as exc:
        logger.warning("Search memo unavailable (type=%s)", type(exc).__name__)
        return None
    if record is None:
        return None
    summary = record.outcome_summary if isinstance(record.outcome_summary, dict) else {}
    current = now or datetime.now(timezone.utc)
    if (record.memo_policy_version != POLICY
            or record.search_policy_version != settings.SEARCH_POLICY_VERSION
            or summary.get("search_policy_version") != settings.SEARCH_POLICY_VERSION
            or summary.get("source_kind") != source_kind
            or summary.get("discovery_outcome") not in ELIGIBLE_OUTCOMES
            or not summary.get("web_attempts")
            or _aware(record.expires_at) <= current
            # A pause longer than the current setting is capped by it.
            or _aware(record.created_at) + timedelta(days=int(settings.SEARCH_REUSE_PAUSE_DAYS)) <= current):
        return None
    return record


def reuse_attempts(record, *, reference_id: str, expected_required_providers: list[str]):
    """Rebuild the memo's bounded-web attempts for the current trace.

    Returns (queries, attempts) or None when the memo cannot be replayed
    faithfully, in which case the caller searches normally.
    """
    from app.services.reference_discovery import (
        ReferenceRouteAttempt, ReferenceSearchQuery, SearchMemoReuse,
    )

    summary = record.outcome_summary
    if list(summary.get("required_web_providers") or []) != list(expected_required_providers):
        return None
    searched_at = datetime.fromisoformat(summary["searched_at"])
    reuse = SearchMemoReuse(memo_id=str(record.id), searched_at=_aware(searched_at),
                            expires_at=_aware(record.expires_at),
                            original_outcome=summary["discovery_outcome"])
    queries, attempts = [], []
    for ordinal, item in enumerate(summary["web_attempts"], start=1):
        bound = []
        for index, query in enumerate(item["queries"], start=1):
            seed = f"{reference_id}:{MEMO_PROVIDER}:{record.id}:{ordinal}:{index}:{query['query_sha256']}"
            bound.append(ReferenceSearchQuery(
                **query,
                query_id=f"query-{hashlib.sha256(seed.encode()).hexdigest()[:24]}",
                route_category="bounded_web", provider=MEMO_PROVIDER, cache_hit=True,
            ))
        audits = item.get("transient_search_audits") or []
        outcome = item["outcome"]
        if outcome == "candidate_found":
            # The original candidates are not carried (they may hold search
            # content); only candidates without identity signal ever qualify.
            outcome = "candidates_processed" if audits else "no_match"
        seed = f"{reference_id}:{MEMO_PROVIDER}:{record.id}:{ordinal}:" + ":".join(q.query_id for q in bound)
        attempts.append(ReferenceRouteAttempt(
            search_memo=reuse,
            transient_search_audits=audits,
            attempt_id=f"attempt-{hashlib.sha256(seed.encode()).hexdigest()[:24]}",
            route_category="bounded_web", provider=MEMO_PROVIDER, required=True, permitted=True,
            query_ids=[q.query_id for q in bound], outcome=outcome, reason_code=REUSED_REASON,
            started_at=_aware(datetime.fromisoformat(item["started_at"])),
            completed_at=_aware(datetime.fromisoformat(item["completed_at"])),
        ))
        queries.extend(bound)
    return queries, attempts
