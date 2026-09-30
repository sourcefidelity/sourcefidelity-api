"""Optional operation-local allowance; never changes deployment defaults."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
from threading import Lock


class CandidateBudgetExceeded(RuntimeError):
    def __init__(self):
        super().__init__("candidate_budget_exhausted")


class CandidateBudget:
    def __init__(self, limit: int):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Candidate limit must be a nonnegative integer")
        self.limit = limit
        self._allowed = set()
        self._denied = set()
        self._lock = Lock()

    def reserve(self, kind: str, key: str) -> bool:
        # Keys are operation-only, including discovery-derived locations.
        identity = (kind, hashlib.sha256(key.encode()).digest())
        with self._lock:
            if identity in self._allowed:
                return True
            if identity in self._denied or len(self._allowed) >= self.limit:
                self._denied.add(identity)
                return False
            self._allowed.add(identity)
            return True

    def snapshot(self):
        with self._lock:
            return {"limit": self.limit, "attempted": len(self._allowed),
                    "not_attempted": len(self._denied)}

    def remaining(self):
        with self._lock:
            return max(0, self.limit - len(self._allowed))

    def clear(self):
        with self._lock:
            self._allowed.clear()
            self._denied.clear()


ACTIVE_CANDIDATE_BUDGET = ContextVar("source_candidate_budget", default=None)


def inspection_capacity(maximum=5):
    """Remaining candidate allowance, not permission to increase a tier's cap."""
    budget = ACTIVE_CANDIDATE_BUDGET.get()
    return maximum if budget is None else min(maximum, budget.remaining())


@contextmanager
def candidate_budget_scope(limit: int):
    existing = ACTIVE_CANDIDATE_BUDGET.get()
    if existing is not None:
        if limit != existing.limit:
            raise ValueError("Nested candidate scope cannot change the allowance")
        yield existing
        return
    budget = CandidateBudget(limit)
    token = ACTIVE_CANDIDATE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        budget.clear()
        ACTIVE_CANDIDATE_BUDGET.reset(token)


def require_source_candidate(key: str):
    budget = ACTIVE_CANDIDATE_BUDGET.get()
    if budget is not None and not budget.reserve("source", key):
        raise CandidateBudgetExceeded()


def reserve_metadata_candidate(provider, result):
    budget = ACTIVE_CANDIDATE_BUDGET.get()
    if budget is None or not result.success:
        return result
    import json
    key = json.dumps([provider, result.doi, result.title, result.authors, result.year,
                      (result.metadata or {}).get("isbn"),
                      (result.metadata or {}).get("book_edition_metadata")], sort_keys=True)
    if not budget.reserve("metadata", key):
        # Adapter caches may own this instance; never poison a later operation.
        return replace(result, success=False,
            metadata={**(result.metadata or {}), "candidate_budget_skipped": True})
    return result
