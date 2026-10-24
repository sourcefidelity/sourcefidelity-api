"""The discovery recorder: one cited reference's search record (discovery-recorder-v1).

Owner decisions 2026-10-21 (architecture review candidate 3, ARCHITECTURE §4).
It replaces the shared trace dictionary the resolver used to write. Writers add
queries, attempts, candidates and limitations through methods that check each
entry when it is written:

- a broken link (an attempt naming a missing query or another route's query, a
  candidate without its found attempt or with another provider, a duplicate
  identifier) is a programming error: refused with `DiscoveryRecordError` when
  `STRICT` is on (the test suite), otherwise dropped, logged and noted as a
  limitation, so one bad entry never fails a paper;
- a size cap reached drops the extra entry and notes a limitation.

`preview()` and `finish()` build the stored trace and record without changing
the recorder; `finish()` also attaches the own-script route, the archive match,
the container identity and a DOI that registers another title. The stored
shape is unchanged (`reference-discovery-trace-v1`, `reference-discovery-v1`).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field

from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryTrace, assess_reference_discovery_trace,
)

logger = logging.getLogger(__name__)

VERSION = "discovery-recorder-v1"
# The stored trace's own caps (ReferenceDiscoveryTrace).
CAPS = {"queries": 64, "attempts": 128, "candidates": 128, "limitations": 16}
# Tests turn this on so a broken link fails where it is written.
STRICT = False

_TRACE_POLICIES = dict(
    candidate_ranking_policy_version="candidate-purpose-ranking-v1",
    search_capacity_policy_version="search-inspection-capacity-v4",
    metadata_screen_resolution_policy_version="screened-metadata-work-binding-v1",
    bounded_review_policy_version="bounded-reference-review-v9",
    metadata_identity_policy_version="corroborated-metadata-identity-v1",
    credibility_policy_version="reference-credibility-v4",
)


class DiscoveryRecordError(ValueError):
    """An entry that would break the record's links (a programming error)."""


@dataclass
class ModelAllowance:
    """The cited reference's model comparisons (web-page-identity-v1), shared
    with nested searches for the same reference."""
    used: int = 0
    answers: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reserve(self, limit: int) -> bool:
        """Take one comparison if any is left (safe across worker threads)."""
        with self._lock:
            if self.used >= limit:
                return False
            self.used += 1
            return True


@dataclass
class Finished:
    trace: dict | None
    record: dict | None
    outcome: str | None


class DiscoveryRecorder:
    def __init__(self, *, reference_id: str, expected: ExpectedBibliographicFields, required=(),
                 submitted_reference: str = "", search_policy_version: str = "configured-search-v1",
                 search_retention_policy: str | None = None, required_web_providers=(),
                 allowance: ModelAllowance | None = None):
        self.reference_id = reference_id
        self.expected = expected
        self.required: set[str] = set(required)
        self.submitted_reference = submitted_reference or ""
        self.search_policy_version = search_policy_version
        self.search_retention_policy = search_retention_policy
        self.required_web_providers = list(required_web_providers or [])
        self.allowance = allowance or ModelAllowance()
        self.queries: list = []
        self.attempts: list = []
        self.candidates: list = []
        self.limitations: list[str] = []
        self.journal_checks: list | None = None
        self.native_script_route: dict | None = None
        self.periodical_archive_match: dict | None = None
        self.container_identity: dict | None = None
        self.doi_title_mismatch: dict | None = None
        self.declared_document_url: str | None = None
        self.search_memo_reused = False
        self.mdpi_file_server_urls: set[str] = set()
        self._dropped_attempts: set[str] = set()
        if not self.required:
            self.add_limitation("No completed-search route policy exists for this source kind.")

    # ---- writing ------------------------------------------------------------

    def require(self, category: str) -> None:
        self.required.add(category)

    def add_limitation(self, text: str) -> None:
        if text and text not in self.limitations and len(self.limitations) < CAPS["limitations"]:
            self.limitations.append(text)

    def add_attempt(self, attempt, queries=()) -> str | None:
        """Add an attempt with the queries it ran; its id, or None when not kept."""
        queries = list(queries)
        known = {q.query_id: q for q in self.queries}
        if attempt.attempt_id in {a.attempt_id for a in self.attempts}:
            return self._refuse("a duplicate attempt id", attempt.attempt_id)
        new_ids = [q.query_id for q in queries]
        if len(set(new_ids)) != len(new_ids) or set(new_ids) & set(known):
            return self._refuse("a duplicate query id", attempt.attempt_id)
        known.update({q.query_id: q for q in queries})
        for query_id in attempt.query_ids:
            query = known.get(query_id)
            if query is None or query.route_category != attempt.route_category or query.provider != attempt.provider:
                return self._refuse("an attempt naming a missing or another route's query", attempt.attempt_id)
        if (len(self.attempts) >= CAPS["attempts"]
                or len(self.queries) + len(queries) > CAPS["queries"]):
            self._dropped_attempts.add(attempt.attempt_id)
            self.add_limitation("Discovery record limit reached: further search attempts were not recorded.")
            return None
        self.queries.extend(queries)
        self.attempts.append(attempt)
        return attempt.attempt_id

    def add_attempts(self, attempts, queries) -> None:
        """Attempts and queries recorded together elsewhere (a reused search memo)."""
        queries = list(queries)
        for attempt in attempts:
            own = [q for q in queries if q.query_id in set(attempt.query_ids)]
            self.add_attempt(attempt, own)

    def add_candidate(self, candidate, *, unique: bool = False) -> bool:
        """Add a candidate of a found attempt. With `unique`, an exact repeat of
        one already recorded is skipped (a provider returning a record twice)."""
        if candidate.attempt_id in self._dropped_attempts:
            return False
        attempt = next((a for a in self.attempts if a.attempt_id == candidate.attempt_id), None)
        if attempt is None or attempt.provider != candidate.provider:
            self._refuse("a candidate without its attempt or with another provider", candidate.candidate_id)
            return False
        if any(c.candidate_id == candidate.candidate_id for c in self.candidates):
            if not unique:
                self._refuse("a duplicate candidate id", candidate.candidate_id)
            return False
        if len(self.candidates) >= CAPS["candidates"]:
            self.add_limitation("Discovery record limit reached: further candidates were not recorded.")
            return False
        self.candidates.append(candidate)
        return True

    def room_for(self, *, queries: int = 0, attempts: int = 0, candidates: int = 0) -> bool:
        """Whether this many more entries fit within the caps."""
        return (len(self.queries) + queries <= CAPS["queries"]
                and len(self.attempts) + attempts <= CAPS["attempts"]
                and len(self.candidates) + candidates <= CAPS["candidates"])

    def _refuse(self, what: str, identifier: str):
        if STRICT:
            raise DiscoveryRecordError(f"Discovery record refused {what} ({identifier})")
        logger.error("Discovery record refused %s", what)
        self.add_limitation("Discovery record refused an inconsistent entry.")
        return None

    # ---- reading ------------------------------------------------------------

    def attempt(self, attempt_id: str):
        return next((a for a in self.attempts if a.attempt_id == attempt_id), None)

    def completion_input(self) -> dict:
        """The parts the completion rule reads (search_completion)."""
        return {"expected": self.expected, "queries": self.queries, "attempts": self.attempts,
                "required_web_providers": self.required_web_providers}

    # ---- finishing ----------------------------------------------------------

    def preview(self, extra_limitations=()) -> Finished:
        """The trace and record as they stand, changing nothing."""
        limitations = list(self.limitations)
        for text in extra_limitations:
            if text and text not in limitations:
                limitations.append(text)
        precursor = ReferenceDiscoveryTrace(
            **_TRACE_POLICIES,
            credibility_reference_sha256=(hashlib.sha256(self.submitted_reference.encode()).hexdigest()
                                          if self.submitted_reference else None),
            **({"journal_checks": self.journal_checks} if self.journal_checks is not None else {}),
            search_retention_policy=self.search_retention_policy,
            search_policy_version=self.search_policy_version,
            required_web_providers=self.required_web_providers,
            reference_id=self.reference_id,
            expected=self.expected,
            required_route_categories=sorted(self.required),
            queries=list(self.queries), attempts=list(self.attempts), candidates=list(self.candidates),
            # Room for the assessor note and a completion note below.
            limitations=limitations[:CAPS["limitations"] - 2] + [
                "The trace is evaluated by the fail-closed completion assessor before any live outcome is attached."],
        )
        completion = assess_reference_discovery_trace(precursor)
        outcome = completion.record.outcome if completion.ready and completion.record is not None else None
        record = None
        final_limitations = list(precursor.limitations)
        if completion.ready and completion.record is not None:
            if completion.record.outcome == "unlocated_after_search":
                final_limitations.append("A completed no-match outcome is suppressed pending its real acceptance control.")
            else:
                record = completion.record.model_dump(mode="json")
        elif completion.blocker_codes:
            final_limitations.append("Completion blockers: " + ", ".join(completion.blocker_codes))
        trace = precursor.model_copy(update={"candidates_complete": completion.ready,
                                             "outcome_derived": record is not None,
                                             "limitations": final_limitations}).model_dump(mode="json")
        return Finished(trace=trace, record=record, outcome=outcome)

    def finish(self, extra_limitations=()) -> Finished:
        """The stored trace and record with their attachments, changing nothing."""
        done = self.preview(extra_limitations)
        for payload in (done.trace, done.record):
            if payload is None:
                continue
            if self.native_script_route:
                payload["native_script_route"] = deepcopy(self.native_script_route)
            if self.periodical_archive_match:
                payload["periodical_archive_match"] = deepcopy(self.periodical_archive_match)
        if done.record is not None:
            if self.container_identity:
                done.record["container_identity"] = deepcopy(self.container_identity)
            if self.doi_title_mismatch:
                done.record["doi_title_mismatch"] = deepcopy(self.doi_title_mismatch)
        return done

    # ---- stored traces ------------------------------------------------------

    @classmethod
    def from_stored(cls, trace_payload: dict, record_payload: dict | None = None) -> "DiscoveryRecorder":
        """A recorder holding a stored trace's entries, its attachments and the
        stored record's, ready to finish again (no searches)."""
        trace = ReferenceDiscoveryTrace.model_validate(
            {k: v for k, v in trace_payload.items() if k not in _ATTACHED})
        recorder = cls(reference_id=trace.reference_id, expected=trace.expected,
                       required=trace.required_route_categories, search_policy_version=trace.search_policy_version,
                       search_retention_policy=trace.search_retention_policy,
                       required_web_providers=trace.required_web_providers)
        recorder.queries, recorder.attempts, recorder.candidates = (list(trace.queries), list(trace.attempts),
                                                                    list(trace.candidates))
        recorder.limitations = [text for text in trace.limitations if not text.startswith(_FINISH_NOTES)]
        recorder.journal_checks = list(trace.journal_checks) if trace.journal_checks else None
        for name in _ATTACHED:
            value = trace_payload.get(name) or (record_payload or {}).get(name)
            if value:
                setattr(recorder, name, deepcopy(value))
        return recorder


# Keys the finish adds to the stored payloads, and the notes it writes itself.
_ATTACHED = ("native_script_route", "periodical_archive_match", "container_identity", "doi_title_mismatch")
_FINISH_NOTES = ("The trace is evaluated by the fail-closed completion assessor",
                 "A completed no-match outcome is suppressed", "Completion blockers:")


def repaired_archive_record(trace_payload: dict | None, record_payload: dict | None = None) -> Finished | None:
    """A stored result whose magazine-archive search was recorded as found with
    no candidate, which failed the record's binding check and dropped the whole
    record (fixed 2026-10-21): the record re-derived from its stored trace with
    the corrected outcome, no searches. None when there is no such attempt or
    the trace cannot be read (owner decision 2026-10-22)."""
    if not isinstance(trace_payload, dict) or record_payload:
        return None
    from app.services.retrieval import media_history_archive as archive
    try:
        recorder = DiscoveryRecorder.from_stored(trace_payload)
    except (ValueError, TypeError):
        return None
    bound = {candidate.attempt_id for candidate in recorder.candidates}
    faulty = [attempt for attempt in recorder.attempts
              if attempt.provider == archive.PROVIDER and attempt.outcome == "candidate_found"
              and attempt.attempt_id not in bound]
    if not faulty:
        return None
    recorder.attempts = [attempt.model_copy(update={"outcome": "no_match", "reason_code": "archive_page_located"})
                         if attempt in faulty else attempt for attempt in recorder.attempts]
    return recorder.finish()
