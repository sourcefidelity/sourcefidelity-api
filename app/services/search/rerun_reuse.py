"""Reuse a paper's earlier search results when it is checked again (`paper-search-reuse-v1`).

Owner decision 2026-10-01: re-running a paper must not pay again for references
already searched. When a later run of a paper in the same scope reaches a
reference whose text and parse are unchanged, the most recent completed run's
result for it is reused, within ``SEARCH_RERUN_REUSE_DAYS``, instead of
searching. Only results that stand on their own are reused:

* a durably stored source (``durable_authorized``), re-authorized for this run;
* a completed result with no full text (``unavailable``, ``abstract_only``,
  ``metadata_only``) whose search finished, or whose every search required
  for the reference's kind finished without locating it (the standard behind
  Cannot be verified), even if an optional route failed (2026-10-02).

A transient text (``transient_authorized``) is never reused: its bytes were
discarded after that run and its address was not kept. A search that was
incomplete, a reference whose parse changed, a cited reference whose earlier
record was identity-only (or the reverse) and a targeted refresh with
``force_search`` all search again. The reused record keeps its original
discovery record and dates, so the report reads the original search.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

from app.config import settings

POLICY = "paper-search-reuse-v2"
_REUSABLE_STATUSES = frozenset({"durable_authorized", "unavailable", "abstract_only", "metadata_only"})
_INCOMPLETE_REASONS = frozenset({"full_text_search_incomplete", "transient_verification_run_unavailable",
                                 "transient_source_admission_failed", "durable_representation_not_authorized"})


def _norm(value) -> str:
    return " ".join(str(value or "").split()).casefold()


def reference_signature(reference, *, identity_only: bool) -> tuple:
    """Unchanged reference text and parse, and the same cited/identity-only mode."""
    get = (lambda name: reference.get(name)) if isinstance(reference, dict) else (lambda name: getattr(reference, name, None))
    return (_norm(get("raw_ref")), _norm(get("source_kind")), _norm(get("title")), _norm(get("author")),
            _norm(get("year")), _norm(get("container_title")), _norm(get("doi")), bool(identity_only))


def _required_searches_completed(item: dict, reference: dict | None) -> bool:
    """reference-verification-v2 found every required search complete."""
    if not reference:
        return False
    from app.services.reference_verification import assess_reference_verification
    from app.services.schemas import ParsedReference
    try:
        parsed = ParsedReference.model_validate(reference)
    except Exception:
        return False
    verdict = assess_reference_verification(parsed, item.get("reference_discovery"))
    return verdict.get("reason_code") == "suited_searches_completed_without_match"


def _reusable(item: dict, reference: dict | None = None) -> bool:
    if item.get("status") == "durable_authorized":
        # The text itself is stored; how the search ended does not matter.
        return bool(item.get("representation_id"))
    discovery = item.get("reference_discovery") or {}
    if (isinstance(reference, dict) and reference.get("source_kind") == "book_section"
            and (reference.get("container_title") or "").strip()
            and discovery.get("outcome") not in {"confirmed", "confirmed_with_minor_differences"}
            and not discovery.get("container_identity")):
        # An unconfirmed chapter whose book was never identified is searched
        # again, so a corrected book lookup reaches it (Gershon, 2026-10-04).
        return False
    if item.get("status") == "link_check_only":
        # An uncited web page's own search (webpage-verification-v1): reused
        # once its web providers answered (paper 10's Statista, 2026-10-04).
        return bool(item.get("web_page_check")) and _web_search_answered(discovery)
    if item.get("status") not in _REUSABLE_STATUSES or item.get("reason_code") in _INCOMPLETE_REASONS:
        return False
    if item.get("full_text_search_incomplete_providers"):
        return False
    if item.get("retryable_provider_dependencies"):
        # An optional provider left a retry, but every search the reference's
        # kind requires finished (paper 10: OpenAlex re-searched each run, 2026-10-04).
        return _required_searches_completed(item, reference)
    if _network_failed(discovery):
        return False
    if discovery.get("outcome") == "search_incomplete":
        return _required_searches_completed(item, reference)
    return bool(discovery)


def _web_search_answered(discovery: dict) -> bool:
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    web = [a for a in attempts if a.get("route_category") == "bounded_web"]
    return bool(web) and all(a.get("outcome") in {"candidate_found", "no_match"} for a in web)


def _network_failed(discovery: dict) -> bool:
    """The earlier run could not reach the student's link, or most indexes:
    a network failure, not a finished search (2026-10-02)."""
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    # A failed student link matters only when nothing else confirmed the work:
    # a confirmed reference is not searched again for it (paper 4 re-run,
    # 2026-10-02: 15 confirmed references re-searched, USD 0.35).
    if (discovery.get("outcome") not in {"confirmed", "confirmed_with_minor_differences"}
            and any(a.get("route_category") == "student_url" and a.get("outcome") == "operational_failure"
                    for a in attempts)):
        return True
    adapters = [a for a in attempts if a.get("route_category") == "academic_adapter"
                and a.get("outcome") != "unavailable"]
    failed = sum(1 for a in adapters if a.get("outcome") == "operational_failure")
    return bool(adapters) and failed * 2 >= len(adapters)


def prior_results_index(session, job) -> dict:
    """signature -> (prior job id, result item), most recent completed run first."""
    days = int(getattr(settings, "SEARCH_RERUN_REUSE_DAYS", 0) or 0)
    if days <= 0 or not hasattr(session, "query") or not getattr(job, "scope_id", None):
        return {}
    from app.models.job import Job, JobStatus
    since = datetime.now(timezone.utc) - timedelta(days=days)
    prior = (session.query(Job)
             .filter(Job.scope_type == job.scope_type, Job.scope_id == job.scope_id, Job.id != job.id,
                     Job.status == JobStatus.COMPLETED, Job.created_at >= since,
                     Job.source_results.isnot(None), Job.extraction_payload.isnot(None))
             .order_by(Job.created_at.desc()).limit(50).all())
    index: dict = {}
    for earlier in prior:
        references = {ref.get("reference_id"): ref for ref in (earlier.extraction_payload or {}).get("references") or []}
        for item in earlier.source_results or []:
            reference = references.get(item.get("reference_id"))
            if reference is None or not _reusable(item, reference):
                continue
            key = reference_signature(reference, identity_only=bool(item.get("identity_only_policy")))
            index.setdefault(key, (str(earlier.id), item))
    return index


def reused_result(index: dict, reference, *, identity_only: bool) -> dict | None:
    """A copy of the earlier result for this reference, re-keyed to it, or None."""
    found = index.get(reference_signature(reference, identity_only=identity_only))
    if found is None:
        return None
    prior_job_id, item = found
    record = copy.deepcopy(item)
    record["reference_id"] = reference.reference_id
    if isinstance(record.get("reference_discovery"), dict):
        record["reference_discovery"]["reference_id"] = reference.reference_id
    # The link checks travel with the result: re-bind each one to this
    # reference when it checked the same submitted address (2026-10-02; a
    # stale binding dropped every reused reference's link findings).
    rows = record.get("submitted_link_observations")
    if isinstance(rows, list) and rows:
        from app.services.submitted_links import initial_observations
        current = {(row.kind, row.submitted_sha256, row.request_sha256): row
                   for row in initial_observations(reference)}
        for row in rows:
            match = current.get((row.get("kind"), row.get("submitted_sha256"), row.get("request_sha256"))) \
                if isinstance(row, dict) else None
            if match is not None:
                row["reference_id"] = reference.reference_id
                row["reference_snapshot_sha256"] = match.reference_snapshot_sha256
    record["search_reuse"] = {"policy_version": POLICY, "from_job_id": prior_job_id,
                              "searched_at": (record.get("reference_discovery") or {}).get("created_at")}
    return record
