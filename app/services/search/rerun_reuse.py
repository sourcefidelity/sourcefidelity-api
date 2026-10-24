"""Reuse a paper's earlier search results when it is checked again (`paper-search-reuse-v1`, now v4).

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

A transient text (``transient_authorized``) is not kept, but an open-access
copy verified with high identity confidence keeps its public address and
content hash (``verified-public-source-access-v1``): it is downloaded again
with the same checks and used only if the file is byte-identical
(``paper-search-reuse-v4``, owner decision 2026-10-07); otherwise the reference
is searched again. A transient text without that record is searched again. A search that was
incomplete, a likely other-language work whose earlier search did not complete
a search in the work's own script (``native-script-route-v1``,
``paper-search-reuse-v5``, owner decision 2026-10-12), a reference whose parse changed, a cited reference whose earlier
record was identity-only (or the reverse) and a targeted refresh with
``force_search`` all search again. The reused record keeps its original
discovery record and dates, so the report reads the original search.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

from app.config import settings
# The reuse rules (source_reuse, 2026-10-15); these names stay importable here.
from app.services.source_reuse import (  # noqa: F401
    _network_failed, _required_searches_completed, downloadable, earlier_result,
    native_script_route_due, public_copy,
)

POLICY = "paper-search-reuse-v5"




def _norm(value) -> str:
    return " ".join(str(value or "").split()).casefold()


def reference_signature(reference, *, identity_only: bool) -> tuple:
    """Unchanged reference text and parse, and the same cited/identity-only mode.
    The parsed link is part of the parse: a repaired address (a query split by
    a space, 2026-10-14) is a link the earlier search never followed."""
    get = (lambda name: reference.get(name)) if isinstance(reference, dict) else (lambda name: getattr(reference, name, None))
    return (_norm(get("raw_ref")), _norm(get("source_kind")), _norm(get("title")), _norm(get("author")),
            _norm(get("year")), _norm(get("container_title")), _norm(get("doi")), bool(identity_only),
            _norm(get("url")))


def excluded_job_ids() -> frozenset[str]:
    """Runs whose results are never reused (``SEARCH_REUSE_EXCLUDED_JOBS``)."""
    raw = getattr(settings, "SEARCH_REUSE_EXCLUDED_JOBS", "") or ""
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def prior_results_index(session, job) -> dict:
    """signature -> (prior job id, result item), most recent completed run first."""
    days = int(getattr(settings, "SEARCH_RERUN_REUSE_DAYS", 0) or 0)
    if days <= 0 or not hasattr(session, "query") or not getattr(job, "scope_id", None):
        return {}
    from app.models.job import Job, JobStatus
    since = datetime.now(timezone.utc) - timedelta(days=days)
    base = (session.query(Job)
            .filter(Job.scope_type == job.scope_type, Job.scope_id == job.scope_id, Job.id != job.id,
                    Job.status == JobStatus.COMPLETED, Job.created_at >= since,
                    Job.source_results.isnot(None), Job.extraction_payload.isnot(None)))
    prior = base.order_by(Job.created_at.desc()).limit(50).all()
    # The same paper's own earlier runs always count, however many other
    # papers ran since (2026-10-08: the long article's run had fallen out of
    # the 50 most recent, so none of its 65 results was reused).
    filename = getattr(job, "filename", None)
    if filename:
        seen = {item.id for item in prior}
        same = base.filter(Job.filename == filename).order_by(Job.created_at.desc()).limit(10).all()
        prior = sorted([*prior, *(item for item in same if item.id not in seen)],
                       key=lambda item: item.created_at, reverse=True)
    excluded = excluded_job_ids()
    index: dict = {}
    for earlier in prior:
        if str(earlier.id).lower() in excluded:
            continue
        references = {ref.get("reference_id"): ref for ref in (earlier.extraction_payload or {}).get("references") or []}
        for item in earlier.source_results or []:
            reference = references.get(item.get("reference_id"))
            if reference is None:
                continue
            key = reference_signature(reference, identity_only=bool(item.get("identity_only_policy")))
            decision = earlier_result(item, reference)
            if not decision.reusable:
                # Why the most recent earlier result was not reused (provenance).
                index.setdefault(("refused",) + key, decision.reason)
                index.setdefault(("refused",) + parsed_signature(key), decision.reason)
                continue
            index.setdefault(key, (str(earlier.id), item))
            index.setdefault(parsed_signature(key), (str(earlier.id), item))
    return index


def parsed_signature(signature: tuple) -> tuple:
    """The same work parsed identically, whatever the reference's punctuation
    or layout: another paper citing it in its own format reuses the result
    (owner decision 2026-10-09). The reference text itself is left out."""
    return ("parsed",) + tuple(signature[1:])


def _identifying(signature: tuple) -> bool:
    """A parse with a DOI, or a title with an author and year, names one work."""
    _, _, title, author, year, _, doi = signature[:7]
    return bool(doi) or bool(title and author and year)


def reused_result(index: dict, reference, *, identity_only: bool) -> dict | None:
    """A copy of the earlier result for this reference, re-keyed to it, or None."""
    signature = reference_signature(reference, identity_only=identity_only)
    found = index.get(signature)
    if found is None and _identifying(signature):
        found = index.get(parsed_signature(signature))
    if found is None:
        return None
    prior_job_id, item = found
    record = copy.deepcopy(item)
    from app.services.discovery_recorder import repaired_archive_record
    repaired = repaired_archive_record(record.get("reference_discovery_trace"), record.get("reference_discovery"))
    if repaired is not None and repaired.record is not None:
        # A magazine-archive search stored as found with no candidate dropped
        # the record; it is re-derived from the stored trace, without searching
        # again (owner decision 2026-10-22).
        record["reference_discovery_trace"], record["reference_discovery"] = repaired.trace, repaired.record
        record["archive_record_repaired"] = True
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


def reuse_refusal(index: dict, reference, *, identity_only: bool) -> str | None:
    """Why the most recent earlier result for this reference was not reused, if one was refused."""
    signature = reference_signature(reference, identity_only=identity_only)
    found = index.get(("refused",) + signature)
    if found is None and _identifying(signature):
        found = index.get(("refused",) + parsed_signature(signature))
    return found
