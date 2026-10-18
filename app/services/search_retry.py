"""Repeat incomplete reference searches of a completed paper (owner decision 2026-09-29).

Two routes start the same targeted refresh (`begin_reference_search_refresh`):

* automatic retries: a scheduled scan retries a reference whose search ended
  incomplete after each configured delay (default 1 hour, 1 day, 3 days),
  measured from the incomplete search being retried, at most once per delay;
* "Search Again": the report reader asks for one reference to be searched
  again, with ``force_search``, at most once per day per reference.

An incomplete search is a discovery outcome of ``search_incomplete`` or a
full-text reason of ``full_text_search_incomplete`` on a reference with no
authorized source. A completed search is never retried here: it may hold a
completed-search memo (search-reuse-memo-v1), and retrying it would only
repeat paid work. Both routes record what they started in the job's
``upload_evidence``.

Neither route runs for a paper whose assessment has marks released
(assessment_marks.py): further searching can no longer affect it.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job, JobStatus
from app.models.report import Report, ReportPaperArtifactRecord
from app.services.assessment_marks import job_marks_released
from app.services.paper_workflow import (
    TARGETED_SOURCE_REFRESH_KEY,
    PaperWorkflowError,
    begin_reference_search_refresh,
)

AUTO_RETRY_KEY = "incomplete_search_retry"
AUTO_RETRY_REASON = "incomplete_search_retry"
_AUTHORIZED_STATUSES = frozenset({"durable_authorized", "transient_authorized"})
_KEPT_RUNS = 50


def source_result_search_incomplete(item: dict) -> bool:
    """Whether one stored source result records a search that did not finish."""
    if not isinstance(item, dict) or not item.get("reference_id"):
        return False
    if item.get("status") in _AUTHORIZED_STATUSES:
        # A source is in hand (possibly uploaded); searching again could only
        # replace it.
        return False
    if "author_metadata_lookup" in item:
        # A bibliography-only author lookup never runs the web route by design.
        return False
    return (
        item.get("reason_code") == "full_text_search_incomplete"
        or (item.get("reference_discovery") or {}).get("outcome") == "search_incomplete"
    )


def incomplete_search_at(item: dict) -> datetime | None:
    """When the incomplete search recorded on ``item`` ran."""
    for record in (item.get("reference_discovery"), item.get("reference_discovery_trace")):
        value = (record or {}).get("created_at") if isinstance(record, dict) else None
        parsed = _parse_time(value)
        if parsed is not None:
            return parsed
    return None


def retry_delays() -> list[timedelta]:
    delays = []
    for part in str(settings.INCOMPLETE_SEARCH_RETRY_DELAYS_HOURS or "").split(","):
        try:
            hours = float(part.strip())
        except ValueError:
            continue
        if hours > 0:
            delays.append(timedelta(hours=hours))
    return delays


def job_accepts_search_refresh(session: Session, job: Job, *, now: datetime) -> bool:
    """A completed, idle job whose report and paper surface are still retained,
    and whose assessment (if any) does not have marks released."""
    if (
        job.status != JobStatus.COMPLETED
        or job.store_only
        or not job.extraction_payload
        or (job.upload_evidence or {}).get(TARGETED_SOURCE_REFRESH_KEY)
    ):
        return False
    if session.scalar(select(Report.id).where(Report.job_id == job.id).limit(1)) is None:
        return False
    if job_marks_released(session, job):
        return False
    artifact = session.scalar(
        select(ReportPaperArtifactRecord).where(ReportPaperArtifactRecord.job_id == job.id)
    )
    return bool(
        artifact is not None
        and artifact.deleted_at is None
        and _as_utc(artifact.expires_at) > now
    )


def due_retry_references(
    job: Job,
    *,
    now: datetime,
    delays: list[timedelta],
    window: timedelta | None,
) -> list[str]:
    """References of ``job`` whose next automatic retry is due now."""
    state = (job.upload_evidence or {}).get(AUTO_RETRY_KEY) or {}
    references = state.get("references") or {}
    due = []
    for item in job.source_results or []:
        if not source_result_search_incomplete(item):
            continue
        reference_id = str(item["reference_id"])
        attempts = len((references.get(reference_id) or {}).get("attempts") or [])
        if attempts >= len(delays):
            continue
        searched_at = incomplete_search_at(item)
        if searched_at is None:
            continue
        due_at = searched_at + delays[attempts]
        if now < due_at or (window is not None and now > due_at + window):
            continue
        due.append(reference_id)
    return sorted(set(due))


def prepare_incomplete_search_retry(
    session: Session,
    job: Job,
    *,
    now: datetime,
    delays: list[timedelta],
    window: timedelta | None,
) -> list[str]:
    """Start one grouped automatic retry for every due reference of ``job``.

    The caller holds the job row lock and commits.
    """
    if not delays or not job_accepts_search_refresh(session, job, now=now):
        return []
    due = due_retry_references(job, now=now, delays=delays, window=window)
    if not due:
        return []
    searched = {
        str(item.get("reference_id")): incomplete_search_at(item)
        for item in job.source_results or []
        if str(item.get("reference_id")) in due
    }
    attempt_id = begin_reference_search_refresh(job, due, reason=AUTO_RETRY_REASON)
    upload_evidence = deepcopy(dict(job.upload_evidence or {}))
    state = dict(upload_evidence.get(AUTO_RETRY_KEY) or {})
    references = dict(state.get("references") or {})
    stamp = now.isoformat()
    for reference_id in due:
        entry = dict(references.get(reference_id) or {})
        entry["attempts"] = [
            *(entry.get("attempts") or []),
            {
                "started_at": stamp,
                "attempt_id": attempt_id,
                "incomplete_search_at": (
                    searched[reference_id].isoformat() if searched.get(reference_id) else None
                ),
            },
        ]
        references[reference_id] = entry
    state["references"] = references
    state["runs"] = [
        *(state.get("runs") or []),
        {"attempt_id": attempt_id, "reference_ids": due, "started_at": stamp},
    ][-_KEPT_RUNS:]
    upload_evidence[AUTO_RETRY_KEY] = state
    job.upload_evidence = upload_evidence
    session.flush()
    return due


def _parse_time(value) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
