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
SEARCH_AGAIN_KEY = "search_again_requests"
AUTO_RETRY_REASON = "incomplete_search_retry"
SEARCH_AGAIN_REASON = "user_search_again"
SEARCH_AGAIN_INTERVAL = timedelta(days=1)
_AUTHORIZED_STATUSES = frozenset({"durable_authorized", "transient_authorized"})
_KEPT_RUNS = 50
_KEPT_REQUESTS = 10


class SearchAgainRateLimited(Exception):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("This reference was already searched again within the last day")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


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


def prepare_search_again_refresh(
    session: Session,
    *,
    report_id: str | uuid.UUID,
    reference_id: str,
    scope_type: str,
    scope_id: str,
    now: datetime | None = None,
    commit: bool = True,
) -> dict:
    """Start a forced search of one reference the reader asked to search again."""
    current = _as_utc(now or datetime.now(timezone.utc))
    try:
        parsed_report_id = (
            report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id))
        )
    except (TypeError, ValueError) as exc:
        raise PaperWorkflowError("report_missing", "Report does not exist") from exc
    requested_report = session.get(Report, parsed_report_id)
    if requested_report is None:
        raise PaperWorkflowError("report_missing", "Report does not exist")
    job = session.scalar(
        select(Job).where(Job.id == requested_report.job_id).with_for_update()
    )
    if job is None or job.scope_type != scope_type or job.scope_id != scope_id:
        raise PaperWorkflowError("report_missing", "Report does not exist")
    if job_marks_released(session, job):
        raise PaperWorkflowError(
            "assessment_marks_released", "Marks have been released for this assessment"
        )
    if job.store_only or not job.extraction_payload:
        raise PaperWorkflowError(
            "report_dependencies_unavailable",
            "Report dependencies are unavailable for a new search",
        )
    requests = list(((job.upload_evidence or {}).get(SEARCH_AGAIN_KEY) or {}).get(reference_id) or [])
    last = max(
        (parsed for parsed in (_parse_time(item.get("requested_at")) for item in requests) if parsed),
        default=None,
    )
    if last is not None and current - last < SEARCH_AGAIN_INTERVAL:
        raise SearchAgainRateLimited((last + SEARCH_AGAIN_INTERVAL - current).total_seconds())
    if job.status != JobStatus.COMPLETED or (job.upload_evidence or {}).get(
        TARGETED_SOURCE_REFRESH_KEY
    ):
        raise PaperWorkflowError(
            "report_reanalysis_busy", "Another report reanalysis is already in progress"
        )
    if not job_accepts_search_refresh(session, job, now=current):
        raise PaperWorkflowError(
            "report_dependencies_unavailable",
            "Report dependencies are unavailable for a new search",
        )
    result = next(
        (item for item in job.source_results or [] if item.get("reference_id") == reference_id),
        None,
    )
    if result is None or not source_result_search_incomplete(result):
        raise PaperWorkflowError(
            "search_not_incomplete", "This reference has no incomplete search"
        )
    latest_report = session.scalar(
        select(Report)
        .where(Report.job_id == job.id)
        .order_by(Report.report_version.desc(), Report.created_at.desc())
        .limit(1)
    )
    attempt_id = begin_reference_search_refresh(
        job,
        [reference_id],
        reason=SEARCH_AGAIN_REASON,
        force_search=True,
        extra={
            "requested_report_id": str(requested_report.id),
            "base_report_id": str(latest_report.id),
            "base_report_version": latest_report.report_version,
        },
    )
    upload_evidence = deepcopy(dict(job.upload_evidence or {}))
    recorded = dict(upload_evidence.get(SEARCH_AGAIN_KEY) or {})
    recorded[reference_id] = [
        *requests,
        {
            "requested_at": current.isoformat(),
            "attempt_id": attempt_id,
            "requested_report_id": str(requested_report.id),
        },
    ][-_KEPT_REQUESTS:]
    upload_evidence[SEARCH_AGAIN_KEY] = recorded
    job.upload_evidence = upload_evidence
    if commit:
        session.commit()
    else:
        session.flush()
    return {
        "job_id": str(job.id),
        "attempt_id": attempt_id,
        "report_id": str(latest_report.id),
        "reference_id": reference_id,
    }


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
