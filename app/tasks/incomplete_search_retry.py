"""Scheduled automatic retries of completed papers' incomplete searches."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import Text, cast, select

from app.config import settings
from app.database import SessionLocal
from app.models.job import Job, JobStatus
from app.services.assessment_marks import marks_released_clause
from app.services.paper_dispatch import attempt_id_for
from app.services.search_retry import prepare_incomplete_search_retry, retry_delays
from app.tasks.celery_app import celery_app
from app.tasks.check_paper import dispatch_paper_workflow

_SCAN_LIMIT = 500


@celery_app.task(name="retry_incomplete_searches")
def retry_incomplete_searches() -> dict:
    """Start one grouped targeted refresh per paper with due incomplete searches."""
    # One switch for every automatic re-run (owner decision 2026-09-29).
    if not settings.PROVIDER_RECOVERY_REFRESH_ENABLED:
        return {"jobs_requeued": 0, "reference_members": 0,
                "disabled": "PROVIDER_RECOVERY_REFRESH_ENABLED"}
    from app.services.assessment_marks import institutional_deployment
    if not institutional_deployment():
        # Personal: searching again is always the user's choice (owner decision 2026-09-29).
        return {"jobs_requeued": 0, "reference_members": 0, "disabled": "personal_deployment"}
    delays = retry_delays()
    if not delays:
        return {"jobs_requeued": 0, "reference_members": 0}
    window_hours = settings.INCOMPLETE_SEARCH_RETRY_WINDOW_HOURS
    window = timedelta(hours=window_hours) if window_hours > 0 else None
    limit = max(1, settings.INCOMPLETE_SEARCH_RETRY_MAX_JOBS)
    now = datetime.now(timezone.utc)
    scheduled: list[tuple[str, int, str]] = []
    with SessionLocal() as session:
        jobs = list(
            session.scalars(
                select(Job)
                .where(
                    Job.status == JobStatus.COMPLETED,
                    Job.store_only.is_(False),
                    Job.source_results.is_not(None),
                    # Matches both `search_incomplete` and
                    # `full_text_search_incomplete`; each record is then
                    # checked exactly.
                    cast(Job.source_results, Text).ilike("%search_incomplete%"),
                    # Released marks stop retries; excluded here so they never
                    # occupy the bounded scan.
                    ~marks_released_clause(),
                )
                .order_by(Job.updated_at)
                .limit(_SCAN_LIMIT)
                .with_for_update(skip_locked=True)
            )
        )
        for job in jobs:
            retried = prepare_incomplete_search_retry(
                session, job, now=now, delays=delays, window=window
            )
            if retried:
                scheduled.append((str(job.id), len(retried), attempt_id_for(job)))
                if len(scheduled) >= limit:
                    break
        session.commit()

    pending = 0
    for job_id, _count, attempt_id in scheduled:
        try:
            dispatch_paper_workflow(job_id, attempt_id)
        except Exception:
            # The committed intent is resumed by recover_pending_paper_workflows.
            pending += 1
    return {
        "jobs_requeued": len(scheduled),
        "reference_members": sum(count for _job, count, _attempt in scheduled),
        "publication_pending": pending,
        "bounded_job_limit": limit,
    }
