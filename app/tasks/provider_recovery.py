"""Bounded refresh of paper references affected by a recovered provider."""

from sqlalchemy import Text, cast, select

from app.config import settings
from app.database import SessionLocal
from app.models.job import Job, JobStatus
from app.services.paper_workflow import prepare_provider_recovery_refresh
from app.tasks.celery_app import celery_app
from app.tasks.check_paper import dispatch_paper_workflow
from app.services.paper_dispatch import attempt_id_for


def schedule_provider_recovery(provider: str) -> None:
    """Delay the scan so a paper that observed recovery can finish checkpointing."""
    celery_app.send_task(
        "requeue_recovered_provider_work",
        args=[provider],
        countdown=60,
    )


@celery_app.task(name="requeue_recovered_provider_work")
def requeue_recovered_provider_work(provider: str, force_search: bool = False, automatic: bool = True) -> dict:
    """Refresh only completed jobs with retryable dependencies on ``provider``.

    ``force_search`` repeats paid web searches even where the job's scope holds
    a completed-search memo (search-reuse-memo-v1).
    """
    normalized = provider.strip().casefold()
    if not normalized:
        return {"provider": "", "jobs_requeued": 0, "reference_members": 0}
    # Both profiles re-run on their own once a provider recovers (owner decision
    # 2026-10-18, replacing the Personal user's Search again of 2026-09-29).
    if not settings.PROVIDER_RECOVERY_REFRESH_ENABLED:
        # The single gate every automatic re-run passes through: live web
        # search reaches this task via `schedule_provider_recovery`.
        return {"provider": normalized, "jobs_requeued": 0, "reference_members": 0,
                "disabled": "PROVIDER_RECOVERY_REFRESH_ENABLED"}
    limit = max(1, settings.PROVIDER_RECOVERY_MAX_JOBS)
    scan_limit = max(100, min(limit * 10, 2_000))
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    scheduled: list[tuple[str, int, str]] = []
    with SessionLocal() as session:
        jobs = list(
            session.scalars(
                select(Job)
                .where(
                    Job.status == JobStatus.COMPLETED,
                    Job.source_results.is_not(None),
                    cast(Job.source_results, Text).ilike(
                        f'%"{escaped}"%', escape="\\"
                    ),
                )
                .order_by(Job.updated_at)
                .limit(scan_limit)
                .with_for_update(skip_locked=True)
            )
        )
        from app.services.storage_policy import finishes_searches_automatically
        for job in jobs:
            # A hand-started run passes automatic=False; otherwise Personal
            # scopes only when their owner turned it on (owner decision 2026-10-18).
            if automatic and not finishes_searches_automatically(session, job.scope_type, job.scope_id):
                continue
            affected = prepare_provider_recovery_refresh(
                session,
                job.id,
                provider=normalized,
                commit=False,
                force_search=force_search,
            )
            if affected:
                scheduled.append((str(job.id), len(affected), attempt_id_for(job)))
                if len(scheduled) >= limit:
                    break
        session.commit()

    pending = 0
    for job_id, _count, attempt_id in scheduled:
        try:
            dispatch_paper_workflow(job_id, attempt_id)
        except Exception:
            pending += 1
    return {
        "provider": normalized,
        "jobs_requeued": len(scheduled),
        "reference_members": sum(count for _job_id, count, _attempt in scheduled),
        "publication_pending": pending,
        "bounded_job_limit": limit,
    }
