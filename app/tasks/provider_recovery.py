"""Bounded refresh of paper references affected by a recovered provider."""

from sqlalchemy import Text, cast, select

from app.config import settings
from app.database import SessionLocal
from app.models.job import Job, JobStatus
from app.services.retrieval.provider_runtime import (
    ProviderHealthStore,
    ProviderPolicy,
    provider_policy,
)
from app.services.search import get_search_provider
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


@celery_app.task(name="probe_retrieval_provider_recovery")
def probe_retrieval_provider_recovery() -> dict:
    """Run one half-open probe for each due managed web-search incident."""
    from app.services.retrieval.web_search import _searx_provider_key

    store = ProviderHealthStore()
    default_policy = ProviderPolicy(
        timeout_seconds=15.0,
        cooldown_seconds=180,
        max_cooldown_seconds=3600,
    )
    probed = 0
    recovered = 0
    configured_searx_keys = {
        _searx_provider_key(group)
        for group in settings.SEARXNG_ENGINE_GROUPS.split(";")
        if group.strip()
    }
    api_first = settings.SEARCH_POLICY_VERSION == "api-first-search-v2"
    if api_first and not settings.SEARCH_SEARXNG_FALLBACK_ENABLED:
        configured_searx_keys = set()
    supported_keys = [
        key
        for key in store.incident_providers()
        # Preserve old/experimental incidents, but do not turn an engine
        # removed from the configured cascade into indefinite probe traffic.
        # A stale incident for a removed provider (duckduckgo) is therefore
        # kept in the store and never probed.
        if key in configured_searx_keys
    ]
    for provider_key in supported_keys:
        if store.cooldown_remaining(provider_key) > 0:
            continue
        if not store.claim_recovery_probe(provider_key):
            continue
        probed += 1
        provider_name = "searxng" if provider_key.startswith("searxng:") else provider_key
        policy = provider_policy(provider_name, default_policy)
        search = get_search_provider(provider_name)
        if search is None:
            store.record_unavailable(
                provider_key, policy, status="provider_not_configured"
            )
            continue
        engines = provider_key.partition(":")[2] or "default"
        search_kwargs = (
            {"engines": None if engines == "default" else engines}
            if provider_name == "searxng"
            else {}
        )
        results = search.search(
            "sourcefidelity provider availability probe",
            num_results=1,
            **search_kwargs,
        )
        status = str(getattr(search, "last_status", "operational_failure"))
        if results or status == "completed":
            if store.record_success(provider_key):
                recovered += 1
                schedule_provider_recovery(provider_name)
            continue
        if status == "timeout":
            store.record_timeout(provider_key, policy)
        else:
            store.record_unavailable(provider_key, policy, status=status)
    return {
        "provider": "managed_web_search",
        "incidents_probed": probed,
        "incidents_recovered": recovered,
    }


@celery_app.task(name="requeue_recovered_provider_work")
def requeue_recovered_provider_work(provider: str, force_search: bool = False, automatic: bool = True) -> dict:
    """Refresh only completed jobs with retryable dependencies on ``provider``.

    ``force_search`` repeats paid web searches even where the job's scope holds
    a completed-search memo (search-reuse-memo-v1).
    """
    normalized = provider.strip().casefold()
    if not normalized:
        return {"provider": "", "jobs_requeued": 0, "reference_members": 0}
    from app.services.assessment_marks import institutional_deployment
    if automatic and not institutional_deployment():
        # Personal: no automatic re-searching; the user decides with Search again
        # (owner decision 2026-09-29). A hand-started run passes automatic=False.
        return {"provider": normalized, "jobs_requeued": 0, "reference_members": 0,
                "disabled": "personal_deployment"}
    if not settings.PROVIDER_RECOVERY_REFRESH_ENABLED:
        # The single gate every automatic re-run passes through: the probe and
        # live web search both reach this task via `schedule_provider_recovery`.
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
        for job in jobs:
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
