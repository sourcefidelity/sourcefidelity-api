"""Celery application instance."""

from celery import Celery

from app.config import settings
from app.log_safety import configure_sensitive_transport_logging

configure_sensitive_transport_logging()

celery_app = Celery(
    "sourcefidelity",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.check_paper",
        "app.tasks.dead_letter",
        "app.tasks.source_retention",
        "app.tasks.verification_run_cleanup",
        "app.tasks.paper_job_cleanup",
        "app.tasks.provider_recovery",
        "app.tasks.source_reanalysis",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # Do not impose one time or rate limit on every task. A paper is a durable,
    # checkpointed workflow whose report deadline is distinct from the safety
    # budget of any one stage. Executable stages and maintenance tasks own their
    # limits; provider pacing belongs to provider-specific queues/adapters.
    task_ignore_result=False,
    # Result backend config
    result_expires=86400,      # Auto-delete results after 24 hours
    # Default retry policy (R4 mitigation)
    task_default_retry_delay=30,      # Wait 30s before first retry
    task_max_retries=3,               # Max 3 retries per task
    # Worker settings
    worker_max_memory_per_child=500_000,  # 500MB memory limit per worker (R4)
    worker_max_tasks_per_child=100,       # Restart worker after 100 tasks
    worker_send_task_events=True,         # Enable task events for monitoring
    task_send_sent_event=True,
    beat_schedule={
        "recover-pending-paper-workflows": {
            "task": "recover_pending_paper_workflows",
            "schedule": 60,
        },
        "cleanup-expired-source-representations": {
            "task": "cleanup_expired_source_representations",
            "schedule": max(
                60,
                settings.SOURCE_RETENTION_CLEANUP_INTERVAL_SECONDS,
            ),
        },
        "cleanup-stale-verification-runs": {
            "task": "cleanup_stale_verification_runs",
            "schedule": max(
                60,
                settings.VERIFICATION_RUN_CLEANUP_INTERVAL_SECONDS,
            ),
        },
        "cleanup-stale-paper-job-inputs": {
            "task": "cleanup_stale_paper_job_inputs",
            "schedule": max(60, settings.PAPER_UPLOAD_CLEANUP_INTERVAL_SECONDS),
        },
        "probe-retrieval-provider-recovery": {
            "task": "probe_retrieval_provider_recovery",
            "schedule": max(60, settings.PROVIDER_HEALTH_PROBE_INTERVAL_SECONDS),
        },
    },
)
