"""Celery application instance."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "sourcefidelity",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
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
    # Task timeouts (R4 mitigation)
    task_soft_time_limit=300,  # 5 minutes before SoftTimeLimitExceeded
    task_time_limit=360,       # 6 minutes hard limit before termination
    task_ignore_result=False,
    # Result backend config
    result_expires=86400,      # Auto-delete results after 24 hours
    # Default retry policy (R4 mitigation)
    task_default_retry_delay=30,      # Wait 30s before first retry
    task_max_retries=3,               # Max 3 retries per task
    task_default_rate_limit="10/m",   # Max 10 tasks per minute
    # Worker settings
    worker_max_memory_per_child=500_000,  # 500MB memory limit per worker (R4)
    worker_max_tasks_per_child=100,       # Restart worker after 100 tasks
    worker_send_task_events=True,         # Enable task events for monitoring
    task_send_sent_event=True,
)
