"""Dead letter queue for tasks that fail after max retries (R4 mitigation).

Tasks that exhaust their retry attempts are logged here for debugging.
In production, this could write to a database table or monitoring system.
"""

import logging
from datetime import datetime

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# In-memory dead letter store (replace with DB table in production)
_dead_letter_store: list[dict] = []


@celery_app.task(name="dead_letter_handler")
def dead_letter_handler(task_name: str, task_id: str, args: tuple, kwargs: dict, error: str):
    """
    Handle a task that failed after max retries.

    This task is called when a task exhausts its retry limit.
    In production, this should:
    - Write to a `failed_tasks` DB table
    - Send an alert to admin
    - Store enough context for debugging

    Args:
        task_name: The name of the failed task.
        task_id: The UUID of the failed task.
        args: Positional arguments of the failed task.
        kwargs: Keyword arguments of the failed task.
        error: The error message from the final failure.
    """
    entry = {
        "task_name": task_name,
        "task_id": task_id,
        "args": args,
        "kwargs": kwargs,
        "error": error,
        "failed_at": datetime.utcnow().isoformat(),
    }

    _dead_letter_store.append(entry)

    logger.critical(
        "Task sent to dead letter queue",
        extra={
            "task_name": task_name,
            "task_id": task_id,
            "error": error,
            "dead_letter_count": len(_dead_letter_store),
        },
    )


def get_dead_letter_count() -> int:
    """Return the number of tasks in the dead letter queue."""
    return len(_dead_letter_store)


def get_dead_letter_entries() -> list[dict]:
    """Return all dead letter entries for inspection."""
    return list(_dead_letter_store)


def clear_dead_letter_queue() -> None:
    """Clear the dead letter queue (for testing)."""
    _dead_letter_store.clear()
