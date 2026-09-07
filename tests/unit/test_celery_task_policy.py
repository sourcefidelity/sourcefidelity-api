"""Celery limits distinguish workflow deadlines from bounded task safety."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.services.paper_workflow import PaperWorkflowError
from app.tasks.check_paper import (
    _is_retryable,
    _job_execution_lock,
    check_paper_task,
)
from app.tasks.dead_letter import dead_letter_handler
from app.tasks.source_retention import cleanup_expired_source_representations
from app.tasks.verification_run_cleanup import cleanup_stale_verification_run_objects
from app.tasks.paper_job_cleanup import cleanup_stale_paper_job_input_objects


def test_paper_workflow_has_no_global_monolithic_limits() -> None:
    assert celery_app.conf.task_soft_time_limit is None
    assert celery_app.conf.task_time_limit is None
    assert celery_app.conf.task_default_rate_limit is None
    assert check_paper_task.soft_time_limit is None
    assert check_paper_task.time_limit is None
    assert celery_app.conf.broker_connection_retry_on_startup is True


def test_existing_executable_tasks_keep_explicit_safety_limits() -> None:
    assert cleanup_expired_source_representations.soft_time_limit == 300
    assert cleanup_expired_source_representations.time_limit == 360
    assert cleanup_stale_verification_run_objects.soft_time_limit == 300
    assert cleanup_stale_verification_run_objects.time_limit == 360
    assert cleanup_stale_paper_job_input_objects.soft_time_limit == 300
    assert cleanup_stale_paper_job_input_objects.time_limit == 360
    assert dead_letter_handler.soft_time_limit == 30
    assert dead_letter_handler.time_limit == 60


def test_job_execution_lock_is_a_noop_outside_postgresql() -> None:
    with Session(create_engine("sqlite+pysqlite:///:memory:")) as session:
        with _job_execution_lock(session, "job-1", "extract"):
            pass


def test_job_execution_lock_fails_closed_when_stage_is_already_running() -> None:
    class Result:
        @staticmethod
        def scalar_one() -> bool:
            return False

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def execute(*_args, **_kwargs):
            return Result()

    bind = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        connect=lambda: Connection(),
    )
    session = SimpleNamespace(get_bind=lambda: bind)

    with pytest.raises(PaperWorkflowError) as exc_info:
        with _job_execution_lock(session, "job-1", "extract"):
            pass

    assert exc_info.value.code == "job_stage_busy"
    assert _is_retryable(exc_info.value)
