"""Celery limits distinguish workflow deadlines from bounded task safety."""

from app.tasks.celery_app import celery_app
from app.tasks.check_paper import check_paper_task
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


def test_existing_executable_tasks_keep_explicit_safety_limits() -> None:
    assert cleanup_expired_source_representations.soft_time_limit == 300
    assert cleanup_expired_source_representations.time_limit == 360
    assert cleanup_stale_verification_run_objects.soft_time_limit == 300
    assert cleanup_stale_verification_run_objects.time_limit == 360
    assert cleanup_stale_paper_job_input_objects.soft_time_limit == 300
    assert cleanup_stale_paper_job_input_objects.time_limit == 360
    assert dead_letter_handler.soft_time_limit == 30
    assert dead_letter_handler.time_limit == 60
