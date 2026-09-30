"""Judgment runs on their own queue, so paper checks are never delayed by them."""
import logging
import uuid

from app.database import SessionLocal
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

JUDGMENT_QUEUE = "judgment"


@celery_app.task(name="app.tasks.judgment.run_report_judgment", acks_late=True, max_retries=0)
def run_report_judgment(run_id: str) -> dict:
    from app.services.judgment_runs import execute_run

    with SessionLocal() as session:
        try:
            run = execute_run(session, uuid.UUID(run_id))
        except Exception as exc:   # record the failure; never retry into paid calls
            session.rollback()
            from app.models.judgment import JudgmentRun
            run = session.get(JudgmentRun, uuid.UUID(run_id))
            if run is not None and run.status in {"queued", "running"}:
                run.status, run.reason_code = "failed", type(exc).__name__[:80]
                session.commit()
            logger.warning("Judgment run failed (type=%s)", type(exc).__name__)
        return {"status": getattr(run, "status", None), "reason": getattr(run, "reason_code", None)}


def dispatch_report_judgment(run_id) -> None:
    run_report_judgment.apply_async(args=[str(run_id)], queue=JUDGMENT_QUEUE)
