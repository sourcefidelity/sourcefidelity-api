"""Private-worker bootstrap for the opt-in isolated scheduling test only."""

import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def configure(prefix):
    if not re.fullmatch(r"sf_dispatch_test_[0-9a-f]{32}", prefix):
        raise ValueError("Invalid isolated test namespace")
    from app.tasks.celery_app import celery_app
    from app.tasks import check_paper
    from celery.signals import worker_ready
    from redis import Redis

    celery_app.conf.update(
        task_default_queue=prefix,
        task_default_exchange=prefix,
        task_default_routing_key=prefix,
        task_routes={"*": {"queue": prefix}},
        broker_transport_options={"global_keyprefix": prefix + ":"},
        result_backend_transport_options={"global_keyprefix": prefix + ":"},
        worker_send_task_events=False,
        task_send_sent_event=False,
        worker_enable_remote_control=False,
    )

    class NoSourceAccess:
        def __getattr__(self, name):
            raise AssertionError("Scheduling test must not access stored source bytes")

    check_paper.get_storage_backend = NoSourceAccess

    @celery_app.task(name="sf_dispatch_test_barrier")
    def barrier():
        return True

    @worker_ready.connect(weak=False)
    def ready(**_kwargs):
        Redis.from_url(celery_app.conf.broker_url).set(prefix + ":ready", "1", ex=600)

    return celery_app


if __name__ == "__main__":
    prefix = os.environ["SOURCEFIDELITY_TEST_NAMESPACE"]
    application = configure(prefix)
    if sys.argv[1] == "worker":
        application.worker_main([
            "worker", "--pool=solo", "--concurrency=1", "--loglevel=WARNING",
            "--without-gossip", "--without-mingle", "--without-heartbeat",
            "--hostname=" + prefix, "--queues=" + prefix,
        ])
    else:
        from app.database import SessionLocal
        from app.services.paper_dispatch import claim_dispatch
        from app.tasks.check_paper import _job_execution_lock, _start_stage
        from sqlalchemy import text
        job_id, attempt_id = sys.argv[2:4]
        with SessionLocal() as session:
            assert session.scalar(text("SELECT current_schema()")) == prefix
        if sys.argv[1] == "publication-exit":
            assert claim_dispatch(SessionLocal, job_id, expected_attempt=attempt_id)
            os._exit(73)  # Only this newly created test process exits.
        elif sys.argv[1] == "stage-exit":
            with SessionLocal() as session:
                with _job_execution_lock(session, job_id, "finalize"):
                    assert _start_stage(session, job_id, attempt_id, "finalize")
                    os._exit(73)  # Tests release of this process's dedicated lock.
        else:
            raise ValueError("Unknown isolated test operation")
