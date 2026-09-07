"""Opt-in, read-only PostgreSQL lock/query check; never processes existing jobs."""

import os
import uuid

import pytest

from app.database import SessionLocal
from app.services.paper_dispatch import recovery_candidates
from app.services.paper_workflow import PaperWorkflowError
from app.tasks.check_paper import _job_execution_lock


@pytest.mark.skipif(os.environ.get("RUN_LIVE_PAPER_DISPATCH") != "1",
                    reason="requires PostgreSQL for read-only workflow lock/query validation")
def test_postgres_job_lock_survives_commits_and_serializes_different_stages():
    # Random lock IDs have no associated job rows. No writes, publication,
    # worker interruption, source access or legacy-job adoption are performed.
    job_id = str(uuid.uuid4())
    with SessionLocal() as first, SessionLocal() as second:
        assert first.get_bind().dialect.name == "postgresql"
        with _job_execution_lock(first, job_id, "verify"):
            first.commit()
            with pytest.raises(PaperWorkflowError) as blocked:
                with _job_execution_lock(second, job_id, "finalize"):
                    pytest.fail("Different stage bypassed job-wide lock")
            assert blocked.value.code == "job_stage_busy"
            with _job_execution_lock(second, str(uuid.uuid4()), "verify"):
                second.commit()
        with _job_execution_lock(second, job_id, "finalize"):
            second.commit()
        # Verify the portable JSON query on PostgreSQL without dispatching IDs.
        assert len(recovery_candidates(first, limit=1)) <= 1
