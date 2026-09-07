"""Database connection ownership survives commits, never reconnection."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.services.workflow_execution import WorkflowOwnershipLost, stage_execution


def test_owned_sessions_commit_and_rollback_independently(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "owned.db"))
    factory = sessionmaker(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE checkpoints (id INTEGER PRIMARY KEY)"))
    with engine.connect() as connection, stage_execution(connection, factory) as owner:
        with owner.session_factory() as session:
            session.execute(text("INSERT INTO checkpoints VALUES (1)"))
            session.commit()
        with factory() as observer:
            assert observer.scalar(text("SELECT count(*) FROM checkpoints")) == 1
        with owner.session_factory() as session:
            session.execute(text("INSERT INTO checkpoints VALUES (2)"))
            session.rollback()
        with owner.session_factory() as session:
            session.execute(text("INSERT INTO checkpoints VALUES (3)"))
            session.commit()
    with factory() as observer:
        assert observer.scalars(text("SELECT id FROM checkpoints ORDER BY id")).all() == [1, 3]
    engine.dispose()


@pytest.mark.parametrize("peer", [False, True])
def test_real_task_workflow_uses_owned_sessions_after_loss(tmp_path, monkeypatch, peer):
    """Exercise actual task/report writes; SQLite does not prove PG locking."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from celery.exceptions import Ignore
    from sqlalchemy import select
    from app.models import Base
    from app.models.job import Job
    from app.models.report import VerificationReportRecord
    from app.services import paper_workflow
    from app.services.paper_dispatch import attempt_id_for
    from app.tasks import check_paper as tasks
    from test_paper_workflow import _retry_test_job

    engine = create_engine("sqlite:///" + str(tmp_path / "workflow.db"))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    _, storage, job_id = _retry_test_job(monkeypatch, factory=factory)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "get_storage_backend", lambda: storage)
    monkeypatch.setattr(paper_workflow.settings, "PAPER_LLM_PROCESSING_ENABLED", False)
    owned = []
    @contextmanager
    def synthetic_lock(*args):
        with engine.connect() as connection:
            owned.append(connection)
            yield connection
    monkeypatch.setattr(tasks, "_job_execution_lock", synthetic_lock)
    with factory() as session:
        attempt = attempt_id_for(session.get(Job, job_id))
    task = SimpleNamespace(request=SimpleNamespace(retries=0),
        retry=lambda **kwargs: pytest.fail("Unexpected retry publication"))
    original = paper_workflow._shadow_artifact
    interrupted = False
    def interleave(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            owned[0].invalidate()
            if peer:
                assert tasks._run_stage(task, str(job_id), attempt, "verify")["reports_persisted"] == 2
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(paper_workflow, "_shadow_artifact", interleave)
        with pytest.raises(Ignore):
            tasks._run_stage(task, str(job_id), attempt, "verify")
    with factory() as session:
        assert session.get(Job, job_id).stage == ("verified" if peer else "verifying")
        assert session.get(Job, job_id).status == "running"
        assert len(session.scalars(select(VerificationReportRecord)).all()) == (2 if peer else 0)
    if not peer:
        assert tasks._run_stage(task, str(job_id), attempt, "verify")["reports_persisted"] == 2
    # Normal finalization and its metrics must still commit on the owned bind.
    result = tasks._run_stage(task, str(job_id), attempt, "finalize")
    assert result["report_id"]
    with factory() as session:
        assert session.get(Job, job_id).status == "completed"
    engine.dispose()


@pytest.mark.parametrize("operation", ["factory", "execute", "driver_sql", "commit"])
def test_connection_loss_cannot_reconnect_and_write(tmp_path, operation):
    engine = create_engine("sqlite:///" + str(tmp_path / "lost.db"))
    factory = sessionmaker(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE checkpoints (id INTEGER PRIMARY KEY)"))
    with engine.connect() as connection:
        with pytest.raises(WorkflowOwnershipLost):
            with stage_execution(connection, factory) as owner:
                with owner.session_factory() as session:
                    session.execute(text("INSERT INTO checkpoints VALUES (1)"))
                    connection.invalidate()
                    session.rollback()  # Rollback must not restore ownership.
                    with pytest.raises(WorkflowOwnershipLost):
                        if operation == "factory":
                            owner.session_factory()
                        elif operation == "execute":
                            connection.execute(text("INSERT INTO checkpoints VALUES (2)"))
                        elif operation == "driver_sql":
                            connection.exec_driver_sql("INSERT INTO checkpoints VALUES (2)")
                        else:
                            session.add_all([])
                            connection.begin()
                            connection.commit()
    with factory() as observer:
        assert observer.scalar(text("SELECT count(*) FROM checkpoints")) == 0
    engine.dispose()
