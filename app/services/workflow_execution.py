"""Fail-stop database ownership for one synchronous paper stage execution."""

from contextlib import contextmanager

from sqlalchemy import event


class WorkflowOwnershipLost(RuntimeError):
    code = "workflow_execution_ownership_lost"


class StageExecution:
    def __init__(self, connection, session_factory):
        self.connection = connection
        self.original_factory = session_factory
        self.driver_connection = None if connection is None else connection.connection.dbapi_connection
        self.lost = False

    def check(self, *args, **kwargs):
        connection = self.connection
        if connection is None:
            return
        if (self.lost or connection.closed or connection.invalidated
                or connection.connection.dbapi_connection is not self.driver_connection
                or getattr(self.driver_connection, "closed", False)):
            self.lost = True
            raise WorkflowOwnershipLost("Paper execution no longer owns its database connection")

    def session_factory(self):
        self.check()
        if self.connection is None:
            return self.original_factory()
        return self.original_factory(bind=self.connection, join_transaction_mode="control_fully")


@contextmanager
def stage_execution(connection, session_factory):
    """Use the lock's physical connection for all stage-owned transactions.

    The caller's lock query is committed first; the session advisory lock
    survives. Stage sessions execute sequentially and retain their ordinary
    real commit/rollback boundaries. Reconnection must never revive ownership.
    SQLite retains the existing test behavior, not concurrency guarantees.
    """
    owner = StageExecution(connection, session_factory)
    listeners = ("before_execute", "before_cursor_execute", "commit")
    if connection is not None:
        for name in listeners:
            event.listen(connection, name, owner.check)
    try:
        if connection is not None:
            connection.commit()
        try:
            yield owner
        finally:
            owner.check()
    finally:
        if connection is not None:
            for name in listeners:
                event.remove(connection, name, owner.check)
