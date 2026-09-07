"""Content-free upload completion receipts and bounded cleanup-watch cadence."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_


class UploadReceipt(str):
    """String-compatible key whose acknowledgement covers all upload attempts.

    Single-attempt adapters may return an ordinary key after acknowledgement.
    Retrying adapters must preserve uncertainty, even after a later success.
    Wrappers must propagate this receipt rather than replacing it with a key.
    """
    def __new__(cls, key, *, confirmed):
        value = super().__new__(cls, key)
        value.confirmed = confirmed is True
        return value


def upload_confirmed(result):
    return isinstance(result, str) and getattr(result, "confirmed", True) is True


def utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def next_check(created_at, now):
    age = utc(now) - utc(created_at)
    delay = timedelta(minutes=5) if age < timedelta(days=1) else (
        timedelta(hours=1) if age < timedelta(days=7) else timedelta(days=1))
    return (utc(now) + delay).isoformat()


def check_due(encoded, now):
    if not encoded:
        return True
    try:
        due = datetime.fromisoformat(encoded)
        return utc(due) <= utc(now) or utc(due) > utc(now) + timedelta(days=1)
    except (ValueError, TypeError):
        return True


def due_clause(json_timestamp, now):
    """Filter waiting records before the batch limit; reject far-future stalls.

    These internal ISO timestamps are always UTC. Invalid metadata is not
    deletion authority; an invalid/far-future schedule gets inspected again.
    """
    value = json_timestamp.as_string()
    return or_(value.is_(None), value <= utc(now).isoformat(),
               value > (utc(now) + timedelta(days=1)).isoformat())
