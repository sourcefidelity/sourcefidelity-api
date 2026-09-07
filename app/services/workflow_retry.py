"""Shared operational-error policy for workflow retries and source retention."""

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import RedisError
from sqlalchemy.exc import OperationalError as DatabaseOperationalError


def is_retryable(exc: BaseException) -> bool:
    retryable_types = (
        ConnectionError, TimeoutError, httpx.RequestError, BotoCoreError,
        ClientError, KombuOperationalError, RedisError, DatabaseOperationalError,
    )
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, retryable_types):
            return True
        if str(getattr(current, "code", "")) in {
            "storage_unavailable", "source_safety_unavailable",
            "transient_verification_run_unavailable", "workflow_enqueue_failed",
            "job_stage_busy",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False
