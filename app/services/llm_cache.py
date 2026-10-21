"""Remember model answers to exact repeated requests (owner decision 2026-10-09).

A paper check sends hundreds of deterministic (temperature 0) model requests;
a re-run, or another paper citing the same source for the same statement,
sends many of them again word for word. Inside a paper stage the scope is set
(`llm_cache_scope`), and `chat_completion_json` reuses an earlier parsed answer
to the identical request in the same scope instead of calling the model. The
key covers the model, endpoint, both prompts and every generation parameter,
so any change asks again. Answers are kept `SEARCH_RERUN_REUSE_DAYS`. Hits and
misses are counted per calling module in the stage metrics.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from sqlalchemy import delete, select

from app.config import settings

CACHE_VERSION = "llm-call-cache-v1"
logger = logging.getLogger(__name__)
_scope: ContextVar[tuple[str, str] | None] = ContextVar("llm_cache_scope", default=None)


@contextmanager
def llm_cache_scope(scope_type: str, scope_id: str):
    token = _scope.set((scope_type, scope_id))
    try:
        yield
    finally:
        _scope.reset(token)


def _enabled() -> bool:
    return _scope.get() is not None and int(settings.SEARCH_RERUN_REUSE_DAYS or 0) > 0


def caller() -> str:
    """The first module outside the model client that asked."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name not in {__name__, "app.services.llm_service"}:
            return name[:120]
        frame = frame.f_back
    return "unknown"


def request_key(*, model, system_prompt, user_prompt, temperature, max_tokens, response_format,
                disable_thinking, reasoning_effort, endpoint=None) -> str:
    body = {"version": CACHE_VERSION, "model": model or settings.LLM_MODEL,
            "endpoint": endpoint or (urlsplit(settings.LLM_BASE_URL or "").hostname or "").casefold(),
            "system": system_prompt, "user": user_prompt, "temperature": temperature,
            "max_tokens": max_tokens, "response_format": response_format,
            "disable_thinking": disable_thinking, "reasoning_effort": reasoning_effort}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _count(hit: bool, who: str) -> None:
    from app.services.processing_metrics import _count as count
    count("llm_cache_hits" if hit else "llm_cache_misses", who)


def lookup(key: str, who: str):
    """The remembered answer, or None (also when anything goes wrong)."""
    if not _enabled():
        return None
    from app.database import SessionLocal
    from app.models.llm_cache import LLMCallCacheRecord
    scope_type, scope_id = _scope.get()
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(settings.SEARCH_RERUN_REUSE_DAYS))
    try:
        with SessionLocal() as session:
            response = session.scalar(select(LLMCallCacheRecord.response).where(
                LLMCallCacheRecord.scope_type == scope_type, LLMCallCacheRecord.scope_id == scope_id,
                LLMCallCacheRecord.cache_key == key, LLMCallCacheRecord.created_at >= cutoff))
    except Exception as exc:
        logger.warning("Model answer cache unavailable (type=%s)", type(exc).__name__)
        return None
    _count(response is not None, who)
    return copy.deepcopy(response) if response is not None else None


def store(key: str, who: str, model: str | None, response) -> None:
    if not _enabled() or not isinstance(response, (dict, list)):
        return
    from app.database import SessionLocal
    from app.models.llm_cache import LLMCallCacheRecord
    scope_type, scope_id = _scope.get()
    try:
        with SessionLocal() as session:
            session.execute(delete(LLMCallCacheRecord).where(
                LLMCallCacheRecord.scope_type == scope_type, LLMCallCacheRecord.scope_id == scope_id,
                LLMCallCacheRecord.cache_key == key))
            session.add(LLMCallCacheRecord(scope_type=scope_type, scope_id=scope_id, cache_key=key, caller=who,
                                           model=str(model or settings.LLM_MODEL)[:120], response=response))
            session.commit()
    except Exception as exc:
        logger.warning("Model answer not remembered (type=%s)", type(exc).__name__)


def expire(session, *, now: datetime | None = None) -> int:
    from app.models.llm_cache import LLMCallCacheRecord
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=max(0, int(settings.SEARCH_RERUN_REUSE_DAYS or 0)))
    deleted = session.execute(delete(LLMCallCacheRecord).where(LLMCallCacheRecord.created_at < cutoff)).rowcount
    session.commit()
    return deleted or 0
