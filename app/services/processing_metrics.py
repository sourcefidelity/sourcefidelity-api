"""Text-free stage resource accounting; historical gaps remain explicit."""

from contextlib import contextmanager
from contextvars import ContextVar
import time
import uuid

_current = ContextVar("paper_processing_metrics", default=None)


def record_llm_attempt() -> None:
    current = _current.get()
    if current is not None:
        current["llm_calls"] += 1
        current["missing_usage_calls"] += 1


def record_llm_usage(usage) -> None:
    current = _current.get()
    if current is None or usage is None:
        return
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int) and total >= 0:
        current["total_tokens"] += total
        current["missing_usage_calls"] = max(0, current["missing_usage_calls"] - 1)


def _snapshot(current):
    return {k:v for k,v in current.items() if not k.startswith('_')} | {
        "wall_seconds": round(time.perf_counter()-current['_wall'], 4),
        "cpu_seconds": round(time.process_time()-current['_cpu'], 4),
    }


@contextmanager
def measure_paper_stage(session_factory, job_id, stage, *, workflow_attempt=None):
    from app.models.job import Job
    from app.services.paper_dispatch import attempt_id_for
    from sqlalchemy import select
    current = {"stage":stage,"attempt_id":str(uuid.uuid4()),"llm_calls":0,"total_tokens":0,"missing_usage_calls":0,"_wall":time.perf_counter(),"_cpu":time.process_time()}
    token = _current.set(current)
    try:
        yield
    finally:
        measured = _snapshot(current)
        _current.reset(token)
        with session_factory() as session:
            job = session.scalar(select(Job).where(Job.id == uuid.UUID(str(job_id))).with_for_update())
            if job is not None and (workflow_attempt is None or attempt_id_for(job) == workflow_attempt):
                evidence = dict(job.upload_evidence or {})
                evidence['processing_stages'] = list(evidence.get('processing_stages') or []) + [measured]
                job.upload_evidence = evidence
                session.commit()


def report_processing_metrics(job) -> dict:
    stages = list((job.upload_evidence or {}).get('processing_stages') or [])
    current = _current.get()
    if current is not None:
        stages.append(_snapshot(current))
    if not stages:
        return {"partial":True}
    expected = {'extract','retrieve','verify'}
    missing_usage = sum(s.get('missing_usage_calls',0) for s in stages)
    return {
        "wall_seconds":round(sum(s.get('wall_seconds',0) for s in stages), 3),
        "cpu_seconds":round(sum(s.get('cpu_seconds',0) for s in stages), 3),
        "llm_calls":sum(s.get('llm_calls',0) for s in stages),
        "total_tokens":None if missing_usage else sum(s.get('total_tokens',0) for s in stages),
        "known_tokens":sum(s.get('total_tokens',0) for s in stages),
        "missing_usage_calls":missing_usage,
        "partial":not expected.issubset({s['stage'] for s in stages}) or bool(missing_usage),
        "scope":"Recorded worker stages through report projection; active duration excludes queue waits; CPU excludes external services and child processes; LLM calls are application requests through the shared provider abstraction, including failed attempts.",
    }
