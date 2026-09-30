"""Text-free stage resource accounting; historical gaps remain explicit."""

from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
import uuid

_current = ContextVar("paper_processing_metrics", default=None)
# Worker threads receive a copy of the context that shares the same stage
# dict; ``+=`` is not atomic, so every mutation holds this lock.
_lock = threading.Lock()
METRICS_VERSION = 2


def record_llm_attempt(*, model: str | None = None, endpoint_host: str | None = None) -> None:
    current = _current.get()
    if current is not None:
        with _lock:
            current["llm_calls"] += 1
            current["missing_usage_calls"] += 1
            if model is not None or endpoint_host is not None:
                current.setdefault("llm_attempt_records", []).append(
                    {"model": str(model or ""), "endpoint_host": str(endpoint_host or "")})


def record_search_usage(provider: str, calls: int, cost_usd=None) -> None:
    current = _current.get()
    if current is not None and calls:
        with _lock:
            current.setdefault('search_usage_records', []).append({
                'execution_provider':provider, 'provider_calls':calls, 'cost_usd':cost_usd,
            })


def _count(bucket: str, key: str) -> None:
    current = _current.get()
    if current is None or not key:
        return
    with _lock:
        counts = current.setdefault(bucket, {})
        counts[key] = counts.get(key, 0) + 1


def record_provider_request(provider: str) -> None:
    """One real HTTP request to an academic adapter; never a cache hit."""
    _count("adapter_requests", str(provider))


def record_direct_fetch(kind: str) -> None:
    """One SSRF-guarded fetch of a page or file outside the adapters."""
    _count("direct_fetches", str(kind))


def record_llm_usage(usage, *, price_context=None, model: str | None = None,
                     endpoint_host: str | None = None) -> None:
    current = _current.get()
    if current is None or usage is None:
        return
    total = getattr(usage, "total_tokens", None)
    # Keep only numeric accounting fields, never provider responses or text.
    fields = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    observed = dict(price_context or {})
    if model is not None:
        observed.setdefault("model", str(model))
    if endpoint_host is not None:
        observed["endpoint_host"] = str(endpoint_host)
    for field in fields:
        value = getattr(usage, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            observed[field] = value
    from app.services.llm_service import cache_split
    for field, value in cache_split(usage).items():
        observed.setdefault(field, value)
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        observed["total_tokens"] = total
    with _lock:
        if isinstance(total, int) and total >= 0:
            current["total_tokens"] += total
            current["missing_usage_calls"] = max(0, current["missing_usage_calls"] - 1)
        current.setdefault("llm_usage_records", []).append(observed)


def _snapshot(current):
    with _lock:
        values = {k:(dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                  for k,v in current.items() if not k.startswith('_')}
    return values | {
        "wall_seconds": round(time.perf_counter()-current['_wall'], 4),
        "cpu_seconds": round(time.process_time()-current['_cpu'], 4),
    }


@contextmanager
def measure_paper_stage(session_factory, job_id, stage, *, workflow_attempt=None):
    from app.models.job import Job
    from app.services.paper_dispatch import attempt_id_for
    from sqlalchemy import select
    current = {"stage":stage,"attempt_id":str(uuid.uuid4()),"metrics_version":METRICS_VERSION,"llm_calls":0,"total_tokens":0,"missing_usage_calls":0,"search_usage_records":[],"adapter_requests":{},"direct_fetches":{},"_wall":time.perf_counter(),"_cpu":time.process_time()}
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
    from app.services.usage_cost import llm_cost, search_trace_cost
    stages = list((job.upload_evidence or {}).get('processing_stages') or [])
    current = _current.get()
    if current is not None:
        stages.append(_snapshot(current))
    if not stages:
        return {"partial":True}
    expected = {'extract','retrieve','verify'}
    missing_usage = sum(s.get('missing_usage_calls',0) for s in stages)
    search_records = [r for s in stages for r in s.get('search_usage_records', [])]
    baseline = (job.upload_evidence or {}).get('historical_search_usage_baseline')
    if baseline:
        search_records = list(baseline['records'])+search_records
    search = search_trace_cost([{'reference_discovery_trace':{'queries':search_records}}]
                               if search_records or any(s['stage']=='retrieve' and 'search_usage_records' in s for s in stages)
                               else getattr(job, 'source_results', None) or [])
    usage = [u for s in stages for u in s.get('llm_usage_records', [])]
    costs = [llm_cost(u) for u in usage]
    unknown_costs = sum(c is None for c in costs) + missing_usage
    unknown_costs += max(0, sum(s.get('llm_calls',0) for s in stages)-len(usage)-missing_usage)
    known_cost = sum(c for c in costs if c is not None)+search['search_cost_usd']
    return {
        **search,
        'estimated_cost_usd': round(known_cost,8),
        'cost_estimate_partial': bool(unknown_costs or search['search_cost_unknown_calls']
                                      or (not baseline and any(s['stage']=='retrieve' and 'search_usage_records' not in s for s in stages))),
        'unpriced_llm_calls': unknown_costs,
        "wall_seconds":round(sum(s.get('wall_seconds',0) for s in stages), 3),
        "cpu_seconds":round(sum(s.get('cpu_seconds',0) for s in stages), 3),
        "llm_calls":sum(s.get('llm_calls',0) for s in stages),
        "total_tokens":None if missing_usage else sum(s.get('total_tokens',0) for s in stages),
        "known_tokens":sum(s.get('total_tokens',0) for s in stages),
        "missing_usage_calls":missing_usage,
        "partial":not expected.issubset({s['stage'] for s in stages}) or bool(missing_usage),
        "scope":"Recorded worker stages through report projection; active duration excludes queue waits; CPU excludes external services and child processes; LLM calls are application requests through the shared provider abstraction, including failed attempts.",
        **_breakdowns(stages, search_records, baseline),
    }


def _breakdowns(stages: list[dict], search_records: list[dict], baseline) -> dict:
    """Per-adapter, per-fetch-kind, per-search-provider and per-model detail.

    Recorded from ``processing-metrics-v2`` onward. When any stage predates it
    the breakdown is marked incomplete rather than presented as a whole.
    """
    from app.services.usage_cost import llm_cost, search_cost_by_provider
    recorded = [s for s in stages if s.get('metrics_version') == METRICS_VERSION]
    result = {
        'metrics_version': 'processing-metrics-v2' if recorded else None,
        'breakdown_complete': bool(recorded) and len(recorded) == len(stages),
    }
    if not recorded:
        return result
    adapters: dict[str, int] = {}
    fetches: dict[str, int] = {}
    for stage in recorded:
        for name, count in (stage.get('adapter_requests') or {}).items():
            adapters[name] = adapters.get(name, 0) + int(count)
        for name, count in (stage.get('direct_fetches') or {}).items():
            fetches[name] = fetches.get(name, 0) + int(count)
    models: dict[tuple, dict] = {}
    for stage in recorded:
        for attempt in stage.get('llm_attempt_records') or []:
            key = (attempt.get('model') or '', attempt.get('endpoint_host') or '')
            models.setdefault(key, {'calls': 0, 'usage': []})['calls'] += 1
        for usage in stage.get('llm_usage_records') or []:
            key = (usage.get('model') or '', usage.get('endpoint_host') or '')
            models.setdefault(key, {'calls': 0, 'usage': []})['usage'].append(usage)
    llm_rows = []
    for (model, host), row in sorted(models.items()):
        usage = row['usage']
        costs = [llm_cost(u) for u in usage]
        missing = max(0, row['calls'] - len(usage))
        tokens = {k: sum(int(u.get(k) or 0) for u in usage)
                  for k in ('prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'prompt_tokens',
                            'completion_tokens', 'total_tokens')}
        known = [c for c in costs if c is not None]
        llm_rows.append({
            'model': model or None, 'endpoint_host': host or None, 'calls': row['calls'],
            'missing_usage_calls': missing, **tokens,
            'cost_usd': round(sum(known), 8) if known else None,
            'cost_complete': bool(usage) and len(known) == len(usage) and not missing,
        })
    return {
        **result,
        'adapter_requests': [{'provider': k, 'requests': v} for k, v in sorted(adapters.items())],
        'direct_fetches': [{'kind': k, 'requests': v} for k, v in sorted(fetches.items())],
        'search_by_provider': search_cost_by_provider(search_records),
        'llm_by_model': llm_rows,
    }
