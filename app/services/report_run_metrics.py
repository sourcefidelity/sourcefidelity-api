"""Technical details for one run of a report (owner request 2026-09-28).

A report's stored metrics total every stage of every version. The owner asked
for what it costs to run the report a single time: the first complete check
(every stage up to its first finalize) and the report's Judgment run, priced
as if nothing were cached. Later targeted refreshes are left out. Unknown
usage stays unknown, never zero.
"""
from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import urlsplit

from app.services.judgment_report import JUDGED
from app.services.processing_metrics import report_processing_metrics
from app.services.usage_cost import llm_cost, llm_price_context


def first_run_stages(stages: list[dict]) -> list[dict]:
    end = next((i for i, stage in enumerate(stages) if stage.get("stage") == "finalize"), None)
    return list(stages) if end is None else list(stages[:end + 1])


def _judge_base_url(arm_id: str, settings) -> str:
    # Repeat samples are "zai_glm:s2", "zai_glm:s3": the same arm and endpoint
    # (they were listed as a second GLM row with no endpoint; 2026-10-02).
    base = str(arm_id or "").split(":", 1)[0]
    return {"zai_glm": settings.ZAI_BASE_URL, "deepseek": settings.LLM_BASE_URL}.get(base) or ""


def judgment_usage(results: list, arms: list, settings) -> list[dict]:
    """Per-model rows for the Judgment run: judge calls and the notes."""
    rows: dict[tuple, dict] = {}

    def row(model, host):
        return rows.setdefault((model, host), {"calls": 0, "missing": 0, "prompt": 0, "completion": 0,
                                               "cost": 0.0, "complete": True})

    for arm in arms:
        base_url = _judge_base_url(arm.arm_id, settings)
        entry = row(arm.model, urlsplit(base_url).hostname or None)
        usage = arm.usage or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        attempts = int(usage.get("attempts") or 1)
        entry["calls"] += attempts
        if type(prompt) is not int or type(completion) is not int:
            entry["missing"] += attempts
            entry["complete"] = False
            continue
        entry["prompt"] += prompt
        entry["completion"] += completion
        rates = llm_price_context(arm.model, base_url).get("usd_per_million")
        cost = llm_cost({"usd_per_million": rates, "prompt_tokens": prompt, "prompt_cache_hit_tokens": 0,
                         "prompt_cache_miss_tokens": prompt, "completion_tokens": completion}) if rates else None
        if cost is None and arm.cost_basis in {"tariff", "provider_reported"} and arm.cost_usd is not None:
            cost = arm.cost_usd
        if cost is None:
            entry["complete"] = False
        else:
            entry["cost"] += cost
    for result in results:
        coaching = result.coaching or {}
        if not coaching.get("attempts"):
            continue
        entry = row(settings.LLM_MODEL, urlsplit(settings.LLM_BASE_URL or "").hostname or None)
        entry["calls"] += int(coaching["attempts"])
        entry["missing"] += int(coaching["attempts"])     # note token counts are not stored
        cost = coaching.get("original_cost_usd") if coaching.get("cached") else coaching.get("cost_usd")
        if coaching.get("cost_basis") in {"tariff", "provider_reported"} and isinstance(cost, (int, float)):
            entry["cost"] += cost
        else:
            entry["complete"] = False
    return [{"model": model, "endpoint_host": host, "calls": r["calls"], "missing_usage_calls": r["missing"],
             "prompt_tokens": r["prompt"], "prompt_cache_hit_tokens": 0, "completion_tokens": r["completion"],
             "cost_usd": round(r["cost"], 8), "cost_complete": r["complete"]}
            for (model, host), r in rows.items()]


def single_run_metrics(job, *, results: list = (), arms: list = (), settings=None) -> dict:
    evidence = dict(job.upload_evidence or {})
    evidence["processing_stages"] = first_run_stages(list(evidence.get("processing_stages") or []))
    metrics = report_processing_metrics(SimpleNamespace(upload_evidence=evidence,
                                                        source_results=getattr(job, "source_results", None)))
    if settings is None or metrics.get("metrics_version") != "processing-metrics-v2":
        return metrics
    by_key = {(r.get("model"), r.get("endpoint_host")): dict(r) for r in metrics.get("llm_by_model") or []}
    extra = 0.0
    for row in judgment_usage(list(results), list(arms), settings):
        key = (row["model"], row["endpoint_host"])
        current = by_key.get(key)
        if current is None:
            by_key[key] = row
        else:
            for field in ("calls", "missing_usage_calls", "prompt_tokens", "prompt_cache_hit_tokens",
                          "completion_tokens"):
                current[field] = int(current.get(field) or 0) + row[field]
            current["cost_usd"] = round((current.get("cost_usd") or 0.0) + row["cost_usd"], 8)
            current["cost_complete"] = bool(current.get("cost_complete")) and row["cost_complete"]
        extra += row["cost_usd"]
    metrics["llm_by_model"] = [by_key[k] for k in sorted(by_key, key=lambda k: (k[0] or "", k[1] or ""))]
    if isinstance(metrics.get("estimated_cost_usd"), (int, float)):
        metrics["estimated_cost_usd"] = round(metrics["estimated_cost_usd"] + extra, 8)
    return metrics


def judgment_states(results: list) -> dict:
    """Judged statements (proposition and source) by result."""
    from app.services.judgment_report import display_state
    counts: dict = {}
    for r in results:
        state = display_state(r.display_state, r.reason_code)
        if state in JUDGED or state == "undecided":
            counts[state] = counts.get(state, 0) + 1
    return counts


def judged_citations(results: list) -> list[int]:
    """Citations with at least one judged proposition."""
    return sorted({r.citation_index for r in results
                   if r.citation_index is not None and r.display_state in JUDGED})
