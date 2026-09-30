from types import SimpleNamespace

from app.services import processing_metrics as metrics


def test_token_accounting_retains_only_numeric_allowlisted_fields():
    state = {"total_tokens": 0, "missing_usage_calls": 1}
    token = metrics._current.set(state)
    try:
        metrics.record_llm_usage(SimpleNamespace(total_tokens=15, prompt_tokens=10,
            completion_tokens=5, prompt_cache_hit_tokens=4, prompt_cache_miss_tokens=6,
            secret="must not persist"))
        assert state["llm_usage_records"] == [{"total_tokens": 15,
            "prompt_tokens": 10, "completion_tokens": 5,
            "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 6}]
        assert state["missing_usage_calls"] == 0
    finally:
        metrics._current.reset(token)


def test_missing_usage_is_not_zero_usage():
    state = {"total_tokens": 0, "missing_usage_calls": 1}
    token = metrics._current.set(state)
    try:
        metrics.record_llm_usage(None)
        assert "llm_usage_records" not in state
        assert state["missing_usage_calls"] == 1
    finally:
        metrics._current.reset(token)
