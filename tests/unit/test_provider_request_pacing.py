import json
import multiprocessing
import time

import pytest

from app.services.retrieval.provider_runtime import ProviderHealthStore, ProviderRequestPacer


def _request_in_child(path, queue):
    with ProviderRequestPacer(path).request("engine", min_interval=0.08):
        queue.put(time.time())


def test_pacing_serializes_independent_processes_without_health_changes(tmp_path):
    path = tmp_path / "health.json"
    path.write_text(json.dumps({"other": {"last_status": "captcha"}}))
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with ProviderRequestPacer(path).request("engine", min_interval=0.08):
        processes = [context.Process(target=_request_in_child, args=(str(path), queue)) for _ in range(2)]
        for process in processes:
            process.start()
        released_at = time.time()
    times = sorted(queue.get(timeout=10) for _ in processes)
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert times[0] >= released_at
    assert times[1] - times[0] >= 0.07
    assert ProviderHealthStore(str(path)).incident_providers() == ["other"]
    assert "engine" not in next((tmp_path / "health.json.requests").iterdir()).name


def test_busy_gate_is_bounded_and_different_engines_are_independent(tmp_path):
    first = ProviderRequestPacer(tmp_path / "state.json")
    second = ProviderRequestPacer(tmp_path / "state.json")
    with first.request("one", min_interval=0.1):
        with second.request("two", min_interval=0.1):
            pass
        with pytest.raises(RuntimeError, match="wait exhausted"):
            with second.request("one", min_interval=0.1, max_wait=0.02):
                pytest.fail("Busy request must not run")


def test_exception_releases_request_lock(tmp_path):
    gate = ProviderRequestPacer(tmp_path / "state.json")
    with pytest.raises(RuntimeError, match="upstream"):
        with gate.request("engine", min_interval=0.001):
            raise RuntimeError("upstream")
    with gate.request("engine", min_interval=0.001):
        pass


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_pacing_fails_closed(tmp_path, interval):
    with pytest.raises(ValueError):
        with ProviderRequestPacer(tmp_path / "state.json").request("engine", min_interval=interval):
            pytest.fail("Invalid pacing must not run")
