"""Shared operational policy and non-secret health state for retrieval providers."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Iterator

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderPolicy:
    enabled: bool = True
    timeout_seconds: float = 15.0
    min_interval_seconds: float = 0.0
    batch_size: int = 1
    max_batches: int = 0  # 0 means no adapter-level ceiling
    retry_delays_seconds: tuple[float, ...] = ()
    cooldown_seconds: int = 900
    max_cooldown_seconds: int = 86400
    max_consecutive_failures: int = 3
    max_timeouts_per_run: int = 0  # 0 disables the cumulative timeout ceiling


_POLICY_FIELDS = set(ProviderPolicy.__dataclass_fields__)


def provider_policy(name: str, defaults: ProviderPolicy) -> ProviderPolicy:
    """Apply validated deployment overrides to an adapter's safe defaults."""
    try:
        configured = json.loads(settings.RETRIEVAL_PROVIDER_CONFIG or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid RETRIEVAL_PROVIDER_CONFIG; using defaults")
        return defaults
    raw = configured.get(name, {}) if isinstance(configured, dict) else {}
    if not isinstance(raw, dict):
        logger.warning("Provider config for %s is not an object; using defaults", name)
        return defaults
    unknown = set(raw) - _POLICY_FIELDS
    if unknown:
        logger.warning("Ignoring unsupported %s provider settings: %s", name, sorted(unknown))
    values = {key: value for key, value in raw.items() if key in _POLICY_FIELDS}
    if "retry_delays_seconds" in values:
        values["retry_delays_seconds"] = tuple(values["retry_delays_seconds"])
    try:
        policy = replace(defaults, **values)
        if policy.timeout_seconds <= 0 or policy.min_interval_seconds < 0:
            raise ValueError("timeouts/intervals must be positive")
        if policy.batch_size < 1 or policy.max_batches < 0:
            raise ValueError("batch_size must be >=1 and max_batches >=0")
        if policy.cooldown_seconds < 0 or policy.max_cooldown_seconds < policy.cooldown_seconds:
            raise ValueError("invalid cooldown range")
        if policy.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >=1")
        if policy.max_timeouts_per_run < 0:
            raise ValueError("max_timeouts_per_run must be >=0")
        return policy
    except (TypeError, ValueError):
        logger.warning("Invalid provider config for %s; using defaults", name)
        return defaults


class ProviderHealthStore:
    """Small persistent store for cross-process provider cooldowns; no secrets."""

    _lock = threading.Lock()

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or settings.PROVIDER_HEALTH_STATE_PATH)

    @property
    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked_data(self) -> Iterator[dict]:
        """Serialize read/modify/write cycles across threads and processes."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield self._read()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
            temporary.replace(self.path)
        except OSError as exc:
            logger.warning(
                "Could not persist provider health state (type=%s)",
                type(exc).__name__,
            )

    def cooldown_remaining(self, provider: str) -> float:
        with self._locked_data() as data:
            state = data.get(provider, {})
        until = state.get("cooldown_until")
        if not until:
            return 0.0
        try:
            parsed = datetime.fromisoformat(until)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return 0.0

    def record_rate_limit(self, provider: str, policy: ProviderPolicy) -> int:
        return self._record_cooldown(provider, policy, status=429)

    def record_timeout(self, provider: str, policy: ProviderPolicy) -> int:
        """Persist a bounded cooldown after a provider timeout circuit opens."""
        return self._record_cooldown(provider, policy, status="timeout")

    def record_unavailable(
        self,
        provider: str,
        policy: ProviderPolicy,
        *,
        status: str,
    ) -> int:
        """Persist a typed provider incident without retaining request data."""
        return self._record_cooldown(provider, policy, status=status)

    def _record_cooldown(
        self,
        provider: str,
        policy: ProviderPolicy,
        *,
        status: int | str,
    ) -> int:
        with self._locked_data() as data:
            previous = data.get(provider, {})
            failures = int(previous.get("consecutive_failures", 0)) + 1
            seconds = min(
                policy.max_cooldown_seconds,
                policy.cooldown_seconds * (4 ** (failures - 1)),
            )
            data[provider] = {
                "consecutive_failures": failures,
                "cooldown_until": (
                    datetime.now(timezone.utc) + timedelta(seconds=seconds)
                ).isoformat(),
                "incident_opened_at": previous.get("incident_opened_at")
                or datetime.now(timezone.utc).isoformat(),
                "last_failure_at": datetime.now(timezone.utc).isoformat(),
                "last_status": status,
            }
            self._write(data)
        return seconds

    def claim_recovery_probe(self, provider: str, *, lease_seconds: int = 60) -> bool:
        """Claim the one half-open probe allowed after a cooldown expires.

        A provider with no open incident does not need a probe claim.  For an
        expired incident, the lease prevents every worker from testing the
        same upstream route simultaneously.
        """
        current = datetime.now(timezone.utc)
        with self._locked_data() as data:
            state = data.get(provider)
            if not isinstance(state, dict):
                return True
            cooldown_until = _parsed_datetime(state.get("cooldown_until"))
            if cooldown_until is not None and cooldown_until > current:
                return False
            probe_until = _parsed_datetime(state.get("probe_lease_until"))
            if probe_until is not None and probe_until > current:
                return False
            state["probe_lease_until"] = (
                current + timedelta(seconds=max(1, lease_seconds))
            ).isoformat()
            data[provider] = state
            self._write(data)
            return True

    def record_success(self, provider: str) -> bool:
        """Close an incident and report whether this call recovered it."""
        with self._locked_data() as data:
            recovered = provider in data
            if recovered:
                data.pop(provider)
                self._write(data)
            return recovered

    def incident(self, provider: str) -> dict | None:
        """Return a non-secret copy of one incident for diagnostics."""
        with self._locked_data() as data:
            state = data.get(provider)
            return dict(state) if isinstance(state, dict) else None

    def incident_providers(self, *, prefix: str | None = None) -> list[str]:
        """List provider keys with open incidents, optionally by adapter prefix."""
        with self._locked_data() as data:
            names = [name for name, state in data.items() if isinstance(state, dict)]
        if prefix is not None:
            names = [name for name in names if name.startswith(prefix)]
        return sorted(names)


class ProviderRequestPacer:
    """Serialize one provider's requests across local workers, without queries.

    Separate files keep pacing timestamps out of incident/recovery state. The
    lock covers the request, so slow requests cannot overlap just because the
    start interval expired. OS locks are released when a worker exits.
    """

    def __init__(self, state_path: str | Path) -> None:
        path = Path(state_path)
        self.directory = path.parent / (path.name + ".requests")

    @contextmanager
    def request(self, key: str, *, min_interval: float, max_wait: float = 15.0):
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("Invalid provider pacing interval")
        if not math.isfinite(max_wait) or max_wait <= 0:
            raise ValueError("Invalid provider pacing wait")
        if min_interval == 0:
            yield 0.0
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / hashlib.sha256(key.encode()).hexdigest()
        started = time.monotonic()
        with path.open("a+") as handle:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - started >= max_wait:
                        raise RuntimeError("Provider pacing wait exhausted")
                    time.sleep(min(0.05, max_wait))
            try:
                handle.seek(0)
                raw = handle.read()
                previous = float(raw) if raw else 0.0
                if not math.isfinite(previous):
                    raise ValueError("Invalid provider pacing timestamp")
                delay = max(0.0, previous + min_interval - time.time())
                if time.monotonic() - started + delay > max_wait:
                    raise RuntimeError("Provider pacing wait exhausted")
                if delay:
                    time.sleep(delay)
                handle.seek(0)
                handle.truncate()
                handle.write(str(time.time()))
                handle.flush()
                yield time.monotonic() - started
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parsed_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def policy_dict(policy: ProviderPolicy) -> dict:
    return asdict(policy)
