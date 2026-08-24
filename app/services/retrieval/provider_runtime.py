"""Shared operational policy and non-secret health state for retrieval providers."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import threading

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
    except json.JSONDecodeError as exc:
        logger.warning("Invalid RETRIEVAL_PROVIDER_CONFIG; using defaults: %s", exc)
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
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid provider config for %s; using defaults: %s", name, exc)
        return defaults


class ProviderHealthStore:
    """Small persistent store for cross-process provider cooldowns; no secrets."""

    _lock = threading.Lock()

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or settings.PROVIDER_HEALTH_STATE_PATH)

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
            logger.warning("Could not persist provider health state: %s", exc)

    def cooldown_remaining(self, provider: str) -> float:
        with self._lock:
            state = self._read().get(provider, {})
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

    def _record_cooldown(
        self,
        provider: str,
        policy: ProviderPolicy,
        *,
        status: int | str,
    ) -> int:
        with self._lock:
            data = self._read()
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
                "last_status": status,
            }
            self._write(data)
        return seconds

    def record_success(self, provider: str) -> None:
        with self._lock:
            data = self._read()
            if provider in data:
                data.pop(provider)
                self._write(data)


def policy_dict(policy: ProviderPolicy) -> dict:
    return asdict(policy)
