"""Credentials held by Settings must never render in reprs, tracebacks or logs."""

import re

import pytest
from pydantic import SecretStr

from app.config import Settings, secret_value, settings

_CREDENTIAL_NAME = re.compile(r"(_API_KEY|_TOKEN|_SECRET_KEY|_ACCESS_KEY|PASSWORD)$")
_SECRET_FIELDS = [
    name for name, field in Settings.model_fields.items()
    if field.annotation in (SecretStr, SecretStr | None)
]
_URL_FIELDS = ["DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]


def _synthetic_settings() -> tuple[Settings, list[str]]:
    secrets = {name: f"synthetic-{name.lower()}-9f3a" for name in _SECRET_FIELDS}
    urls = {name: f"redis://user:synthetic-{name.lower()}-pw@db:6379/0" for name in _URL_FIELDS}
    return Settings(_env_file=None, **secrets, **urls), [*secrets.values(), *urls.values()]


def test_every_credential_named_field_is_a_secret() -> None:
    credential_fields = [n for n in Settings.model_fields if _CREDENTIAL_NAME.search(n)]
    assert credential_fields
    assert sorted(credential_fields) == sorted(_SECRET_FIELDS)


def test_repr_and_str_contain_no_configured_secret() -> None:
    configured, values = _synthetic_settings()
    rendered = repr(configured) + str(configured)
    for value in values:
        assert value not in rendered
    for name in _SECRET_FIELDS:
        assert secret_value(getattr(configured, name)) in values


def test_failed_monkeypatch_error_does_not_leak_secrets(monkeypatch) -> None:
    configured, values = _synthetic_settings()
    with pytest.raises(AttributeError) as excinfo:
        monkeypatch.setattr(configured, "MISSING", "x")
    rendered = str(excinfo.value) + repr(excinfo.value)
    for value in values:
        assert value not in rendered


def test_live_settings_repr_contains_no_configured_secret() -> None:
    rendered = repr(settings) + str(settings)
    for name in _SECRET_FIELDS:
        value = secret_value(getattr(settings, name))
        default = secret_value(Settings.model_fields[name].default)
        # Public development defaults (e.g. the local MinIO key) are not secrets.
        if value and value != default:
            assert value not in rendered, name
    for name in _URL_FIELDS:
        assert getattr(settings, name) not in rendered, name


def test_secret_value_passes_through_none() -> None:
    assert secret_value(None) is None
    assert secret_value(SecretStr("abc")) == "abc"
