import logging
import traceback
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.log_safety import (
    configure_sensitive_transport_logging,
    private_value_id,
)


def test_private_value_id_is_stable_and_excludes_original_value():
    value = "https://provider.example/search?q=private&api_key=secret"

    identifier = private_value_id("URL", value)

    assert identifier == private_value_id("url", value)
    assert identifier.startswith("url_sha256=")
    assert value not in identifier
    assert "private" not in identifier
    assert "secret" not in identifier


def test_sensitive_transport_loggers_are_warning_or_higher():
    names = ("httpx", "httpcore", "urllib3.connectionpool", "openai._base_client")
    prior = {name: logging.getLogger(name).level for name in names}
    try:
        for name in names:
            logging.getLogger(name).setLevel(logging.INFO)

        configure_sensitive_transport_logging()

        assert all(logging.getLogger(name).level >= logging.WARNING for name in names)
    finally:
        for name, level in prior.items():
            logging.getLogger(name).setLevel(level)


def test_terminal_paper_validation_error_does_not_expose_document_input(monkeypatch):
    from app.tasks import check_paper

    class BoundedInput(BaseModel):
        text: str = Field(max_length=1)

    persisted = []
    monkeypatch.setattr(check_paper, "_fail", lambda job, exc: persisted.append(type(exc).__name__))
    private_text = "PRIVATE_DOCUMENT_SENTINEL"
    try:
        BoundedInput(text=private_text)
    except ValidationError as exc:
        with pytest.raises(RuntimeError) as failure:
            check_paper._retry_or_fail(SimpleNamespace(), "acceptance-job", exc)
    rendered = "".join(traceback.format_exception(failure.value))
    assert private_text not in str(failure.value)
    assert private_text not in rendered
    assert failure.value.__suppress_context__
    assert str(failure.value) == "Paper workflow failed: validation_error"
    assert persisted == ["ValidationError"]
