"""The credential scanner reports names and counts, never values."""
import base64
import io
import json
import sys
from urllib.parse import quote

import pytest

from app import secret_scan
from app.config import Settings

_VALUE = "synthetic-scan-9f3a/+key"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(secret_scan, "settings", Settings(_env_file=None, TAVILY_API_KEY=_VALUE, CORE_API_KEY="short"))
    return secret_scan.configured_secrets()


def _run(monkeypatch, capsys, data: bytes, *argv):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    status = secret_scan.main(list(argv))
    out = capsys.readouterr().out
    assert _VALUE not in out and quote(_VALUE, safe="") not in out
    return status, [json.loads(line) for line in out.splitlines()]


def test_only_non_default_credentials_of_useful_length_are_checked(configured):
    assert configured == {"TAVILY_API_KEY": _VALUE}   # "short" and the MinIO defaults are not credentials


@pytest.mark.parametrize("form", [_VALUE, quote(_VALUE, safe=""), base64.b64encode(_VALUE.encode()).decode()])
def test_raw_url_encoded_and_base64_forms_are_found(configured, form):
    assert secret_scan.scan_bytes(f"prefix {form} suffix".encode(), configured) == {"TAVILY_API_KEY": 1}


def test_stdin_hit_prints_name_and_count_only(configured, monkeypatch, capsys):
    status, (report,) = _run(monkeypatch, capsys, f"log line {_VALUE}\n".encode())
    assert status == 1 and report["found"] == {"TAVILY_API_KEY": 1}


def test_clean_input_exits_zero(configured, monkeypatch, capsys):
    status, (report,) = _run(monkeypatch, capsys, b"nothing here\n")
    assert status == 0 and report["found"] == {}


def test_lines_mode_reports_position_and_metadata_only(configured, monkeypatch, capsys):
    data = (json.dumps({"type": "user", "timestamp": "t1", "message": "ok"}) + "\n"
            + json.dumps({"type": "user", "timestamp": "t2", "message": _VALUE}) + "\n").encode()
    status, (located, report) = _run(monkeypatch, capsys, data, "--lines")
    assert status == 1
    assert located["located"] == [{"line": 2, "names": ["TAVILY_API_KEY"], "timestamp": "t2", "type": "user", "sessionId": None}]


def test_internal_error_reports_class_only(monkeypatch, capsys):
    def boom():
        raise ValueError(_VALUE)
    monkeypatch.setattr(secret_scan, "configured_secrets", boom)
    status, (report,) = _run(monkeypatch, capsys, b"")
    assert status == 2 and report == {"error": "ValueError"}
