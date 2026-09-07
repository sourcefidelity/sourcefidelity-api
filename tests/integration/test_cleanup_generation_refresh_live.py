"""Opt-in cached-generation cleanup against isolated PostgreSQL/MinIO."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit'))
from test_cleanup_generation_refresh import CASES, exercise_generation_cleanup
from test_source_generation_live import isolated

pytestmark = pytest.mark.skipif(os.environ.get('RUN_LIVE_CLEANUP_REFRESH') != '1',
    reason='requires isolated PostgreSQL/MinIO cached-generation cleanup acceptance')


@pytest.mark.parametrize('case', CASES)
def test_cleanup_refreshes_locked_generation(isolated, monkeypatch, case):
    factory, backend, engine, receipt = isolated
    exercise_generation_cleanup(factory, backend, monkeypatch, case)
    receipt.update(case='generation_refresh_' + case, regression_passed=True)
