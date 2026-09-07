"""Opt-in source mutation freshness against isolated PostgreSQL/MinIO."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit'))
from test_source_mutation_refresh import DETACH_CASES, RENEWAL_CASES, exercise_detach, exercise_renewal
from test_source_generation_live import isolated

pytestmark = pytest.mark.skipif(os.environ.get('RUN_LIVE_SOURCE_MUTATION_REFRESH') != '1',
    reason='requires isolated PostgreSQL/MinIO source-mutation freshness acceptance')


@pytest.mark.parametrize('case', DETACH_CASES)
def test_last_detach_refreshes_locked_content(isolated, monkeypatch, case):
    factory, backend, engine, receipt = isolated
    exercise_detach(factory, backend, monkeypatch, case)
    receipt.update(case='detach_refresh_' + case, regression_passed=True)


@pytest.mark.parametrize('case', RENEWAL_CASES)
def test_renewal_refreshes_locked_representation(isolated, case):
    factory, backend, engine, receipt = isolated
    exercise_renewal(factory, backend, case)
    receipt.update(case='renewal_refresh_' + case, regression_passed=True)
