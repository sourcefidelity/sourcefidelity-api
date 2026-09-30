import pytest


@pytest.fixture(autouse=True)
def _model_calls_off_unless_a_test_enables_them(monkeypatch):
    """LLM processing and GLM are on by default (owner decision 2026-09-29).

    Tests keep them off so no test reaches a live provider; a test that needs
    either sets it explicitly.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "PAPER_LLM_PROCESSING_ENABLED", False)
    monkeypatch.setattr(settings, "ZAI_ENABLED", False)
