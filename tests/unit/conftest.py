"""Unit-test defaults shared by every module in this directory."""
import pytest


@pytest.fixture(autouse=True)
def _process_local_provider_pacing(monkeypatch):
    """Provider pacing tests control their own clock; never touch a real Redis."""
    from app.config import settings
    monkeypatch.setattr(settings, "RETRIEVAL_SHARED_PACING_ENABLED", False)


@pytest.fixture(autouse=True)
def _code_default_web_fallback(monkeypatch):
    """Tests see the code default, not the local deployment's fallback choice."""
    from app.config import settings
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "searxng")
    monkeypatch.setattr(settings, "TAVILY_USD_PER_CREDIT", None)


@pytest.fixture(autouse=True)
def _judgment_off_unless_a_test_enables_it(monkeypatch):
    """Tests keep Judgment off and enable it explicitly."""
    from app.config import settings
    monkeypatch.setattr(settings, "JUDGMENT_FAKE_PANEL", False)
    monkeypatch.setattr(settings, "SOURCE_TEXT_QUALITY_CHECK_ENABLED", False)
    # The GLM evidence selector and judge must never reach the live provider from tests.
    monkeypatch.setattr(settings, "ZAI_ENABLED", False)
    monkeypatch.setattr(settings, "JUDGMENT_JUDGES", "zai_glm")
