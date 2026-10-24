"""Unit-test defaults shared by every module in this directory."""
import pytest


@pytest.fixture(autouse=True)
def _process_local_provider_pacing(monkeypatch):
    """Provider pacing tests control their own clock; never touch a real Redis."""
    from app.config import settings
    monkeypatch.setattr(settings, "RETRIEVAL_SHARED_PACING_ENABLED", False)
    # The cross-process request gate is tested on its own (test_provider_gate.py).
    monkeypatch.setattr(settings, "PROVIDER_GATE_ENABLED", False)


@pytest.fixture(autouse=True)
def _fresh_authorized_pages():
    """Each test extracts its own pages (some replace the extractor)."""
    from app.services.authorized_pages import clear_cache
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _strict_discovery_recorder(monkeypatch):
    """A broken discovery-record link fails the test where it is written."""
    monkeypatch.setattr("app.services.discovery_recorder.STRICT", True)


@pytest.fixture
def page_judge():
    """The fixed-answer model at the web page identity seam (web-page-identity-v1).
    Set `page_judge.answer`; `page_judge.calls` lists (reference, page) asked."""
    from app.services.web_page_identity import FixedAnswerJudge, use_judge
    with use_judge(FixedAnswerJudge()) as judge:
        yield judge


@pytest.fixture(autouse=True)
def _code_default_web_fallback(monkeypatch):
    """Tests see the code default, not the local deployment's fallback choice."""
    from app.config import settings
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "none")
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



@pytest.fixture(autouse=True)
def _second_required_provider_is_exa(monkeypatch, request):
    """The suite's search tests were written against Exa as the second required
    provider; DeepSeek is the code default since 2026-10-19 and has its own
    tests (test_deepseek_search.py), which set it explicitly."""
    from app.config import settings as _settings
    monkeypatch.setattr(_settings, "SEARCH_SECOND_REQUIRED_PROVIDER", "exa")
