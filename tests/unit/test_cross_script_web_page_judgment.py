"""Model comparison of fetched web-search pages titled in another writing system (2026-10-18, default off)."""
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import source_resolver
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate
from app.services.retrieval.base import RetrievalResult
from tests.unit.discovery_support import recorder_from

CITED = "Frozen archives and the politics of memory"
HAN_TITLE = "冻结的档案与记忆政治"


@pytest.fixture
def judge(monkeypatch, page_judge):
    page_judge.answer = {"version": "cross-script-identity-v1", "same_work": "no", "author": "not_shown",
                         "page_authors": ["Name On Page"], "payload_sha256": "a" * 64}
    monkeypatch.setattr("trafilatura.extract", lambda *a, **k: "page text")
    return page_judge.calls


def _judgment(**kwargs):
    response = SimpleNamespace(text="<html></html>")
    return source_resolver._cross_script_page_judgment(
        response, kwargs.pop("titles", [HAN_TITLE]), expected_title=CITED,
        expected_author="Writer, A.", expected_year="2020", **kwargs)


def test_on_by_default_and_off_when_switched_off(judge, monkeypatch):
    from app.config import Settings
    assert Settings.model_fields["CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED"].default is True
    monkeypatch.setattr(settings, "CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED", False)
    assert _judgment() is None and judge == []


def test_on_judges_a_page_titled_only_in_another_script(judge, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED", True)
    judged = _judgment()
    assert judged["same_work"] == "no" and judged["version"] == "cross-script-identity-v1"
    assert "page_authors" not in judged          # page names are not kept
    assert judge[0][1]["titles"] == [HAN_TITLE] and judge[0][1]["text"] == "page text"


def test_a_page_with_a_title_in_the_cited_script_is_not_sent(judge, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED", True)
    assert _judgment(titles=[HAN_TITLE, "Frozen archives"]) is None and judge == []


def test_bounded_per_reference(judge, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_SCRIPT_WEB_PAGE_JUDGMENT_ENABLED", True)
    monkeypatch.setattr(settings, "CROSS_SCRIPT_PAGE_JUDGMENTS_PER_REFERENCE", 2)
    token = source_resolver._ACTIVE_DISCOVERY_TRACE.set(recorder_from({}))
    try:
        results = [_judgment(titles=[HAN_TITLE + "一二三"[i]]) for i in range(3)]
        again = _judgment(titles=[HAN_TITLE + "一"])
    finally:
        source_resolver._ACTIVE_DISCOVERY_TRACE.reset(token)
    assert [r is not None for r in results] == [True, True, False]
    # A page already asked about gets its first answer, with no new call.
    assert again == results[0] and len(judge) == 2


@pytest.mark.parametrize("same_work,outcome,reason", [
    (None, "unknown", "cross_script_title_unresolved"),
    ("no", "material_conflict", "cross_script_model_different_work"),
    ("unsure", "unknown", "cross_script_model_unsure"),
    ("yes", "agreement", "cross_script_model_same_work"),
])
def test_the_judgment_decides_the_web_candidates_title_comparison(same_work, outcome, reason):
    metadata = {"cross_script_judgment": {"same_work": same_work}} if same_work else {}
    candidate = build_reference_discovery_candidate(
        attempt_id="a1", provider="deepseek",
        expected=ExpectedBibliographicFields(title=CITED, authors=["Writer, A."], year="2020"),
        result=RetrievalResult(source_name="deepseek", success=True, title=HAN_TITLE, metadata=metadata),
    )
    title = next(c for c in candidate.comparisons if c.field_name == "title")
    assert (title.outcome, title.reason_code) == (outcome, reason)
