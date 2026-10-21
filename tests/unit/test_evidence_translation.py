"""English translations of other-language evidence sentences (evidence-translation-v1).

Owner decision 2026-10-13: a source sentence shown as evidence in another
language than the student's paper is followed by its English translation in
square brackets, with no other added wording. Invented fixtures only.
"""
import json
from pathlib import Path

import pytest

from app.services import evidence_translation as et
from test_judgment_runs import store  # noqa: F401 - the in-memory report fixture
from app.services.evidence_translation import (
    BATCH_SIZE,
    MAX_SENTENCES_PER_MEMBER,
    apply_translations,
    judgment_translations,
    sentence_key,
    sentence_units,
    stored_translations,
    translate_report_view,
    translate_sentences,
    translated_text_html,
)

HAN_1 = "河流治理需要长期的社区参与。"
HAN_2 = "该研究调查了三个城市的公共图书馆。"
ENGLISH = {HAN_1: "River management requires long-term community participation.",
           HAN_2: "The study surveyed public libraries in three cities."}
SOURCE = {"author": "Lin, Q.", "year": "2019", "title": "Heliu zhili yanjiu", "raw_reference": "Lin, Q. (2019). Heliu zhili yanjiu."}
STATEMENT = "River care depends on local residents (Lin, 2019)."


def fake_completion(calls, answers=None):
    """A DeepSeek stand-in: translates each supplied sentence from a fixed table."""
    def complete(system, payload, **kwargs):
        body = json.loads(payload)
        calls.append({"system": system, "payload": body, "kwargs": kwargs})
        kwargs["receipt"].update(returned_model="deepseek-test", total_tokens=120, reported_cost_usd=0.0004)
        table = answers if answers is not None else ENGLISH
        return {"translations": [{"id": s["id"], "translation": table.get(s["text"], f"English {s['id']}")}
                                 for s in body["sentences"]]}
    return complete


def member(sentences, **extra):
    return {"reference_id": "r1", "source": SOURCE, "coverage_level": "full_text",
            "verification_report_id": "00000000-0000-0000-0000-000000000001",
            "evidence_sentences": [{"key": f"0:{i}:{i + 1}", "text": text, "page": 3, "reason": "bears_on_statement"}
                                   for i, text in enumerate(sentences)], **extra}


def view_of(*members, statement=STATEMENT):
    return {"citations": [{"student_text": statement, "members": list(members)}]}


def test_selected_other_language_sentences_are_translated_stored_and_rendered():
    from app.services.evidence_report import _render_member
    calls = []
    han = member([HAN_1, "Table 2 (2019)."])
    same = {**member(["Rivers need local care."]), "reference_id": "r2"}
    store, view = translate_report_view(view_of(han, same), completion=fake_completion(calls))
    # One call, carrying only the other-language sentence.
    assert len(calls) == 1 and [s["text"] for s in calls[0]["payload"]["sentences"]] == [HAN_1]
    assert set(calls[0]["payload"]) == {"target_language", "sentences"}
    assert store["sentences"] == {sentence_key(HAN_1): ENGLISH[HAN_1]}
    provenance = store["calls"][0]
    assert provenance["call"] == "answered" and len(provenance["payload_sha256"]) == 64
    assert provenance["returned_model"] == "deepseek-test" and provenance["total_tokens"] == 120
    assert store["cost_usd"] == pytest.approx(0.0004)
    shown = view["citations"][0]["members"]
    html = _render_member(shown[0])
    assert (f'<q>{HAN_1}</q> <span class="ev-translation">[{ENGLISH[HAN_1]}]</span></li>') in html
    assert "<q>Table 2 (2019).</q></li>" in html
    # A source in the paper's own language is unchanged.
    assert "ev-translation" not in _render_member(shown[1]) and "translation" not in shown[1]["evidence_sentences"][0]


def test_a_failed_or_invalid_translation_shows_the_original_alone():
    from app.services.evidence_report import _render_member

    def down(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    store, view = translate_report_view(view_of(member([HAN_1])), completion=down)
    assert store["sentences"] == {} and store["calls"][0]["call"] == "failed"
    html = _render_member(view["citations"][0]["members"][0])
    assert f"<q>{HAN_1}</q></li>" in html and "ev-translation" not in html and "[" not in html.split(HAN_1)[1][:20]
    # A translation not written in English letters is rejected; so is a malformed answer.
    calls = []
    store, _ = translate_report_view(view_of(member([HAN_1])), completion=fake_completion(calls, {HAN_1: "河流"}))
    assert store["sentences"] == {} and store["calls"][0]["call"] == "partial"
    store, _ = translate_report_view(view_of(member([HAN_1])), completion=lambda *a, **k: {"text": "x"})
    assert store["sentences"] == {} and store["calls"][0]["call"] == "invalid_answer"


def test_translation_never_fails_the_report():
    broken = {"citations": [{"student_text": STATEMENT, "members": [{"evidence_sentences": [{"text": None}]}]}]}
    store, view = translate_report_view(broken, completion=lambda *a, **k: 1 / 0)
    assert view is broken


def test_batches_are_bounded_and_known_sentences_are_not_sent_again():
    texts = [f"第{n}号河流的水质在雨季明显下降。" for n in range(MAX_SENTENCES_PER_MEMBER + 5)]
    calls = []
    translated, made = translate_sentences(texts, completion=fake_completion(calls))
    assert len(translated) == MAX_SENTENCES_PER_MEMBER
    assert [len(c["payload"]["sentences"]) for c in calls] == [BATCH_SIZE, MAX_SENTENCES_PER_MEMBER - BATCH_SIZE]
    assert all(c["kwargs"]["temperature"] == 0.0 for c in calls)
    again = []
    assert translate_sentences(texts[:MAX_SENTENCES_PER_MEMBER], known=translated,
                               completion=fake_completion(again)) == ({}, [])
    assert again == []
    # A whole member is bounded the same way.
    calls = []
    store, _ = translate_report_view(view_of(member(texts)), completion=fake_completion(calls))
    assert len(store["sentences"]) == MAX_SENTENCES_PER_MEMBER


def test_stored_translations_are_reused_and_reapplied_by_a_refresh():
    calls = []
    store, _ = translate_report_view(view_of(member([HAN_1])), completion=fake_completion(calls))
    again = []
    known = stored_translations({"evidence_translations": store})
    store_2, view = translate_report_view(view_of(member([HAN_1])), known=known, completion=fake_completion(again))
    assert again == [] and store_2["sentences"] == store["sentences"]
    # The projection (build_evidence_report_view) applies the kept store without any call.
    fresh = apply_translations(view_of(member([HAN_1])), known)
    assert fresh["citations"][0]["members"][0]["evidence_sentences"][0]["translation"] == ENGLISH[HAN_1]
    # With the model off nothing is sent, and kept translations still apply.
    store_3, view = translate_report_view(view_of(member([HAN_1])), known=known, enabled=False,
                                          completion=lambda *a, **k: pytest.fail("no call"))
    assert view["citations"][0]["members"][0]["evidence_sentences"][0]["translation"] == ENGLISH[HAN_1]


def test_an_abstract_shows_each_translated_sentence_followed_by_its_translation():
    from app.services.evidence_report import _abstract_disclosure
    abstract = HAN_1 + HAN_2
    assert "".join(sentence_units(abstract)) == abstract and len(sentence_units(abstract)) == 2
    item = {"text": abstract, "display_text": abstract, "evidence_kind": "abstract"}
    calls = []
    abstract_member = {"reference_id": "r1", "source": SOURCE, "coverage_level": "abstract_only",
                       "best_evidence": item, "evidence_sentences": []}
    _, view = translate_report_view(view_of(abstract_member), completion=fake_completion(calls))
    assert [s["text"] for s in calls[0]["payload"]["sentences"]] == [HAN_1, HAN_2]
    html = _abstract_disclosure(view["citations"][0]["members"][0])
    assert (f'<blockquote>{HAN_1} <span class="ev-translation">[{ENGLISH[HAN_1]}]</span> '
            f'{HAN_2} <span class="ev-translation">[{ENGLISH[HAN_2]}]</span></blockquote>') in html
    # Without translations the abstract is unchanged.
    assert translated_text_html(abstract, {}) == abstract


def test_the_judgment_result_delivers_translations_for_the_page_script():
    from app.services.judgment_report import judgment_result
    payload = {
        "claim": {"text": STATEMENT},
        "verification_candidates": {"candidates": [{"candidate_id": "c1", "text": STATEMENT, "segments": []}]},
        "facet_evidence_foundation": {
            "source_sentences": [{"sentence_id": "s1", "text": HAN_1, "passage_id": "p1",
                                  "passage_start": 0, "passage_end": 14}],
            "candidate_bundles": [{"candidate_id": "c1", "evidence_sentence_ids": ["s1"], "facets": [
                {"facet_id": "f1", "kind": "candidate_as_written", "text": STATEMENT, "material_to_aggregate": True}]}]},
        "passages": [{"passage_id": "p1", "page_index": 0, "page_label": "5", "character_start": 100}],
    }
    arm = {"arm_id": "zai_glm", "status": "valid", "label": "supports", "mappings": [
        {"facet_id": "f1", "direction": "supports", "evidence_sentence_ids": ["s1"], "rationale": "Agrees."}]}
    row = {"display_state": "supported", "reason_code": "judge_supports", "candidate_id": "c1",
           "panel": {"arms": [arm]}}
    result = judgment_result(row, payload, None, translations={sentence_key(HAN_1): ENGLISH[HAN_1]})
    assert result["evidence"][0]["translation"] == ENGLISH[HAN_1]
    assert f'<q>{HAN_1}</q> <span class="ev-translation">[{ENGLISH[HAN_1]}]</span>' in result["html"]
    assert "&quot;translation&quot;" in result["html"]     # the key sentence's alternatives carry it too
    plain = judgment_result(row, payload, None)
    assert "translation" not in plain["evidence"][0] and "ev-translation" not in plain["html"]


def test_judge_cited_sentences_in_another_language_are_translated_once():
    panel = {"arms": [{"status": "valid", "in_majority": True, "mappings": [
        {"evidence_sentence_ids": ["s1", "s2"]}]}, {"status": "failed", "mappings": [
        {"evidence_sentence_ids": ["s3"]}]}]}
    sentences = {"s1": HAN_1, "s2": "Rivers flood in spring.", "s3": HAN_2}
    calls = []
    translated, made = judgment_translations(panel, sentences, STATEMENT, known={},
                                             completion=fake_completion(calls))
    assert [s["text"] for s in calls[0]["payload"]["sentences"]] == [HAN_1]
    assert translated == {sentence_key(HAN_1): ENGLISH[HAN_1]} and made[0]["call"] == "answered"
    again = []
    assert judgment_translations(panel, sentences, STATEMENT, known=translated,
                                 completion=fake_completion(again)) == (translated, [])
    assert again == []
    assert judgment_translations(panel, sentences, STATEMENT, known={}, enabled=False,
                                 completion=fake_completion(again)) == ({}, [])


def test_the_judgment_worker_keeps_translations_with_the_result_and_results_deliver_them(store, monkeypatch):
    import test_judgment_runs as jr
    from test_judgment_panel import _fake_call
    from app.services.judgment_results import result_items
    session, report, _ = store
    seen = []

    def fake(panel, sentences, statement, *, known, completion=None, enabled=True):
        texts = et.judged_sentence_texts(panel, sentences)
        seen.append(texts)
        return ({sentence_key(t): f"EN {i}" for i, t in enumerate(texts)},
                [{"call": "answered", "cost_usd": 0.001, "sentence_count": len(texts)}])
    monkeypatch.setattr(et, "judgment_translations", fake)
    run = jr._run(session, report, call=_fake_call(jr.SUPPORTS))
    judged = [r for r in jr._rows(session, run) if r.citation_index == 1]
    assert judged and all(r.panel.get("evidence_translations") for r in judged)
    assert all(r.panel["evidence_translation"]["version"] == et.VERSION for r in judged)
    assert run.spend_usd == pytest.approx((0.003 + 0.001) * len(judged))
    items = [i for i in result_items(session, run, jr.PRINCIPAL) if i["citation_number"] == 1]
    assert items and all(s.get("translation", "").startswith("EN ") for i in items for s in i["evidence"])


def test_the_page_script_appends_translations_delivered_with_results():
    script = (Path(__file__).resolve().parents[2] / "app/services/report_judgment.js").read_text()
    assert "sentence.translation" in script and "a.translation" in script
    assert "span.className = 'ev-translation'" in script and "`[${value}]`" in script
    assert "showTranslations();" in script and "noteTranslations();" in script


def test_the_backfill_counts_in_a_dry_run_and_otherwise_refreshes_with_the_translations(store, monkeypatch):
    from app.services import evidence_report_refresh
    from app.services.evidence_translation_backfill import backfill_report_translations
    session, report, _ = store
    report.report_json = {**report.report_json, "evidence_report": {
        **report.report_json["evidence_report"], "citations": [
            {"student_text": STATEMENT, "members": [member([HAN_1, HAN_2, "Rivers flood."])]}]}}
    session.commit()

    def forbidden(*args, **kwargs):
        raise AssertionError("no model call in a dry run")
    counts = backfill_report_translations(session, None, report.id, dry_run=True, completion=forbidden)
    assert counts == {"report_id": str(report.id), "cited_sources": 1, "sentences": 2, "already_translated": 0,
                      "to_translate": 2, "estimated_calls": 1, "dry_run": True, "refreshed": False}
    refreshed = []
    monkeypatch.setattr(evidence_report_refresh, "refresh_evidence_report_projection",
                        lambda session, backend, *, report_id, evidence_translations: refreshed.append(
                            evidence_translations) or {"report_id": "new", "report_version": 2})
    calls = []
    summary = backfill_report_translations(session, None, report.id, completion=fake_completion(calls))
    assert len(calls) == 1 and summary["translated"] == 2 and summary["new_report_id"] == "new"
    assert refreshed[0]["sentences"] == {sentence_key(HAN_1): ENGLISH[HAN_1], sentence_key(HAN_2): ENGLISH[HAN_2]}
    assert refreshed[0]["source"] == "backfill"
