"""Owner requests of 2026-09-28: proposition windows, single-run costs, GLM prices, DOIs."""
from types import SimpleNamespace

import pytest

from app.services.evidence_report import _render_panel_template, attach_propositions
from app.services.ref_field_extractor import _clean_doi, decode_doi
from app.services.report_run_metrics import first_run_stages, judged_citations, judgment_usage, single_run_metrics
from app.services.usage_cost import USD_PER_CNY, llm_cost, llm_price_context

TEXT = "Production became project-oriented, with teams formed around tasks (Grabher, 2002)."


def _citation():
    return {"citation_number": 10, "student_text": TEXT, "paper_character_start": 100, "members": []}


def _layer():
    a_end = TEXT.index(",")
    b_start, b_end = TEXT.index("with"), TEXT.index(" (")
    return {"marks": [
        {"citation": 10, "group": f"10:{100 + b_start}-{100 + b_end}", "order": 100 + b_start},
        {"citation": 10, "group": f"10:100-{100 + a_end}", "order": 100},
        {"citation": 10, "group": "10:whole", "order": 100},
    ]}


def test_a_citation_with_two_propositions_gets_lettered_windows_with_bold_wording():
    citations = [_citation()]
    attach_propositions(citations, _layer())
    assert [p["letter"] for p in citations[0]["propositions"]] == ["A", "B"]
    html = _render_panel_template(citations[0], 10)
    assert '<h2>Citation 10<span data-proposition-suffix></span></h2>' in html
    assert '<strong class="proposition">Production became project-oriented</strong>' in html
    assert '<strong class="proposition">with teams formed around tasks</strong>' in html
    assert html.count('class="selected-citation"') == 2
    assert 'data-proposition-letter="B" hidden' in html


def test_one_proposition_keeps_one_window():
    citations = [_citation()]
    layer = _layer()
    layer["marks"] = layer["marks"][1:]
    attach_propositions(citations, layer)
    assert "propositions" not in citations[0]
    assert "data-proposition=" not in _render_panel_template(citations[0], 10)


def test_windows_carry_no_caution_line_and_how_to_read_does():
    from app.services.evidence_report import render_evidence_report_html, _render_how_to_read
    from bs4 import BeautifulSoup
    view = {"title": "T", "citation_format": "APA", "paper_surface": {},
            "citations": [{"student_text": "A claim (A, 2020).", "members": []}]}
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce="caution-line-nonce"), "html.parser")
    assert "AI can make mistakes" not in soup.select_one("template#citation-panel-1").decode_contents()
    guide = BeautifulSoup(_render_how_to_read({}), "html.parser").select_one("details")
    assert guide.select("p")[-1].get_text() == "AI can make mistakes. Check the sources to verify judgments."


def test_a_supports_result_has_the_fixed_supports_note():
    from app.services.judgment_report import judgment_result
    html = judgment_result({"display_state": "supported", "reason_code": "judge_supports", "candidate_id": "c1",
                            "panel": {"arms": []}, "coaching": {"status": "not_applicable"}}, {}, None)["html"]
    assert "The source supports this statement." in html and "Read the evidence" not in html


def test_single_run_is_the_first_check_up_to_its_first_finalize():
    stages = [{"stage": s} for s in ("extract", "retrieve", "verify", "finalize", "verify", "finalize")]
    assert [s["stage"] for s in first_run_stages(stages)] == ["extract", "retrieve", "verify", "finalize"]
    assert first_run_stages(stages[:3]) == stages[:3]


def test_glm_is_priced_from_the_bigmodel_list_price():
    context = llm_price_context("glm-5.3-flash", "https://open.bigmodel.cn/api/paas/v4")
    assert context["usd_per_million"] == pytest.approx([0.23 * USD_PER_CNY, 0.8 * USD_PER_CNY, 2.8 * USD_PER_CNY])
    assert llm_price_context("glm-5.3-flash", "https://api.z.ai/api/paas/v4")["usd_per_million"] == [0.03, 0.15, 0.5]
    assert llm_price_context("other-model", "https://open.bigmodel.cn/x") == {}
    usage = {**context, "prompt_tokens": 1_000_000, "prompt_cache_hit_tokens": 0,
             "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 0}
    assert llm_cost(usage) == pytest.approx(0.8 * USD_PER_CNY)


def test_judgment_run_is_costed_as_if_nothing_were_cached():
    settings = SimpleNamespace(ZAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
                               LLM_BASE_URL="https://api.deepseek.com", LLM_MODEL="deepseek-v4-flash")
    arms = [SimpleNamespace(arm_id="zai_glm", model="glm-5.3-flash", cost_usd=0.0, cost_basis="ceiling_estimate",
                            usage={"prompt_tokens": 1000, "completion_tokens": 100, "attempts": 1})]
    results = [SimpleNamespace(citation_index=3, display_state="qualified",
                               coaching={"attempts": 1, "cached": True, "cost_basis": "tariff", "original_cost_usd": 0.0002}),
               SimpleNamespace(citation_index=4, display_state="not_judged", coaching=None)]
    rows = {r["model"]: r for r in judgment_usage(results, arms, settings)}
    glm = rows["glm-5.3-flash"]
    assert glm["endpoint_host"] == "open.bigmodel.cn" and glm["calls"] == 1
    assert glm["cost_usd"] == pytest.approx((1000 * 0.8 + 100 * 2.8) * USD_PER_CNY / 1_000_000, abs=1e-8)
    note = rows["deepseek-v4-flash"]
    assert note["cost_usd"] == pytest.approx(0.0002) and note["missing_usage_calls"] == 1
    assert judged_citations(results) == [3]


def test_single_run_metrics_add_the_judgment_run():
    job = SimpleNamespace(source_results=[], upload_evidence={"processing_stages": [
        {"stage": "extract", "metrics_version": 2, "llm_calls": 0, "total_tokens": 0, "missing_usage_calls": 0,
         "wall_seconds": 2.0, "cpu_seconds": 1.0},
        {"stage": "finalize", "metrics_version": 2, "llm_calls": 0, "total_tokens": 0, "missing_usage_calls": 0,
         "wall_seconds": 1.0, "cpu_seconds": 1.0},
        {"stage": "verify", "metrics_version": 2, "llm_calls": 0, "total_tokens": 0, "missing_usage_calls": 0,
         "wall_seconds": 100.0, "cpu_seconds": 50.0}]})
    settings = SimpleNamespace(ZAI_BASE_URL="https://api.z.ai/api/paas/v4",
                               LLM_BASE_URL="https://api.deepseek.com", LLM_MODEL="deepseek-v4-flash")
    arms = [SimpleNamespace(arm_id="zai_glm", model="glm-5.3-flash", cost_usd=None, cost_basis="unpriced",
                            usage={"prompt_tokens": 2_000_000, "completion_tokens": 0})]
    metrics = single_run_metrics(job, arms=arms, settings=settings)
    assert metrics["wall_seconds"] == 3.0            # the later refresh is left out
    assert metrics["estimated_cost_usd"] == pytest.approx(0.30)
    assert [r["model"] for r in metrics["llm_by_model"]] == ["glm-5.3-flash"]


@pytest.mark.parametrize("raw,clean", [
    ("10.21066/CARCL.LIBRI.2015-04%2802%29.0001", "10.21066/CARCL.LIBRI.2015-04(02).0001"),
    ("https://doi.org/10.1234/x(02)", "10.1234/x(02)"),
    ("10.1000/abc).", "10.1000/abc"),
])
def test_a_percent_encoded_doi_is_decoded_once(raw, clean):
    assert _clean_doi(raw) == clean


def test_decoding_never_produces_a_non_doi():
    assert decode_doi("10.1/a%20b") == "10.1/a%20b"
