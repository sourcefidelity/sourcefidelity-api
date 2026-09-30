"""Judgment claim underline geometry (Phase C)."""
from app.services.judgment_layer import claim_span_rectangles, render_judgment_assets

TEXT = "Teachers said feedback helped (Rivera, 2019)."
# Line 1: "Teachers said feed-"   Line 2: "back helped (Rivera, 2019)."
WORDS = {0: [
    (10, 10, 50, 20, "Teachers", 0, 0, 0), (52, 10, 70, 20, "said", 0, 0, 1), (72, 10, 95, 20, "feed-", 0, 0, 2),
    (10, 22, 30, 32, "back", 0, 1, 0), (32, 22, 60, 32, "helped", 0, 1, 1),
    (62, 22, 90, 32, "(Rivera,", 0, 1, 2), (92, 22, 115, 32, "2019).", 0, 1, 3),
    (10, 60, 60, 70, "Unrelated", 0, 3, 0),
]}
CITATION = {"student_text": TEXT, "paper_character_start": 1000, "paper_location": {
    "localization_level": "exact_rectangle",
    "rectangles": [{"page_index": 0, "x0": 10, "y0": 10, "x1": 95, "y1": 20},
                   {"page_index": 0, "x0": 10, "y0": 22, "x1": 115, "y1": 32}]}}


def _range(fragment):
    start = TEXT.index(fragment)
    return [(1000 + start, 1000 + start + len(fragment))]


def test_exact_claim_words_across_a_hyphenated_line_break():
    rects, placement = claim_span_rectangles(CITATION, _range("said feedback"), WORDS)
    assert placement == "exact"
    assert [(r["x0"], r["x1"], r["y0"]) for r in rects] == [(52, 95, 10), (10, 30, 22)]


def test_a_discontinuous_claim_selects_only_its_segments():
    rects, placement = claim_span_rectangles(CITATION, _range("Teachers") + _range("helped"), WORDS)
    assert placement == "exact" and [(r["x0"], r["x1"]) for r in rects] == [(10, 50), (32, 60)]


def test_words_outside_the_citation_are_never_selected():
    rects, _ = claim_span_rectangles(CITATION, _range(TEXT), WORDS)
    assert all(r["y0"] < 40 for r in rects)


def test_a_mismatch_falls_back_to_the_whole_citation():
    changed = {**CITATION, "student_text": "Teachers said something else entirely (Rivera, 2019)."}
    rects, placement = claim_span_rectangles(changed, [(1000, 1010)], WORDS)
    assert placement == "citation_span" and len(rects) == 2


def test_an_unlocated_citation_has_no_underline():
    unlocated = {**CITATION, "paper_location": {"localization_level": "page_only", "rectangles": []}}
    assert claim_span_rectangles(unlocated, _range("said"), WORDS) == ([], "none")


def test_assets_escape_the_data_block_and_have_no_notice():
    layer = {"marks": [{"key": "k", "group": "g", "citation": 1, "record": None, "candidate": "c",
                        "state": "pending", "rects": [], "placement": "none", "order": 0}],
             "static_windows": {"k": "</script><script>alert(1)</script>"}}
    html = render_judgment_assets(layer, report_id="r1", nonce="n" * 20)
    assert "</script><script>alert" not in html
    assert "judgment-notice" not in html and "<dialog" not in html
    assert html.count("<script") == 2


def test_marks_show_in_the_one_layout():
    from app.services.judgment_layer import JUDGMENT_CSS
    assert ".judgment-overlay{cursor:pointer}" in JUDGMENT_CSS and "layout-judgment" not in JUDGMENT_CSS

