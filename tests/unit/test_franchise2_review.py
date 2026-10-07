"""Owner review of a re-run APA report, 2026-10-07: defects found in one report,
each pinned here with invented fixtures."""
from types import SimpleNamespace

import fitz

from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.text_extractor import isolate_pdf_headings


def test_a_slanted_regular_font_reads_as_italic():
    # Some PDFs slant a regular font for italics; the font name says regular.
    from app.services.reference_layout import _slanted_characters, _split_by_slant
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Plain words then ")
    start = 72 + fitz.get_text_length("Plain words then ", fontsize=11)
    page.insert_text((start, 100), "Journal of Ports", morph=(fitz.Point(start, 100), fitz.Matrix(1, 0, -0.33, 1, 0, 0)))
    content = doc.tobytes()
    slanted = _slanted_characters(content)[0]
    assert any(s for *_box, s in slanted) and not all(s for *_box, s in slanted)
    raw = fitz.open(stream=content, filetype="pdf")[0].get_text("rawdict")
    spans = [part for block in raw["blocks"] for line in block.get("lines", []) for span in line["spans"]
             for part in _split_by_slant(span, slanted)]
    assert any(p["slanted"] and "Ports" in p["text"] for p in spans)
    assert any(not p["slanted"] and "Plain" in p["text"] for p in spans)


def test_a_wholly_bold_line_is_a_heading_in_any_case():
    text = ("The film earned a great deal of money.\nPerformance and influencing factors\n"
            "The video game sold over 10,000 copies (Ames, 2023).")
    assert "\n\nPerformance and influencing factors\n\n" not in isolate_pdf_headings(text)
    isolated = isolate_pdf_headings(text, bold_lines=frozenset({"Performance and influencing factors"}))
    assert "\n\nPerformance and influencing factors\n\n" in isolated
    # A bold line that ends a sentence is not a heading.
    sentence = "The film earned a great deal of money.\nIt sold well in many markets.\nThe game followed."
    assert isolate_pdf_headings(sentence, bold_lines=frozenset({"It sold well in many markets."})) == sentence


def test_a_query_split_or_a_long_slug_on_the_next_line_completes_the_url():
    from app.services.reference_parser import extract_and_parse_references
    from app.services.reference_url_repair import repair_reference_urls
    layout = ("References\nBox office. (n.d.) Retrieved from: https://example.test/i/52/show?share=iOS&utm_sour\n"
              "ce=app_share\nGame sales. (2023). Retrieved from: https://example.test/news/s\n"
              "ales_of_the_game_within_ten_days\nShort. (2020). Retrieved from: https://example.test/p/1_The_P\n"
              "ractice_of_Media\nTail. (2021). Retrieved from: https://example.test/news/sales_within\n_10_days\n")
    refs = extract_and_parse_references(layout, format_hint="apa", use_regex_first=True, use_llm_fallback=False,
                                        paper_version_id="x")
    urls = [r.url for r in repair_reference_urls(refs, layout)]   # physical lines, as the pipeline passes them
    assert urls[0] == "https://example.test/i/52/show?share=iOS&utm_source=app_share"
    assert urls[1] == "https://example.test/news/sales_of_the_game_within_ten_days"
    assert urls[2] == "https://example.test/p/1_The_P"      # a short fragment still needs proof
    assert urls[3] == "https://example.test/news/sales_within_10_days"   # a line opening with a separator


def test_a_same_titled_record_outside_the_cited_journal_is_not_an_author_conflict():
    from app.services.reference_verification import same_title_author_difference
    reference = SimpleNamespace(title="Harbour Master: Detective", author="Ames, R", container_title="Antique Weekly",
                                raw_ref="Ames, R. (2015). Harbour Master: Detective. Antique Weekly, 28(10), 26.",
                                publisher="", source_kind="journal_article")
    book = {"provider": "web_search", "candidate_id": "c1",
            "observed": {"title": "Harbour Master: Detective", "authors": ["Theodore Blane"], "container_title": ""}}
    assert same_title_author_difference(reference, {"candidates": [book]}) is None
    same_journal = {**book, "observed": {**book["observed"], "container_title": "Antique Weekly"}}
    assert same_title_author_difference(reference, {"candidates": [same_journal]}) is not None


def test_a_web_found_pdf_is_checked_against_the_references_page_range():
    # A copy with no provider record took no page range, so an 18-page article
    # PDF was judged "uncertain" and shown as limited text.
    from app.services.retrieval.base import RetrievalResult
    from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE, _expected_page_range
    result = RetrievalResult(source_name="web_search", success=True)
    assert _expected_page_range(result) is None
    token = _ACTIVE_DISCOVERY_TRACE.set({"expected": ExpectedBibliographicFields(title="A title", pages="537-554")})
    try:
        assert _expected_page_range(result) == (537, 554)
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def _member(observation, outcome="response", coverage="unavailable"):
    return {"reference_id": "r1", "coverage_level": coverage, "submitted_link_observations": [
        {"state": "observed", "requests": [{"completed_at": "2026-10-07T00:00:00Z", "outcome": outcome,
                                            "page_observation": observation}]}]}


def test_an_unretrieved_link_says_why_in_the_owners_words():
    from app.services.evidence_report import _submitted_link_unretrieved_reason as reason
    assert reason(_member("page_title_mismatch_unconfirmed")) == (
        "The submitted link opened a page with a different title, so the source could not be retrieved.")
    assert reason(_member("readable_text_unavailable")) == "The submitted link opened the page, but its text could not be read."
    assert reason(_member("not_assessed", outcome="timeout")) == "The submitted link could not be reached when checked."
    # Text was retrieved, or a finding already says what the link did: nothing more.
    assert reason(_member("page_title_mismatch_unconfirmed", coverage="full_text")) is None
    assert reason(_member("page_title_mismatch_unconfirmed"),
                  [{"reference_id": "r1", "finding_type": "submitted_link_issue"}]) is None
    assert reason(_member("not_assessed", outcome="not_found")) is None


def test_a_volume_number_alone_in_regular_type_is_flagged():
    import hashlib
    from app.services.reference_formatting import ReferenceTitleStyleResult, reference_style_findings
    raw = "Ames, R. (1992). How costly is it? Journal of Ports, 6(3), 159-178."
    ref = SimpleNamespace(reference_id="r1", raw_ref=raw)
    start = raw.index("6(3)")
    result = ReferenceTitleStyleResult(
        rule_id="apa7_periodical_volume_italics_v1", reference_id="r1",
        reference_text_sha256=hashlib.sha256(raw.encode()).hexdigest(), source_kind="journal_article",
        status="difference", reason_code="periodical_volume_not_italic", title_start=start, title_end=start + 1,
        title_sha256=hashlib.sha256(b"6").hexdigest(), expected_italic=True, observed_italic=False)
    payload = {"assessment_version": "reference-formatting-v2", "title_results": [dict(result)]}
    [finding] = reference_style_findings(payload, {"r1": ref})
    assert finding["finding"] == "The volume number is not italicized in this APA reference."


def test_a_repeated_number_is_placed_at_its_own_offset():
    import hashlib
    from app.services.evidence_report import _attach_reference_field_geometry
    entry = "Ames, R. (1996). Six ways. Journal of Ports, 6(3), 156-178."
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), entry)
    offset = entry.index("6(3)")
    probe = {"finding_type": "reference_title_style", "source": {"raw_reference": entry},
             "field_difference": {"submitted_value": "6", "raw_offset": offset}}
    _attach_reference_field_geometry({"reference_practice": [probe]}, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    assert len(probe["rectangles"]) == 1
    without = {**probe, "field_difference": {"submitted_value": "6"}}
    _attach_reference_field_geometry({"reference_practice": [without]}, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    assert without["rectangles"] == []


def test_a_passage_a_page_break_interrupts_is_found_between_its_ends():
    import hashlib
    from app.services.evidence_report import _attach_reference_field_geometry
    first = "The film Harbour Lights opened to strong reviews and long queues across the country that year,"
    second = "and later critics called Harbour Lights a turning point for the studio and its younger directors."
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 700), first)
    page.insert_text((72, 780), "Running Header 7")
    doc.new_page().insert_text((72, 72), second)
    probe = {"finding_type": "body_title_style", "citation_text": f"{first} {second}",
             "field_difference": {"submitted_value": "Harbour Lights"}}
    twin = {**probe, "field_difference": {"submitted_value": "Harbour Lights"}}
    _attach_reference_field_geometry({"reference_practice": [probe, twin]}, doc, hashlib.sha256(doc.tobytes()).hexdigest())
    assert probe["rectangles"] and twin["rectangles"]
    assert probe["rectangles"][0]["page_index"] == 0 and twin["rectangles"][0]["page_index"] == 1


def test_a_line_break_hyphen_inside_a_word_does_not_change_the_title():
    from app.services.reference_discovery import build_reference_discovery_candidate
    from app.services.retrieval.base import RetrievalResult
    expected = ExpectedBibliographicFields(title="Revolution-izing harbour teaching with chatbots", authors=["Lin, Q."], year="2023")
    result = RetrievalResult(source_name="semantic_scholar", success=True,
                             title="Revolutionizing Harbour Teaching with Chatbots", authors=["Q. Lin"], year="2023")
    built = build_reference_discovery_candidate(attempt_id="a", provider="semantic_scholar", expected=expected, result=result)
    assert {c.field_name: c.outcome for c in built.comparisons}["title"] == "agreement"
    # A real compound keeps its hyphen's meaning against a different word.
    other = RetrievalResult(source_name="semantic_scholar", success=True,
                            title="Revolution in harbour teaching with chatbots", authors=["Q. Lin"], year="2023")
    built = build_reference_discovery_candidate(attempt_id="b", provider="semantic_scholar", expected=expected, result=other)
    assert {c.field_name: c.outcome for c in built.comparisons}["title"] != "agreement"


def test_a_url_cut_after_a_slash_continues_on_the_next_line():
    from app.services.reference_parser import extract_and_parse_references
    from app.services.reference_url_repair import repair_reference_urls
    layout = "References\nAmes, A. (2022). Harbour usage notes. Open Archive. https://archive.example.test/\nhal-03913837\n"
    refs = extract_and_parse_references(layout, format_hint="apa", use_regex_first=True, use_llm_fallback=False,
                                        paper_version_id="x")
    assert repair_reference_urls(refs, layout)[0].url == "https://archive.example.test/hal-03913837"


def test_a_doi_address_broken_after_doi_dot_is_one_link():
    from app.services.evidence_report import _render_formatted_reference
    raw = "Ames, A. (2025). Title of a study. Journal of Ports, 3, 100121. https://doi. org/10.1016/j.ports.2025.100121"
    html = _render_formatted_reference({"raw_reference": raw}, raw)
    assert 'href="https://doi.org/10.1016/j.ports.2025.100121"' in html


def test_a_table_without_rules_is_kept_apart_from_the_prose_after_it():
    from app.services.text_extractor import isolate_pdf_tables
    text = ("Earlier prose ends here (Ames, 2001).\n\nTable 1 Harbour essay marking rubric\n0–39 40–49 50–59\n"
            "Partially Primarily Correct,\nCompletely Repeatedly Occasionally\n– 15% writing conciseness writing\n"
            "It is difficult for the models to produce analyses because they are dependent on the data\n"
            "and do not truly understand context (Lin, 2023). Unless the analyses are common, it fails.")
    out = isolate_pdf_tables(text)
    assert "\n\nIt is difficult for the models" in out
    assert out.split("\n\n")[1].startswith("Table 1")
    # A paragraph without a table caption is untouched.
    plain = "Some prose.\n\nIt is difficult for the models to produce analyses because they are dependent. Yes."
    assert isolate_pdf_tables(plain) == plain


def test_a_long_reference_list_starting_before_the_last_pages_counts_as_back_matter():
    from app.services.completeness_checker import check_completeness
    doc = fitz.open()
    for page in range(12):
        p = doc.new_page()
        if page < 7:
            p.insert_text((72, 100), f"Body text of the article on page {page + 1}, discussing harbour teaching.")
        else:
            if page == 7:
                p.insert_text((72, 80), "REFERENCES")
            for row in range(6):
                p.insert_text((72, 120 + row * 40), f"Ames, A. ({2000 + row}). Harbour study {row}. "
                                                    f"https://doi.org/10.1000/h{page}{row}")
    signals = check_completeness(doc.tobytes(), is_article=True, title="Harbour", author="Ames, A").signals
    assert any("found heading 'REFERENCES'" in s for s in signals)


def test_title_first_entries_are_told_apart_by_their_titles():
    from app.services.reference_consistency import apa_in_text_form
    one = SimpleNamespace(author="", title="Harbour season 1", year="n.d.")
    four = SimpleNamespace(author="", title="Harbour season 4", year="n.d.")
    assert apa_in_text_form(one) != apa_in_text_form(four)


def test_a_bot_check_page_is_a_wall():
    from app.services.source_resolver import interstitial_page_title
    assert interstitial_page_title("Making sure you're not a bot!")
    assert not interstitial_page_title("Notes on a robotics festival")
