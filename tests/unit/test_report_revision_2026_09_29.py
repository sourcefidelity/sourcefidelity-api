"""Owner requests of 2026-09-29: link retry, judgment summary, summary wording, marks."""
from pathlib import Path

from bs4 import BeautifulSoup

from app.services import source_resolver
from app.services.evidence_report import (
    _build_report_summary, _render_reference_panel_template, _render_upload_priorities,
    judgment_summary_sentence, render_evidence_report_html, summary_text,
)
from app.services.retrieval.base import RetrievalResult


def test_a_student_link_is_tried_again_after_a_network_error(monkeypatch):
    monkeypatch.setattr(source_resolver, "STUDENT_LINK_RETRY_DELAY_SECONDS", 0)
    resolver = object.__new__(source_resolver.SourceResolver)
    calls = []

    def attempt():
        calls.append(1)
        if len(calls) == 1:
            return RetrievalResult(source_name="web_fetch", success=False, error="web_fetch_failed:connect_error")
        return RetrievalResult(source_name="web_fetch", success=True)
    result = resolver._retry_after_network_error(attempt)
    assert result.success and len(calls) == 2
    assert result.metadata["student_link_retry"] == {"first_failure": "connect_error"}


def test_an_answer_from_the_site_is_not_retried(monkeypatch):
    monkeypatch.setattr(source_resolver, "STUDENT_LINK_RETRY_DELAY_SECONDS", 0)
    resolver = object.__new__(source_resolver.SourceResolver)
    for error in ("Page title does not match the cited title", "web_fetch_failed:http_404",
                  "web_fetch_failed:http_403"):
        calls = []
        resolver._retry_after_network_error(
            lambda: calls.append(1) or RetrievalResult(source_name="web_fetch", success=False, error=error))
        assert len(calls) == 1, error


def test_judgment_summary_lists_only_results_that_occur():
    text = judgment_summary_sentence({"supported": 6, "qualified": 2}, 9, 28)
    assert text == ("6/8 statements are supported by the sources, 2/8 have qualified or mixed support "
                    "in the sources, and 9/28 citations lack full texts and are not judged.")
    assert "contradict" not in text and "insufficient" not in text
    assert judgment_summary_sentence({"contradicts": 1}, 0, 5) == "1/1 statements contradict the sources."
    assert judgment_summary_sentence({"supported": 1, "undecided": 1}, 0, 5) == (
        "1/2 statements are supported by the sources, and 1/2 cannot be decided upon by the LLM.")
    # "statements" goes with whichever part comes first.
    assert judgment_summary_sentence({"qualified": 1, "insufficient": 2, "undecided": 1}, 3, 9) == (
        "1/4 statements have qualified or mixed support in the sources, 2/4 are not supported by the "
        "sources, 1/4 cannot be decided upon by the LLM, and 3/9 citations lack full texts "
        "and are not judged.")
    script = Path("app/services/report_judgment.js").read_text()
    assert "i === 0 ? ' statements' : ''" in script and "statements are supported" not in script
    assert judgment_summary_sentence({}, 0, 5) == ""


SOURCE = {"source": {"author": "A", "year": "2020", "title": "T", "raw_reference": "A. (2020). T."}}


def test_the_judgment_summary_follows_the_retrieval_counts_in_sources():
    view = {"title": "T", "citation_format": "APA", "paper_surface": {},
            "overview": {"reference_count": 2, "verified_full_text_sources": 1},
            "judgment_summary": {"judged_citations": [1], "states": {"supported": 1}},
            "citations": [{"student_text": "A (A, 2020).", "members": [{"coverage_level": "full_text", **SOURCE}]},
                          {"student_text": "B (B, 2021).", "members": [{"coverage_level": "unavailable", **SOURCE}]}]}
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce="judgment-summary-nonce"), "html.parser")
    items = soup.select(".summary-column")[0].select("li")
    assert items[0]["class"] == ["retrieval-summary"]
    assert items[1].get_text() == ("1/1 statements are supported by the sources, and 1/2 citations lack full texts "
                                   "and are not judged.")


def _finding(kind, n, **extra):
    return {"finding_type": kind, "reference_id": f"r{n}", "rectangles": [{"page_index": 0, "x0": 1, "y0": 1,
                                                                             "x1": 2, "y1": 2}], **extra}


def test_submitted_links_have_two_specific_lines_and_singular_wording():
    practice = [_finding("submitted_link_issue", 1, link_outcome="site_homepage"),
                _finding("submitted_link_issue", 2, link_outcome="missing_page")]
    summary = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
                                    reference_practice=practice, reference_numbers={"r1": 1, "r2": 2})
    texts = list(map(summary_text, summary["academic_practice"]))
    assert "1 reference has a submitted link that returned a missing page or conflicting destination details." in texts
    assert "1 reference links to a website address rather than a page for the cited work." in texts
    assert not any("submitted-link issues" in t for t in texts)


def test_single_reference_findings_read_in_the_singular():
    summary = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
                                    reference_practice=[_finding("required_doi_missing", 1),
                                                        _finding("reference_order", 2)],
                                    reference_numbers={"r1": 1, "r2": 2})
    texts = list(map(summary_text, summary["reference_formatting"]))
    assert "1 reference omits a DOI for the cited work." in texts
    assert "1 reference is not in alphabetical order." in texts


def test_unverifiable_mark_has_no_outline_and_headings_no_underline():
    html = render_evidence_report_html({"title": "T", "citation_format": "APA", "citations": [], "paper_surface": {}},
                                       csp_nonce="unverified-outline-nonce")
    assert "stroke:#c5221f" not in html and "outline:1px solid #c5221f" not in html
    assert ".issue-heading{font-weight:700;text-decoration:none;" in html


def test_the_verified_doi_window_does_not_name_the_provider():
    html = _render_reference_panel_template({"finding_type": "required_doi_missing", "finding": "DOI missing.",
                                             "verified_doi": "10.1/x", "provider": "crossref",
                                             "source": {"raw_reference": "Ref."}}, 1)
    assert "crossref" not in html and "provider" not in html


def test_the_upload_sentence_is_on_the_heading_line_and_not_bold():
    citations = [{"members": [{"reference_id": "A", "coverage_level": "unavailable",
                               "source": {"raw_reference": "A. A source.", "source_kind": "monograph"}}]}]
    soup = BeautifulSoup(_render_upload_priorities(citations, judgments=14), "html.parser")
    heading = soup.select_one("h2.upload-heading")
    assert heading.select_one("strong").get_text() == "Sources to Upload"
    assert heading.get_text() == ("Sources to Upload - This paper has 14 judgments. If this source is uploaded, "
                                  "the paper will have 1 more judgment.")


def test_a_serial_list_before_a_participle_stays_in_its_clause():
    from app.services.candidate_relationship_judgment import _decompose
    text = ("The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, and the nature "
            "of reality, allowing her to grow from a confused girl to a brave and confident hero (Flegar & Wertag, 2015).")
    specs = _decompose(text, 0, len(text))
    parts = [" | ".join(text[a:b] for a, b in spec.spans) for spec in specs]
    main = "The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, and the nature of reality"
    assert parts[0] == main
    assert parts[1].startswith(main + " | allowing her to grow")
    assert len(parts) == 2


def test_about_is_a_quantity_only_before_a_number():
    from app.services.facet_evidence_judgment import _FACET_PATTERNS
    quantity = dict(_FACET_PATTERNS)["quantity"]
    assert not quantity.search("cultural and political factors have brought about changes")
    assert not quantity.search("a film about Alice")
    assert quantity.search("about 40 films") and quantity.search("approximately half of the studios")


def test_a_main_clause_and_its_participle_are_both_judgeable():
    import sys
    sys.path.insert(0, "tests/unit")
    from test_candidate_relationship_judgment import _artifact
    from app.services.candidate_relationship_judgment import attach_verification_candidates
    text = ("The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, and the nature "
            "of reality, allowing her to grow from a confused girl to a brave and confident hero (Flegar & Wertag, 2015).")
    generated = attach_verification_candidates(_artifact(text, "(Flegar & Wertag, 2015)"))
    eligible = [c for c in generated.verification_candidates.candidates if c.relationship_eligible]
    assert [c.generation_method for c in eligible] == ["structural_clause_fallback", "participial_clause_with_exact_parent"]
    assert eligible[0].text.endswith("identity, and the nature of reality")


def test_this_portrayal_resolves_to_the_portraying_phrase_before_it():
    import sys
    sys.path.insert(0, "tests/unit")
    from test_antecedent_resolver import _claim
    from app.services.antecedent_resolver import resolve_claim_antecedents
    from app.services.verification_report import _validate_claim_antecedents
    cited = ("This portrayal reflects broader societal expectations and cultural conventions of this period "
             "(Doster, 2002).")
    body = ("In contrast, Disney's early adaptation simplified these themes, portraying Alice simply as a naive "
            "child whose adventures emphasize whimsy instead of introspection. " + cited)
    resolved = resolve_claim_antecedents(body, _claim(body, cited, marker="(Doster, 2002)"))
    dependency = resolved.antecedent_dependencies[0]
    assert dependency.mention_text == "This portrayal"
    assert dependency.resolution_status == "resolved"
    assert dependency.antecedent_text.startswith("portraying Alice simply as a naive child")
    _validate_claim_antecedents(resolved)


def test_secondary_citation_is_flagged_only_when_every_bearing_sentence_cites_another_work():
    from app.services.secondary_citation import attributed_names, secondary_citation
    doster = {"author": "Doster, I. V.", "raw_reference": "Doster, I. V. (2002). The Disney dilemma."}
    assert attributed_names("In fact, some authors (Boltanski and Chiapello, 1999) see projects as a logic.") == [
        "Boltanski", "Chiapello"]
    assert attributed_names("According to Bordwell, the system ended.") == ["Bordwell"]
    assert attributed_names("The system ended in 1948 (p. 12).") == []
    flagged = secondary_citation([
        {"key": "a", "reason": "bears_on_statement", "text": "Grabher (2002) argues projects are temporary."},
        {"key": "b", "reason": "necessary_context", "text": "Context without a citation."}], doster)
    assert flagged["sentence_keys"] == ["a"] and flagged["attributed_to"] == ["Grabher"]
    # A bearing sentence in the source's own voice, or its own author, clears the flag.
    assert secondary_citation([
        {"key": "a", "reason": "bears_on_statement", "text": "Grabher (2002) argues projects are temporary."},
        {"key": "c", "reason": "bears_on_statement", "text": "Disney films simplified Alice."}], doster) is None
    assert secondary_citation([
        {"key": "a", "reason": "bears_on_statement", "text": "Doster (2002) shows the dilemma."}], doster) is None


def test_a_flagged_source_is_summarized_and_highlighted_yellow():
    member = {"reference_id": "r1", "coverage_level": "full_text",
              "secondary_citation": {"sentence_keys": ["a"], "attributed_to": ["Grabher"]}}
    summary = _build_report_summary(citations=[{"members": [member]}], overview={}, pervasive_hanging_indent=False)
    assert ("1 citation currently relies on passages where the cited source represents another work."
            in [text.strip() for text in map(summary_text, summary["academic_practice"])])


def test_an_exported_report_carries_its_finished_judgments():
    import json
    view = {"title": "T", "citation_format": "APA", "paper_surface": {}, "portable_export": True,
            "report_id": "r", "citations": [{"student_text": "A (A, 2020).", "members": []}],
            "judgment_summary": {"states": {"supported": 1}},
            "judgment_layer": {"static": True, "marks": [{"key": "1:x:c", "group": "1:0-5", "citation": 1,
                                                          "state": "pending", "rects": [], "order": 0}],
                               "static_results": [{"seq": 1, "citation_number": 1, "verification_report_id": "x",
                                                   "candidate_id": "c", "display_state": "supported",
                                                   "window_html": "<section></section>", "evidence": []}]}}
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce="static-judgment-nonce"), "html.parser")
    data = json.loads(soup.find(id="judgment-data").string)
    assert data["static_results"][0]["display_state"] == "supported"
    assert soup.select_one(".judgment-summary").get_text().startswith("1/1 statements are supported by the sources")
    assert "const remember=false" in str(soup)       # How to read shows on every opening of an export


def test_a_media_reference_window_says_it_cannot_be_assessed():
    from app.services.evidence_report import _render_reference_window_template
    entry = {"number": 3, "template_id": "reference-entry-panel-3", "source": {"raw_reference": "Burton, T. (2010). Alice [Film]."},
             "member": {"status": "citation_not_assessed", "coverage_level": "unavailable",
                        "source": {"source_kind": "traditional_media", "raw_reference": "Burton, T. (2010). Alice [Film]."}},
             "citation_numbers": [1]}
    html = _render_reference_window_template(entry, [], [{}])
    assert '<h3 class="reference-availability">Media Reference - Cannot Assess</h3>' in html


def test_an_undecided_judge_has_its_own_label_and_note():
    from app.services.judgment_report import judgment_result
    result = judgment_result({"display_state": "not_judged", "reason_code": "judge_undecided", "candidate_id": "c1",
                              "panel": {"arms": []}}, {}, None)
    assert result["label"] == "LLM Undecided" and result["state"] == "undecided"
    assert "The model cannot decide if this statement is supported." in result["html"]
    assert "Check the evidence yourself" not in result["html"]


def test_a_note_crediting_the_student_with_the_evidence_is_rejected():
    from app.services.judgment_coaching import check_note
    bound = {"citation_marker": "(Doster, 2002)", "claim": "A claim.", "evidence_text": [], "facets": {},
             "sentences": {}}
    note, violations = check_note({"note": "The evidence sentences you supplied concern Basile.",
                                   "facet_ids": [], "sentence_ids": []}, bound)
    assert note is None and "evidence_attributed_to_student" in violations


def test_a_contained_proposition_is_underlined_only_where_it_adds_words():
    from app.services.judgment_layer import _separate_contained
    marks = [{"group": "17:0-10", "ranges": [(0, 10)], "rects": ["whole"]},
             {"group": "17:0-10,12-20", "ranges": [(0, 10), (12, 20)], "rects": ["whole"]}]
    import app.services.judgment_layer as layer
    original = layer.claim_span_rectangles
    layer.claim_span_rectangles = lambda citation, ranges, words: ([{"ranges": ranges}], "exact")
    try:
        _separate_contained(marks, {}, {})
    finally:
        layer.claim_span_rectangles = original
    assert "display_ranges" not in marks[0]
    assert marks[1]["display_ranges"] == [(12, 20)] and marks[1]["rects"] == [{"ranges": [(12, 20)]}]


def test_secondary_citation_window_line_names_the_attributed_authors():
    # Owner wording 2026-09-29, option A with the names.
    from app.services.evidence_report import _render_member, secondary_citation_line
    flag = {"version": "glm-sentence-secondary-citation-v1", "sentence_keys": ["1:0:9"],
            "attributed_to": ["Campbell", "Propp"]}
    line = "This citation relies on source passages that report another author's work (Campbell; Propp)."
    assert secondary_citation_line(flag) == line
    assert secondary_citation_line({"attributed_to": []}) == (
        "This citation relies on source passages that report another author's work.")
    assert secondary_citation_line(None) == ""
    member = {"reference_id": "ref-1", "coverage_level": "full_text", "secondary_citation": flag,
              "source": {"author": "Writer, A.", "year": "2020", "title": "Title",
                         "raw_reference": "Writer, A. (2020). Title."}}
    soup = BeautifulSoup(_render_member(member), "html.parser")
    heading = soup.select_one(".issue-heading.academic")
    assert heading.get_text() == "Academic Practice"
    assert heading.find_parent("p").find_next_sibling("p").get_text() == line


def test_unverified_references_head_the_sources_column():
    # Owner request 2026-09-30.
    from app.services.evidence_report import _render_report_summary
    summary = {"evidence": [{"kind": "identity_conflict", "text": "1 reference contains conflicting fields."},
                            {"kind": "unverified_reference", "text": "2 reference(s) cannot be verified."}],
               "academic_practice": [], "reference_formatting": []}
    soup = BeautifulSoup(_render_report_summary(summary, retrieval="5 of 9 sources retrieved."), "html.parser")
    rows = [li.get_text() for li in soup.select(".summary-column")[0].select("li")]
    assert rows == ["2 reference(s) cannot be verified.", "5 of 9 sources retrieved.",
                    "1 reference contains conflicting fields."]


def test_references_that_cannot_be_verified_are_not_suggested_for_upload():
    # Owner decision 2026-09-30 (Gerbner, Paper 1).
    from app.services.evidence_report import _upload_priorities
    member = lambda rid, **extra: {"reference_id": rid, "coverage_level": "unavailable",
                                   "source": {"source_kind": "journal_article", "raw_reference": f"{rid} (2020)."},
                                   **extra}
    citations = [{"members": [member("ref-a"), member("ref-b", unverified=True), member("ref-c")]}]
    assert [row["member"]["reference_id"] for row in _upload_priorities(citations)] == ["ref-a", "ref-c"]
    assert [row["member"]["reference_id"] for row in _upload_priorities(citations, frozenset({"ref-c"}))] == ["ref-a"]


def test_a_film_named_in_the_paper_is_not_called_uncited():
    # Owner decision 2026-09-30 (Paper 1, reference 1).
    from app.services.evidence_report import _media_named_in_paper, _paper_key
    title_words = ["The", "Last", "Temptation", "of", "Christ"]
    body = [(0, 0, 1, 1, w) for w in ["Scorsese's", *title_words, "divided"]]
    entry = [(0, 0, 1, 1, w) for w in ["Scorsese,", "M.", "(1988).", *title_words, "[Motion", "Picture]."]]
    surface = {"selectable_words": {0: body, 1: entry}}
    # The reference entry alone is not a mention in the paper (review 2026-09-30).
    assert not _media_named_in_paper(
        {"title": "The last temptation of Christ [Motion Picture]", "source_kind": "traditional_media",
         "raw_reference": "Scorsese, M. (1988). The last temptation of Christ [Motion Picture]."},
        _paper_key({"selectable_words": {1: entry}}))
    film = {"title": "The last temptation of Christ [Motion Picture]", "source_kind": "traditional_media",
            "raw_reference": "Scorsese, M. (1988). The last temptation of Christ [Motion Picture]."}
    key = _paper_key(surface)
    assert _media_named_in_paper(film, key)
    assert not _media_named_in_paper({**film, "title": "Another film entirely"}, key)
    webpage = {"title": "The Last Temptation of Christ", "source_kind": "webpage", "raw_reference": "Wiki."}
    assert not _media_named_in_paper(webpage, key)


def test_citation_after_final_punctuation_is_detected_and_summarised():
    # Owner wording 2026-09-30.
    from app.services.evidence_report import _build_report_summary, citation_after_punctuation, summary_text
    after = {"citation_marker": "(Hess,1974)", "student_text": "Films calm audiences with the status quo. (Hess,1974)"}
    opening = {"citation_marker": "(Hess,1974)", "student_text": "(Hess,1974) These films produce satisfaction."}
    correct = {"citation_marker": "(Hess, 1974)", "student_text": "Films calm audiences (Hess, 1974)."}
    narrative = {"citation_marker": "Hess (1974)", "student_text": "Hess (1974) argues this. More follows."}
    assert [citation_after_punctuation(c) for c in (after, opening, correct, narrative)] == [True, True, False, False]
    one = _build_report_summary(citations=[correct, after], overview={}, pervasive_hanging_indent=False)
    assert summary_text(one["reference_formatting"][-1]) == (
        "1 parenthetical citation is placed after the sentence's final punctuation (citation 2).")
    two = _build_report_summary(citations=[after, correct, opening], overview={}, pervasive_hanging_indent=False)
    assert summary_text(two["reference_formatting"][-1]) == (
        "2 parenthetical citations are placed after the sentences' final punctuation (citations 1, 3).")


def test_uncited_references_are_summarised_in_academic_practice():
    # Owner wording 2026-09-30.
    from app.services.evidence_report import _build_report_summary, summary_text
    numbers = {"ref-a": 1, "ref-b": 3, "ref-c": 5}
    one = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
                                reference_numbers=numbers, uncited_reference_ids=["ref-b"])
    assert [summary_text(i) for i in one["academic_practice"]] == ["1 reference is not cited in the paper (reference 3)."]
    two = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=False,
                                reference_numbers=numbers, uncited_reference_ids=["ref-c", "ref-b"])
    assert [summary_text(i) for i in two["academic_practice"]] == [
        "2 references are not cited in the paper (references 3, 5)."]


def test_references_that_cannot_be_verified_keep_their_upload_button():
    # Owner decision 2026-09-30: the reader can supply the source if the search was wrong.
    from app.services.evidence_report import _member_accepts_upload
    member = {"coverage_level": "unavailable", "source": {"source_kind": "journal_article", "raw_reference": "A."}}
    assert _member_accepts_upload(member) and _member_accepts_upload({**member, "unverified": True})


def test_a_quotation_difference_is_shown_beside_the_source_wording_not_highlighted():
    # Owner request 2026-09-30: the same layout as patchwriting.
    from app.services.evidence_report import (_quotation_difference_diagnostics, _render_citation_text,
                                              render_quotation_comparisons)
    text = 'She calls it “a brutal and unflinching portrayal of youth violence in the city” (Lee, 2020, p. 4).'
    passages = [{"text": "The film offers a brutal and honest portrayal of youth violence in the city today.",
                 "page_label": "4"}]
    [difference] = _quotation_difference_diagnostics(text, passages)
    assert difference["source_excerpt"] == {"text": "a brutal and honest portrayal of youth violence in the city",
                                            "page": "4"}
    soup = BeautifulSoup(render_quotation_comparisons({"differences": [difference]}, text), "html.parser")
    student, label, source = soup.select("p")
    assert [b.get_text() for b in student.select("strong")] == ["unflinching"]
    assert student.get_text() == "a brutal and unflinching portrayal of youth violence in the city"
    assert label.get_text() == "Source (p. 4):"
    assert source.get_text() == "a brutal and honest portrayal of youth violence in the city"
    citation = {"student_text": text, "quotation_differences": [difference]}
    assert "<mark" not in _render_citation_text(citation)


def test_line_break_hyphens_and_bracketed_clarifications_are_not_quotation_differences():
    # 2026-09-30: "cultivat-ing" and "[Google Translate]" (Academic Article - Long).
    from app.services.evidence_report import _positive_quote_check, _quotation_difference_diagnostics
    source = ("students have positive attitudes towards GT and employ diverse search strategies to address "
              "various language-related challenges")
    quote = ("students have positive attitudes towards [Google Translate] and employ diverse search strategies "
             "to address vari-ous language-related challenges")
    assert _positive_quote_check([quote], [source])["attention"] is False
    assert _quotation_difference_diagnostics(f"They found that “{quote}” (p. 5).", [source]) == []
    # A real hyphenated compound in the source is left alone, and real changes still count.
    changed = quote.replace("diverse", "many")
    [difference] = _quotation_difference_diagnostics(f"They found that “{changed}” (p. 5).", [source])
    assert [span["paper_text"] for span in difference["spans"]] == ["many"]


def test_a_reference_window_says_cannot_be_verified_once():
    # Owner decision 2026-09-30.
    from app.services.evidence_report import _render_reference_window_template
    from app.services.reference_verification import FINDING_TYPE
    source = {"raw_reference": "Writer, A. (2020). A title. Journal, 1, 1-2.", "author": "Writer, A.",
              "year": "2020", "title": "A title"}
    member = {"reference_id": "r1", "coverage_level": "unavailable", "unverified": True, "source": source}
    finding = {"finding_type": FINDING_TYPE, "reference_id": "r1", "source": source, "rectangles": [{}]}
    entry = {"number": 1, "template_id": "reference-entry-panel-1", "source": source, "member": member,
             "citation_numbers": [1], "first_citation": (1, 0), "finding_indexes": [1]}
    html = _render_reference_window_template(entry, [finding], [{"upload_action": {}}])
    template = BeautifulSoup(html, "html.parser").select_one("template")
    text = BeautifulSoup(template.decode_contents(), "html.parser").get_text(" ")
    assert text.count("Cannot be verified") == 1 and "Searches for this source could not locate the reference." in text


def test_the_citation_window_explains_a_citation_after_the_final_punctuation():
    # Owner wording 2026-09-30.
    from app.services.evidence_report import render_evidence_report_html
    view = {"title": "Report", "citation_format": "APA", "reference_practice": [],
            "paper_surface": {"page_dimensions": [{"page_index": 0, "width": 612, "height": 792}],
                              "page_href_template": "p-{page_index}", "selectable_words": {0: []}},
            "citations": [{"student_text": "Films calm audiences with the status quo. (Hess,1974)",
                           "citation_marker": "(Hess,1974)", "members": []}]}
    soup = BeautifulSoup(render_evidence_report_html(view, csp_nonce="patchwriting-report-nonce"), "html.parser")
    window = BeautifulSoup(soup.select_one("template#citation-panel-1").decode_contents(), "html.parser")
    assert "This parenthetical citation is placed after the sentence's final punctuation." in window.get_text()
    assert window.select_one(".issue-heading.formatting").get_text() == "Citation and reference formatting"


def test_windows_carry_no_coaching_sentences():
    # Owner decision 2026-09-30: no coaching text.
    from app.services.evidence_report import _panel_statement
    text = ("The located work Title is a single-authored book, not an edited collection, so this reference cites "
            "a part of it as though it were a chapter in an edited volume. Cite the book itself and give the page "
            "range of the part used.")
    assert _panel_statement(text) == ("The located work Title is a single-authored book, not an edited collection, "
                                      "so this reference cites a part of it as though it were a chapter in an "
                                      "edited volume.")
    assert _panel_statement("Cited in two places.") == "Cited in two places."


def test_author_name_order_differences_are_not_record_conflicts():
    # Academic Article, 2026-09-30: "Le-Ha, P." / "Le-Ha Phan"; "Nguyen, P.A." / "Anh Nguyen Phuong".
    from app.services.evidence_report import _same_people_other_name_order
    assert _same_people_other_name_order({"submitted_value": "Le-Ha, P", "located_value": "Le-Ha Phan"})
    assert _same_people_other_name_order({"submitted_value": "Nguyen, P.A., Nguyen, T.L.T., & Nguyen, H.N.T",
                                          "located_value": "Anh Nguyen Phuong, Loan Nguyen T. Thanh"})
    assert not _same_people_other_name_order({"submitted_value": "Singer, B", "located_value": "Mario Falsetto"})


def test_different_in_text_forms_are_not_a_shared_author_and_year():
    # Academic Article, 2026-09-30: "Yang et al., 2019" and "Yang & Wang, 2019".
    from types import SimpleNamespace
    from app.services.reference_consistency import apa_in_text_form
    many = SimpleNamespace(author="Yang, M., O’Sullivan, P.S., Irby, D.M., Chen, Z., Lin, C., & Lin, C", year="2019")
    two = SimpleNamespace(author="Yang, Y., & Wang, X", year="2019")
    same = SimpleNamespace(author="Yang, Y., & Wang, X", year="2019")
    assert apa_in_text_form(many) == "yang et al. 2019" and apa_in_text_form(two) == "yang & wang 2019"
    assert apa_in_text_form(two) == apa_in_text_form(same)


def test_broad_multi_source_citations_carry_no_topical_mismatch():
    # Academic Article, 2026-09-30.
    from app.services import report_layers
    member = {"coverage_level": "abstract_only"}
    assert report_layers.topical_mismatch(member, {"members": [{}, {}, {}], "student_text": "x"}) is False


def test_review_fixes_2026_09_30():
    from types import SimpleNamespace
    from app.services.evidence_report import citation_after_punctuation, _render_summary_instances
    from app.services.judgment_coaching import _ADVICE
    from app.services.source_resolver import _registration_bound_doi
    # A block quotation's citation belongs after its final punctuation.
    block = {"citation_marker": "(Schatz, 1981, p. 12)", "claim_type": "quotation",
             "student_text": "Genre films of order celebrate the community. (Schatz, 1981, p. 12)"}
    assert citation_after_punctuation(block) is False
    assert citation_after_punctuation({**block, "claim_type": "paraphrase"}) is True
    # A stored summary naming passages shows them as text, not dead links.
    html = _render_summary_instances({"instances": [{"type": "passage", "number": 1, "target": "passage-panel-1"},
                                                    {"type": "passage", "number": 3, "target": "passage-panel-3"}]})
    assert html == "passages 1, 3" and "Instance" not in html
    # Descriptive openings are not advice; imperatives are.
    assert not _ADVICE.search("Focus groups in the source reported mixed views.")
    assert _ADVICE.search("The source differs. Check whether the sample is the same.")
    # A reused source passes on only a registration-bound or supplied DOI.
    record = lambda evidence: SimpleNamespace(canonical_work=SimpleNamespace(doi="10.1234/abc"),
                                              validation_evidence={"canonical_work": {"identity_evidence": evidence}})
    assert _registration_bound_doi(record([{"provider": "crossref", "doi": "10.1234/ABC"}]), None) == "10.1234/abc"
    assert _registration_bound_doi(record([{"provider": "core", "doi": "10.1234/abc"}]), None) is None
    assert _registration_bound_doi(record([]), "https://doi.org/10.1234/abc") == "10.1234/abc"
