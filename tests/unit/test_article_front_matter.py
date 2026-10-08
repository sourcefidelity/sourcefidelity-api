"""Owner review 2026-10-07 of the owner's article: keywords, headings and a
page's licence footer are not paper text, a source's title block is not
evidence, and a source's own title words are not patchwriting."""
from types import SimpleNamespace

from app.services.facet_evidence_judgment import _passage_sentences
from app.services.patchwriting_report import _only_title_words
from app.services.text_extractor import _strip_publication_footer, isolate_pdf_headings


def test_a_page_licence_footer_is_not_paper_text():
    page = "\n".join(["through the last body line, they have", "disrupted it.",
                      "Harbour Studies, April 2026 https://doi.org/10.1000/hs.2026.13",
                      "Published open access under a CC BY licence.",
                      "https://creativecommons.org/licenses/by/4.0/"])
    assert _strip_publication_footer(page) == "through the last body line, they have\ndisrupted it."
    body = "Ports use a Creative Commons licence for maps (Ames, 2020).\nMore prose follows here."
    assert _strip_publication_footer(body) == body


def test_keywords_and_the_heading_after_them_are_not_the_first_sentence():
    text = ("to speed up the port.\nKeywords: harbour cranes, port automation\nBackground\n"
            "The development of port automation has changed harbours (Ames, 2025).")
    paragraphs = [p.strip() for p in isolate_pdf_headings(text).split("\n\n") if p.strip()]
    assert paragraphs[-1].startswith("The development of port automation")
    assert "Keywords: harbour cranes, port automation" in paragraphs and "Background" in paragraphs


def test_a_sources_title_block_before_its_abstract_is_not_a_sentence():
    text = ("Port automation in harbours: Evidence from forty ports\nA. Ames a,*\na Harbour University, 1 Quay Road\n"
            "A B S T R A C T\nThe release of new cranes prompted a massive uptake of automation across ports. "
            "Ports then regulated its use.")
    sentences = [s.text for s in _passage_sentences(SimpleNamespace(text=text, passage_id="p1"))]
    assert sentences == ["The release of new cranes prompted a massive uptake of automation across ports.",
                         "Ports then regulated its use."]


def _region(text):
    return {"text": text}


def test_a_sources_own_title_words_are_not_patchwriting():
    text = "of port crane automation (PCA) has greatly changed harbour logistics"
    title = "Port crane automation in harbour logistics"
    spans = [(0, 23), (25, 28), (53, 69)]   # "of port crane automation", "PCA", "harbour logistics"
    assert _only_title_words(_region(text), 0, spans, title)
    borrowed = "These cranes produce feelings of safety, calm, and trust rather than haste or alarm"
    spans = [(6, 20), (33, 39), (51, 56), (78, 83)]   # cranes produce, safety, trust, alarm
    assert not _only_title_words(_region(borrowed), 0, spans, "Port cranes and the status quo")


def test_the_report_passes_the_reference_title_to_the_patchwriting_filter():
    # The report's reference view carries the title at its top level (2026-10-07).
    from app.services.patchwriting_report import build_passages
    text = "of port crane automation (PCA) has greatly changed harbour logistics"
    finding = {"kind": "close_paraphrase", "student_region": {"paper_start": 0, "paper_end": len(text), "text": text},
               "source_sentences": [{"text": "Port crane automation (PCA) across harbour logistics."}],
               "student_matched_spans": [{"paper_start": a, "paper_end": b} for a, b in ((0, 24), (26, 29), (51, 68))]}
    block = {"sources": {"r1": {"status": "compared", "findings": [finding]}}}
    view = {"title": "Port crane automation in harbour logistics"}
    assert build_passages(block, lambda rid: view) == []


def test_a_dummy_it_opening_needs_no_antecedent():
    # Citation 10 of the owner's article, 2026-10-08.
    from app.services.antecedent_resolver import _EXPLETIVE_IT
    for text in ("It is difficult for port cranes to lift heavy loads (Ames, 2020).",
                 "It is clear that ports automate (Ames, 2020).", "It seems that cranes rust (Ames, 2020)."):
        assert _EXPLETIVE_IT.match(text)
    for text in ("It is a powerful crane (Ames, 2020).", "It reduces delays (Ames, 2020)."):
        assert not _EXPLETIVE_IT.match(text)


def test_propositions_sharing_words_are_underlined_once(monkeypatch):
    # The owner's article, 2026-10-08: the smaller keeps the shared words.
    from app.services import judgment_layer
    calls = []
    monkeypatch.setattr(judgment_layer, "claim_span_rectangles",
                        lambda citation, ranges, words: (calls.append(ranges) or [{"page_index": 0}], "exact"))
    small = {"group": "a", "ranges": [(0, 60)]}
    large = {"group": "b", "ranges": [(0, 40), (70, 200)]}
    judgment_layer._separate_contained([small, large], {}, {})
    # Underlined only where it adds words; bolded whole in the window (citation 10a, 2026-10-08).
    assert calls == [[(70, 200)]] and large["rects"] == [{"page_index": 0}]
    assert "display_ranges" not in large and "display_ranges" not in small
    inner = {"group": "c", "ranges": [(0, 60)]}
    outer = {"group": "d", "ranges": [(0, 200)]}
    judgment_layer._separate_contained([inner, outer], {}, {})
    assert outer["display_ranges"] == [(60, 200)]   # containing: bolded on its added words (2026-09-29)


def test_the_no_passage_line_is_tagged_for_the_judgment_window():
    from app.services.evidence_report import _render_member
    source = {"author": "Ames, A.", "year": "2020", "title": "Ports", "raw_reference": "Ames, A. (2020). Ports."}
    member = {"reference_id": "r1", "availability": "No clearly relevant passage was found for this source's part of "
              "the citation. Its title, study setting or metadata may be relevant; check the source manually.",
              "source": source, "coverage_level": "full_text",
              "verification_report_id": "00000000-0000-0000-0000-000000000001"}
    assert "data-no-passage-note" in _render_member(member)
    assert "data-no-passage-note" not in _render_member({**member, "availability": "Only part of the source is available."})


def test_a_dummy_it_part_does_not_ask_the_judge_for_a_referent():
    from app.services.candidate_relationship_judgment import _requires_parent_context
    text = "It is difficult for port cranes to lift loads because they rust (Ames, 2020)."
    assert not _requires_parent_context(text, [(0, 63)])
    assert _requires_parent_context("It reduces delays (Ames, 2020).", [(0, 17)])


def test_underline_boxes_do_not_bridge_other_words_on_the_line():
    from app.services.judgment_layer import _merge_per_line
    words = [(0, (10, 0, 20, 10, "a")), (0, (22, 0, 30, 10, "b")), (0, (60, 0, 70, 10, "e"))]
    assert len(_merge_per_line(words, [0, 1, 4])) == 2
    assert len(_merge_per_line(words)) == 1
