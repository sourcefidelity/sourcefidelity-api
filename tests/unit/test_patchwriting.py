"""patchwriting-v3: deterministic unquoted-wording and close-paraphrase comparison."""
from types import SimpleNamespace

import pytest

from app.services import patchwriting as pw
from app.services.patchwriting import (
    POLICY_VERSION,
    StudentStatement,
    build_source_index,
    detect_patchwriting,
    sentence_spans,
    source_sentences_from_pages,
    statement_from_claim,
)

SOURCE_PAGE = (
    "Early studies focused on television. "
    "Social media platforms have fundamentally transformed the ways in which adolescents "
    "communicate with their peers and families. "
    "The rapid expansion of online networks has altered how young people form friendships "
    "and maintain close relationships over long distances. "
    "However, researchers disagree about the long-term effects of these changes on wellbeing."
)


def _index(*pages):
    page_objects = [
        SimpleNamespace(index=i + 2, label=str(i + 3), text=text, structural_spans=())
        for i, text in enumerate(pages or (SOURCE_PAGE,))
    ]
    return build_source_index(source_sentences_from_pages(page_objects),
                              representation_id="rep-1", content_sha256="c" * 64)


def _statement(text, paper_start=5_000, claim_type="paraphrase", marker=None):
    markers = ()
    if marker:
        start = text.index(marker)
        markers = ((start, start + len(marker)),)
    return StudentStatement(text=text, segments=((0, len(text), paper_start),),
                            marker_spans=markers, claim_type=claim_type, claim_id="claim-1")


def test_unquoted_verbatim_run_is_flagged_with_exact_spans():
    text = ("Researchers observe that social media platforms have fundamentally transformed "
            "the ways in which adolescents communicate (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index())
    assert result.policy_version == POLICY_VERSION == "patchwriting-v3"
    assert result.status == "compared"
    assert result.decision_applied is False
    kinds = [finding.kind for finding in result.findings]
    assert kinds[0] == "unquoted_verbatim"
    finding = result.findings[0]
    span = finding.student_matched_spans[0]
    assert span.text == ("social media platforms have fundamentally transformed the ways in which "
                         "adolescents communicate")
    assert finding.measures.qualifying_run_words >= 8
    source_page = SOURCE_PAGE
    matched = finding.source.matched_spans[0]
    assert source_page[matched.absolute_start:matched.absolute_end].casefold() == span.text.casefold()
    assert finding.source.sentence_key == (
        f"2:{finding.source.absolute_start}:{finding.source.absolute_end}")
    assert source_page[finding.source.absolute_start:finding.source.absolute_end] == finding.source.text
    assert finding.source.page_label == "3"


def test_spans_map_to_exact_paper_offsets():
    paper = "Intro sentence here. " * 10
    text = ("As noted, social media platforms have fundamentally transformed the ways in which "
            "adolescents communicate with their peers (Smith, 2020).")
    paper += text
    passage_start = paper.index(text)
    statement = statement_from_claim({
        "text": text,
        "passage_start": passage_start,
        "citation_markers": [{"text": "Smith, 2020", "local_start": text.index("Smith, 2020"),
                              "local_end": text.index("Smith, 2020") + len("Smith, 2020")}],
        "claim_type": "paraphrase",
        "claim_id": "c",
        "source_segments": [],
    })
    result = detect_patchwriting(statement, _index())
    finding = result.findings[0]
    for span in finding.student_matched_spans + [finding.student_sentence]:
        assert paper[span.paper_start:span.paper_end] == span.text == text[span.local_start:span.local_end]


def test_same_wording_inside_quotation_marks_is_not_flagged():
    for opening, closing in (('"', '"'), ("“", "”")):
        text = (f"Smith argues that {opening}social media platforms have fundamentally transformed "
                f"the ways in which adolescents communicate with their peers{closing} (2020).")
        result = detect_patchwriting(_statement(text), _index())
        assert result.findings == []
        assert any(span.kind == "double_quotation" for span in result.excluded_student_spans)
        assert result.coverage.student_words_excluded_quotation >= 14


def test_quotation_does_not_bridge_a_verbatim_run():
    # Removing the quoted words must not make the words around them contiguous.
    text = ('Social media platforms have "fundamentally transformed" the ways in which adolescents '
            "talk online (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index())
    assert all(f.kind != "unquoted_verbatim" for f in result.findings)


def test_block_quotation_without_marks_is_not_assessed():
    text = ("Social media platforms have fundamentally transformed the ways in which adolescents "
            "communicate with their peers and families.")
    result = detect_patchwriting(_statement(text, claim_type="quotation"), _index())
    assert result.status == "not_assessed"
    assert result.reason == "no_unquoted_student_wording"
    assert result.excluded_student_spans[0].kind == "block_quotation"


def test_close_paraphrase_with_synonym_swaps_and_kept_order_is_flagged():
    text = ("The fast expansion of online networks has changed how young people form friendships "
            "and keep close relationships over long distances (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index())
    assert [f.kind for f in result.findings][:1] == ["close_paraphrase"]
    measures = result.findings[0].measures
    assert measures.structural_retention >= pw.PARAPHRASE_MIN_RETAINED
    assert measures.matched_content_words >= pw.PARAPHRASE_MIN_MATCHED
    assert measures.aligned_substitutions >= 2
    matched = {s.text for s in result.findings[0].student_matched_spans}
    assert "fast" not in matched and "online networks" in matched


def test_genuine_paraphrase_with_different_structure_is_not_flagged():
    text = ("Friendships among teenagers increasingly survive geographic separation, since "
            "digital services let them stay in touch cheaply (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index())
    assert result.status == "compared"
    assert result.findings == []


@pytest.mark.parametrize("text", [
    "The results show that crime fell in the United States (Smith, 2020).",
    "In the United States, the results show that on the other hand (Smith, 2020).",
    "As a result of this, in the context of the study, it is important to note (Smith, 2020).",
])
def test_conventional_short_phrases_are_not_flagged(text):
    source = ("In the United States, the results show that crime fell sharply. "
              "As a result of this, in the context of the study, it is important to note the limits. "
              "On the other hand, other regions differed.")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index(source))
    assert result.findings == []


def test_citation_marker_is_excluded():
    source = "Smith and Jones 2020 reported that attendance rose among rural students after reform."
    text = ("Attendance rose among rural students after reform "
            "(Smith and Jones 2020 reported that attendance).")
    marker = "Smith and Jones 2020 reported that attendance"
    result = detect_patchwriting(_statement(text, marker=marker), _index(source))
    assert result.coverage.student_words_excluded_marker == 7
    assert any(span.kind == "citation_marker" for span in result.excluded_student_spans)
    for finding in result.findings:
        for span in finding.student_matched_spans:
            assert span.local_end <= text.index(marker)


def test_statement_from_claim_uses_legacy_marker_when_unique():
    statement = statement_from_claim({"text": "Rates rose (Lee, 2019).", "passage_start": 10,
                                      "citation_marker": "(Lee, 2019)", "claim_type": "paraphrase"})
    assert statement.marker_spans == ((11, 22),)
    assert statement.segments == ((0, 23, 10),)


def test_long_and_odd_input_does_not_raise():
    long_statement = ("word " * 30_000) + "(Smith, 2020)."
    index = _index("x" * 100_000 + ". " + ("alpha beta gamma " * 5_000))
    result = detect_patchwriting(StudentStatement(text=long_statement), index)
    assert result.status in {"compared", "not_assessed"}
    assert "statement_truncated" in result.limitations
    assert "paper_offsets_unavailable" in result.limitations
    for odd in ("", "   ", '"unclosed quotation with many words here', "ﬁ​\x00�"):
        assert detect_patchwriting(StudentStatement(text=odd), index).status in {"compared", "not_assessed"}
    empty = build_source_index([])
    assert detect_patchwriting(StudentStatement(text="Some words here."), empty).reason == "no_source_text"
    assert detect_patchwriting(None, index).status == "not_assessed"  # type: ignore[arg-type]


def test_hyphenated_line_break_and_ligature_are_normalized():
    source = ("The inter-\nnational ﬁnancial community quickly rejected the proposed monetary "
              "stabilization agreement in early spring.")
    text = ("The international financial community quickly rejected the proposed monetary "
            "stabilization agreement (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index(source))
    assert result.findings and result.findings[0].kind == "unquoted_verbatim"


def test_sentence_boundaries_match_the_report_sentence_splitter():
    from app.services.facet_evidence_judgment import _passage_sentences

    text = ("D. Ricardo argued this. Prices rose in 1990! Did wages follow? \"Yes,\" said A. Smith. "
            "(Later) work disagreed.\n\nA new paragraph starts here. 3 cases remained.")
    expected = [(s.passage_start, s.passage_end)
                for s in _passage_sentences(SimpleNamespace(text=text, passage_id="p"))]
    assert sentence_spans(text) == expected


def test_reference_list_sentences_are_labelled():
    page = "Body sentence about teachers.\nReferences\nSmith, J. (2020). A long title about teachers."
    sentences = source_sentences_from_pages([SimpleNamespace(index=0, label=None, text=page,
                                                             structural_spans=())])
    assert sentences[0].role == "body"
    assert sentences[-1].role == "reference_list"


def test_structural_role_needs_majority_overlap():
    page = "A long body sentence about reading habits among rural teenagers today. Header"
    furniture = SimpleNamespace(start=page.index("Header"), end=len(page), role="page_furniture")
    sentences = source_sentences_from_pages([SimpleNamespace(index=0, label=None, text=page,
                                                             structural_spans=(furniture,))])
    assert [s.role for s in sentences] == ["body", "page_furniture"]


OWNER_SOURCE = (
    "In view of Fokkema’s definition of postmodern character which is “in the most general sense, "
    "caught up in power relations [and which is c]onstituted by history, its own paranoid beliefs, "
    "other narratives, language, or Foucauldian discourse, its autonomy is endangered or lost” and "
    "whose circumstances “may testify to a deteriorating situation, resulting in madness or in the "
    "increasing power of a public identity” (1991: 184), Alice’s visit to Wonderland signifies a "
    "postmodern crisis pertaining to power, identity and the nature of reality. "
    "Her growth through the story is gradual and uneven."
)
OWNER_STATEMENT = (
    "The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, and the "
    "nature of reality, allowing her to grow from a confused girl to a brave and confident hero "
    "(Flegar & Wertag, 2015)."
)


def test_owner_example_flags_the_whole_clause_as_close_paraphrase():
    result = detect_patchwriting(_statement(OWNER_STATEMENT, marker="Flegar & Wertag, 2015"),
                                 _index(OWNER_SOURCE))
    finding = result.findings[0]
    assert finding.kind == "close_paraphrase"
    assert finding.student_region.text == (
        "The visit of Alice to Wonderland marks a postmodern crisis related to power, identity, "
        "and the nature of reality")
    m = finding.measures
    assert (m.transpositions, m.aligned_substitutions) == (1, 2)   # visit/Alice; marks, related
    assert m.qualifying_run_words == 8                             # the verbatim tail is inside
    assert finding.source.page_label == "3"
    region = OWNER_SOURCE[finding.source.matched_spans[0].absolute_start:
                          finding.source.matched_spans[-1].absolute_end]
    assert region.startswith("Alice") and region.endswith("reality")


@pytest.mark.parametrize("source, text", [
    # reordered clause with inflection changes and a dropped article
    ("Teachers who received sustained coaching reported greater confidence in managing "
     "disruptive classroom behaviour.",
     "Teachers receiving sustained coaching reported more confidence in managing disruptive "
     "behaviour in the classroom (Lee, 2019)."),
    # synonym swaps in the same slots, same clause order
    ("Rising sea levels threaten coastal communities by accelerating erosion and contaminating "
     "freshwater supplies.",
     "Increasing sea levels endanger coastal communities by speeding erosion and polluting "
     "freshwater supplies (Lee, 2019)."),
    # clause embedded in a longer student sentence
    ("The policy reduced unemployment among young workers but widened regional wage inequality.",
     "Although it was popular, the policy lowered unemployment among young workers but widened "
     "regional wage inequality, which later governments tried to address (Lee, 2019)."),
])
def test_realistic_patchwriting_variants_flag(source, text):
    result = detect_patchwriting(_statement(text, marker="Lee, 2019"), _index(source))
    assert result.findings, result.sentence_comparisons
    assert result.findings[0].kind in {"close_paraphrase", "unquoted_verbatim"}


@pytest.mark.parametrize("source, text", [
    ("Teachers who received sustained coaching reported greater confidence in managing "
     "disruptive classroom behaviour.",
     "Long-term mentoring appears to make instructors feel better equipped when pupils act out "
     "(Lee, 2019)."),
    ("Rising sea levels threaten coastal communities by accelerating erosion and contaminating "
     "freshwater supplies.",
     "For towns on the shore, the ocean's slow advance means land loss and saltier drinking "
     "water (Lee, 2019)."),
    ("The policy reduced unemployment among young workers but widened regional wage inequality.",
     "Regional pay gaps grew under the policy, even as fewer young people were out of work "
     "(Lee, 2019)."),
])
def test_genuine_paraphrases_do_not_flag(source, text):
    result = detect_patchwriting(_statement(text, marker="Lee, 2019"), _index(source))
    assert result.findings == []


def test_stemming_and_possessives_are_conservative():
    assert pw.stem("signifies") == pw.stem("signify") == pw.stem("signified")
    assert pw.stem("related") == pw.stem("relate") == pw.stem("relates")
    assert pw.stem("stopped") == pw.stem("stop")
    assert pw._normalize_word("Alice’s") == pw._normalize_word("Alice's") == "alice"
    assert pw.stem("analysis") == "analysis" and pw.stem("status") == "status"


def test_body_comparison_labels_citation_statements_and_excludes_parentheticals():
    source_a = _index(OWNER_SOURCE)
    source_a.representation_id = "rep-a"
    other = _index("Rising sea levels threaten coastal communities by accelerating erosion and "
                   "contaminating freshwater supplies.")
    other.representation_id = "rep-b"
    body = ("Introduction to the essay. " + OWNER_STATEMENT + " Later, increasing sea levels "
            "endanger coastal communities by speeding erosion and polluting freshwater supplies (Lee 12).")
    start = body.index(OWNER_STATEMENT)
    marker = OWNER_STATEMENT.index("Flegar & Wertag, 2015")
    statement = pw.CitationStatement(
        claim_id="claim-a", paper_start=start, paper_end=start + len(OWNER_STATEMENT),
        cited_representation_ids=("rep-a",),
        marker_spans=((start + marker, start + marker + len("Flegar & Wertag, 2015")),))
    result = pw.detect_in_body(body, [source_a, other], statements=[statement])
    by_rep = {f.source_representation_id: f for f in result.findings}
    assert by_rep["rep-a"].label == "citation_statement_vs_cited_source"
    assert by_rep["rep-a"].claim_ids == ["claim-a"]
    assert by_rep["rep-b"].label == "body_sentence"
    region = by_rep["rep-a"].student_region
    assert body[region.paper_start:region.paper_end] == region.text
    assert any(s.kind == "citation_parenthetical" for s in result.excluded_student_spans)
    assert [c.representation_id for c in result.coverage.sources] == ["rep-a", "rep-b"]
    assert result.coverage.student_sentences == 3


def test_body_comparison_never_raises():
    assert pw.detect_in_body(None, [None]).status == "not_assessed"  # type: ignore[list-item]
    assert pw.detect_in_body("x" * (pw.MAX_BODY_CHARACTERS + 10), [_index()]).limitations[0] == "body_truncated"


def test_v1_results_remain_readable():
    pw.PatchwritingResult.model_validate({
        "policy_version": "patchwriting-v1", "status": "compared",
        "coverage": {"comparison_scope": "full_text"}})


def test_bibliographic_source_sentences_and_titles_are_measured_but_not_flagged():
    source = ("Body sentence about fandom and participation in online spaces today.\n"
              "27 Henry Jenkins, Convergence Culture: Where Old and New Media Collide "
              "(New York: New York University Press, 2006).\n"
              "References\nJenkins, H. (2006). Convergence culture: Where old and new media collide. NYU Press.")
    text = ("In 2006 Henry Jenkins redefined the term in Convergence Culture: Where Old and New Media "
            "Collide to describe tightly integrated narratives.")
    result = detect_patchwriting(_statement(text), _index(source))
    assert result.findings == []
    # Since 2026-09-30 a title repeated from the source does not count toward a
    # finding at all, so the later suppression reasons need not be reached.
    reasons = {c.best_measures.suppressed_reason for c in result.sentence_comparisons if c.best_measures}
    assert reasons <= {None, "source_role:citation_notes", "source_role:reference_list", "title_or_name"}


def test_source_wording_quoted_in_the_source_is_measured():
    source = ("For Jenkins, ‘a transmedia story unfolds across multiple media\nplatforms, with each text "
              "making a distinctive contribution to the whole’.")
    text = "A transmedia story unfolds across multiple media platforms, with each text making its own mark."
    result = detect_patchwriting(_statement(text), _index(source))
    assert result.findings and result.findings[0].measures.source_quoted_share == 1.0


def test_one_finding_per_sentence_and_source():
    sentence = "Rising sea levels threaten coastal communities by accelerating erosion and contaminating freshwater supplies. "
    source = sentence + "Other material follows here. " + sentence
    text = ("Increasing sea levels endanger coastal communities by speeding erosion and polluting "
            "freshwater supplies (Lee, 2019).")
    result = detect_patchwriting(_statement(text, marker="Lee, 2019"), _index(source))
    assert len(result.findings) == 1


GARDEN_SOURCE = ("A central feature of the garden is its terrace, which is usually reached through the gates, "
                 "using the hedge as a decorative border.")


def test_v3_five_matched_words_with_same_slot_swaps_is_close_paraphrase():
    # Owner calibration 2026-09-29: synonym swaps in the source's own slots keep
    # its structure; five matched words suffice when the swaps are counted.
    text = ("A main aspect of the garden is its terrace, which is often entered through gates, "
            "using the hedge as an ornamental edge (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index(GARDEN_SOURCE))
    [finding] = result.findings
    measures = finding.measures
    assert finding.kind == "close_paraphrase"
    assert measures.matched_content_words == pw.PARAPHRASE_MIN_MATCHED == 5
    assert measures.student_density < 0.8 and measures.aligned_substitutions >= 2
    assert measures.structural_retention >= pw.PARAPHRASE_MIN_RETAINED


def test_v3_four_matched_words_are_not_enough():
    text = ("A main aspect of the garden is its terrace, which is often entered through doors, "
            "using the hedge as an ornamental edge (Smith, 2020).")
    result = detect_patchwriting(_statement(text, marker="Smith, 2020"), _index(GARDEN_SOURCE))
    assert result.status == "compared" and result.findings == []


def test_a_title_repeated_from_the_source_is_not_patchwriting():
    # Franchise 1, 2026-09-30: a long film title plus two ordinary words.
    source = ("Of the eight, River Dreams and the Northern Lights: Part 2 made the most of any film in the series "
              "and closed the story.")
    text = ("By the time River Dreams and the Northern Lights - Part 2 was released, the eight-film series "
            "had changed the studio (Lake, 2023).")
    result = detect_patchwriting(_statement(text, marker="Lake, 2023"), _index(source))
    assert result.findings == []
