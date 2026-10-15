"""Citation projection: which citations the report shows, numbered how, and
which references count as cited (2026-10-15: the rules tested together)."""
from app.services.citation_projection import heading_citation, project_citations, uncited_reference_ids
from app.services.paper_extraction import PaperExtractionArtifact
from app.services.report_references import reference_numbers
from app.services.schemas import CitationMarkerMember
from app.services.verification_evidence import ClaimEvidence

EXTRACTION = PaperExtractionArtifact(paper_version_id="p", citation_format="APA")


def _claim(claim_id, text, start, *markers, marker_type="parenthetical"):
    built = [CitationMarkerMember(text=marker, local_start=text.index(marker),
                                  local_end=text.index(marker) + len(marker), reference_ids=[ref],
                                  marker_type="parenthetical")
             for marker, ref in markers]
    return ClaimEvidence(claim_id=claim_id, paper_version_id="p", text=text, passage_start=start,
                         passage_end=start + len(text), citation_marker=markers[0][0],
                         citation_markers=built, citation_marker_type=marker_type,
                         reference_ids=[ref for _, ref in markers])


def _group(claim, *references):
    members = [{"reference_id": ref, "source": {"title": f"Work {ref}"}} for ref in references]
    return claim, {"members": [{"reference_id": ref} for ref in references]}, members, False


def _project(*groups):
    return project_citations(EXTRACTION, {}, list(groups), {}, {})


def _uncited(citations, headed, bibliography, paper_key=""):
    view = {"citations": citations, "heading_reference_ids": headed,
            "bibliography": [{"reference_id": ref, "source": {"title": f"Work {ref}"}} for ref in bibliography]}
    return uncited_reference_ids(view, reference_numbers(view), paper_key)


def test_a_heading_naming_a_work_is_not_a_citation():
    heading = {'student_text': 'Classical Hollywood: The Day the Earth Stood Still (Wise, 1951)',
               'citation_marker': '(Wise, 1951)'}
    assert heading_citation(heading)
    assert heading_citation({'student_text': 'Post-Classical Hollywood: Star Trek: The Motion Picture (Wise, 1979)',
                             'citation_marker': '(Wise, 1979)'})
    sentence = {'student_text': 'The film was released across 2,000 theatres (Bond, 2020).', 'citation_marker': '(Bond, 2020)'}
    no_stop = {'student_text': 'the studio controlled every stage of release (Bond, 2020)', 'citation_marker': '(Bond, 2020)'}
    assert not heading_citation(sentence) and not heading_citation(no_stop)
    assert heading_citation({'student_text': "Killer's Kiss (1955): Classical Hollywood Film Analysis",
                             'citation_marker': "Killer's Kiss (1955)"})


def test_a_clause_cut_at_its_citation_is_kept_and_its_reference_is_cited():
    # Owner review 2026-10-14 (Harlow): the title-cased words before the one
    # marker alone read as a heading, which dropped the citation.
    text = "In Wife vs. Secretary (Brown, 1936), she played a secretary suspected of an affair."
    citations, headed = _project(_group(_claim("c1", text, 100, ("(Brown, 1936)", "brown")), "brown"))
    assert [(c["student_text"], c["citation_number"], bool(c.get("clause_of"))) for c in citations] == [
        ("In Wife vs. Secretary (Brown, 1936)", 1, True)]
    assert headed == []
    assert _uncited(citations, headed, ["brown"]) == []


def test_a_reference_cited_only_by_a_heading_is_neither_a_citation_nor_uncited():
    heading = "Classical Hollywood: The Day the Earth Stood Still (Wise, 1951)"
    sentence = "Production became project-oriented across the studios (Ames, 2020)."
    citations, headed = _project(_group(_claim("c1", heading, 0, ("(Wise, 1951)", "wise")), "wise"),
                                 _group(_claim("c2", sentence, 80, ("(Ames, 2020)", "ames")), "ames"))
    assert [c["claim_id"] for c in citations] == ["c2"] and citations[0]["citation_number"] == 1
    assert headed == ["wise"]
    assert _uncited(citations, headed, ["wise", "ames", "lone"]) == ["lone"]


def test_a_split_sentence_is_numbered_in_paper_order_after_earlier_citations():
    first = "Ports automate their cranes (Ames, 2025)."
    split = "Cranes are slow in storms (Bell, 2022), and they break often (Dunn, 2023)."
    citations, _ = _project(_group(_claim("late", split, 200, ("(Bell, 2022)", "bell"), ("(Dunn, 2023)", "dunn")),
                                   "bell", "dunn"),
                            _group(_claim("early", first, 10, ("(Ames, 2025)", "ames")), "ames"))
    assert [(c["claim_id"], c["citation_number"], [m["reference_id"] for m in c["members"]]) for c in citations] == [
        ("early", 1, ["ames"]), ("late", 2, ["bell"]), ("late", 3, ["dunn"])]
    assert citations[1]["paper_character_start"] == 200 and citations[2]["student_text"].endswith("(Dunn, 2023).")
