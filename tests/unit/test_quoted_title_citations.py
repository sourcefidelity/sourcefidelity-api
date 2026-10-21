"""Owner decision 2026-10-12: a quoted title followed by a bare year,
"Section 7 of 'Title' (2002)", cites the reference with that title, whatever
its author; a different year still links and the difference is kept."""
from types import SimpleNamespace

from app.services.citation_extractor import extract_citations, quoted_title_references
from app.services.reference_consistency import assess_reference_consistency
from app.services.schemas import InTextCitation, ParsedReference


def _ref(rid, raw, author="", year="", title=""):
    return ParsedReference(reference_id=rid, raw_ref=raw, author=author, year=year, title=title,
                           citation_key=rid)


def _media_refs():
    return [_ref("r1", "Northfield Ministry of Media. (2002). The interim measures on the supervision of coastal "
                 "publishing. https://example.test/a", author="Northfield Ministry of Media", year="2002",
                 title="The interim measures on the supervision of coastal publishing"),
            _ref("r2", "Northfield Ministry of Media. (2016). Rules on network audio services. https://example.test/b",
                 author="Northfield Ministry of Media", year="2016", title="Rules on network audio services"),
            _ref("r3", "Lowe, P., & Marsh, T. (2016). Streaming and the state. Journal of Media Policy, 4(2), 1-20.",
                 author="Lowe, P., & Marsh, T", year="2016", title="Streaming and the state"),
            _ref("r4", "Northfield Archive Office. (2011). Measures for broadcast archives. https://example.test/c",
                 author="Northfield Archive Office", year="2011", title="Measures for broadcast archives")]


def _title_links(body, refs):
    return {c.citation_marker: (c.link_status, c.reference_ids, sorted(c.candidate_reference_ids), c.link_differences)
            for c in extract_citations(body, refs, format_hint="apa") if c.marker_type == "narrative"}


def test_a_quoted_title_with_a_bare_year_links_in_every_quote_style():
    for opening, closing in [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”"), ("‘", "'"), ("“", '"')]:
        body = f"Every outlet registers under {opening}Rules on Network Audio Services{closing} (2016)."
        assert list(_title_links(body, _media_refs()).values()) == [("linked", ["r2"], [], [])], (opening, closing)


def test_a_corporate_author_reference_matches_a_close_title_variant():
    body = ("Section 4 of 'The Interim Measures on the Supervision of Coastal Publication' (2002) "
            "requires every publisher to hold a licence.")
    assert list(_title_links(body, _media_refs()).values()) == [("linked", ["r1"], [], [])]
    [citation] = extract_citations(body, _media_refs(), format_hint="apa")
    # The quoted title names the work; it is not quoted wording.
    assert citation.claim_type == "paraphrase"


def test_a_quoted_title_with_a_different_year_links_and_keeps_the_difference():
    body = "Regulators also cite ‘Measures for Broadcast Archives' (2010) when reviewing appeals."
    assert list(_title_links(body, _media_refs()).values()) == [("linked", ["r4"], [], ["year:2011"])]


def test_two_similar_titles_stay_ambiguous_and_no_title_match_links_nothing():
    refs = [_ref("r1", "Northfield Ministry of Media. (2016). Rules on network audio service.",
                 author="Northfield Ministry of Media", year="2016", title="Rules on network audio service"),
            _ref("r2", "Eastvale Press Office. (2015). Rules on network audio services.",
                 author="Eastvale Press Office", year="2015", title="Rules on network audio services")]
    assert list(_title_links('Every outlet registers under "Rules on Network Audio Servicing" (2016).',
                             refs).values()) == [("ambiguous", [], ["r1", "r2"], [])]
    # The same title and year as exactly one of them decides it.
    assert [r.reference_id for r in quoted_title_references("Rules on Network Audio Services", "2015", refs)[0]] == ["r2"]
    assert _title_links('Every outlet registers under "Rules on Broadcast Licensing" (2016).', refs) == {}
    # A short title is never matched by a variant.
    assert quoted_title_references("Audio Service", "2016", [_ref("r9", "x", title="Audio Services", year="2016")]) == ([], [])


def test_title_and_year_legislation_markers_are_unchanged():
    refs = [_ref("r1", "Measures for the Licensing of Coastal Radio Receivers 1990. http://example.test/law", year="1990")]
    linked = {c.citation_marker: c.reference_ids for c in extract_citations(
        "Users must hold licences (Measures for the Licensing of Coastal Radio Receivers 1990, reg 4).", refs,
        format_hint="apa") if c.link_status == "linked"}
    assert list(linked.values()) == [["r1"]]
    assert _title_links('This essay takes the "Measures for the Licensing of Coastal Radio Receivers (1990)" as its case.',
                        refs) == {}
    # The authorless entry's opening title also matches a quoted title.
    assert list(_title_links("Section 4 of 'Measures for the Licensing of Coastal Radio Receivers' (1990) applies.",
                             refs).values()) == [("linked", ["r1"], [], [])]


def test_a_rejected_low_confidence_detection_does_not_make_a_reference_cited():
    refs = _media_refs()
    rejected = InTextCitation(text="Regulators review appeals in time.", citation_marker="(2016)", reference_ids=["r2"],
                              link_status="linked", passage_start=0, passage_end=34, confidence="low",
                              drop_reason="hallucinated_key")
    result = assess_reference_consistency(paper_version_id="p", citation_format="apa", references=refs,
                                          citations=[rejected])
    uncited = {rid for f in result.findings if f.finding_type == "reference_not_linked_in_extracted_citations"
               for rid in f.reference_ids}
    assert "r2" in uncited
    # A structural link set aside only to regroup its sentence still counts.
    regrouped = rejected.model_copy(update={"confidence": "high",
                                            "drop_reason": "multi_marker_citation_unit_requires_grouping"})
    result = assess_reference_consistency(paper_version_id="p", citation_format="apa", references=refs,
                                          citations=[regrouped])
    assert "r2" not in {rid for f in result.findings if f.finding_type == "reference_not_linked_in_extracted_citations"
                        for rid in f.reference_ids}


def _paper():
    return ("Media Rules\n\n"
            "Section 4 of 'The Interim Measures on the Supervision of Coastal Publication' (2002) requires every "
            "publisher to hold a licence. The later \"Rules on Network Audio Services\" (2016) extend this to streaming "
            "platforms, and the platforms complied quickly (Lowe & Marsh, 2016, p.7). Regulators also cite "
            "‘Measures for Broadcast Archives' (2010) when reviewing appeals.\n\nReferences\n\n"
            + "\n".join(ref.raw_ref for ref in _media_refs()) + "\n")


def test_a_unit_with_a_linked_quoted_title_is_assessed_with_its_other_citation():
    from app.services.paper_extraction import extract_paper_evidence
    artifact = extract_paper_evidence(_paper(), paper_version_id="v1", format_hint="apa", use_llm_boundaries=False,
                                      use_llm_atomizer=False, use_llm_reference_fallback=False)
    assert artifact.claim_boundary_rejections == []
    assert all(marker.link_status == "linked" for marker in artifact.citation_marker_census)
    by_title = {ref.title: ref.reference_id for ref in artifact.references}
    claimed = sorted(sorted(claim.reference_ids) for claim in artifact.citation_claims)
    assert sorted([by_title["Rules on network audio services"], by_title["Streaming and the state"]]) in claimed
    assert [by_title["Measures for broadcast archives"]] in claimed
    assert all(claim.claim_type == "paraphrase" for claim in artifact.citation_claims)
    assert "reference_not_linked_in_extracted_citations" not in artifact.reference_consistency.finding_counts
    archives = next(m for m in artifact.citation_marker_census if m.text.startswith("‘Measures"))
    assert archives.link_differences == ["year:2011"]


def test_a_year_difference_is_listed_once_even_inside_a_grouped_sentence():
    from app.services.evidence_report import _reference_mismatch_members
    from app.services.paper_extraction import extract_paper_evidence
    paper = _paper().replace("‘Measures for Broadcast Archives' (2010) when reviewing appeals",
                             "appeals (Lowe & Marsh, 2016) under ‘Measures for Broadcast Archives' (2010)")
    artifact = extract_paper_evidence(paper, paper_version_id="v1", format_hint="apa", use_llm_boundaries=False,
                                      use_llm_atomizer=False, use_llm_reference_fallback=False)
    marker = next(m for m in artifact.citation_marker_census if m.text.startswith("‘Measures"))
    rows = _reference_mismatch_members(artifact, marker.report_passage_start, marker.report_passage_end)
    assert [row for row in rows if "reference year 2011" in row] and len(rows) == 1
    assert _reference_mismatch_members(SimpleNamespace(citations=[], citation_marker_census=[marker]),
                                       marker.passage_start, marker.passage_end) == [
        "‘Measures for Broadcast Archives' (2010): reference year 2011"]
