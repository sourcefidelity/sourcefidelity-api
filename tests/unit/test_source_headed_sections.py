"""Owner decision 2026-10-04: in a source-by-source review each heading's citation
covers the section's uncited paragraphs, keeping its exact marker (paper 7)."""
from types import SimpleNamespace

from app.services.paper_extraction import CitationMarkerCensusEntry, source_headed_sections
from app.services.schemas import InTextCitation

DISCUSS = "The author traces how professional ethics developed in early newspapers and the press."
REFS = [SimpleNamespace(reference_id="ref-smith", source_kind="journal_article", title=""),
        SimpleNamespace(reference_id="ref-jones", source_kind="monograph", title=""),
        SimpleNamespace(reference_id="ref-film", source_kind="traditional_media", title="")]


def _census(body, marker, reference_id, index):
    start = body.index(marker)
    return CitationMarkerCensusEntry(marker_id=f"m{index}", text=marker, passage_start=start,
                                     passage_end=start + len(marker), reference_ids=[reference_id],
                                     link_status="linked", marker_type="narrative", paragraph_index=0,
                                     member_count=1)


def _heading(body, line, marker, reference_id):
    start = body.index(line)
    return InTextCitation(reference_ids=[reference_id], text=line, citation_marker=marker,
                          passage_start=start, passage_end=start + len(line), citation_key=reference_id)


def _active(citations):
    return [c for c in citations if c.drop_reason is None]


def test_each_heading_citation_covers_its_sections_uncited_paragraphs():
    first, second = "Article 1: Smith, J. (2018).", "Article 2: Jones, K. (2020)."
    body = (f"{first}\n\n{DISCUSS}\n\nA second paragraph about the same book and its argument.\n\n"
            f"{second}\n\n{DISCUSS}\n\nAnother view (Lee, 2019) is cited here with its own marker for this paragraph.")
    census = [_census(body, "Smith, J. (2018)", "ref-smith", 1), _census(body, "Jones, K. (2020)", "ref-jones", 2),
              _census(body, "(Lee, 2019)", "ref-lee", 3)]
    citations = [_heading(body, first, "Smith, J. (2018)", "ref-smith"),
                 _heading(body, second, "Jones, K. (2020)", "ref-jones")]
    result = source_headed_sections(citations, census, body, REFS)
    active = _active(result)
    assert [c.reference_ids[0] for c in active] == ["ref-smith", "ref-jones"]
    assert active[0].text.startswith(first) and active[0].text.endswith("its argument.")
    assert active[1].text.startswith(second) and active[1].text.endswith(DISCUSS)
    assert all(body[c.passage_start:c.passage_end] == c.text and c.citation_marker in c.text for c in active)
    assert {c.drop_reason for c in result if c.drop_reason} == {"base_of_source_headed_section_v1"}


def test_one_cited_line_or_a_film_heading_changes_nothing():
    line = "Smith, J. (2018)."
    body = f"{line}\n\n{DISCUSS}\n\n{DISCUSS}"
    census = [_census(body, "Smith, J. (2018)", "ref-smith", 1)]
    citations = [_heading(body, line, "Smith, J. (2018)", "ref-smith")]
    assert source_headed_sections(citations, census, body, REFS) == citations
    films = f"Alien (1979):\n\n{DISCUSS}\n\nJaws (1975):\n\n{DISCUSS}"
    census = [_census(films, "Alien (1979)", "ref-film", 1), _census(films, "Jaws (1975)", "ref-film", 2)]
    assert source_headed_sections([], census, films, REFS) == []


def test_a_sentence_with_a_citation_is_not_a_heading():
    one, two = "Smith, J. (2018) argues that ethics began with newspapers.", "Jones, K. (2020) also argues this point about newspapers."
    body = f"{one}\n\n{DISCUSS}\n\n{two}\n\n{DISCUSS}"
    census = [_census(body, "Smith, J. (2018)", "ref-smith", 1), _census(body, "Jones, K. (2020)", "ref-jones", 2)]
    citations = [_heading(body, one, "Smith, J. (2018)", "ref-smith"), _heading(body, two, "Jones, K. (2020)", "ref-jones")]
    assert source_headed_sections(citations, census, body, REFS) == citations


def test_headings_inside_one_block_and_a_closing_section_and_a_model_continuation():
    # Paper 7: heading, discussion and the next heading on single line breaks;
    # a "Conclusion:" heading ends the last book's section; the model's own
    # continuation inside a section is kept for audit only.
    first, second = "Article 1: Smith, J. (2018). The Fourth Estate", "Article 2: Jones, K. (2020). Press Freedom"
    body = f"{first}\n{DISCUSS}\n{second}\n\n{DISCUSS}\n\nConclusion: Ethical Trends in the U.S.A\n\n{DISCUSS}"
    census = [_census(body, "Smith, J. (2018)", "ref-smith", 1), _census(body, "Jones, K. (2020)", "ref-jones", 2)]
    start = body.index(DISCUSS)
    continuation = InTextCitation(reference_ids=["ref-smith"], text=DISCUSS, citation_marker="implicit_continuation",
                                  passage_start=start, passage_end=start + len(DISCUSS))
    citations = [_heading(body, first, "Smith, J. (2018)", "ref-smith"), continuation,
                 _heading(body, second, "Jones, K. (2020)", "ref-jones")]
    result = source_headed_sections(citations, census, body, REFS)
    active = _active(result)
    assert [c.text for c in active] == [f"{first}\n{DISCUSS}", f"{second}\n\n{DISCUSS}"]
    assert "within_source_headed_section_v1" in {c.drop_reason for c in result}


def test_the_heading_only_names_the_source():
    # Owner decision 2026-10-04: the paragraphs are the statement shown and judged.
    from app.services.paper_extraction import SOURCE_HEADING_MARKER_TYPE, source_heading_statement_start
    first, second = "Article 1: Smith, J. (2018). The Fourth Estate", "Article 2: Jones, K. (2020). Press Freedom"
    body = f"{first}\n{DISCUSS}\n{second}\n\n{DISCUSS}"
    census = [_census(body, "Smith, J. (2018)", "ref-smith", 1), _census(body, "Jones, K. (2020)", "ref-jones", 2)]
    citations = [_heading(body, first, "Smith, J. (2018)", "ref-smith"), _heading(body, second, "Jones, K. (2020)", "ref-jones")]
    active = _active(source_headed_sections(citations, census, body, REFS))
    assert {c.marker_type for c in active} == {SOURCE_HEADING_MARKER_TYPE}
    assert [c.text[source_heading_statement_start(c.text, c.citation_marker):] for c in active] == [DISCUSS, DISCUSS]
