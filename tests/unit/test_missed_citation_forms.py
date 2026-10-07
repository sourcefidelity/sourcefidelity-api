"""Owner decision 2026-10-06: citation forms the owner's highlights showed missed
(APA test papers): an undated title used as the author, legislation cited by
title and year, an author written first name first, an author with no year."""
from app.services.citation_extractor import extract_citations
from app.services.paper_extraction import yearless_author_mentions
from app.services.schemas import InTextCitation, ParsedReference


def _ref(rid, raw, author="", year="", title=""):
    return ParsedReference(reference_id=rid, raw_ref=raw, author=author, year=year, title=title,
                           citation_key=rid)


def _linked(body, refs):
    return {c.citation_marker: c.reference_ids for c in extract_citations(body, refs, format_hint="apa")
            if c.link_status == "linked"}


def test_an_undated_title_used_as_the_author_links_its_entry():
    refs = [_ref("r1", "Harbour lights box office. (n.d) Retrieve from: https://example.test/a", year="n.d."),
            _ref("r2", "STUDIO AND LICENSING. (n.d.) Retrieved from: https://example.test/b",
                 author="STUDIO AND LICENSING", year="n.d.")]
    body = ("The film grossed 536 million dollars (“Harbour lights box office”, n.d.). "
            "The studio employs designers (“STUDIO AND\nLICENSING”, n.d.).")
    linked = _linked(body, refs)
    assert linked["(“Harbour lights box office”, n.d.)"] == ["r1"]
    assert linked["(“STUDIO AND\nLICENSING”, n.d.)"] == ["r2"]


def test_legislation_cited_by_title_and_year_links_its_entry():
    refs = [_ref("r1", "Measures for the Licensing of Coastal Radio Receivers 1990. http://example.test/law", year="1990"),
            _ref("r2", "Rules on Joint Production of Radio Serials 2004. http://example.test/rules", year="2004")]
    body = ("Section 4 limits users to licensed units (Measures for the Licensing of Coastal Radio Receivers 1990, reg 4). "
            "The rules also bar unlicensed joint production (Rules on Joint Production of Radio Serials 2004, reg. 4).")
    linked = _linked(body, refs)
    assert ["r1"] in linked.values() and ["r2"] in linked.values()
    # The law named in a sentence is the student's topic, not a citation.
    assert not _linked('This essay takes the "Measures for the Licensing of Coastal Radio Receivers (1990)" as its case.', refs)


def test_an_author_written_first_name_first_holds_the_cited_surname():
    refs = [_ref("r1", "Wilma Ostrander Media Tutor (2005). Notes on realism.", author="Wilma Ostrander Media Tutor", year="2005"),
            _ref("r2", "[Review of the motion picture Harbour Express]. (1932). Trade Weekly, 9.",
                 author="[Review of the motion picture Harbour Express]", year="1932")]
    assert _linked("All messages are constructed (Ostrander, 2005).", refs) == {"(Ostrander, 2005)": ["r1"]}
    # A bracketed description is not a person's name.
    assert not _linked("The film Harbour Express (1932) opened in spring.", refs)


def test_an_author_with_no_year_links_and_records_the_missing_year():
    refs = [_ref("r1", "National Society of Paediatrics. (2014). Some things to know about media. https://example.test/m",
                 author="National Society of Paediatrics", year="2014"),
            _ref("r2", "Moonlit Detective The Game. (n.d.). Retrieved from: https://example.test/g",
                 author="Moonlit Detective The Game", year="n.d.")]
    [citation] = [c for c in extract_citations("Media have their own codes (National Society of Paediatrics).", refs,
                                               format_hint="apa") if c.link_status == "linked"]
    assert citation.reference_ids == ["r1"] and citation.link_differences == ["year:2014"]
    # A title in the author slot is the work's name: "(Moonlit Detective The Game)" names a game.
    assert not _linked("Spin-offs include video games (Moonlit Detective The Game).", refs)


def test_a_sentence_naming_a_cited_author_without_a_year_is_a_citation():
    refs = [_ref("r1", "Halvorsen, G. R. (2004). A life on screen. Press.", author="Halvorsen, G. R.", year="2004"),
            _ref("r2", "Quint, B. (2010). Unrelated. Press.", author="Quint, B.", year="2010")]
    first = "She told interviewers she was tired (Halvorsen, 2004, p. 63)."
    body = f"Halvorsen documents how the actor left in 1928. {first} Quint argues otherwise."
    start = body.index(first)
    cited = InTextCitation(reference_ids=["r1"], link_status="linked", text=first, citation_marker="(Halvorsen, 2004, p. 63)",
                           passage_start=start, passage_end=start + len(first), citation_key="r1")
    result = yearless_author_mentions([cited], [], body, refs)
    added = [c for c in result if c is not cited]
    assert [(c.citation_marker, c.reference_ids) for c in added] == [("Halvorsen documents", ["r1"])]
    # Quint is never cited with a marker, so naming Quint links nothing.
    assert body[added[0].passage_start:added[0].passage_end] == added[0].text


def test_the_students_own_framing_is_not_a_yearless_citation():
    refs = [_ref("r1", "Halvorsen, G. R. (2004). A life on screen. Press.", author="Halvorsen, G. R.", year="2004")]
    first = "She told interviewers she was tired (Halvorsen, 2004)."
    body = f"{first} Drawing on Halvorsen's theory, this essay examines the actor's image."
    cited = InTextCitation(reference_ids=["r1"], link_status="linked", text=first, citation_marker="(Halvorsen, 2004)",
                           passage_start=0, passage_end=len(first), citation_key="r1")
    assert yearless_author_mentions([cited], [], body, refs) == [cited]


def test_a_title_first_web_entry_keeps_its_title_and_is_cited_by_it():
    from app.services.reference_parser import extract_and_parse_references
    refs = extract_and_parse_references(
        "References\nHarbour season 1. (n.d.) Retrieved from: https://example.test/tv/s01\n"
        "Harbour box office. (n.d) Retrieve from: https://example.test/i/377?share=x&utm_ source=app\n"
        "Pictorial Weekly. (1945). https://example.test/details/pictorial-1945\n"
        "Ames, T. (2022). A regular article title here. https://example.test/a\n",
        format_hint="apa", use_regex_first=True, use_llm_fallback=False, paper_version_id="x")
    assert [(r.author, r.title) for r in refs] == [
        ("", "Harbour season 1"), ("", "Harbour box office"), ("", "Pictorial Weekly"),
        ("Ames, T", "A regular article title here")]
    linked = _linked("Covers often featured her (Pictorial Weekly, 1945). It rated 98 percent (“Harbour season 1”, n.d.).", refs)
    assert linked == {"(Pictorial Weekly, 1945)": [refs[2].reference_id], "(“Harbour season 1”, n.d.)": [refs[0].reference_id]}
