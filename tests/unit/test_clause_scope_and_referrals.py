"""Owner decisions 2026-10-07 from the review of the owner's article: each source
is judged on its own clause; "see" citations are not judged; a narrower hanging
indent is stated as narrower, not missing."""
from types import SimpleNamespace

from app.services.candidate_relationship_judgment import _member_clause_spans
from app.services.judgment_input import judgment_eligibility, referral_citation


def _claim(text, *markers):
    built = []
    for marker, ref in markers:
        start = text.index(marker)
        built.append(SimpleNamespace(text=marker, local_start=start, local_end=start + len(marker), reference_ids=[ref]))
    return SimpleNamespace(text=text, citation_markers=built)


LIST = ("While harbour cranes are impressive, they remain slow in storms (Ames, 2022), fail to lift heavy loads "
        "(Bell & Cole, 2024), and break often (Dunn, 2023).")


def test_each_source_is_judged_on_its_own_clause():
    claim = _claim(LIST, ("(Ames, 2022)", "a"), ("(Bell & Cole, 2024)", "b"), ("(Dunn, 2023)", "d"))
    text = lambda ref: [LIST[a:b] for a, b in _member_clause_spans(claim, ref)]
    assert text("a") == ["While harbour cranes are impressive, they remain slow in storms"]
    assert text("b") == ["fail to lift heavy loads"]
    assert text("d") == ["and break often"]


def test_wording_after_a_mid_sentence_citation_belongs_to_no_source():
    text = ("Many ports have encouraged automation (Ames, 2025), so the revision was a chance to guide its use. "
            "The ports also report fewer delays.")
    claim = _claim(text, ("(Ames, 2025)", "a"))
    assert [text[a:b] for a, b in _member_clause_spans(claim, "a")] == [
        "Many ports have encouraged automation", "The ports also report fewer delays"]
    # A citation at the end of its sentence keeps the whole statement.
    end = _claim("Many ports have encouraged automation (Ames, 2025).", ("(Ames, 2025)", "a"))
    assert _member_clause_spans(end, "a") is None
    # A narrative citation is not a parenthetical marker and is left alone.
    narrative = _claim("Ames (2025) argues that ports automate, so delays fall.", ("Ames (2025)", "a"))
    assert _member_clause_spans(narrative, "a") is None
    year_only = _claim("Hartmn (2016) highlights that ports automate, so delays fall.", ("(2016)", "a"))
    assert _member_clause_spans(year_only, "a") is None


def _payload(marker):
    return {"source_binding": {"reference_id": "r1"},
            "claim": {"citation_markers": [{"text": marker, "reference_ids": ["r1"]}]}}


def test_a_see_citation_is_a_referral_and_is_not_judged():
    assert referral_citation(_payload("(see Edwards, 2024)"))
    assert referral_citation(_payload("(see also Edwards, 2024)")) and referral_citation(_payload("(cf. Edwards, 2024)"))
    assert not referral_citation(_payload("(e.g., Edwards, 2024)")) and not referral_citation(_payload("(Edwards, 2024)"))
    eligible = {**_payload("(see Edwards, 2024)"),
                "coverage": {"level": "full_text", "completeness_verdict": "complete"},
                "source_identity": {"status": "verified"},
                "facet_evidence_foundation": {"foundation_version": "exact-facet-evidence-foundation-v12", "status": "complete"}}
    assert judgment_eligibility(eligible).reason_code == "referral_citation"


def test_the_referral_note_is_shown_in_the_owners_words():
    from app.services.judgment_report import REFERRAL_NOTE
    assert REFERRAL_NOTE == "Not judged: a “see” citation points to further reading rather than supporting a statement."


def test_a_narrower_hanging_indent_is_stated_as_narrower():
    from app.services.evidence_report import HANGING_INDENT_NARROW_TEXT, _build_report_summary
    narrow = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=True, hanging_indent_narrow=True)
    missing = _build_report_summary(citations=[], overview={}, pervasive_hanging_indent=True)
    lines = lambda summary: [i.get("text") for i in summary.get("reference_formatting") or []]
    assert HANGING_INDENT_NARROW_TEXT in lines(narrow)
    assert HANGING_INDENT_NARROW_TEXT == "The reference list's hanging indent is narrower than APA's 0.5 inch."
    assert not any(HANGING_INDENT_NARROW_TEXT == line for line in lines(missing))


def test_one_wider_entry_does_not_hide_a_narrow_indent():
    from app.services.evidence_report import _hanging_indent_narrow
    assert _hanging_indent_narrow([{"observed_points": 15.0}] * 13 + [{"observed_points": 55.6}])
    assert not _hanging_indent_narrow([{"observed_points": 0.0}] * 10)
    assert not _hanging_indent_narrow([{"observed_points": 72.0}] * 10)
    assert not _hanging_indent_narrow([{}])


def test_each_sources_clause_is_its_own_citation():
    from app.services.candidate_relationship_judgment import citation_clause_segments
    claim = _claim(LIST, ("(Ames, 2022)", "a"), ("(Bell & Cole, 2024)", "b"), ("(Dunn, 2023)", "d"))
    pieces = [(LIST[s:e], refs, opens, closes) for s, e, refs, opens, closes in citation_clause_segments(claim, ["a", "b", "d"])]
    assert pieces == [
        ("While harbour cranes are impressive, they remain slow in storms (Ames, 2022)", ["a"], True, False),
        ("fail to lift heavy loads (Bell & Cole, 2024)", ["b"], False, False),
        ("and break often (Dunn, 2023).", ["d"], False, True)]


def test_wording_after_a_single_citation_is_left_out_of_it():
    from app.services.candidate_relationship_judgment import citation_clause_segments
    text = "Many ports have encouraged automation (Ames, 2025), so the revision was a chance to guide its use."
    [(s, e, refs, opens, closes)] = citation_clause_segments(_claim(text, ("(Ames, 2025)", "a")), ["a"])
    assert (text[s:e], opens, closes) == ("Many ports have encouraged automation (Ames, 2025)", True, False)


def test_citations_that_stay_whole():
    from app.services.candidate_relationship_judgment import citation_clause_segments
    # One marker at the end, two sources in one bracket, a later sentence, a narrative citation.
    assert citation_clause_segments(_claim("Ports automate (Ames, 2025).", ("(Ames, 2025)", "a")), ["a"]) is None
    both = "Ports automate (Ames, 2025; Bell, 2024)."
    one = _claim(both, ("(Ames, 2025; Bell, 2024)", "a"))
    one.citation_markers[0].reference_ids = ["a", "b"]
    assert citation_clause_segments(one, ["a", "b"]) is None
    later = "Ports automate (Ames, 2025), and cranes rust (Bell, 2024). Delays fall."
    assert citation_clause_segments(_claim(later, ("(Ames, 2025)", "a"), ("(Bell, 2024)", "b")), ["a", "b"]) is None
    narrative = "Ames (2025) argues ports automate, and cranes rust (Bell, 2024)."
    assert citation_clause_segments(_claim(narrative, ("(2025)", "a"), ("(Bell, 2024)", "b")), ["a", "b"]) is None


def test_an_upload_matches_a_title_with_a_line_break_hyphen():
    # A reference list title broken at a line end keeps its hyphen ("automa-tion").
    from app.services.pdf_verifier import _title_matches
    page = {"title": "Harbour Studies", "first_page_text": "Harbours in the Era of Crane Automation (CA): "
            "Understanding the Potential Benefits of Remote Loading"}
    assert _title_matches("Harbours in the era of crane automa-tion (CA): Understanding the "
                          "potential benefits of remote loading", page)
    assert not _title_matches("Airports in the era of crane automa-tion (CA): Misreading the "
                              "costs of remote parking", page)


def test_sources_sharing_a_bracket_share_its_clause():
    # Stored one by one, without the bracket (the owner's article, citation 3, 2026-10-07).
    from app.services.candidate_relationship_judgment import citation_clause_segments
    text = "Ports feared automation (Ames, 2023), as well as claims that cranes will change them (Bell, 2023; Cole, 2023)."
    claim = _claim(text, ("(Ames, 2023)", "a"), ("Bell, 2023", "b"), ("Cole, 2023", "c"))
    pieces = [(text[s:e], refs) for s, e, refs, _o, _c in citation_clause_segments(claim, ["a", "b", "c"])]
    assert pieces == [("Ports feared automation (Ames, 2023)", ["a"]),
                      ("as well as claims that cranes will change them (Bell, 2023; Cole, 2023).", ["b", "c"])]
    assert [text[s:e] for s, e in _member_clause_spans(claim, "c")] == ["as well as claims that cranes will change them"]
