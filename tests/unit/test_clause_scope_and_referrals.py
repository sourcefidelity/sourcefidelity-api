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
