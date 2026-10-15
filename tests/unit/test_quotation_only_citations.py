"""Owner rule 2026-10-12: a citation that is only a quotation is not judged;
quotation and locator checks still run. Quoted words inside the student's own
claim get both. Invented sentences only."""
import pytest

from app.services.judgment_input import judgment_eligibility
from app.services.quotation_form import QUOTATION_FORM_VERSION, classify_quotation_form


def _form(text, *markers, names=("Hale, R.",), titles=()):
    spans = [(text.index(m), text.index(m) + len(m)) for m in markers]
    return classify_quotation_form(text, marker_spans=spans, names=names, titles=titles)


@pytest.mark.parametrize("text, markers", [
    ('Hale (2019) argues that "harbour cranes set the pace of every port."', ("Hale (2019)",)),
    ('According to Hale (2019), "harbour cranes set the pace of every port" (p. 4).', ("Hale (2019)",)),
    ('As Hale (2019) puts it, “the crane is the port”.', ("Hale (2019)",)),
    ('In Hale’s words, "the crane is the port" (2019, p. 4).', ("(2019, p. 4)",)),
    ('Moreover, Hale states: "the crane is the port" (2019).', ("(2019)",)),
    ('"The crane is the port" (Hale, 2019, p. 4).', ("(Hale, 2019, p. 4)",)),
    ("She was called 'the harbour queen' and 'the last stevedore' (Hale, 2019).", ("(Hale, 2019)",)),
    ('However, the crane was described as "a slow giant" by Hale (2019).', ("Hale (2019)",)),
])
def test_a_signal_phrase_and_a_quotation_is_quotation_only(text, markers):
    # "crane" in the last case is the student's word, so it is embedded.
    expected = "embedded" if "the crane was described" in text else "quotation_only"
    assert _form(text, *markers) == expected


@pytest.mark.parametrize("text, markers", [
    ("'Productivity growth' and 'rising prosperity' are hampered by slow cranes (Hale, 2019).", ("(Hale, 2019)",)),
    ('Ports that automate create "dead time" for crews (Hale, 2019).', ("(Hale, 2019)",)),
    ('Hale (2019) shows that "cranes set the pace" in every northern port.', ("Hale (2019)",)),
    ('This is "the worst policy" for ports (Hale, 2019).', ("(Hale, 2019)",)),
])
def test_quoted_words_in_the_students_claim_are_embedded(text, markers):
    assert _form(text, *markers) == "embedded"


def test_a_quoted_reference_title_is_a_title():
    text = 'Hale (2019) wrote "Cranes and the modern port".'
    assert _form(text, "Hale (2019)", titles=["Cranes and the modern port"]) == "title"
    # Without the title in the reference list it is a quotation.
    assert _form(text, "Hale (2019)") == "quotation_only"


@pytest.mark.parametrize("text", [
    'Hale (2019) argues that "cranes set the pace of every port.',
    'Hale (2019) argues that “cranes set the pace of every port.',
    "Hale (2019) argues that ‘cranes set the pace of every port.",
])
def test_unbalanced_quotation_marks_are_embedded(text):
    assert _form(text, "Hale (2019)") == "embedded"


def test_no_quotation_marks_and_apostrophes_are_none():
    assert _form("Cranes set the pace of every port (Hale, 2019).", "(Hale, 2019)") == "none"
    assert _form("The authors’ view is that cranes don't set the pace (Hale, 2019).", "(Hale, 2019)") == "none"


def _eligible_payload(form):
    return {"source_binding": {"reference_id": "r1"},
            "claim": {"text": "x", "quotation_form": form,
                      "citation_markers": [{"text": "(Hale, 2019)", "reference_ids": ["r1"]}]},
            "coverage": {"level": "full_text", "completeness_verdict": "complete"},
            "source_identity": {"status": "verified"},
            "facet_evidence_foundation": {"foundation_version": "exact-facet-evidence-foundation-v12",
                                          "status": "complete"}}


def test_a_quotation_only_citation_is_not_judged():
    assert judgment_eligibility(_eligible_payload("quotation_only")).reason_code == "quotation_only"
    for form in ("embedded", "title", "none", None):
        assert judgment_eligibility(_eligible_payload(form)).eligible


def test_extraction_stores_the_form_and_keeps_the_quotation_check():
    from app.services.paper_extraction import extract_paper_evidence
    from app.services.quotation_locator_requirement import assess_quotation_locators
    paper = ('Ports and cranes\n\nHale (2019) argues that "harbour cranes set the pace of every port." '
             'Busy ports also run night shifts (Hale, 2019).\n\nReferences\n\n'
             'Hale, R. (2019). Cranes and the modern port. Port Studies, 4(2), 11-19.\n')
    artifact = extract_paper_evidence(paper, paper_version_id="v1", format_hint="apa",
                                      use_llm_boundaries=False, use_llm_atomizer=False,
                                      use_llm_reference_fallback=False)
    forms = {claim.text: (claim.quotation_form, claim.quotation_form_version, claim.claim_type)
             for claim in artifact.citation_claims}
    quoted = next(value for text, value in forms.items() if "harbour cranes" in text)
    assert quoted == ("quotation_only", QUOTATION_FORM_VERSION, "quotation")
    plain = next(value for text, value in forms.items() if "night shifts" in text)
    assert plain[0] == "none"
    # The quotation and locator checks are unchanged by the form.
    dump = lambda rows: [row.model_dump(mode="json") for row in rows]
    body = paper.split("\n\nReferences")[0]
    unset = [c.model_copy(update={"quotation_form": None, "quotation_form_version": None})
             for c in artifact.citation_claims]
    assert artifact.quotation_locator_requirements
    assert dump(assess_quotation_locators(body_text=body, citation_format="apa", claims=unset,
                                          references=artifact.references)) == dump(
        assess_quotation_locators(body_text=body, citation_format="apa", claims=artifact.citation_claims,
                                  references=artifact.references))


def test_an_unset_form_dumps_as_before():
    from app.services.verification_evidence import ClaimEvidence
    claim = ClaimEvidence(claim_id="c", paper_version_id="p", text="Cranes set the pace.")
    assert "quotation_form" not in claim.model_dump(mode="json")
    stored = claim.model_copy(update={"quotation_form": "embedded", "quotation_form_version": QUOTATION_FORM_VERSION})
    assert ClaimEvidence.model_validate(stored.model_dump(mode="json")).quotation_form == "embedded"


def test_the_heading_reads_the_owners_wording():
    # Owner decision 2026-10-13: "Not Judged - Quotation only" replaces the
    # evidence kind in the heading; no separate note in the window.
    from pathlib import Path
    from app.services.judgment_report import judgment_result
    result = judgment_result({"display_state": "not_judged", "reason_code": "quotation_only"}, {}, None)
    assert result["label"] == "Not Judged" and "Quotation only" not in result["html"]
    script = (Path(__file__).resolve().parents[2] / "app/services/report_judgment.js").read_text()
    # The server marks the item from the reason table; the page only reads the mark (2026-10-15).
    from app.services.judgment_reasons import quotation_only
    assert quotation_only("quotation_only") and not quotation_only("referral_citation")
    assert "i.quotation_only" in script and "'Quotation only'" in script and "'quotation_only'" not in script
