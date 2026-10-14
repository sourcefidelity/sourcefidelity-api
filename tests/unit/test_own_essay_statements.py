"""The writer stating their own essay's aims is no claim about a source (owner decision 2026-10-14)."""
import pytest

from app.services.paper_extraction import OWN_ESSAY_STATEMENT, states_own_essay_aims, unresolved_marker_report_groups


@pytest.mark.parametrize("text,expected", [
    ("This essay endeavors to take the 'Regulations for the Administration' (2016) as an example to discuss.", True),
    ("This essay will compare Smith's (2019) framework with Lee's.", True),
    ("My paper aims to examine the Copyright Act (1976) in practice.", True),
    # The cited paper is the subject.
    ("This paper (Smith, 2019) aims to show that trade falls.", False),
    # A proposition attributed to the source stays a citation (Stardom 1).
    ("This essay will provide a case study based on Bordwell's (2006) argument that technology changed little.", False),
    ("This study found that tariffs fell (Smith, 2019).", False),
    ("In this essay, I argue that tariffs harm workers (Smith, 2019).", False),
])
def test_only_statements_of_the_essays_own_aims_match(text, expected):
    assert states_own_essay_aims(text) is expected


def test_its_marker_is_not_shown_as_an_unresolved_citation():
    from app.services.paper_extraction import CitationMarkerCensusEntry
    fields = CitationMarkerCensusEntry.model_fields
    sample = {name: (1 if field.annotation in (int,) else "x") for name, field in fields.items() if field.is_required()}
    marker = CitationMarkerCensusEntry.model_validate({**sample, "coverage_status": OWN_ESSAY_STATEMENT,
                                                       "passage_start": 0, "passage_end": 10, "text": "(2016)"})
    assert unresolved_marker_report_groups([marker]) == []


def test_an_assessment_of_a_cited_works_effects_is_not_a_claim_about_it():
    from app.services.paper_extraction import CITED_WORK_EFFECTS_STATEMENT, not_a_claim_about_the_source
    assert not_a_claim_about_the_source(
        'The results of the "Regulations for the Administration of Online Publishing Services" (2016) '
        "largely confirm the pros and cons of such regulations.") == CITED_WORK_EFFECTS_STATEMENT
    # A statement of the work's content stays a claim.
    assert not_a_claim_about_the_source(
        "Section 7 of 'The Interim Provisions on the Administration of Internet Publication' (2002) "
        "specifically regulated the conditions for Internet publishing.") is None
