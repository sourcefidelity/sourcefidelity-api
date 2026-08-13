from app.services.ref_field_extractor import (
    extract_fields_apa,
    extract_fields_from_llm_response,
    extract_fields_mla,
)


def test_apa_regex_extracts_identity_fields() -> None:
    raw = (
        "Smith, J. (2024). Platform governance and academic integrity. "
        "Journal of Source Studies, 8(2). https://doi.org/10.1234/example"
    )

    reference = extract_fields_apa(raw)

    assert reference is not None
    assert reference.author == "Smith, J"
    assert reference.year == "2024"
    assert reference.title == "Platform governance and academic integrity"
    assert reference.doi == "10.1234/example"
    assert reference.extraction_method == "regex"
    assert reference.needs_review is False


def test_mla_regex_extracts_quoted_article_title() -> None:
    raw = (
        'Smith, Jane. "Platform Governance and Academic Integrity." '
        "Journal of Source Studies, vol. 8, no. 2, 2024, pp. 1-20."
    )

    reference = extract_fields_mla(raw)

    assert reference is not None
    assert reference.author == "Smith, Jane"
    assert reference.year == "2024"
    assert reference.title == "Platform Governance and Academic Integrity."
    assert reference.extraction_method == "regex"


def test_llm_fallback_is_always_marked_for_review() -> None:
    response = """Author: Smith, J.
Year: Published 2024
Title: Platform governance
DOI: https://doi.org/10.1234/example
URL: none
"""

    reference = extract_fields_from_llm_response(response, "original reference")

    assert reference.extraction_method == "llm"
    assert reference.needs_review is True
    assert reference.year == "2024"
    assert reference.doi == "10.1234/example"
    assert reference.raw_ref == "original reference"
