from app.services.schemas import ParsedReference


def test_parsed_reference_normalizes_doi_and_year() -> None:
    reference = ParsedReference(
        author="Smith, J.",
        year="Published online in 2024",
        title="A source",
        doi="https://doi.org/10.1234/example",
    )

    assert reference.doi == "10.1234/example"
    assert reference.year == "2024"


def test_missing_year_is_explicitly_unknown() -> None:
    assert ParsedReference(year=None).year == "n.d."
