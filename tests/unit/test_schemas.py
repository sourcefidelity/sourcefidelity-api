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


def test_parsed_reference_carries_deterministic_source_kind_evidence() -> None:
    reference = ParsedReference(
        author="Wu, T.",
        year="2010",
        title="The master switch",
        raw_ref=(
            "Wu, T. (2010). The master switch: The rise and fall of "
            "information empires. Knopf."
        ),
    )

    assert reference.source_kind == "monograph"
    assert reference.source_kind_confidence == "high"
    assert reference.source_kind_evidence
