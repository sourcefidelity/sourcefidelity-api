"""Public regression tests for corpus-independent APA entry boundaries."""

from app.services.parsers.apa_parser import ApaParser


def test_dated_work_title_is_a_continuation():
    raw = """Rivera, A. (2020). A study of adaptations:
The clockwork city (1922) and its reception. Quarterly Review of Media, 2(1), 1-10.
Singh, B. (2021). Another work. Example Press.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 2
    assert "The clockwork city (1922)" in references[0]
    assert references[1].startswith("Singh, B.")


def test_period_terminated_dated_title_after_incomplete_line_is_a_continuation():
    raw = """Rivera, A. (2020). A study of adaptations:
The clockwork city (1922). Reception and interpretation. Quarterly Review, 2(1), 1-10.
Singh, B. (2021). Another work. Example Press.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 2
    assert "The clockwork city (1922)." in references[0]


def test_references_do_not_require_known_publisher_words():
    raw = """Rivera, A. (2020). Community archives in practice. North Coast Historical Society.
Singh, B. (2021). Small collections and public memory. Lakeside Museum.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 2
    assert references[0].endswith("North Coast Historical Society.")
    assert references[1].endswith("Lakeside Museum.")


def test_year_disambiguation_suffixes_are_entry_dates():
    raw = """Rivera, A. (2020a). First related work. Lakeside Museum.
Rivera, A. (2020b). Second related work. North Coast Historical Society.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 2
    assert references[0].startswith("Rivera, A. (2020a)")
    assert references[1].startswith("Rivera, A. (2020b)")


def test_wrapped_author_lines_remain_together_until_the_date():
    raw = """Hendy, A., Vignoles, V., and another author with an extended name,
Finalauthor, Z. (2023). A jointly authored study. Quarterly Review, 2(1), 1-20.
Rivera, A. (2024). A following study. Lakeside Museum.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 2
    assert references[0].startswith("Hendy, A.")
    assert "Finalauthor, Z. (2023)" in references[0]
    assert references[1].startswith("Rivera, A. (2024)")


def test_long_institutional_author_survives_missing_separator():
    raw = """Rivera, A. (2020). A preceding study. Lakeside Museum.
International Council for Community Archives and Regional Public Memory (2021). Annual report. North Coast Historical Society.
Singh, B. (2022). A following study. Quarterly Review, 3(1), 1-10.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 3
    assert references[1].startswith("International Council")


def test_parenthesized_institutional_date_field_starts_an_entry():
    raw = """Rivera, A. (2020). A preceding study. Lakeside Museum.
Regional Media Council (2021). Annual report. North Coast Historical Society.
Singh, B. (2022). A following study. Quarterly Review, 3(1), 1-10.
"""

    references = ApaParser.split_references(raw)

    assert len(references) == 3
    assert references[1].startswith("Regional Media Council (2021)")
