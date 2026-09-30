"""Title-alone merges identify a work but cannot accuse the reference.

Owner decision, September 23 (option C). A distinctive title is enough to show
a reader which work was probably cited, and measured over the development
corpus 1,613 of 5,702 accepts rested on it, 23 references being identified that
way and no other. But a title is one signal: a reprint, a later edition or
another work sharing the phrase satisfies it equally. So the record stays
visible and keeps supplying locations and text, and is barred from supplying a
disagreement with what the student wrote.
"""
from app.services.retrieval import RetrievalResult
from app.services.retrieval.canonical_work import (
    CanonicalWorkGraph,
    assess_work_identity,
)

_TITLE = "The Classical Hollywood Cinema Film Style and Mode of Production"


def _record(name: str, **fields) -> RetrievalResult:
    return RetrievalResult(source_name=name, success=True, **fields)


def _graph(**expected) -> CanonicalWorkGraph:
    return CanonicalWorkGraph(expected_title=_TITLE, **expected)


def test_a_distinctive_title_alone_still_identifies_the_work() -> None:
    assessment = assess_work_identity(
        _record("openalex", title=_TITLE, year="1985"),
        expected_doi=None, expected_title=_TITLE, expected_author=None,
        expected_year=None)

    assert assessment.accepted is True
    assert assessment.corroboration == "title_alone"


def test_an_agreeing_second_field_makes_the_match_corroborated() -> None:
    assessment = assess_work_identity(
        _record("openalex", title=_TITLE, year="1985", authors=["Bordwell"]),
        expected_doi=None, expected_title=_TITLE, expected_author="Bordwell",
        expected_year=None)

    assert assessment.corroboration == "corroborated"


def test_a_title_alone_record_cannot_create_a_year_conflict() -> None:
    graph = _graph(expected_author="Bordwell")
    graph.add(_record("openalex", title=_TITLE, year="1985", authors=["Bordwell"]))
    # A different edition of the same title, matching on nothing else.
    graph.add(_record("core", title=_TITLE, year="2003"))

    merged = graph.to_result()
    work = merged.metadata["canonical_work"]

    assert work["accepted_providers"] == ["openalex", "core"]
    assert work["title_alone_providers"] == ["core"]
    # Both were accepted, so the reader sees both; only the corroborated one
    # may say the reference's year is wrong.
    assert "year" not in work["metadata_conflicts"]


def test_two_corroborated_records_still_conflict() -> None:
    graph = _graph(expected_author="Bordwell")
    graph.add(_record("openalex", title=_TITLE, year="1985", authors=["Bordwell"]))
    graph.add(_record("core", title=_TITLE, year="2003", authors=["Bordwell"]))

    conflicts = graph.to_result().metadata["canonical_work"]["metadata_conflicts"]

    assert sorted(c["value"] for c in conflicts["year"]) == ["1985", "2003"]


def test_a_corroborated_year_is_preferred_when_the_reference_gave_none() -> None:
    graph = _graph(expected_author="Bordwell")
    graph.add(_record("core", title=_TITLE, year="2003"))
    graph.add(_record("openalex", title=_TITLE, year="1985", authors=["Bordwell"]))

    assert graph.to_result().year == "1985"


def test_a_title_alone_year_is_shown_when_it_is_all_there_is() -> None:
    graph = _graph()
    graph.add(_record("core", title=_TITLE, year="2003"))

    merged = graph.to_result()

    # Identity for display is kept: withholding it would leave the reader with
    # no located record at all, which is not what a single weak signal means.
    assert merged.success is True
    assert merged.year == "2003"
    assert merged.metadata["canonical_work"]["corroborated_providers"] == []
