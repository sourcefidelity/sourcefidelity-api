from app.services.reference_formatting import assess_reference_formatting
from app.services.reference_layout import ReferenceEntryLayout, ReferenceLayoutArtifact


def _layout(*entries: ReferenceEntryLayout, citation_format: str = "apa"):
    return ReferenceLayoutArtifact(
        content_sha256="a" * 64,
        media_type="application/pdf",
        extraction_backend="pymupdf_dict",
        location_kind="page",
        citation_format=citation_format,
        status="complete",
        heading_status="matched",
        heading_page_index=1,
        reference_count=len(entries),
        matched_reference_count=sum(e.mapping_status == "matched" for e in entries),
        entries=list(entries),
    )


def _entry(reference_id: str, observed: float | None):
    return ReferenceEntryLayout(
        reference_id=reference_id,
        reference_text_sha256="b" * 64,
        mapping_status="matched",
        match_confidence=1.0,
        location_indexes=[1],
        line_count=2 if observed is not None else 1,
        first_line_x_points=72.0,
        continuation_x_median_points=(
            None if observed is None else 72.0 + observed
        ),
        observed_hanging_indent_points=observed,
        italic_character_fraction=0.1,
        bold_character_fraction=0.0,
    )


def test_hanging_indent_rule_preserves_match_difference_and_unobservable():
    result = assess_reference_formatting(
        _layout(_entry("good", 36.0), _entry("bad", 0.0), _entry("one-line", None))
    )

    assert result.status == "partial"
    assert result.result_counts == {
        "difference": 1,
        "matches_rule": 1,
        "not_assessed": 1,
    }
    assert [item.status for item in result.results] == [
        "matches_rule",
        "difference",
        "not_assessed",
    ]
    assert result.primary_rule_sources == [
        "https://www.apa.org/ed/precollege/psn/2020/09/apa-style-student-papers"
    ]


def test_mla_uses_primary_sources_and_does_not_assess_unmatched_entry():
    entry = ReferenceEntryLayout(
        reference_id="ambiguous",
        reference_text_sha256="c" * 64,
        mapping_status="ambiguous",
        match_confidence=0.9,
        line_count=0,
    )
    result = assess_reference_formatting(_layout(entry, citation_format="mla"))

    assert result.status == "not_assessed"
    assert result.result_counts == {"not_assessed": 1}
    assert all(source.startswith("https://style.mla.org/") for source in result.primary_rule_sources)
