from app.services.reference_consistency import assess_reference_consistency
from app.services.schemas import InTextCitation, ParsedReference


def _reference(reference_id: str, author: str, year: str, title: str, **updates):
    return ParsedReference(
        reference_id=reference_id,
        author=author,
        year=year,
        title=title,
        raw_ref=f"{author}. ({year}). {title}.",
        citation_key=f"{author.split(',')[0]}{year}",
        **updates,
    )


def _citation(**updates):
    values = {
        "text": "A bounded claim (Smith, 2020).",
        "citation_marker": "(Smith, 2020)",
        "passage_start": 10,
        "passage_end": 40,
        "marker_start": 16,
        "marker_end": 29,
        "reference_ids": ["ref-1"],
        "link_status": "linked",
    }
    values.update(updates)
    return InTextCitation(**values)


def test_consistency_keeps_definite_and_bounded_review_findings_separate():
    references = [
        _reference("ref-1", "Smith, J", "2020", "First"),
        _reference("ref-2", "Jones, A", "2021", "Unused"),
    ]
    citations = [
        _citation(),
        _citation(
            text="Another bounded claim (Missing, 2019).",
            citation_marker="(Missing, 2019)",
            passage_start=50,
            passage_end=90,
            reference_ids=[],
            link_status="missing_reference",
        ),
    ]

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="apa",
        references=references,
        citations=citations,
    )

    assert result.status == "complete"
    assert result.formatting_status == "not_assessed"
    assert result.formatting_reason_codes == [
        "normalized_text_lacks_layout_and_typography"
    ]
    assert result.finding_counts == {
        "missing_reference_entry": 1,
        "reference_not_linked_in_extracted_citations": 1,
    }
    missing, unlinked = result.findings
    assert missing.level == "neutral"
    assert missing.marker_text_sha256 is not None
    assert missing.passage_start == 50
    assert unlinked.level == "neutral"
    assert "bounded to current citation extraction" in result.limitations[0]


def test_consistency_records_ambiguous_candidates_as_observed_references():
    references = [
        _reference("ref-a", "Smith, J", "2020a", "First"),
        _reference("ref-b", "Smith, J", "2020b", "Second"),
    ]
    citation = _citation(
        reference_ids=[],
        candidate_reference_ids=["ref-a", "ref-b"],
        link_status="ambiguous",
    )

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="apa",
        references=references,
        citations=[citation],
    )

    assert result.finding_counts == {"ambiguous_reference_link": 1}
    assert result.findings[0].candidate_reference_ids == ["ref-a", "ref-b"]


def test_consistency_preserves_unbound_missing_marker_without_invalid_coordinates():
    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="apa",
        references=[],
        citations=[
            _citation(
                reference_ids=[],
                link_status="missing_reference",
                passage_start=-1,
                passage_end=-1,
            )
        ],
    )

    finding = result.findings[0]
    assert finding.finding_type == "missing_reference_entry"
    assert finding.passage_start is None
    assert finding.passage_end is None
    assert finding.marker_text_sha256 is None


def test_consistency_detects_duplicate_entry_key_and_parse_review():
    first = _reference("ref-a", "Smith, J", "2020", "First", doi="10.1/example")
    second = _reference(
        "ref-b",
        "Smith, J",
        "2020",
        "First",
        doi="10.1/example",
        needs_review=True,
        extraction_method="llm",
    )

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="mla",
        references=[first, second],
        citations=[_citation(reference_ids=["ref-a", "ref-b"])],
    )

    assert result.finding_counts == {
        "duplicate_citation_key": 1,
        "duplicate_reference_entry": 1,
        "reference_parse_review": 1,
    }
    duplicate = next(
        item for item in result.findings if item.finding_type == "duplicate_reference_entry"
    )
    assert duplicate.level == "attention"
    assert duplicate.reason_code == "same_normalized_reference_entry"


def test_consistency_surfaces_near_identical_entries_as_likely_repetition():
    first = ParsedReference(
        reference_id="ref-a",
        author="Smith, John",
        year="2021",
        title="Media Regulation in Britain",
        raw_ref=(
            "Smith, John. Media Regulation in Britain. Media Studies, 2021."
        ),
        citation_key="Smith2021",
    )
    second = ParsedReference(
        reference_id="ref-b",
        author="Smith, J.",
        year="2021",
        title="Media Regulation in Britain",
        raw_ref=(
            "Smith, J. Media Regulation in Britain. Media Studies Quarterly, 2021."
        ),
        citation_key="Smith2021",
    )

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="mla",
        references=[first, second],
        citations=[_citation(reference_ids=["ref-a", "ref-b"])],
    )

    repetition = next(
        item
        for item in result.findings
        if item.finding_type == "likely_reference_repetition"
    )
    assert repetition.level == "attention"
    assert repetition.reason_code == "same_author_year_near_identical_title"
    assert repetition.reference_ids == ["ref-a", "ref-b"]
    assert repetition.explanation == (
        "These reference entries appear to repeat the same source."
    )


def test_consistency_keeps_ordinary_title_similarity_neutral():
    first = _reference(
        "ref-a", "Smith, John", "2021", "Media Regulation in Britain"
    )
    second = _reference(
        "ref-b", "Smith, John", "2021", "British Media Regulation after 2020"
    )

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="mla",
        references=[first, second],
        citations=[_citation(reference_ids=["ref-a", "ref-b"])],
    )

    assert not any(
        item.finding_type == "likely_reference_repetition"
        for item in result.findings
    )


def test_consistency_does_not_duplicate_exact_duplicate_finding():
    first = _reference("ref-a", "Smith, John", "2021", "A Substantive Shared Title")
    second = _reference("ref-b", "Smith, John", "2021", "A Substantive Shared Title")

    result = assess_reference_consistency(
        paper_version_id="paper-v1",
        citation_format="mla",
        references=[first, second],
        citations=[_citation(reference_ids=["ref-a", "ref-b"])],
    )

    assert result.finding_counts["duplicate_reference_entry"] == 1
    assert "likely_reference_repetition" not in result.finding_counts


def test_paper_extraction_attaches_consistency_assessment(monkeypatch):
    from app.services import paper_extraction

    reference = _reference("ref-1", "Smith, J", "2020", "First")
    citation = _citation()
    paper = "A bounded claim (Smith, 2020).\n\nReferences\nSmith, J. (2020). First."
    monkeypatch.setattr(
        paper_extraction, "extract_and_parse_references", lambda *_a, **_k: [reference]
    )
    monkeypatch.setattr(
        paper_extraction, "extract_citations", lambda *_a, **_k: [citation]
    )

    artifact = paper_extraction.extract_paper_evidence(
        paper,
        paper_version_id="paper-v1",
        format_hint="apa",
        use_llm_atomizer=False,
    )

    assert artifact.reference_consistency is not None
    assert artifact.reference_consistency.status == "complete"
    assert artifact.reference_consistency.findings == []
