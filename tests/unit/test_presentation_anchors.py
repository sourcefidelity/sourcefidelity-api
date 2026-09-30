"""Citation-to-page anchor tests for the fixed presentation surface."""

import fitz

from app.services.evidence_report import _paper_location
from app.services.presentation_anchors import bind_citations_to_pdf
from app.services.schemas import InTextCitation


def _pdf_with_text(*lines: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 24
    value = document.tobytes()
    document.close()
    return value


def _citation(text: str, start: int = 10) -> InTextCitation:
    return InTextCitation(
        text=text,
        citation_marker="(Smith, 2020)",
        passage_start=start,
        passage_end=start + len(text),
    )


def test_unique_exact_tokens_create_stable_page_geometry():
    sentence = "Careful verification improves accuracy (Smith, 2020)."
    artifact = bind_citations_to_pdf(
        _pdf_with_text(sentence), citations=[_citation(sentence)]
    )
    assert artifact.status == "complete"
    assert artifact.matched_citation_count == 1
    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "matched"
    assert anchor.page_indexes == [0]
    assert anchor.rectangles[0].x1 > anchor.rectangles[0].x0


def test_missing_sentence_space_uses_exact_character_boundary():
    sentence = 'The report describes the result (Smith, 2020).'
    content = _pdf_with_text(sentence+'Another sentence follows.')
    artifact = bind_citations_to_pdf(content, citations=[_citation(sentence)])
    assert artifact.matched_citation_count == 1
    with fitz.open(stream=content,filetype='pdf') as doc:
        next_word=doc[0].search_for('Another')[0]
        assert artifact.anchors[0].rectangles[-1].x1 <= next_word.x0+.01


def test_duplicate_detections_of_one_exact_span_share_one_surface_anchor():
    sentence = "Careful verification improves accuracy (Smith, 2020)."
    citation = _citation(sentence)
    duplicate = citation.model_copy(
        update={"citation_marker": "Smith (2020)", "reference_ids": ["ref-2"]}
    )

    artifact = bind_citations_to_pdf(
        _pdf_with_text(sentence), citations=[citation, duplicate]
    )

    assert artifact.citation_count == 1
    assert artifact.matched_citation_count == 1
    assert len(artifact.anchors) == 1


def test_repeated_sentence_fails_closed_as_ambiguous():
    sentence = "Repeated evidence (Smith, 2020)."
    artifact = bind_citations_to_pdf(
        _pdf_with_text(sentence, sentence), citations=[_citation(sentence)]
    )
    anchor = artifact.anchors[0]
    assert artifact.status == "not_assessed"
    assert anchor.mapping_status == "ambiguous"
    assert anchor.match_count == 2
    assert anchor.rectangles == []


def test_visible_line_end_hyphenation_maps_to_one_semantic_token():
    sentence = "The representation is supported (Smith, 2020)."
    artifact = bind_citations_to_pdf(
        _pdf_with_text("The represen-", "tation is supported (Smith, 2020)."),
        citations=[_citation(sentence)],
    )
    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "matched"
    assert len(anchor.rectangles) == 2


def test_unique_compact_characters_resolve_pdf_word_segmentation_only():
    sentence = "SourceFidelity preserves bounded evidence (Smith, 2020)."
    artifact = bind_citations_to_pdf(
        _pdf_with_text("Source Fidelity preserves bounded evidence (Smith, 2020)."),
        citations=[_citation(sentence)],
    )
    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "matched"
    assert anchor.mapping_method == "compact_alphanumeric"


def test_page_spanning_citation_skips_running_header_but_not_body_words():
    sentence = "The finding begins here and continues with evidence (Smith, 2020)."
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 700), "The finding begins here and")
    second = document.new_page()
    second.insert_text((72, 50), "[2] Author Name")
    second.insert_text((72, 110), "continues with evidence (Smith, 2020).")
    pdf = document.tobytes()
    document.close()

    artifact = bind_citations_to_pdf(pdf, citations=[_citation(sentence)])

    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "matched"
    assert anchor.mapping_method == "normalized_tokens_with_margin_skip"
    assert anchor.page_indexes == [0, 1]
    assert len(anchor.rectangles) == 2


def test_page_spanning_citation_does_not_skip_intervening_body_prose():
    sentence = "The finding begins here and continues with evidence (Smith, 2020)."
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 700), "The finding begins here and")
    second = document.new_page()
    second.insert_text((72, 50), "[2] Author Name")
    second.insert_text((72, 110), "unrelated body prose")
    second.insert_text((72, 140), "continues with evidence (Smith, 2020).")
    pdf = document.tobytes()
    document.close()

    artifact = bind_citations_to_pdf(pdf, citations=[_citation(sentence)])

    assert artifact.anchors[0].mapping_status == "not_matched"


def test_unique_paragraph_context_disambiguates_repeated_citation_text():
    sentence = "Repeated evidence (Smith, 2020)."
    first = f"First unique discussion contains {sentence}"
    second = f"Second distinct discussion contains {sentence}"
    artifact = bind_citations_to_pdf(
        _pdf_with_text(first, second),
        citations=[_citation(sentence)],
        paragraphs=[first],
    )
    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "matched"
    assert anchor.mapping_method == "paragraph_context_disambiguation"
    assert anchor.localization_level == "exact_rectangle"


def test_exact_paragraph_context_can_supply_page_without_guessing_rectangle():
    paragraph = "Context words around AB that uniquely locate the paragraph."
    citation = _citation("AB")
    artifact = bind_citations_to_pdf(
        _pdf_with_text("Context words around A B that uniquely locate the paragraph."),
        citations=[citation],
        paragraphs=[paragraph],
    )
    anchor = artifact.anchors[0]
    assert anchor.mapping_status == "not_matched"
    assert anchor.localization_level == "page_only"
    assert anchor.page_indexes == [0]
    assert anchor.rectangles == []


def test_report_location_contract_distinguishes_exact_page_and_structural_levels():
    surface = {
        "citation_anchors": [
            {
                "passage_start": 10,
                "passage_end": 20,
                "localization_level": "exact_rectangle",
                "page_indexes": [1],
                "rectangles": [
                    {"page_index": 1, "x0": 10, "y0": 20, "x1": 30, "y1": 40}
                ],
            },
            {
                "passage_start": 30,
                "passage_end": 40,
                "localization_level": "page_only",
                "page_indexes": [2],
                "rectangles": [],
            },
            {
                "passage_start": 50,
                "passage_end": 60,
                "localization_level": "structural_only",
            },
        ]
    }
    exact = _paper_location(surface, passage_start=10, passage_end=20)
    page = _paper_location(surface, passage_start=30, passage_end=40)
    structural = _paper_location(surface, passage_start=50, passage_end=60)

    assert exact["action"]["label"] == "View in paper"
    assert exact["rectangles"]
    assert page["action"]["label"] == "View paper"
    assert page["rectangles"] == []
    assert structural["action"] is None
    assert structural["reason_code"] == (
        "paragraph_known_presentation_location_unresolved"
    )
