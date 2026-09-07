"""Phase 3.8 evidence artifact and authorization-bound passage regressions."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.relationship_signal import NLIScore
from app.services.schemas import CitationMarkerMember, InTextCitation
from app.services.source_repository import AdmissionRequest, WorkIdentity, admit_representation
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    _PassageCandidate,
    _bm25_concept_candidates,
    _bounded_concept_rescue_candidates,
    _consolidate_nested_passage_entries,
    _candidate_union_candidates,
    _context_bounds,
    _context_bounds_on_page,
    _PdfLayoutSpan,
    _pdf_layout_spans,
    _pdf_reading_order_text_and_spans,
    _normalize_pdf_extracted_text,
    _normalize_pdf_page_label,
    _ocr_token_sequence_match,
    _visible_pdf_page_label,
    _pdf_structural_spans,
    _page_text_blocks,
    _passage_boundary_status,
    _passage_overlap_ratio,
    _retrieve_candidates,
    _SourcePage,
    _SourceStructuralSpan,
    _source_blocks,
    _text_blocks,
    ClaimAntecedentDependency,
    ClaimEvidence,
    CoverageLevel,
    EvidenceAuthorizationError,
    RelationshipStatus,
    PassageRelevanceGateEvidence,
    SourcePassageEvidence,
    VerificationVerdict,
    authorize_representation,
    attach_candidate_passage_retrieval,
    attach_local_semantic_retrieval_rescue,
    build_passage_evidence,
    claim_evidence_from_citation,
    passage_role_from_text,
)


class _SemanticRescueScorer:
    model_id = "local-test-nli"
    model_revision = "revision-1"

    def score_pairs(self, premises, hypotheses):
        assert len(premises) == len(hypotheses)
        return [
            NLIScore(entailment=0.94, neutral=0.04, contradiction=0.02)
            if "machine translation strategically" in premise.casefold()
            else NLIScore(entailment=0.05, neutral=0.90, contradiction=0.05)
            for premise in premises
        ]


def _ocr_authorized_source(
    text: str, *, label: str | None = "685"
) -> AuthorizedRepresentation:
    content = text.encode("utf-8")
    return AuthorizedRepresentation(
        representation_id="ocr-derivative",
        canonical_work_id="zeitz-work",
        content_object_id="development-only",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        representation_kind="plain_text",
        media_type="text/plain",
        provenance="development_local_ocr_shadow",
        scope_type="verification_run",
        scope_id="ocr-shadow-v1",
        identity_verdict="verified",
        identity_confidence=0.8,
        completeness_verdict="complete",
        text_quality="scan_ocr",
        edition_or_version="JSTOR scan",
        created_at=datetime.now(timezone.utc),
        admitted_at=None,
        verification_run_id="ocr-shadow-v1",
        parent_content_sha256="a" * 64,
        derivation_method="local-pdf-ocr-derivative-v1",
        derivation_manifest_sha256="b" * 64,
        page_labels=(label,),
    )


def test_visible_pdf_page_label_uses_one_standalone_furniture_line() -> None:
    text = "84\nQUALITATIVE SOCIOLOGY\nBody prose."
    spans = (
        _SourceStructuralSpan(start=0, end=24, role="page_furniture"),
    )

    assert _visible_pdf_page_label(text, spans) == "84"


def test_visible_pdf_page_label_rejects_ambiguous_furniture_numbers() -> None:
    text = "84\nHeader\n2024\nBody prose."
    spans = (
        _SourceStructuralSpan(start=0, end=17, role="page_furniture"),
    )

    assert _visible_pdf_page_label(text, spans) is None

def test_nested_same_page_passages_merge_channels_and_keep_broader_span() -> None:
    shorter = _PassageCandidate(
        page_index=0,
        page_label="10",
        start=100,
        end=500,
        text="a" * 400,
        method="whole_citation_context",
        score=0.91,
    )
    broader = _PassageCandidate(
        page_index=0,
        page_label="10",
        start=100,
        end=650,
        text="a" * 400 + "b" * 150,
        method="lexical_overlap",
        score=0.83,
    )
    other_page = _PassageCandidate(
        page_index=1,
        page_label="11",
        start=100,
        end=650,
        text="c" * 550,
        method="lexical_overlap",
        score=0.80,
    )

    consolidated = _consolidate_nested_passage_entries(
        [
            (shorter, {"whole_citation_context"}),
            (broader, {"candidate_lexical"}),
            (other_page, {"candidate_lexical"}),
        ]
    )

    assert len(consolidated) == 2
    merged, channels = consolidated[0]
    assert (merged.page_index, merged.start, merged.end) == (0, 100, 650)
    assert merged.text == broader.text
    assert merged.score == shorter.score
    assert channels == {"whole_citation_context", "candidate_lexical"}
    assert consolidated[1][0].page_index == 1


def test_substantial_partial_overlap_keeps_one_window_and_records_provenance() -> None:
    first = _PassageCandidate(
        page_index=14,
        page_label="15",
        start=0,
        end=1_741,
        text="a" * 1_741,
        method="whole_citation_context",
        score=0.92,
    )
    overlapping = _PassageCandidate(
        page_index=14,
        page_label="15",
        start=653,
        end=2_436,
        text="b" * 1_783,
        method="lexical_overlap",
        score=0.86,
    )
    later = _PassageCandidate(
        page_index=14,
        page_label="15",
        start=1_972,
        end=3_672,
        text="c" * 1_700,
        method="lexical_overlap",
        score=0.90,
    )
    later_overlap = _PassageCandidate(
        page_index=14,
        page_label="15",
        start=2_584,
        end=3_948,
        text="d" * 1_364,
        method="lexical_overlap",
        score=0.84,
    )

    consolidated = _consolidate_nested_passage_entries(
        [
            (first, {"whole_citation_context"}),
            (overlapping, {"candidate_lexical"}),
            (later, {"candidate_facet_1_lexical"}),
            (later_overlap, {"candidate_concept_rescue"}),
        ]
    )

    assert len(consolidated) == 2
    assert all(
        _passage_overlap_ratio(left[0], right[0]) < 0.50
        for left, right in zip(consolidated, consolidated[1:])
    )
    provenance = [set(item[0].consolidated_from_spans) for item in consolidated]
    assert {(0, 1_741), (653, 2_436)} in provenance
    assert {(1_972, 3_672), (2_584, 3_948)} in provenance
    assert any(item.start == 653 and item.end == 2_436 for item, _channels in consolidated)
    assert all("substantial_overlap_consolidated" in channels for _item, channels in consolidated)


def test_clear_two_column_pdf_uses_left_then_right_reading_order() -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox(
        fitz.Rect(330, 80, 560, 260),
        "RIGHT FIRST BLOCK " + "right evidence words " * 16,
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(330, 300, 560, 500),
        "RIGHT SECOND BLOCK " + "right conclusion words " * 16,
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(40, 80, 270, 260),
        "LEFT FIRST BLOCK " + "left evidence words " * 16,
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(40, 300, 270, 500),
        "LEFT SECOND BLOCK " + "left conclusion words " * 16,
        fontsize=9,
    )

    text, spans, reordered = _pdf_reading_order_text_and_spans(page, 0)

    assert reordered is True
    assert text.index("LEFT FIRST BLOCK") < text.index("LEFT SECOND BLOCK")
    assert text.index("LEFT SECOND BLOCK") < text.index("RIGHT FIRST BLOCK")
    assert text.index("RIGHT FIRST BLOCK") < text.index("RIGHT SECOND BLOCK")
    assert "".join(span.text for span in spans) == text


def test_literal_utf16_pdf_page_label_is_decoded() -> None:
    assert _normalize_pdf_page_label("<FEFF0053006500630032003A>100") == "Sec2:100"
    assert _normalize_pdf_page_label("15") == "15"


def test_passage_roles_exclude_references_and_citation_only_notes_but_keep_body_prose():
    body = (
        "Examination of media representations suggests inaccurate portrayals. "
        "Several studies reach related conclusions (Jones, 2009; Smith, 2011; "
        "Draaisma, 2012). The present discussion then explains the substantive "
        "difference between those findings."
    )
    references = (
        "References\n"
        "Jones, A. (2009). Media representations and disability. Journal One.\n"
        "Smith, B. (2011). Public understanding. Journal Two.\n"
        "Draaisma, C. (2012). Film stereotypes. Journal Three."
    )
    notes = (
        "1 Smith (2009) discusses the earlier statute.\n"
        "2 Jones (2010) supplies the historical citation.\n"
        "3 Brown (2011) records the parallel proceeding.\n"
        "4 White (2012) provides another citation."
    )

    assert passage_role_from_text(body) == "body_prose"
    assert passage_role_from_text(references) == "reference_list"
    assert passage_role_from_text(notes) == "citation_notes"


def test_table_rows_and_demographic_prose_do_not_look_like_legal_citations():
    text = (
        "2 Free response analyses describe the results in full.\n"
        "1 Poor social skills 92 56.1\n"
        "2 Introverted and withdrawn 52 31.7\n"
        "3 Poor communication 48 29.3\n"
        "4 Difficult personality 46 28.0\n"
        "5 Poor emotional intelligence 38 23.2\n"
        "6 Special abilities 30 18.3\n"
        "The survey was completed by 42 students, including 40 female "
        "participants, using traits identified by Nario-Redmond (2010). "
        "The article then explains the study method and results."
    )

    assert passage_role_from_text(text) == "body_prose"


def test_short_body_sentence_about_copyright_is_not_publication_furniture():
    text = "The article explains how copyright constraints affect documentary reuse."

    assert passage_role_from_text(text) != "publication_metadata"


def test_reference_heading_excludes_later_blank_line_blocks_and_pages():
    pages = [
        _SourcePage(
            index=0,
            label="1",
            text=(
                "Substantive language discussion appears in the article body. "
                "It contains several complete explanatory sentences.\n"
                "References\n"
                "Jones, A. (2009). Language and film."
            ),
        ),
        _SourcePage(
            index=1,
            label="2",
            text="Smith, B. (2011). Translation and culture.\n\nBrown, C. (2012). Loneliness.",
        ),
    ]

    blocks = _source_blocks(pages)
    roles = [role for _page, _start, _end, _text, role in blocks]

    assert roles[0] == "body_prose"
    assert roles[1:] == ["reference_list", "reference_list", "reference_list"]


def test_table_of_contents_reference_line_does_not_exclude_later_pages():
    pages = [
        _SourcePage(
            index=0,
            label="i",
            text="Contents\nIntroduction 1\nReferences 57\nAppendix 63",
        ),
        _SourcePage(
            index=1,
            label="1",
            text=(
                "The report begins with a substantive account of market entry. "
                "It then explains how compliance costs affect smaller firms."
            ),
        ),
    ]

    later_roles = [
        role
        for page, _start, _end, _text, role in _source_blocks(pages)
        if page.index == 1
    ]

    assert later_roles == ["body_prose"]


def test_chapter_reference_section_stops_before_following_chapter():
    pages = [
        _SourcePage(
            index=0,
            label="56",
            text=(
                "References\n"
                "Jones, A. (2009). Market entry.\n"
                "Smith, B. (2011). Regulatory costs."
            ),
        ),
        _SourcePage(
            index=1,
            label="57",
            text=(
                "Brown, C. (2012). Competition policy.\n"
                "White, D. (2015). Firm innovation."
            ),
        ),
        _SourcePage(
            index=2,
            label="58",
            text=(
                "The next chapter examines a different part of the economy. "
                "Its evidence concerns new firms and changing market structure."
            ),
        ),
    ]

    roles_by_page: dict[int, list[str]] = {}
    for page, _start, _end, _text, role in _source_blocks(pages):
        roles_by_page.setdefault(page.index, []).append(role)

    assert set(roles_by_page[0]) == {"reference_list"}
    assert set(roles_by_page[1]) == {"reference_list"}
    assert roles_by_page[2] == ["body_prose"]


def test_oversized_source_block_preserves_middle_paragraph_in_complete_windows():
    prefix = " ".join(
        f"Background sentence {index} explains an unrelated preliminary issue in sufficient detail."
        for index in range(12)
    )
    target_sentences = [
        "The relevant paragraph introduces one precise relationship between the works.",
        "It explains that the characters combine traits previously treated as incompatible.",
        "Their presentation permits audiences to reconcile those apparent contradictions.",
        "The films consequently reflect changing expectations without erasing tradition.",
        "This combination is the paragraph's central factual relationship.",
        "The final sentence completes that bounded discussion for the reader.",
    ]
    target = " ".join(target_sentences)
    suffix = " ".join(
        f"Following sentence {index} examines a separate historical topic in sufficient detail."
        for index in range(12)
    )
    text = f"{prefix} {target} {suffix}"

    blocks = _text_blocks(text)

    assert len(text) > 1_800
    assert len(blocks) > 1
    assert any(target in block_text for _start, _end, block_text in blocks)
    for start, end, block_text in blocks:
        assert text[start:end] == block_text
        assert len(block_text) <= 1_800
        assert block_text[0].isupper()
        assert block_text.endswith(".")


def test_oversized_source_windows_overlap_instead_of_splitting_at_pdf_line_breaks():
    text = " ".join(
        f"Sentence {index} preserves complete evidence despite an extracted\nvisual line break."
        for index in range(40)
    )

    blocks = _text_blocks(text)

    assert len(blocks) > 2
    assert all(text[start:end] == block for start, end, block in blocks)
    assert all(block.endswith(".") for _start, _end, block in blocks)
    assert all(
        next_start < current_end
        for (_current_start, current_end, _current_text),
        (next_start, _next_end, _next_text) in zip(blocks, blocks[1:])
    )


def test_pdf_block_boundary_inside_sentence_is_merged_before_windowing():
    text = (
        "A complete opening sentence. The next sentence continues at the extracted"
        "\n\n"
        "column boundary and ends only in the second block. A final sentence follows."
    )

    blocks = _text_blocks(text)

    assert blocks == [(0, len(text), text)]


def test_passage_boundary_status_exposes_fragment_or_nonprose_output():
    assert _passage_boundary_status("A complete sentence.") == "sentence_complete"
    assert (
        _passage_boundary_status("continuation from a prior column. A full sentence.")
        == "bounded_fragment_or_nonprose"
    )
    assert (
        _passage_boundary_status("a bounded continuation without terminal punctuation")
        == "bounded_fragment_or_nonprose"
    )


def test_complete_paragraph_boundary_does_not_merge():
    first = "The first paragraph ends completely."
    second = "The second paragraph begins independently."
    text = first + "\n\n" + second

    blocks = _text_blocks(text)

    assert [block for _start, _end, block in blocks] == [first, second]


def test_numbered_section_heading_is_a_hard_passage_boundary():
    prior = (
        "The preceding discussion explains an international trade mechanism. "
        "It does not address the next section's protected agricultural policy."
    )
    heading = "6. Criticism of the corn laws"
    following = (
        "Ricardo discussed agricultural protection in a separately titled work. "
        "The section then develops that subject independently."
    )
    text = f"{prior}\n{heading}\n{following}"

    blocks = _text_blocks(text)

    assert [block for _start, _end, block in blocks] == [
        prior,
        f"{heading}\n{following}",
    ]
    assert all(text[start:end] == block for start, end, block in blocks)


def test_decimal_numbered_section_heading_is_a_hard_passage_boundary():
    prior = "An earlier section closes with a complete substantive sentence."
    heading = "3.4 Institutional Reputation and TNHE Value"
    following = "The new section discusses institutional reputation directly."
    text = f"{prior}\n{heading}\n{following}"

    blocks = _text_blocks(text)

    assert [block for _start, _end, block in blocks] == [
        prior,
        f"{heading}\n{following}",
    ]


def test_exact_phrase_context_does_not_cross_numbered_section_heading():
    text = (
        "Earlier discussion describes a different problem and ends here.\n"
        "3.4 Institutional Reputation and TNHE Value\n"
        "Machine translation can damage the institution's reputation and lower "
        "the value of its degree."
    )
    page = _SourcePage(index=0, label="9", text=text)
    target = "damage the institution's reputation"
    start = text.index(target)

    bounded_start, bounded_end = _context_bounds_on_page(
        page, start, start + len(target)
    )

    context = text[bounded_start:bounded_end]
    assert context.startswith("3.4 Institutional Reputation and TNHE Value")
    assert "Earlier discussion" not in context


def test_pdf_control_debris_does_not_discard_recovered_body_windows():
    body = " ".join(
        f"Body sentence {index} retains exact source coordinates and complete punctuation."
        for index in range(36)
    )
    text = body + "\n" + ("\x00" * 20) + " damaged footer without a stable mapping"

    blocks = _text_blocks(text)

    assert blocks
    assert all(text[start:end] == block for start, end, block in blocks)
    assert any("Body sentence 20" in block for _start, _end, block in blocks)
    assert all("damaged footer" not in block for _start, _end, block in blocks)


def test_pdf_control_normalization_preserves_character_coordinates():
    raw = "First\x03sentence.\nSecond\x11sentence."

    normalized, substitutions = _normalize_pdf_extracted_text(raw)

    assert normalized == "First sentence.\nSecond sentence."
    assert substitutions == 2
    assert len(normalized) == len(raw)


def test_pathological_single_sentence_respects_hard_passage_limit():
    text = "A " + " ".join(f"term{index}" for index in range(700)) + "."

    blocks = _text_blocks(text)

    assert len(text) > 1_800
    assert len(blocks) > 1
    assert all(text[start:end] == block for start, end, block in blocks)
    assert all(0 < len(block) <= 1_800 for _start, _end, block in blocks)


def test_pdf_layout_mapping_keeps_repeated_blocks_on_distinct_coordinates():
    document = fitz.open()
    try:
        page = document.new_page()
        repeated = "Repeated journal header"
        page.insert_text((72, 72), repeated)
        page.insert_text((72, 144), repeated)
        page_text = page.get_text("text")

        spans = _pdf_layout_spans(page, page_text, 0)
    finally:
        document.close()

    repeated_spans = [span for span in spans if span.text.strip() == repeated]
    assert len(repeated_spans) == 2
    assert repeated_spans[0].end <= repeated_spans[1].start
    assert all(page_text[span.start:span.end] == span.text for span in repeated_spans)


def test_exact_match_context_uses_sentence_complete_overlapping_window():
    sentences = [
        f"Sentence {index} provides enough surrounding context for a bounded retrieval window."
        for index in range(40)
    ]
    sentences[20] = "The exact target phrase appears in this complete source sentence."
    text = " ".join(sentences)
    target_start = text.index("exact target phrase")
    target_end = target_start + len("exact target phrase")

    start, end = _context_bounds(text, target_start, target_end)

    assert start <= target_start < target_end <= end
    assert len(text[start:end]) <= 1_800
    assert text[start].isupper()
    assert text[start:end].endswith(".")


def _layout_span(
    page: int,
    start: int,
    text: str,
    *,
    y0: float,
    y1: float,
    x0: float = 60.0,
    x1: float = 500.0,
) -> _PdfLayoutSpan:
    return _PdfLayoutSpan(
        page_index=page,
        start=start,
        end=start + len(text),
        text=text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        page_width=560.0,
        page_height=800.0,
    )


def test_layout_roles_separate_repeated_furniture_biography_and_notes():
    header = "Example Journal 30 (2026)"
    body = "Substantive article prose explains the relevant relationship in full."
    biography = "Morgan Author is Lecturer in Film Studies at Example University."
    address = "Example City, EX1 2AB, Country."
    notes = "1 Earlier evidence appears in Smith (1996).\n2 A related source is Jones (1998)."
    note_continuation = "The second note continues with complete bibliographic detail."
    download = "This content downloaded from the journal archive"
    debris = "192.0.2.1 on a recorded access date"
    page_zero = [
        _layout_span(0, 0, header, y0=15, y1=25),
        _layout_span(0, 100, body, y0=180, y1=195),
        _layout_span(0, 500, biography, y0=500, y1=512),
        _layout_span(0, 565, address, y0=512, y1=524),
        _layout_span(0, 650, notes, y0=550, y1=570),
        _layout_span(0, 740, note_continuation, y0=570, y1=582),
        _layout_span(0, 850, download, y0=650, y1=662),
        _layout_span(0, 900, debris, y0=664, y1=676),
    ]
    page_one = [
        _layout_span(1, 0, header, y0=15, y1=25),
        _layout_span(1, 100, body, y0=180, y1=195),
        _layout_span(1, 850, download, y0=650, y1=662),
        _layout_span(1, 900, debris, y0=664, y1=676),
    ]

    structural = _pdf_structural_spans({0: page_zero, 1: page_one})
    roles_zero = [(item.start, item.end, item.role) for item in structural[0]]

    assert (0, len(header), "page_furniture") in roles_zero
    assert any(role == "author_biography" and start == 500 for start, _end, role in roles_zero)
    assert any(role == "citation_notes" and start == 650 for start, _end, role in roles_zero)
    assert any(role == "page_furniture" and start == 850 for start, _end, role in roles_zero)
    assert all(not (start == 100 and role != "body_prose") for start, _end, role in roles_zero)


def test_first_page_article_title_is_document_metadata_and_byline_is_excluded():
    title = "A Large Article Title\nwith a Second Line\nand a Third Line"
    byline = "NORA GILBERT"
    body = "Substantive article prose explains the relevant relationship in full."
    spans = [
        _layout_span(0, 0, title, y0=170, y1=240, x0=155, x1=400),
        _layout_span(0, 100, byline, y0=260, y1=272, x0=155, x1=240),
        _layout_span(0, 200, body, y0=330, y1=390, x0=155, x1=440),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert any(item.start == 0 and item.role == "document_metadata" for item in structural)
    assert any(item.start == 100 and item.role == "publication_metadata" for item in structural)
    assert all(not (item.start <= 200 < item.end) for item in structural)


def test_split_article_title_extends_backward_without_including_masthead():
    masthead = "Journal of Language Research Volume 12"
    first_title_line = "Strategic Use of Machine Translation in"
    final_title_line = "Across Multilingual Education"
    byline = "Morgan Scholar"
    body = "The study reports findings from student interviews."
    spans = [
        _layout_span(0, 0, masthead, y0=150, y1=160, x0=200, x1=470),
        _layout_span(
            0, 100, first_title_line, y0=210, y1=228, x0=145, x1=475
        ),
        _layout_span(
            0, 150, final_title_line, y0=242, y1=260, x0=240, x1=375
        ),
        _layout_span(0, 200, byline, y0=278, y1=290, x0=240, x1=340),
        _layout_span(0, 300, body, y0=340, y1=400, x0=80, x1=510),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert any(
        item.start == 100 and item.role == "document_metadata"
        for item in structural
    )
    assert any(
        item.start == 150 and item.role == "document_metadata"
        for item in structural
    )
    assert any(
        item.start == 200 and item.role == "publication_metadata"
        for item in structural
    )
    assert all(
        not (item.start <= 0 < item.end and item.role == "document_metadata")
        for item in structural
    )


def test_compact_first_page_title_is_document_metadata_and_byline_is_excluded():
    title = "Universally speaking: Lost in Translation and polyglot cinema"
    byline_affiliation = "Tessa Dwyer\nUniversity of Melbourne"
    abstract = (
        "Conceived from the start as a cultural form with international appeal, "
        "cinema bears a relationship to translation."
    )
    byline_start = len(title) + 1
    abstract_start = byline_start + len(byline_affiliation) + 1
    spans = [
        _layout_span(0, 0, title, y0=82.5, y1=96.3, x0=66.3, x1=383.4),
        _layout_span(
            0,
            byline_start,
            byline_affiliation,
            y0=107.9,
            y1=132.0,
            x0=66.3,
            x1=170.8,
        ),
        _layout_span(
            0,
            abstract_start,
            abstract,
            y0=156.4,
            y1=300.0,
            x0=66.3,
            x1=386.7,
        ),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert any(item.start == 0 and item.role == "document_metadata" for item in structural)
    assert any(
        item.start == byline_start and item.role == "publication_metadata"
        for item in structural
    )
    assert all(not (item.start <= abstract_start < item.end) for item in structural)


def test_article_opening_after_cover_separates_title_byline_and_affiliations():
    title = "The Impact of Protectionism on Cultural Industries\nand Imported Films"
    byline = "Jimmyn Parc a b, Patrick Messerlin c, and Kyuchan Kim d"
    affiliations = (
        "a Department of East Asian Studies, University of Malaya; "
        "b Institute of Communication Research; c Sciences Po Paris"
    )
    abstract = (
        "ABSTRACT\nHollywood studios have actively sought a large market. "
        "The article examines the resulting policy relationship."
    )
    byline_start = len(title)
    affiliation_start = byline_start + len(byline)
    abstract_start = affiliation_start + len(affiliations)
    spans = [
        _layout_span(1, 0, title, y0=100, y1=134, x0=100, x1=500),
        _layout_span(1, byline_start, byline, y0=139, y1=152, x0=100, x1=460),
        _layout_span(1, affiliation_start, affiliations, y0=160, y1=180, x0=100, x1=500),
        _layout_span(1, abstract_start, abstract, y0=199, y1=300, x0=100, x1=500),
    ]

    structural = _pdf_structural_spans({1: spans})[1]

    assert len(structural) == 3
    assert structural[0].start == 0
    assert structural[0].end == byline_start
    assert structural[0].role == "document_metadata"
    assert structural[1].start == byline_start
    assert structural[1].end == abstract_start
    assert structural[1].role == "publication_metadata"
    assert structural[2].start == abstract_start
    assert structural[2].end == abstract_start + len(abstract)
    assert structural[2].role == "abstract"


def test_shifted_title_and_name_without_scholarly_opening_support_remain_body():
    heading = "General Class Policies and Participation"
    apparent_name = "Attendance Requirements"
    body = (
        "Students should consult the course schedule and complete assigned work. "
        "This page contains instructional content rather than article metadata."
    )
    spans = [
        _layout_span(1, 0, heading, y0=100, y1=134, x0=48, x1=440),
        _layout_span(1, 100, apparent_name, y0=139, y1=152, x0=48, x1=300),
        _layout_span(1, 200, body, y0=170, y1=260, x0=48, x1=440),
    ]

    structural = _pdf_structural_spans({1: spans})[1]

    assert all(not (item.start <= 0 < item.end) for item in structural)
    assert all(not (item.start <= 100 < item.end) for item in structural)


def test_large_first_page_section_heading_without_byline_remains_body():
    heading = "Introduction and Historical Background"
    body = "Substantive article prose explains the relevant relationship in full."
    spans = [
        _layout_span(0, 0, heading, y0=170, y1=190, x0=155, x1=400),
        _layout_span(0, 100, body, y0=210, y1=270, x0=155, x1=440),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert all(not (item.start <= 0 < item.end) for item in structural)


def test_award_heading_and_award_name_are_not_treated_as_article_header():
    heading = "2009 Award Winners"
    award_name = "RAY AND PAT BROWNE AWARDS"
    recipient = "Gerard J. DeGroot"
    body = "The listed award recipients and works are substantive source content."
    spans = [
        _layout_span(0, 0, heading, y0=170, y1=190, x0=155, x1=400),
        _layout_span(0, 100, award_name, y0=210, y1=230, x0=155, x1=400),
        _layout_span(0, 200, recipient, y0=240, y1=252, x0=155, x1=280),
        _layout_span(0, 300, body, y0=270, y1=330, x0=155, x1=440),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert all(not (item.start <= 0 < item.end) for item in structural)
    assert all(not (item.start <= 100 < item.end) for item in structural)
    assert all(not (item.start <= 200 < item.end) for item in structural)


def test_work_title_and_distant_name_across_content_are_not_article_header():
    work_title = "A Kaleidoscopic History of a Disorderly Decade"
    intervening = "The work received the award for its historical contribution."
    person = "Gerard J. DeGroot"
    spans = [
        _layout_span(0, 0, work_title, y0=280, y1=311, x0=72, x1=355),
        _layout_span(0, 100, intervening, y0=330, y1=365, x0=72, x1=410),
        _layout_span(0, 200, person, y0=389, y1=406, x0=72, x1=250),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert all(not (item.start <= 0 < item.end) for item in structural)
    assert all(not (item.start <= 200 < item.end) for item in structural)


def test_marginal_note_continuation_does_not_cross_into_interleaved_body_column():
    note_one = "25 Letter from an archive (1936)."
    note_two = "26 Related archival correspondence (1937)."
    note_continuation = "The marginal note continues here."
    body_one = "The main article column contains substantive prose and analysis."
    body_two = "A second body paragraph remains ordinary retrievable evidence."
    spans = [
        _layout_span(0, 500, note_one, y0=300, y1=330, x0=45, x1=135),
        _layout_span(0, 600, note_two, y0=335, y1=365, x0=45, x1=135),
        _layout_span(0, 0, body_one, y0=315, y1=360, x0=155, x1=435),
        _layout_span(0, 700, note_continuation, y0=367, y1=380, x0=55, x1=130),
        _layout_span(0, 100, body_two, y0=365, y1=410, x0=155, x1=435),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert any(item.role == "citation_notes" and item.start == 500 for item in structural)
    assert all(not (item.start <= 0 < item.end) for item in structural)
    assert all(not (item.start <= 100 < item.end) for item in structural)
    ordered = sorted(structural, key=lambda item: item.start)
    assert all(left.end <= right.start for left, right in zip(ordered, ordered[1:]))


def test_numbered_lower_half_main_column_is_not_misclassified_as_notes():
    method_block = (
        "2 Experiment\n"
        "The study compares two systems reported in 2023 and 2024.\n"
        "3 Results\n"
        "The main-column analysis continues with substantive findings."
    )
    spans = [
        _layout_span(
            0,
            100,
            method_block,
            y0=390,
            y1=560,
            x0=70,
            x1=500,
        )
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert structural == ()


def test_numbered_two_column_section_is_not_a_marginal_note():
    introduction = (
        "1 Introduction\n"
        "The left article column discusses a 2023 evaluation and its findings."
    )
    spans = [
        _layout_span(
            0,
            100,
            introduction,
            y0=430,
            y1=700,
            x0=45,
            x1=190,
        )
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert structural == ()


def test_short_numbered_section_heading_does_not_seed_note_continuation():
    heading = "1 Introduction"
    body = (
        "The article body continues in the same column and discusses findings "
        "reported in 2023 without becoming citation-note evidence."
    )
    spans = [
        _layout_span(0, 100, heading, y0=590, y1=605, x0=65, x1=145),
        _layout_span(0, 115, body, y0=610, y1=700, x0=65, x1=275),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert structural == ()


def test_unique_margin_copyright_and_download_notice_are_furniture_not_body():
    copyright_line = "© The Author 2026. All rights reserved."
    copyright_discussion = (
        "The article explains how copyright constraints affect documentary reuse."
    )
    download_notice = "This content downloaded from an institutional archive"
    spans = [
        _layout_span(0, 0, copyright_line, y0=35, y1=48, x0=150, x1=410),
        _layout_span(0, 100, copyright_discussion, y0=250, y1=270, x0=150, x1=430),
        _layout_span(0, 300, download_notice, y0=120, y1=560, x0=458, x1=468),
    ]

    structural = _pdf_structural_spans({0: spans})[0]

    assert any(item.role == "page_furniture" and item.start == 0 for item in structural)
    assert any(item.role == "page_furniture" and item.start == 300 for item in structural)
    assert all(not (item.start <= 100 < item.end) for item in structural)


def test_structural_spans_split_body_without_losing_exact_coordinates():
    first = "First complete body sentence."
    biography = "Morgan Author is Lecturer in Film Studies."
    second = "Second complete body sentence."
    text = first + "\n" + biography + "\n" + second
    bio_start = text.index(biography)
    page = _SourcePage(
        index=0,
        label="1",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(
                start=bio_start,
                end=bio_start + len(biography),
                role="author_biography",
            ),
        ),
    )

    blocks = _page_text_blocks(page)

    assert any(role == "author_biography" and block == biography for _s, _e, block, role in blocks)
    assert any(role is None and first in block for _s, _e, block, role in blocks)
    assert any(role is None and second in block for _s, _e, block, role in blocks)
    assert all(text[start:end] == block for start, end, block, _role in blocks)


def test_exact_quotation_cannot_reenter_excluded_biography_span():
    biography = "Morgan Author is Lecturer in Film Studies."
    page = _SourcePage(
        index=0,
        label="1",
        text=biography,
        structural_spans=(
            _SourceStructuralSpan(
                start=0,
                end=len(biography),
                role="author_biography",
            ),
        ),
    )

    candidates = _retrieve_candidates(
        [page],
        claim_text=biography,
        claim_type="quotation",
        page_locator="",
        top_k=3,
    )

    assert candidates == []


def test_ocr_token_sequence_requires_long_unchanged_word_sequence() -> None:
    source = (
        "The coalition rejected the political center; because its members believed "
        "that durable reform required a much broader social movement."
    )
    punctuation_variant = (
        "The coalition rejected the political center because its members believed "
        "that durable reform required a much broader social movement"
    )
    changed_word = punctuation_variant.replace("broader", "narrower")

    match = _ocr_token_sequence_match(source, punctuation_variant)

    assert match is not None
    assert match[2] == "ocr_token_sequence"
    assert _ocr_token_sequence_match(source, changed_word) is None
    assert _ocr_token_sequence_match(source, "durable reform required") is None


def test_ocr_derivative_preserves_parent_provenance_and_locator() -> None:
    source_text = (
        "685\n"
        "The coalition rejected the political center; because its members believed "
        "that durable reform required a much broader social movement."
    )
    source = _ocr_authorized_source(source_text)
    quote = (
        "The coalition rejected the political center because its members believed "
        "that durable reform required a much broader social movement"
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(f'The author writes, "{quote}."', claim_type="quotation", page_locator="685"),
    )

    assert artifact.coverage.level is CoverageLevel.FULL_TEXT
    assert artifact.coverage.confidence.value == "medium"
    assert artifact.coverage.method == "validated_ocr_derivative_text_extraction"
    assert artifact.source_identity.parent_content_sha256 == "a" * 64
    assert artifact.source_identity.derivation_manifest_sha256 == "b" * 64
    assert artifact.quotation_check.outcome == "all_spans_ocr_token_sequence_match"
    assert artifact.locator_check.outcome == "located_span_matches_supplied_locator"
    assert artifact.passages[0].page_label == "685"
    assert artifact.passages[0].retrieval_method == "ocr_token_sequence"


def test_ocr_derivative_nonmatch_remains_not_assessable() -> None:
    source = _ocr_authorized_source(
        "685\nThe coalition rejected the political center because its members believed "
        "that durable reform required a much broader social movement."
    )
    changed_quote = (
        "The coalition rejected the political center because its members believed "
        "that durable reform required a much narrower social movement"
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            f'The author writes, "{changed_quote}."',
            claim_type="quotation",
            page_locator="685",
        ),
    )

    assert artifact.quotation_check.status == "not_assessable"
    assert artifact.locator_check.status == "not_assessable"


def test_ocr_derivative_missing_printed_page_label_never_uses_physical_page() -> None:
    source = _ocr_authorized_source(
        "The coalition rejected the political center because its members believed "
        "that durable reform required a much broader social movement.",
        label=None,
    )
    quote = (
        "The coalition rejected the political center because its members believed "
        "that durable reform required a much broader social movement"
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            f'The author writes, "{quote}."',
            claim_type="quotation",
            page_locator="1",
        ),
    )

    assert artifact.quotation_check.outcome == "all_spans_literal_match"
    assert artifact.locator_check.status == "not_assessable"
    assert artifact.locator_check.outcome == "ocr_page_label_unavailable"


def test_incomplete_ocr_derivative_provenance_fails_closed() -> None:
    source = _ocr_authorized_source(
        "685\nThis sufficiently long OCR page has complete source text."
    )
    source = replace(source, derivation_manifest_sha256=None)

    with pytest.raises(EvidenceAuthorizationError, match="provenance"):
        build_passage_evidence(source, claim=_claim("A source-dependent claim."))


def test_stale_broad_passage_cannot_reenter_newly_excluded_article_metadata():
    metadata = "Large Article Title\nMorgan Author\nExample University\n"
    abstract = (
        "ABSTRACT\nHollywood depends materially on the large market. "
        "The policy relationship is examined in this article."
    )
    text = metadata + abstract
    abstract_start = len(metadata)
    page = _SourcePage(
        index=1,
        label="2",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(
                start=0,
                end=abstract_start,
                role="publication_metadata",
            ),
            _SourceStructuralSpan(
                start=abstract_start,
                end=len(text),
                role="abstract",
            ),
        ),
    )
    stale = SourcePassageEvidence(
        passage_id="stale-opening",
        representation_id="representation",
        content_sha256="a" * 64,
        authorization_scope_type="verification_run",
        authorization_scope_id="run",
        page_index=1,
        page_label="2",
        character_start=0,
        character_end=len(text),
        text=text,
        retrieval_method="whole_citation_context",
        retrieval_score=0.9,
        passage_role="body_prose",
        boundary_status="sentence_complete",
    )

    candidates, _rescue, _facets = _candidate_union_candidates(
        [page],
        query_text="Hollywood depends materially on the large market.",
        page_locator="",
        broad_passages=[stale],
        top_k=3,
    )

    assert candidates
    assert all(candidate.start >= abstract_start for candidate, _channels in candidates)
    assert all(candidate.passage_role == "abstract" for candidate, _channels in candidates)
    assert all("whole_citation_context" not in channels for _candidate, channels in candidates)


def test_labelled_notes_are_searched_only_as_a_separate_fallback_channel():
    body = "The ordinary body discusses a wholly unrelated production history."
    notes = "1 The archive documents a distinctive cobalt process (1996).\n2 Further cobalt evidence appears in Jones (1998)."
    text = body + "\n" + notes
    note_start = text.index(notes)
    page = _SourcePage(
        index=0,
        label="1",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(
                start=note_start,
                end=note_start + len(notes),
                role="citation_notes",
            ),
        ),
    )

    ranked, rescue_applied, _facets = _candidate_union_candidates(
        [page],
        query_text="The archive documents a distinctive cobalt process.",
        page_locator="",
        broad_passages=[],
        top_k=3,
    )

    assert rescue_applied is True
    assert len(ranked) == 1
    candidate, channels = ranked[0]
    assert candidate.passage_role == "citation_notes"
    assert channels == ["candidate_note_fallback"]


def test_accepted_complete_facet_query_adds_distinct_retrieval_channel():
    first = (
        "The report describes the platform's general commercial strategy and "
        "its international audience."
    )
    facet_evidence = (
        "The service reduced creators' editorial autonomy when outside investors "
        "required approval of controversial material."
    )
    page = _SourcePage(
        index=0,
        label="1",
        text=first + "\n\n" + facet_evidence,
    )

    baseline, _baseline_rescue, baseline_facets = _candidate_union_candidates(
        [page],
        query_text="Foreign financing constrained what creators could say.",
        page_locator="",
        broad_passages=[],
        top_k=3,
    )
    enriched, _enriched_rescue, enriched_facets = _candidate_union_candidates(
        [page],
        query_text="Foreign financing constrained what creators could say.",
        page_locator="",
        broad_passages=[],
        top_k=3,
        accepted_facet_queries=[
            "Outside investors required approval of controversial material, reducing creators' editorial autonomy."
        ],
    )

    assert baseline_facets == []
    assert enriched_facets == [
        "Outside investors required approval of controversial material, reducing creators' editorial autonomy."
    ]
    assert not any(
        any(channel.startswith("candidate_facet_") for channel in channels)
        for _candidate, channels in baseline
    )
    matched = [
        channels
        for candidate, channels in enriched
        if facet_evidence in candidate.text
    ]
    assert matched
    assert any(channel.startswith("candidate_facet_1_") for channel in matched[0])


def test_concept_rescue_preserves_each_included_blocks_role():
    body = "The archive documents a distinctive cobalt production process."
    furniture = "Downloaded for authorized personal use."
    text = body + "\n" + furniture
    furniture_start = text.index(furniture)
    page = _SourcePage(
        index=0,
        label="1",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(
                start=furniture_start,
                end=furniture_start + len(furniture),
                role="page_furniture",
            ),
        ),
    )

    candidates = _bounded_concept_rescue_candidates(
        [page],
        query_text="The archive describes a cobalt production process.",
        page_locator="",
        top_k=3,
    )

    assert candidates
    assert all(candidate.passage_role not in {"page_furniture", "author_biography"} for candidate in candidates)


def test_bm25_concept_channel_ranks_rare_relationship_terms_without_noise():
    noise = [
        _SourcePage(
            index=index,
            label=str(index + 1),
            text=(
                "The study discusses general language learning, classrooms, "
                f"and educational technology in background section {index}."
            ),
        )
        for index in range(6)
    ]
    evidence = _SourcePage(
        index=6,
        label="7",
        text=(
            "Human-translated texts and machine-generated translations exhibit "
            "different error patterns that require different forms of revision."
        ),
    )
    furniture_text = "Downloaded from the journal platform for authorized use."
    furniture = _SourcePage(
        index=7,
        label="8",
        text=furniture_text,
        structural_spans=(
            _SourceStructuralSpan(
                start=0,
                end=len(furniture_text),
                role="page_furniture",
            ),
        ),
    )

    candidates = _bm25_concept_candidates(
        [*noise, evidence, furniture],
        query_text="Humans and machine translation make different kinds of errors.",
        page_locator="",
        top_k=5,
    )

    assert candidates
    assert candidates[0].page_index == 6
    assert candidates[0].method == "bm25_concept"
    assert all(candidate.passage_role != "page_furniture" for candidate in candidates)


def test_document_title_is_retrievable_without_reintroducing_byline_metadata():
    title = "Strategic Machine Translation Use in Multilingual Classrooms"
    byline = "Morgan Scholar\nExample University"
    body = "The article reports classroom observations and participant interviews."
    text = f"{title}\n{byline}\n{body}"
    title_end = len(title)
    byline_start = title_end + 1
    byline_end = byline_start + len(byline)
    page = _SourcePage(
        index=0,
        label="1",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(0, title_end, "document_metadata"),
            _SourceStructuralSpan(byline_start, byline_end, "publication_metadata"),
        ),
    )

    ranked, _rescue, _facets = _candidate_union_candidates(
        [page],
        query_text="The study concerns strategic machine translation use.",
        page_locator="",
        broad_passages=[],
        top_k=5,
    )

    assert any(
        candidate.passage_role == "document_metadata" and title in candidate.text
        for candidate, _channels in ranked
    )
    assert all(byline not in candidate.text for candidate, _channels in ranked)


def test_multi_source_member_title_has_explicit_document_level_channel():
    title = "Strategic Machine Translation Use in Multilingual Classrooms"
    byline = "Morgan Scholar\nExample University"
    body = "The article reports classroom observations and participant interviews."
    text = f"{title}\n{byline}\n{body}"
    title_end = len(title)
    byline_start = title_end + 1
    byline_end = byline_start + len(byline)
    page = _SourcePage(
        index=0,
        label="1",
        text=text,
        structural_spans=(
            _SourceStructuralSpan(0, title_end, "document_metadata"),
            _SourceStructuralSpan(
                byline_start, byline_end, "publication_metadata"
            ),
        ),
    )

    ranked, _rescue, _facets = _candidate_union_candidates(
        [page],
        query_text="Learners sometimes evaluate digital tools critically.",
        page_locator="",
        broad_passages=[],
        top_k=5,
        include_document_metadata=True,
    )

    assert any(
        candidate.text == title
        and candidate.passage_role == "document_metadata"
        and "candidate_document_metadata" in channels
        for candidate, channels in ranked
    )
    assert all(byline not in candidate.text for candidate, _channels in ranked)


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.objects.setdefault(key, file_bytes)
        return key

    def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _pdf(*pages: str) -> bytes:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 540, 760), text, fontsize=11)
    content = document.tobytes()
    document.close()
    return content


def _admit(
    session: Session,
    storage: MemoryStorage,
    content: bytes,
    *,
    scope_type: str = "personal_owner",
    scope_id: str = "owner-1",
    expires_at: datetime | None = None,
    kind: RepresentationKind = RepresentationKind.PDF,
    media_type: str = "application/pdf",
):
    record = admit_representation(
        session,
        storage,
        AdmissionRequest(
            work=WorkIdentity(
                title="A Verified Source",
                work_type="journal_article",
                doi="10.1234/verified-source",
                author="Scholar, A.",
                year="2025",
            ),
            representation=SourceRepresentation(
                kind=kind,
                media_type=media_type,
                content=content,
            ),
            provenance="instructor_upload",
            license_class="commercial_user_upload",
            scope_type=scope_type,
            scope_id=scope_id,
            identity_verdict="verified",
            identity_confidence=0.98,
            completeness_verdict="complete",
            cleanliness_verdict="clean",
            text_quality="digital",
            expires_at=expires_at,
            admitted_by="owner-1",
        ),
    )
    session.commit()
    return record


def _claim(
    text: str,
    *,
    claim_type: str = "paraphrase",
    page_locator: str = "",
) -> ClaimEvidence:
    return ClaimEvidence(
        claim_id="claim-1",
        paper_version_id="paper-v1",
        text=text,
        claim_type=claim_type,
        reference_ids=["ref-1"],
        page_locator=page_locator,
        passage_start=20,
        passage_end=20 + len(text),
    )


def test_local_semantic_rescue_is_bounded_additive_and_source_bound() -> None:
    paragraphs = [
        f"Students discuss language tools and unrelated classroom topic {index}."
        for index in range(12)
    ]
    paragraphs.append(
        "Students use machine translation strategically while drawing on their linguistic resources."
    )
    content = "\n\n".join(paragraphs).encode("utf-8")
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="semantic-source",
        canonical_work_id="semantic-work",
        content_object_id="semantic-object",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        representation_kind="plain_text",
        media_type="text/plain",
        provenance="authorized_upload",
        scope_type="personal_owner",
        scope_id="owner-1",
        identity_verdict="verified",
        identity_confidence=0.99,
        completeness_verdict="complete",
        text_quality="digital",
        edition_or_version=None,
        created_at=now,
        admitted_at=now,
    )
    claim_text = "Students deploy language tools deliberately (Smith, 2020)."
    claim = ClaimEvidence(
        claim_id="semantic-claim",
        paper_version_id="paper-v1",
        text=claim_text,
        reference_ids=["ref-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        passage_start=0,
        passage_end=len(claim_text),
    )
    artifact = build_passage_evidence(source, claim=claim)
    artifact = attach_verification_candidates(artifact)
    artifact = attach_citation_use_routes(artifact)
    artifact = attach_candidate_passage_retrieval(source, artifact)
    protected_ids = {passage.passage_id for passage in artifact.passages}
    original = artifact.candidate_passage_retrieval
    artifact = artifact.model_copy(
        update={
            "candidate_passage_retrieval": original.model_copy(
                update={
                    "selections": [
                        selection.model_copy(update={"passages": []})
                        for selection in original.selections
                    ]
                }
            ),
            "passage_relevance": PassageRelevanceGateEvidence(
                status="complete",
                method="test_gate",
                outcome="no_relevant_candidate_passage",
            ),
        }
    )

    rescued = attach_local_semantic_retrieval_rescue(
        source,
        artifact,
        scorer=_SemanticRescueScorer(),
        prefilter_count=13,
        max_additions=4,
    )

    retrieval = rescued.candidate_passage_retrieval
    assert retrieval.semantic_rescue_status == "complete"
    assert retrieval.semantic_rescue_version == "bm25-prefilter-local-nli-v1"
    assert retrieval.semantic_model_id == "local-test-nli"
    assert retrieval.semantic_model_revision == "revision-1"
    assert 0 < retrieval.semantic_prefilter_count <= 13
    assert 0 < retrieval.semantic_addition_count <= 4
    assert protected_ids <= {passage.passage_id for passage in rescued.passages}
    selected = retrieval.selections[0].passages
    assert len(selected) <= 4
    assert selected[0].rank == 1
    assert "candidate_local_nli_rescue" in selected[0].channels
    by_id = {passage.passage_id: passage for passage in rescued.passages}
    assert "machine translation strategically" in by_id[selected[0].passage_id].text


def test_local_semantic_rescue_does_not_run_before_a_confirmed_candidate_miss() -> None:
    artifact = build_passage_evidence(
        _ocr_authorized_source("A complete source passage."),
        claim=_claim("A complete source passage."),
    )

    assert (
        attach_local_semantic_retrieval_rescue(
            _ocr_authorized_source("A complete source passage."),
            artifact,
            scorer=_SemanticRescueScorer(),
        )
        is artifact
    )


def test_exact_quotation_builds_scope_bound_inspectable_artifact(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "Opening material about another subject.",
        "The evidence states that careful verification improves accuracy.\n\n"
        "A neighboring sentence provides necessary context.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            'The author writes, "careful verification improves accuracy."',
            claim_type="quotation",
            page_locator="p. 2",
        ),
    )

    assert artifact.source_identity.status.value == "verified"
    assert artifact.coverage.level is CoverageLevel.FULL_TEXT
    assert artifact.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE
    assert artifact.passages[0].retrieval_method == "exact_quotation"
    assert artifact.passages[0].page_index == 1
    assert artifact.passages[0].representation_id == str(record.id)
    assert artifact.passages[0].authorization_scope_type == "personal_owner"
    assert artifact.passages[0].authorization_scope_id == "owner-1"
    assert artifact.passages[0].content_sha256 == hashlib.sha256(content).hexdigest()
    assert "necessary context" in artifact.passages[0].text
    payload = artifact.report_payload()
    assert payload["verdict"] == "inconclusive"
    assert "content" not in payload
    assert "candidate_passages_located_relation_not_assessed" in payload["reason_codes"]


def test_lexical_retrieval_ranks_relevant_passage_and_does_not_claim_support(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "Vampire cinema and adaptation are discussed in this paragraph.",
        "Anime diplomacy connects popular culture, national image, and foreign policy.\n\n"
        "The authors examine cultural influence in international relations.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            "Anime can influence national image and foreign policy through popular culture.",
            page_locator="2",
        ),
        top_k=2,
    )

    assert artifact.passages
    assert artifact.passages[0].page_index == 1
    assert artifact.passages[0].retrieval_method == "lexical_overlap"
    assert "foreign policy" in artifact.passages[0].text
    assert artifact.relationship.status is RelationshipStatus.NOT_ASSESSED
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE


def test_resolved_antecedent_expands_retrieval_without_rewriting_claim(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The report explains that competition law creates market rivalry among firms."
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    claim = _claim("This change improves outcomes.")
    dependency = ClaimAntecedentDependency(
        mention_text="This change",
        mention_local_start=0,
        mention_local_end=len("This change"),
        mention_paper_start=20,
        mention_paper_end=20 + len("This change"),
        resolution_status="resolved",
        confidence="high",
        antecedent_context_index=0,
        antecedent_text="competition law creates market rivalry among firms",
        antecedent_paper_start=0,
        antecedent_paper_end=51,
        method="test",
    )
    claim = claim.model_copy(
        update={
            "granularity": "atomic_claim",
            "antecedent_dependencies": [dependency],
            "context_dependency_status": "resolved",
        }
    )

    artifact = build_passage_evidence(source, claim=claim)

    assert artifact.claim.text == "This change improves outcomes."
    assert artifact.passages
    assert "competition law" in artifact.passages[0].text


def test_no_located_passage_abstains_with_explicit_reason(session: Session) -> None:
    storage = MemoryStorage()
    record = _admit(session, storage, _pdf("This source concerns botanical taxonomy."))
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("Quantum processors improve cryptographic throughput."),
    )

    assert artifact.passages == []
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE
    assert artifact.reason_codes == [
        "no_passage_located_in_available_evidence",
        "claim_not_atomized",
    ]


def test_exact_scope_is_required_even_when_same_bytes_exist_elsewhere(
    session: Session,
) -> None:
    storage = MemoryStorage()
    record = _admit(
        session,
        storage,
        _pdf("Authorized course evidence."),
        scope_type="assessment",
        scope_id="assessment-1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )

    with pytest.raises(EvidenceAuthorizationError, match="requesting scope"):
        authorize_representation(
            session,
            storage,
            representation_id=record.id,
            scope_type="assessment",
            scope_id="assessment-2",
        )


def test_expired_or_nonaccepted_representation_fails_closed(session: Session) -> None:
    storage = MemoryStorage()
    expired = _admit(
        session,
        storage,
        _pdf("Expired evidence."),
        scope_type="assessment",
        scope_id="assessment-1",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(EvidenceAuthorizationError, match="expired"):
        authorize_representation(
            session,
            storage,
            representation_id=expired.id,
            scope_type="assessment",
            scope_id="assessment-1",
        )

    expired.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    expired.admission_state = "needs_review"
    session.commit()
    with pytest.raises(EvidenceAuthorizationError, match="not accepted"):
        authorize_representation(
            session,
            storage,
            representation_id=expired.id,
            scope_type="assessment",
            scope_id="assessment-1",
        )


def test_content_hash_mismatch_fails_closed(session: Session) -> None:
    storage = MemoryStorage()
    record = _admit(session, storage, _pdf("Original verified bytes."))
    storage.objects[record.content_object.storage_key] = _pdf("Tampered bytes.")

    with pytest.raises(EvidenceAuthorizationError, match="immutable-object"):
        authorize_representation(
            session,
            storage,
            representation_id=record.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )


def test_plain_text_representation_uses_same_evidence_contract(session: Session) -> None:
    storage = MemoryStorage()
    content = (
        b"First page unrelated.\f"
        b"The study reports a relationship between cultural exports and national image."
    )
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("Cultural exports shape national image.", page_locator="2"),
    )

    assert artifact.coverage.representation_kind == "plain_text"
    assert artifact.passages[0].page_index == 1
    assert artifact.verdict is VerificationVerdict.INCONCLUSIVE


def test_exact_quotation_normalizes_line_end_hyphenation(session: Session) -> None:
    storage = MemoryStorage()
    content = b"Careful verifi-\ncation improves the reliability of academic evidence."
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"Careful verification improves the reliability of academic evidence."',
            claim_type="quotation",
        ),
    )

    assert artifact.passages[0].retrieval_method == "exact_quotation"
    assert artifact.quotation_check.outcome == "all_spans_normalized_match"


def test_full_span_quotation_accepts_bounded_bracket_change(session: Session) -> None:
    storage = MemoryStorage()
    content = b"The evidence demonstrates careful and complete source verification."
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"[T]he evidence demonstrates careful and complete source verification."',
            claim_type="quotation",
        ),
    )

    assert artifact.passages[0].retrieval_method == "exact_quotation"
    assert artifact.quotation_check.outcome == "all_spans_normalized_match"


def test_full_span_quotation_accepts_bounded_ellipsis(session: Session) -> None:
    storage = MemoryStorage()
    content = (
        b"The evidence demonstrates careful source verification across every "
        b"retrieved document and therefore improves the reliability of academic review."
    )
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"The evidence demonstrates careful source verification ... improves the reliability of academic review."',
            claim_type="quotation",
        ),
    )

    assert artifact.passages[0].retrieval_method == "exact_quotation"
    assert artifact.quotation_check.outcome == "all_spans_normalized_match"


def test_ellipsis_cannot_turn_short_prefix_into_full_span_match(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = b"The evidence demonstrates something materially different."
    record = _admit(
        session,
        storage,
        content,
        kind=RepresentationKind.PLAIN_TEXT,
        media_type="text/plain",
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim('"The evidence demonstrates ..."', claim_type="quotation"),
    )

    assert artifact.quotation_check.status == "not_assessable"
    assert artifact.quotation_check.outcome == "marked_editorial_changes_require_review"
    assert not artifact.quotation_check.evidence_passage_ids


def test_quotation_check_requires_every_quoted_span(session: Session) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The source says alpha evidence is present and complete.",
        "A different passage supplies ordinary context.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            'The author writes "alpha evidence is present" but also "missing beta evidence".',
            claim_type="quotation",
            page_locator="1",
        ),
    )

    assert artifact.quotation_check.status == "complete"
    assert artifact.quotation_check.outcome == "some_spans_not_located"
    assert artifact.quotation_check.evidence_passage_ids
    assert artifact.locator_check.status == "not_assessable"
    assert artifact.locator_check.outcome == "quotation_not_fully_located"


def test_quotation_locator_reports_match_on_different_page(session: Session) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "Opening material on the supplied first page.",
        "The exact quoted evidence appears only on the second page.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"The exact quoted evidence appears only on the second page."',
            claim_type="quotation",
            page_locator="1",
        ),
    )

    assert artifact.quotation_check.outcome == "all_spans_literal_match"
    assert artifact.locator_check.status == "complete"
    assert artifact.locator_check.outcome == "located_span_outside_supplied_locator"


def test_author_accepted_manuscript_does_not_claim_publisher_page_match(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf("The exact quoted evidence appears on this manuscript page.")
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    source = replace(
        source,
        edition_or_version="author_accepted_manuscript",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"The exact quoted evidence appears on this manuscript page."',
            claim_type="quotation",
            page_locator="1",
        ),
    )

    assert artifact.quotation_check.outcome == "all_spans_literal_match"
    assert artifact.locator_check.status == "not_assessable"
    assert (
        artifact.locator_check.outcome
        == "source_version_pagination_may_differ"
    )


def test_full_span_quotation_and_locator_cross_page_boundary(session: Session) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The participant explained that careful source checking begins on this page",
        "and continues on the next page before reaching a clear conclusion.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"careful source checking begins on this page and continues on the next page before reaching a clear conclusion"',
            claim_type="quotation",
            page_locator="1–2",
        ),
    )

    assert artifact.quotation_check.status == "complete"
    assert artifact.quotation_check.outcome == "all_spans_normalized_match"
    assert artifact.quotation_check.evidence_passage_ids
    assert artifact.locator_check.status == "complete"
    assert artifact.locator_check.outcome == "located_span_matches_supplied_locator"


def test_cross_page_quotation_requires_locator_to_cover_every_page(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The participant explained that careful source checking begins on this page",
        "and continues on the next page before reaching a clear conclusion.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"careful source checking begins on this page and continues on the next page before reaching a clear conclusion"',
            claim_type="quotation",
            page_locator="1",
        ),
    )

    assert artifact.quotation_check.outcome == "all_spans_normalized_match"
    assert artifact.locator_check.status == "complete"
    assert artifact.locator_check.outcome == "located_span_outside_supplied_locator"


def test_cross_page_quotation_cannot_bridge_into_reference_list(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf(
        "The quoted claim begins in ordinary body prose and",
        "References\nSmith, J. (2020). continues only in bibliography material.",
    )
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim(
            '"The quoted claim begins in ordinary body prose and Smith, J. (2020). continues only in bibliography material."',
            claim_type="quotation",
            page_locator="1–2",
        ),
    )

    assert artifact.quotation_check.status == "complete"
    assert artifact.quotation_check.outcome == "no_span_located"
    assert artifact.locator_check.status == "not_assessable"
    assert artifact.locator_check.outcome == "quotation_not_fully_located"


def test_paraphrase_locator_does_not_inherit_retrieval_as_validation(
    session: Session,
) -> None:
    storage = MemoryStorage()
    content = _pdf("A paraphrased idea appears on this page.")
    record = _admit(session, storage, content)
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("A paraphrased idea appears.", page_locator="1"),
    )

    assert artifact.locator_check.status == "not_assessable"
    assert artifact.locator_check.outcome == "paraphrase_locator_requires_relevant_evidence"


def test_claim_conversion_preserves_application_identity_and_coordinates() -> None:
    citation = InTextCitation(
        reference_ids=["paper-v1-ref-0001"],
        link_status="linked",
        text="A verified cited claim.",
        claim_type="paraphrase",
        citation_marker="(Scholar, 2025, p. 14)",
        page_number="14",
        passage_start=100,
        passage_end=123,
    )

    first = claim_evidence_from_citation(citation, paper_version_id="paper-v1")
    second = claim_evidence_from_citation(citation, paper_version_id="paper-v1")

    assert first.claim_id == second.claim_id
    assert first.reference_ids == ["paper-v1-ref-0001"]
    assert first.passage_start == 100
    assert first.passage_end == 123
    assert first.page_locator == "14"
    assert first.granularity == "citation_unit"
    assert first.atomization_method == "not_run"


@pytest.mark.parametrize(
    "citation, message",
    [
        (
            InTextCitation(
                reference_ids=["ref-1"],
                text="Rejected model text.",
                passage_start=0,
                passage_end=20,
                drop_reason="text_not_in_original",
            ),
            "Rejected citation",
        ),
        (
            InTextCitation(
                reference_ids=[],
                candidate_reference_ids=["ref-1", "ref-2"],
                link_status="ambiguous",
                text="Ambiguous claim.",
                passage_start=0,
                passage_end=16,
            ),
            "uniquely linked",
        ),
        (
            InTextCitation(
                reference_ids=["ref-1"],
                text="Unlocated claim.",
                passage_start=-1,
                passage_end=-1,
            ),
            "coordinates",
        ),
    ],
)
def test_claim_conversion_rejects_untrusted_or_ambiguous_spans(
    citation: InTextCitation,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        claim_evidence_from_citation(citation, paper_version_id="paper-v1")


def test_image_only_pdf_is_not_reported_as_full_text_coverage(session: Session) -> None:
    storage = MemoryStorage()
    document = fitz.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    record = _admit(session, storage, content)
    record.text_quality = "pure_scan"
    session.commit()
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    artifact = build_passage_evidence(
        source,
        claim=_claim("A claim that requires source text."),
    )

    assert artifact.coverage.level is CoverageLevel.UNAVAILABLE
    assert artifact.verdict is VerificationVerdict.NOT_ASSESSED
    assert "source_text_unavailable" in artifact.reason_codes


def test_collective_claim_requires_and_preserves_source_specific_binding(
    session: Session,
) -> None:
    storage = MemoryStorage()
    record = _admit(
        session,
        storage,
        _pdf("Evidence matters when sources are checked independently."),
    )
    source = authorize_representation(
        session,
        storage,
        representation_id=record.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    text = "Smith (2020) and Jones (2021) argue that evidence matters."
    smith_text = "Smith (2020)"
    jones_text = "Jones (2021)"
    jones_start = text.index(jones_text)
    claim = ClaimEvidence(
        claim_id="collective-claim",
        paper_version_id="paper-v1",
        text=text,
        reference_ids=["ref-smith", "ref-jones"],
        citation_marker="Smith (2020) and Jones (2021)",
        citation_markers=[
            CitationMarkerMember(
                text=smith_text,
                local_start=0,
                local_end=len(smith_text),
                reference_ids=["ref-smith"],
                marker_type="narrative",
            ),
            CitationMarkerMember(
                text=jones_text,
                local_start=jones_start,
                local_end=jones_start + len(jones_text),
                reference_ids=["ref-jones"],
                marker_type="narrative",
            ),
        ],
        citation_marker_type="narrative",
        passage_start=100,
        passage_end=100 + len(text),
    )

    with pytest.raises(ValueError, match="active reference binding"):
        build_passage_evidence(source, claim=claim)

    smith = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="ref-smith",
        cited_author_label="Smith",
    )
    jones = build_passage_evidence(
        source,
        claim=claim,
        active_reference_id="ref-jones",
        cited_author_label="Jones",
    )

    assert smith.source_binding.reference_id == "ref-smith"
    assert smith.source_binding.marker_text == smith_text
    assert smith.source_binding.cited_author_label == "Smith"
    assert jones.source_binding.reference_id == "ref-jones"
    assert jones.source_binding.marker_text == jones_text
    assert jones.source_binding.cited_author_label == "Jones"
    assert smith.verification_id != jones.verification_id
