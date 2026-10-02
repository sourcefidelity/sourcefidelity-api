"""Regression tests for deterministic citation-unit quotation boundaries."""

from app.services.citation_extractor import (
    _extract_attributed_text_and_index,
    extract_citations,
    split_sentences,
)
from app.services.schemas import ParsedReference


def test_compact_authors_and_nonstandard_et_period_preserve_exact_marker():
    ref = ParsedReference(reference_id='three', author='Example,Q.,Other, J., &Third, Y', year='2025')
    for marker in ['(Example et al., 2025)', '(Example et. al., 2025)']:
        text = 'This is the student claim ' + marker + '.'
        citations = extract_citations(text, [ref], format_hint='apa', use_llm_boundaries=False)
        assert len(citations) == 1
        assert citations[0].reference_ids == ['three']
        assert citations[0].citation_marker == marker


def test_et_al_style_error_does_not_invent_a_missing_reference():
    refs = [ParsedReference(reference_id='two', author='Garcia Example, A., & Other, B.', year='2022')]
    rows = extract_citations('A claim (Example et. al., 2022).', refs,
                             format_hint='apa', use_llm_boundaries=False)
    assert rows[0].reference_ids == ['two']


def _extract(text: str, marker: str) -> tuple[str, int, int, int]:
    start = text.index(marker)
    return _extract_attributed_text_and_index(
        text,
        start,
        start + len(marker),
        split_sentences(text),
        "parenthetical",
    )


def test_narrative_marker_keeps_literal_quote_across_sentence_boundary():
    text='Smith (2020) states: "The first sentence. The second sentence." Next claim.'
    found,_,start,end=_extract(text,'Smith (2020)')
    assert found==text[:text.index(' Next')]
    assert text[start:end]==found


def test_missing_space_after_parenthetical_keeps_exact_offsets():
    text='A statement (Smith, 2020).Another statement (Jones, 2021).'
    assert split_sentences(text)==['A statement (Smith, 2020).','Another statement (Jones, 2021).']
    found,_,start,end=_extract(text,'(Jones, 2021)')
    assert found=='Another statement (Jones, 2021).' and text[start:end]==found


def test_multiword_surname_and_secondary_author_link_without_rewriting_marker():
    refs=[ParsedReference(author='von Sternberg, J.',year='1932',title='Example film',raw_ref='von Sternberg, J. (1932). Example film.',reference_id='film'),
          ParsedReference(author='Hodges, G.',year='2012',title='Example book',raw_ref='Hodges, G. (2012). Example book.',reference_id='book')]
    citations=extract_citations('A claim (von Sternberg, 1932). Another claim (as cited in Hodges, 2012).',refs,format_hint='apa',use_llm_boundaries=False)
    assert [(c.reference_ids,c.citation_marker) for c in citations]==[(['film'],'(von Sternberg, 1932)'),(['book'],'(as cited in Hodges, 2012)')]


def test_marker_sentence_extends_to_complete_multisentence_curly_quote() -> None:
    text = (
        "The source states, “The first quoted sentence ends here. "
        "The second quoted sentence ends here” (Walsh 84). "
        "The student's next sentence is separate."
    )

    attributed, sentence_index, start, end = _extract(text, "(Walsh 84)")

    assert attributed == text[: text.index(" The student's")]
    assert sentence_index == 1
    assert (start, end) == (0, len(attributed))


def test_marker_sentence_extends_to_complete_multisentence_straight_quote() -> None:
    text = (
        'The source states, "The first quoted sentence ends here. '
        'The second quoted sentence ends here" (Walsh 84). '
        "The student's next sentence is separate."
    )

    attributed, _sentence_index, start, end = _extract(text, "(Walsh 84)")

    assert attributed == text[: text.index(" The student's")]
    assert (start, end) == (0, len(attributed))


def test_ordinary_marker_sentence_does_not_extend_backward() -> None:
    text = "An earlier sentence is unrelated. The cited proposition is here (Walsh 84)."

    attributed, sentence_index, start, end = _extract(text, "(Walsh 84)")

    assert attributed == "The cited proposition is here (Walsh 84)."
    assert sentence_index == 1
    assert text[start:end] == attributed


def test_apa_linking_normalizes_unicode_and_uses_lead_author_structure() -> None:
    references = [
        ParsedReference(
            reference_id="oneill",
            author="O’Neill, E.M",
            year="2016",
            citation_key="ONeill2016",
        ),
        ParsedReference(
            reference_id="kir",
            author="Kir, E., & Akyüz, A",
            year="2020",
            citation_key="Kir2020",
        ),
        ParsedReference(
            reference_id="yang-many",
            author="Yang, M., O’Sullivan, P.S., Irby, D.M., Chen, Z., Lin, C., & Lin, C",
            year="2019",
            citation_key="Yang2019a",
        ),
        ParsedReference(
            reference_id="yang-two",
            author="Yang, Y., & Wang, X",
            year="2019",
            citation_key="Yang2019b",
        ),
        ParsedReference(
            reference_id="chen-peng",
            author="Chen, Y., & Peng, J",
            year="2019",
            citation_key="Chen2019",
        ),
    ]
    body = (
        "Each source is cited (O’Neill, 2016; Kir & Akyüz, 2020; "
        "Yang et al., 2019; Chen & Peng, 2019)."
    )

    citations = extract_citations(
        body,
        references,
        format_hint="apa",
        use_llm_boundaries=False,
    )

    assert [citation.reference_ids for citation in citations] == [
        ["oneill"],
        ["kir"],
        ["yang-many"],
        ["chen-peng"],
    ]
    assert all(citation.link_status == "linked" for citation in citations)


def test_apa_narrative_year_marker_accepts_unicode_surname() -> None:
    references = [
        ParsedReference(
            reference_id="akyuz",
            author="Akyüz, A",
            year="2020",
            citation_key="Akyuz2020",
        )
    ]

    citations = extract_citations(
        "Akyüz (2020) reports the finding.",
        references,
        format_hint="apa",
        use_llm_boundaries=False,
    )

    assert len(citations) == 1
    assert citations[0].reference_ids == ["akyuz"]
    assert citations[0].citation_marker == "Akyüz (2020)"


def test_apa_year_only_marker_recovers_coordinated_narrative_authors() -> None:
    references = [
        ParsedReference(
            reference_id="garcia-pena",
            author="Garcia, P., & Pena, M.I",
            year="2011",
            citation_key="Garcia2011",
        )
    ]

    citations = extract_citations(
        "As Garcia and Pena propose, the change must be autonomous (2011, p. 486).",
        references,
        format_hint="apa",
        use_llm_boundaries=False,
    )

    assert len(citations) == 1
    assert citations[0].reference_ids == ["garcia-pena"]
    assert citations[0].citation_marker.startswith("Garcia and Pena")
    assert citations[0].page_number == "486"


def test_apa_year_only_marker_recovers_pdf_split_accented_surname() -> None:
    references = [
        ParsedReference(
            reference_id="kir-akyuz",
            author="Kir, E., & Akyüz, A",
            year="2020",
            citation_key="Kir2020",
        )
    ]

    citations = extract_citations(
        "Kir and Akyu\u0308 z (2020) report the finding.",
        references,
        format_hint="apa",
        use_llm_boundaries=False,
    )

    assert len(citations) == 1
    assert citations[0].reference_ids == ["kir-akyuz"]
    assert citations[0].citation_marker.startswith("Kir and Akyu")


def test_film_title_year_links_exact_work_not_previous_corporate_author():
    refs = [
        ParsedReference(reference_id="film", author="Morgan, C. (Director)", year="1946", title="Marina [Film]", raw_ref="Morgan, C. (Director). (1946). Marina [Film]. Studio.", citation_key="Morgan1946"),
        ParsedReference(reference_id="magazine", author="Screen Herald", year="1946", title="Screen Herald", citation_key="Screen1946"),
    ]
    text = "Screen Herald discussed the actor (Screen, 1946). Marina (1946) is the film discussed here."
    citations = extract_citations(text, refs, format_hint="apa", use_llm_boundaries=False)
    films = [c for c in citations if c.citation_marker == "Marina (1946)"]
    assert len(films) == 1 and films[0].reference_ids == ["film"]
    assert films[0].text == "Marina (1946) is the film discussed here."
    assert not any("Screen Herald discussed" in c.citation_marker for c in citations)


def test_parenthetical_exact_film_title_links_without_title_variants():
    ref = ParsedReference(reference_id='film', author='Morgan, C. (Director)', year='1946',
        title='Marina [Film]', raw_ref='Morgan, C. (Director). (1946). Marina [Film]. Studio.')
    for marker, expected in [('(Marina, 1946)', ['film']), ('(Marinas, 1946)', []), ('(Marina, 1947)', [])]:
        found = extract_citations('The scene ends '+marker+'.', [ref], format_hint='apa', use_llm_boundaries=False)
        assert found[0].reference_ids == expected
    other = ref.model_copy(update={'reference_id':'other'})
    found = extract_citations('The scene ends (Marina, 1946).', [ref, other], format_hint='apa', use_llm_boundaries=False)
    assert found[0].link_status == 'ambiguous'


def test_second_explicit_surname_without_initials_still_links_exactly():
    ref = ParsedReference(reference_id='book',author='Smith, J. & Jones',year='1984',
        title='A complete title',raw_ref='Smith, J. & Jones (1984). A complete title. Example Press.')
    # Owner decision 2026-10-01 (citation-reference-tolerance-v1): a differing second
    # author still links the one matching reference, and the difference is kept.
    for marker, expected, differences in [('(Smith & Jones, 1984)', ['book'], []),
                                          ('(Smith & James, 1984)', ['book'], ['coauthor:Jones'])]:
        found=extract_citations('The claim is stated '+marker+'.',[ref],format_hint='apa',use_llm_boundaries=False)
        assert found[0].reference_ids==expected and found[0].link_differences==differences


def test_explicit_unmatched_narrative_attribution_is_retained_without_identity():
    refs=[ParsedReference(reference_id='known',author='Jones, A.',year='2020',title='Known work')]
    text='Smith (1939) noted that the commercial decision shaped later roles.'
    found=extract_citations(text,refs,format_hint='apa',use_llm_boundaries=False)
    citation,=found
    assert citation.citation_marker=='Smith (1939)'
    assert citation.link_status=='missing_reference' and not citation.reference_ids
    assert citation.confidence=='low'
    assert text[citation.passage_start:citation.passage_end]==citation.text


def test_unmatched_narrative_recovery_preserves_noise_and_spelling_abstention():
    refs=[ParsedReference(reference_id='known',author='Bolton, A.',year='2017',title='Known work')]
    for text in ['The work was autonomous (2011) and changed later.',
                 'Boltn (2017) pointed out that the image changed.',
                 'Smith (1939) was discussed by the audience.']:
        found=extract_citations(text,refs,format_hint='apa',use_llm_boundaries=False)
        assert not any(c.link_status=='missing_reference' for c in found)


def test_explicit_group_author_abbreviation_links_and_preserves_collisions():
    ref=ParsedReference(reference_id='group',author='Example Motion Pictures (EMP)',year='1941',title='A letter')
    found=extract_citations('The studio wrote a letter (EMP, 1941).',[ref],format_hint='apa',use_llm_boundaries=False)
    assert found[0].reference_ids==['group']
    other=ref.model_copy(update={'reference_id':'other','author':'Another Organization (EMP)'})
    found=extract_citations('The studio wrote a letter (EMP, 1941).',[ref,other],format_hint='apa',use_llm_boundaries=False)
    assert found[0].link_status=='ambiguous'
    hidden=ref.model_copy(update={'author':'Example Motion Pictures','title':'A letter (EMP)'})
    found=extract_citations('The studio wrote a letter (EMP, 1941).',[hidden],format_hint='apa',use_llm_boundaries=False)
    assert not found[0].reference_ids


def test_same_creator_film_and_article_use_explicit_printed_locator():
    refs = [
        ParsedReference(reference_id="film", author="Morgan, C. (Director)", year="1946", title="Marina [Film]", raw_ref="Morgan, C. (Director). (1946). Marina [Film]. Studio.", citation_key="Morgan1946"),
        ParsedReference(reference_id="article", author="Morgan, C", year="1946", title="The performance", raw_ref="Morgan, C. (1946, August). The performance. Screen Review, pp. 42, 86-87.", citation_key="Morgan1946"),
    ]
    linked = extract_citations('The performance was described as "restrained" (Morgan, 1946, p. 86).', refs, format_hint="apa", use_llm_boundaries=False)
    assert len(linked) == 1 and linked[0].reference_ids == ["article"]
    ambiguous = extract_citations("The work is discussed (Morgan, 1946).", refs, format_hint="apa", use_llm_boundaries=False)
    assert ambiguous[0].link_status == "ambiguous"


def test_unknown_title_year_cannot_borrow_author_from_previous_sentence():
    refs = [ParsedReference(reference_id="smith", author="Smith", year="1946", citation_key="Smith1946")]
    citations = extract_citations("Smith discussed another subject. Marina (1946) was released later.", refs, format_hint="apa", use_llm_boundaries=False)
    assert not citations
