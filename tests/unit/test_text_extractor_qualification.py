from app.services import text_extractor


def test_native_spacing_recovery_with_shared_long_url(monkeypatch):
    good = ('The cultural context provides several useful observations.\n' * 20
            + 'https://example.org/' + 'a' * 100)
    bad = good.replace('cultural context provides', 'culturalcontextprovides').replace(
        'several useful observations', 'severalusefulobservations')
    monkeypatch.setattr(text_extractor, '_extract_pdfplumber_bytes', lambda _: bad)
    monkeypatch.setattr(text_extractor, '_extract_pymupdf_bytes', lambda _: good)
    result = text_extractor.extract_qualified_text_from_bytes(b'pdf', 'paper.pdf')
    assert result.selected.backend == 'pymupdf'
    assert result.selection_reason == 'alternate_recovers_native_word_boundaries'


def test_more_words_without_native_line_agreement_does_not_repair_spacing():
    make = text_extractor._text_candidate
    a = make('a', 'some text ' * 40, layout_text='some text\n' * 40)
    b = make('b', 'other many different words ' * 40, layout_text='other many different words\n' * 40)
    assert not text_extractor._native_spacing_recovery(a, b)


def test_pdf_qualification_rejects_nonempty_missing_space_output(monkeypatch):
    bad = " ".join(["ordinary"] * 50) + " " + ("joined" * 30)
    good = "This is cleanly spaced semantic paper text. " * 30
    monkeypatch.setattr(text_extractor, "_extract_pdfplumber_bytes", lambda _: bad)
    monkeypatch.setattr(text_extractor, "_extract_pymupdf_bytes", lambda _: good)

    result = text_extractor.extract_qualified_text_from_bytes(b"pdf", "paper.pdf")

    assert result.selected.backend == "pymupdf"
    assert result.selected.text == good
    assert result.selection_reason == "alternate_materially_cleaner_than_preferred"
    assert {item.backend for item in result.candidates} == {"pdfplumber", "pymupdf"}


def test_pdf_qualification_keeps_clean_preferred_backend(monkeypatch):
    preferred = "Clean preferred text with ordinary spacing. " * 20
    alternate = "Clean alternate text with ordinary spacing. " * 30
    monkeypatch.setattr(
        text_extractor, "_extract_pdfplumber_bytes", lambda _: preferred
    )
    monkeypatch.setattr(text_extractor, "_extract_pymupdf_bytes", lambda _: alternate)

    result = text_extractor.extract_qualified_text_from_bytes(b"pdf", "paper.pdf")

    assert result.selected.backend == "pdfplumber"
    assert result.selection_reason == "preferred_backend_quality_acceptable"


def test_reference_selection_keeps_clean_native_title_when_reference_counts_tie(monkeypatch):
    from app.services.paper_workflow import _select_reference_candidate
    from app.services.paper_extraction import extract_paper_evidence
    bad_body=' '.join(['ordinary']*50)+' '+('joined'*30)
    good_body='This is readable semantic paper text. '*30
    bad=bad_body+'\n\nReferences\n\nWriter,A.(2020).CultureandSocietyandPoliticsandMedia.Sage.'
    good=good_body+'\n\nReferences\n\nWriter, A. (2020). Culture and Society and Politics and Media. Sage.'
    monkeypatch.setattr(text_extractor,'_extract_pdfplumber_bytes',lambda _:bad)
    monkeypatch.setattr(text_extractor,'_extract_pymupdf_bytes',lambda _:good)
    qualified=text_extractor.extract_qualified_text_from_bytes(b'pdf','paper.pdf')
    candidate,counts=_select_reference_candidate(qualified,citation_format='apa')
    assert candidate.backend=='pymupdf'
    assert counts=={'pdfplumber':1,'pymupdf':1}
    artifact=extract_paper_evidence(qualified.selected.text,paper_version_id='test-title-quality',
        reference_text=candidate.text,reference_layout_text=candidate.layout_text,format_hint='apa',
        use_llm_boundaries=False,use_llm_atomizer=False,use_llm_reference_fallback=False)
    assert artifact.references[0].title=='Culture and Society and Politics and Media'
