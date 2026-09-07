from app.services import text_extractor


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
