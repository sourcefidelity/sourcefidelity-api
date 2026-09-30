import hashlib
from unittest.mock import patch

import fitz
import pytest

from app.services.file_safety import FileSafetyReport, SafetyVerdict
from app.services.reprint_pdf_screen import screen_reprint_pdf


CLEAN = FileSafetyReport(SafetyVerdict.CLEAN, SafetyVerdict.CLEAN, SafetyVerdict.CLEAN)


def pdf(text):
    with fitz.open() as doc:
        doc.new_page().insert_text((50, 70), text)
        return doc.tobytes()


def run(raw):
    return screen_reprint_pdf(raw, submitted_reference_sha256='b'*64,
                             cited_year='2000', retrieved_year='2002')


def test_inspected_bytes_bind_the_finding():
    raw = pdf('Copyright 2000\nFirst published 2000\nReprinted 2002\nISBN 123')
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN) as scan:
        result = run(raw)
    scan.assert_called_once_with(raw)
    assert result.status == 'screened'
    assert result.finding.status == 'documented_reprint'
    assert result.representation_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.finding.representation_sha256 == result.representation_sha256


@pytest.mark.parametrize('verdict', [SafetyVerdict.REJECTED, SafetyVerdict.NOT_ASSESSED, SafetyVerdict.UNAVAILABLE])
def test_nonclean_bytes_never_reach_academic_parser(verdict):
    report = FileSafetyReport(verdict, verdict, verdict)
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=report), \
         patch('app.services.reprint_pdf_screen.fitz.open') as parser:
        assert run(b'not a pdf').reason == 'safety_not_clean'
        parser.assert_not_called()


def test_scanner_failure_is_neutral():
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', side_effect=RuntimeError('private detail')):
        result = run(b'bytes')
    assert result.reason == 'safety_inspection_unavailable'
    assert 'private detail' not in result.model_dump_json()


def test_image_only_publication_page_abstains():
    with fitz.open() as doc:
        page = doc.new_page()
        page.draw_rect(fitz.Rect(30, 30, 200, 300))
        raw = doc.tobytes()
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN):
        result = run(raw)
        assert result.reason == 'publication_text_unavailable'
        assert result.unassessed_page_indexes == (0,)


def test_blank_pdf_abstains():
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN):
        assert run(pdf('')).reason == 'publication_text_unavailable'


def test_white_graphical_page_is_bound_and_does_not_block_history():
    with fitz.open() as doc:
        page = doc.new_page()
        page.draw_rect(page.rect, color=(1, 1, 1), fill=(1, 1, 1))
        doc.new_page().insert_text((50, 70),
            'Copyright 2000\nFirst published 2000\nReprinted 2002\nISBN 123')
        raw = doc.tobytes()
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN):
        result = run(raw)
    assert result.status == 'screened'
    assert result.finding.status == 'documented_reprint'
    assert result.blank_page_renders[0].page_index == 0
    assert len(result.blank_page_renders[0].render_sha256) == 64


def test_faint_graphical_content_is_not_treated_as_white():
    with fitz.open() as doc:
        page = doc.new_page()
        page.draw_rect(fitz.Rect(30, 30, 200, 200), color=(0.99, 0.99, 0.99))
        raw = doc.tobytes()
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN):
        assert run(raw).unassessed_page_indexes == (0,)


def test_oversize_graphical_page_is_not_rendered():
    from app.services.reprint_pdf_screen import _blank_page_render
    from unittest.mock import Mock
    page = Mock(rect=fitz.Rect(0, 0, 10000, 10000))
    assert _blank_page_render(page, 0) is None
    page.get_pixmap.assert_not_called()


def test_page_text_budget_has_specific_reason_without_truncation():
    from unittest.mock import MagicMock
    from app.services.edition_statement_verifier import MAX_PAGE_CHARACTERS
    doc = MagicMock()
    doc.__enter__.return_value = doc
    doc.__len__.return_value = 1
    doc.__getitem__.return_value.get_text.return_value = 'x' * (MAX_PAGE_CHARACTERS + 1)
    with patch('app.services.reprint_pdf_screen.inspect_uploaded_pdf', return_value=CLEAN), \
         patch('app.services.reprint_pdf_screen.fitz.open', return_value=doc), \
         patch('app.services.reprint_pdf_screen.verify_reprint_history') as verifier:
        result = run(b'bounded fixture')
    assert result.status == 'not_assessed'
    assert result.reason == 'publication_page_text_budget_exceeded'
    assert result.unassessed_page_indexes == (0,)
    assert result.finding is None
    verifier.assert_not_called()
