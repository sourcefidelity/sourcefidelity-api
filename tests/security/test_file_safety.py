"""Hostile-file inspection rejects active, malformed, and unscanned PDFs."""

import fitz
import pytest

from app.config import settings
from app.services.file_safety import (
    FileSafetyUnavailable,
    SafetyVerdict,
    _parse_clamd_reply,
    inspect_uploaded_pdf,
)


def _pdf_bytes(*, pages: int = 1, encrypted: bool = False) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), "A benign academic source")
    options = {}
    if encrypted:
        options = {
            "encryption": fitz.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner-password",
            "user_pw": "user-password",
        }
    content = document.tobytes(**options)
    document.close()
    return content


def _pdf_with_open_action(action: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "An academic source with an initial view")
    catalog = document.pdf_catalog()
    if action == "destination":
        document.xref_set_key(catalog, "OpenAction", f"[{page.xref} 0 R /Fit]")
    elif action == "indirect_destination":
        destination_xref = document.get_new_xref()
        document.update_object(destination_xref, f"[{page.xref} 0 R /Fit]")
        document.xref_set_key(catalog, "OpenAction", f"{destination_xref} 0 R")
    else:
        action_xref = document.get_new_xref()
        if action == "GoTo":
            action_body = f"/S /GoTo /D [{page.xref} 0 R /Fit]"
        elif action == "malformed_goto":
            action_body = "/S /GoTo"
        elif action == "chained":
            action_body = (
                f"/S /GoTo /D [{page.xref} 0 R /Fit] "
                "/Next << /S /GoToR >>"
            )
        else:
            action_body = f"/S /{action}"
        document.update_object(action_xref, f"<< {action_body} >>")
        document.xref_set_key(catalog, "OpenAction", f"{action_xref} 0 R")
    content = document.tobytes()
    document.close()
    return content


@pytest.mark.security
def test_clean_pdf_requires_both_malware_and_structural_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    report = inspect_uploaded_pdf(_pdf_bytes())

    assert report.verdict is SafetyVerdict.CLEAN
    assert report.malware_verdict is SafetyVerdict.CLEAN
    assert report.structural_verdict is SafetyVerdict.CLEAN


@pytest.mark.security
def test_active_pdf_content_is_rejected_before_parser(monkeypatch) -> None:
    content = _pdf_bytes().replace(b"%%EOF", b"/JavaScript /OpenAction\n%%EOF")
    scanner_called = False

    def scanner(_content):
        nonlocal scanner_called
        scanner_called = True
        return SafetyVerdict.CLEAN, "stream: OK"

    monkeypatch.setattr("app.services.file_safety.scan_with_clamd", scanner)
    report = inspect_uploaded_pdf(content)

    assert report.verdict is SafetyVerdict.REJECTED
    assert "PDF JavaScript" in report.findings
    assert scanner_called is False


@pytest.mark.security
def test_benign_open_action_destination_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("destination"))

    assert report.verdict is SafetyVerdict.CLEAN
    assert report.findings == ()


@pytest.mark.security
def test_internal_goto_open_action_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("GoTo"))

    assert report.verdict is SafetyVerdict.CLEAN
    assert report.findings == ()


@pytest.mark.security
def test_indirect_open_action_destination_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("indirect_destination"))

    assert report.verdict is SafetyVerdict.CLEAN
    assert report.findings == ()


@pytest.mark.security
def test_external_open_action_is_rejected_structurally(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("GoToR"))

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == ("automatic open action (GoToR)",)


@pytest.mark.security
def test_internal_open_action_with_external_chain_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("chained"))

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == ("automatic open action (chained action)",)


@pytest.mark.security
def test_internal_open_action_without_destination_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )

    report = inspect_uploaded_pdf(_pdf_with_open_action("malformed_goto"))

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == (
        "automatic open action (malformed internal destination)",
    )


@pytest.mark.security
def test_encrypted_pdf_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    report = inspect_uploaded_pdf(_pdf_bytes(encrypted=True))

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == ("encrypted or password-protected PDF",)


@pytest.mark.security
def test_malformed_pdf_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    report = inspect_uploaded_pdf(b"%PDF-1.7\nnot actually a PDF")

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings[0].startswith("malformed PDF")


@pytest.mark.security
def test_html_download_with_pdf_filename_is_rejected_before_scanning(monkeypatch) -> None:
    scanner_called = False

    def scanner(_content):
        nonlocal scanner_called
        scanner_called = True
        return SafetyVerdict.CLEAN, "stream: OK"

    monkeypatch.setattr("app.services.file_safety.scan_with_clamd", scanner)
    report = inspect_uploaded_pdf(
        b"<!doctype html><title>Download failed</title><p>Access page</p>"
    )

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == ("content is not a PDF",)
    assert scanner_called is False


@pytest.mark.security
def test_page_limit_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 1)
    monkeypatch.setattr(
        "app.services.file_safety.scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    report = inspect_uploaded_pdf(_pdf_bytes(pages=2))

    assert report.verdict is SafetyVerdict.REJECTED
    assert report.findings == ("PDF exceeds configured page limit",)


@pytest.mark.security
def test_required_scanner_unavailability_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MALWARE_SCAN_REQUIRED", True)

    def unavailable(_content):
        raise FileSafetyUnavailable("not configured")

    monkeypatch.setattr("app.services.file_safety.scan_with_clamd", unavailable)
    with pytest.raises(FileSafetyUnavailable):
        inspect_uploaded_pdf(_pdf_bytes())


@pytest.mark.security
def test_optional_scanner_unavailability_is_not_called_clean(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MALWARE_SCAN_REQUIRED", False)

    def unavailable(_content):
        raise FileSafetyUnavailable("not configured")

    monkeypatch.setattr("app.services.file_safety.scan_with_clamd", unavailable)
    report = inspect_uploaded_pdf(_pdf_bytes())

    assert report.verdict is SafetyVerdict.NOT_ASSESSED
    assert report.structural_verdict is SafetyVerdict.CLEAN


@pytest.mark.security
@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        (b"stream: OK\0", SafetyVerdict.CLEAN),
        (b"stream: Eicar-Signature FOUND\0", SafetyVerdict.REJECTED),
        (b"stream: size limit exceeded ERROR\0", SafetyVerdict.UNAVAILABLE),
        (b"", SafetyVerdict.UNAVAILABLE),
    ],
)
def test_clamd_replies_are_fail_closed(reply: bytes, verdict: SafetyVerdict) -> None:
    assert _parse_clamd_reply(reply)[0] is verdict
