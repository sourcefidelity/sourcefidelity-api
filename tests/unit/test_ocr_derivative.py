"""Development-only immutable OCR derivative regressions."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import fitz
import pytest

from app.services import ocr_derivative
from app.services.ocr_derivative import (
    OcrDerivativeError,
    build_isolated_pdf_ocr_derivative,
    build_local_pdf_ocr_derivative,
    validate_ocr_derivative,
)


def _one_page_pdf() -> bytes:
    document = fitz.open()
    document.new_page(width=300, height=400)
    content = document.tobytes()
    document.close()
    return content


def test_ocr_derivative_binds_parent_render_text_and_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        ocr_derivative,
        "_tesseract_version",
        lambda _executable: "tesseract 5.test",
    )
    monkeypatch.setattr(
        ocr_derivative,
        "_run_tesseract_page",
        lambda *_args, **_kwargs: (
            "685\nThis sufficiently long page text represents deterministic OCR output "
            "and includes enough additional material to pass the bounded usability gate.\n",
            91.25,
        ),
    )

    derivative = build_local_pdf_ocr_derivative(_one_page_pdf())
    validate_ocr_derivative(derivative)

    assert derivative.page_labels == ("685",)
    assert derivative.page_results[0].mean_word_confidence == 91.25
    assert derivative.manifest["page_count"] == 1
    assert derivative.manifest["parent_content_sha256"] == derivative.parent_content_sha256
    assert "This sufficiently" not in str(derivative.manifest)


def test_ocr_derivative_validation_rejects_altered_content(monkeypatch) -> None:
    monkeypatch.setattr(
        ocr_derivative,
        "_tesseract_version",
        lambda _executable: "tesseract 5.test",
    )
    monkeypatch.setattr(
        ocr_derivative,
        "_run_tesseract_page",
        lambda *_args, **_kwargs: (
            "This sufficiently long page text represents deterministic OCR output "
            "and includes enough additional material to pass the bounded usability gate.\n",
            90.0,
        ),
    )
    derivative = build_local_pdf_ocr_derivative(_one_page_pdf())

    with pytest.raises(OcrDerivativeError, match="content hash"):
        validate_ocr_derivative(replace(derivative, content=derivative.content + b"changed"))


def test_ocr_budget_rejects_before_raster_allocation(monkeypatch) -> None:
    monkeypatch.setattr(ocr_derivative, "_tesseract_version", lambda _: "test")
    raw = _one_page_pdf()
    def forbidden_render(*args, **kwargs):
        pytest.fail("Over-budget raster must not be allocated")
    monkeypatch.setattr(fitz.Page, "get_pixmap", forbidden_render)
    for bounds in [{"max_pixels_per_page": 1}, {"max_total_pixels": 1}]:
        with pytest.raises(OcrDerivativeError, match="OCR bound"):
            build_local_pdf_ocr_derivative(raw, **bounds)


def test_ocr_derivative_rejects_unvalidated_language() -> None:
    with pytest.raises(OcrDerivativeError, match="English"):
        build_local_pdf_ocr_derivative(_one_page_pdf(), language="chi_sim")


def test_isolated_ocr_revalidates_child_outputs(monkeypatch) -> None:
    pdf_bytes = _one_page_pdf()
    content = b"685\nA bounded isolated OCR derivative contains enough useful source text.\n"
    page = ocr_derivative.OcrPageResult(
        page_index=0,
        page_label="685",
        render_sha256="a" * 64,
        page_label_render_sha256="b" * 64,
        text_sha256=hashlib.sha256(content).hexdigest(),
        character_count=len(content),
        mean_word_confidence=91.0,
    )
    manifest = {
        "version": ocr_derivative.OCR_DERIVATIVE_VERSION,
        "parent_content_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "derivative_content_sha256": hashlib.sha256(content).hexdigest(),
        "engine": "tesseract",
        "engine_version": "test",
        "language": "eng",
        "render_dpi": 250,
        "page_segmentation_mode": 6,
        "page_count": 1,
        "pages": [page.__dict__],
        "resource_bounds": {
            "max_pages": 200,
            "max_pixels_per_page": 25_000_000,
            "max_total_pixels": 500_000_000,
            "timeout_seconds_per_page": 45,
        },
    }

    def fake_run(command, **_kwargs):
        derivative_path = Path(command[command.index("--derivative-output") + 1])
        manifest_path = Path(command[command.index("--manifest-output") + 1])
        derivative_path.write_bytes(content)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(ocr_derivative.subprocess, "run", fake_run)

    derivative = build_isolated_pdf_ocr_derivative(pdf_bytes)

    assert derivative.content == content
    assert derivative.page_labels == ("685",)


def test_isolated_ocr_rejects_manifest_configuration_drift(monkeypatch) -> None:
    pdf_bytes = _one_page_pdf()
    content = b"A bounded isolated OCR derivative contains enough useful source text.\n"
    page = ocr_derivative.OcrPageResult(
        page_index=0,
        page_label=None,
        render_sha256="a" * 64,
        page_label_render_sha256="b" * 64,
        text_sha256=hashlib.sha256(content).hexdigest(),
        character_count=len(content),
        mean_word_confidence=91.0,
    )
    manifest = {
        "version": ocr_derivative.OCR_DERIVATIVE_VERSION,
        "parent_content_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "derivative_content_sha256": hashlib.sha256(content).hexdigest(),
        "engine": "tesseract",
        "engine_version": "test",
        "language": "eng",
        "render_dpi": 300,
        "page_segmentation_mode": 6,
        "page_count": 1,
        "pages": [page.__dict__],
        "resource_bounds": {
            "max_pages": 200,
            "max_pixels_per_page": 25_000_000,
            "max_total_pixels": 500_000_000,
            "timeout_seconds_per_page": 45,
        },
    }

    def fake_run(command, **_kwargs):
        derivative_path = Path(command[command.index("--derivative-output") + 1])
        manifest_path = Path(command[command.index("--manifest-output") + 1])
        derivative_path.write_bytes(content)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(ocr_derivative.subprocess, "run", fake_run)

    with pytest.raises(OcrDerivativeError, match="requested configuration"):
        build_isolated_pdf_ocr_derivative(pdf_bytes)


def test_isolated_ocr_enforces_whole_job_deadline(monkeypatch) -> None:
    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr(ocr_derivative.subprocess, "run", timeout)

    with pytest.raises(OcrDerivativeError, match="total deadline"):
        build_isolated_pdf_ocr_derivative(_one_page_pdf(), total_timeout_seconds=1)
