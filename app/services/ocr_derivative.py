"""Development-only construction of immutable local PDF OCR derivatives.

This module does not admit a source or alter the durable source repository.  It
binds bounded local OCR output to the exact parent PDF and rendered page bytes
so the derivative can be evaluated before a production OCR intake path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import fitz


OCR_DERIVATIVE_VERSION = "local-pdf-ocr-derivative-v1"


class OcrDerivativeError(ValueError):
    """The input or OCR output failed a bounded derivative requirement."""


@dataclass(frozen=True)
class OcrPageResult:
    page_index: int
    page_label: str | None
    render_sha256: str
    page_label_render_sha256: str | None
    text_sha256: str
    character_count: int
    mean_word_confidence: float | None


@dataclass(frozen=True)
class OcrDerivative:
    content: bytes
    content_sha256: str
    parent_content_sha256: str
    page_labels: tuple[str | None, ...]
    page_results: tuple[OcrPageResult, ...]
    manifest: dict
    manifest_sha256: str


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_ocr_text(value: str) -> str:
    lines = [
        line.rstrip()
        for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _visible_page_label(text: str) -> str | None:
    candidates = set(re.findall(r"(?<!\d)(\d{1,4})(?!\d)", text))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _tesseract_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OcrDerivativeError("The configured local OCR engine is unavailable.") from exc
    first_line = (result.stdout or result.stderr).splitlines()
    if not first_line:
        raise OcrDerivativeError("The local OCR engine did not report a version.")
    return first_line[0].strip()[:120]


def _run_tesseract_page(
    image_path: Path,
    output_base: Path,
    *,
    executable: str,
    language: str,
    dpi: int,
    page_segmentation_mode: int,
    timeout_seconds: int,
) -> tuple[str, float | None]:
    try:
        subprocess.run(
            [
                executable,
                str(image_path),
                str(output_base),
                "-l",
                language,
                "--dpi",
                str(dpi),
                "--psm",
                str(page_segmentation_mode),
                "txt",
                "tsv",
            ],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        text = output_base.with_suffix(".txt").read_text(
            encoding="utf-8", errors="replace"
        )
        tsv = output_base.with_suffix(".tsv").read_text(
            encoding="utf-8", errors="replace"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OcrDerivativeError("Local OCR failed within the configured bounds.") from exc
    confidences: list[float] = []
    for line in tsv.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) < 12 or not columns[11].strip():
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence >= 0:
            confidences.append(confidence)
    mean = sum(confidences) / len(confidences) if confidences else None
    return text, mean


def build_local_pdf_ocr_derivative(
    pdf_bytes: bytes,
    *,
    language: str = "eng",
    dpi: int = 250,
    page_segmentation_mode: int = 6,
    max_pages: int = 200,
    max_pixels_per_page: int = 25_000_000,
    max_total_pixels: int = 500_000_000,
    timeout_seconds_per_page: int = 45,
    executable: str = "tesseract",
) -> OcrDerivative:
    """Render and OCR a bounded PDF without mutating or admitting the original."""
    if language != "eng":
        raise OcrDerivativeError("Only the validated English OCR language is enabled.")
    if not 150 <= dpi <= 400:
        raise OcrDerivativeError("OCR DPI must be between 150 and 400.")
    if page_segmentation_mode != 6:
        raise OcrDerivativeError(
            "Only the validated OCR page segmentation mode is enabled."
        )
    if max_pages < 1 or timeout_seconds_per_page < 1:
        raise OcrDerivativeError("OCR resource bounds must be positive.")
    parent_sha256 = _sha256(pdf_bytes)
    engine_version = _tesseract_version(executable)
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except (fitz.FileDataError, fitz.mupdf.FzErrorBase, RuntimeError) as exc:
        raise OcrDerivativeError("The parent bytes are not a readable PDF.") from exc
    try:
        if document.needs_pass:
            raise OcrDerivativeError("Encrypted PDFs are not eligible for local OCR.")
        if not 1 <= document.page_count <= max_pages:
            raise OcrDerivativeError("PDF page count exceeds the local OCR bound.")
        texts: list[str] = []
        page_results: list[OcrPageResult] = []
        total_pixels = 0
        with tempfile.TemporaryDirectory(prefix="sourcefidelity-ocr-") as directory:
            temp_dir = Path(directory)
            for page_index, page in enumerate(document):
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixels = pixmap.width * pixmap.height
                total_pixels += pixels
                if pixels > max_pixels_per_page or total_pixels > max_total_pixels:
                    raise OcrDerivativeError("Rendered PDF pixels exceed the OCR bound.")
                png_bytes = pixmap.tobytes("png")
                image_path = temp_dir / f"page-{page_index + 1:04d}.png"
                image_path.write_bytes(png_bytes)
                raw_text, mean_confidence = _run_tesseract_page(
                    image_path,
                    temp_dir / f"page-{page_index + 1:04d}",
                    executable=executable,
                    language=language,
                    dpi=dpi,
                    page_segmentation_mode=page_segmentation_mode,
                    timeout_seconds=timeout_seconds_per_page,
                )
                text = _normalize_ocr_text(raw_text)
                header_rect = fitz.Rect(
                    page.rect.x0,
                    page.rect.y0,
                    page.rect.x1,
                    page.rect.y0 + page.rect.height * 0.18,
                )
                header_pixmap = page.get_pixmap(
                    matrix=matrix, clip=header_rect, alpha=False
                )
                header_png = header_pixmap.tobytes("png")
                header_path = temp_dir / f"page-{page_index + 1:04d}-header.png"
                header_path.write_bytes(header_png)
                raw_label_text, _label_confidence = _run_tesseract_page(
                    header_path,
                    temp_dir / f"page-{page_index + 1:04d}-header",
                    executable=executable,
                    language=language,
                    dpi=dpi,
                    page_segmentation_mode=page_segmentation_mode,
                    timeout_seconds=timeout_seconds_per_page,
                )
                label = _visible_page_label(raw_label_text)
                texts.append(text)
                page_results.append(
                    OcrPageResult(
                        page_index=page_index,
                        page_label=label,
                        render_sha256=_sha256(png_bytes),
                        page_label_render_sha256=_sha256(header_png),
                        text_sha256=_sha256(text.encode("utf-8")),
                        character_count=len(text),
                        mean_word_confidence=(
                            round(mean_confidence, 3)
                            if mean_confidence is not None
                            else None
                        ),
                    )
                )
    finally:
        document.close()
    content = "\f".join(texts).encode("utf-8")
    if sum(item.character_count for item in page_results) < 100:
        raise OcrDerivativeError("OCR output did not contain enough usable text.")
    manifest = {
        "version": OCR_DERIVATIVE_VERSION,
        "parent_content_sha256": parent_sha256,
        "derivative_content_sha256": _sha256(content),
        "engine": "tesseract",
        "engine_version": engine_version,
        "language": language,
        "render_dpi": dpi,
        "page_segmentation_mode": page_segmentation_mode,
        "page_count": len(page_results),
        "pages": [item.__dict__ for item in page_results],
        "resource_bounds": {
            "max_pages": max_pages,
            "max_pixels_per_page": max_pixels_per_page,
            "max_total_pixels": max_total_pixels,
            "timeout_seconds_per_page": timeout_seconds_per_page,
        },
    }
    return OcrDerivative(
        content=content,
        content_sha256=_sha256(content),
        parent_content_sha256=parent_sha256,
        page_labels=tuple(item.page_label for item in page_results),
        page_results=tuple(page_results),
        manifest=manifest,
        manifest_sha256=_sha256(_canonical_json_bytes(manifest)),
    )


def validate_ocr_derivative(derivative: OcrDerivative) -> None:
    """Fail closed when derivative bytes or text-free provenance were altered."""
    if _sha256(derivative.content) != derivative.content_sha256:
        raise OcrDerivativeError("OCR derivative content hash does not match.")
    pages = derivative.content.decode("utf-8", "strict").split("\f")
    if len(pages) != len(derivative.page_results):
        raise OcrDerivativeError("OCR derivative page count does not match.")
    for text, result in zip(pages, derivative.page_results):
        if _sha256(text.encode("utf-8")) != result.text_sha256:
            raise OcrDerivativeError("OCR derivative page hash does not match.")
    if (
        derivative.manifest.get("parent_content_sha256")
        != derivative.parent_content_sha256
    ):
        raise OcrDerivativeError("OCR derivative parent binding does not match.")
    if (
        derivative.manifest.get("derivative_content_sha256")
        != derivative.content_sha256
    ):
        raise OcrDerivativeError("OCR derivative manifest content binding does not match.")
    if derivative.manifest.get("pages") != [
        item.__dict__ for item in derivative.page_results
    ]:
        raise OcrDerivativeError("OCR derivative page provenance does not match.")
    if _sha256(_canonical_json_bytes(derivative.manifest)) != derivative.manifest_sha256:
        raise OcrDerivativeError("OCR derivative manifest hash does not match.")


def build_isolated_pdf_ocr_derivative(
    pdf_bytes: bytes,
    *,
    language: str = "eng",
    dpi: int = 250,
    page_segmentation_mode: int = 6,
    max_pages: int = 200,
    max_pixels_per_page: int = 25_000_000,
    max_total_pixels: int = 500_000_000,
    timeout_seconds_per_page: int = 45,
    total_timeout_seconds: int = 900,
    max_derivative_bytes: int = 20_000_000,
    executable: str = "tesseract",
) -> OcrDerivative:
    """Build a derivative in a child process with one hard overall deadline."""
    if total_timeout_seconds < 1 or max_derivative_bytes < 1:
        raise OcrDerivativeError("Isolated OCR bounds must be positive.")
    with tempfile.TemporaryDirectory(prefix="sourcefidelity-ocr-isolated-") as directory:
        temp_dir = Path(directory)
        input_path = temp_dir / "parent.pdf"
        derivative_path = temp_dir / "derivative.txt"
        manifest_path = temp_dir / "manifest.json"
        input_path.write_bytes(pdf_bytes)
        command = [
            sys.executable,
            "-m",
            "app.services.ocr_worker_cli",
            "--input",
            str(input_path),
            "--derivative-output",
            str(derivative_path),
            "--manifest-output",
            str(manifest_path),
            "--language",
            language,
            "--dpi",
            str(dpi),
            "--page-segmentation-mode",
            str(page_segmentation_mode),
            "--max-pages",
            str(max_pages),
            "--max-pixels-per-page",
            str(max_pixels_per_page),
            "--max-total-pixels",
            str(max_total_pixels),
            "--timeout-seconds-per-page",
            str(timeout_seconds_per_page),
            "--executable",
            executable,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=total_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrDerivativeError("Isolated OCR exceeded its total deadline.") from exc
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OcrDerivativeError("Isolated OCR worker failed.") from exc
        if result.stdout or result.stderr:
            raise OcrDerivativeError("Isolated OCR worker emitted unexpected output.")
        if not derivative_path.is_file() or not manifest_path.is_file():
            raise OcrDerivativeError("Isolated OCR worker omitted required outputs.")
        if derivative_path.stat().st_size > max_derivative_bytes:
            raise OcrDerivativeError("OCR derivative exceeds its byte limit.")
        if manifest_path.stat().st_size > 10_000_000:
            raise OcrDerivativeError("OCR derivative manifest exceeds its byte limit.")
        content = derivative_path.read_bytes()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            page_results = tuple(
                OcrPageResult(**item) for item in manifest.get("pages", [])
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise OcrDerivativeError("OCR worker manifest is invalid.") from exc
        derivative = OcrDerivative(
            content=content,
            content_sha256=_sha256(content),
            parent_content_sha256=_sha256(pdf_bytes),
            page_labels=tuple(item.page_label for item in page_results),
            page_results=page_results,
            manifest=manifest,
            manifest_sha256=_sha256(_canonical_json_bytes(manifest)),
        )
        validate_ocr_derivative(derivative)
        expected_manifest_values = {
            "version": OCR_DERIVATIVE_VERSION,
            "parent_content_sha256": _sha256(pdf_bytes),
            "derivative_content_sha256": _sha256(content),
            "engine": "tesseract",
            "language": language,
            "render_dpi": dpi,
            "page_segmentation_mode": page_segmentation_mode,
            "page_count": len(page_results),
            "resource_bounds": {
                "max_pages": max_pages,
                "max_pixels_per_page": max_pixels_per_page,
                "max_total_pixels": max_total_pixels,
                "timeout_seconds_per_page": timeout_seconds_per_page,
            },
        }
        if any(
            manifest.get(key) != value
            for key, value in expected_manifest_values.items()
        ) or not isinstance(manifest.get("engine_version"), str):
            raise OcrDerivativeError(
                "OCR worker manifest does not match the requested configuration."
            )
        return derivative
