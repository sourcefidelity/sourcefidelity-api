"""Private command-line boundary for one isolated local OCR job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.ocr_derivative import build_local_pdf_ocr_derivative


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--derivative-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--dpi", required=True, type=int)
    parser.add_argument("--page-segmentation-mode", required=True, type=int)
    parser.add_argument("--max-pages", required=True, type=int)
    parser.add_argument("--max-pixels-per-page", required=True, type=int)
    parser.add_argument("--max-total-pixels", required=True, type=int)
    parser.add_argument("--timeout-seconds-per-page", required=True, type=int)
    parser.add_argument("--executable", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    input_path = Path(arguments.input)
    derivative_path = Path(arguments.derivative_output)
    manifest_path = Path(arguments.manifest_output)
    if input_path.parent != derivative_path.parent or input_path.parent != manifest_path.parent:
        raise ValueError("OCR worker files must share one private temporary directory")
    derivative = build_local_pdf_ocr_derivative(
        input_path.read_bytes(),
        language=arguments.language,
        dpi=arguments.dpi,
        page_segmentation_mode=arguments.page_segmentation_mode,
        max_pages=arguments.max_pages,
        max_pixels_per_page=arguments.max_pixels_per_page,
        max_total_pixels=arguments.max_total_pixels,
        timeout_seconds_per_page=arguments.timeout_seconds_per_page,
        executable=arguments.executable,
    )
    derivative_path.write_bytes(derivative.content)
    manifest_path.write_text(
        json.dumps(
            derivative.manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
