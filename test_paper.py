#!/usr/bin/env python3
"""Manual test script for SourceFidelity functions.

Run individual functions without the FastAPI server:
    python test_paper.py --extract path/to/paper.pdf

Use VS Code debugger (F5) with breakpoints to step through.
"""

import argparse
import sys
from pathlib import Path


def test_extract_text(file_path: str):
    """Test text extraction from a PDF or DOCX file."""
    from app.services.text_extractor import extract_text

    print(f"Extracting text from: {file_path}")
    text = extract_text(file_path)
    print(f"\nExtracted {len(text)} characters")
    print("\n--- First 500 characters ---")
    print(text[:500])
    print("\n--- Last 500 characters ---")
    print(text[-500:])
    return text


def test_pdfplumber(file_path: str):
    """Test extraction with pdfplumber specifically."""
    from app.services.text_extractor import extract_from_pdf_pdfplumber

    print(f"Testing pdfplumber on: {file_path}")
    text = extract_from_pdf_pdfplumber(file_path)
    print(f"Extracted {len(text)} characters")
    print("\n--- First 500 characters ---")
    print(text[:500])
    return text


def test_pymupdf(file_path: str):
    """Test extraction with PyMuPDF specifically."""
    from app.services.text_extractor import extract_from_pdf_pymupdf

    print(f"Testing PyMuPDF on: {file_path}")
    text = extract_from_pdf_pymupdf(file_path)
    print(f"Extracted {len(text)} characters")
    print("\n--- First 500 characters ---")
    print(text[:500])
    return text


def test_extract_references(text: str):
    """Test reference extraction."""
    from app.services.reference_parser import extract_reference_section

    print("Testing reference section extraction...")
    refs = extract_reference_section(text)
    print(f"Found reference section: {len(refs) if refs else 0} characters")
    return refs


def main():
    parser = argparse.ArgumentParser(description="Test SourceFidelity functions")
    parser.add_argument("--extract", type=str, help="Test text extraction on a file")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pdfplumber", "pymupdf"],
        default="pdfplumber",
        help="PDF extraction backend to use",
    )
    parser.add_argument(
        "--refs", type=str, help="Extract references from extracted text file"
    )

    args = parser.parse_args()

    if args.extract:
        test_extract_text(args.extract)
    elif args.backend and args.extract:
        if args.backend == "pdfplumber":
            test_pdfplumber(args.extract)
        else:
            test_pymupdf(args.extract)
    elif args.refs:
        with open(args.refs, "r") as f:
            text = f.read()
        test_extract_references(text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
