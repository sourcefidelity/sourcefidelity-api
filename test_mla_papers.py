#!/usr/bin/env python3
"""Test LLM-first reference extraction on MLA papers.

Usage:
    python test_mla_papers.py                    # Test all papers in test_data/MLA/
    python test_mla_papers.py --file "Name.pdf"  # Test a single file
    python test_mla_papers.py --verbose           # Show full reference details

This script:
1. Extracts text from each paper (PDF + DOCX)
2. Detects citation format (should detect MLA)
3. Extracts the reference section
4. Runs LLM-first extraction + parsing
5. Prints results for manual verification
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_single_paper(file_path: Path, verbose: bool = False) -> dict:
    """Test extraction on a single paper.

    Returns a dict with results for summary reporting.
    """
    from app.services.text_extractor import extract_text
    from app.services.parsers import detect_format
    from app.services.reference_parser import extract_reference_section, extract_and_parse_references

    filename = file_path.name
    result = {
        "filename": filename,
        "format_detected": None,
        "ref_section_found": False,
        "ref_section_chars": 0,
        "num_refs_parsed": 0,
        "refs": [],
        "error": None,
    }

    print(f"\n{'='*70}")
    print(f"FILE: {filename}")
    print(f"{'='*70}")

    # Step 1: Extract text
    try:
        text = extract_text(str(file_path))
        print(f"  Text extracted: {len(text):,} chars")
    except Exception as e:
        result["error"] = f"Text extraction failed: {e}"
        print(f"  ✗ {result['error']}")
        return result

    # Step 2: Detect format
    parser = detect_format(text)
    result["format_detected"] = parser.__name__
    print(f"  Format detected: {parser.__name__}")

    if parser.__name__ != "MlaParser":
        print(f"  ⚠ WARNING: Expected MlaParser, got {parser.__name__}")

    # Step 3: Extract reference section
    ref_section = extract_reference_section(text, format_hint="mla")
    if ref_section:
        result["ref_section_found"] = True
        result["ref_section_chars"] = len(ref_section)
        print(f"  Reference section: {len(ref_section):,} chars")
    else:
        result["error"] = "No reference section found"
        print(f"  ✗ {result['error']}")
        return result

    # Step 4: LLM-first extraction + parsing
    print(f"  Calling LLM (this may take 10-30 seconds)...")
    start_time = time.time()
    try:
        parsed_refs = extract_and_parse_references(
            ref_section,
            format_hint="mla",
            use_llm_split=True,
        )
    except NotImplementedError:
        # If format detection misroutes to MLA regex, try with explicit apa fallback
        print("  ⚠ NotImplementedError from MLA regex — this shouldn't happen with LLM-first")
        print("  Retrying with format_hint='apa' for section extraction...")
        ref_section = extract_reference_section(text, format_hint="apa")
        if not ref_section:
            result["error"] = "No reference section found (APA fallback also failed)"
            print(f"  ✗ {result['error']}")
            return result
        parsed_refs = extract_and_parse_references(
            ref_section,
            format_hint="mla",
            use_llm_split=True,
        )
    except Exception as e:
        result["error"] = f"LLM extraction failed: {e}"
        print(f"  ✗ {result['error']}")
        return result

    elapsed = time.time() - start_time
    result["num_refs_parsed"] = len(parsed_refs)
    result["refs"] = parsed_refs

    print(f"  Parsed {len(parsed_refs)} references in {elapsed:.1f}s")

    # Step 5: Display results
    if verbose:
        print(f"\n  --- Parsed References ---")
        for i, ref in enumerate(parsed_refs, 1):
            print(f"\n  [{i}] Author:   {ref.author}")
            print(f"      Year:     {ref.year}")
            print(f"      Title:    {ref.title[:80]}{'...' if len(ref.title) > 80 else ''}")
            print(f"      DOI:      {ref.doi or '(none)'}")
            print(f"      URL:      {ref.url[:80] if ref.url else '(none)'}")
            print(f"      Key:      {ref.citation_key}")
            print(f"      Media:    {ref.is_media_source}")
            print(f"      Raw:      {ref.raw_ref[:100]}{'...' if len(ref.raw_ref) > 100 else ''}")
    else:
        # Compact display
        for i, ref in enumerate(parsed_refs, 1):
            print(f"  [{i}] {ref.author} ({ref.year}). {ref.title[:60]}{'...' if len(ref.title) > 60 else ''}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test MLA reference extraction")
    parser.add_argument("--file", type=str, help="Test a single file by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full reference details")
    args = parser.parse_args()

    mla_dir = Path("test_data/MLA")

    if args.file:
        files = [mla_dir / args.file]
        if not files[0].exists():
            print(f"File not found: {files[0]}")
            sys.exit(1)
    else:
        files = sorted(mla_dir.iterdir())

    print(f"Testing {len(files)} MLA paper(s)")
    print(f"Directory: {mla_dir.resolve()}")

    all_results = []
    for f in files:
        if f.is_dir():
            continue
        result = test_single_paper(f, verbose=args.verbose)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'File':<55} {'Format':<12} {'Refs':>5}")
    print(f"{'-'*55} {'-'*12} {'-'*5}")

    total_refs = 0
    for r in all_results:
        fname = r["filename"][:55]
        fmt = r["format_detected"] or "ERROR"
        nrefs = r["num_refs_parsed"]
        total_refs += nrefs
        status = "✗" if r["error"] else "✓"
        print(f"{status} {fname:<53} {fmt:<12} {nrefs:>5}")

    print(f"\nTotal references parsed: {total_refs}")
    errors = [r for r in all_results if r["error"]]
    if errors:
        print(f"Errors: {len(errors)}")
        for r in errors:
            print(f"  {r['filename']}: {r['error']}")


if __name__ == "__main__":
    main()
