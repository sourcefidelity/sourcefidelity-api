#!/usr/bin/env python3
"""Test the full reference parsing pipeline on real student papers.

Tests Phases 3.1-3.4:
- 3.1: Text extraction (PDF/DOCX)
- 3.2: Reference section extraction
- 3.3: Reference splitting
- 3.4: LLM parsing with caching

Usage:
    python test_real_papers.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env explicitly (backup if VS Code doesn't inject)
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


from app.services.text_extractor import extract_text
from app.services.reference_parser import (
    extract_reference_section,
    split_references,
    parse_reference_batch,
)
from app.services.doi_cache import get_cache_stats, clear_cache
from app.config import settings


# Test files
TEST_DATA_DIR = project_root / "test_data"

STUDENT_PAPERS = [
    "Paper - APA - Regulation.docx",
    "Paper - APA - Stardom - 1.docx",
    "Paper - APA - Stardom - 2.pdf",
    "Paper - APA - Stardom with Archive.org.docx",
    "Paper - MLA - Digital.docx",
    "Paper - MLA - Moral.pdf",
    "Paper - MLA - TV Comedy.pdf",
]


def test_paper(filename: str, use_llm: bool = True) -> dict:
    """Test the full pipeline on one paper.

    Returns dict with:
        - filename
        - file_type
        - text_length
        - format_detected
        - refs_section_found
        - refs_count
        - parsed_count
        - parse_errors
        - sample_parsed (first 3 parsed references)
        - elapsed_time
    """
    filepath = TEST_DATA_DIR / filename
    result = {
        "filename": filename,
        "file_type": Path(filename).suffix,
        "text_length": 0,
        "format_detected": None,
        "refs_section_found": False,
        "refs_count": 0,
        "parsed_count": 0,
        "parse_errors": [],
        "sample_parsed": [],
        "elapsed_time": 0,
    }

    start_time = time.time()

    # Step 1: Extract text
    try:
        text = extract_text(filepath)
        result["text_length"] = len(text)
        print(f"  ✓ Extracted {len(text):,} characters")
    except Exception as e:
        result["parse_errors"].append(f"Text extraction failed: {e}")
        print(f"  ✗ Text extraction failed: {e}")
        return result

    # Step 2: Extract reference section
    try:
        ref_section = extract_reference_section(text)
        if ref_section:
            result["refs_section_found"] = True
            result["format_detected"] = "APA" if "References" in text[:5000] else "MLA" if "Works Cited" in text[:5000] else "Unknown"
            print(f"  ✓ Found reference section ({len(ref_section):,} chars, format: {result['format_detected']})")
        else:
            result["parse_errors"].append("No reference section found")
            print(f"  ✗ No reference section found")
            return result
    except Exception as e:
        result["parse_errors"].append(f"Reference section extraction failed: {e}")
        print(f"  ✗ Reference section extraction failed: {e}")
        return result

    # Step 3: Split references
    try:
        refs = split_references(ref_section)
        result["refs_count"] = len(refs)
        print(f"  ✓ Split into {len(refs)} references")
        
        # Show first 2 raw refs for debugging
        for i, ref in enumerate(refs[:2]):
            preview = ref[:80] + "..." if len(ref) > 80 else ref
            print(f"    Ref {i+1}: {preview}")
    except Exception as e:
        result["parse_errors"].append(f"Reference splitting failed: {e}")
        print(f"  ✗ Reference splitting failed: {e}")
        return result

    # Step 4: Parse with LLM (optional)
    if use_llm and refs:
        try:
            print(f"  → Calling LLM to parse {len(refs)} references...")
            parsed = parse_reference_batch(refs)
            result["parsed_count"] = len(parsed)
            
            # Count successful parses (non-empty)
            successful = sum(1 for p in parsed if p.author or p.title)
            print(f"  ✓ Parsed {successful}/{len(parsed)} references successfully")
            
            # Sample first 3
            result["sample_parsed"] = [
                {
                    "author": p.author,
                    "year": p.year,
                    "title": p.title[:50] + "..." if len(p.title) > 50 else p.title,
                    "doi": p.doi,
                    "citation_key": p.citation_key,
                    "is_media": p.is_media_source,
                }
                for p in parsed[:3]
            ]
            
        except Exception as e:
            result["parse_errors"].append(f"LLM parsing failed: {e}")
            print(f"  ✗ LLM parsing failed: {e}")

    result["elapsed_time"] = time.time() - start_time
    return result


def run_all_tests(use_llm: bool = True):
    """Run pipeline tests on all student papers."""
    
    print("=" * 70)
    print("SourceFidelity Reference Parsing Pipeline Test")
    print("=" * 70)
    print(f"\nLLM Config:")
    print(f"  Model: {settings.LLM_MODEL}")
    print(f"  Base URL: {settings.LLM_BASE_URL}")
    print(f"  API Key: {'✓ Set' if settings.LLM_API_KEY and settings.LLM_API_KEY != 'sk-placeholder' else '✗ Not set or placeholder'}")
    print(f"  Batch Size: {settings.LLM_BATCH_SIZE}")
    print(f"  Cache Enabled: {settings.CACHE_ENABLED}")
    print()
    
    # Clear cache for fresh test
    clear_cache()
    
    results = []
    total_refs = 0
    total_parsed = 0
    
    for filename in STUDENT_PAPERS:
        print(f"\n{'='*70}")
        print(f"Testing: {filename}")
        print("=" * 70)
        
        result = test_paper(filename, use_llm=use_llm)
        results.append(result)
        
        total_refs += result["refs_count"]
        total_parsed += result["parsed_count"]
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nPapers tested: {len(results)}")
    print(f"Total references found: {total_refs}")
    print(f"Total parsed: {total_parsed}")
    
    # Per-paper summary
    print(f"\nPer-paper results:")
    for r in results:
        status = "✓" if r["parsed_count"] > 0 or not use_llm else "✗"
        print(f"  {status} {r['filename']}: {r['refs_count']} refs, {r['parsed_count']} parsed, {r['elapsed_time']:.1f}s")
        if r["parse_errors"]:
            for err in r["parse_errors"]:
                print(f"      Error: {err}")
    
    # Cache stats
    stats = get_cache_stats()
    print(f"\nCache stats: {stats}")
    
    # Show sample parsed references
    print(f"\nSample parsed references (first 3 from each paper):")
    for r in results:
        if r["sample_parsed"]:
            print(f"\n  {r['filename']}:")
            for i, p in enumerate(r["sample_parsed"]):
                print(f"    {i+1}. {p['citation_key']} ({p['year']})")
                print(f"       Author: {p['author']}")
                print(f"       Title: {p['title']}")
                print(f"       DOI: {p['doi']}")
                if p['is_media']:
                    print(f"       ⚠ Media source detected")
    
    return results


if __name__ == "__main__":
    # Check API key before running
    if not settings.LLM_API_KEY or settings.LLM_API_KEY == "sk-placeholder":
        print("WARNING: LLM_API_KEY not set or placeholder")
        print("Running extraction + splitting only (no LLM parsing)")
        print()
        run_all_tests(use_llm=False)
    else:
        run_all_tests(use_llm=True)