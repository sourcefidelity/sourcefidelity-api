#!/usr/bin/env python3
"""Test script for Phase 3.4 LLM Reference Parsing.

This script tests:
1. LLM service connection
2. Reference parsing with mock data
3. Caching layer
4. Pydantic validation

Usage:
    python test_llm_parsing.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up test environment
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "deepseek-chat")
os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com/v1")

from app.services.schemas import ParsedReference, validate_llm_reference_array
from app.services.prompts import REFERENCE_PARSE_SYSTEM_PROMPT, build_reference_parse_user_prompt
from app.services import doi_cache


def test_schemas():
    """Test Pydantic schema validation."""
    print("\n=== Testing Schemas ===")
    
    # Test valid reference
    ref_data = {
        "author": "Smith, J. & Doe, A.",
        "year": "2023",
        "title": "A Great Paper",
        "doi": "10.1234/test",
        "url": "https://doi.org/10.1234/test",
        "raw_ref": "Smith, J. & Doe, A. (2023). A Great Paper. Journal of Testing, 1(1), 1-10.",
        "citation_key": "Smith2023",
        "is_media_source": False,
    }
    
    ref = ParsedReference(**ref_data)
    assert ref.author == "Smith, J. & Doe, A."
    assert ref.year == "2023"
    assert ref.doi == "10.1234/test"
    print(f"✓ Valid reference: {ref.citation_key}")
    
    # Test DOI normalization
    ref_data2 = {
        "author": "Jones, B.",
        "year": "2022",
        "title": "Another Paper",
        "doi": "https://doi.org/10.5678/example",  # Should be normalized
        "url": "",
        "raw_ref": "Jones, B. (2022). Another Paper.",
        "citation_key": "Jones2022",
        "is_media_source": False,
    }
    
    ref2 = ParsedReference(**ref_data2)
    assert ref2.doi == "10.5678/example", f"Expected normalized DOI, got: {ref2.doi}"
    print(f"✓ DOI normalized: 'https://doi.org/10.5678/example' → '{ref2.doi}'")
    
    # Test year normalization
    ref_data3 = {
        "author": "Brown, C.",
        "year": "2021, March",  # Should extract 2021
        "title": "Paper with Month",
        "doi": "",
        "url": "",
        "raw_ref": "Brown, C. (2021, March). Paper with Month.",
        "citation_key": "Brown2021",
        "is_media_source": False,
    }
    
    ref3 = ParsedReference(**ref_data3)
    assert ref3.year == "2021", f"Expected year extraction, got: {ref3.year}"
    print(f"✓ Year normalized: '2021, March' → '{ref3.year}'")
    
    print("✓ All schema tests passed")


def test_validate_llm_response():
    """Test LLM response validation."""
    print("\n=== Testing LLM Response Validation ===")
    
    # Test valid response
    valid_response = '''[
        {
            "author": "Smith, J.",
            "year": "2023",
            "title": "Test Paper",
            "doi": "10.1234/test",
            "url": "https://doi.org/10.1234/test",
            "raw_ref": "Smith, J. (2023). Test Paper.",
            "citation_key": "Smith2023",
            "is_media_source": false
        }
    ]'''
    
    refs = validate_llm_reference_array(valid_response, expected_count=1)
    assert len(refs) == 1
    assert refs[0]["author"] == "Smith, J."
    print(f"✓ Valid response validated: {len(refs)} reference(s)")
    
    # Test count mismatch
    try:
        validate_llm_reference_array(valid_response, expected_count=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected 2 references" in str(e)
        print(f"✓ Count mismatch detected correctly")
    
    # Test markdown removal
    markdown_response = '''```json
    [
        {
            "author": "Doe, A.",
            "year": "2024",
            "title": "Markdown Test",
            "doi": "",
            "url": "",
            "raw_ref": "Doe, A. (2024). Markdown Test.",
            "citation_key": "Doe2024",
            "is_media_source": false
        }
    ]
    ```'''
    
    refs = validate_llm_reference_array(markdown_response, expected_count=1)
    assert len(refs) == 1
    print(f"✓ Markdown code block removed correctly")
    
    print("✓ All validation tests passed")


def test_prompts():
    """Test prompt generation."""
    print("\n=== Testing Prompt Generation ===")
    
    refs = [
        "Smith, J. (2023). First Paper. Journal A, 1(1), 1-10.",
        "Doe, A. (2022). Second Paper. Journal B, 2(2), 11-20.",
    ]
    
    system = REFERENCE_PARSE_SYSTEM_PROMPT.format(reference_count=2)
    assert "{reference_count}" not in system
    assert "exactly 2 references" in system
    print(f"✓ System prompt formatted correctly (count: 2)")
    
    user = build_reference_parse_user_prompt(refs)
    assert "exactly 2 references" in user
    assert "Smith, J." in user
    print(f"✓ User prompt formatted correctly")
    
    print(f"\n--- System Prompt Preview ---")
    print(system[:200] + "...")
    
    print(f"\n--- User Prompt Preview ---")
    print(user[:300] + "...")
    
    print("✓ All prompt tests passed")


def test_cache():
    """Test DOI/title-hash caching."""
    print("\n=== Testing Cache Layer ===")
    
    # Clear cache
    doi_cache.clear_cache()
    
    # Test DOI normalization
    doi1 = doi_cache.normalize_doi("https://doi.org/10.1234/test")
    assert doi1 == "10.1234/test"
    print(f"✓ DOI normalized: 'https://doi.org/10.1234/test' → '{doi1}'")
    
    doi2 = doi_cache.normalize_doi("doi:10.5678/example")
    assert doi2 == "10.5678/example"
    print(f"✓ DOI normalized: 'doi:10.5678/example' → '{doi2}'")
    
    # Test title hashing
    hash1 = doi_cache.compute_title_hash("A Great Paper")
    hash2 = doi_cache.compute_title_hash("A  GREAT  paper")  # Normalized
    assert hash1 == hash2
    print(f"✓ Title hash consistent for normalized titles")
    
    # Test cache operations
    ref_data = {
        "author": "Smith, J.",
        "year": "2023",
        "title": "Cached Paper",
        "doi": "10.9999/cached",
        "url": "https://doi.org/10.9999/cached",
        "raw_ref": "Smith, J. (2023). Cached Paper.",
        "citation_key": "Smith2023",
        "is_media_source": False,
    }
    
    # Cache by DOI
    doi_cache.cache_reference(ref_data, doi=ref_data["doi"], title=ref_data["title"])
    
    # Retrieve by DOI
    cached = doi_cache.get_cached_reference(doi="10.9999/cached")
    assert cached is not None
    assert cached["author"] == "Smith, J."
    print(f"✓ Cache hit by DOI: {cached['citation_key']}")
    
    # Retrieve by title
    cached2 = doi_cache.get_cached_reference(title="Cached Paper")
    assert cached2 is not None
    print(f"✓ Cache hit by title hash")
    
    # Test cache stats
    stats = doi_cache.get_cache_stats()
    print(f"✓ Cache stats: {stats}")
    
    # Clear cache
    doi_cache.clear_cache()
    stats = doi_cache.get_cache_stats()
    assert stats["size"] == 0
    print(f"✓ Cache cleared")
    
    print("✓ All cache tests passed")


def test_heuristics():
    """Test DOI/title extraction heuristics."""
    print("\n=== Testing Extraction Heuristics ===")
    
    # Test DOI extraction
    ref_with_doi = "Smith, J. (2023). Paper Title. Journal, 1, 1-10. https://doi.org/10.1234/paper"
    doi = doi_cache._extract_doi_heuristic(ref_with_doi)
    assert doi == "10.1234/paper"
    print(f"✓ DOI extracted: '{doi}'")
    
    ref_with_doi2 = "Doe, A. (2022). Paper. doi:10.5678/example"
    doi2 = doi_cache._extract_doi_heuristic(ref_with_doi2)
    assert doi2 == "10.5678/example"
    print(f"✓ DOI extracted from 'doi:' prefix: '{doi2}'")
    
    # Test title extraction
    ref_with_title = "Brown, C. (2021). An Amazing Discovery in Science. Journal, 3, 30-40."
    title = doi_cache._extract_title_heuristic(ref_with_title)
    assert title is not None
    print(f"✓ Title extracted: '{title}'")
    
    print("✓ All heuristic tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Phase 3.4 LLM Reference Parsing - Unit Tests")
    print("=" * 60)
    
    try:
        test_schemas()
        test_validate_llm_response()
        test_prompts()
        test_cache()
        test_heuristics()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()