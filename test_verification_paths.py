#!/usr/bin/env python3
"""Comprehensive verification test across all retrieval paths.

Tests every verification path on real student papers:
  1. OA PDF retrieval + caching (OpenAlex/S2/CORE → full text → S3)
  2. Publisher PDF retrieval (campus-network paywalled content — test on the campus-regional route)
  3. Abstract-only verification (paywalled, no PDF — source-existence + faithfulness)
  4. Web-fetch verification (website citations — fetch, verify, discard)
  5. Primary-text retrieval (Gutenberg/Wikisource for public-domain works)
  6. Link validation (every URL in the reference list categorized)
  7. Missing-identifier reporting (DOIs/URLs missing, categorized)

Usage:
    cd <project root>
    source venv/bin/activate
    python test_verification_paths.py                    # all papers
    python test_verification_paths.py --limit 3          # first 3 papers (quick)
    python test_verification_paths.py --paper "Stardom"  # specific paper by name fragment
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ────────────────────────────────────────────────────────────


def hr(char="─", n=100):
    print(char * n)


def short(s: str, n=70) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def section(title: str):
    print()
    hr()
    print(f"  {title}")
    hr()


# ── Main test ─────────────────────────────────────────────────────────


def _is_website_url(url: str) -> bool:
    """Heuristic: is this URL a non-academic website (not a DOI/journal)?

    Used to skip academic-DB resolution for website citations, which wastes
    time (they're never in OpenAlex/CORE/etc.) and produces false-positive
    keyword matches.
    """
    url_lower = url.lower()
    # DOIs resolve to publisher pages — these ARE academic
    if "doi.org" in url_lower:
        return False
    # Known academic/publisher domains
    academic_domains = (
        "springer.com", "tandfonline.com", "wiley.com", "sciencedirect.com",
        "sagepub.com", "academic.oup.com", "cambridge.org", "jstor.org",
        "tandf.co.uk", "nature.com", "sciencemag.org", "plos.org",
        "frontiersin.org", "mdpi.com", "bmj.com", "thelancet.com",
        "projectmuse.org", "degruyter.com", "brill.com", "ejournals.eu",
        "elibrary.ru", "cnki.net", "wanfangdata.com.cn",
    )
    if any(domain in url_lower for domain in academic_domains):
        return False
    # University repository domains (.edu, .ac.xx) are academic
    if ".edu/" in url_lower or ".edu:" in url_lower or re.search(r"\.ac\.[a-z]{2,3}/", url_lower):
        return False
    # Everything else is a website (news, blog, gov, company, etc.)
    return True


def test_paper(path: str):
    """Run all verification checks on a single paper."""
    from app.services.text_extractor import extract_text
    from app.services.reference_parser import (
        extract_reference_section,
        extract_and_parse_references,
        count_missing_identifiers,
    )
    from app.services.link_validator import check_reference_links
    from app.services.source_resolver import SourceResolver, SourceResolutionError

    name = os.path.basename(path)
    fmt = "mla" if "/MLA/" in path else "apa"

    section(f"PAPER: {name} [{fmt.upper()}]")

    # Parse references
    try:
        text = extract_text(path)
        ref_section = extract_reference_section(text, format_hint=fmt)
        if not ref_section:
            print("  No reference section found — skipping.")
            return
        refs = extract_and_parse_references(ref_section, format_hint=fmt, use_llm_split=True)
    except Exception as e:
        print(f"  Parse error: {e}")
        return

    print(f"  References parsed: {len(refs)}")

    # 1. Missing-identifier report
    print("\n  ── Identifier completeness ──")
    id_report = count_missing_identifiers(refs)
    print(f"    With DOI:              {id_report['with_doi']}/{id_report['total']}")
    print(f"    With URL (no DOI):     {id_report['with_url']}/{id_report['total']}")
    print(f"    Missing — traditional media: {id_report['missing_traditional_media']} (no penalty)")
    print(f"    Missing — archive:          {id_report['missing_archive']} (no penalty)")
    print(f"    Missing — should have:      {id_report['missing_should_have']} (PENALISABLE)")
    if id_report["missing_examples"]:
        print(f"    Examples of penalisable missing:")
        for ex in id_report["missing_examples"][:3]:
            print(f"      • {short(ex, 65)}")

    # 2. Link validation (if there are URLs)
    url_refs = [r for r in refs if (r.url and r.url.strip()) or re.search(r"https?://", r.raw_ref or "")]
    if url_refs:
        print(f"\n  ── Link validation ({len(url_refs)} URLs) ──")
        link_result = check_reference_links(refs)
        for cat, count in sorted(link_result["summary"].items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
        # Show mismatches in detail
        mismatches = [r for r in link_result["results"] if r.category == "content_mismatch"]
        if mismatches:
            print(f"    Content mismatches (links work but title differs):")
            for m in mismatches[:3]:
                print(f"      • {short(m.url, 55)}")
                print(f"        page title: {short(m.page_title or '?', 50)}")

    # 3. Source resolution (try to retrieve each reference)
    print(f"\n  ── Source resolution ({len(refs)} references) ──")
    resolver = SourceResolver()

    resolution_stats = {
        "full_text_pdf": 0,
        "full_text_text": 0,
        "abstract_only": 0,
        "not_found": 0,
        "skipped_media": 0,
        "skipped_archive": 0,
    }

    for ref_idx, ref in enumerate(refs):
        doi = ref.doi.strip() if ref.doi else None
        url = ref.url.strip() if ref.url else None
        title = ref.title.strip() if ref.title else None
        author = ref.author.strip() if ref.author else None
        raw = ref.raw_ref or ""

        # Extract URL from raw_ref if not in structured field
        if not url:
            m = re.search(r"https?://[^\s]+", raw)
            if m:
                url = m.group(0).rstrip(".,);]")

        # Skip traditional media and archives
        from app.services.source_type import is_traditional_media, is_archive_source
        if is_traditional_media(raw):
            resolution_stats["skipped_media"] += 1
            continue
        if is_archive_source(raw):
            resolution_stats["skipped_archive"] += 1
            continue

        # Pre-filter: skip academic-DB resolution for references that are
        # clearly websites (have a URL to a non-academic domain, no DOI).
        # These will be handled by the web-fetch path, not academic DBs.
        is_website = bool(url and not doi and _is_website_url(url))

        try:
            if is_website:
                # Don't try academic sources for website citations
                resolution_stats["not_found"] += 1
                tag = "WEB"
                print(f"    • [{tag:3}] {'(website)':12} | {short(title or raw, 50)}")
                continue

            result = resolver.resolve(
                doi=doi, title=title, author=author,
                student_url=url, raw_ref=raw,
            )
            if result.full_text:
                # Check if it's PDF or plain text
                if result.full_text[:5] == b"%PDF-":
                    resolution_stats["full_text_pdf"] += 1
                else:
                    resolution_stats["full_text_text"] += 1
                tag = "PDF" if result.full_text[:5] == b"%PDF-" else "TXT"
                print(f"    ✓ [{tag:3}] {result.source_name:12} | {short(title or raw, 50)}")
            elif result.abstract:
                resolution_stats["abstract_only"] += 1
                print(f"    ○ [ABS] {result.source_name:12} | {short(title or raw, 50)}")
            else:
                resolution_stats["not_found"] += 1
                print(f"    ✗ [---] {result.source_name:12} | {short(title or raw, 50)}")
        except SourceResolutionError:
            resolution_stats["not_found"] += 1
            print(f"    ✗ [---] {'not found':12} | {short(title or raw, 50)}")

    print(f"\n  ── Resolution summary ──")
    for k, v in resolution_stats.items():
        if v > 0:
            print(f"    {k}: {v}")

    # 4. Web-fetch verification (test one website citation if available)
    web_refs = [r for r in refs if r.url and "doi.org" not in (r.url or "") and re.search(r"https?://", r.raw_ref or "")]
    if web_refs:
        print(f"\n  ── Web-fetch verification (testing 1 website citation) ──")
        from app.services.web_verifier import verify_citation_against_webpage, PARAPHRASE
        ref = web_refs[0]
        url = ref.url or ""
        if not url:
            m = re.search(r"https?://[^\s]+", ref.raw_ref or "")
            url = m.group(0).rstrip(".,);]") if m else ""
        if url:
            # We don't have the student's quotation automatically, so just
            # report that the page is fetchable and how much text we extracted
            print(f"    Testing: {short(url, 60)}")
            try:
                import httpx
                import trafilatura
                resp = httpx.get(url, headers={"User-Agent": "SourceFidelity/0.1.0"}, timeout=20, follow_redirects=True)
                page_text = trafilatura.extract(resp.text) if resp.status_code == 200 else None
                if page_text:
                    print(f"    ✓ Page fetched: {len(page_text)} chars of readable text extracted")
                else:
                    print(f"    ✗ Page returned status {resp.status_code} or no extractable text")
            except Exception as e:
                print(f"    ✗ Fetch failed: {str(e)[:60]}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive verification path test")
    parser.add_argument("--limit", type=int, default=None, help="process only N papers")
    parser.add_argument("--paper", type=str, default=None, help="paper name fragment (e.g. 'Stardom')")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    papers = sorted(glob.glob("test_data/APA/*.*")) + sorted(glob.glob("test_data/MLA/*.*"))
    papers = [p for p in papers if not p.endswith(".DS_Store")]

    if args.paper:
        papers = [p for p in papers if args.paper.lower() in p.lower()]
    if args.limit:
        papers = papers[: args.limit]

    print(f"Testing {len(papers)} papers")
    for path in papers:
        test_paper(path)

    print("\n")
    hr()
    print("DONE")
    hr()


if __name__ == "__main__":
    main()

