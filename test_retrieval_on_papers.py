#!/usr/bin/env python3
"""Test retrieval coverage against parsed references from APA/MLA student papers.

Parses references from every student paper via the LLM (Phase 3.4), then
looks each unique reference up via OpenAlex and CORE (DOI-first, then title).
Reports the real-world retrieval rate students would experience.

Usage:
    cd <project root>
    source venv/bin/activate
    python test_retrieval_on_papers.py
    python test_retrieval_on_papers.py --limit 5      # first 5 papers only (quick check)
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bibliographic sources have DOI (journal articles) or ISBN (books) or neither.
# We attempt DOI lookup first; if no DOI, we attempt title+author lookup.


def hr(char="─", n=90):
    print(char * n)


def short(s: str, n=60) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    parser = argparse.ArgumentParser(description="Retrieval coverage on student-paper references")
    parser.add_argument("--limit", type=int, default=None, help="process only N papers (for quick checks)")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    from app.services.text_extractor import extract_text
    from app.services.reference_parser import extract_reference_section, extract_and_parse_references
    from app.services.retrieval.openalex import OpenAlexRetriever
    from app.services.retrieval.core import CoreRetriever

    oa = OpenAlexRetriever()
    cr = CoreRetriever()

    papers = sorted(glob.glob("test_data/APA/*.*")) + sorted(glob.glob("test_data/MLA/*.*"))
    papers = [p for p in papers if not p.endswith(".DS_Store")]
    if args.limit:
        papers = papers[: args.limit]

    hr()
    print(f"STUDENT PAPERS: {len(papers)} ({len(glob.glob('test_data/APA/*.*'))} APA + {len(glob.glob('test_data/MLA/*.*'))} MLA)")
    hr()

    # Aggregate across all papers (dedup by DOI or title-key).
    all_refs = []          # list of (paper, ParsedReference)
    total_parsed = 0
    parse_failures = []

    for path in papers:
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        fmt = "mla" if "/MLA/" in path else "apa"
        try:
            text = extract_text(path)
            if not text.strip():
                parse_failures.append((name, "no text extracted"))
                continue
            ref_section = extract_reference_section(text, format_hint=fmt)
            if not ref_section or not ref_section.strip():
                parse_failures.append((name, "no reference section found"))
                continue
            refs = extract_and_parse_references(ref_section, format_hint=fmt, use_llm_split=True)
            total_parsed += len(refs)
            for r in refs:
                all_refs.append((name, r))
            print(f"  {short(name, 45):47s} [{fmt}]  refs={len(refs)}")
        except Exception as e:
            parse_failures.append((name, str(e)[:60]))
            print(f"  {short(name, 45):47s} [{fmt}]  PARSE ERROR: {str(e)[:50]}")

    if parse_failures:
        print(f"\n  Parse failures: {len(parse_failures)}")
        for n, err in parse_failures:
            print(f"    - {short(n, 40)}: {err}")

    # Deduplicate references across papers. A reference is "the same" if it
    # shares a DOI, or (if no DOI) a normalized title.
    seen = set()
    unique_refs = []
    for _paper, r in all_refs:
        key = r.doi.strip().lower() if r.doi.strip() else _norm_title(r.title)
        if not key:
            key = r.raw_ref[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append(r)

    hr()
    print(f"REFERENCES: {total_parsed} parsed | {len(unique_refs)} unique (deduped across papers)")
    hr()

    # Classification
    with_doi = [r for r in unique_refs if r.doi.strip()]
    without_doi = [r for r in unique_refs if not r.doi.strip()]
    print(f"  With DOI:    {len(with_doi)} ({len(with_doi)/len(unique_refs)*100:.0f}%)")
    print(f"  Without DOI: {len(without_doi)} ({len(without_doi)/len(unique_refs)*100:.0f}%)")
    hr()

    # Retrieval — DOI first, then title fallback
    stats = {
        "doi_oa": 0, "doi_core": 0,
        "title_oa": 0, "title_core": 0,
        "not_found": 0,
        "no_doi_no_title": 0,
    }
    found_by_any = set()

    print(f"\n{'#':<4} {'DOI?':<5} {'OA':<4} {'CORE':<5} {'title':<55}")
    print("-" * 90)

    for i, r in enumerate(unique_refs, 1):
        title_disp = short(r.title or "(no title)", 53)
        has_doi = bool(r.doi.strip())
        doi_tag = "DOI" if has_doi else "ttl"
        oa_hit = core_hit = False

        # 1. DOI lookup (if available)
        if has_doi:
            ro = oa.search_by_doi(r.doi.strip())
            if ro.success:
                oa_hit = True
                stats["doi_oa"] += 1
            rc = cr.search_by_doi(r.doi.strip())
            if rc.success:
                core_hit = True
                stats["doi_core"] += 1
        # 2. Title lookup (fallback, or primary if no DOI)
        elif r.title.strip():
            ro = oa.search_by_title_author(r.title.strip(), r.author.strip() or None)
            if ro.success:
                oa_hit = True
                stats["title_oa"] += 1
            rc = cr.search_by_title_author(r.title.strip(), r.author.strip() or None)
            if rc.success:
                core_hit = True
                stats["title_core"] += 1
        else:
            stats["no_doi_no_title"] += 1

        if oa_hit or core_hit:
            found_by_any.add(i)
        else:
            stats["not_found"] += 1

        oa_s = "✓" if oa_hit else "✗"
        core_s = "✓" if core_hit else "✗"
        print(f"{i:<4} {doi_tag:<5} {oa_s:<4} {core_s:<5} {title_disp}")

    # Summary
    hr()
    n = len(unique_refs)
    found = len(found_by_any)
    print(f"\nRETRIEVAL SUMMARY ({n} unique references)")
    print(f"  Found by ≥1 source:        {found}/{n}  ({found/n*100:.0f}%)")
    print(f"  Not found:                 {stats['not_found']}/{n}")
    print(f"  No DOI and no title:       {stats['no_doi_no_title']}/{n} (unsearchable)")
    print()
    print(f"  By path:")
    print(f"    DOI lookups  — OpenAlex: {stats['doi_oa']}, CORE: {stats['doi_core']}")
    print(f"    Title lookups — OpenAlex: {stats['title_oa']}, CORE: {stats['title_core']}")
    doi_found = sum(1 for r in found_by_any if unique_refs[r-1].doi.strip()) if False else None
    print()
    print(f"  Among {len(with_doi)} refs WITH a DOI:    {stats['doi_oa']}/{len(with_doi)} found via OpenAlex, {stats['doi_core']}/{len(with_doi)} via CORE")
    no_doi_searchable = len(without_doi) - stats["no_doi_no_title"]
    no_doi_found = stats["title_oa"]  # rough proxy (OpenAlex)
    if no_doi_searchable > 0:
        print(f"  Among {no_doi_searchable} refs WITHOUT DOI but with title: {no_doi_found}/{no_doi_searchable} found via OpenAlex title search")
    hr()


def _norm_title(title: str) -> str:
    import re
    return re.sub(r"\s+", " ", (title or "").strip().lower())[:80]


if __name__ == "__main__":
    main()
