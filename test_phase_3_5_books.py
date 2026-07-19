#!/usr/bin/env python3
"""Test script for Phase 3.5 hardening: N-up detection, completeness, splitting,
and retrieval coverage across all books and articles in test_data/.

Usage:
    cd <project root>
    source venv/bin/activate
    python test_phase_3_5_books.py              # books + articles (external lookups optional)
    python test_phase_3_5_books.py --books      # books only
    python test_phase_3_5_books.py --articles   # articles only
    python test_phase_3_5_books.py --no-external  # skip all external API calls

External API calls (OpenAlex, CORE, Google Books, Open Library) are attempted
only when the relevant API key / network is available; they degrade gracefully.
"""

import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ────────────────────────────────────────────────────────────


def hr(char="─", n=80):
    print(char * n)


def short(name: str, n=52) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def env(key: str) -> str | None:
    v = os.environ.get(key)
    return v if v else None


# ── Book tests ─────────────────────────────────────────────────────────


def test_books(external: bool = True):
    from app.services.page_layout import detect_n_up, classify_text_quality, PURE_SCAN
    from app.services.completeness_checker import check_completeness
    from app.services.chapter_splitter import is_edited_collection, split_into_chapters

    book_paths = sorted(glob.glob("test_data/Books/*.pdf"))
    hr()
    print(f"BOOKS: {len(book_paths)} files")
    hr()

    n_up_count = 0
    incomplete_count = 0
    edited_count = 0
    total_chapters = 0
    scan_counts = {"pure_scan": 0, "scan_ocr": 0, "digital": 0}

    for path in book_paths:
        name = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()

        # N-up
        layout = detect_n_up(data)
        if layout.is_n_up:
            n_up_count += 1

        # Text quality
        tq = classify_text_quality(data)
        scan_counts[tq.verdict] = scan_counts.get(tq.verdict, 0) + 1

        # Completeness (local signals always; external optional)
        report = check_completeness(data, external_lookup=external)
        if report.verdict == "INCOMPLETE":
            incomplete_count += 1

        # Edited collection + splitting
        edited = is_edited_collection(data)
        n_chapters = 0
        if edited:
            edited_count += 1
            chapters = split_into_chapters(data)
            n_chapters = len(chapters)
            total_chapters += n_chapters

        # Print a compact row
        flag = ""
        if report.verdict == "INCOMPLETE":
            flag += " ⚠INCOMPLETE"
        if layout.is_n_up:
            flag += f" {layout.pages_per_sheet}-up"
        if tq.verdict == PURE_SCAN:
            flag += " ⚠NEEDS-OCR"
        elif tq.verdict == "scan_ocr":
            flag += " [OCR]"
        lp = report.n_up_layout.logical_pages if report.n_up_layout else "?"
        print(f"  {short(name):50s} pp={lp:>4} {report.verdict:10s} ed={edited!s:5} ch={n_chapters}{flag}")

    hr()
    print(f"SUMMARY: {len(book_paths)} books | text-quality: {scan_counts['digital']} digital, {scan_counts['scan_ocr']} scan_ocr, {scan_counts['pure_scan']} pure_scan | {n_up_count} N-up | {incomplete_count} INCOMPLETE | {edited_count} edited | {total_chapters} chapters")
    hr()


# ── Article tests ──────────────────────────────────────────────────────


def test_articles(external: bool = True):
    from app.services.pdf_verifier import extract_metadata_from_pdf

    article_paths = sorted(glob.glob("test_data/Academic Articles/*.pdf"))
    # filter out non-pdf (e.g. .DS_Store)
    article_paths = [p for p in article_paths if p.lower().endswith(".pdf")]
    hr()
    print(f"ARTICLES: {len(article_paths)} files")
    hr()

    doi_found = 0
    title_found = 0
    openalex_hits = 0
    core_hits = 0
    openalex_attempted = 0
    core_attempted = 0

    from app.config import settings
    has_openalex_key = bool(settings.OPENALEX_API_KEY)
    has_core_key = bool(settings.CORE_API_KEY)

    if external and has_openalex_key:
        from app.services.retrieval.openalex import OpenAlexRetriever
        oa = OpenAlexRetriever()
    else:
        oa = None

    if external and has_core_key:
        from app.services.retrieval.core import CoreRetriever
        cr = CoreRetriever()
    else:
        cr = None

    for path in article_paths:
        name = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()

        try:
            meta = extract_metadata_from_pdf(data)
        except Exception as e:
            print(f"  {short(name):54s} EXTRACT ERROR: {e}")
            continue

        doi = meta.get("doi")
        title = (meta.get("title") or "")[:40]
        year = meta.get("year") or "?"
        if doi:
            doi_found += 1
        if meta.get("title"):
            title_found += 1

        oa_status = "-"
        cr_status = "-"
        if doi and oa is not None:
            openalex_attempted += 1
            r = oa.search_by_doi(doi)
            if r.success:
                openalex_hits += 1
                oa_status = "✓"
            else:
                oa_status = "✗"
        if doi and cr is not None:
            core_attempted += 1
            r = cr.search_by_doi(doi)
            if r.success:
                core_hits += 1
                cr_status = "✓"
            else:
                cr_status = "✗"

        print(f"  {short(name):54s} doi={doi or 'NONE':30s} yr={year} OA={oa_status} CORE={cr_status}")

    hr()
    print(f"SUMMARY: {len(article_paths)} articles | DOI extracted: {doi_found}/{len(article_paths)} | title: {title_found}/{len(article_paths)}")
    if oa is not None:
        print(f"  OpenAlex: {openalex_hits}/{openalex_attempted} hits (key present)")
    else:
        print(f"  OpenAlex: skipped (no OPENALEX_API_KEY)")
    if cr is not None:
        print(f"  CORE: {core_hits}/{core_attempted} hits (key present)")
    else:
        print(f"  CORE: skipped (no CORE_API_KEY)")
    hr()


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5 hardening test")
    parser.add_argument("--books", action="store_true", help="test books only")
    parser.add_argument("--articles", action="store_true", help="test articles only")
    parser.add_argument("--no-external", action="store_true", help="skip external API calls")
    args = parser.parse_args()

    run_books = args.books or not args.articles
    run_articles = args.articles or not args.books
    external = not args.no_external

    os.chdir(PROJECT_ROOT)

    try:
        if run_books:
            test_books(external=external)
        if run_articles:
            test_articles(external=external)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
