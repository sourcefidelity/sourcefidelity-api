"""List admitted search-found PDFs that no longer reach high identity.

Read-only. Run inside the api container:

    docker compose exec -T api python -m app.identity_recheck
    docker compose exec -T api python -m app.identity_recheck --provenance web_search --provenance landing_page

Re-runs the current identity rule (opening-byline-identity-v1) over accepted
representations of the given provenance (default: web_search) and prints their
ids and content hashes only. Changes nothing and makes no network request
beyond the configured database and object store.
"""
from __future__ import annotations

import argparse
import sys

from app.database import SessionLocal
from app.services.identity_recheck import DEFAULT_PROVENANCE, recheck_admitted_representations
from app.services.storage.backend import get_storage_backend


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provenance", action="append",
                        help="representation provenance to check (repeatable; default web_search)")
    args = parser.parse_args(argv)
    session = SessionLocal()
    try:
        summary = recheck_admitted_representations(
            session, get_storage_backend(),
            provenance=tuple(args.provenance or DEFAULT_PROVENANCE))
    finally:
        session.rollback()
        session.close()
    for item in summary["flagged"]:
        print(f"{item['representation_id']} work={item['canonical_work_id']} "
              f"sha256={item['content_sha256']} identity={item['identity_confidence']} "
              f"author_conflict={'yes' if item['author_conflict'] else 'no'}")
    print(f"{summary['checked']} PDFs checked, {len(summary['flagged'])} below high identity, "
          f"{summary['skipped_not_pdf']} non-PDF skipped, "
          f"{summary['object_unavailable']} objects unavailable ({summary['policy']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
