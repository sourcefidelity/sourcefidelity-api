"""Add English translations of other-language evidence sentences to one report.

Run inside the api container:

    docker compose exec -T api python -m app.translate_evidence <report_id> --dry-run
    docker compose exec -T api python -m app.translate_evidence <report_id>

evidence-translation-v1 (owner decision 2026-10-13). Translates the sentences
in another writing system than the student's statement that the report shows,
with the configured model, and refreshes the report projection as a new report
version. No search is run and no Evidence Package changes. A dry run prints
counts only and makes no model call. Prints counts and ids only.
"""
from __future__ import annotations

import argparse
import sys

from app.database import SessionLocal
from app.services.evidence_translation_backfill import backfill_report_translations
from app.services.storage.backend import get_storage_backend


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_id")
    parser.add_argument("--dry-run", action="store_true", help="count only; no model call, no change")
    args = parser.parse_args(argv)
    session = SessionLocal()
    try:
        summary = backfill_report_translations(session, get_storage_backend(), args.report_id,
                                               dry_run=args.dry_run)
    finally:
        session.rollback()
        session.close()
    print(" ".join(f"{key}={value}" for key, value in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
