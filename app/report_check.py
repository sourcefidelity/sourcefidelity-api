"""Check the latest rendered report of each paper (owner request 2026-10-07).

Run inside the api container, with the owner-confirmed expectations (kept out
of the repository) on standard input:

    docker compose exec -T api python -m app.report_check < test_data/report_expectations.json
    docker compose exec -T api python -m app.report_check --paper "Franchise 2" < /dev/null

Expectations file: {"<paper filename>": [{"window": "reference-entry-panel-3",
"contains": "...", "note": "..."}, {"window": "", "absent": "..."}]}.
Prints violations and exits 1 when any check fails. Makes no network request
beyond the local report server and changes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request

import fitz
from sqlalchemy import select

from app.database import SessionLocal
from app.models.job import Job
from app.models.report import Report
from app.services.evidence_report import load_authorized_evidence_report_bundle, project_reference_flags
from app.services.report_check import check_expectations, check_report
from app.services.storage.backend import get_storage_backend


def _latest_reports(session, paper: str | None):
    seen = set()
    for job in session.scalars(select(Job).where(Job.status == "completed").order_by(Job.created_at.desc())):
        if job.filename in seen or (paper and paper.casefold() not in job.filename.casefold()):
            continue
        report = session.scalar(select(Report).where(Report.job_id == job.id)
                                .order_by(Report.report_version.desc(), Report.created_at.desc()).limit(1))
        if report is None:
            continue
        seen.add(job.filename)
        yield job, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", help="only papers whose filename contains this text")
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args(argv)
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    expectations = json.loads(raw) if raw.strip() else {}
    session = SessionLocal()
    backend = get_storage_backend()
    failures = checked = 0
    for job, report in _latest_reports(session, args.paper):
        try:
            view, _artifact, paper = load_authorized_evidence_report_bundle(
                session, backend, report_id=str(report.id), scope_type=job.scope_type, scope_id=job.scope_id)
        except Exception:  # noqa: BLE001 - reports outside the local scope are not checked
            continue
        with fitz.open(stream=paper, filetype="pdf") as document:
            projected = project_reference_flags(view, document, hashlib.sha256(paper).hexdigest())
        with urllib.request.urlopen(f"{args.base}/report/{report.id}", timeout=120) as response:
            html = response.read().decode("utf-8")
        violations = check_report(html, projected) + check_expectations(html, expectations.get(job.filename) or [])
        checked += 1
        status = "FAIL" if violations else "ok"
        print(f"{status:4} {job.filename} (report {str(report.id)[:8]})")
        for v in violations:
            print(f"     {v['check']:24} {v['window'][:28]:28} {v['detail']}")
        failures += bool(violations)
    print(f"{checked} reports checked, {failures} with violations")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
