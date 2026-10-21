"""Judgment result items for a report, as the page script consumes them.

Shared by the results endpoint (polled by the live report) and the exported
interactive report, which carries the finished results inside the file
(owner request 2026-09-29) because it cannot contact the server.
"""
from __future__ import annotations

from app.services import judgment_runs as runs


def result_items(session, run, principal, *, after: int = 0, limit: int = 50) -> list[dict]:
    from app.models.report import VerificationReportRecord
    from app.services.judgment_report import display_state, judgment_result
    from app.services.judgment_reasons import quotation_only
    from app.services.judgment_reserve import load_reserve
    items: list[dict] = []
    if run is None:
        return items
    fake = bool((run.policy_snapshot or {}).get("fake_panel"))
    cache: dict = {}
    # English translations of other-language sentences (evidence-translation-v1):
    # those kept with the report, and those the run added for judge-cited sentences.
    from app.models.report import Report
    from app.services.evidence_translation import stored_translations
    report = session.get(Report, run.report_id)
    kept = stored_translations(report.report_json if report is not None else None)
    for row in runs.results_after(session, run, after, limit=limit):
        key = row.verification_report_id
        if key not in cache:
            record = session.get(VerificationReportRecord, key)
            in_scope = record is not None and (record.scope_type, record.scope_id) == (
                principal.scope_type, principal.scope_id)
            cache[key] = ((record.report_payload or {}) if in_scope else {},
                          load_reserve(session, key, principal.scope_type, principal.scope_id) if in_scope else None)
        payload, reserve = cache[key]
        stored = {"display_state": row.display_state, "reason_code": row.reason_code,
                  "candidate_id": row.candidate_id, "panel": row.panel, "wider_search": row.wider_search,
                  "coaching": row.coaching}
        translations = {**kept, **((row.panel or {}).get("evidence_translations") or {})}
        result = judgment_result(stored, payload, reserve, fake_panel=fake, translations=translations)
        items.append({
            "seq": row.seq, "citation_number": row.citation_index,
            "verification_report_id": str(row.verification_report_id), "candidate_id": row.candidate_id,
            "display_state": display_state(row.display_state, row.reason_code), "reason_code": row.reason_code,
            "labels": (row.panel or {}).get("labels") or {},
            "formal_status": (row.panel or {}).get("formal_status"),
            "wider_search": row.wider_search,
            "label": result["label"], "window_html": result["html"], "evidence": result["evidence"],
            "quotation_only": quotation_only(row.reason_code)})
    return items


def all_result_items(session, run, principal) -> list[dict]:
    """Every result of a run, in order."""
    items, after = [], 0
    while True:
        page = result_items(session, run, principal, after=after, limit=200)
        if not page:
            return items
        items.extend(page)
        after = page[-1]["seq"]
