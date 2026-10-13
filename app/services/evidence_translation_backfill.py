"""Add English translations to one existing report (evidence-translation-v1).

For a report checked before translations existed: translates the
other-language sentences its windows show (the selected evidence sentences,
abstracts, the PDF's extracts and the sentences its latest Judgment run cites),
then refreshes the report projection with the translations kept beside it.
No search is run and no Evidence Package changes. Bounded like a paper check:
`MAX_SENTENCES_PER_MEMBER` per cited source, `MAX_REPORT_SENTENCES` per report.
A dry run counts only and calls nothing.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.services.evidence_translation import (
    MAX_REPORT_SENTENCES,
    MAX_SENTENCES_PER_MEMBER,
    BATCH_SIZE,
    VERSION,
    call_cost,
    is_other_language,
    report_sentences,
    sentence_key,
    statement_script,
    stored_translations,
    translate_sentences,
)


def _latest_report(session, report_id):
    from app.models.report import Report
    report = session.get(Report, report_id if isinstance(report_id, uuid.UUID) else uuid.UUID(str(report_id)))
    if report is None:
        raise ValueError("Report not found")
    return session.scalar(select(Report).where(Report.job_id == report.job_id)
                          .order_by(Report.report_version.desc(), Report.created_at.desc()).limit(1))


def judged_sentences(session, report) -> list[list[str]]:
    """Per cited source, the other-language sentences the latest Judgment run of the report's lineage cites."""
    from app.models.judgment import JudgmentCandidateResult, JudgmentRun
    from app.models.report import Report, VerificationReportRecord
    from app.services.judgment_report import judgment_result
    from app.services.judgment_reserve import load_reserve
    lineage = select(Report.id).where(Report.job_id == report.job_id)
    run = session.scalar(select(JudgmentRun).where(JudgmentRun.report_id.in_(lineage))
                         .order_by(JudgmentRun.created_at.desc()).limit(1))
    if run is None:
        return []
    groups: dict[uuid.UUID, dict[str, str]] = {}
    cache: dict = {}
    for row in session.scalars(select(JudgmentCandidateResult).where(JudgmentCandidateResult.run_id == run.id)
                               .order_by(JudgmentCandidateResult.seq)):
        key = row.verification_report_id
        if key not in cache:
            record = session.get(VerificationReportRecord, key)
            in_scope = record is not None and (record.scope_type, record.scope_id) == (run.scope_type, run.scope_id)
            cache[key] = ((record.report_payload or {}) if in_scope else {},
                          load_reserve(session, key, run.scope_type, run.scope_id) if in_scope else None)
        payload, reserve = cache[key]
        if not payload:
            continue
        result = judgment_result({"display_state": row.display_state, "reason_code": row.reason_code,
                                  "candidate_id": row.candidate_id, "panel": row.panel,
                                  "wider_search": row.wider_search, "coaching": row.coaching}, payload, reserve)
        script = statement_script((payload.get("claim") or {}).get("text") or "")
        chosen = groups.setdefault(key, {})
        for sentence in result["evidence"]:
            text = sentence.get("text") or ""
            if len(chosen) < MAX_SENTENCES_PER_MEMBER and is_other_language(text, script):
                chosen.setdefault(sentence_key(text), text)
    return [list(texts.values()) for texts in groups.values() if texts]


def backfill_report_translations(session, backend, report_id, *, dry_run: bool = False,
                                 completion=None) -> dict:
    """Translate one report's displayed other-language sentences and refresh its projection."""
    report = _latest_report(session, report_id)
    view = (report.report_json or {}).get("evidence_report") or {}
    known = stored_translations(report.report_json)
    groups = report_sentences(view) + judged_sentences(session, report)
    keys = {sentence_key(text) for texts in groups for text in texts}
    missing = [[text for text in texts if sentence_key(text) not in known] for texts in groups]
    counts = {"report_id": str(report.id), "cited_sources": len(groups), "sentences": len(keys),
              "already_translated": len(keys & set(known)),
              "to_translate": len({sentence_key(t) for texts in missing for t in texts}),
              "estimated_calls": sum(-(-len(texts) // BATCH_SIZE) for texts in missing if texts)}
    if dry_run or not counts["to_translate"]:
        return {**counts, "dry_run": dry_run, "refreshed": False}
    calls: list[dict] = []
    budget = MAX_REPORT_SENTENCES
    for texts in missing:
        if budget <= 0 or not texts:
            continue
        new, made = translate_sentences(texts, known=known, completion=completion,
                                        limit=min(MAX_SENTENCES_PER_MEMBER, budget))
        budget -= sum(c.get("sentence_count", 0) for c in made)
        known.update(new)
        calls += made
    previous = (report.report_json or {}).get("evidence_translations") or {}
    store = {"version": VERSION, "sentences": known, "source": "backfill",
             "calls": [*(previous.get("calls") or []), *calls][-60:],
             "cost_usd": float(previous.get("cost_usd") or 0.0) + call_cost(calls)}
    from app.services.evidence_report_refresh import refresh_evidence_report_projection
    refreshed = refresh_evidence_report_projection(session, backend, report_id=report.id,
                                                   evidence_translations=store)
    return {**counts, "dry_run": False, "refreshed": True, "translated": sum(int(c.get("translated_count") or 0) for c in calls),
            "calls": len(calls), "failed_calls": sum(c.get("call") in {"failed", "invalid_answer"} for c in calls),
            "cost_usd": call_cost(calls), "new_report_id": refreshed["report_id"],
            "new_report_version": refreshed["report_version"]}
