"""Judgment runs over one report (ARCHITECTURE §7).

A run starts when the paper is checked (or when Judgment is first opened on a
report checked earlier). It walks the report's citations in paper order. For
each cited source with a stored verification record it checks eligibility
(complete full text, verified identity), rebuilds the exact prompts from the
stored record, and judges each clause candidate with the configured judges
(one GLM judge answering three times since 2026-09-30). Results are written one
row per candidate in arrival order (`seq`), so the viewer can show them as they
finish. Nothing is written to the Evidence Package.

No provider is called unless every judge's policy gate passes (key, verified
terms date, price ceiling); otherwise the run ends `unavailable` with no call.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.job import Job
from app.models.judgment import (
    JudgmentArmResult,
    JudgmentCandidateResult,
    JudgmentRun,
)
from app.models.report import Report, VerificationReportRecord
from app.services.facet_evidence_judgment import FACET_JUDGMENT_VERSION, CandidateFacetFinding
from app.services.judge_arms import JudgePanelUnavailable, coaching_route, formal_judge_arms, policy_hash
from app.services.judgment_input import JudgmentInputUnavailable, judgment_eligibility, prepare_from_payload
from app.services.judgment_coaching import COACHED_STATES, coach
from app.services.judgment_report import candidate_text
from app.services.judgment_panel import PANEL_VERSION, ArmResult, CandidatePanelResult, judge_candidate

UNDECIDED_REASONS = frozenset({"judge_undecided", "judges_undecided"})
from app.services.llm_service import chat_completion_json

logger = logging.getLogger(__name__)

WHOLE_CITATION = "__citation__"
# The notice was removed (owner decision 2026-09-28); the column records that.
NO_NOTICE = "none"
_FAILURES_BEFORE_STOPPING = 3
ACTIVE = frozenset({"queued", "running"})
_PRE_CALL_REASONS = (
    ("antecedent", "antecedent_unresolved"),
    ("exceeded its configured budget", "prompt_over_budget"),
    ("sentences were unavailable", "evidence_unavailable"),
)


def _now():
    return datetime.now(timezone.utc)


def latest_run(session: Session, report_id: uuid.UUID, principal) -> JudgmentRun | None:
    return session.scalars(select(JudgmentRun).where(
        JudgmentRun.report_id == report_id,
        JudgmentRun.scope_type == principal.scope_type,
        JudgmentRun.scope_id == principal.scope_id,
    ).order_by(JudgmentRun.created_at.desc()).limit(1)).first()


def create_run(session: Session, report_id: uuid.UUID, principal, *, force_new: bool = False) -> JudgmentRun:
    """Reuse an active or completed run unless a new one is asked for (retry).

    `run.created_now` says whether this call created it, so only a new run is
    dispatched.
    """
    current = latest_run(session, report_id, principal)
    if current is not None and (current.status in ACTIVE or (current.status == "completed" and not force_new)):
        current.created_now = False
        return current
    run = JudgmentRun(report_id=report_id, scope_type=principal.scope_type, scope_id=principal.scope_id,
                      requested_by=principal.subject, notice_version=NO_NOTICE,
                      status="queued", panel_version=PANEL_VERSION, candidate_total=0,
                      candidate_done=0, spend_usd=0.0, unpriced_calls=0)
    session.add(run)
    session.flush()
    run.created_now = True
    return run


def eligible_members(view: dict) -> list[tuple[int | None, str]]:
    """(citation number, verification report id) in paper order, de-duplicated."""
    seen, members = set(), []
    for citation in view.get("citations") or []:
        for member in citation.get("members") or []:
            record_id = member.get("verification_report_id")
            if record_id and (citation.get("citation_number"), record_id) not in seen:
                seen.add((citation.get("citation_number"), record_id))
                members.append((citation.get("citation_number"), str(record_id)))
    return members


def _pre_call_reason(finding: CandidateFacetFinding) -> str:
    text = " ".join(finding.limitations or [])
    for needle, reason in _PRE_CALL_REASONS:
        if needle in text:
            return reason
    return "not_assessed_before_judging"


def _panel_json(result: CandidatePanelResult) -> dict:
    """Labels, agreement and each arm's reading; ids and bounded rationales only."""
    arms = []
    for arm in result.arms:
        item = {"arm_id": arm.arm_id, "model": arm.model, "status": arm.status, "label": arm.label,
                "failure": arm.failure, "cached": arm.cached, "cost_usd": arm.cost_usd,
                "cost_basis": arm.cost_basis, "in_majority": arm.in_majority}
        if arm.finding is not None:
            item.update(derived_outcome=arm.finding.derived_outcome,
                        evidence_coverage=arm.finding.evidence_coverage,
                        context_resolution=arm.finding.context_resolution,
                        locator_status=arm.finding.locator_status,
                        mappings=[m.model_dump(mode="json") for m in arm.finding.mappings])
        arms.append(item)
    assessment = result.assessment
    return {"panel_version": PANEL_VERSION, "contract_version": FACET_JUDGMENT_VERSION,
            "prompt_sha256": result.prompt_sha256, "arms": arms,
            "formal_status": assessment.formal_status if assessment else None,
            "formal_max_distance": assessment.formal_max_distance if assessment else None,
            "labels": {a.arm_id: a.label for a in result.arms if a.label}}


def _spend_limit_reached(session: Session, run: JudgmentRun) -> str | None:
    per_report = settings.JUDGMENT_MAX_USD_PER_REPORT
    if per_report is not None and run.spend_usd >= per_report:
        return "report_spend_limit"
    per_day = settings.JUDGMENT_MAX_USD_PER_DAY
    if per_day is not None:
        spent = session.scalar(select(func.coalesce(func.sum(JudgmentRun.spend_usd), 0.0)).where(
            JudgmentRun.scope_type == run.scope_type, JudgmentRun.scope_id == run.scope_id,
            JudgmentRun.created_at >= _now() - timedelta(days=1)))
        if spent >= per_day:
            return "daily_spend_limit"
    return None


class _ScopedCache:
    def __init__(self, session: Session, run: JudgmentRun):
        self.session, self.run = session, run

    def coaching_lookup(self, key: str) -> dict | None:
        from app.models.judgment import JudgmentCoachingNote
        row = self.session.scalars(select(JudgmentCoachingNote).where(
            JudgmentCoachingNote.scope_type == self.run.scope_type,
            JudgmentCoachingNote.scope_id == self.run.scope_id,
            JudgmentCoachingNote.cache_key == key)).first()
        return row.payload if row is not None else None

    def coaching_store(self, key: str, payload: dict) -> None:
        from app.models.judgment import JudgmentCoachingNote
        try:
            with self.session.begin_nested():
                self.session.add(JudgmentCoachingNote(
                    scope_type=self.run.scope_type, scope_id=self.run.scope_id, cache_key=key,
                    prompt_version=payload.get("version", ""), payload=payload))
        except IntegrityError:
            pass

    def lookup(self, key: str) -> dict | None:
        row = self.session.scalars(select(JudgmentArmResult).where(
            JudgmentArmResult.scope_type == self.run.scope_type,
            JudgmentArmResult.scope_id == self.run.scope_id,
            JudgmentArmResult.cache_key == key)).first()
        return row.response if row is not None else None

    def store(self, arm: ArmResult) -> None:
        receipt = arm.receipt or {}
        row = JudgmentArmResult(
            scope_type=self.run.scope_type, scope_id=self.run.scope_id, cache_key=arm.cache_key,
            arm_id=arm.arm_id, model=arm.model, returned_model=receipt.get("returned_model"),
            returned_provider=receipt.get("returned_provider"), contract_version=FACET_JUDGMENT_VERSION,
            prompt_sha256=receipt.get("prompt_sha256", ""), response=arm.response or {},
            usage={k: receipt.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                                                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                                "attempts") if receipt.get(k) is not None},
            cost_usd=arm.cost_usd, cost_basis=arm.cost_basis)
        try:
            with self.session.begin_nested():
                self.session.add(row)
        except IntegrityError:      # the same prompt judged concurrently; keep the first
            pass

    def ids(self, keys: list[str]) -> list[str]:
        rows = self.session.scalars(select(JudgmentArmResult.id).where(
            JudgmentArmResult.scope_type == self.run.scope_type,
            JudgmentArmResult.scope_id == self.run.scope_id,
            JudgmentArmResult.cache_key.in_(keys)))
        return [str(r) for r in rows]


def execute_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    routes_factory: Callable = formal_judge_arms,
    call=chat_completion_json,
    wider_search_factory: Callable | None = None,
    coaching_route_factory: Callable | None = None,
) -> JudgmentRun:
    """Judge every eligible candidate of one run; safe to call twice.

    `wider_search_factory(session, verification_report_id, run)` returns the
    wider-search step for one cited source (default: its stored reserve).
    """
    if wider_search_factory is None:
        from app.services.judgment_reserve import make_wider_search as wider_search_factory
    run = session.scalars(select(JudgmentRun).where(JudgmentRun.id == run_id).with_for_update()).first()
    if run is None or run.status != "queued":
        return run
    run.status, run.updated_at = "running", _now()
    session.commit()

    def finish(status: str, reason: str | None = None) -> JudgmentRun:
        run.status, run.reason_code, run.updated_at = status, reason, _now()
        session.commit()
        return run

    try:
        routes, snapshot = routes_factory()
    except JudgePanelUnavailable as exc:
        return finish("unavailable", f"panel_unavailable:{exc.arm_id}")
    run.policy_snapshot, run.policy_hash = snapshot, policy_hash(snapshot)

    report = session.get(Report, run.report_id)
    job = session.get(Job, report.job_id) if report is not None else None
    if job is None or (job.scope_type, job.scope_id) != (run.scope_type, run.scope_id):
        return finish("failed", "report_not_in_scope")
    view = (report.report_json or {}).get("evidence_report") or {}
    cache = _ScopedCache(session, run)
    seq = 0
    # DeepSeek writes the notes whichever model judges (owner decision 2026-09-27).
    note_route = next((r for r in routes if r.arm_id == "deepseek"), None)
    if note_route is None:
        note_route = (coaching_route_factory or coaching_route)()
    # Stop calling judges that keep failing: each failure costs a full timeout.
    consecutive_failures = 0

    def write(citation_number, record_id, candidate_id, state, reason, panel=None, wider=None,
              arm_ids=None, spend=0.0, coaching=None):
        nonlocal seq
        seq += 1
        session.add(JudgmentCandidateResult(
            run_id=run.id, seq=seq, citation_index=citation_number,
            verification_report_id=uuid.UUID(record_id), candidate_id=candidate_id,
            display_state=state, reason_code=reason, panel=panel or {}, wider_search=wider,
            arm_result_ids=arm_ids or [], spend_usd=spend, coaching=coaching))
        run.candidate_done = seq
        run.updated_at = _now()
        session.commit()

    for citation_number, record_id in eligible_members(view):
        record = session.get(VerificationReportRecord, uuid.UUID(record_id))
        if (record is None or (record.scope_type, record.scope_id) != (run.scope_type, run.scope_id)
                or record.paper_version_id != view.get("paper_version_id")):
            continue
        payload = record.report_payload or {}
        eligibility = judgment_eligibility(payload)
        if not eligibility.eligible:
            write(citation_number, record_id, WHOLE_CITATION, "not_judged", eligibility.reason_code)
            continue
        try:
            context, prepared = prepare_from_payload(payload)
        except JudgmentInputUnavailable as exc:
            write(citation_number, record_id, WHOLE_CITATION, "not_judged", exc.reason_code)
            continue
        run.candidate_total += sum(1 for _ in prepared.items)
        wider_search = wider_search_factory(session, record_id, run)
        for item in prepared.items:
            if isinstance(item, CandidateFacetFinding):
                write(citation_number, record_id, item.candidate_id, "not_judged", _pre_call_reason(item))
                continue
            limit = _spend_limit_reached(session, run)
            if limit:
                return finish("completed", limit)
            if consecutive_failures >= _FAILURES_BEFORE_STOPPING:
                write(citation_number, record_id, item.bundle.candidate_id, "not_judged", "judge_unavailable")
                continue
            result = judge_candidate(item, context, prepared.sentences, routes, run.policy_hash,
                                     cache_lookup=cache.lookup, cache_store=cache.store, call=call)
            spend = result.spend_usd
            consecutive_failures = consecutive_failures + 1 if result.reason_code == "judge_failed" else 0
            wider = None
            if result.display_state == "insufficient":
                # Red only after a wider search of new sentences agrees again.
                outcome = wider_search(context, item, routes, run, cache, call) if wider_search else None
                if outcome is None:
                    wider = {"status": "unavailable"}
                    result.display_state, result.reason_code = "not_judged", "wider_search_unavailable"
                else:
                    wider = outcome["record"]
                    spend += outcome.get("spend_usd", 0.0)
                    result.display_state, result.reason_code = outcome["display_state"], outcome["reason_code"]
            coaching = None
            note_state = result.display_state
            if (result.display_state == "not_judged" and result.reason_code in UNDECIDED_REASONS
                    and any(a.finding is not None and a.finding.status == "uncertain"
                            for a in result.arms if a.status == "valid" and a.in_majority)):
                # A judge left part of the statement uncertain: the note says why
                # (owner decision 2026-09-30). An unresolved statement keeps its fixed note.
                note_state = "undecided"
            if note_state in COACHED_STATES or note_state == "undecided":
                wider_record = wider or {}
                sentences = {sid: sentence.text for sid, sentence in prepared.sentences.items()}
                sentences.update((wider_record.get("sentences") or {}))
                facets = {f.facet_id: {"kind": f.kind, "text": f.text,
                                       "material_to_aggregate": f.material_to_aggregate}
                          for f in item.bundle.facets}
                coaching = coach(note_state, candidate_text(payload, item.bundle.candidate_id),
                                 context.claim.citation_marker or "", facets,
                                 wider_record.get("panel") or _panel_json(result), sentences,
                                 route=note_route, call=call, cache_lookup=cache.coaching_lookup,
                                 cache_store=cache.coaching_store)
                spend += coaching.get("cost_usd") or 0.0
                coaching = {k: v for k, v in coaching.items() if k != "cache_key"}
            run.spend_usd += spend
            run.unpriced_calls += sum(1 for a in result.arms if not a.cached and a.cost_basis == "unpriced"
                                      and a.receipt)
            write(citation_number, record_id, item.bundle.candidate_id, result.display_state,
                  result.reason_code, panel=_panel_json(result),
                  wider={k: v for k, v in (wider or {}).items() if k != "sentences"} or None,
                  arm_ids=cache.ids([a.cache_key for a in result.arms if a.status == "valid"]),
                  spend=spend, coaching=coaching)
    return finish("completed")


def results_after(session: Session, run: JudgmentRun, after_seq: int, limit: int = 50) -> list[JudgmentCandidateResult]:
    return list(session.scalars(select(JudgmentCandidateResult).where(
        JudgmentCandidateResult.run_id == run.id, JudgmentCandidateResult.seq > after_seq,
    ).order_by(JudgmentCandidateResult.seq).limit(limit)))


def schedule_run_at_check(session: Session, report, job) -> JudgmentRun | None:
    """Judge a checked paper straight away (owner decision 2026-09-27).

    Never raises: a Judgment problem must not fail the paper check.
    """
    try:
        from app.tasks.judgment import dispatch_report_judgment
        run = JudgmentRun(report_id=report.id, scope_type=job.scope_type, scope_id=job.scope_id,
                          requested_by="paper_check", notice_version=NO_NOTICE, status="queued",
                          panel_version=PANEL_VERSION, candidate_total=0, candidate_done=0,
                          spend_usd=0.0, unpriced_calls=0)
        session.add(run)
        session.commit()
        dispatch_report_judgment(run.id)
        return run
    except Exception as exc:
        session.rollback()
        logger.warning("Judgment was not scheduled at check time (type=%s)", type(exc).__name__)
        return None
