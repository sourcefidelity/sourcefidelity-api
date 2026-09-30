"""Sentence reserve and wider search for the Judgment layout (owner decisions 12-13).

The judges read at most 24 selected sentences, so an agreed "no evidence" may
only mean the selection missed it. Before "insufficient evidence" (red) is
shown, the judges read up to 24 NEW sentences from the same source and must
agree on "no evidence" again.

The new sentences are prepared at paper-run time, while the source file is
still available, by the same deterministic machinery as the first selection:
candidate retrieval with a wider passage limit, excluding every passage the
first selection used, then the unchanged facet foundation builder. That keeps
the sentences' voice and attribution annotations identical in kind, so the
wider prompt is an ordinary prompt of the same contract. No model is used to
build the reserve and building it can never fail a paper run.

Retention: kept with the verification report (deleted with it) under
`JUDGMENT_RESERVE_RETENTION`; `off` builds none. An "until_grades_released"
reserve is purged when marks are released for the paper's assessment
(assessment_marks.py; set manually now, by the planned LMS signal later) and
is not stored for a paper whose assessment already has marks released. A paper
without an assessment keeps it with the verification report.
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

from sqlalchemy import select

from app.config import settings
from app.services.facet_evidence_judgment import (
    CandidateFacetFinding,
    attach_facet_evidence_foundation,
    prepare_candidate_prompts,
)
from app.services.verification_evidence import (
    FacetEvidenceFoundation,
    attach_candidate_passage_retrieval,
)

logger = logging.getLogger(__name__)

RESERVE_VERSION = "judgment-reserve-v1"


def reserve_enabled() -> bool:
    return settings.JUDGMENT_RESERVE_RETENTION != "off"


def build_judgment_reserve(source, artifact) -> dict | None:
    """New sentences per candidate from passages the first selection did not use.

    Complete full texts only. Returns None when nothing new exists or on any
    error: an unexpected input degrades this one optional record, never the run.
    """
    try:
        level = getattr(artifact.coverage, "level", None)
        if str(getattr(level, "value", level) or "") != "full_text":
            return None
        original = artifact.facet_evidence_foundation
        retrieval = artifact.candidate_passage_retrieval
        if original.status not in {"complete", "incomplete"} or retrieval.status not in {"complete", "incomplete"}:
            return None
        used = {sel.candidate_id: {p.passage_id for p in sel.passages} for sel in retrieval.selections}
        known_text = {" ".join(s.text.split()) for s in original.source_sentences}
        # The same ranking, skipping every passage the first selection used.
        wide = attach_candidate_passage_retrieval(source, artifact, excluded_passage_ids_by_candidate=used)
        foundation = attach_facet_evidence_foundation(wide).facet_evidence_foundation
        if foundation.status not in {"complete", "incomplete"}:
            return None
        by_id = {s.sentence_id: s for s in foundation.source_sentences}
        bundles, any_new = [], False
        for bundle in foundation.candidate_bundles:
            fresh = [sid for sid in bundle.evidence_sentence_ids
                     if sid in by_id and " ".join(by_id[sid].text.split()) not in known_text]
            any_new = any_new or bool(fresh)
            bundles.append(bundle.model_copy(update={"evidence_sentence_ids": fresh}))
        if not any_new:
            return None
        kept_ids = {sid for b in bundles for sid in [*b.evidence_sentence_ids, *b.source_discourse_sentence_ids]}
        reserve = foundation.model_copy(update={
            "candidate_bundles": bundles,
            "source_sentences": [s for s in foundation.source_sentences if s.sentence_id in kept_ids],
        })
        passage_ids = {s.passage_id for s in reserve.source_sentences}
        return {
            "reserve_version": RESERVE_VERSION,
            "retention": settings.JUDGMENT_RESERVE_RETENTION,
            "foundation": reserve.model_dump(mode="json"),
            "passages": [{"passage_id": p.passage_id, "page_index": p.page_index, "page_label": p.page_label,
                          "character_start": p.character_start}
                         for p in wide.passages if p.passage_id in passage_ids],
        }
    except Exception as exc:        # optional record: degrade, never fail the run
        logger.warning("Judgment reserve not built (type=%s)", type(exc).__name__)
        return None


def store_reserve(session, record, reserve: dict) -> None:
    from app.models.judgment import JudgmentSourceReserve

    existing = session.scalar(select(JudgmentSourceReserve).where(
        JudgmentSourceReserve.verification_report_id == record.id))
    if existing is not None:
        return
    if reserve.get("retention") == "until_grades_released":
        from app.services.assessment_marks import paper_version_marks_released
        if paper_version_marks_released(session, paper_version_id=record.paper_version_id,
                                        scope_type=record.scope_type, scope_id=record.scope_id):
            return
    session.add(JudgmentSourceReserve(
        verification_report_id=record.id, scope_type=record.scope_type, scope_id=record.scope_id,
        reserve_version=reserve["reserve_version"], retention_policy=reserve["retention"], payload=reserve))
    session.flush()


def load_reserve(session, verification_report_id, scope_type: str, scope_id: str) -> dict | None:
    from app.models.judgment import JudgmentSourceReserve

    row = session.scalar(select(JudgmentSourceReserve).where(
        JudgmentSourceReserve.verification_report_id == uuid.UUID(str(verification_report_id)),
        JudgmentSourceReserve.scope_type == scope_type, JudgmentSourceReserve.scope_id == scope_id))
    return row.payload if row is not None and row.reserve_version == RESERVE_VERSION else None


def make_wider_search(session, verification_report_id, run):
    """The wider-search step `execute_run` calls for one agreed "no evidence"."""
    from app.services.judgment_panel import judge_candidate

    reserve = load_reserve(session, verification_report_id, run.scope_type, run.scope_id)

    def wider(context, item, routes, _run, cache, call):
        if reserve is None:
            return None
        foundation = FacetEvidenceFoundation.model_validate(reserve["foundation"])
        wide_context = SimpleNamespace(**{**vars(context), "facet_evidence_foundation": foundation,
                                          "passages": [*context.passages, *[SimpleNamespace(**p)
                                                                            for p in reserve["passages"]]]})
        prepared = prepare_candidate_prompts(wide_context, max_input_tokens=settings.JUDGMENT_MAX_INPUT_TOKENS)
        match = next((i for i in prepared.items
                      if getattr(i, "candidate_id", None) == item.bundle.candidate_id
                      or getattr(getattr(i, "bundle", None), "candidate_id", None) == item.bundle.candidate_id),
                     None)
        if match is None or isinstance(match, CandidateFacetFinding) or not match.bundle.evidence_sentence_ids:
            return None
        result = judge_candidate(match, wide_context, prepared.sentences, routes, run.policy_hash,
                                 cache_lookup=cache.lookup, cache_store=cache.store, call=call)
        from app.services.judgment_runs import _panel_json
        record = {"status": "completed", "new_sentences": len(match.bundle.evidence_sentence_ids),
                  "panel": _panel_json(result),
                  # For the coaching note only; not stored with the result.
                  "sentences": {sid: sentence.text for sid, sentence in prepared.sentences.items()}}
        if result.display_state == "insufficient":
            state, reason = "insufficient", "agreed_no_evidence_after_wider_search"
        elif result.display_state == "not_judged":
            state, reason = "not_judged", f"wider_search_{result.reason_code}"
        else:
            state, reason = result.display_state, "found_in_wider_search"
        return {"display_state": state, "reason_code": reason, "record": record, "spend_usd": result.spend_usd}

    return wider
