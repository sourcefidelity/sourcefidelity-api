"""The whole source for the evidence selector and Judgment across writing systems.

`cross-language-whole-source-v1` (owner decision 2026-10-13). When a claim's
source is written in another writing system than the student's statement, a
translated retrieval query can miss the passage that matters because the
source says it in other words. When the whole source, split into the existing
real sentences, fits the evidence selector's input budget (the existing
estimate and `MAX_INPUT_TOKENS`, statement and instructions included), the
selector numbers every sentence of the source in reading order rather than
only the retrieved candidates, and the Judgment foundation is drawn from the
same sentences within its existing per-claim cap. When it does not fit, the
retrieved candidates (translated channel included) are used as before and the
claim's evidence records that the whole source did not fit.

Retrieved passages keep their ids and offsets: a whole-source sentence with
the same absolute span as a retrieved sentence, or lying inside one, is not
offered twice. Whole-source passages are held in memory only (the artifact's
private `_whole_source`); the report keeps their page coordinates and text
hashes, never their text, in the claim's `cross_language` record, beside the
sentences the foundation itself keeps. Same-language sources are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

VERSION = "cross-language-whole-source-v1"
WHOLE_SOURCE_METHOD = "whole_source_cross_language"


def _page_order(page_index) -> int:
    return page_index if page_index is not None else -1


def absolute_key(passage, start: int, end: int) -> tuple:
    """A sentence's place in the source: page and absolute character span."""
    return (passage.page_index, passage.character_start + start, passage.character_start + end)


def _reading_order(key: tuple) -> tuple:
    return (_page_order(key[0]), key[1], key[2])


@dataclass(frozen=True)
class WholeSourceEntry:
    key: tuple
    passage: object          # SourcePassageEvidence (retrieved or whole-source)
    sentence: object         # SourceEvidenceSentence of that passage
    retrieved: bool


@dataclass(frozen=True)
class WholeSourcePlan:
    """Every sentence of the source, in reading order, deduplicated."""

    passages: tuple          # whole-source passages contributing a sentence
    entries: tuple           # WholeSourceEntry, reading order
    excluded: int            # retrieved passages left out by role, as the selector counts them

    def request_spans(self) -> list:
        return [(e.passage.passage_id, e.passage.page_index, e.passage.character_start, e.sentence.passage_start,
                 e.sentence.passage_end, e.sentence.text, e.passage.passage_role) for e in self.entries]


def _excluded_roles() -> frozenset:
    from app.services.source_roles import NON_EVIDENCE_ROLES, SELECTION_EXCLUDED_ROLES
    return NON_EVIDENCE_ROLES | SELECTION_EXCLUDED_ROLES


def whole_source_passages(source) -> list:
    """The source's body blocks as passages, as retrieval reads them, excluded roles left out."""
    from app.services.verification_evidence import (
        _PassageCandidate,
        _extract_pages,
        _passage_evidence,
        _source_blocks,
        _stable_id,
    )
    pages, _limitations = _extract_pages(source)
    excluded = _excluded_roles()
    passages = []
    for page, start, end, text, role in _source_blocks(pages):
        if role in excluded or not text.strip():
            continue
        passage = _passage_evidence(source, _PassageCandidate(
            page_index=page.index, page_label=page.label, start=start, end=end, text=text,
            method=WHOLE_SOURCE_METHOD, score=0.0, passage_role=role))
        # Its own id, so a whole-source passage never stands in for a retrieved one.
        passages.append(passage.model_copy(update={"passage_id": _stable_id(
            VERSION, source.content_sha256, str(page.index), str(start), str(end))}))
    return passages


def build_plan(artifact, whole_passages) -> WholeSourcePlan:
    """Retrieved sentences first (ids kept), then every other sentence of the source."""
    from app.services.facet_evidence_judgment import bounded_passage_sentences
    excluded_roles = _excluded_roles()
    items, seen, excluded = [], set(), 0
    retrieved = sorted(artifact.passages, key=lambda p: (_page_order(p.page_index), p.character_start))
    for passage in retrieved:
        if passage.passage_role in excluded_roles:
            excluded += 1
            continue
        sentences = bounded_passage_sentences(passage)
        if sentences is None:   # a sentence over the bound: skip this passage, never the run
            excluded += 1
            continue
        for sentence in sentences:
            key = absolute_key(passage, sentence.passage_start, sentence.passage_end)
            if key not in seen:
                seen.add(key)
                items.append(WholeSourceEntry(key, passage, sentence, True))
    for passage in whole_passages:
        if passage.passage_role in excluded_roles:
            continue
        for sentence in bounded_passage_sentences(passage) or ():
            key = absolute_key(passage, sentence.passage_start, sentence.passage_end)
            if key not in seen:
                seen.add(key)
                items.append(WholeSourceEntry(key, passage, sentence, False))
    # A whole-source sentence inside an earlier-starting sentence already kept
    # (a retrieved one, or an overlapping window's) is the same text again.
    items.sort(key=lambda e: (*_reading_order(e.key)[:2], -e.key[2], not e.retrieved))
    kept, reach = [], {}
    for entry in items:
        page = _page_order(entry.key[0])
        if not entry.retrieved and reach.get(page, -1) >= entry.key[2]:
            continue
        kept.append(entry)
        reach[page] = max(reach.get(page, -1), entry.key[2])
    kept.sort(key=lambda e: _reading_order(e.key))
    used = {e.passage.passage_id for e in kept if not e.retrieved}
    return WholeSourcePlan(passages=tuple(p for p in whole_passages if p.passage_id in used),
                           entries=tuple(kept), excluded=excluded)


def _descriptor(passage) -> dict:
    return {"passage_id": passage.passage_id, "page_index": passage.page_index, "page_label": passage.page_label,
            "character_start": passage.character_start, "character_end": passage.character_end,
            "passage_role": passage.passage_role,
            "text_sha256": hashlib.sha256(passage.text.encode("utf-8")).hexdigest()}


def _with_record(artifact, record: dict, plan=None):
    cross = copy.deepcopy(artifact.cross_language)
    cross["whole_source"] = record
    updated = artifact.model_copy(update={"cross_language": cross})
    updated._whole_source = plan
    return updated


def attach_whole_source(source, artifact, *, max_input_tokens: int | None = None):
    """Give the selector and the foundation the whole source when it fits.

    Only for a claim whose `cross_language` record says the writing systems
    differ; any other artifact is returned unchanged. Never fails the run: an
    unreadable source records why and keeps the retrieved candidates.
    """
    cross = artifact.cross_language
    if not cross or not cross.get("differs"):
        return artifact
    from app.services.evidence_sentence_selection import MAX_INPUT_TOKENS, SYSTEM_PROMPT, numbered_prompt
    from app.services.llm_input_boundary import estimate_prompt_tokens
    from app.services.verification_evidence import _source_matches_artifact
    budget = MAX_INPUT_TOKENS if max_input_tokens is None else max_input_tokens
    record = {"version": VERSION, "max_input_tokens": budget}
    try:
        if not _source_matches_artifact(source, artifact):
            return _with_record(artifact, {**record, "status": "not_available",
                                           "reason": "authorized_source_mismatch"})
        plan = build_plan(artifact, whole_source_passages(source))
    except Exception as exc:  # noqa: BLE001 - one optional record degrades, never the run
        logger.info("Whole source not prepared (type=%s)", type(exc).__name__)
        return _with_record(artifact, {**record, "status": "not_available", "reason": "source_not_split"})
    if not plan.entries:
        return _with_record(artifact, {**record, "status": "not_available", "reason": "no_source_sentence"})
    claim = artifact.claim
    _sentences, user_prompt = numbered_prompt(claim.text, claim.citation_marker or "", plan.request_spans())
    estimate = estimate_prompt_tokens(SYSTEM_PROMPT, user_prompt)
    record.update(sentence_count=len(plan.entries), token_estimate=estimate,
                  retrieved_sentence_count=sum(1 for e in plan.entries if e.retrieved))
    if estimate > budget:
        return _with_record(artifact, {**record, "status": "did_not_fit", "reason": "over_selector_input_budget"})
    record.update(status="applied", passages=[_descriptor(p) for p in plan.passages])
    return _with_record(artifact, record, plan)


def foundation_sentences(plan: WholeSourcePlan, artifact, sentence_groups, *, cap: int) -> tuple[list, dict]:
    """One candidate's foundation sentences from the whole source, within `cap`, in reading order.

    Everything when the source fits the cap. Otherwise the selector's chosen
    sentences come first, then the candidate's retrieved sentences in
    retrieval order, then the rest of the source in reading order, and the
    truncation is recorded.
    """
    by_key = {entry.key: entry.sentence for entry in plan.entries}
    passages = {p.passage_id: p for p in [*plan.passages, *artifact.passages]}
    chosen = []
    selection = artifact.sentence_evidence
    if selection.status == "selected":
        for item in selection.items:
            passage = passages.get(item.passage_id)
            if passage is None:
                continue
            key = absolute_key(passage, item.passage_start, item.passage_end)
            if key in by_key:
                chosen.append(key)
    retrieval_order, extra = [], {}
    for _rank, passage, sentences in sorted(sentence_groups, key=lambda group: group[0]):
        for sentence in sentences:
            key = absolute_key(passage, sentence.passage_start, sentence.passage_end)
            retrieval_order.append(key)
            if key not in by_key:
                extra.setdefault(key, sentence)
    every = {**by_key, **extra}
    if len(every) <= cap:
        keep = list(every)
    else:
        keep = list(dict.fromkeys([*chosen, *retrieval_order, *by_key]))[:cap]
    kept = set(keep)
    info = {"whole_source_sentences": len(every), "sentences": len(keep), "truncated": len(every) > len(keep),
            "omitted_sentences": len(every) - len(keep),
            "selector_sentences_omitted": sum(1 for key in dict.fromkeys(chosen) if key not in kept)}
    return [every[key] for key in sorted(keep, key=_reading_order)], info


def record_foundation(artifact, candidates: dict):
    """Add the foundation's per-candidate whole-source counts to the claim's record (data only)."""
    cross = copy.deepcopy(artifact.cross_language)
    cross["whole_source"]["foundation"] = candidates
    updated = artifact.model_copy(update={"cross_language": cross})
    return updated


def stored_passages(payload: dict) -> list:
    """A stored report's passage coordinates, with any whole-source passages (no text)."""
    whole = ((payload.get("cross_language") or {}).get("whole_source") or {}).get("passages") or []
    return [*(payload.get("passages") or []), *whole]
