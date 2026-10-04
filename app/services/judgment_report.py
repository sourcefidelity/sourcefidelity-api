"""Judgment result block for one claim in a citation window (ARCHITECTURE §7).

One layout (owner decisions 2026-09-28): each source part of a citation has one
evidence list, the sentences GLM selected, collapsed under "Evidence". The
judgment result is the note (the judge's reasoning is not shown); the label goes
into the part's heading. A supported statement also shows, outside the
collapsed list, the one sentence that most carried the judgment (owner decision
2026-09-30). The sentences the judge relied on are returned as keys
(page and exact position), and the page script appends any the list does not
already hold. Every model-, student- and source-derived string is escaped.
"""
from __future__ import annotations

import json
import re
from html import escape

from app.services.judgment_coaching import (
    COACHING_PROMPT_VERSION,
    SPLIT_NOTE_VERSION,
    TEMPLATE_NOTES,
    UNDECIDED_NOTE_VERSION,
    without_advice,
)
from app.services.text_quality import readable_text

# Result labels (owner-approved wording, 2026-09-28; "Not Supported" 2026-09-30).
LABELS = {
    "supported": "Supports",
    "qualified": "Qualified or Mixed",
    "contradicts": "Contradicts",
    "insufficient": "Not Supported",
    "undecided": "LLM Undecided",
    "not_judged": "Not Judged",
}
# Owner wording 2026-09-29 (hedged revision), the fixed note for an undecided judge.
# Owner decision 2026-09-30: no instruction sentence.
UNDECIDED_NOTE = TEMPLATE_NOTES["undecided"]
# Owner wording 2026-09-30, for a judge that could not resolve what the statement refers to.
UNDECIDED_WORDING_NOTE = "The model cannot decide if this statement is supported. It could not tell what the statement refers to."


def display_state(state: str | None, reason: str | None) -> str:
    """The shown state: an undecided judge is its own result (owner decision 2026-09-29)."""
    # The same results after a wider evidence search carry a "wider_search_"
    # prefix (paper 4 citation 8 showed Not Judged; 2026-10-02).
    if state == "not_judged" and str(reason or "").removeprefix("wider_search_") in {"judge_undecided", "samples_split"}:
        return "undecided"
    return state if state in LABELS else "not_judged"
JUDGED = frozenset({"supported", "qualified", "contradicts", "insufficient"})
_RETRY_REASONS = frozenset({"judge_failed", "wider_search_judge_failed", "judge_unavailable"})
# The judge names sentences by prompt alias (s1, s2, ...).
_SENTENCE_ALIAS = re.compile(r"\bs(\d{1,3})\b")
_REF = "\x00{}\x00"
_REF_TOKEN = re.compile(r"\x00([^\x00]+)\x00")
_BRACKETED_ONLY = re.compile(r"\((\x00[^\x00]+\x00(?:\s*(?:[-–,]|and)\s*\x00[^\x00]+\x00)*)\)")


def _sentences(payload: dict, reserve: dict | None) -> dict:
    """sentence id -> {key, text, page}; key = page and absolute position, as the selector's."""
    from app.services.evidence_report import evidence_sentence_key
    passages = {p.get("passage_id"): p for p in [*(payload.get("passages") or []),
                                                   *((reserve or {}).get("passages") or [])]}
    found = {}
    for source in (payload.get("facet_evidence_foundation") or {}, (reserve or {}).get("foundation") or {}):
        for s in source.get("source_sentences") or []:
            passage = passages.get(s.get("passage_id")) or {}
            index = passage.get("page_index")
            label = passage.get("page_label") or (index + 1 if index is not None else None)
            if passage.get("character_start") is not None:
                key = evidence_sentence_key(index, passage["character_start"] + s["passage_start"],
                                            passage["character_start"] + s["passage_end"])
            else:      # reserves built before 2026-09-28 lack the passage start
                key = f"{s.get('passage_id')}:{s['passage_start']}:{s['passage_end']}"
            found.setdefault(s["sentence_id"], {"key": key, "text": readable_text(s.get("text") or ""),
                                                "page": label})
    return found


def _facets(payload: dict, candidate_id: str) -> dict:
    for bundle in (payload.get("facet_evidence_foundation") or {}).get("candidate_bundles") or []:
        if bundle.get("candidate_id") == candidate_id:
            return {f["facet_id"]: f for f in bundle.get("facets") or []}
    return {}


def candidate_text(payload: dict, candidate_id: str) -> str:
    claim = (payload.get("claim") or {}).get("text") or ""
    for candidate in (payload.get("verification_candidates") or {}).get("candidates") or []:
        if candidate.get("candidate_id") == candidate_id:
            # Exact student wording, segment by segment (a candidate may be discontinuous).
            parts = [s.get("text") or "" for s in candidate.get("segments") or []]
            return " … ".join(p for p in parts if p) or candidate.get("text") or claim
    return claim


def candidate_paper_ranges(payload: dict, candidate_id: str) -> list[tuple[int, int]]:
    """The candidate's exact student wording as paper character ranges."""
    for candidate in (payload.get("verification_candidates") or {}).get("candidates") or []:
        if candidate.get("candidate_id") == candidate_id:
            return [(s["paper_start"], s["paper_end"]) for s in candidate.get("segments") or []
                    if isinstance(s.get("paper_start"), int) and isinstance(s.get("paper_end"), int)
                    and s["paper_end"] > s["paper_start"]]
    return []


def _arms(row: dict) -> list[dict]:
    panel = (row.get("wider_search") or {}).get("panel") or row.get("panel") or {}
    # With repeated answers, only those the shown result rests on (owner decision 2026-09-30).
    return [a for a in panel.get("arms") or [] if a.get("status") == "valid" and a.get("in_majority", True)]


def _reasoning(arm: dict, facets: dict) -> str:
    for mapping in arm.get("mappings") or []:
        if (facets.get(mapping.get("facet_id")) or {}).get("kind") == "candidate_as_written":
            return mapping.get("rationale") or ""
    return next((m.get("rationale") for m in arm.get("mappings") or [] if m.get("rationale")), "")


def _sentence_aliases(foundation: dict, candidate_id: str) -> dict:
    """Prompt alias -> sentence id, rebuilt exactly as prepare_candidate_prompts assigns it."""
    for bundle in foundation.get("candidate_bundles") or []:
        if bundle.get("candidate_id") == candidate_id:
            ids = dict.fromkeys([*(bundle.get("evidence_sentence_ids") or []),
                                 *(bundle.get("source_discourse_sentence_ids") or [])])
            return {f"s{index}": sid for index, sid in enumerate(ids, start=1)}
    return {}


def _cited(arms: list[dict], sentences: dict, reasoning: str, aliases: dict) -> tuple[list[dict], str]:
    """The sentences the judge relied on, and its reasoning with each alias as a key token.

    Sentences the reasoning names but did not bind as evidence are included, so
    every reference in the reasoning points at a listed sentence.
    """
    ids = [sid for sid in dict.fromkeys(sid for arm in arms for m in arm.get("mappings") or []
                                        for sid in m.get("evidence_sentence_ids") or []) if sid in sentences]
    for match in _SENTENCE_ALIAS.finditer(reasoning):
        sid = aliases.get(match.group(0))
        if sid in sentences and sid not in ids:
            ids.append(sid)

    def replace(match):
        sid = aliases.get(match.group(0))
        return _REF.format(sentences[sid]["key"]) if sid in ids else match.group(0)
    tokenized = _BRACKETED_ONLY.sub(r"\1", _SENTENCE_ALIAS.sub(replace, reasoning))
    return [sentences[sid] for sid in ids], tokenized


def _left_uncertain(arms: list[dict]) -> bool:
    """A judge left part of the statement uncertain, rather than failing to resolve what it refers to.

    Uncertainty is derived only from an uncertain facet reading, so a judge
    with none was undecided because the statement's reference was unresolved.
    """
    return any(m.get("direction") == "uncertain" for arm in arms for m in arm.get("mappings") or [])


def _key_sentence(arms: list[dict], facets: dict, sentences: dict) -> dict | None:
    """The one sentence that most carried a Supports judgment.

    Among the sentences the whole-statement reading cites, the one the most
    supporting readings cite; a tie goes to the whole-statement reading's first.
    """
    supporting = [m for arm in arms for m in arm.get("mappings") or [] if m.get("direction") == "supports"]
    whole_readings = [m for m in supporting if (facets.get(m.get("facet_id")) or {}).get("kind") == "candidate_as_written"]
    whole = [sid for m in whole_readings for sid in m.get("evidence_sentence_ids") or []]
    order = [sid for sid in dict.fromkeys(whole or [sid for m in supporting for sid in m.get("evidence_sentence_ids") or []])
             if sid in sentences]
    if not order:
        return None
    # A sentence cited only for whose claim it is (attribution or discourse
    # scope) does not outvote one that bears on the statement (citation 21,
    # 2026-10-02).
    voters = [m for m in supporting if (facets.get(m.get("facet_id")) or {}).get("kind")
              not in {"source_attribution", "inherited_discourse_scope"}] or supporting
    votes = {sid: sum(sid in (m.get("evidence_sentence_ids") or []) for m in voters) for sid in order}
    ranked = sorted(order, key=lambda sid: (-votes[sid], order.index(sid)))
    return dict(sentences[ranked[0]], alternatives=[sentences[sid] for sid in ranked])


def _with_references(tokenized: str) -> str:
    """Escape the reasoning; each key token becomes a reference the page script numbers."""
    parts, last = [], 0
    for match in _REF_TOKEN.finditer(tokenized):
        parts.append(escape(tokenized[last:match.start()]))
        parts.append(f'<span class="ev-ref" data-evidence-ref="{escape(match.group(1), quote=True)}"></span>')
        last = match.end()
    parts.append(escape(tokenized[last:]))
    return "".join(parts)


def judgment_result(row: dict, payload: dict, reserve: dict | None, *, fake_panel: bool = False) -> dict:
    """{state, label, html, evidence}: the block below the evidence list, and the sentences it cites."""
    state = display_state(row.get("display_state"), row.get("reason_code"))
    reason = row.get("reason_code") or ""
    parts = [f'<section class="judgment-window" data-state="{escape(state)}" data-reason="{escape(reason)}">']
    if fake_panel:   # development stand-in judges only
        parts.append('<p class="jw-standin">Stand-in judges: development answers, not judgments.</p>')
    evidence: list[dict] = []
    arms = _arms(row)
    undecided_uncertain = state == "undecided" and _left_uncertain(arms)
    if state in JUDGED or undecided_uncertain:
        candidate = row.get("candidate_id") or ""
        facets = _facets(payload, candidate)
        # The wider search judged the reserve's sentences, so its aliases come from there.
        wider = bool((row.get("wider_search") or {}).get("panel"))
        foundation = ((reserve or {}).get("foundation") if wider else payload.get("facet_evidence_foundation")) or {}
        sentences = _sentences(payload, reserve)
        evidence, _ = _cited(arms, sentences,
                             _reasoning(arms[0], facets) if arms else "",
                             _sentence_aliases(foundation, candidate))
        coaching = row.get("coaching") or {}
        # A fixed note is shown in its current approved wording, not as stored.
        note = TEMPLATE_NOTES.get(state) if coaching.get("status") == "template" else coaching.get("note")
        if (note and coaching.get("status") == "model"
                and coaching.get("version") not in {COACHING_PROMPT_VERSION, UNDECIDED_NOTE_VERSION}):
            # Notes written before v4 keep only their description (owner decision 2026-09-30).
            note = without_advice(note) or TEMPLATE_NOTES.get(state)
        if note and coaching.get("status") in {"model", "template"}:
            parts.append(f'<p class="jw-coaching">{escape(note)}</p>')
        elif state == "supported":
            # Notes are written only for the other results; Supports has a fixed note.
            parts.append(f'<p class="jw-coaching">{escape(TEMPLATE_NOTES["supported"])}</p>')
        elif state == "undecided":
            parts.append(f'<p class="jw-coaching">{escape(UNDECIDED_NOTE)}</p>')
        key = _key_sentence(arms, facets, sentences) if state == "supported" else None
        if key is not None:
            # Shown outside the collapsed list; the page script leaves it out of that list.
            parts.append(
                f'<ul class="evidence-sentences jw-key-evidence"><li data-key-evidence="{escape(key["key"], quote=True)}" '
                f'data-key-alternatives="{escape(json.dumps([{"key": a["key"], "text": a["text"], "page": a.get("page")} for a in key["alternatives"]]), quote=True)}">'
                + (f'<span class="ev-page">p. {escape(str(key["page"]))}</span> ' if key.get("page") else "")
                + f'<q>{escape(key["text"])}</q></li></ul>')
    elif state == "undecided":
        # Answers with no shared result show their note about the statement and the
        # source (2026-10-03), else the fixed note; otherwise the statement was unresolved.
        coaching = row.get("coaching") or {}
        if str(reason or "").removeprefix("wider_search_") != "samples_split":
            note = UNDECIDED_WORDING_NOTE
        elif coaching.get("status") == "model" and coaching.get("version") == SPLIT_NOTE_VERSION and coaching.get("note"):
            note = coaching["note"]
        else:
            note = UNDECIDED_NOTE
        parts.append(f'<p class="jw-coaching">{escape(note)}</p>')
    if reason in _RETRY_REASONS:
        parts.append('<p><button type="button" class="jw-retry" data-judgment-retry>Try Again</button></p>')
    parts.append("</section>")
    return {"state": state, "label": LABELS[state], "html": "".join(p for p in parts if p), "evidence": evidence}
