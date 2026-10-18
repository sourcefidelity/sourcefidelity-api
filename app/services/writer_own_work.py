"""Parts of a cited statement that describe the writer's own work (`writer-own-work-v2`).

Owner direction 2026-10-18 (Synthetic Embrace review). A student comparing
their own invented film to a real one ("Semantically, it draws on the
dystopian sci-fi tradition exemplified by Blade Runner 2049 ... (Guynes,
2020)"), or an author setting their own findings against others', writes
parts the cited source cannot support: they are about the writer's work, not
the source. Judging them against the source reads as a failed search or a
red result. Such a part is shown as the writer's own and not judged; the
part the source can support is judged as before.

One DeepSeek call per cited statement, at Judgment time, sees only the
student's text (the statement, the sentences just before it) and the cited
reference; never the source. Only a part classified as the writer's own work
is skipped: "unsure" is judged as before. Built without a labelled
validation set (owner decision 2026-10-18); verification is post-prototype.
"""
from __future__ import annotations

import hashlib
import json

from app.services.llm_service import chat_completion_json

VERSION = "writer-own-work-v2"
OWN_WORK = "writer_own_work"
_ANSWERS = frozenset({OWN_WORK, "cited_source", "unsure"})
_MAX_CONTEXT_CHARACTERS = 1_200

SYSTEM_PROMPT = """You read one statement from a student's paper that cites a source, the
sentences just before it, and the cited reference. The statement is split
into numbered parts. For each part, say what it is about:

- "writer_own_work": the part's subject is the writer's OWN work: a film,
  story, design or project the writer made or invented, the writer's own
  study, data or findings, or what this paper itself does ("This reflection
  examines ...", "my film draws on ...", "it echoes ..." where "it" is the
  writer's work). This holds even when the part names another work that the
  writer's work draws on, echoes, builds on or resembles: the cited source
  cannot say anything about the writer's own work.
- "cited_source": the part's subject is another work, a theory, a fact, or
  what the cited source says (a real film's story, a scholar's argument).
- "unsure": the part mixes both, or you cannot tell.

Use the sentences before the statement to tell what "it", "the film" or
"this" refers to. Do not judge whether the source supports anything. Return
JSON only: {"parts": [{"id": "<part id>", "about": "writer_own_work" |
"cited_source" | "unsure"}]}."""


def _user_prompt(statement: str, before: str, reference: str, parts: list[tuple[str, str]]) -> str:
    return json.dumps({
        "sentences_before": before[-_MAX_CONTEXT_CHARACTERS:],
        "statement": statement,
        "cited_reference": reference,
        "parts": [{"id": part_id, "text": text} for part_id, text in parts],
    }, ensure_ascii=False)


def cache_key(route, user_prompt: str) -> str:
    parts = [VERSION, getattr(route, "arm_id", ""), getattr(route, "model", ""),
             hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), hashlib.sha256(user_prompt.encode()).hexdigest()]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def own_work_parts(statement: str, before: str, reference: str, parts: list[tuple[str, str]], *,
                   route, call=chat_completion_json, cache_lookup=None, cache_store=None) -> dict:
    """{"own_work": set of part ids, "cost_usd", "status"}; never raises.

    A failed or malformed answer skips nothing: every part is judged as before.
    """
    result = {"version": VERSION, "own_work": set(), "cost_usd": 0.0, "status": "not_run"}
    if route is None or not parts or not statement.strip():
        return result
    user_prompt = _user_prompt(statement, before, reference, parts)
    key = cache_key(route, user_prompt)
    stored = cache_lookup(key) if cache_lookup else None
    if stored and stored.get("version") == VERSION:
        return {**result, "own_work": set(stored.get("own_work") or []), "status": "cached"}
    receipt: dict = {}
    try:
        raw = call(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=400, max_retries=1,
                   route=route, receipt=receipt)
    except Exception:  # noqa: BLE001 - the check is optional; parts are judged as before
        return {**result, "status": "failed"}
    from app.services.judge_arms import call_cost_usd
    cost, _basis = call_cost_usd(receipt)
    result["cost_usd"] = cost or 0.0
    known = {part_id for part_id, _ in parts}
    answers = (raw or {}).get("parts") if isinstance(raw, dict) else None
    if not isinstance(answers, list):
        return {**result, "status": "invalid"}
    own = set()
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("about") not in _ANSWERS:
            return {**result, "status": "invalid"}
        if answer.get("about") == OWN_WORK and answer.get("id") in known:
            own.add(answer["id"])
    result.update(own_work=own, status="model")
    if cache_store:
        cache_store(key, {"version": VERSION, "status": "model", "own_work": sorted(own),
                          "original_cost_usd": result["cost_usd"]})
    return result
