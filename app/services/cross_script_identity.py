"""Whether a linked page in another writing system names the cited work.

Owner decision 2026-10-06 (Xu & Luo): a reference cited in English to a page
titled in another script cannot be compared by spelling, and was flagged as
"Cannot be verified" although the student's link named the work. The
configured model (DeepSeek) compares the cited reference with the page's own
title, date and opening text and answers in a fixed schema. Its answer is one
field comparison among the existing ones: 'yes' agrees on the title, 'no'
conflicts, 'unsure' leaves the reference not assessed. Payload: the reference
fields and at most PAGE_EXCERPT_CHARS of page text; no complete source.
"""
from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

VERSION = 'cross-script-identity-v1'
PAGE_EXCERPT_CHARS = 1500
_ANSWERS = {'yes', 'no', 'unsure'}
_AUTHOR_ANSWERS = {'agrees', 'differs', 'not_shown'}

_SYSTEM = (
    "You compare a bibliographic reference with a web page that may be written in another language "
    "or writing system. Decide whether the page presents the same work the reference cites, judging "
    "the meaning of the titles across languages, not their spelling. Judge the authors by whether the "
    "page names people whose names correspond to the cited names (for example a romanized surname and "
    "the same surname in its original script). Answer 'unsure' when the page does not let you decide. "
    'Return JSON only: {"same_work": "yes"|"no"|"unsure", '
    '"author": "agrees"|"differs"|"not_shown", "page_authors": [names as written on the page]}.'
)


def _payload(reference: dict, page: dict) -> str:
    return json.dumps({
        'reference': {k: str(reference.get(k) or '')[:400] for k in ('title', 'author', 'year', 'container_title')},
        'page': {'titles': [str(t)[:400] for t in (page.get('titles') or [])[:3]],
                 'date': str(page.get('date') or '')[:40],
                 'opening_text': str(page.get('text') or '')[:PAGE_EXCERPT_CHARS]},
    }, ensure_ascii=False)


def validate_answer(raw) -> dict | None:
    """The model's answer in the fixed schema, or None."""
    if not isinstance(raw, dict) or raw.get('same_work') not in _ANSWERS:
        return None
    author = raw.get('author') if raw.get('author') in _AUTHOR_ANSWERS else 'not_shown'
    names = [str(n).strip()[:120] for n in raw.get('page_authors') or [] if isinstance(n, str) and n.strip()][:6]
    if author != 'not_shown' and not names:
        author = 'not_shown'
    return {'same_work': raw['same_work'], 'author': author, 'page_authors': names}


def judge_cross_script_identity(reference: dict, page: dict, *, completion=None) -> dict:
    """The model's comparison with its provenance; 'unsure' when the call fails."""
    from app.config import settings
    if completion is None:
        from app.services.llm_service import chat_completion_json as completion
    payload = _payload(reference, page)
    receipt: dict = {}
    record = {'version': VERSION, 'payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
              'configured_model': str(getattr(settings, 'LLM_MODEL', '') or '')[:120]}
    try:
        raw = completion(_SYSTEM, payload, temperature=0.0, max_tokens=300, disable_thinking=True, receipt=receipt)
    except Exception as exc:  # noqa: BLE001 - a failed call decides nothing
        logger.info("Cross-script identity call failed: %s", type(exc).__name__)
        return {**record, 'same_work': 'unsure', 'author': 'not_shown', 'page_authors': [], 'call': 'failed'}
    answer = validate_answer(raw)
    provenance = {k: receipt.get(k) for k in ('returned_model', 'returned_provider', 'total_tokens', 'reported_cost_usd')}
    if answer is None:
        return {**record, **provenance, 'same_work': 'unsure', 'author': 'not_shown', 'page_authors': [],
                'call': 'invalid_answer'}
    judged = {**record, **provenance, **answer, 'call': 'answered'}
    logger.info("Cross-script identity %s", json.dumps(
        {k: v for k, v in judged.items() if k != 'page_authors'}, sort_keys=True))
    return judged
