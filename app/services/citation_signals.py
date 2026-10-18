"""Signals a citation marker carries about how its source is used.

One rule for "this source is cited as further reading", read by the
citation-use router and by Judgment eligibility. They had disagreed: the
router excluded every source of a sentence when "see also" appeared anywhere
in it, so "X rose (Jones, 2021; see also Smith, 2020)" left Jones unjudged
with no note (rules audit 2026-10-18; owner decision: only the referral
source is skipped).
"""
from __future__ import annotations

import re

# "see", "see also", "cf." opening a marker or the part of a grouped marker
# that names this source.
_SIGNAL_OPENING = re.compile(r"^[\s(\[]*(?:see(?:\s+also)?|cf\.?)\s", re.IGNORECASE)
# The words just before a narrative marker ("see also Smith (2020)"), or a
# pointer earlier in the source's own clause ("For further discussion, see
# the related literature (Smith, 2020)").
_SIGNAL_BEFORE = re.compile(r"\b(?:see(?:\s+also)?|cf\.?)\s*$", re.IGNORECASE)
_POINTER_IN_CLAUSE = re.compile(
    r"\bfor\s+(?:further\s+)?(?:background|discussion|overview|review|examples?|details?)\b.{0,80}\bsee\b"
    r"|\bsee\s+also\b|\bcf\.", re.IGNORECASE | re.DOTALL)


def _own_part(marker_text: str, surname: str) -> str:
    parts = [part for part in re.split(r";", marker_text) if part.strip()]
    if len(parts) > 1 and surname:
        named = [part for part in parts if surname.casefold() in part.casefold()]
        if len(named) == 1:
            return named[0]
    return marker_text


def referral_source(claim_text: str, markers: list[dict], reference_id: str | None,
                    cited_author_label: str = "") -> bool:
    """Every marker naming this source cites it as further reading.

    `markers` are dicts with text, local_start, local_end and reference_ids
    (claim-local offsets).
    """
    surname = re.split(r"[,\s]", (cited_author_label or "").strip(), 1)[0]
    ordered = sorted((m for m in markers if isinstance(m, dict)), key=lambda m: int(m.get("local_start") or 0))
    own = [m for m in ordered if not reference_id or reference_id in (m.get("reference_ids") or [])]
    if not own:
        return False
    for marker in own:
        start = int(marker.get("local_start") or 0)
        previous_end = max([int(m.get("local_end") or 0) for m in ordered if int(m.get("local_end") or 0) <= start],
                           default=0)
        clause = claim_text[previous_end:start]
        if not (_SIGNAL_OPENING.match(_own_part(str(marker.get("text") or ""), surname))
                or _SIGNAL_BEFORE.search(clause) or _POINTER_IN_CLAUSE.search(clause)):
            return False
    return True
