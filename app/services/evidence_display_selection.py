"""Two-stage compact projection of already authorized evidence.

Never edits the candidate reservoir, infers support, or establishes absence.
Model observations are optional; legacy/unassessed inputs use a labelled lexical
fallback. No provider call, source fetch, or source-specific preference lives here.
"""
from __future__ import annotations

import hashlib

DISPLAY_SELECTION_VERSION = "probative-additive-display-v4"


def bound_observation(passage, assessment, claim_text):
    observation = assessment.get("display_observation") or {}
    text = str(passage.get("excerpt") or "")
    start = assessment.get("assessed_text_offset_start", 0)
    end = assessment.get("assessed_text_offset_end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        return {}
    assessed = text[start:end]
    span = observation.get("source_span") or ""
    claims = observation.get("claim_spans") or []
    if (not span or span not in assessed or not isinstance(claims, list)
            or not claims or any(not s.strip() or s not in claim_text for s in claims)
            or hashlib.sha256(assessed.encode()).hexdigest() != assessment.get("assessed_text_sha256")):
        return {}
    from app.services.verification_evidence import _passage_boundary_status
    from app.services.evidence_report import _closed_display_delimiters, _display_prose_blocks, _normalize_display_text
    if _passage_boundary_status(span) != "sentence_complete" or not _closed_display_delimiters(span):
        return {}
    if not any(_normalize_display_text(span) in block for block in _display_prose_blocks(text)):
        return {}
    return observation


def select_display_passages(ranked, claim_text, assessments, preferred_ids, *, terms, excerpt, context=None):
    """Choose a primary, then only additive context (at most two passages)."""
    if not ranked:
        return [], dict(version=DISPLAY_SELECTION_VERSION, selected=[], omitted=[])
    preferred = set(preferred_ids)
    observations = {p["passage_id"]: bound_observation(p, assessments.get(p["passage_id"], {}), claim_text)
                    for p in ranked}
    # Ordinary relevance is not a support score. A directly diagnostic partial
    # passage may lead over a broadly relevant framework/example.
    basis_rank = {"direct_attribution": 3, "necessary_context": 2,
                  "general_framework": 1, "illustrative_example": 0, "unclear": 0}
    ordered = sorted(enumerate(ranked), key=lambda pair: (
        pair[1]["passage_id"] in preferred,
        basis_rank.get(observations[pair[1]["passage_id"]].get("basis"), 1),
        # Within the same attribution tier, claim-span length must not demote
        # an explicitly relevant candidate below an explicitly partial one.
        {"relevant": 3, "partially_relevant": 2, "uncertain": 1}.get(
            assessments.get(pair[1]["passage_id"], {}).get("relevance"), 0),
        # Equal advisory labels preserve the earlier probative ranking.
        # A longer claim phrase is not stronger evidence of the attribution.
        -pair[0]), reverse=True)
    primary = ordered[0][1]
    selected = [primary]
    reasons = {primary['passage_id']: 'protected_primary' if primary['passage_id'] in preferred else
               'attribution_primary' if observations[primary['passage_id']] else 'legacy_ranked_primary'}
    claim_terms = set(terms(claim_text))

    def text(p):
        return observations[p['passage_id']].get('source_span') or excerpt(str(p.get('excerpt') or ''), claim_text)

    def coverage(p):
        obs = observations[p['passage_id']]
        # The reader also sees the primary's longer context. Marginal gain must
        # account for that, not pretend only the compact sentence was supplied.
        visible = context(p) if context else str(p.get('excerpt') or '')
        return (set(terms(' '.join(obs['claim_spans']))) if obs else set()) | (set(terms(visible)) & claim_terms)

    remaining = [p for _, p in ordered if p is not primary]
    while remaining and len(selected) < 3:
        covered = set().union(*(coverage(p) for p in selected))
        shown = [set(terms(text(p))) for p in selected]

        def gain(p):
            pid = p['passage_id']
            obs = observations[pid]
            words = set(terms(text(p)))
            # Overlapping windows with the same visible evidence add nothing.
            redundant = any(words and len(words & s) / max(1, min(len(words), len(s))) >= .85 for s in shown)
            if redundant:
                return (0, 0, 0)
            new = len(coverage(p) - covered)
            basis = obs.get('basis')
            context = bool(obs and basis == 'necessary_context' and
                           not any(observations[s['passage_id']].get('basis') == basis for s in selected))
            # An illustration is not additive just because its unrelated names
            # differ. With no bound observations, require new claim vocabulary.
            if basis == 'illustrative_example':
                new = 0
            return (int(pid in preferred), int(context), new if new >= 2 else 0)

        best = max(remaining, key=gain)
        score = gain(best)
        if not any(score):
            break
        selected.append(best)
        reasons[best['passage_id']] = ('protected_context' if score[0] else 'necessary_context' if score[1]
                                      else 'additional_attributed_aspect' if observations[best['passage_id']]
                                      else 'additional_lexical_context')
        remaining.remove(best)
    return selected, dict(version=DISPLAY_SELECTION_VERSION,
        bound_observation_count=sum(bool(o) for o in observations.values()),
        selection_status='bound_observations' if all(observations.values()) else 'mixed_or_lexical_fallback',
        selected=[dict(passage_id=p['passage_id'], reason=reasons[p['passage_id']]) for p in selected],
        omitted=[dict(passage_id=p['passage_id'], reason='no_added_value_or_display_limit')
                 for p in ranked if p not in selected],
        eligible_passage_ids=[p['passage_id'] for p in ranked],
        support_assessed=False)
