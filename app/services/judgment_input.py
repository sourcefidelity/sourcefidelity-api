"""Rebuild the Judgment layout's model input from a stored verification report.

The Judgment layout runs on demand, after the paper run, and may run long after
a temporary source file is gone. Everything the fixed-ID facet prompt reads is
already in the immutable `VerificationReportRecord.report_payload`: the claim,
the source binding, the clause-level verification candidates, the facet
foundation (the selected source sentences) and the passages' page coordinates.
This module rebuilds exactly that, so `prepare_candidate_prompts` produces the
same prompt the paper run would have produced, and refuses when it cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

from app.config import settings
from app.services.facet_evidence_judgment import (
    FACET_FOUNDATION_VERSION,
    PreparedFacetJudgment,
    prepare_candidate_prompts,
)
from app.services.verification_evidence import (
    CitationSourceBinding,
    ClaimEvidence,
    FacetEvidenceFoundation,
    VerificationCandidateSet,
)

# Foundation versions whose stored sentences this prompt contract can read.
# v10 foundations (24 merged units) stay judgeable; v11 keeps real sentences.
JUDGEABLE_FOUNDATION_VERSIONS = frozenset({FACET_FOUNDATION_VERSION, "exact-facet-evidence-foundation-v11",
                                          "exact-facet-evidence-foundation-v10"})
_CLAIM_STORAGE_FIELDS = ("text_truncated", "text_sha256")


class JudgmentInputUnavailable(ValueError):
    """The stored record cannot reproduce the judgment input exactly."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason_code: str


def judgment_eligibility(payload: dict) -> Eligibility:
    """Only complete full texts with a verified identity are judged (owner decision 6).

    Every refusal carries a reason the window can explain; none is a result.
    """
    # A "see" citation is never judged, whatever text was retrieved (2026-10-07).
    if referral_citation(payload):
        return Eligibility(False, "referral_citation")
    coverage = (payload.get("coverage") or {}).get("level")
    if coverage != "full_text":
        return Eligibility(False, "not_complete_full_text")
    completeness = ((payload.get("coverage") or {}).get("completeness_verdict")
                    or (payload.get("coverage") or {}).get("completeness") or "")
    if isinstance(completeness, dict):
        completeness = completeness.get("verdict") or ""
    if completeness and completeness not in {"complete", "not_applicable"}:
        return Eligibility(False, "not_complete_full_text")
    if (payload.get("source_identity") or {}).get("status") != "verified":
        return Eligibility(False, "identity_unconfirmed")
    foundation = payload.get("facet_evidence_foundation") or {}
    if foundation.get("foundation_version") not in JUDGEABLE_FOUNDATION_VERSIONS:
        return Eligibility(False, "prepared_before_feature")
    if foundation.get("status") not in {"complete", "incomplete"}:
        return Eligibility(False, "evidence_unavailable")
    if (payload.get("claim") or {}).get("text_truncated"):
        return Eligibility(False, "claim_text_truncated")
    return Eligibility(True, "eligible")


_REFERRAL_SIGNAL = re.compile(r"^\(\s*(?:see(?:\s+also)?|cf\.?)\s", re.IGNORECASE)


def referral_citation(payload: dict) -> bool:
    """Every marker naming this source is a "see"/"see also"/"cf." signal,
    which points to further reading rather than supporting the statement
    (owner decision 2026-10-07)."""
    reference_id = (payload.get("source_binding") or {}).get("reference_id")
    markers = [m for m in (payload.get("claim") or {}).get("citation_markers") or []
               if isinstance(m, dict) and (not reference_id or reference_id in (m.get("reference_ids") or []))]
    return bool(markers) and all(_REFERRAL_SIGNAL.match(str(m.get("text") or "")) for m in markers)


def judgment_context_from_payload(payload: dict) -> SimpleNamespace:
    """The artifact view `prepare_candidate_prompts` and its interpreter read."""
    eligibility = judgment_eligibility(payload)
    if not eligibility.eligible:
        raise JudgmentInputUnavailable(eligibility.reason_code)
    claim = {k: v for k, v in (payload.get("claim") or {}).items() if k not in _CLAIM_STORAGE_FIELDS}
    binding = payload.get("source_binding")
    try:
        # A bare "This …" stored as unresolved reads as the previous sentence
        # (owner decision 2026-10-03), for the prompt and its interpretation alike.
        from app.services.antecedent_resolver import (
            _EXPLETIVE_IT, previous_sentence_antecedent, without_expletive_dependency)
        candidates = VerificationCandidateSet.model_validate(payload.get("verification_candidates") or {})
        # A stored part opening with a dummy "It" was marked as needing an
        # antecedent, and the judge then declined it (citation 10, 2026-10-08).
        candidates = candidates.model_copy(update={"candidates": [
            candidate.model_copy(update={"requires_antecedent_context": False, "requires_parent_context": False})
            if candidate.requires_antecedent_context and _EXPLETIVE_IT.match(candidate.text or "") else candidate
            for candidate in candidates.candidates]})
        return SimpleNamespace(
            claim=previous_sentence_antecedent(without_expletive_dependency(ClaimEvidence.model_validate(claim))),
            source_binding=CitationSourceBinding.model_validate(binding) if binding else None,
            source_identity=SimpleNamespace(status=(payload.get("source_identity") or {}).get("status")),
            verification_candidates=candidates,
            facet_evidence_foundation=FacetEvidenceFoundation.model_validate(
                payload.get("facet_evidence_foundation") or {}),
            # Only page coordinates are read (locator status), never passage text.
            passages=[SimpleNamespace(passage_id=p.get("passage_id"), page_index=p.get("page_index"),
                                      page_label=p.get("page_label"))
                      for p in payload.get("passages") or []],
        )
    except (ValueError, TypeError) as exc:
        raise JudgmentInputUnavailable("stored_record_invalid") from exc


def prepare_from_payload(payload: dict) -> tuple[SimpleNamespace, PreparedFacetJudgment]:
    context = judgment_context_from_payload(payload)
    return context, prepare_candidate_prompts(
        context, max_input_tokens=settings.JUDGMENT_MAX_INPUT_TOKENS)
