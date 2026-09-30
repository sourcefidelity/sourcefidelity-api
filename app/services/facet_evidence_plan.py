"""Operation-local reuse and separate projections of shadow usefulness results.

This is not an admission, facet-validation, permission or relationship gate.
Callers must supply already authorized, source-bound inputs and invoke only
inside an explicitly enabled experiment. No global/durable cache is created.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.config import settings
from app.services import facet_passage_selector as selector
from app.services.factual_facet_composition import FactualFacetCompositionResult

PLAN_VERSION = "shared-facet-evidence-plan-v1"


def group_provisional_compositions(draws):
    """Lossless exact grouping, not a semantic vote or selected interpretation.

    Each input is (draw_id, composition). Statistically/model-independent
    authorship is a caller provenance question; this verifies identical input
    identity and preserves rejected and minority proposals, too.
    """
    if not draws or len({key for key, _ in draws}) != len(draws):
        raise ValueError("invalid_draw_ids")
    if len({(r.candidate_id, r.candidate_text_sha256, r.proposal_prompt_sha256) for _, r in draws}) != 1:
        raise ValueError("different_student_inputs")
    groups = {}
    outcomes = []
    total = 0
    for key, result in draws:
        outcomes.append(dict(draw_id=key, status=result.status, coverage=result.coverage_status,
                             failure_code=result.failure_code, proposed_count=len(result.proposed_facets)))
        for facet in result.proposed_facets:
            shape = dict(gloss=facet.checking_gloss, inherited=facet.gloss_inherits_from_complete_unit,
                segments=[s.model_dump(mode="json") for s in facet.segments],
                constraints=[dict(kind=c.kind,scope=c.scope,segments=[s.model_dump(mode="json") for s in c.segments])
                             for c in result.scope_constraints if facet.facet_id in c.applies_to_facet_ids],
                edges=[dict(kind=e.kind,segments=[s.model_dump(mode="json") for s in e.segments],
                            connected_glosses=[f.checking_gloss for f in result.proposed_facets if f.facet_id in e.connects_facet_ids])
                       for e in result.structural_edges if facet.facet_id in e.connects_facet_ids])
            digest=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()
            group=groups.setdefault(digest,dict(signature=digest,members=[]))
            group['members'].append(dict(draw_id=key,facet_id=facet.facet_id,
                                         preserved=facet.facet_id in result.accepted_facet_ids))
            total+=1
    return dict(groups=list(groups.values()),draw_outcomes=outcomes,proposed_count=total,
                retained_count=sum(len(g['members']) for g in groups.values()),canonical_reading=None)


def composed_selector_facets(
    artifact, composition: FactualFacetCompositionResult,
    sentences: list[selector.SelectorSentence],
) -> list[selector.SelectorFacet]:
    """Adapt existing source-blind propositions, not source-conditioned guesses.

    Development-only. The caller retains the original proposal/preservation
    receipts and independently checks source/model permissions. A faithful
    preservation label is a provisional model observation, not semantic gold.
    Partial compositions and semantic repairs deliberately abstain here.
    """
    candidate = next((c for c in artifact.verification_candidates.candidates
                      if c.candidate_id == composition.candidate_id), None)
    if (candidate is None or composition.status != "complete"
            or composition.coverage_status != "complete"
            or composition.failure_code != "none"
            or composition.interpretation_status != "as_written"
            or composition.interpretation_id is not None
            or composition.material_uncovered_ids
            or not composition.proposal_prompt_sha256
            or not composition.preservation_prompt_sha256
            or composition.candidate_text_sha256 != hashlib.sha256(candidate.text.encode()).hexdigest()):
        raise ValueError("unusable_source_blind_composition")
    facets = composition.proposed_facets
    constraints = composition.scope_constraints
    facet_ids = [f.facet_id for f in facets]
    constraint_ids = [c.constraint_id for c in constraints]
    sentence_ids = [s.sentence_id for s in sentences]
    if (not facets or not sentences or len(set(facet_ids)) != len(facet_ids)
            or len(set(constraint_ids)) != len(constraint_ids)
            or len(set(sentence_ids)) != len(sentence_ids)
            or len(facets) > selector.MAX_SELECTOR_FACETS
            or len(sentences) > selector.MAX_SELECTOR_SENTENCES
            or len(facets) * len(sentences) > selector.MAX_SELECTOR_PAIRS):
        raise ValueError("invalid_composition_grid")
    for ids, accepted, findings, field in (
        (facet_ids, composition.accepted_facet_ids, composition.preservation_findings, "facet_id"),
        (constraint_ids, composition.accepted_constraint_ids, composition.constraint_preservation_findings, "constraint_id"),
    ):
        if (len(accepted) != len(ids) or set(accepted) != set(ids)
                or len(findings) != len(ids)
                or {getattr(f, field) for f in findings} != set(ids)
                or any(not f.accepted or f.status != "faithful" for f in findings)):
            raise ValueError("unpreserved_composition")
    for item in [*facets, *constraints, *composition.structural_edges]:
        if item.candidate_id != candidate.candidate_id:
            raise ValueError("wrong_composition_candidate")
        for seg in item.segments:
            if (not 0 <= seg.local_start < seg.local_end <= len(artifact.claim.text)
                    or artifact.claim.text[seg.local_start:seg.local_end] != seg.text
                    or seg.paper_start != artifact.claim.passage_start + seg.local_start
                    or seg.paper_end != artifact.claim.passage_start + seg.local_end
                    or not any(s.local_start <= seg.local_start < seg.local_end <= s.local_end
                               for s in candidate.segments)):
                raise ValueError("stale_composition_span")
    for constraint in constraints:
        if not set(constraint.applies_to_facet_ids).issubset(facet_ids):
            raise ValueError("unknown_constraint_facet")
    for facet in facets:
        if set(facet.constraint_ids) != {c.constraint_id for c in constraints
                                        if facet.facet_id in c.applies_to_facet_ids}:
            raise ValueError("incomplete_constraint_binding")
    for edge in composition.structural_edges:
        if not set(edge.connects_facet_ids).issubset(facet_ids):
            raise ValueError("unknown_edge_facet")
    # Include the complete exact unit and the entire composition fingerprint in
    # each fixed obligation so changed context/scope cannot reuse an old map.
    digest = hashlib.sha256(json.dumps(composition.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
    return [selector.SelectorFacet(
        facet_id=f.facet_id, allowed_sentence_ids=sentence_ids,
        claim_spans=[(s.local_start,s.local_end) for s in f.segments]
            + [(s.local_start,s.local_end) for c in constraints if f.facet_id in c.applies_to_facet_ids for s in c.segments]
            + [(s.local_start,s.local_end) for e in composition.structural_edges if f.facet_id in e.connects_facet_ids for s in e.segments],
        text=json.dumps({
            "complete_proposition": f.checking_gloss,
            "exact_segments": [s.text for s in f.segments],
            "complete_citation_unit": artifact.claim.text,
            "scope_constraints": [dict(kind=c.kind, scope=c.scope, exact_segments=[s.text for s in c.segments]) for c in constraints
                                  if f.facet_id in c.applies_to_facet_ids],
            "structural_dependencies": [dict(kind=e.kind, exact_segments=[s.text for s in e.segments], connects=e.connects_facet_ids) for e in composition.structural_edges
                                        if f.facet_id in e.connects_facet_ids],
            "composition_sha256": digest,
            "interpretation": "Provisional source-blind proposition; exact wording and unresolved scope remain authoritative.",
        }, ensure_ascii=False, sort_keys=True),
    ) for f in facets]


@dataclass(frozen=True)
class FacetDisplayPlan:
    selected_sentence_ids: tuple[str, ...]
    additional_sentence_ids: tuple[str, ...]
    # These are selection annotations, never support or accuracy verdicts.
    unresolved_facet_ids: tuple[str, ...]
    partially_addressed_facet_ids: tuple[str, ...]
    judgment_sentence_ids: tuple[str, ...]
    selection_status: str
    projection_version: str = PLAN_VERSION


def project_facet_evidence(
    facets: list[selector.SelectorFacet],
    sentences: list[selector.SelectorSentence],
    result: selector.FacetPassageSelectorResult,
    *,
    display_limit: int = 3,
    protected_sentence_ids: tuple[str, ...] = (),
    complementary_only: bool = False,
) -> FacetDisplayPlan:
    """Prefer new facet context; never prune the judgment evidence reservoir."""
    if not 1 <= display_limit <= selector.MAX_SELECTOR_SENTENCES:
        raise ValueError("invalid_display_limit")
    ids = [s.sentence_id for s in sentences]
    facet_ids = [f.facet_id for f in facets]
    if (not facets or not sentences or len(facets) > selector.MAX_SELECTOR_FACETS
            or len(sentences) > selector.MAX_SELECTOR_SENTENCES):
        raise ValueError("invalid_input_size")
    if len(set(ids)) != len(ids) or len(set(facet_ids)) != len(facet_ids):
        raise ValueError("duplicate_input_id")
    if any(s not in ids for f in facets for s in f.allowed_sentence_ids):
        raise ValueError("unknown_sentence_id")
    if any(s not in ids for s in protected_sentence_ids):
        raise ValueError("unknown_protected_sentence")
    expected = {(f.facet_id, s) for f in facets for s in f.allowed_sentence_ids}
    if len(expected) > selector.MAX_SELECTOR_PAIRS:
        raise ValueError("pair_budget_exceeded")
    pairs = [(a.facet_id, a.sentence_id) for a in result.assessments]
    valid = (
        result.status == "complete" and result.failure_code == "none"
        and result.facet_sentence_sha256 == selector.facet_sentence_fingerprint(facets, sentences)
        and len(pairs) == len(set(pairs)) and set(pairs) == expected
    )
    labels = {(a.facet_id, a.sentence_id): a.usefulness for a in result.assessments} if valid else {}
    selected = list(dict.fromkeys(protected_sentence_ids))
    sufficient: set[str] = set()
    partial: set[str] = set()

    def update(sid):
        for fid in facet_ids:
            label = labels.get((fid, sid))
            if label == "sufficient":
                sufficient.add(fid)
            elif label == "partially_useful":
                partial.add(fid)

    for sid in selected:
        update(sid)
    remaining = [s for s in ids if s not in selected]
    while remaining and len(selected) < display_limit:
        def gain(sid):
            new_full = sum(labels.get((f, sid)) == "sufficient" for f in facet_ids if f not in sufficient)
            new_partial = sum(labels.get((f, sid)) == "partially_useful" for f in facet_ids if f not in sufficient and f not in partial)
            # A second partial passage may be essential together with the first;
            # it never upgrades the facet to sufficient automatically.
            extra_partial = (0 if complementary_only else sum(
                labels.get((f, sid)) == "partially_useful"
                for f in partial if f not in sufficient
            ))
            return new_full, new_partial, extra_partial
        sid = max(remaining, key=gain) if valid else remaining[0]
        if valid and not any(gain(sid)):
            break  # A display cap is not a quota; the reservoir stays complete.
        selected.append(sid)
        remaining.remove(sid)
        update(sid)
    return FacetDisplayPlan(
        selected_sentence_ids=tuple(selected),
        additional_sentence_ids=tuple(s for s in ids if s not in selected),
        unresolved_facet_ids=tuple(f for f in facet_ids if f not in sufficient),
        partially_addressed_facet_ids=tuple(f for f in facet_ids if f in partial and f not in sufficient),
        judgment_sentence_ids=tuple(ids),
        selection_status="advisory_complete" if valid else "not_assessed",
        projection_version=("shared-facet-evidence-plan-v2-complementary"
                            if complementary_only else PLAN_VERSION),
    )


def project_inspection_evidence(candidate_text, facets, sentences, result, *, display_limit=3, protected_sentence_ids=()):
    """Use bound material-part/context annotations, never sum partials to whole.

    New token coverage is a display diagnostic, not semantic completeness. The
    model's material-part boundaries still require development quality review.
    """
    baseline = project_facet_evidence(facets, sentences, result, display_limit=display_limit,
                                     protected_sentence_ids=protected_sentence_ids, complementary_only=True)
    pairs = {(a.facet_id, a.sentence_id): a.usefulness for a in result.assessments}
    details = {(d.facet_id, d.sentence_id): d for d in result.inspection_details}
    valid = (baseline.selection_status == "advisory_complete"
             and result.selector_version == "bounded-facet-inspection-v7-scoped"
             and result.inspection_claim_sha256 == hashlib.sha256(candidate_text.encode()).hexdigest()
             and len(details) == len(result.inspection_details)
             and set(details) == {p for p, label in pairs.items() if label in {"sufficient", "partially_useful"}})
    for pair, detail in details.items():
        if detail.role == "C":
            valid = valid and not detail.claim_spans and bool(detail.context_for) and all(
                sid != detail.sentence_id and (detail.facet_id, sid) in details
                and details[(detail.facet_id, sid)].role in {"W", "M"} for sid in detail.context_for)
        else:
            valid = valid and bool(detail.claim_spans) and not detail.context_for and all(
                0 <= start < end <= len(candidate_text) for start, end in detail.claim_spans)
            facet = next((f for f in facets if f.facet_id == detail.facet_id), None)
            bounded = bool(facet and facet.claim_spans and all(
                0 <= a < b <= len(candidate_text) for a,b in facet.claim_spans))
            allowed = {i for a,b in facet.claim_spans for i in range(a,b)} if bounded else set()
            valid = valid and bool(allowed) and all(
                i in allowed or candidate_text[i].isspace()
                for a,b in detail.claim_spans if 0 <= a < b <= len(candidate_text) for i in range(a,b))
        valid = valid and pairs.get(pair) == ("sufficient" if detail.role == "W" else "partially_useful")
    if not valid:
        failed = result.model_copy(update={"status": "not_assessed"})
        return project_facet_evidence(facets, sentences, failed, display_limit=display_limit,
                                      protected_sentence_ids=protected_sentence_ids)
    ids = [s.sentence_id for s in sentences]
    selected = list(dict.fromkeys(protected_sentence_ids))
    coverage = {f.facet_id: set() for f in facets}
    def chars(d):
        return {i for start, end in d.claim_spans for i in range(start, end) if not candidate_text[i].isspace()}
    def update(sid):
        for (fid, unit), d in details.items():
            if unit == sid and d.role in {"W", "M"}:coverage[fid].update(chars(d))
    for sid in selected:update(sid)
    while len(selected) < display_limit:
        def gain(sid):
            additions = [(fid,d) for (fid,unit),d in details.items() if unit==sid]
            necessary = sum(d.role == "C" and any(x in selected for x in d.context_for) for _,d in additions)
            new_parts = sum(d.role in {"W","M"} and bool(chars(d)-coverage[fid]) for fid,d in additions)
            return necessary, new_parts
        remaining = [sid for sid in ids if sid not in selected]
        if not remaining:break
        best = max(remaining,key=gain)
        if not any(gain(best)):break
        selected.append(best);update(best)
    whole = {fid for (fid,sid),d in details.items() if sid in selected and d.role == "W"}
    return FacetDisplayPlan(tuple(selected),tuple(s for s in ids if s not in selected),
        tuple(f.facet_id for f in facets if f.facet_id not in whole),
        tuple(f.facet_id for f in facets if coverage[f.facet_id] and f.facet_id not in whole),
        tuple(ids),"advisory_complete","shared-facet-inspection-plan-v3")


class OperationFacetSelection:
    """Single-operation, single-entry memo. Discard/clear at operation end.

permission_context must identify current authorization/retention/model-processing
capabilities, not credentials. source_binding includes manifestation, extraction,
page/sentence coordinates and coverage hashes supplied by the owning pipeline.
This key detects changes; it does not validate those permissions or bindings.
"""

    def __init__(self):
        self._key = None
        self._result = None
        self.selector_invocations = 0
        self.reuse_hits = 0

    def clear(self):
        self._key = None
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.clear()

    def select(self, candidate_text, complete_citation_unit, facets, sentences, *,
               permission_context: str, source_binding: str, enabled: bool = False,
               complete_propositions: bool = False, inspection_parts: bool = False):
        if not enabled:
            self.clear()
            raise ValueError("facet_selection_not_enabled")
        if not permission_context or not source_binding:
            self.clear()
            raise ValueError("missing_reuse_boundary")
        payload = {
            "version": PLAN_VERSION,
            "selector_version": selector.FACET_PASSAGE_SELECTOR_VERSION,
            "prompt": selector._INSPECTION_PROMPT if inspection_parts else selector._PROPOSITION_PROMPT if complete_propositions else selector._SYSTEM_PROMPT,
            "complete_propositions": complete_propositions,
            "inspection_parts": inspection_parts,
            "candidate": candidate_text, "citation": complete_citation_unit,
            "facets": [f.model_dump(mode="json") for f in facets],
            "sentences": [s.model_dump(mode="json") for s in sentences],
            "permission_context": permission_context, "source_binding": source_binding,
            "model": settings.LLM_MODEL, "endpoint": settings.LLM_BASE_URL,
            "input_budget": settings.VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS,
            "output_budget": settings.VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key == self._key and self._result is not None:
            self.reuse_hits += 1
            return self._result.model_copy(deep=True)
        self.clear()
        self.selector_invocations += 1
        result = selector.classify_facet_sentence_usefulness(
            candidate_text, complete_citation_unit, facets, sentences,
            **({"complete_propositions": True} if complete_propositions else {}),
            **({"inspection_parts": True} if inspection_parts else {}),
        )
        # Do not convert operational failure into a reusable completed search.
        try:
            reusable = project_facet_evidence(
                facets, sentences, result,
            ).selection_status == "advisory_complete"
        except ValueError:
            reusable = False
        if reusable:
            self._key = key
            self._result = result.model_copy(deep=True)
        return result
