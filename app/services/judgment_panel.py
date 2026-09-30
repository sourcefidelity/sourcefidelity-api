"""Three-judge panel for the Judgment layout (ARCHITECTURE §7, owner decisions 2026-09-24/25).

For one clause-level candidate, every arm receives the identical prepared
prompt (`facet_evidence_judgment.prepare_candidate_prompts`). Each response is
validated and aggregated in application code exactly as the single-judge path
does (`interpret_candidate_response`), mapped to a panel label, and the three
labels are scored by `judicial_proximity.assess_judicial_panel`. The display
state follows only from that assessment:

    a judge missing or invalid         -> not judged: a judge failed
    moderate/material/directional gap  -> not judged: judges disagree
    any agreed `uncertain`             -> not judged: judges could not decide
    agreement on {supports}            -> supported (teal)
    {contradicts} or {contradicts, mixed} -> contradicts (purple)
    other agreement with qualifies/mixed  -> qualified or mixed (amber)
    agreement on {none}                -> insufficient evidence (red), but only
                                          after a wider search agrees again

Nothing here writes to the Evidence Package, the Sources report, summaries,
counts or exports. No provider is called unless the caller passes routes
built by `judge_arms.formal_judge_arms`, which a run does when the paper is
checked (or when Judgment is first opened on a report checked before that).
"""
from __future__ import annotations

import contextvars
import hashlib
import time
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import ValidationError

from app.config import settings
from app.services.facet_evidence_judgment import (
    FACET_JUDGMENT_VERSION,
    CandidateFacetFinding,
    PreparedCandidate,
    interpret_candidate_response,
)
from app.services.judge_arms import call_cost_usd
from app.services.judicial_proximity import JudicialPanelAssessment, assess_judicial_panel
from app.services.llm_service import LLMCallFailure, LLMRoute, chat_completion_json

PANEL_VERSION = "three-judge-panel-v1"
REQUIRED_ARMS = 3
DisplayState = Literal["supported", "qualified", "contradicts", "insufficient", "not_judged"]
# Owner decision 10: the most serious judged result is shown for a claim that
# cites several sources.
SEVERITY = {"insufficient": 4, "contradicts": 3, "qualified": 2, "supported": 1}
_DISAGREEMENT = frozenset({"moderate_review", "material_disagreement", "directional_disagreement"})


@dataclass
class ArmResult:
    arm_id: str
    model: str
    status: Literal["valid", "failed"]
    label: str | None = None
    finding: CandidateFacetFinding | None = None
    response: dict | None = None          # validated model JSON, for the cache
    failure: str | None = None
    receipt: dict = field(default_factory=dict)
    cost_usd: float | None = None
    cost_basis: str = "unpriced"
    cache_key: str = ""
    cached: bool = False


@dataclass
class CandidatePanelResult:
    candidate_id: str
    display_state: DisplayState
    reason_code: str
    arms: list[ArmResult]
    assessment: JudicialPanelAssessment | None = None
    prompt_sha256: str = ""
    wider_search: dict | None = None

    @property
    def spend_usd(self) -> float:
        return sum(a.cost_usd or 0.0 for a in self.arms if not a.cached)


def prompt_sha256(prepared: PreparedCandidate) -> str:
    return hashlib.sha256((prepared.system_prompt + "\x00" + prepared.user_prompt).encode()).hexdigest()


def arm_cache_key(route: LLMRoute, policy_hash: str, prepared: PreparedCandidate) -> str:
    """Reused only for the same arm, model, policy, contract and exact prompt."""
    parts = [route.arm_id, route.model, policy_hash, FACET_JUDGMENT_VERSION,
             hashlib.sha256(prepared.system_prompt.encode()).hexdigest(),
             hashlib.sha256(prepared.user_prompt.encode()).hexdigest()]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def arm_label(finding: CandidateFacetFinding) -> str | None:
    """A candidate outcome as a panel label; None when the response is unusable."""
    if finding.status == "uncertain":
        return "uncertain"
    if finding.status == "not_assessed":
        # The judge could not resolve the student's antecedent: undecided, not failed.
        return "uncertain" if finding.context_resolution in {"ambiguous", "unresolved"} else None
    outcome = finding.derived_outcome
    if outcome == "supports":
        return "supports"
    if outcome == "contradicts":
        return "contradicts"
    if outcome == "insufficient_evidence":
        return "none"
    if outcome == "mixed_or_qualified":
        directions = {mapping.direction for mapping in finding.mappings}
        if "mixed" in directions or {"supports", "contradicts"} <= directions:
            return "mixed"
        return "qualifies"
    return None


def agreed_display_state(values: set[str]) -> tuple[DisplayState, str]:
    """The state for labels already known not to disagree (one label, or a proximate panel)."""
    if "uncertain" in values:
        return "not_judged", "judges_undecided"
    if values == {"none"}:
        return "insufficient", "agreed_no_evidence"
    if values == {"supports"}:
        return "supported", "agreed_supports"
    if "contradicts" in values and values <= {"contradicts", "mixed"}:
        return "contradicts", "agreed_contradicts"
    return "qualified", "agreed_qualified_or_mixed"


_SINGLE_JUDGE = {"supports": ("supported", "judge_supports"), "qualifies": ("qualified", "judge_qualifies"),
                 "mixed": ("qualified", "judge_mixed"), "contradicts": ("contradicts", "judge_contradicts"),
                 "none": ("insufficient", "judge_no_evidence"), "uncertain": ("not_judged", "judge_undecided")}
_TRANSIENT = frozenset({"timeout", "connection_failed", "rate_limited", "provider_server_error", "empty_response"})
_RETRY_PAUSE_SECONDS = 2.0


def panel_display_state(
    labels: dict[str, str], required: int = REQUIRED_ARMS,
) -> tuple[DisplayState, str, JudicialPanelAssessment | None]:
    if len(labels) < required:
        return "not_judged", "judge_failed", None
    if required == 1:
        # One judge (owner decision 2026-09-27): its label is the result.
        state, reason = _SINGLE_JUDGE[next(iter(labels.values()))]
        return state, reason, None
    assessment = assess_judicial_panel(labels, required_formal_count=required)
    if assessment.formal_status in _DISAGREEMENT:
        return "not_judged", "judges_disagree", assessment
    state, reason = agreed_display_state(set(labels.values()))
    return state, reason, assessment


def most_serious(states: list[str]) -> str:
    """The claim's shown state across its sources; not judged only if none was judged."""
    judged = [s for s in states if s in SEVERITY]
    return max(judged, key=SEVERITY.__getitem__) if judged else "not_judged"


CacheLookup = Callable[[str], dict | None]
CacheStore = Callable[[ArmResult], None]


def _run_arm(route, key, cached, prepared, context, sentences, call, temperature: float = 0.0) -> ArmResult:
    """Runs in a worker thread: no database access here (sessions are not thread-safe)."""
    result = ArmResult(arm_id=route.arm_id, model=route.model, status="failed", cache_key=key)
    raw = cached
    if raw is None:
        receipt: dict = {}
        for attempt in (1, 2):
            try:
                raw = call(prepared.system_prompt, prepared.user_prompt, temperature=temperature,
                           max_tokens=route.max_output_tokens
                           or min(settings.VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS, 1_600),
                           max_retries=1, route=route, receipt=receipt)
                result.failure = None
                break
            except LLMCallFailure as exc:
                result.failure = exc.category
                # A dropped connection or an overload usually clears: try once more.
                if attempt == 1 and exc.category in _TRANSIENT:
                    time.sleep(_RETRY_PAUSE_SECONDS)
                    continue
                break
            except Exception as exc:   # an arm failure never fails the run
                result.failure = type(exc).__name__
                break
        receipt["prompt_sha256"] = prompt_sha256(prepared)
        result.receipt = receipt
        result.cost_usd, result.cost_basis = call_cost_usd(receipt)
        if result.failure:
            return result
    else:
        result.cached = True
    try:
        finding = interpret_candidate_response(context, prepared, raw, sentences)
    except (ValidationError, ValueError, TypeError, RuntimeError):
        result.failure = "invalid_response"
        return result
    label = arm_label(finding)
    if label is None:
        result.failure = "invalid_response"
        return result
    result.status, result.label, result.finding = "valid", label, finding
    result.response = raw if isinstance(raw, dict) else json.loads(json.dumps(raw))
    return result


def judge_candidate(
    prepared: PreparedCandidate,
    context,
    sentences: dict,
    routes: list[LLMRoute],
    policy_hash: str,
    *,
    cache_lookup: CacheLookup | None = None,
    cache_store: CacheStore | None = None,
    call=chat_completion_json,
) -> CandidatePanelResult:
    """Run the configured judges in parallel on the identical prompt and derive the state."""
    if not routes:
        raise ValueError("Judgment needs at least one judge")
    keys = [arm_cache_key(route, policy_hash, prepared) for route in routes]
    # Cache reads and writes stay on the calling thread.
    cached = [cache_lookup(key) if cache_lookup else None for key in keys]
    parent = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        futures = [pool.submit(parent.copy().run, _run_arm, route, key, hit, prepared,
                               context, sentences, call)
                   for route, key, hit in zip(routes, keys, cached)]
        arms = [future.result() for future in futures]
    if cache_store:
        for arm in arms:
            if arm.status == "valid" and not arm.cached:
                cache_store(arm)
    labels = {arm.arm_id: arm.label for arm in arms if arm.status == "valid"}
    state, reason, assessment = panel_display_state(labels, required=len(routes))
    return CandidatePanelResult(candidate_id=prepared.bundle.candidate_id, display_state=state,
                                reason_code=reason, arms=arms, assessment=assessment,
                                prompt_sha256=prompt_sha256(prepared))


def not_judged(candidate_id: str, reason_code: str) -> CandidatePanelResult:
    """A candidate decided before any call (e.g. unresolved antecedent, over budget)."""
    return CandidatePanelResult(candidate_id=candidate_id, display_state="not_judged",
                                reason_code=reason_code, arms=[])
