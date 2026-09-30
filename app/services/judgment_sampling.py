"""Repeat sampling of one Judgment arm, for Phase E validation only (ROADMAP P7).

The panel asks three different models once each. This module asks one arm the
same candidate several times, so Phase E can measure how often a single judge
disagrees with itself and whether that instability falls on the same claims
the panel marks "judges disagree". Three kinds of repeat are supported:

    temperature 0, original order     -> provider nondeterminism only
    temperature > 0, original order   -> sampling spread of the judge
    temperature 0, reordered passages -> sensitivity to evidence order

A reordered prompt moves whole passage groups; every sentence keeps its fixed
ID, passage group and passage sequence, so the response is validated and
interpreted by the unchanged contract. Sample cache keys are separate from the
panel's, so a sample can never answer a production lookup or be served one.

Nothing here is called by the Judgment run, the report or any export. Labels
from samples are an evaluation measurement, never a displayed result.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from app.services.judgment_panel import (
    ArmResult,
    CacheLookup,
    _run_arm,
    agreed_display_state,
    arm_cache_key,
    panel_display_state,
)
from app.services.llm_input_boundary import json_data_envelope
from app.services.llm_service import LLMRoute, chat_completion_json

SAMPLING_VERSION = "judgment-repeat-sampling-v1"
MAX_TEMPERATURE = 1.0


@dataclass(frozen=True)
class SampleSpec:
    """One repeat: its index within a series, temperature and passage-order seed."""

    index: int
    temperature: float = 0.0
    order_seed: int | None = None     # None keeps the prepared order

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("sample index must be non-negative")
        if not 0.0 <= self.temperature <= MAX_TEMPERATURE:
            raise ValueError("sample temperature is outside 0-1")

    def as_json(self) -> dict:
        return {"index": self.index, "temperature": self.temperature, "order_seed": self.order_seed}


def reorder_passages(prepared, seed: int):
    """The same candidate with its passage groups in a seeded order.

    Returns (prepared, changed). Sentences stay together within their passage
    group and keep every field; only the list order changes. One group cannot
    be reordered, and a permutation equal to the original is reported as
    unchanged rather than silently counted as a perturbation.
    """
    data = json.loads(prepared.user_prompt)
    sentences = data.get("evidence_sentences") or []
    groups: dict[int, list] = {}
    for sentence in sentences:
        groups.setdefault(sentence.get("passage_group"), []).append(sentence)
    order = list(groups)
    random.Random(seed).shuffle(order)
    if order == list(groups):
        return prepared, False
    data["evidence_sentences"] = [s for group in order for s in groups[group]]
    return replace(prepared, user_prompt=json_data_envelope(data)), True


def sample_cache_key(route: LLMRoute, policy_hash: str, prepared, spec: SampleSpec) -> str:
    """Keyed on the panel key of the unperturbed prompt plus the sample spec."""
    base = arm_cache_key(route, policy_hash, prepared)
    parts = [SAMPLING_VERSION, base, str(spec.index), repr(float(spec.temperature)),
             "" if spec.order_seed is None else str(spec.order_seed)]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass
class SampleResult:
    spec: SampleSpec
    arm: ArmResult
    order_changed: bool
    prompt_sha256: str


def sample_arm(
    prepared,
    context,
    sentences: dict,
    route: LLMRoute,
    policy_hash: str,
    spec: SampleSpec,
    *,
    cache_lookup: CacheLookup | None = None,
    call=chat_completion_json,
) -> SampleResult:
    """One repeat of one arm; a failure is recorded, never raised."""
    key = sample_cache_key(route, policy_hash, prepared, spec)
    changed = False
    sent = prepared
    if spec.order_seed is not None:
        sent, changed = reorder_passages(prepared, spec.order_seed)
    cached = cache_lookup(key) if cache_lookup else None
    arm = _run_arm(route, key, cached, sent, context, sentences, call, temperature=spec.temperature)
    digest = hashlib.sha256((sent.system_prompt + "\x00" + sent.user_prompt).encode()).hexdigest()
    return SampleResult(spec=spec, arm=arm, order_changed=changed, prompt_sha256=digest)


# ---- decision rules compared in Phase E -------------------------------------

def single_judge_state(label: str | None) -> tuple[str, str]:
    """What one judge alone would show: no disagreement state exists."""
    if label is None:
        return "not_judged", "judge_failed"
    return agreed_display_state({label})


def self_consistency_state(labels: Iterable[str | None], required: int = 3) -> tuple[str, str]:
    """The panel rule applied to repeats of one judge instead of three models."""
    valid = {f"s{i}": label for i, label in enumerate(labels) if label is not None}
    state, reason, _ = panel_display_state(valid, required=required)
    return state, reason


# ---- Phase E summary -----------------------------------------------------------

def _share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def summarize(rows: list[dict], rules: dict[str, Callable[[dict], tuple[str, str]]]) -> dict:
    """State distributions per rule and their overlap with the panel.

    Each row is one candidate. `rules` maps a rule name to a function of the
    row returning (display_state, reason_code); a rule named "panel" is the
    reference. Counts only: agreement between rules is not correctness, which
    needs the owner's model-blind labels.
    """
    names = list(rules)
    states = {name: [rules[name](row)[0] for row in rows] for name in names}
    reasons = {name: [rules[name](row)[1] for row in rows] for name in names}
    total = len(rows)
    summary = {"sampling_version": SAMPLING_VERSION, "candidates": total, "rules": {}}
    for name in names:
        counts = Counter(states[name])
        summary["rules"][name] = {
            "states": dict(sorted(counts.items())),
            "not_judged_share": _share(counts.get("not_judged", 0), total),
            "red_share": _share(counts.get("insufficient", 0), total),
            "reasons": dict(sorted(Counter(reasons[name]).items())),
        }
    if "panel" in states:
        reference = states["panel"]
        disagree = [r == "judges_disagree" for r in reasons["panel"]]
        for name in names:
            if name == "panel":
                continue
            other = states[name]
            both_judged = [(a, b) for a, b in zip(reference, other)
                           if a != "not_judged" and b != "not_judged"]
            unstable = [r in {"judges_disagree", "judges_undecided"} for r in reasons[name]]
            summary["rules"][name]["versus_panel"] = {
                "same_state": sum(a == b for a, b in zip(reference, other)),
                "both_judged": len(both_judged),
                "both_judged_same_state": sum(a == b for a, b in both_judged),
                # Where the panel abstains for disagreement, does this rule also abstain?
                "panel_disagree": sum(disagree),
                "panel_disagree_and_rule_unstable": sum(d and u for d, u in zip(disagree, unstable)),
                "panel_agrees_but_rule_unstable": sum((not d) and u for d, u in zip(disagree, unstable)),
                # Labels this rule would show where the panel shows nothing.
                "rule_judged_where_panel_not": sum(a == "not_judged" and b != "not_judged"
                                                   for a, b in zip(reference, other)),
            }
    return summary
