"""Protected, ID-only evaluation contract for passage-retrieval ablations.

The harness compares fixed baseline and experimental passage rankings without
carrying source text or changing a verification result.  Human evidence labels
remain external to the retriever.  Positive-only labels can measure recall but
cannot be used to claim precision.
"""

from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RETRIEVAL_ABLATION_VERSION = "protected-retrieval-ablation-v1"
MAX_ABLATION_PASSAGES = 10
DEFAULT_SAFETY_WINDOW = 5

PassageRole = Literal[
    "body_prose",
    "reference_list",
    "citation_notes",
    "publication_metadata",
    "unknown",
]

_NOISE_ROLES = {
    "reference_list",
    "citation_notes",
    "publication_metadata",
}


class RetrievalAblationError(ValueError):
    """The fixed retrieval comparison violated its evaluation contract."""


class RetrievalInputSnapshot(BaseModel):
    """Immutable identity of one private input used to build the manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RankedRetrievalPassage(BaseModel):
    """One exact authorized passage in one retrieval arm."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str = Field(min_length=1, max_length=128)
    rank: int = Field(ge=1, le=MAX_ABLATION_PASSAGES)
    channels: list[str] = Field(min_length=1, max_length=12)
    passage_role: PassageRole
    redundancy_group_id: str | None = Field(default=None, max_length=128)
    source_context_sufficient: bool | None = None


class RetrievalArmCost(BaseModel):
    """Measured local operational cost for one candidate retrieval."""

    model_config = ConfigDict(extra="forbid")

    query_latency_ms: float | None = Field(default=None, ge=0.0)
    index_build_ms: float | None = Field(default=None, ge=0.0)
    index_bytes: int | None = Field(default=None, ge=0)


class RetrievalArm(BaseModel):
    """A complete ordered retrieval result for one fixed candidate."""

    model_config = ConfigDict(extra="forbid")

    passages: list[RankedRetrievalPassage] = Field(
        min_length=1, max_length=MAX_ABLATION_PASSAGES
    )
    cost: RetrievalArmCost = Field(default_factory=RetrievalArmCost)

    @model_validator(mode="after")
    def validate_order(self):
        ranks = [item.rank for item in self.passages]
        if ranks != list(range(1, len(self.passages) + 1)):
            raise ValueError("Passage ranks must be contiguous and ordered from one")
        passage_ids = [item.passage_id for item in self.passages]
        if len(passage_ids) != len(set(passage_ids)):
            raise ValueError("A retrieval arm cannot repeat a passage ID")
        return self


class RetrievalFacetGold(BaseModel):
    """Human-selected passages that bear materially on one exact facet."""

    model_config = ConfigDict(extra="forbid")

    facet_id: str = Field(min_length=1, max_length=128)
    useful_passage_ids: list[str] = Field(min_length=1, max_length=32)


class RetrievalAblationCase(BaseModel):
    """One source-separated, human-labelled retrieval comparison."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_group_id: str = Field(min_length=1, max_length=128)
    gold_review_scope: Literal["bounded_candidates", "complete_source"] = (
        "bounded_candidates"
    )
    gold_reviewer: str = Field(default="unspecified", min_length=1, max_length=80)
    evidence_labels: Literal["positive_only", "exhaustive"] = "positive_only"
    useful_passage_ids: list[str] = Field(min_length=1, max_length=64)
    facet_gold: list[RetrievalFacetGold] = Field(default_factory=list, max_length=24)
    baseline: RetrievalArm
    experiment: RetrievalArm | None = None

    @model_validator(mode="after")
    def validate_gold(self):
        useful_ids = set(self.useful_passage_ids)
        if len(useful_ids) != len(self.useful_passage_ids):
            raise ValueError("Useful passage IDs must be unique")
        facet_ids = [item.facet_id for item in self.facet_gold]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("Facet gold IDs must be unique within a case")
        for facet in self.facet_gold:
            if not set(facet.useful_passage_ids).issubset(useful_ids):
                raise ValueError("Facet gold passages must be in the case useful set")
        return self


class RetrievalAblationManifest(BaseModel):
    """Fixed development-only retrieval experiment and safety boundary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["protected-retrieval-ablation-v1"] = (
        RETRIEVAL_ABLATION_VERSION
    )
    partition: Literal["development"] = "development"
    holdout_included: Literal[False] = False
    protected_depth: int = Field(default=2, ge=1, le=2)
    safety_window: int = Field(
        default=DEFAULT_SAFETY_WINDOW, ge=2, le=MAX_ABLATION_PASSAGES
    )
    input_snapshots: list[RetrievalInputSnapshot] = Field(
        default_factory=list, max_length=12
    )
    limitations: list[str] = Field(default_factory=list, max_length=12)
    cases: list[RetrievalAblationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases_and_safety(self):
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Ablation case IDs must be unique")
        candidate_ids = [case.candidate_id for case in self.cases]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate IDs must be unique across the manifest")
        has_experiment = [case.experiment is not None for case in self.cases]
        if any(has_experiment) and not all(has_experiment):
            raise ValueError("An ablation cannot mix baseline-only and experiment cases")
        if all(has_experiment):
            for case in self.cases:
                self._validate_protected_passages(case)
        return self

    def _validate_protected_passages(self, case: RetrievalAblationCase) -> None:
        assert case.experiment is not None
        protected = {
            item.passage_id
            for item in case.baseline.passages[: self.protected_depth]
        }
        experiment_window = {
            item.passage_id: item
            for item in case.experiment.passages[: self.safety_window]
        }
        missing = protected - set(experiment_window)
        if missing:
            raise ValueError(
                "The experiment must retain every protected baseline passage "
                f"inside the safety window: {sorted(missing)}"
            )
        unmarked = [
            passage_id
            for passage_id in protected
            if "protected_baseline" not in experiment_window[passage_id].channels
        ]
        if unmarked:
            raise ValueError(
                "Protected baseline passages require an inspectable "
                f"protected_baseline channel: {sorted(unmarked)}"
            )


class RetrievalArmMetrics(BaseModel):
    """Capability-separated measurements for one fixed retrieval arm."""

    model_config = ConfigDict(extra="forbid")

    positive_cases: int = Field(ge=1)
    labelled_passages: int = Field(ge=1)
    material_facets: int = Field(ge=0)
    hit_rate_at_3: float = Field(ge=0.0, le=1.0)
    hit_rate_at_5: float = Field(ge=0.0, le=1.0)
    labelled_passage_recall_at_3: float = Field(ge=0.0, le=1.0)
    labelled_passage_recall_at_5: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    facet_coverage_at_3: float | None = Field(default=None, ge=0.0, le=1.0)
    facet_coverage_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    useful_precision_at_3: float | None = Field(default=None, ge=0.0, le=1.0)
    useful_precision_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    excluded_role_rate_at_5: float = Field(ge=0.0, le=1.0)
    redundant_slot_rate_at_5: float = Field(ge=0.0, le=1.0)
    source_context_sufficiency_rate_at_5: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    mean_query_latency_ms: float | None = Field(default=None, ge=0.0)
    mean_index_build_ms: float | None = Field(default=None, ge=0.0)
    mean_index_bytes: float | None = Field(default=None, ge=0.0)


class RetrievalContinuationCriteria(BaseModel):
    """Pre-registered gate for continuing an additive retrieval channel."""

    model_config = ConfigDict(extra="forbid")

    max_hit_rate_at_5_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_labelled_recall_at_5_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_facet_coverage_at_5_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_material_improvement: float = Field(default=0.01, ge=0.0, le=1.0)
    max_query_latency_ratio: float | None = Field(default=None, ge=1.0)
    max_mean_index_bytes: float | None = Field(default=None, ge=0.0)


class RetrievalAblationResult(BaseModel):
    """Shadow-only comparison; it can never apply a verification decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["protected-retrieval-ablation-result-v1"] = (
        "protected-retrieval-ablation-result-v1"
    )
    case_count: int = Field(ge=1)
    source_group_count: int = Field(ge=1)
    baseline: RetrievalArmMetrics
    experiment: RetrievalArmMetrics | None = None
    protected_passage_retention: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    decision: Literal["baseline_only", "continue", "stop", "inconclusive"]
    reason_codes: list[str] = Field(default_factory=list)
    decision_applied: Literal[False] = False
    limitations: list[str] = Field(default_factory=list)


def evaluate_retrieval_ablation(
    manifest: RetrievalAblationManifest,
    *,
    criteria: RetrievalContinuationCriteria | None = None,
) -> RetrievalAblationResult:
    """Measure baseline/experiment rankings without reading source text."""
    gate = criteria or RetrievalContinuationCriteria()
    baseline = _arm_metrics(manifest.cases, "baseline")
    if manifest.cases[0].experiment is None:
        return RetrievalAblationResult(
            case_count=len(manifest.cases),
            source_group_count=len(
                {case.source_group_id for case in manifest.cases}
            ),
            baseline=baseline,
            decision="baseline_only",
            reason_codes=["experiment_not_supplied"],
            limitations=_metric_limitations(manifest),
        )

    experiment = _arm_metrics(manifest.cases, "experiment")
    protected_retention = _protected_retention(manifest)
    stop_reasons: list[str] = []
    if protected_retention < 1.0:
        stop_reasons.append("protected_safety_passage_regression")
    if (
        baseline.hit_rate_at_5 - experiment.hit_rate_at_5
        > gate.max_hit_rate_at_5_drop
    ):
        stop_reasons.append("positive_case_recall_regression")
    if (
        baseline.labelled_passage_recall_at_5
        - experiment.labelled_passage_recall_at_5
        > gate.max_labelled_recall_at_5_drop
    ):
        stop_reasons.append("labelled_passage_recall_regression")
    if (
        baseline.facet_coverage_at_5 is not None
        and experiment.facet_coverage_at_5 is not None
        and baseline.facet_coverage_at_5 - experiment.facet_coverage_at_5
        > gate.max_facet_coverage_at_5_drop
    ):
        stop_reasons.append("material_facet_coverage_regression")

    if gate.max_query_latency_ratio is not None:
        if (
            baseline.mean_query_latency_ms is not None
            and experiment.mean_query_latency_ms is not None
            and baseline.mean_query_latency_ms > 0
            and experiment.mean_query_latency_ms / baseline.mean_query_latency_ms
            > gate.max_query_latency_ratio
        ):
            stop_reasons.append("query_latency_budget_exceeded")
    if (
        gate.max_mean_index_bytes is not None
        and experiment.mean_index_bytes is not None
        and experiment.mean_index_bytes > gate.max_mean_index_bytes
    ):
        stop_reasons.append("index_storage_budget_exceeded")
    if stop_reasons:
        return RetrievalAblationResult(
            case_count=len(manifest.cases),
            source_group_count=len(
                {case.source_group_id for case in manifest.cases}
            ),
            baseline=baseline,
            experiment=experiment,
            protected_passage_retention=protected_retention,
            decision="stop",
            reason_codes=stop_reasons,
            limitations=_metric_limitations(manifest),
        )

    improvements = _material_improvements(baseline, experiment, gate)
    return RetrievalAblationResult(
        case_count=len(manifest.cases),
        source_group_count=len({case.source_group_id for case in manifest.cases}),
        baseline=baseline,
        experiment=experiment,
        protected_passage_retention=protected_retention,
        decision="continue" if improvements else "inconclusive",
        reason_codes=improvements or ["no_measured_material_improvement"],
        limitations=_metric_limitations(manifest),
    )


def build_overlap_cleanup_experiment(
    manifest: RetrievalAblationManifest,
) -> RetrievalAblationManifest:
    """Add a deterministic overlap-cleanup arm without discovering new passages.

    The baseline's protected prefix is always retained.  Later passages that
    share an inspectable redundancy group with an already retained passage are
    skipped, and the remaining baseline ranking backfills the safety window.
    A separately measured discovery channel may fill any still-open slots in a
    later experiment; this function deliberately does not invent candidates.
    """
    if any(case.experiment is not None for case in manifest.cases):
        raise RetrievalAblationError(
            "Overlap cleanup requires a baseline-only manifest"
        )

    experiment_cases: list[RetrievalAblationCase] = []
    for case in manifest.cases:
        selected: list[RankedRetrievalPassage] = []
        selected_groups: set[str] = set()
        ordered = sorted(case.baseline.passages, key=lambda item: item.rank)
        for item in ordered:
            protected = item.rank <= manifest.protected_depth
            group_id = item.redundancy_group_id or item.passage_id
            if not protected and group_id in selected_groups:
                continue
            channels = list(item.channels)
            if "deterministic_overlap_cleanup" not in channels:
                channels.append("deterministic_overlap_cleanup")
            if protected and "protected_baseline" not in channels:
                channels.append("protected_baseline")
            selected.append(
                item.model_copy(
                    update={
                        "rank": len(selected) + 1,
                        "channels": channels,
                    }
                )
            )
            selected_groups.add(group_id)
            if len(selected) >= manifest.safety_window:
                break
        experiment_cases.append(
            case.model_copy(
                update={
                    "experiment": RetrievalArm(
                        passages=selected,
                        cost=case.baseline.cost.model_copy(),
                    )
                }
            )
        )

    return manifest.model_copy(
        update={
            "cases": experiment_cases,
            "limitations": list(
                dict.fromkeys(
                    [
                        *manifest.limitations,
                        "The overlap-cleanup arm adds no newly discovered passage; open top-five slots are measured before a local additive discovery channel is introduced.",
                    ]
                )
            ),
        }
    )


def build_protected_discovery_experiment(
    manifest: RetrievalAblationManifest,
    *,
    discoveries: dict[str, list[RankedRetrievalPassage]],
    costs: dict[str, RetrievalArmCost] | None = None,
    discovery_slots: int = 2,
) -> RetrievalAblationManifest:
    """Add bounded discovery passages while preserving the baseline safety set.

    The first two baseline passages remain protected.  At most two discovered
    body passages are admitted next, and the old ranking then backfills the
    top-five window.  This guarantees only the configured protected depth
    (currently the baseline top two) inside the top-five safety window; later
    baseline ranks may move or be displaced by the measured discovery arm.
    """
    if any(case.experiment is not None for case in manifest.cases):
        raise RetrievalAblationError(
            "Protected discovery requires a baseline-only manifest"
        )
    if discovery_slots < 1 or discovery_slots > (
        manifest.safety_window - manifest.protected_depth
    ):
        raise RetrievalAblationError("Discovery slots exceed the safety window")
    case_ids = {case.case_id for case in manifest.cases}
    if set(discoveries) != case_ids:
        raise RetrievalAblationError(
            "Discovery rankings must be supplied for every and only manifest case"
        )
    if costs is not None and set(costs) != case_ids:
        raise RetrievalAblationError(
            "Discovery costs must be supplied for every and only manifest case"
        )

    experiment_cases: list[RetrievalAblationCase] = []
    for case in manifest.cases:
        baseline = sorted(case.baseline.passages, key=lambda item: item.rank)
        selected: list[RankedRetrievalPassage] = []
        selected_ids: set[str] = set()
        selected_groups: set[str] = set()

        def retain(item: RankedRetrievalPassage, *, channel: str) -> bool:
            if item.passage_id in selected_ids:
                return False
            group_id = item.redundancy_group_id or item.passage_id
            if group_id in selected_groups:
                return False
            channels = list(dict.fromkeys([*item.channels, channel]))
            selected.append(
                item.model_copy(
                    update={"rank": len(selected) + 1, "channels": channels}
                )
            )
            selected_ids.add(item.passage_id)
            selected_groups.add(group_id)
            return True

        for item in baseline[: manifest.protected_depth]:
            # Protected passages cannot be removed even if their deterministic
            # overlap groups collide with one another.
            channels = list(dict.fromkeys([*item.channels, "protected_baseline"]))
            selected.append(
                item.model_copy(
                    update={"rank": len(selected) + 1, "channels": channels}
                )
            )
            selected_ids.add(item.passage_id)
            selected_groups.add(item.redundancy_group_id or item.passage_id)

        added = 0
        for item in discoveries[case.case_id]:
            if item.passage_role in _NOISE_ROLES:
                continue
            if retain(item, channel="bge_m3_hybrid_discovery"):
                added += 1
            if added >= discovery_slots:
                break

        for item in baseline[manifest.protected_depth :]:
            if len(selected) >= manifest.safety_window:
                break
            retain(item, channel="baseline_backfill")

        experiment_cases.append(
            case.model_copy(
                update={
                    "experiment": RetrievalArm(
                        passages=selected,
                        cost=(costs or {}).get(case.case_id, RetrievalArmCost()),
                    )
                }
            )
        )
    return manifest.model_copy(update={"cases": experiment_cases})


def _arm_metrics(
    cases: list[RetrievalAblationCase], arm_name: Literal["baseline", "experiment"]
) -> RetrievalArmMetrics:
    arms = []
    for case in cases:
        arm = case.baseline if arm_name == "baseline" else case.experiment
        if arm is None:
            raise RetrievalAblationError("Every case must supply the requested arm")
        arms.append(arm)

    useful_total = sum(len(case.useful_passage_ids) for case in cases)
    reciprocal_ranks: list[float] = []
    hits: dict[int, int] = {3: 0, 5: 0}
    useful_found: dict[int, int] = {3: 0, 5: 0}
    facet_total = sum(len(case.facet_gold) for case in cases)
    facets_found: dict[int, int] = {3: 0, 5: 0}
    selected_total: dict[int, int] = {3: 0, 5: 0}
    selected_useful: dict[int, int] = {3: 0, 5: 0}
    noise_slots = 0
    redundant_slots = 0
    top_five_slots = 0
    context_values: list[bool] = []

    for case, arm in zip(cases, arms, strict=True):
        useful = set(case.useful_passage_ids)
        ordered = sorted(arm.passages, key=lambda item: item.rank)
        first_rank = next(
            (item.rank for item in ordered if item.passage_id in useful), None
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in (3, 5):
            selected = ordered[:k]
            selected_ids = {item.passage_id for item in selected}
            hits[k] += bool(selected_ids & useful)
            useful_found[k] += len(selected_ids & useful)
            selected_total[k] += len(selected)
            selected_useful[k] += len(selected_ids & useful)
            for facet in case.facet_gold:
                facets_found[k] += bool(
                    selected_ids & set(facet.useful_passage_ids)
                )

        top_five = ordered[:5]
        top_five_slots += len(top_five)
        noise_slots += sum(item.passage_role in _NOISE_ROLES for item in top_five)
        groups = [item.redundancy_group_id or item.passage_id for item in top_five]
        redundant_slots += len(groups) - len(set(groups))
        context_values.extend(
            item.source_context_sufficient
            for item in top_five
            if item.source_context_sufficient is not None
        )

    exhaustive = all(case.evidence_labels == "exhaustive" for case in cases)
    return RetrievalArmMetrics(
        positive_cases=len(cases),
        labelled_passages=useful_total,
        material_facets=facet_total,
        hit_rate_at_3=hits[3] / len(cases),
        hit_rate_at_5=hits[5] / len(cases),
        labelled_passage_recall_at_3=useful_found[3] / useful_total,
        labelled_passage_recall_at_5=useful_found[5] / useful_total,
        mean_reciprocal_rank=mean(reciprocal_ranks),
        facet_coverage_at_3=(facets_found[3] / facet_total if facet_total else None),
        facet_coverage_at_5=(facets_found[5] / facet_total if facet_total else None),
        useful_precision_at_3=(
            selected_useful[3] / selected_total[3]
            if exhaustive and selected_total[3]
            else None
        ),
        useful_precision_at_5=(
            selected_useful[5] / selected_total[5]
            if exhaustive and selected_total[5]
            else None
        ),
        excluded_role_rate_at_5=(noise_slots / top_five_slots if top_five_slots else 0.0),
        redundant_slot_rate_at_5=(
            redundant_slots / top_five_slots if top_five_slots else 0.0
        ),
        source_context_sufficiency_rate_at_5=(
            sum(context_values) / len(context_values) if context_values else None
        ),
        mean_query_latency_ms=_mean_optional(
            [arm.cost.query_latency_ms for arm in arms]
        ),
        mean_index_build_ms=_mean_optional(
            [arm.cost.index_build_ms for arm in arms]
        ),
        mean_index_bytes=_mean_optional([arm.cost.index_bytes for arm in arms]),
    )


def _protected_retention(manifest: RetrievalAblationManifest) -> float:
    retained = 0
    total = 0
    for case in manifest.cases:
        assert case.experiment is not None
        protected = {
            item.passage_id
            for item in case.baseline.passages[: manifest.protected_depth]
        }
        experiment = {
            item.passage_id
            for item in case.experiment.passages[: manifest.safety_window]
        }
        retained += len(protected & experiment)
        total += len(protected)
    return retained / total if total else 1.0


def _material_improvements(
    baseline: RetrievalArmMetrics,
    experiment: RetrievalArmMetrics,
    gate: RetrievalContinuationCriteria,
) -> list[str]:
    threshold = gate.minimum_material_improvement
    improvements = []
    if experiment.hit_rate_at_3 - baseline.hit_rate_at_3 >= threshold:
        improvements.append("positive_hit_rate_at_3_improved")
    if (
        experiment.labelled_passage_recall_at_5
        - baseline.labelled_passage_recall_at_5
        >= threshold
    ):
        improvements.append("labelled_passage_recall_at_5_improved")
    if experiment.mean_reciprocal_rank - baseline.mean_reciprocal_rank >= threshold:
        improvements.append("mean_reciprocal_rank_improved")
    if (
        baseline.facet_coverage_at_3 is not None
        and experiment.facet_coverage_at_3 is not None
        and experiment.facet_coverage_at_3 - baseline.facet_coverage_at_3
        >= threshold
    ):
        improvements.append("material_facet_coverage_at_3_improved")
    if (
        baseline.excluded_role_rate_at_5 - experiment.excluded_role_rate_at_5
        >= threshold
    ):
        improvements.append("excluded_role_noise_reduced")
    if (
        baseline.redundant_slot_rate_at_5 - experiment.redundant_slot_rate_at_5
        >= threshold
    ):
        improvements.append("redundant_slots_reduced")
    return improvements


def _metric_limitations(manifest: RetrievalAblationManifest) -> list[str]:
    cases = manifest.cases
    limitations = [
        "The ablation is shadow-only and cannot change a citation relationship or report outcome.",
        "Only exact authorized passage IDs and human evidence labels determine retrieval metrics.",
        *manifest.limitations,
    ]
    if any(case.evidence_labels == "positive_only" for case in cases):
        limitations.append(
            "At least one case has positive-only labels; unselected passages are not proven irrelevant, so useful-passage precision is not reported."
        )
    return list(dict.fromkeys(limitations))


def _mean_optional(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None
