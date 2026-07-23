from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from token_prediction.experiment import CandidateResult


STAGE4_MATCHED_COVERAGE_POLICY_ID = "stage4_matched_task_coverage_v1"
STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID = "stage4_same_task_condition_mean_v1"


def _calibration_comparability_key(result: CandidateResult) -> tuple[object, ...]:
    return (
        result.dataset_id,
        result.split_plan_id,
        result.eligibility_hash,
        result.position,
        result.target,
        result.condition_id,
        result.alpha,
        result.metric_suite_id,
    )


def assert_calibration_raw_prediction_parity(
    reference: CandidateResult,
    candidate: CandidateResult,
) -> None:
    if _calibration_comparability_key(reference) != _calibration_comparability_key(
        candidate
    ):
        raise ValueError("calibration results differ outside the calibrator axis")
    if (
        reference.candidate_id != candidate.candidate_id
        or reference.candidate_hash != candidate.candidate_hash
    ):
        raise ValueError("calibration results use different model candidates")
    reference_by_point = {
        record.point_id: record for record in reference.predictions
    }
    candidate_by_point = {
        record.point_id: record for record in candidate.predictions
    }
    if set(reference_by_point) != set(candidate_by_point):
        raise ValueError("calibration results use different prediction cohorts")
    for point_id in sorted(reference_by_point):
        left = reference_by_point[point_id]
        right = candidate_by_point[point_id]
        if (
            left.task_id != right.task_id
            or left.trajectory_id != right.trajectory_id
            or left.fold != right.fold
            or left.target != right.target
            or left.sample_weight != right.sample_weight
        ):
            raise ValueError("calibration results changed prediction provenance")
        left_raw = (
            left.forecast.raw_lower,
            left.forecast.raw_point,
            left.forecast.raw_upper,
        )
        right_raw = (
            right.forecast.raw_lower,
            right.forecast.raw_point,
            right.forecast.raw_upper,
        )
        if left_raw != right_raw:
            raise ValueError("calibration ablation changed raw predictions")


@dataclass(frozen=True)
class MatchedCoverageComparison:
    reference_id: str
    candidate_id: str
    coverage_metric: str
    interval_score_metric: str
    reference_coverage: float
    candidate_coverage: float
    reference_interval_score: float
    candidate_interval_score: float
    tolerance: float
    matched: bool
    winner: str | None
    policy_id: str = STAGE4_MATCHED_COVERAGE_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != STAGE4_MATCHED_COVERAGE_POLICY_ID:
            raise ValueError("unsupported matched-coverage policy")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("coverage tolerance must be finite and non-negative")
        values = (
            self.reference_coverage,
            self.candidate_coverage,
            self.reference_interval_score,
            self.candidate_interval_score,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("matched-coverage metrics must be finite and non-negative")
        if self.matched != (
            abs(self.reference_coverage - self.candidate_coverage) <= self.tolerance
        ):
            raise ValueError("matched-coverage status disagrees with achieved coverage")
        expected_winner = None
        if self.matched:
            if self.candidate_interval_score < self.reference_interval_score:
                expected_winner = self.candidate_id
            elif self.candidate_interval_score > self.reference_interval_score:
                expected_winner = self.reference_id
            else:
                expected_winner = "tie"
        if self.winner != expected_winner:
            raise ValueError("matched-coverage winner is invalid")


def compare_matched_coverage(
    reference: CandidateResult,
    candidate: CandidateResult,
    *,
    coverage_metric: str = "task_simultaneous_coverage",
    interval_score_metric: str = "interval_score",
    tolerance: float = 0.02,
) -> MatchedCoverageComparison:
    assert_calibration_raw_prediction_parity(reference, candidate)
    try:
        reference_coverage = float(reference.metrics[coverage_metric])
        candidate_coverage = float(candidate.metrics[coverage_metric])
        reference_score = float(reference.metrics[interval_score_metric])
        candidate_score = float(candidate.metrics[interval_score_metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("matched-coverage metrics are missing or non-numeric") from exc
    matched = abs(reference_coverage - candidate_coverage) <= tolerance
    winner: str | None = None
    if matched:
        if candidate_score < reference_score:
            winner = candidate.calibrator_id
        elif candidate_score > reference_score:
            winner = reference.calibrator_id
        else:
            winner = "tie"
    return MatchedCoverageComparison(
        reference_id=reference.calibrator_id,
        candidate_id=candidate.calibrator_id,
        coverage_metric=coverage_metric,
        interval_score_metric=interval_score_metric,
        reference_coverage=reference_coverage,
        candidate_coverage=candidate_coverage,
        reference_interval_score=reference_score,
        candidate_interval_score=candidate_score,
        tolerance=tolerance,
        matched=matched,
        winner=winner,
    )


@dataclass(frozen=True)
class CrossConditionPairedComparison:
    candidate_id: str
    reference_id: str
    condition_count: int
    task_count: int
    mean_mae_delta: float
    mae_delta_ci_lower: float
    mae_delta_ci_upper: float
    candidate_win_probability: float
    iterations: int
    seed: int
    policy_id: str = STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID:
            raise ValueError("unsupported Stage 4 pairing policy")
        if self.condition_count < 1 or self.task_count < 2:
            raise ValueError("paired analysis requires conditions and at least two tasks")
        if self.iterations < 1 or self.seed < 0:
            raise ValueError("paired bootstrap iterations and seed are invalid")
        values = (
            self.mean_mae_delta,
            self.mae_delta_ci_lower,
            self.mae_delta_ci_upper,
            self.candidate_win_probability,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paired analysis contains non-finite values")
        if not 0 <= self.candidate_win_probability <= 1:
            raise ValueError("candidate win probability must be in [0, 1]")


def compare_same_tasks_across_conditions(
    pairs: Mapping[str, tuple[CandidateResult, CandidateResult]],
    *,
    iterations: int = 10_000,
    seed: int,
) -> CrossConditionPairedComparison:
    if not pairs:
        raise ValueError("cross-condition pairing input is empty")
    if iterations < 1 or seed < 0:
        raise ValueError("paired bootstrap iterations and seed are invalid")
    candidate_ids = {candidate.candidate_id for candidate, _reference in pairs.values()}
    reference_ids = {reference.candidate_id for _candidate, reference in pairs.values()}
    if len(candidate_ids) != 1 or len(reference_ids) != 1:
        raise ValueError("cross-condition pairing candidates are inconsistent")

    validated: list[tuple[CandidateResult, CandidateResult, set[str]]] = []
    for condition_id, (candidate, reference) in sorted(pairs.items()):
        if (
            candidate.dataset_id != reference.dataset_id
            or candidate.split_plan_id != reference.split_plan_id
            or candidate.eligibility_hash != reference.eligibility_hash
            or candidate.position != reference.position
            or candidate.target != reference.target
            or candidate.condition_id != condition_id
            or reference.condition_id != condition_id
            or candidate.calibrator_id != reference.calibrator_id
            or candidate.alpha != reference.alpha
            or candidate.metric_suite_id != reference.metric_suite_id
        ):
            raise ValueError("cross-condition pair differs outside candidate or feature path")
        shared_tasks = set(candidate.task_metrics) & set(reference.task_metrics)
        if shared_tasks != set(candidate.task_metrics) or shared_tasks != set(
            reference.task_metrics
        ):
            raise ValueError("cross-condition candidate task cohorts differ")
        validated.append((candidate, reference, shared_tasks))

    matched_tasks = set.intersection(*(tasks for _candidate, _reference, tasks in validated))
    if len(matched_tasks) < 2:
        raise ValueError("cross-condition pairing has fewer than two matched tasks")
    task_deltas: dict[str, list[float]] = {}
    for candidate, reference, _shared_tasks in validated:
        for task_id in matched_tasks:
            try:
                candidate_mae = float(candidate.task_metrics[task_id]["weighted_mae"])
                reference_mae = float(reference.task_metrics[task_id]["weighted_mae"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("task metrics lack weighted_mae") from exc
            task_deltas.setdefault(task_id, []).append(candidate_mae - reference_mae)

    collapsed = [
        sum(values) / len(values)
        for _task_id, values in sorted(task_deltas.items())
    ]
    observed = sum(collapsed) / len(collapsed)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(collapsed[rng.randrange(len(collapsed))] for _ in collapsed)
        / len(collapsed)
        for _iteration in range(iterations)
    )
    lower_index = max(0, math.ceil(iterations * 0.025) - 1)
    upper_index = min(iterations - 1, math.ceil(iterations * 0.975) - 1)
    return CrossConditionPairedComparison(
        candidate_id=next(iter(candidate_ids)),
        reference_id=next(iter(reference_ids)),
        condition_count=len(pairs),
        task_count=len(collapsed),
        mean_mae_delta=observed,
        mae_delta_ci_lower=bootstrap[lower_index],
        mae_delta_ci_upper=bootstrap[upper_index],
        candidate_win_probability=(
            sum(value < 0 for value in bootstrap) / len(bootstrap)
        ),
        iterations=iterations,
        seed=seed,
    )


__all__ = [
    "STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID",
    "STAGE4_MATCHED_COVERAGE_POLICY_ID",
    "CrossConditionPairedComparison",
    "MatchedCoverageComparison",
    "assert_calibration_raw_prediction_parity",
    "compare_matched_coverage",
    "compare_same_tasks_across_conditions",
]
