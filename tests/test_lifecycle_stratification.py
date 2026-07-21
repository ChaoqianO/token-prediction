from __future__ import annotations

import hashlib
import json
import unittest

from token_prediction.dataset import (
    LIFECYCLE_SCHEMA_VERSION,
    LabelStatus,
    LifecycleSequence,
    LifecycleStep,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    lifecycle_scored_hash,
    point_input_semantic,
)
from token_prediction.estimators import ObservedTransition, SessionSeed, TokenForecast
from token_prediction.evaluation import (
    evaluate_progress_checkpoints,
    evaluate_same_task_run_variance,
    evaluate_termination_strata,
)
from token_prediction.lifecycle import LifecyclePrediction, LifecycleRun


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
CONTRACT = "a" * 64


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run(
    task_id: str,
    run_id: str,
    *,
    labels: tuple[int | None, ...],
    statuses: tuple[LabelStatus, ...] | None = None,
    reasons: tuple[str, ...] | None = None,
    predictions: tuple[float, ...] | None = None,
    progress_features: tuple[float, ...] | None = None,
) -> LifecycleRun:
    statuses = statuses or tuple(
        LabelStatus.OBSERVED if value is not None else LabelStatus.MISSING
        for value in labels
    )
    reasons = reasons or tuple("" if value is not None else "missing_usage" for value in labels)
    predictions = predictions or tuple(float(value or 0) for value in labels)
    progress_features = progress_features or tuple(99.0 - index for index in range(len(labels)))
    trajectory_id = f"{task_id}:{run_id}:trajectory"
    points = tuple(
        PredictionPoint(
            point_id=f"{trajectory_id}:point:{index}",
            source_event_id=f"{trajectory_id}:event:{index}",
            task_id=task_id,
            trajectory_id=trajectory_id,
            run_id=run_id,
            prediction_context_id=f"{trajectory_id}:context:{index}",
            condition_id="condition:test",
            logical_call_id=f"{trajectory_id}:call:{index}",
            attempt_id=None,
            cutoff_event_seq=index + 1,
            position=(
                PredictionPosition.TASK_PRE
                if index == 0
                else PredictionPosition.TASK_UPDATE
            ),
            target=TARGET,
            features={
                "step_progress_ratio": (
                    -1.0 if index == 0 else progress_features[index - 1]
                ),
                "missing_usage_attempts": 0,
                "cumulative_provider_input_tokens": index,
                "cumulative_provider_output_tokens": 0,
            },
            known_offset_tokens=0,
        )
        for index in range(len(labels) + 1)
    )
    steps = (
        LifecycleStep(
            points[0],
            None,
            LabelStatus.MISSING,
            "redacted_task_pre_label",
            False,
            False,
            0.0,
        ),
        *(
            LifecycleStep(
                points[index + 1],
                label,
                statuses[index],
                reasons[index],
                label is not None,
                label is not None,
                1.0 / sum(value is not None for value in labels) if label is not None else 0.0,
            )
            for index, label in enumerate(labels)
        ),
    )
    context_hash = _hash(
        {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "input_contract_hash": CONTRACT,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "run_id": run_id,
            "condition_id": "condition:test",
            "target": TARGET.value,
            "points": [point_input_semantic(point) for point in points],
        }
    )
    sequence = LifecycleSequence(
        dataset_id="dataset",
        input_contract_hash=CONTRACT,
        task_id=task_id,
        trajectory_id=trajectory_id,
        run_id=run_id,
        condition_id="condition:test",
        target=TARGET,
        steps=steps,
        context_hash=context_hash,
        scored_hash=lifecycle_scored_hash(context_hash, steps),
    )
    seed = SessionSeed(
        task_pre_point=points[0],
        forecast=TokenForecast(
            points[0].point_id,
            TARGET,
            0,
            1,
            2,
            raw_lower=0,
            raw_point=1,
            raw_upper=2,
        ),
        initializer_id="empirical_quantile",
        initializer_hash="b" * 64,
        inner_split_id="inner",
        component_bundle_hashes=("c" * 64,),
        seed_policy_id="policy",
        seed_policy_hash="d" * 64,
    )
    lifecycle_predictions = tuple(
        LifecyclePrediction(
            step=steps[index + 1],
            forecast=TokenForecast(
                points[index + 1].point_id,
                TARGET,
                max(0.0, prediction - 1),
                prediction,
                prediction + 1,
                latency_ms=float(index + 1),
            ),
            transition=ObservedTransition(
                points[index].point_id,
                points[index + 1].point_id,
                1,
            ),
        )
        for index, prediction in enumerate(predictions)
    )
    return LifecycleRun(sequence, "offline", seed, lifecycle_predictions)


class LifecycleStratificationTests(unittest.TestCase):
    def test_progress_is_derived_from_sequence_order_and_preserves_unscored_context(self) -> None:
        run = _run(
            "task-a",
            "run-a",
            labels=(10, None, 30, 40),
            predictions=(11, 999, 33, 44),
            progress_features=(0.99, 0.01, 0.50, -20.0),
        )
        report = evaluate_progress_checkpoints((run,))
        self.assertEqual(
            report["selection_policy"],
            "first_boundary_at_or_after_sequence_fraction_v1",
        )
        self.assertEqual(report["strata"]["p25"]["metrics"]["mae"], 1.0)
        self.assertEqual(report["strata"]["p50"]["n_scored"], 0)
        self.assertIsNone(report["strata"]["p50"]["metrics"])
        self.assertEqual(report["strata"]["p75"]["metrics"]["mae"], 3.0)

    def test_censored_termination_has_counts_but_no_fabricated_metrics(self) -> None:
        observed = _run("task-a", "run-a", labels=(10, 5))
        censored = _run(
            "task-b",
            "run-b",
            labels=(None, None),
            statuses=(LabelStatus.CENSORED, LabelStatus.CENSORED),
            reasons=("step_limit", "step_limit"),
            predictions=(20, 10),
        )
        report = evaluate_termination_strata((observed, censored))
        self.assertEqual(report["strata"]["censored:step_limit"]["n_sequences"], 1)
        self.assertEqual(report["strata"]["censored:step_limit"]["n_scored"], 0)
        self.assertIsNone(report["strata"]["censored:step_limit"]["metrics"])
        self.assertEqual(report["strata"]["observed_termination"]["n_scored"], 2)

    def test_same_task_run_variance_uses_run_level_mae(self) -> None:
        runs = (
            _run("task-a", "run-1", labels=(10,), predictions=(10,)),
            _run("task-a", "run-2", labels=(10,), predictions=(12,)),
            _run("task-b", "run-1", labels=(20,), predictions=(25,)),
        )
        report = evaluate_same_task_run_variance(runs)
        self.assertEqual(report["n_repeated_tasks"], 1)
        self.assertEqual(report["status"], "estimable")
        self.assertEqual(report["mean_within_task_run_mae_variance"], 1.0)


if __name__ == "__main__":
    unittest.main()
