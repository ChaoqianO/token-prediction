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
from token_prediction.estimators import SessionSeed, TokenForecast
from token_prediction.estimators.cross_position_deduct import FittedCrossPositionDeduct
from token_prediction.lifecycle import run_lifecycle_sequence, visible_spend_delta


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
CONTRACT = "a" * 64


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _point(index: int, *, missing: int = 0, known: int = 0) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"point-{index}",
        source_event_id=f"event-{index}",
        task_id="task",
        trajectory_id="trajectory",
        run_id="run",
        prediction_context_id=f"context-{index}",
        condition_id="condition:a",
        logical_call_id=f"call-{index}",
        attempt_id=None,
        cutoff_event_seq=index + 1,
        position=(
            PredictionPosition.TASK_PRE
            if index == 0
            else PredictionPosition.TASK_UPDATE
        ),
        target=TARGET,
        features={
            "missing_usage_attempts": missing,
            "cumulative_provider_input_tokens": known,
            "cumulative_provider_output_tokens": 0,
        },
        known_offset_tokens=0,
    )


def _sequence() -> LifecycleSequence:
    points = (_point(0), _point(1, known=10), _point(2, known=20))
    steps = (
        LifecycleStep(points[0], 20, LabelStatus.OBSERVED, "", False, False, 0.0),
        LifecycleStep(points[1], None, LabelStatus.MISSING, "missing", False, False, 0.0),
        LifecycleStep(points[2], 0, LabelStatus.OBSERVED, "", True, True, 1.0),
    )
    context_hash = _hash(
        {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "input_contract_hash": CONTRACT,
            "task_id": "task",
            "trajectory_id": "trajectory",
            "run_id": "run",
            "condition_id": "condition:a",
            "target": TARGET.value,
            "points": [point_input_semantic(point) for point in points],
        }
    )
    scored_hash = lifecycle_scored_hash(context_hash, steps)
    return LifecycleSequence(
        dataset_id="dataset",
        input_contract_hash=CONTRACT,
        task_id="task",
        trajectory_id="trajectory",
        run_id="run",
        condition_id="condition:a",
        target=TARGET,
        steps=steps,
        context_hash=context_hash,
        scored_hash=scored_hash,
    )


def _seed(point: PredictionPoint) -> SessionSeed:
    return SessionSeed(
        task_pre_point=point,
        forecast=TokenForecast(
            point.point_id,
            point.target,
            10,
            20,
            30,
            raw_lower=10,
            raw_point=20,
            raw_upper=30,
        ),
        initializer_id="empirical_quantile",
        initializer_hash="b" * 64,
        inner_split_id="inner",
        component_bundle_hashes=("c" * 64,),
        seed_policy_id="policy",
        seed_policy_hash="d" * 64,
    )


class LifecycleDriverTests(unittest.TestCase):
    def test_unscored_context_is_observed_and_offline_shadow_order_matches(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            "dataset",
            TARGET,
            "condition:a",
            CONTRACT,
        )
        seed = _seed(sequence.steps[0].point)
        offline = run_lifecycle_sequence(fitted, sequence, seed, runtime_mode="offline")
        shadow = run_lifecycle_sequence(fitted, sequence, seed, runtime_mode="shadow")
        self.assertEqual(
            [item.step.point.point_id for item in offline.predictions],
            ["point-1", "point-2"],
        )
        self.assertEqual(len(offline.scored_predictions), 1)
        self.assertEqual(
            [item.forecast for item in offline.predictions],
            [item.forecast for item in shadow.predictions],
        )

    def test_missing_counter_growth_blocks_only_that_transition(self) -> None:
        first = _point(0, missing=0, known=0)
        polluted = _point(1, missing=1, known=10)
        recovered = _point(2, missing=1, known=25)
        self.assertIsNone(visible_spend_delta(first, polluted))
        self.assertEqual(visible_spend_delta(polluted, recovered), 15)


if __name__ == "__main__":
    unittest.main()
