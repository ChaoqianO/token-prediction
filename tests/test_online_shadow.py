from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

from token_prediction.contracts import Observable, SourceCapabilities
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
from token_prediction.dataset.points import (
    prediction_input_contract_hash_from_capability,
)
from token_prediction.estimators import SessionSeed, TokenForecast
from token_prediction.estimators.cross_position_deduct import FittedCrossPositionDeduct
from token_prediction.lifecycle import run_lifecycle_sequence
from token_prediction.online_shadow import (
    OnlineShadowProvenance,
    OnlineShadowSession,
    fitted_model_provenance_hash,
)
from token_prediction.telemetry import TelemetryCapabilityError


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS


def _capabilities(source_id: str = "source") -> SourceCapabilities:
    return SourceCapabilities(
        source_id,
        frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
            }
        ),
    )


CAPABILITIES = _capabilities()
CONTRACT = prediction_input_contract_hash_from_capability(
    capability_contract_hash=CAPABILITIES.contract_hash
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _point(index: int, known: int = 0) -> PredictionPoint:
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
            "missing_usage_attempts": 0,
            "cumulative_provider_input_tokens": known,
            "cumulative_provider_output_tokens": 0,
        },
        known_offset_tokens=0,
    )


def _sequence() -> LifecycleSequence:
    points = (_point(0), _point(1, 10), _point(2, 25))
    steps = (
        LifecycleStep(points[0], 30, LabelStatus.OBSERVED, "", False, False, 0.0),
        LifecycleStep(points[1], 20, LabelStatus.OBSERVED, "", True, True, 0.5),
        LifecycleStep(points[2], 5, LabelStatus.OBSERVED, "", True, True, 0.5),
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
        scored_hash=lifecycle_scored_hash(context_hash, steps),
    )


def _seed(point: PredictionPoint) -> SessionSeed:
    return SessionSeed(
        task_pre_point=point,
        forecast=TokenForecast(
            point.point_id,
            point.target,
            20,
            30,
            40,
            raw_lower=20,
            raw_point=30,
            raw_upper=40,
        ),
        initializer_id="empirical_quantile",
        initializer_hash="b" * 64,
        inner_split_id="inner",
        component_bundle_hashes=("c" * 64,),
        seed_policy_id="policy",
        seed_policy_hash="d" * 64,
    )


def _provenance(
    fitted: FittedCrossPositionDeduct,
    seed: SessionSeed,
    *,
    capabilities: SourceCapabilities = CAPABILITIES,
) -> OnlineShadowProvenance:
    return OnlineShadowProvenance(
        source_id=capabilities.source_id,
        capability_contract_hash=capabilities.contract_hash,
        dataset_id=fitted.dataset_id,
        input_contract_hash=fitted.input_contract_hash,
        condition_id=fitted.condition_id,
        target=fitted.target,
        estimator_id=fitted.estimator_id,
        fitted_model_provenance_hash=fitted_model_provenance_hash(fitted),
        seed_content_hash=seed.content_hash,
        seed_component_bundle_hashes=seed.component_bundle_hashes,
    )


class OnlineShadowTests(unittest.TestCase):
    def test_incremental_shadow_uses_the_exact_offline_driver_order(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        offline = run_lifecycle_sequence(fitted, sequence, seed)
        emitted = []
        shadow = OnlineShadowSession(
            fitted,
            capabilities=CAPABILITIES,
            provenance=_provenance(fitted, seed),
            dataset_id=sequence.dataset_id,
            input_contract_hash=sequence.input_contract_hash,
            condition_id=sequence.condition_id,
            task_pre_point=sequence.steps[0].point,
            seed=seed,
            sink=emitted.append,
        )
        actual = [
            shadow.observe_boundary(step.point)
            for step in sequence.steps[1:]
        ]
        self.assertEqual([item.ordinal for item in actual], [1, 2])
        self.assertEqual(emitted, actual)
        self.assertEqual(
            [
                replace(item.forecast, latency_ms=0.0)
                for item in offline.predictions
            ],
            [
                replace(item.forecast, latency_ms=0.0)
                for item in actual
            ],
        )
        self.assertEqual(
            [item.transition for item in offline.predictions],
            [item.transition for item in actual],
        )

    def test_online_shadow_fails_before_start_when_telemetry_is_incomplete(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        with self.assertRaises(TelemetryCapabilityError):
            OnlineShadowSession(
                fitted,
                capabilities=SourceCapabilities(
                    "source",
                    frozenset({Observable.REQUEST_BOUNDARIES}),
                ),
                provenance=_provenance(fitted, _seed(sequence.steps[0].point)),
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=_seed(sequence.steps[0].point),
            )

    def test_online_shadow_rejects_capabilities_from_another_source(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        with self.assertRaisesRegex(ValueError, "source_id"):
            OnlineShadowSession(
                fitted,
                capabilities=_capabilities("other-source"),
                provenance=_provenance(fitted, seed),
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )

    def test_online_shadow_rejects_wrong_capability_contract(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        changed = SourceCapabilities(
            CAPABILITIES.source_id,
            CAPABILITIES.observables | {Observable.TASK_TERMINATION},
        )
        with self.assertRaisesRegex(ValueError, "capability contract"):
            OnlineShadowSession(
                fitted,
                capabilities=changed,
                provenance=_provenance(fitted, seed),
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )

    def test_online_shadow_rejects_wrong_dataset_or_input_contract(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        provenance = _provenance(fitted, seed)
        with self.assertRaisesRegex(ValueError, "dataset"):
            OnlineShadowSession(
                fitted,
                capabilities=CAPABILITIES,
                provenance=provenance,
                dataset_id="other-dataset",
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )
        with self.assertRaisesRegex(ValueError, "input contract"):
            OnlineShadowSession(
                fitted,
                capabilities=CAPABILITIES,
                provenance=provenance,
                dataset_id=sequence.dataset_id,
                input_contract_hash="f" * 64,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )

    def test_online_shadow_rejects_wrong_target_provenance(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        provenance = replace(
            _provenance(fitted, seed),
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        )
        with self.assertRaisesRegex(ValueError, "target"):
            OnlineShadowSession(
                fitted,
                capabilities=CAPABILITIES,
                provenance=provenance,
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )

    def test_online_shadow_rejects_wrong_model_or_seed_bundle_identity(self) -> None:
        sequence = _sequence()
        fitted = FittedCrossPositionDeduct(
            "cross_position_deduct",
            sequence.dataset_id,
            TARGET,
            sequence.condition_id,
            sequence.input_contract_hash,
        )
        seed = _seed(sequence.steps[0].point)
        provenance = _provenance(fitted, seed)
        with self.assertRaisesRegex(ValueError, "fitted model provenance"):
            OnlineShadowSession(
                fitted,
                capabilities=CAPABILITIES,
                provenance=replace(
                    provenance,
                    fitted_model_provenance_hash="f" * 64,
                ),
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )
        with self.assertRaisesRegex(ValueError, "seed bundle identity"):
            OnlineShadowSession(
                fitted,
                capabilities=CAPABILITIES,
                provenance=replace(
                    provenance,
                    seed_component_bundle_hashes=("e" * 64,),
                ),
                dataset_id=sequence.dataset_id,
                input_contract_hash=sequence.input_contract_hash,
                condition_id=sequence.condition_id,
                task_pre_point=sequence.steps[0].point,
                seed=seed,
            )
        changed_seeds = {
            "forecast": replace(
                seed,
                forecast=TokenForecast(
                    seed.forecast.point_id,
                    seed.forecast.target,
                    21,
                    31,
                    41,
                    raw_lower=21,
                    raw_point=31,
                    raw_upper=41,
                ),
            ),
            "initializer": replace(seed, initializer_hash="e" * 64),
            "seed-policy": replace(seed, seed_policy_hash="f" * 64),
        }
        for name, changed_seed in changed_seeds.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "complete session seed"):
                    OnlineShadowSession(
                        fitted,
                        capabilities=CAPABILITIES,
                        provenance=provenance,
                        dataset_id=sequence.dataset_id,
                        input_contract_hash=sequence.input_contract_hash,
                        condition_id=sequence.condition_id,
                        task_pre_point=sequence.steps[0].point,
                        seed=changed_seed,
                    )


if __name__ == "__main__":
    unittest.main()
