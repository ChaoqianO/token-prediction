from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import (
    CrossPositionDeductEstimator,
    FitContext,
    GRUResidualEstimator,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.neural_encoder import OptionalNeuralDependencyError
from token_prediction.lifecycle import visible_spend_delta


HAS_NEURAL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"))
TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
CONDITION = "condition:gru"
CONTRACT_HASH = "a" * 64


def _point(
    sequence_index: int,
    step_index: int,
    *,
    missing_usage_attempts: int = 0,
) -> PredictionPoint:
    position = (
        PredictionPosition.TASK_PRE
        if step_index == 0
        else PredictionPosition.TASK_UPDATE
    )
    cumulative_input = (0, 20 + sequence_index, 31 + sequence_index)[step_index]
    cumulative_output = (0, 10, 19)[step_index]
    return PredictionPoint(
        point_id=f"gru-{sequence_index}-{step_index}",
        source_event_id=f"event-{sequence_index}-{step_index}",
        task_id=f"task-{sequence_index}",
        trajectory_id=f"trajectory-{sequence_index}",
        run_id=f"run-{sequence_index}",
        prediction_context_id=f"context-{sequence_index}-{step_index}",
        condition_id=CONDITION,
        logical_call_id=f"call-{sequence_index}-{step_index}",
        attempt_id=None,
        cutoff_event_seq=step_index * 10,
        position=position,
        target=TARGET,
        features={
            "request_content_chars": float(sequence_index + step_index),
            "model_id": "even" if sequence_index % 2 == 0 else "odd",
            "step_progress_ratio": None if step_index == 1 else float(step_index) / 2,
            "missing_usage_attempts": missing_usage_attempts,
            "cumulative_provider_input_tokens": cumulative_input,
            "cumulative_provider_output_tokens": cumulative_output,
        },
        known_offset_tokens=0,
    )


def _seed(point: PredictionPoint) -> SessionSeed:
    forecast = TokenForecast(
        point.point_id,
        point.target,
        150.0,
        180.0,
        220.0,
        raw_lower=150.0,
        raw_point=180.0,
        raw_upper=220.0,
    )
    return SessionSeed(
        point,
        forecast,
        "empirical_quantile",
        "b" * 64,
        "c" * 64,
        ("d" * 64,),
        "uncalibrated_repaired_quantile_ensemble_v1",
        "e" * 64,
    )


def _sequence(sequence_index: int) -> SimpleNamespace:
    points = tuple(_point(sequence_index, step) for step in range(3))
    total = 200.0 + sequence_index * 3
    labels = (None, total - 30 - sequence_index, total - 50 - sequence_index)
    steps = tuple(
        SimpleNamespace(
            point=point,
            label=labels[index],
            loss_mask=index > 0,
            score_mask=index > 0,
            sample_weight=0.0 if index == 0 else 0.5,
            invalid_reason="redacted_task_pre_label" if index == 0 else "",
        )
        for index, point in enumerate(points)
    )
    context_hash = hashlib.sha256(f"sequence-{sequence_index}".encode()).hexdigest()
    return SimpleNamespace(
        dataset_id="gru-dataset",
        input_contract_hash=CONTRACT_HASH,
        task_id=points[0].task_id,
        trajectory_id=points[0].trajectory_id,
        run_id=points[0].run_id,
        condition_id=CONDITION,
        target=TARGET,
        context_hash=context_hash,
        steps=steps,
        session_seed=_seed(points[0]),
    )


def _view(indices: range) -> TrainingView:
    sequences = tuple(_sequence(index) for index in indices)
    examples = tuple(
        TrainingExample(step.point, float(step.label), step.sample_weight)
        for sequence in sequences
        for step in sequence.steps[1:]
    )
    return TrainingView(
        dataset_id="gru-dataset",
        position=PredictionPosition.TASK_UPDATE,
        target=TARGET,
        examples=examples,
        lifecycle_sequences=sequences,
        input_contract_hash=CONTRACT_HASH,
    )


def _context(sequence: SimpleNamespace, *, runtime_mode: str = "offline") -> RunContext:
    return RunContext(
        sequence.task_id,
        sequence.trajectory_id,
        sequence.run_id,
        dataset_id=sequence.dataset_id,
        condition_id=sequence.condition_id,
        target=sequence.target,
        runtime_mode=runtime_mode,  # type: ignore[arg-type]
        input_contract_hash=sequence.input_contract_hash,
        session_seed=sequence.session_seed,
    )


def _trajectory_forecasts(fitted: object, sequence: SimpleNamespace) -> tuple[TokenForecast, ...]:
    session = fitted.start(_context(sequence))
    forecasts: list[TokenForecast] = []
    previous = sequence.steps[0].point
    for step in sequence.steps[1:]:
        transition = ObservedTransition(
            previous.point_id,
            step.point.point_id,
            visible_spend_delta(previous, step.point),
        )
        session.observe(transition)
        forecasts.append(session.predict(step.point))
        previous = step.point
    return tuple(forecasts)


class GRUResidualContractTests(unittest.TestCase):
    def test_import_does_not_load_optional_dependencies(self) -> None:
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'safetensors'}:
        raise AssertionError('optional dependency imported eagerly: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import token_prediction.estimators.gru
import token_prediction.estimators.gru_bundle
print('safe')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "safe")

    def test_hyperparameter_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "residual_scale"):
            GRUResidualEstimator(residual_scale=-1)
        with self.assertRaisesRegex(ValueError, "no_recurrence"):
            GRUResidualEstimator(no_recurrence=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"\[1, 200\]"):
            GRUResidualEstimator(max_epochs=201)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            GRUResidualEstimator(hidden_dim=True)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            GRUResidualEstimator(quantiles=(0.1, 0.5, 0.8))

    def test_missing_optional_dependency_is_actionable(self) -> None:
        with mock.patch(
            "token_prediction.estimators.gru._load_neural_dependencies",
            side_effect=OptionalNeuralDependencyError(
                "install token-prediction[neural]"
            ),
        ):
            with self.assertRaisesRegex(
                OptionalNeuralDependencyError,
                r"token-prediction\[neural\]",
            ):
                GRUResidualEstimator(max_epochs=2, patience=1).fit(
                    _view(range(4)),
                    _view(range(4, 6)),
                    FitContext(1, 0),
                )

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_zero_residual_is_exact_cross_position_deduct_trajectory(self) -> None:
        train = _view(range(4))
        validation = _view(range(4, 6))
        context = FitContext(17, 0)
        gru = GRUResidualEstimator(
            transition_dim=8,
            hidden_dim=8,
            residual_head_dim=8,
            residual_scale=0.0,
            max_epochs=2,
            patience=1,
        ).fit(train, validation, context)
        deduct = CrossPositionDeductEstimator(
            expected_condition_id=CONDITION,
            expected_input_contract_hash=CONTRACT_HASH,
        ).fit(train, validation, context)
        sequence = _sequence(9)
        self.assertEqual(
            _trajectory_forecasts(gru, sequence),
            _trajectory_forecasts(deduct, sequence),
        )
        self.assertEqual(gru.fit_report.parameters["teacher_forcing"], False)

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_deterministic_fit_run_reset_and_offline_shadow_parity(self) -> None:
        estimator = GRUResidualEstimator(
            transition_dim=8,
            hidden_dim=8,
            residual_head_dim=8,
            no_recurrence=True,
            max_epochs=4,
            patience=2,
        )
        fit_context = FitContext(23, 1)
        first = estimator.fit(_view(range(4)), _view(range(4, 6)), fit_context)
        second = estimator.fit(_view(range(4)), _view(range(4, 6)), fit_context)
        sequence = _sequence(8)
        self.assertEqual(
            _trajectory_forecasts(first, sequence),
            _trajectory_forecasts(second, sequence),
        )
        self.assertEqual(
            _trajectory_forecasts(first, sequence),
            _trajectory_forecasts(first, sequence),
        )
        offline = first.start(_context(sequence, runtime_mode="offline"))
        shadow = first.start(_context(sequence, runtime_mode="shadow"))
        previous = sequence.steps[0].point
        offline_forecasts: list[TokenForecast] = []
        shadow_forecasts: list[TokenForecast] = []
        for step in sequence.steps[1:]:
            transition = ObservedTransition(
                previous.point_id,
                step.point.point_id,
                visible_spend_delta(previous, step.point),
            )
            offline.observe(transition)
            shadow.observe(transition)
            offline_forecasts.append(offline.predict(step.point))
            shadow_forecasts.append(shadow.predict(step.point))
            previous = step.point
        self.assertEqual(offline_forecasts, shadow_forecasts)
        self.assertTrue(first.no_recurrence)
        self.assertFalse(first.fit_report.parameters["teacher_forcing"])

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_labels_cannot_enter_seed_or_inference_session(self) -> None:
        fitted = GRUResidualEstimator(
            transition_dim=8,
            hidden_dim=8,
            residual_head_dim=8,
            max_epochs=2,
            patience=1,
        ).fit(_view(range(4)), _view(range(4, 6)), FitContext(7, 0))
        sequence = _sequence(7)
        session = fitted.start(_context(sequence))
        point = sequence.steps[1].point
        with self.assertRaisesRegex(RuntimeError, "observe"):
            session.predict(point)
        with self.assertRaisesRegex(ValueError, "dataset_id"):
            fitted.start(replace(_context(sequence), dataset_id="other"))

        leaked = _view(range(4))
        first_sequence = leaked.lifecycle_sequences[0]
        first_sequence.steps[0].label = 999  # SimpleNamespace fixture only.
        with self.assertRaisesRegex(ValueError, "Task-pre"):
            GRUResidualEstimator(max_epochs=2, patience=1).fit(
                leaked,
                _view(range(4, 6)),
                FitContext(1, 0),
            )


if __name__ == "__main__":
    unittest.main()
