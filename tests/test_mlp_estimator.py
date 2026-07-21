from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest import mock

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.mlp import (
    MAX_MLP_DIMENSION,
    IndependentMLPQuantileEstimator,
    MLPArchitecture,
)
from token_prediction.estimators.neural_encoder import (
    NeuralFeatureEncoder,
    OptionalNeuralDependencyError,
)


HAS_NEURAL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"))


def _point(index: int, *, condition_id: str = "condition-a") -> PredictionPoint:
    return PredictionPoint(
        point_id=f"mlp-point-{index}",
        source_event_id=f"event-{index}",
        task_id=f"task-{index // 2}",
        trajectory_id=f"trajectory-{index}",
        run_id=f"run-{index}",
        prediction_context_id=f"context-{index}",
        condition_id=condition_id,
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features={
            "task_tokens": float(index + 1) if index % 7 else None,
            "model_id": "model-a" if index % 2 else "model-b",
            "task_embedding": (float(index % 5), float((index * 3) % 7)),
        },
        known_offset_tokens=0,
    )


def _target(index: int) -> float:
    return float(2 * index + (11 if index % 2 else 3))


def _view(indices: range) -> TrainingView:
    return TrainingView(
        dataset_id="mlp-dataset",
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        input_contract_hash="a" * 64,
        examples=tuple(
            TrainingExample(
                point=_point(index),
                target_value=_target(index),
                sample_weight=0.5 if index % 3 else 2.0,
            )
            for index in indices
        ),
    )


class IndependentMLPContractTests(unittest.TestCase):
    def test_import_does_not_load_optional_dependencies(self) -> None:
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'safetensors'}:
        raise AssertionError('optional dependency imported eagerly: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import token_prediction.estimators.mlp
import token_prediction.estimators.neural_bundle
print('safe')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "safe")

    def test_missing_optional_dependency_fails_with_actionable_error(self) -> None:
        with mock.patch(
            "token_prediction.estimators.mlp._load_neural_dependencies",
            side_effect=OptionalNeuralDependencyError(
                "install token-prediction[neural]"
            ),
        ):
            with self.assertRaisesRegex(
                OptionalNeuralDependencyError, r"token-prediction\[neural\]"
            ):
                IndependentMLPQuantileEstimator(max_epochs=2, patience=1).fit(
                    _view(range(4)), _view(range(4, 6)), FitContext(1, 0)
                )

    def test_hyperparameter_and_alpha_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[1, 200\]"):
            IndependentMLPQuantileEstimator(max_epochs=201)
        with self.assertRaisesRegex(ValueError, "q50_huber_delta"):
            IndependentMLPQuantileEstimator(q50_huber_delta=0)
        with self.assertRaisesRegex(ValueError, "ordered"):
            IndependentMLPQuantileEstimator(quantiles=(0.1, 0.4, 0.9))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            IndependentMLPQuantileEstimator(quantiles=(0.05, 0.5, 0.9))
        for hidden_dims in ((8.9, 4.1), ("8", "4"), (True, 4)):
            with self.subTest(hidden_dims=hidden_dims):
                with self.assertRaisesRegex(ValueError, "integer"):
                    IndependentMLPQuantileEstimator(hidden_dims=hidden_dims)
        with self.assertRaisesRegex(ValueError, "integer"):
            IndependentMLPQuantileEstimator(max_epochs=3.9)
        with self.assertRaisesRegex(ValueError, "safe"):
            MLPArchitecture(input_dim=10_000_000)
        with self.assertRaisesRegex(ValueError, "safe"):
            MLPArchitecture(input_dim=16, hidden_dims=(10_000, 8))

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_weighted_pinball_uses_sample_weights(self) -> None:
        import torch

        from token_prediction.estimators.mlp import _weighted_quantile_loss

        predictions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        targets = torch.tensor([1.0, 10.0])
        equal = _weighted_quantile_loss(
            torch,
            predictions,
            targets,
            torch.tensor([1.0, 1.0]),
            (0.05, 0.5, 0.95),
            q50_huber_delta=None,
        )
        first_heavy = _weighted_quantile_loss(
            torch,
            predictions,
            targets,
            torch.tensor([100.0, 1.0]),
            (0.05, 0.5, 0.95),
            q50_huber_delta=None,
        )
        self.assertLess(float(first_heavy), float(equal))

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_deterministic_fit_scope_repair_and_stateless_observe(self) -> None:
        estimator = IndependentMLPQuantileEstimator(
            max_epochs=30,
            patience=6,
            hidden_dims=(16, 8),
        )
        context = FitContext(seed=17, fold=2, interval_alpha=0.2)
        first = estimator.fit(_view(range(40)), _view(range(40, 50)), context)
        second = estimator.fit(_view(range(40)), _view(range(40, 50)), context)
        self.assertEqual(first.quantiles, (0.1, 0.5, 0.9))
        self.assertEqual(first.architecture.hidden_dims, (16, 8))
        self.assertEqual(first.fit_report.parameters["q50_huber_delta"], None)
        self.assertEqual(first.fit_report.parameters["optimizer"], "adamw")
        self.assertEqual(first.fit_report.parameters["device"], "cpu")
        self.assertTrue(first.fit_report.parameters["deterministic"])
        with self.assertRaises(TypeError):
            first.fit_report.parameters["hidden_dims"][0] = 999
        self.assertLessEqual(len(first.fit_report.validation_history), 30)

        point = _point(51)
        context = RunContext(
            point.task_id,
            point.trajectory_id,
            point.run_id,
            dataset_id=first.dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            input_contract_hash=first.input_contract_hash,
        )
        session = first.start(context)
        before = session.predict(point)
        session.observe(ObservedTransition("previous", point.point_id, 999_999))
        after = session.predict(point)
        other = second.start(context).predict(point)
        self.assertEqual(before, after)
        self.assertEqual(before, other)
        self.assertGreaterEqual(before.lower, 0.0)
        self.assertLessEqual(before.lower, before.point)
        self.assertLessEqual(before.point, before.upper)

        with self.assertRaisesRegex(ValueError, "condition_id"):
            session.predict(replace(point, condition_id="condition-b"))
        with self.assertRaisesRegex(ValueError, "target"):
            session.predict(
                replace(point, target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS)
            )
        with self.assertRaisesRegex(ValueError, "dataset_id"):
            first.start(replace(context, dataset_id="another-dataset"))
        with self.assertRaisesRegex(ValueError, "input_contract_hash"):
            first.start(replace(context, input_contract_hash="b" * 64))
        for field in ("dataset_id", "condition_id", "target", "input_contract_hash"):
            with self.subTest(missing_context_field=field):
                with self.assertRaisesRegex(ValueError, field):
                    first.start(replace(context, **{field: None}))
        wrong_task = first.start(replace(context, task_id="another-task"))
        with self.assertRaisesRegex(ValueError, "task_id"):
            wrong_task.predict(point)

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_configured_quantiles_must_match_interval_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match experiment"):
            IndependentMLPQuantileEstimator(
                quantiles=(0.05, 0.5, 0.95), max_epochs=2, patience=1
            ).fit(
                _view(range(10)),
                _view(range(10, 14)),
                FitContext(1, 0, interval_alpha=0.2),
            )

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_validation_condition_cannot_expand_train_scope(self) -> None:
        validation = _view(range(10, 14))
        validation = replace(
            validation,
            examples=tuple(
                replace(example, point=replace(example.point, condition_id="condition-b"))
                for example in validation.examples
            ),
        )
        with self.assertRaisesRegex(ValueError, "condition scope"):
            IndependentMLPQuantileEstimator(max_epochs=2, patience=1).fit(
                _view(range(10)), validation, FitContext(1, 0)
            )

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_encoder_width_limit_precedes_matrix_materialization(self) -> None:
        encoder = mock.Mock()
        encoder.schema.output_width = MAX_MLP_DIMENSION + 1
        encoder.transform.side_effect = AssertionError(
            "transform reached before architecture limit"
        )
        with mock.patch.object(NeuralFeatureEncoder, "fit", return_value=encoder):
            with self.assertRaisesRegex(ValueError, "input_dim"):
                IndependentMLPQuantileEstimator(max_epochs=2, patience=1).fit(
                    _view(range(10)),
                    _view(range(10, 14)),
                    FitContext(1, 0),
                )
        encoder.transform.assert_not_called()

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_encoded_cell_limit_precedes_matrix_materialization(self) -> None:
        encoder = mock.Mock()
        encoder.schema.output_width = 500_000
        encoder.transform.side_effect = AssertionError(
            "transform reached before encoded-cell limit"
        )
        with mock.patch.object(NeuralFeatureEncoder, "fit", return_value=encoder):
            with self.assertRaisesRegex(ValueError, "cell-count"):
                IndependentMLPQuantileEstimator(
                    hidden_dims=(1, 1), max_epochs=2, patience=1
                ).fit(
                    _view(range(30)),
                    _view(range(30, 60)),
                    FitContext(1, 0),
                )
        encoder.transform.assert_not_called()


if __name__ == "__main__":
    unittest.main()
