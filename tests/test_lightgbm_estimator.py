from __future__ import annotations

import unittest
from dataclasses import replace

from token_prediction.dataset import (
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
    make_task_split_plan,
)
from token_prediction.estimators import (
    FitContext,
    LightGBMQuantileEstimator,
    RunContext,
    TrainingExample,
    TrainingView,
    builtin_registry,
)
from token_prediction.experiment import (
    CandidateRole,
    CandidateSpec,
    ExperimentRunner,
    ExperimentSpec,
    compare_candidate_results,
)
from token_prediction.features import FeatureSet

from tests.helpers import make_two_call_trajectory


def _point(index: int) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"point-{index}",
        source_event_id=f"event-{index}",
        task_id=f"task-{index // 2}",
        trajectory_id=f"trajectory-{index}",
        run_id=f"run-{index}",
        prediction_context_id=f"context-{index}",
        condition_id="condition",
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features={
            "task_tokens": index + 1,
            "model_id": "model-a" if index % 2 == 0 else "model-b",
            "task_embedding": (float(index % 5), float((index * 3) % 7)),
        },
        known_offset_tokens=0,
    )


def _target(index: int) -> float:
    return float(3 * (index + 1) + (10 if index % 2 else 0) + (index % 5))


def _view(indices: range) -> TrainingView:
    examples = tuple(
        TrainingExample(_point(index), _target(index), sample_weight=1.0)
        for index in indices
    )
    return TrainingView(
        dataset_id="dataset",
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        examples=examples,
    )


def _estimator() -> LightGBMQuantileEstimator:
    return LightGBMQuantileEstimator(
        num_boost_round=60,
        early_stopping_rounds=8,
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=2,
    )


class LightGBMEstimatorTests(unittest.TestCase):
    def test_three_quantiles_early_stop_and_mixed_features(self) -> None:
        fitted = _estimator().fit(_view(range(60)), _view(range(60, 80)), FitContext(17, 2))
        self.assertEqual(
            tuple(report.quantile for report in fitted.fit_report.quantiles),
            (0.05, 0.5, 0.95),
        )
        for report in fitted.fit_report.quantiles:
            self.assertGreater(report.best_iteration, 0)
            self.assertLessEqual(report.best_iteration, 60)
            self.assertTrue(report.validation_history)
            self.assertEqual(report.parameters["device_type"], "cpu")
            self.assertTrue(report.parameters["deterministic"])
            self.assertEqual(report.parameters["num_threads"], 1)

        session = fitted.start(RunContext("task", "trajectory", "run"))
        forecast = session.predict(_point(81))
        self.assertEqual(forecast.point_id, "point-81")
        self.assertLessEqual(forecast.lower, forecast.point)
        self.assertLessEqual(forecast.point, forecast.upper)
        self.assertGreaterEqual(forecast.lower, 0.0)
        self.assertIsNotNone(session.last_raw_quantiles)
        self.assertEqual(session.prediction_count, 1)
        self.assertEqual(
            (forecast.raw_lower, forecast.raw_point, forecast.raw_upper),
            (
                session.last_raw_quantiles.q05,
                session.last_raw_quantiles.q50,
                session.last_raw_quantiles.q95,
            ),
        )

        importance = fitted.feature_importance()
        self.assertEqual(
            len(importance), 3 * len(fitted.encoder.schema.feature_names)
        )
        self.assertEqual(
            {record.source_feature_name for record in importance},
            {"task_tokens", "model_id", "task_embedding"},
        )
        source_importance = fitted.source_feature_importance()
        self.assertEqual(len(source_importance), 9)
        for quantile in fitted.quantiles:
            expanded_vector_gain = sum(
                record.gain
                for record in importance
                if record.quantile == quantile
                and record.source_feature_name == "task_embedding"
            )
            aggregated_vector_gain = next(
                record.gain
                for record in source_importance
                if record.quantile == quantile
                and record.source_feature_name == "task_embedding"
            )
            self.assertEqual(expanded_vector_gain, aggregated_vector_gain)

    def test_same_seed_produces_same_semantic_predictions(self) -> None:
        train = _view(range(60))
        validation = _view(range(60, 80))
        first = _estimator().fit(train, validation, FitContext(23, 1))
        second = _estimator().fit(train, validation, FitContext(23, 1))
        point = _point(82)
        first_forecast = first.start(RunContext("t", "r", "run")).predict(point)
        second_forecast = second.start(RunContext("t", "r", "run")).predict(point)
        self.assertEqual(
            (first_forecast.lower, first_forecast.point, first_forecast.upper),
            (second_forecast.lower, second_forecast.point, second_forecast.upper),
        )
        self.assertEqual(first.best_iterations, second.best_iterations)
        self.assertEqual(
            first.fit_report.encoder_schema_hash,
            second.fit_report.encoder_schema_hash,
        )
        self.assertEqual(first.model_strings(), second.model_strings())

    def test_registry_model_uses_existing_experiment_path(self) -> None:
        dataset = build_supervised_dataset(
            make_two_call_trajectory(task, run)
            for task in range(5)
            for run in range(2)
        )
        split = make_task_split_plan(
            dataset.task_ids,
            dataset_id=dataset.dataset_id,
            folds=5,
            seed=31,
        )
        request_feature = FeatureSet(
            "request",
            include_all=False,
            include_features=frozenset({"current_request_tokens_local"}),
        )
        candidates = (
            CandidateSpec(
                "length",
                "length_only",
                request_feature,
                role=CandidateRole.BASELINE,
            ),
            CandidateSpec(
                "lightgbm",
                "lightgbm_quantile",
                request_feature,
                params={
                    "num_boost_round": 15,
                    "early_stopping_rounds": 3,
                    "learning_rate": 0.15,
                    "num_leaves": 3,
                    "min_data_in_leaf": 1,
                },
                role=CandidateRole.MODEL,
            ),
        )
        results = ExperimentRunner(builtin_registry()).run(
            dataset,
            split,
            ExperimentSpec(
                "lightgbm-contract",
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                candidates,
            ),
            seed=31,
        )
        compare_candidate_results(results)
        self.assertEqual(
            {record.point_id for record in results[0].predictions},
            {record.point_id for record in results[1].predictions},
        )
        self.assertEqual(results[0].split_plan_id, results[1].split_plan_id)
        self.assertEqual(results[0].eligibility_hash, results[1].eligibility_hash)

    def test_quantiles_must_match_experiment_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match experiment"):
            LightGBMQuantileEstimator(
                quantiles=(0.05, 0.5, 0.95),
                num_boost_round=10,
                early_stopping_rounds=2,
                min_data_in_leaf=2,
            ).fit(
                _view(range(20)),
                _view(range(20, 30)),
                FitContext(1, 0, interval_alpha=0.2),
            )

    def test_fit_rejects_a_validation_condition_outside_the_train_cell(self) -> None:
        validation = _view(range(20, 30))
        mismatched = replace(
            validation,
            examples=tuple(
                replace(
                    example,
                    point=replace(example.point, condition_id="other-condition"),
                )
                for example in validation.examples
            ),
        )
        with self.assertRaisesRegex(ValueError, "condition scope"):
            _estimator().fit(_view(range(20)), mismatched, FitContext(1, 0))

    def test_default_quantiles_are_derived_from_experiment_alpha(self) -> None:
        fitted = _estimator().fit(
            _view(range(20)),
            _view(range(20, 30)),
            FitContext(1, 0, interval_alpha=0.2),
        )
        self.assertEqual(fitted.quantiles, (0.1, 0.5, 0.9))


if __name__ == "__main__":
    unittest.main()
