from __future__ import annotations

import unittest
from dataclasses import dataclass

from token_prediction.crossfit import (
    InitializerComponent,
    SEED_POLICY_HASH,
    SEED_POLICY_ID,
    _seed_forecast,
    ensemble_repaired_forecasts,
    generate_crossfit_seeds,
)
from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import ObservedTransition, RunContext, TokenForecast


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS


def _point(task: str, run: int = 0) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"{task}:run-{run}:task-pre",
        source_event_id=f"{task}:event-{run}",
        task_id=task,
        trajectory_id=f"{task}:trajectory-{run}",
        run_id=f"run-{run}",
        prediction_context_id=f"{task}:context-{run}",
        condition_id="condition:a",
        logical_call_id=f"call-{run}",
        attempt_id=None,
        cutoff_event_seq=1,
        position=PredictionPosition.TASK_PRE,
        target=TARGET,
        features={"visible": run},
        known_offset_tokens=0,
    )


@dataclass
class _Session:
    value: float

    def predict(self, point: PredictionPoint) -> TokenForecast:
        return TokenForecast(
            point.point_id,
            point.target,
            self.value - 1,
            self.value,
            self.value + 2,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass
class _Fitted:
    estimator_id: str
    value: float

    def start(self, context: RunContext) -> _Session:
        if context.session_seed is not None:
            raise AssertionError("initializer received an inference seed")
        return _Session(self.value)


def _component(fold: int, holdout: str, *, value: float) -> InitializerComponent:
    tasks = {"a", "b", "c", "d", "e"}
    validation = chr(ord("a") + ((fold + 1) % 5))
    fit = tasks - {holdout, validation}
    return InitializerComponent(
        inner_fold=fold,
        component_id=f"component-{fold}",
        component_hash=f"{fold + 1:x}" * 64,
        bundle_hashes=(f"{fold + 6:x}" * 64,),
        fit_tasks=frozenset(fit),
        validation_tasks=frozenset({validation}),
        holdout_tasks=frozenset({holdout}),
        fitted=_Fitted("empirical_quantile", value),
    )


class CrossfitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = tuple(
            _component(fold, task, value=10.0 * (fold + 1))
            for fold, task in enumerate(("a", "b", "c", "d", "e"))
        )

    def test_oof_seed_uses_only_holdout_model_and_external_uses_all_models(self) -> None:
        points = tuple(_point(task) for task in ("a", "b", "c", "d", "e", "x"))
        seeds = generate_crossfit_seeds(
            points,
            self.components,
            dataset_id="dataset-v2",
            input_contract_hash="a" * 64,
            initializer_id="empirical_quantile",
            initializer_hash="b" * 64,
            inner_split_id="outer-0-inner-v1",
            oof_tasks=frozenset({"a", "b", "c", "d", "e"}),
            external_tasks=frozenset({"x"}),
        )
        by_task = {record.task_id: record for record in seeds.records}
        self.assertEqual(by_task["a"].producer_inner_folds, (0,))
        self.assertEqual(by_task["a"].seed.forecast.point, 10.0)
        self.assertEqual(by_task["a"].seed.component_bundle_hashes, ("6" * 64,))
        self.assertEqual(by_task["x"].producer_inner_folds, (0, 1, 2, 3, 4))
        self.assertEqual(by_task["x"].seed.forecast.point, 30.0)
        self.assertEqual(by_task["x"].seed.seed_policy_id, SEED_POLICY_ID)
        self.assertEqual(by_task["x"].seed.seed_policy_hash, SEED_POLICY_HASH)
        self.assertEqual(seeds.content_hash, seeds.content_hash)

    def test_task_runs_share_partition_without_sharing_point_identity(self) -> None:
        points = tuple(
            [_point(task) for task in ("a", "b", "c", "d", "e", "x")]
            + [_point("a", run=1), _point("x", run=1)]
        )
        seeds = generate_crossfit_seeds(
            points,
            self.components,
            dataset_id="dataset-v2",
            input_contract_hash="a" * 64,
            initializer_id="empirical_quantile",
            initializer_hash="b" * 64,
            inner_split_id="outer-0-inner-v1",
            oof_tasks=frozenset({"a", "b", "c", "d", "e"}),
            external_tasks=frozenset({"x"}),
        )
        a_records = [record for record in seeds.records if record.task_id == "a"]
        x_records = [record for record in seeds.records if record.task_id == "x"]
        self.assertEqual({record.producer_inner_folds for record in a_records}, {(0,)})
        self.assertEqual(
            {record.producer_inner_folds for record in x_records},
            {(0, 1, 2, 3, 4)},
        )
        self.assertEqual(len({record.point_id for record in seeds.records}), len(seeds.records))

    def test_leakage_and_incomplete_inner_partitions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            first = self.components[0]
            InitializerComponent(
                first.inner_fold,
                first.component_id,
                first.component_hash,
                first.bundle_hashes,
                first.fit_tasks | {"a"},
                first.validation_tasks,
                first.holdout_tasks,
                first.fitted,
            )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            generate_crossfit_seeds(
                tuple(_point(task) for task in ("a", "b", "c", "d", "e")),
                self.components,
                dataset_id="dataset-v2",
                input_contract_hash="a" * 64,
                initializer_id="empirical_quantile",
                initializer_hash="b" * 64,
                inner_split_id="outer-0-inner-v1",
                oof_tasks=frozenset({"a", "b", "c", "d", "e"}),
                external_tasks=frozenset({"x"}),
            )

    def test_ensemble_operates_on_repaired_quantiles(self) -> None:
        point = _point("x")
        forecast = ensemble_repaired_forecasts(
            point,
            (
                TokenForecast(point.point_id, point.target, 0, 10, 20),
                TokenForecast(point.point_id, point.target, 2, 20, 30),
            ),
        )
        self.assertEqual((forecast.lower, forecast.point, forecast.upper), (1, 15, 25))
        self.assertEqual(
            (forecast.raw_lower, forecast.raw_point, forecast.raw_upper),
            (1, 15, 25),
        )

    def test_calibrated_initializer_output_is_reduced_to_repaired_raw_seed(self) -> None:
        point = _point("x")
        calibrated = TokenForecast(
            point.point_id,
            point.target,
            0,
            8,
            30,
            raw_lower=-2,
            raw_point=8,
            raw_upper=12,
        )
        seed = _seed_forecast(calibrated)
        self.assertEqual((seed.lower, seed.point, seed.upper), (0, 8, 12))
        self.assertEqual(
            (seed.raw_lower, seed.raw_point, seed.raw_upper),
            (-2, 8, 12),
        )


if __name__ == "__main__":
    unittest.main()
