from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import FittedExpansionCalibrator
from token_prediction.final_ensemble import (
    EmpiricalFoldState,
    ensemble_forecasts,
    ensemble_prediction_maps,
    final_holdout_dataset_id,
    final_task_pseudonym,
    predict_point_rows,
)


def _point(
    point_id: str,
    *,
    cutoff: int = 1,
    trajectory_id: str = "trajectory-1",
) -> PredictionPoint:
    return PredictionPoint(
        point_id=point_id,
        source_event_id=f"event-{point_id}",
        task_id="task-1",
        trajectory_id=trajectory_id,
        run_id="run-1",
        prediction_context_id="context-1",
        condition_id="condition-1",
        logical_call_id=f"call-{point_id}",
        attempt_id=None,
        cutoff_event_seq=cutoff,
        position=PredictionPosition.CALL_PRE,
        target=PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
        features={"request_chars": cutoff * 10},
        known_offset_tokens=0,
    )


class _Session:
    def predict(self, point: PredictionPoint) -> TokenForecast:
        value = float(point.features["request_chars"])
        return TokenForecast(point.point_id, point.target, value, value, value)

    def observe(self, transition: object) -> None:
        del transition


class _Fitted:
    def start(self, context: object) -> _Session:
        del context
        return _Session()


class _BatchEncoder:
    def transform(self, points: tuple[PredictionPoint, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            matrix=[[float(point.features["request_chars"])] for point in points]
        )


class _BatchBooster:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(
        self,
        matrix: list[list[float]],
        *,
        num_iteration: int,
        num_threads: int,
    ) -> list[float]:
        self.assert_controls(num_iteration, num_threads)
        return [row[0] + self.offset for row in matrix]

    @staticmethod
    def assert_controls(num_iteration: int, num_threads: int) -> None:
        if num_iteration != 1 or num_threads != 1:
            raise AssertionError("batch predictor changed frozen controls")


class _BatchFitted:
    estimator_id = "lightgbm_quantile"
    target = PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS
    position = PredictionPosition.CALL_PRE
    allowed_condition_ids = ("condition-1",)
    encoder = _BatchEncoder()
    quantiles = (0.05, 0.5, 0.95)
    boosters = {
        0.05: _BatchBooster(-5.0),
        0.5: _BatchBooster(0.0),
        0.95: _BatchBooster(5.0),
    }
    best_iterations = {0.05: 1, 0.5: 1, 0.95: 1}


class FinalEnsembleTests(unittest.TestCase):
    def test_forecast_ensemble_averages_calibrated_and_raw_values(self) -> None:
        first = TokenForecast(
            "point-1",
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
            1.0,
            2.0,
            5.0,
            latency_ms=2.0,
            raw_lower=2.0,
            raw_point=2.0,
            raw_upper=4.0,
        )
        second = TokenForecast(
            "point-1",
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
            3.0,
            4.0,
            7.0,
            latency_ms=3.0,
            raw_lower=4.0,
            raw_point=4.0,
            raw_upper=6.0,
        )
        actual = ensemble_forecasts((first, second))
        self.assertEqual((actual.lower, actual.point, actual.upper), (2.0, 3.0, 6.0))
        self.assertEqual(
            (actual.raw_lower, actual.raw_point, actual.raw_upper),
            (3.0, 3.0, 5.0),
        )
        self.assertEqual(actual.latency_ms, 5.0)

    def test_prediction_map_ensemble_rejects_cohort_drift(self) -> None:
        first = {
            "point-1": TokenForecast(
                "point-1",
                PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                1.0,
                1.0,
                1.0,
            )
        }
        second = {
            "point-2": TokenForecast(
                "point-2",
                PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                1.0,
                1.0,
                1.0,
            )
        }
        with self.assertRaisesRegex(ValueError, "cohorts"):
            ensemble_prediction_maps((first, second))

    def test_empirical_state_round_trip_is_strict_and_calibrated(self) -> None:
        state = EmpiricalFoldState(
            target=PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
            lower=10.0,
            point=20.0,
            upper=30.0,
            calibrator=FittedExpansionCalibrator(
                "task_max_conformal",
                0.1,
                3.0,
            ),
            development_dataset_id="development-dataset",
            split_plan_id="split-plan",
            split_seed=20260719,
            fold=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(
                json.dumps(state.to_dict(), sort_keys=True),
                encoding="utf-8",
            )
            loaded = EmpiricalFoldState.load(path)
        self.assertEqual(loaded, state)
        forecast = loaded.predict(_point("point-1"))
        self.assertEqual(
            (forecast.lower, forecast.point, forecast.upper),
            (7.0, 20.0, 33.0),
        )
        self.assertEqual(
            (forecast.raw_lower, forecast.raw_point, forecast.raw_upper),
            (10.0, 20.0, 30.0),
        )

    def test_empirical_state_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text('{"state_schema_version":1,"state_schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unreadable"):
                EmpiricalFoldState.load(path)

    def test_point_prediction_preserves_trajectory_order_and_exact_cohort(self) -> None:
        rows = tuple(
            DatasetRow(
                point=point,
                label=100,
                status=LabelStatus.OBSERVED,
                invalid_reason="",
            )
            for point in (_point("point-2", cutoff=2), _point("point-1", cutoff=1))
        )
        predictions = predict_point_rows(
            _Fitted(),
            rows,
            dataset_id="dataset-1",
            input_contract_hash=None,
        )
        self.assertEqual(set(predictions), {"point-1", "point-2"})
        self.assertEqual(predictions["point-1"].point, 10.0)
        self.assertEqual(predictions["point-2"].point, 20.0)

    def test_lightgbm_final_inference_is_batched_without_changing_quantiles(self) -> None:
        rows = tuple(
            DatasetRow(
                point=point,
                label=100,
                status=LabelStatus.OBSERVED,
            )
            for point in (_point("point-2", cutoff=2), _point("point-1", cutoff=1))
        )
        predictions = predict_point_rows(
            _BatchFitted(),
            rows,
            dataset_id="dataset-1",
            input_contract_hash=None,
        )
        self.assertEqual(
            (
                predictions["point-1"].raw_lower,
                predictions["point-1"].raw_point,
                predictions["point-1"].raw_upper,
            ),
            (5.0, 10.0, 15.0),
        )
        self.assertEqual(
            (
                predictions["point-2"].lower,
                predictions["point-2"].point,
                predictions["point-2"].upper,
            ),
            (15.0, 20.0, 25.0),
        )

    def test_final_identity_and_pseudonyms_are_deterministic(self) -> None:
        first = final_holdout_dataset_id(
            parent_dataset_id="parent",
            holdout_plan_id="plan",
            task_ids=("task-b", "task-a"),
        )
        second = final_holdout_dataset_id(
            parent_dataset_id="parent",
            holdout_plan_id="plan",
            task_ids=("task-a", "task-b"),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            final_task_pseudonym("task-a", final_dataset_id=first),
            final_task_pseudonym("task-a", final_dataset_id=second),
        )


if __name__ == "__main__":
    unittest.main()
