from __future__ import annotations

import copy
import unittest

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.neural_encoder import (
    NEURAL_ENCODER_SCHEMA_VERSION,
    NeuralEncoderSchema,
    NeuralFeatureEncoder,
)


def _point(index: int, features: dict[str, object]) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"neural-encoder-point-{index}",
        source_event_id=f"event-{index}",
        task_id=f"task-{index}",
        trajectory_id=f"trajectory-{index}",
        run_id=f"run-{index}",
        prediction_context_id=f"context-{index}",
        condition_id="condition-a",
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features=features,
        known_offset_tokens=0,
    )


class NeuralEncoderTests(unittest.TestCase):
    def test_train_only_statistics_and_explicit_missing_unknown_columns(self) -> None:
        train = (
            _point(
                0,
                {
                    "task_tokens": 1.0,
                    "model_id": "model-a",
                    "task_embedding": (1.0, 10.0),
                },
            ),
            _point(
                1,
                {
                    "task_tokens": 3.0,
                    "model_id": "model-b",
                    "task_embedding": (3.0, 30.0),
                },
            ),
            _point(
                2,
                {
                    "task_tokens": None,
                    "model_id": None,
                    "task_embedding": None,
                },
            ),
        )
        encoder = NeuralFeatureEncoder.fit(train)
        schema_before = encoder.to_dict()
        batch = encoder.transform(
            (
                _point(
                    3,
                    {
                        "task_tokens": None,
                        "model_id": "never-seen",
                        "task_embedding": None,
                    },
                ),
                _point(
                    4,
                    {
                        "task_tokens": 1_000_000.0,
                        "model_id": "model-a",
                        "task_embedding": (1_000_000.0, -1_000_000.0),
                    },
                ),
            )
        )

        self.assertEqual(encoder.to_dict(), schema_before)
        names = batch.feature_names
        first = dict(zip(names, batch.matrix[0].tolist()))
        self.assertEqual(first["model_id__missing"], 0.0)
        self.assertEqual(first["model_id__category_0000"], 0.0)
        self.assertEqual(first["model_id__category_0001"], 0.0)
        self.assertEqual(first["model_id__unknown"], 1.0)
        self.assertEqual(first["task_tokens__missing"], 1.0)
        self.assertEqual(first["task_embedding__missing"], 1.0)
        self.assertEqual(encoder.schema.numeric[0].median, 2.0)

    def test_missing_category_is_not_unknown_and_known_values_are_one_hot(self) -> None:
        encoder = NeuralFeatureEncoder.fit(
            (
                _point(0, {"model_id": "b"}),
                _point(1, {"model_id": "a"}),
                _point(2, {"model_id": None}),
            )
        )
        batch = encoder.transform(
            (
                _point(3, {"model_id": None}),
                _point(4, {"model_id": "a"}),
                _point(5, {"model_id": "c"}),
            )
        )
        self.assertEqual(
            batch.feature_names,
            (
                "model_id__missing",
                "model_id__category_0000",
                "model_id__category_0001",
                "model_id__unknown",
            ),
        )
        self.assertEqual(
            batch.matrix.tolist(),
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        )

    def test_vector_width_mask_and_all_missing_train_vector(self) -> None:
        encoder = NeuralFeatureEncoder.fit(
            (
                _point(0, {"task_embedding": (1.0, 2.0), "new_context_embedding": None}),
                _point(1, {"task_embedding": None, "new_context_embedding": None}),
            )
        )
        self.assertEqual(
            encoder.schema.dropped_all_missing_vectors, ("new_context_embedding",)
        )
        batch = encoder.transform(
            (_point(2, {"task_embedding": None, "new_context_embedding": (9.0,)}),)
        )
        row = dict(zip(batch.feature_names, batch.matrix[0].tolist()))
        self.assertEqual(row["task_embedding__missing"], 1.0)
        self.assertNotIn("new_context_embedding__missing", row)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            encoder.transform((_point(3, {"task_embedding": (1.0,)}),))

    def test_inconsistent_train_vector_width_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent train-fold widths"):
            NeuralFeatureEncoder.fit(
                (
                    _point(0, {"task_embedding": (1.0, 2.0)}),
                    _point(1, {"task_embedding": (1.0, 2.0, 3.0)}),
                )
            )

    def test_schema_round_trip_hash_and_strict_keys(self) -> None:
        encoder = NeuralFeatureEncoder.fit(
            (
                _point(0, {"task_tokens": 1, "model_id": "a"}),
                _point(1, {"task_tokens": None, "model_id": None}),
            )
        )
        restored = NeuralFeatureEncoder.from_dict(encoder.to_dict())
        self.assertEqual(restored.schema.content_hash, encoder.schema.content_hash)
        self.assertEqual(restored.schema.feature_names, encoder.schema.feature_names)
        self.assertEqual(restored.schema.schema_version, NEURAL_ENCODER_SCHEMA_VERSION)

        extra = copy.deepcopy(encoder.to_dict())
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            NeuralEncoderSchema.from_dict(extra)

        non_finite = copy.deepcopy(encoder.to_dict())
        non_finite["numeric"][0]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            NeuralEncoderSchema.from_dict(non_finite)


if __name__ == "__main__":
    unittest.main()
