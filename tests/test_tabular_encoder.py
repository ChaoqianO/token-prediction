from __future__ import annotations

import math
import unittest

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import EncoderSchema, FoldTabularEncoder


def _point(index: int, features: dict[str, object]) -> PredictionPoint:
    return PredictionPoint(
        point_id=f"point-{index}",
        source_event_id=f"event-{index}",
        task_id=f"task-{index}",
        trajectory_id=f"trajectory-{index}",
        run_id=f"run-{index}",
        prediction_context_id=f"context-{index}",
        condition_id="condition",
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        features=features,
        known_offset_tokens=0,
    )


class FoldTabularEncoderTests(unittest.TestCase):
    def test_mixed_schema_is_train_only_and_round_trips(self) -> None:
        train = (
            _point(
                0,
                {
                    "task_tokens": 10,
                    "model_id": "model-a",
                    "task_embedding": (1.0, 2.0),
                },
            ),
            _point(
                1,
                {
                    "task_tokens": 20,
                    "model_id": "model-b",
                    "task_embedding": (3.0, 4.0),
                },
            ),
        )
        encoder = FoldTabularEncoder.fit(train)
        self.assertEqual(
            encoder.schema.feature_names,
            (
                "model_id",
                "task_embedding__v0000",
                "task_embedding__v0001",
                "task_tokens",
            ),
        )
        self.assertEqual(encoder.schema.categorical_indices, (0,))
        self.assertEqual(
            encoder.schema.category_mapping("model_id"), {"model-a": 0, "model-b": 1}
        )
        self.assertEqual(encoder.schema.vector_width("task_embedding"), 2)

        test = _point(
            2,
            {
                "task_tokens": None,
                "model_id": "test-only-model",
                "task_embedding": (5.0, 6.0),
                "agent_id": "test-only-feature",
            },
        )
        batch = encoder.transform((test,))
        self.assertEqual(batch.matrix.shape, (1, 4))
        self.assertEqual(float(batch.matrix[0, 0]), -1.0)
        self.assertEqual(tuple(batch.matrix[0, 1:3]), (5.0, 6.0))
        self.assertTrue(math.isnan(float(batch.matrix[0, 3])))
        self.assertNotIn("agent_id", batch.feature_names)

        restored = FoldTabularEncoder.from_dict(encoder.to_dict())
        self.assertEqual(restored.schema, EncoderSchema.from_dict(encoder.to_dict()))
        self.assertEqual(restored.schema.content_hash, encoder.schema.content_hash)
        restored_row = restored.transform((test,)).matrix[0]
        self.assertEqual(tuple(restored_row[:3]), tuple(batch.matrix[0, :3]))
        self.assertTrue(math.isnan(float(restored_row[3])))

    def test_inconsistent_train_vector_width_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent train-fold widths"):
            FoldTabularEncoder.fit(
                (
                    _point(0, {"task_embedding": (1.0, 2.0)}),
                    _point(1, {"task_embedding": (1.0, 2.0, 3.0)}),
                )
            )

    def test_transform_rejects_vector_width_drift(self) -> None:
        encoder = FoldTabularEncoder.fit(
            (
                _point(0, {"task_embedding": (1.0, 2.0)}),
                _point(1, {"task_embedding": (3.0, 4.0)}),
            )
        )
        with self.assertRaisesRegex(ValueError, "expected 2"):
            encoder.transform((_point(2, {"task_embedding": (5.0,)}),))

    def test_all_missing_train_vector_is_dropped(self) -> None:
        encoder = FoldTabularEncoder.fit(
            (
                _point(0, {"task_tokens": 1, "task_embedding": None}),
                _point(1, {"task_tokens": 2, "task_embedding": None}),
            )
        )
        self.assertEqual(encoder.schema.dropped_all_missing_vectors, ("task_embedding",))
        self.assertEqual(encoder.schema.feature_names, ("task_tokens",))
        batch = encoder.transform(
            (_point(2, {"task_tokens": 3, "task_embedding": (9.0, 10.0)}),)
        )
        self.assertEqual(batch.matrix.tolist(), [[3.0]])


if __name__ == "__main__":
    unittest.main()
