from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from token_prediction.dataset import (
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.estimators import (
    FitContext,
    LightGBMQuantileEstimator,
    RunContext,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.lightgbm_bundle import (
    LightGBMBundleError,
    lightgbm_bundle_files,
    load_lightgbm_bundle,
    save_lightgbm_bundle,
)


def _point(index: int, *, condition_id: str = "condition-a") -> PredictionPoint:
    return PredictionPoint(
        point_id=f"bundle-point-{index}",
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
            "task_tokens": index + 1,
            "model_id": "model-a" if index % 2 == 0 else "model-b",
            "task_embedding": (float(index % 5), float((index * 3) % 7)),
        },
        known_offset_tokens=0,
    )


def _view(indices: range) -> TrainingView:
    return TrainingView(
        dataset_id="bundle-dataset",
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        examples=tuple(
            TrainingExample(
                _point(index),
                float(3 * (index + 1) + (10 if index % 2 else 0)),
                sample_weight=1.0,
            )
            for index in indices
        ),
    )


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_manifest(bundle: Path, manifest: dict[str, object]) -> None:
    payload = _canonical_json_bytes(manifest)
    (bundle / "manifest.json").write_bytes(payload)
    (bundle / "manifest.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii", newline="\n"
    )


class LightGBMBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fitted = LightGBMQuantileEstimator(
            num_boost_round=60,
            early_stopping_rounds=8,
            learning_rate=0.1,
            num_leaves=7,
            min_data_in_leaf=2,
        ).fit(
            _view(range(60)),
            _view(range(60, 80)),
            FitContext(seed=29, fold=2, interval_alpha=0.05),
        )

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="lightgbm-bundle-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.bundle = save_lightgbm_bundle(self.fitted, self.root / "bundle")

    def test_round_trip_predictions_are_exact_and_scope_is_preserved(self) -> None:
        point = _point(81)
        context = RunContext("task", "trajectory", "run")
        expected = self.fitted.start(context).predict(point)

        loaded = load_lightgbm_bundle(self.bundle)
        actual = loaded.start(context).predict(point)

        self.assertEqual(actual, expected)
        self.assertEqual(loaded.dataset_id, "bundle-dataset")
        self.assertEqual(loaded.position, PredictionPosition.TASK_PRE)
        self.assertEqual(loaded.allowed_condition_ids, ("condition-a",))
        self.assertEqual(loaded.fit_report, self.fitted.fit_report)

    def test_in_memory_file_builder_matches_saved_directory_exactly(self) -> None:
        expected = lightgbm_bundle_files(self.fitted)
        actual = {path.name: path.read_bytes() for path in self.bundle.iterdir()}

        self.assertEqual(actual, expected)
        with self.assertRaises(TypeError):
            expected["extra"] = b"not mutable"

    def test_loaded_session_rejects_wrong_target_position_and_condition(self) -> None:
        session = load_lightgbm_bundle(self.bundle).start(
            RunContext("task", "trajectory", "run")
        )
        cases = (
            replace(
                _point(81),
                target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            ),
            replace(_point(81), position=PredictionPosition.TASK_LAUNCH),
            replace(_point(81), condition_id="condition-b"),
        )
        for point in cases:
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "bundle|condition_id"):
                    session.predict(point)

    def test_model_encoder_and_manifest_tampering_fail_closed(self) -> None:
        model_file = next(self.bundle.glob("model-*.txt"))
        cases = (
            self.bundle / "manifest.json",
            self.bundle / "encoder.json",
            model_file,
        )
        for index, path in enumerate(cases):
            if index:
                shutil.rmtree(self.bundle)
                save_lightgbm_bundle(self.fitted, self.bundle)
                if path.name.startswith("model-"):
                    path = next(self.bundle.glob("model-*.txt"))
            path.write_bytes(path.read_bytes() + b"\n")
            with self.subTest(filename=path.name):
                with self.assertRaises(LightGBMBundleError):
                    load_lightgbm_bundle(self.bundle)

    def test_encoder_content_hash_is_checked_after_file_checksums(self) -> None:
        encoder_path = self.bundle / "encoder.json"
        encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
        encoder["category_vocabularies"][0]["values"].append("tampered-model")
        encoder_payload = _canonical_json_bytes(encoder)
        encoder_path.write_bytes(encoder_payload)

        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["encoder"]["sha256"] = hashlib.sha256(encoder_payload).hexdigest()
        _rewrite_manifest(self.bundle, manifest)

        with self.assertRaisesRegex(LightGBMBundleError, "content hash"):
            load_lightgbm_bundle(self.bundle)

    def test_manifest_schema_and_quantile_mapping_are_strict(self) -> None:
        manifest_path = self.bundle / "manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))

        unknown_field = dict(original)
        unknown_field["unexpected"] = True
        _rewrite_manifest(self.bundle, unknown_field)
        with self.assertRaisesRegex(LightGBMBundleError, "keys do not match"):
            load_lightgbm_bundle(self.bundle)

        _rewrite_manifest(self.bundle, original)
        mismatched_quantile = json.loads(json.dumps(original))
        mismatched_quantile["quantiles"][0]["value"] = 0.03
        _rewrite_manifest(self.bundle, mismatched_quantile)
        with self.assertRaisesRegex(LightGBMBundleError, "quantile identifier"):
            load_lightgbm_bundle(self.bundle)

    def test_non_default_alpha_has_collision_free_model_names(self) -> None:
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        quantiles = tuple(record["value"] for record in manifest["quantiles"])
        filenames = tuple(record["filename"] for record in manifest["models"].values())

        self.assertEqual(quantiles, (0.025, 0.5, 0.975))
        self.assertEqual(len(filenames), 3)
        self.assertEqual(len(set(filenames)), 3)
        self.assertTrue(all(filename.startswith("model-q") for filename in filenames))
        self.assertTrue(all(filename.endswith(".txt") for filename in filenames))

    def test_missing_and_extra_files_are_both_rejected(self) -> None:
        model = next(self.bundle.glob("model-*.txt"))
        model.unlink()
        with self.assertRaisesRegex(LightGBMBundleError, "file set"):
            load_lightgbm_bundle(self.bundle)

        shutil.rmtree(self.bundle)
        save_lightgbm_bundle(self.fitted, self.bundle)
        (self.bundle / "notes.txt").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(LightGBMBundleError, "file set"):
            load_lightgbm_bundle(self.bundle)

    def test_save_requires_a_fresh_destination(self) -> None:
        with self.assertRaises(FileExistsError):
            save_lightgbm_bundle(self.fitted, self.bundle)


if __name__ == "__main__":
    unittest.main()
