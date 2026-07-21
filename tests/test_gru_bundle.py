from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from token_prediction.estimators import FitContext, GRUResidualEstimator
from token_prediction.estimators.gru_bundle import (
    GRUBundleError,
    gru_bundle_files,
    load_gru_bundle,
    save_gru_bundle,
)
from token_prediction.evaluation import FittedExpansionCalibrator

from tests.test_gru_estimator import _sequence, _trajectory_forecasts, _view


HAS_NEURAL = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"))


def _provenance() -> dict[str, object]:
    return {
        "role": "lifecycle_updater",
        "candidate_id": "gru-residual",
        "candidate_hash": "1" * 64,
        "candidate_graph": {
            "initializer_estimator_id": "empirical_quantile",
            "updater_estimator_id": "gru_residual",
            "lifecycle_schema_id": "task_lifecycle_v1",
            "seed_policy_id": "uncalibrated_repaired_quantile_ensemble_v1",
            "inner_split_policy_id": "five_fold_rotating_holdout_validation_v1",
        },
        "dataset_id": "gru-dataset",
        "split_plan_id": "2" * 64,
        "eligibility_hash": "3" * 64,
        "lifecycle_context_hash": "4" * 64,
        "lifecycle_scored_hash": "5" * 64,
        "outer_fold": 0,
        "outer_task_partitions_sha256": {
            "train": ["6" * 64],
            "validation": ["7" * 64],
            "calibration": ["8" * 64],
            "test": ["9" * 64],
        },
        "initializer_hash": "a" * 64,
        "inner_split_id": "b" * 64,
        "seed_set_hash": "c" * 64,
        "interval_alpha": 0.1,
        "calibrator_id": "task_max_conformal",
    }


@unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
class GRUBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fitted = GRUResidualEstimator(
            transition_dim=8,
            hidden_dim=8,
            residual_head_dim=8,
            max_epochs=3,
            patience=2,
        ).fit(_view(range(4)), _view(range(4, 6)), FitContext(31, 0))
        cls.calibrator = FittedExpansionCalibrator(
            "task_max_conformal",
            0.1,
            7.0,
        )
        cls.files = dict(
            gru_bundle_files(
                cls.fitted,
                calibrator=cls.calibrator.to_dict(),
                provenance=_provenance(),
            )
        )

    def test_mapping_reload_reproduces_raw_and_calibrated_trajectory(self) -> None:
        sequence = _sequence(11)
        raw = _trajectory_forecasts(self.fitted, sequence)
        loaded_raw = load_gru_bundle(
            self.files,
            expected_provenance=_provenance(),
            apply_calibrator=False,
        )
        loaded_calibrated = load_gru_bundle(
            self.files,
            expected_provenance=_provenance(),
        )
        self.assertEqual(_trajectory_forecasts(loaded_raw, sequence), raw)
        self.assertEqual(
            _trajectory_forecasts(loaded_calibrated, sequence),
            tuple(self.calibrator.transform(forecast) for forecast in raw),
        )
        self.assertTrue(all("pickle" not in name for name in self.files))
        self.assertEqual(
            sum(name.endswith("weights.safetensors") for name in self.files),
            1,
        )

    def test_tamper_extra_unsafe_path_and_provenance_mismatch_fail_closed(self) -> None:
        weights_name = next(
            name for name in self.files if name.endswith("weights.safetensors")
        )
        tampered = dict(self.files)
        payload = bytearray(tampered[weights_name])
        payload[-1] ^= 1
        tampered[weights_name] = bytes(payload)
        with self.assertRaisesRegex(GRUBundleError, "checksum"):
            load_gru_bundle(tampered)

        extra = {**self.files, "extra.json": b"{}\n"}
        with self.assertRaisesRegex(GRUBundleError, "extra"):
            load_gru_bundle(extra)

        unsafe = {**self.files, "../escape": b"x"}
        with self.assertRaisesRegex(GRUBundleError, "safe relative"):
            load_gru_bundle(unsafe)

        expected = _provenance()
        expected["candidate_hash"] = "d" * 64
        with self.assertRaisesRegex(GRUBundleError, "provenance"):
            load_gru_bundle(self.files, expected_provenance=expected)

    def test_disk_reload_rejects_extra_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = save_gru_bundle(
                self.fitted,
                root / "bundle",
                calibrator=self.calibrator.to_dict(),
                provenance=_provenance(),
            )
            loaded = load_gru_bundle(bundle, expected_provenance=_provenance())
            self.assertEqual(
                _trajectory_forecasts(loaded, _sequence(12)),
                tuple(
                    self.calibrator.transform(item)
                    for item in _trajectory_forecasts(self.fitted, _sequence(12))
                ),
            )
            (bundle / "extra.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(GRUBundleError, "extra"):
                load_gru_bundle(bundle)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = save_gru_bundle(
                self.fitted,
                root / "bundle",
                calibrator=self.calibrator.to_dict(),
                provenance=_provenance(),
            )
            weights = next(bundle.rglob("weights.safetensors"))
            target = root / "outside.safetensors"
            target.write_bytes(weights.read_bytes())
            weights.unlink()
            try:
                os.symlink(target, weights)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(GRUBundleError, "unsafe"):
                load_gru_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
