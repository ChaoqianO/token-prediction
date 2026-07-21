from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
import unittest
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import token_prediction.estimators.neural_bundle as neural_bundle_module
from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    prediction_input_contract_hash_from_capability,
)
from token_prediction.estimators.base import FitContext, RunContext
from token_prediction.estimators.mlp import IndependentMLPQuantileEstimator
from token_prediction.estimators.neural_bundle import (
    NeuralBundleError,
    load_neural_bundle,
    neural_bundle_files,
    save_neural_bundle,
)
from token_prediction.estimators.neural_encoder import OptionalNeuralDependencyError
from token_prediction.features import FEATURE_SCHEMA_VERSION

from tests.test_mlp_estimator import HAS_NEURAL, _point, _view


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
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
        newline="\n",
    )


class NeuralBundleOptionalDependencyTests(unittest.TestCase):
    @unittest.skipIf(HAS_NEURAL, "only applies to the base-only environment")
    def test_bundle_builder_fails_closed_without_neural_extra(self) -> None:
        with self.assertRaises(OptionalNeuralDependencyError):
            # Dependency loading occurs before fitted-model access.
            neural_bundle_files(None)  # type: ignore[arg-type]


class NeuralBundleMetadataTests(unittest.TestCase):
    def test_zero_windows_identity_fails_closed(self) -> None:
        enumerated = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o644,
            st_size=65,
            st_dev=0,
            st_ino=0,
            st_nlink=0,
            st_mtime_ns=100,
            st_ctime_ns=200,
        )
        inspected = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o644,
            st_size=65,
            st_dev=0,
            st_ino=0,
            st_nlink=0,
            st_mtime_ns=101,
            st_ctime_ns=201,
        )
        changed_size = SimpleNamespace(**{**vars(inspected), "st_size": 66})

        self.assertFalse(neural_bundle_module._same_snapshot(enumerated, inspected))
        self.assertFalse(
            neural_bundle_module._same_snapshot(enumerated, changed_size)
        )


@unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
class NeuralBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_descriptor = SourceDescriptor(
            source_id="neural-fixture",
            revision="revision-1",
            manifest_path="workspace/neural-fixture.json",
            manifest_sha256="9" * 64,
            capabilities=SourceCapabilities(
                source_id="neural-fixture",
                observables=frozenset(
                    {
                        Observable.ATTEMPT_USAGE,
                        Observable.REQUEST_BOUNDARIES,
                        Observable.REQUEST_LOCAL_COUNT,
                        Observable.TASK_TERMINATION,
                    }
                ),
            ),
        )
        cls.input_contract_hash = prediction_input_contract_hash_from_capability(
            capability_contract_hash=cls.source_descriptor.capabilities.contract_hash
        )
        fitted = IndependentMLPQuantileEstimator(
            hidden_dims=(16, 8), max_epochs=30, patience=6
        ).fit(
            _view(range(40)),
            _view(range(40, 50)),
            FitContext(seed=23, fold=1, interval_alpha=0.2),
        )
        cls.fitted = replace(fitted, input_contract_hash=cls.input_contract_hash)
        cls.calibrator = {
            "calibrator_schema_version": 1,
            "calibrator_id": "task_max_conformal",
            "interval_alpha": 0.2,
            "expansion": 1.5,
        }
        cls.provenance = {
            "bundle_role": "point_model",
            "candidate_id": "independent-mlp-test",
            "candidate_hash": "b" * 64,
            "candidate_graph": {
                "initializer_estimator_id": "none",
                "updater_estimator_id": "independent_mlp",
                "lifecycle_schema_id": "point_cell_v1",
                "seed_policy_id": "none",
                "inner_split_policy_id": "none",
            },
            "dataset_id": "mlp-dataset",
            "dataset_schema_version": CAPABILITY_DATASET_SCHEMA_VERSION,
            "source_descriptor": cls.source_descriptor.to_dict(),
            "source_descriptor_hash": cls.source_descriptor.descriptor_hash,
            "capability_contract_hash": cls.source_descriptor.capabilities.contract_hash,
            "input_contract_hash": cls.input_contract_hash,
            "split_plan_id": "e" * 64,
            "eligibility_hash": "f" * 64,
            "feature_set_hash": "1" * 64,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "position": "task_pre",
            "target": "task_unknown_remaining_tokens",
            "condition_id": "condition-a",
            "fold": 1,
            "interval_alpha": 0.2,
            "calibrator_id": "task_max_conformal",
            "code_hash": "2" * 64,
        }
        cls.source_provenance = {
            "source_descriptor": cls.source_descriptor.to_dict(),
            "source_descriptor_hash": cls.source_descriptor.descriptor_hash,
            "code_hash": "2" * 64,
            "runtime_versions": {
                "python_version": platform.python_version(),
                "token_prediction_version": TOKEN_PREDICTION_VERSION,
                "numpy_version": cls.fitted.fit_report.numpy_version,
                "torch_version": cls.fitted.fit_report.torch_version,
                "safetensors_version": version("safetensors"),
            },
        }

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="neural-bundle-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.bundle = save_neural_bundle(
            self.fitted,
            self.root / "bundle",
            calibrator=self.calibrator,
            provenance=self.provenance,
        )

    def test_nested_safetensors_round_trip_is_exact(self) -> None:
        point = _point(52)
        context = RunContext(
            point.task_id,
            point.trajectory_id,
            point.run_id,
            dataset_id=self.fitted.dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            input_contract_hash=self.fitted.input_contract_hash,
        )
        raw = self.fitted.start(context).predict(point)
        from token_prediction.evaluation.calibration import FittedExpansionCalibrator

        expected = FittedExpansionCalibrator.from_dict(self.calibrator).transform(raw)
        loaded = load_neural_bundle(
            self.bundle,
            expected_source_provenance=self.source_provenance,
        )
        actual = loaded.start(context).predict(point)

        self.assertEqual(actual, expected)
        self.assertEqual(loaded.dataset_id, self.fitted.dataset_id)
        self.assertEqual(loaded.input_contract_hash, self.fitted.input_contract_hash)
        self.assertEqual(loaded.fit_report, self.fitted.fit_report)
        self.assertEqual(dict(loaded.calibrator_document or {}), self.calibrator)
        self.assertEqual(dict(loaded.provenance or {}), self.provenance)
        weights = next(self.bundle.rglob("weights.safetensors"))
        self.assertNotIn(b"pickle", weights.read_bytes().lower())

    def test_in_memory_file_mapping_is_nested_complete_and_immutable(self) -> None:
        files = neural_bundle_files(
            self.fitted,
            calibrator=self.calibrator,
            provenance=self.provenance,
        )
        self.assertEqual(set(files), {path.relative_to(self.bundle).as_posix() for path in self.bundle.rglob("*") if path.is_file()})
        self.assertTrue(any(path.startswith("components/") for path in files))
        self.assertTrue(any(path.endswith("weights.safetensors") for path in files))
        with self.assertRaises(TypeError):
            files["extra"] = b"forbidden"  # type: ignore[index]

    def test_manifest_component_calibrator_and_weights_tampering_fail(self) -> None:
        cases = (
            self.bundle / "manifest.json",
            self.bundle / "calibrator.json",
            next(self.bundle.rglob("component.json")),
            next(self.bundle.rglob("weights.safetensors")),
        )
        for index, original in enumerate(cases):
            if index:
                shutil.rmtree(self.bundle)
                save_neural_bundle(
                    self.fitted,
                    self.bundle,
                    calibrator=self.calibrator,
                    provenance=self.provenance,
                )
                original = next(self.bundle.rglob(original.name))
            original.write_bytes(original.read_bytes() + b"\n")
            with self.subTest(filename=original.name):
                with self.assertRaises(NeuralBundleError):
                    load_neural_bundle(self.bundle)

    def test_generation_rejects_empty_provenance_and_mismatched_calibrator(self) -> None:
        with self.assertRaisesRegex(NeuralBundleError, "provenance"):
            neural_bundle_files(
                self.fitted,
                calibrator=self.calibrator,
                provenance={},
            )
        with self.assertRaisesRegex(NeuralBundleError, "provenance"):
            neural_bundle_files(
                self.fitted,
                calibrator=self.calibrator,
                provenance={"x": "y"},
            )
        bad_calibrator = dict(self.calibrator)
        bad_calibrator["interval_alpha"] = 0.1
        with self.assertRaisesRegex(NeuralBundleError, "alpha"):
            neural_bundle_files(
                self.fitted,
                calibrator=bad_calibrator,
                provenance=self.provenance,
            )

    def test_provenance_scope_mismatch_fails_closed(self) -> None:
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["provenance"]["dataset_id"] = "another-dataset"
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "dataset id"):
            load_neural_bundle(self.bundle)

    def test_resigned_schema_capability_and_input_contract_tampering_fail(self) -> None:
        original = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        cases = (
            ("dataset_schema_version", 999, "dataset schema"),
            ("feature_schema_version", 999, "feature schema"),
        )
        for field, value, message in cases:
            manifest = json.loads(json.dumps(original))
            manifest["provenance"][field] = value
            _rewrite_manifest(self.bundle, manifest)
            with self.subTest(field=field):
                with self.assertRaisesRegex(NeuralBundleError, message):
                    load_neural_bundle(self.bundle)
            _rewrite_manifest(self.bundle, original)

        manifest = json.loads(json.dumps(original))
        manifest["provenance"]["capability_contract_hash"] = "0" * 64
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "capability contract"):
            load_neural_bundle(self.bundle)
        _rewrite_manifest(self.bundle, original)

        insufficient = replace(
            self.source_descriptor,
            capabilities=SourceCapabilities(
                source_id=self.source_descriptor.source_id,
                observables=frozenset(
                    {
                        Observable.ATTEMPT_USAGE,
                        Observable.REQUEST_BOUNDARIES,
                        Observable.TASK_TERMINATION,
                    }
                ),
            ),
        )
        insufficient_contract = prediction_input_contract_hash_from_capability(
            capability_contract_hash=insufficient.capabilities.contract_hash
        )
        manifest = json.loads(json.dumps(original))
        manifest["provenance"]["source_descriptor"] = insufficient.to_dict()
        manifest["provenance"]["source_descriptor_hash"] = insufficient.descriptor_hash
        manifest["provenance"]["capability_contract_hash"] = (
            insufficient.capabilities.contract_hash
        )
        manifest["provenance"]["input_contract_hash"] = insufficient_contract
        manifest["scope"]["input_contract_hash"] = insufficient_contract
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "cannot produce"):
            load_neural_bundle(self.bundle)
        _rewrite_manifest(self.bundle, original)

        manifest = json.loads(json.dumps(original))
        manifest["scope"]["input_contract_hash"] = "0" * 64
        manifest["provenance"]["input_contract_hash"] = "0" * 64
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "input contract"):
            load_neural_bundle(self.bundle)

    def test_runtime_and_expected_source_provenance_are_anchored(self) -> None:
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["runtime"]["python_version"] = "0.0.0-impossible"
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "Python.*incompatible"):
            load_neural_bundle(self.bundle)

        shutil.rmtree(self.bundle)
        save_neural_bundle(
            self.fitted,
            self.bundle,
            calibrator=self.calibrator,
            provenance=self.provenance,
        )
        original = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(json.dumps(original))
        torch_major = manifest["runtime"]["torch_version"].split(".", 1)[0]
        manifest["runtime"]["torch_version"] = f"{torch_major}.999.0"
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "expected provenance"):
            load_neural_bundle(
                self.bundle,
                expected_source_provenance=self.source_provenance,
            )
        _rewrite_manifest(self.bundle, original)

        unexpected = {**self.source_provenance, "code_hash": "0" * 64}
        with self.assertRaisesRegex(NeuralBundleError, "expected provenance"):
            load_neural_bundle(
                self.bundle,
                expected_source_provenance=unexpected,
            )

    def test_bundle_tree_entry_and_depth_limits_fail_closed(self) -> None:
        for index in range(65):
            (self.bundle / f"extra-{index}").mkdir()
        with self.assertRaisesRegex(NeuralBundleError, "entry-count"):
            load_neural_bundle(self.bundle)

        shutil.rmtree(self.bundle)
        save_neural_bundle(
            self.fitted,
            self.bundle,
            calibrator=self.calibrator,
            provenance=self.provenance,
        )
        nested = self.bundle
        for index in range(9):
            nested = nested / f"depth-{index}"
            nested.mkdir()
        with self.assertRaisesRegex(NeuralBundleError, "directory-depth"):
            load_neural_bundle(self.bundle)

    def test_manifest_checksum_whitespace_and_oversize_fail_closed(self) -> None:
        checksum = self.bundle / "manifest.sha256"
        checksum.write_bytes(checksum.read_bytes() + b" \n")
        with self.assertRaisesRegex(NeuralBundleError, "checksum|size"):
            load_neural_bundle(self.bundle)

        shutil.rmtree(self.bundle)
        save_neural_bundle(
            self.fitted,
            self.bundle,
            calibrator=self.calibrator,
            provenance=self.provenance,
        )
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["provenance"]["oversized"] = "x" * 1_100_000
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "size"):
            load_neural_bundle(self.bundle)

    def test_fit_report_must_match_architecture(self) -> None:
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["fit_report"]["parameters"]["hidden_dims"] = [999, 2]
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "fit report"):
            load_neural_bundle(self.bundle)

    def test_missing_extra_files_and_extra_directories_fail(self) -> None:
        next(self.bundle.rglob("weights.safetensors")).unlink()
        with self.assertRaisesRegex(NeuralBundleError, "file set"):
            load_neural_bundle(self.bundle)

        shutil.rmtree(self.bundle)
        save_neural_bundle(
            self.fitted,
            self.bundle,
            calibrator=self.calibrator,
            provenance=self.provenance,
        )
        (self.bundle / "notes.txt").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(NeuralBundleError, "file set"):
            load_neural_bundle(self.bundle)

        (self.bundle / "notes.txt").unlink()
        (self.bundle / "empty-extra-directory").mkdir()
        with self.assertRaisesRegex(NeuralBundleError, "file set"):
            load_neural_bundle(self.bundle)

    def test_backslash_traversal_and_unknown_manifest_fields_fail(self) -> None:
        original = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        for invalid in ("components\\bad", "components/../bad", "/components/bad"):
            manifest = json.loads(json.dumps(original))
            manifest["component"]["path"] = invalid
            _rewrite_manifest(self.bundle, manifest)
            with self.subTest(path=invalid):
                with self.assertRaisesRegex(NeuralBundleError, "POSIX|backslashes"):
                    load_neural_bundle(self.bundle)

        manifest = json.loads(json.dumps(original))
        manifest["unexpected"] = True
        _rewrite_manifest(self.bundle, manifest)
        with self.assertRaisesRegex(NeuralBundleError, "keys do not match"):
            load_neural_bundle(self.bundle)

    def test_symlink_or_reparse_entry_fails_when_supported(self) -> None:
        weights = next(self.bundle.rglob("weights.safetensors"))
        target = self.root / "external.safetensors"
        shutil.copyfile(weights, target)
        weights.unlink()
        try:
            os.symlink(target, weights)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable on this host: {exc}")
        with self.assertRaisesRegex(NeuralBundleError, "symlink|reparse"):
            load_neural_bundle(self.bundle)

    def test_component_directory_swap_to_symlink_during_load_fails(self) -> None:
        manifest = json.loads((self.bundle / "manifest.json").read_text("utf-8"))
        component_relative = manifest["component"]["path"]
        component = self.bundle.joinpath(*component_relative.split("/"))
        outside = self.root / "outside-component"
        detached = self.root / "detached-component"
        shutil.copytree(component, outside)
        probe = self.root / "directory-symlink-probe"
        try:
            os.symlink(outside, probe, target_is_directory=True)
            probe.unlink()
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable on this host: {exc}")

        original = neural_bundle_module._validated_manifest
        swapped = False

        def swap_after_manifest(root: Path, tree: object) -> object:
            nonlocal swapped
            result = original(root, tree)
            component.rename(detached)
            os.symlink(outside, component, target_is_directory=True)
            swapped = True
            return result

        with patch(
            "token_prediction.estimators.neural_bundle._validated_manifest",
            side_effect=swap_after_manifest,
        ):
            with self.assertRaisesRegex(
                NeuralBundleError,
                "changed|symlink|reparse|real directory",
            ):
                load_neural_bundle(self.bundle)
        self.assertTrue(swapped)

    def test_same_size_regular_file_swap_after_snapshot_fails(self) -> None:
        victim = self.bundle / "calibrator.json"
        detached = self.root / "detached-calibrator.json"
        replacement = self.root / "replacement-calibrator.json"
        replacement_payload = victim.read_bytes().replace(b"1.5", b"9.5")
        self.assertEqual(len(replacement_payload), victim.stat().st_size)
        replacement.write_bytes(replacement_payload)
        original = neural_bundle_module._validated_manifest
        swapped = False

        def swap_after_manifest(root: Path, tree: object) -> object:
            nonlocal swapped
            result = original(root, tree)
            victim.rename(detached)
            replacement.rename(victim)
            swapped = True
            return result

        with patch(
            "token_prediction.estimators.neural_bundle._validated_manifest",
            side_effect=swap_after_manifest,
        ):
            with self.assertRaisesRegex(NeuralBundleError, "changed"):
                load_neural_bundle(self.bundle)
        self.assertTrue(swapped)

    def test_zero_identity_direntry_is_upgraded_by_path_and_handle(self) -> None:
        if neural_bundle_module._reliable_identity(self.bundle.lstat()) is None:
            self.skipTest("bundle filesystem does not expose a reliable path identity")
        original_scandir = os.scandir

        class ZeroIdentityEntry:
            def __init__(self, entry: object) -> None:
                self._entry = entry
                self.name = entry.name  # type: ignore[attr-defined]
                self.path = entry.path  # type: ignore[attr-defined]

            def stat(self, *, follow_symlinks: bool = True) -> object:
                metadata = self._entry.stat(  # type: ignore[attr-defined]
                    follow_symlinks=follow_symlinks
                )
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_size=metadata.st_size,
                    st_dev=0,
                    st_ino=0,
                    st_nlink=0,
                    st_mtime_ns=metadata.st_mtime_ns,
                    st_ctime_ns=metadata.st_ctime_ns,
                    st_file_attributes=getattr(metadata, "st_file_attributes", 0),
                )

        class ZeroIdentityScandir:
            def __init__(self, path: object) -> None:
                self._context = original_scandir(path)

            def __enter__(self) -> object:
                entries = self._context.__enter__()
                return iter(ZeroIdentityEntry(entry) for entry in entries)

            def __exit__(self, *args: object) -> object:
                return self._context.__exit__(*args)

        with patch(
            "token_prediction.estimators.neural_bundle.os.scandir",
            side_effect=ZeroIdentityScandir,
        ):
            loaded = load_neural_bundle(
                self.bundle,
                expected_source_provenance=self.source_provenance,
            )
        self.assertEqual(loaded.dataset_id, self.fitted.dataset_id)

    def test_save_requires_fresh_destination(self) -> None:
        with self.assertRaises(FileExistsError):
            save_neural_bundle(self.fitted, self.bundle)


if __name__ == "__main__":
    unittest.main()
