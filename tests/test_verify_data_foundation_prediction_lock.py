from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_data_foundation_baseline as runner
from scripts import verify_data_foundation_prediction_lock as verifier
from tests.test_run_data_foundation_baseline import (
    _build_synthetic_results,
    _rehash_results,
)


LOCK_RELATIVE = Path("configs/data_foundation_prediction_baseline.json")
ARTIFACT_RELATIVE = Path("workspace/data_foundation/baselines/synthetic-lock")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build_fixture(root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    results, bundles = _build_synthetic_results()
    binding = results["source_binding"]

    data_lock = root / binding["baseline_lock_path"]
    _write_json(data_lock, {"fixture": "data-lock"})
    binding["baseline_lock_file_sha256"] = runner._sha256_file(data_lock)

    audit = {
        "audit_schema_version": 1,
        "implementation": {
            "git_commit": binding["data_foundation_audit_git_commit"],
            "source_tree_sha256": binding[
                "data_foundation_audit_source_tree_sha256"
            ],
        },
    }
    audit["audit_payload_sha256"] = runner._semantic_sha256(audit)
    audit_path = root / binding["data_foundation_audit_path"]
    _write_json(audit_path, audit)
    binding["data_foundation_audit_file_sha256"] = runner._sha256_file(audit_path)
    binding["data_foundation_audit_payload_sha256"] = audit[
        "audit_payload_sha256"
    ]

    for index, (name, source) in enumerate(sorted(results["sources"].items())):
        descriptor = root / source["descriptor_path"]
        _write_json(descriptor, {"fixture_source": name})
        source["descriptor_file_sha256"] = runner._sha256_file(descriptor)
        manifest = root / source["manifest_path"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(f"fixture-manifest-{name}\n".encode())
        source["manifest_sha256"] = runner._sha256_file(manifest)
        raw = root / source["raw_artifact_path"]
        if source["raw_artifact_sha256_kind"] == "framed_file_index_v1":
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "fixture.bin").write_bytes(bytes([index]))
        else:
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(b"fixture raw")

    code_payloads = {
        path: f"frozen-code:{path}\n".encode() for path in binding["code_paths"]
    }
    binding["runner_and_src_code_tree_sha256"] = runner._framed_code_hash(
        sorted(code_payloads.items())
    )
    controls = verifier._control_paths(results)
    control_items = [
        (path, (root / path).read_bytes()) for path in controls
    ]
    binding["tracked_control_tree_sha256"] = verifier._control_tree_sha256(
        control_items
    )
    _rehash_results(results)
    artifact = runner.publish_artifact(
        root,
        ARTIFACT_RELATIVE.as_posix(),
        results=results,
        bundles=bundles,
        pre_publish_check=lambda: None,
    )
    manifest = runner.verify_artifact(artifact)
    lock = verifier.build_lock_projection(
        root,
        artifact_relative=ARTIFACT_RELATIVE.as_posix(),
        manifest=manifest,
        results=results,
    )
    _write_json(root / LOCK_RELATIVE, lock)
    return code_payloads, lock


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        [runner._git_executable(), "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(f"Git fixture command failed: {arguments!r}: {result.stderr}")
    return result.stdout.strip()


class DataFoundationPredictionLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls._temporary.name) / "fixture"
        cls.fixture_root.mkdir()
        cls.code_payloads, cls.lock = _build_fixture(cls.fixture_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _copy_fixture(self, destination: Path) -> None:
        shutil.copytree(self.fixture_root, destination, dirs_exist_ok=True)

    def _verify(
        self,
        root: Path,
        *,
        workspace_drift: bool = False,
        require_workspace_source_match: bool = False,
    ) -> dict[str, object]:
        code_paths = tuple(sorted(self.code_payloads))
        code_hash = self.lock["runner"]["code_tree_sha256"]

        def fake_git_tree(
            _: Path, *, commit: str, paths: tuple[str, ...]
        ) -> verifier.GitTreeEvidence:
            canonical = tuple(sorted(paths))
            if set(canonical) == set(code_paths):
                payloads = tuple(
                    (path, self.code_payloads[path]) for path in canonical
                )
            else:
                payloads = tuple((path, (root / path).read_bytes()) for path in canonical)
            return verifier.GitTreeEvidence(commit, canonical, payloads)

        def fake_runner_tree(
            _: Path, *, commit: str, claimed_paths: tuple[str, ...]
        ) -> verifier.GitTreeEvidence:
            return fake_git_tree(_, commit=commit, paths=claimed_paths)

        workspace_paths = code_paths
        workspace_hash = code_hash
        if workspace_drift:
            workspace_paths = (*workspace_paths, "src/token_prediction/stage2_new.py")
            workspace_hash = "9" * 64
        with (
            patch.object(verifier, "frozen_git_tree", side_effect=fake_git_tree),
            patch.object(
                verifier,
                "frozen_runner_code_tree",
                side_effect=fake_runner_tree,
            ),
            patch.object(verifier, "verify_tracked_prediction_lock"),
            patch.object(
                verifier,
                "_workspace_code_tree",
                return_value=(tuple(workspace_paths), workspace_hash),
            ),
        ):
            return verifier.verify_prediction_lock(
                root,
                lock_path=LOCK_RELATIVE,
                artifact_path=ARTIFACT_RELATIVE,
                require_workspace_source_match=require_workspace_source_match,
            )

    def test_exact_synthetic_artifact_lock_closes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_fixture(root)
            summary = self._verify(root)
            self.assertEqual(summary["bundle_count"], 90)
            self.assertEqual(summary["cell_count"], 6)
            self.assertEqual(summary["gated_condition_count"], 4)
            self.assertGreater(summary["prediction_count"], 0)
            self.assertEqual(
                summary["numerical_prediction_projection_id"],
                verifier.NUMERICAL_PREDICTION_PROJECTION_ID,
            )
            results = json.loads(
                (root / ARTIFACT_RELATIVE / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["numerical_prediction_projection_sha256"],
                verifier.numerical_prediction_projection_sha256(results),
            )
            self.assertTrue(summary["workspace_source_matches_frozen"])
            rendered = json.dumps(summary, sort_keys=True)
            self.assertNotIn(str(root), rendered)
            self.assertNotIn("condition:", rendered)

    def test_repository_production_lock_and_frozen_git_objects_close(self) -> None:
        summary = verifier.verify_tracked_prediction_lock_only(verifier.REPO_ROOT)
        self.assertEqual(summary["verification_scope"], "tracked_lock_and_git_objects")
        self.assertEqual(summary["bundle_count"], 90)
        self.assertEqual(summary["estimable_condition_count"], 6)
        self.assertEqual(summary["gated_condition_count"], 4)
        self.assertGreater(summary["prediction_count"], 0)

    def test_lock_extra_fields_and_all_explicit_digests_reject_tamper(self) -> None:
        cases = (
            ("extra", lambda lock: lock.update({"unexpected": True})),
            (
                "artifact id",
                lambda lock: lock["artifact"].update({"artifact_id": "0" * 64}),
            ),
            (
                "metrics",
                lambda lock: lock["metrics"].update(
                    {"aggregate_metrics_sha256": "1" * 64}
                ),
            ),
            (
                "conditions",
                lambda lock: lock["conditions"].update(
                    {"condition_projection_sha256": "2" * 64}
                ),
            ),
            (
                "holdout",
                lambda lock: lock["holdout"].update(
                    {"policy_payload_sha256": "3" * 64}
                ),
            ),
            (
                "control",
                lambda lock: lock["runner"].update(
                    {"tracked_control_tree_sha256": "4" * 64}
                ),
            ),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._copy_fixture(root)
                lock = copy.deepcopy(self.lock)
                mutate(lock)
                _write_json(root / LOCK_RELATIVE, lock)
                with self.assertRaises(verifier.PredictionLockError):
                    self._verify(root)

    def test_artifact_and_bound_input_tamper_fail_closed(self) -> None:
        targets = (
            ARTIFACT_RELATIVE / "results.json",
            Path(self.lock["data_foundation"]["baseline_lock_path"]),
            Path(self.lock["data_foundation"]["audit_path"]),
            Path(self.lock["sources"]["bagen_swebench"]["descriptor_path"]),
            Path(self.lock["sources"]["spend_openhands"]["manifest_path"]),
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._copy_fixture(root)
                path = root / target
                path.write_bytes(path.read_bytes() + b"tamper")
                with self.assertRaises(verifier.PredictionLockError):
                    self._verify(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO support is unavailable")
    def test_artifact_special_nodes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_fixture(root)
            fifo = root / ARTIFACT_RELATIVE / "unexpected.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                verifier.PredictionLockError,
                "only regular files and directories",
            ):
                self._verify(root)

    def test_unsafe_paths_symlinks_and_privacy_identity_keys_are_rejected(self) -> None:
        for value in (
            "../artifact",
            "C:/artifact",
            "/tmp/artifact",
            "workspace\\artifact",
            " workspace/artifact",
            "workspace/artifact ",
            "workspace/\x00artifact",
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._copy_fixture(root)
                lock = copy.deepcopy(self.lock)
                lock["artifact"]["relative_path"] = value
                _write_json(root / LOCK_RELATIVE, lock)
                with self.assertRaises(verifier.PredictionLockError):
                    self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_fixture(root)
            lock = copy.deepcopy(self.lock)
            lock["point_id"] = "private"
            _write_json(root / LOCK_RELATIVE, lock)
            with self.assertRaisesRegex(
                verifier.PredictionLockError, "forbidden row-level identity"
            ):
                self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_fixture(root)
            lock_path = root / LOCK_RELATIVE
            real = lock_path.with_suffix(".real.json")
            lock_path.replace(real)
            try:
                lock_path.symlink_to(real.name)
            except OSError as exc:  # pragma: no cover - host policy dependent
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(
                verifier.PredictionLockError, "symlink|junction|reparse"
            ):
                self._verify(root)

    def test_workspace_drift_is_reported_and_optional_strict_gate_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_fixture(root)
            summary = self._verify(root, workspace_drift=True)
            self.assertFalse(summary["workspace_source_matches_frozen"])
            with self.assertRaisesRegex(
                verifier.PredictionLockError, "workspace runner/source tree differs"
            ):
                self._verify(
                    root,
                    workspace_drift=True,
                    require_workspace_source_match=True,
                )

    def test_historical_commit_object_and_blob_drift_are_proven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _git(root, "init", "--quiet")
            _git(root, "config", "user.name", "Fixture")
            _git(root, "config", "user.email", "fixture@example.invalid")
            _git(root, "config", "core.autocrlf", "false")
            paths = (
                "configs/control.json",
                runner.RUNNER_RELATIVE,
                "src/token_prediction/__init__.py",
            )
            for index, relative in enumerate(paths):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"VALUE = {index}\n".encode())
            _git(root, "add", "--", *paths)
            _git(root, "commit", "--quiet", "-m", "prediction fixture")
            commit = _git(root, "rev-parse", "HEAD")
            evidence = verifier.frozen_git_tree(root, commit=commit, paths=paths)
            self.assertEqual(evidence.paths, tuple(sorted(paths)))
            self.assertEqual(len(evidence.payloads), 3)
            code_paths = (
                runner.RUNNER_RELATIVE,
                "src/token_prediction/__init__.py",
            )
            code_evidence = verifier.frozen_runner_code_tree(
                root,
                commit=commit,
                claimed_paths=code_paths,
            )
            self.assertEqual(code_evidence.paths, tuple(sorted(code_paths)))
            with self.assertRaisesRegex(
                verifier.PredictionLockError, "full frozen runner/source tree"
            ):
                verifier.frozen_runner_code_tree(
                    root,
                    commit=commit,
                    claimed_paths=(runner.RUNNER_RELATIVE,),
                )
            with self.assertRaises(verifier.PredictionLockError):
                verifier.frozen_git_tree(root, commit="f" * 40, paths=paths)

            changed = root / runner.RUNNER_RELATIVE
            changed.write_bytes(b"VALUE = 'drift'\n")
            _git(root, "add", "--", runner.RUNNER_RELATIVE)
            _git(root, "commit", "--quiet", "-m", "prediction drift")
            drift_commit = _git(root, "rev-parse", "HEAD")
            drift = verifier.frozen_git_tree(root, commit=drift_commit, paths=paths)
            self.assertNotEqual(
                runner._framed_code_hash(evidence.payloads),
                runner._framed_code_hash(drift.payloads),
            )

    def test_prediction_lock_must_be_tracked_clean_and_equal_to_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _git(root, "init", "--quiet")
            _git(root, "config", "user.name", "Fixture")
            _git(root, "config", "user.email", "fixture@example.invalid")
            _git(root, "config", "core.autocrlf", "false")
            lock = root / LOCK_RELATIVE
            _write_json(lock, {"tracked": True})
            frozen_bytes = lock.read_bytes()
            _git(root, "add", "--", LOCK_RELATIVE.as_posix())
            _git(root, "commit", "--quiet", "-m", "tracked prediction lock")
            verifier.verify_tracked_prediction_lock(
                root, LOCK_RELATIVE.as_posix(), lock
            )

            lock.write_bytes(frozen_bytes + b" ")
            with self.assertRaisesRegex(verifier.PredictionLockError, "dirty"):
                verifier.verify_tracked_prediction_lock(
                    root, LOCK_RELATIVE.as_posix(), lock
                )

            lock.write_bytes(frozen_bytes)
            untracked_relative = "configs/untracked_prediction_lock.json"
            untracked = root / untracked_relative
            _write_json(untracked, {"tracked": False})
            with self.assertRaises(verifier.PredictionLockError):
                verifier.verify_tracked_prediction_lock(
                    root, untracked_relative, untracked
                )


if __name__ == "__main__":
    unittest.main()
