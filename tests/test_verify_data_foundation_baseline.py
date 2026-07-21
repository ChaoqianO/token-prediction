from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.audit_data_foundation_v2 import (
    BUILD_COMMAND,
    ArtifactEvidence,
    _default_git_executable,
    _source_tree_hash_from_file_hashes,
    build_data_foundation_audit,
    build_source_audit,
)
from scripts.verify_data_foundation_baseline import (
    DataFoundationAuditError,
    FULL_SOURCE_LOADERS,
    GitSourceEvidence,
    _assert_privacy_safe,
    strict_workspace_source_tree_sha256,
    verify_data_foundation_baseline,
    verify_frozen_git_source_tree,
)
from tests.helpers import make_two_call_trajectory
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor


FIXTURE_COMMIT = "1" * 40
FIXTURE_TREE_HASH = "2" * 64
FIXTURE_SOURCE_PATHS = (
    "scripts/audit_data_foundation_v2.py",
    *(f"src/token_prediction/fixture_{index:02d}.py" for index in range(41)),
)
BASELINE_RELATIVE = Path("configs/data_foundation_v2_baseline.json")
AUDIT_RELATIVE = Path("workspace/data_foundation/data_foundation_v2_audit.json")
RERUN_RELATIVE = Path(
    "workspace/data_foundation/data_foundation_v2_audit_rerun.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _semantic_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(rendered)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(_default_git_executable()), "-C", str(root), *arguments],
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


def _make_git_source_fixture(root: Path) -> tuple[str, str, tuple[str, ...]]:
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    paths = tuple(FIXTURE_SOURCE_PATHS)
    file_hashes: dict[str, str] = {}
    for index, relative in enumerate(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"VALUE = {index}\n".encode()
        path.write_bytes(content)
        file_hashes[relative] = _sha256_bytes(content)
    _git(root, "add", "--", "scripts/audit_data_foundation_v2.py", "src/token_prediction")
    _git(root, "commit", "--quiet", "-m", "fixture source")
    commit = _git(root, "rev-parse", "HEAD")
    return commit, _source_tree_hash_from_file_hashes(file_hashes), paths


def _refresh_audit_payload(audit: dict[str, object]) -> None:
    payload = copy.deepcopy(audit)
    payload.pop("audit_payload_sha256", None)
    audit["audit_payload_sha256"] = _semantic_sha256(payload)


def _baseline_source(source: dict[str, object]) -> dict[str, object]:
    descriptor = source["source_descriptor"]
    dataset = source["dataset"]
    identity = source["identity_counts"]
    artifacts = source["artifacts"]
    assert isinstance(descriptor, dict)
    assert isinstance(dataset, dict)
    assert isinstance(identity, dict)
    assert isinstance(artifacts, dict)
    descriptor_artifact = artifacts["descriptor"]
    assert isinstance(descriptor_artifact, dict)
    manifest = descriptor["manifest"]
    assert isinstance(manifest, dict)
    return {
        "capability_contract_hash": source["capability_contract_hash"],
        "condition_count": identity["condition_count"],
        "dataset_id": dataset["dataset_id"],
        "dataset_status_counts": dict(dataset["status_counts"]),
        "descriptor_file_sha256": descriptor_artifact["sha256"],
        "manifest_sha256": manifest["sha256"],
        "revision": descriptor["revision"],
        "row_count": dataset["row_count"],
        "run_count": identity["run_count"],
        "source_descriptor_hash": source["source_descriptor_hash"],
        "source_id": descriptor["source_id"],
        "task_count": identity["task_count"],
        "trajectory_count": identity["trajectory_count"],
    }


def _refresh_baseline_audit_pin(
    root: Path,
    baseline: dict[str, object],
    audit: dict[str, object],
) -> None:
    audit_path = root / AUDIT_RELATIVE
    _write_json(audit_path, audit)
    rerun_path = root / RERUN_RELATIVE
    _write_json(rerun_path, audit)
    production = baseline["production_audit"]
    assert isinstance(production, dict)
    audit_bytes = audit_path.read_bytes()
    production["audit_payload_sha256"] = audit["audit_payload_sha256"]
    production["bytes"] = len(audit_bytes)
    production["file_sha256"] = _sha256_bytes(audit_bytes)
    rerun_bytes = rerun_path.read_bytes()
    production["rerun_bytes"] = len(rerun_bytes)
    production["rerun_file_sha256"] = _sha256_bytes(rerun_bytes)
    _write_json(root / BASELINE_RELATIVE, baseline)


def _make_fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest_relative = "workspace/fixtures/manifest.json"
    manifest_path = root / manifest_relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"fixture manifest\n")
    manifest_sha = _sha256_bytes(manifest_path.read_bytes())

    capabilities = SourceCapabilities(
        source_id="fixture-source",
        observables=frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
                Observable.TASK_USAGE,
            }
        ),
    )
    descriptor = SourceDescriptor(
        source_id="fixture-source",
        revision="fixture-revision",
        manifest_path=manifest_relative,
        manifest_sha256=manifest_sha,
        capabilities=capabilities,
    )
    descriptor_relative = "configs/source_descriptors/fixture.json"
    descriptor_path = root / descriptor_relative
    _write_json(descriptor_path, descriptor.to_dict())
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor_sha = _sha256_bytes(descriptor_bytes)

    source = build_source_audit(
        source_name="fixture_source",
        trajectories=(make_two_call_trajectory(0),),
        descriptor=descriptor,
        artifacts={
            "descriptor": ArtifactEvidence(
                path=descriptor_relative,
                bytes=len(descriptor_bytes),
                sha256=descriptor_sha,
            ),
            "manifest": ArtifactEvidence(
                path=manifest_relative,
                bytes=manifest_path.stat().st_size,
                sha256=manifest_sha,
            ),
        },
    )
    audit = build_data_foundation_audit(
        {"fixture_source": source},
        git_commit=FIXTURE_COMMIT,
        source_tree_sha256=FIXTURE_TREE_HASH,
        runtime={"python_implementation": "CPython", "python_version": "3.11.0"},
    )
    audit_path = root / AUDIT_RELATIVE
    _write_json(audit_path, audit)
    rerun_path = root / RERUN_RELATIVE
    _write_json(rerun_path, audit)
    audit_bytes = audit_path.read_bytes()
    rerun_bytes = rerun_path.read_bytes()
    baseline = {
        "baseline_schema_version": 1,
        "baseline_type": "data_foundation_v2",
        "build_command": BUILD_COMMAND,
        "implementation": {
            "git_commit": FIXTURE_COMMIT,
            "git_source_binding_policy": "tracked_clean_head_blob_tree_v1",
            "source_blob_count": 42,
            "source_tree_sha256": FIXTURE_TREE_HASH,
        },
        "production_audit": {
            "audit_payload_sha256": audit["audit_payload_sha256"],
            "bytes": len(audit_bytes),
            "deterministic_run_count": 2,
            "file_sha256": _sha256_bytes(audit_bytes),
            "relative_path": AUDIT_RELATIVE.as_posix(),
            "rerun_byte_identical": True,
            "rerun_bytes": len(rerun_bytes),
            "rerun_file_sha256": _sha256_bytes(rerun_bytes),
            "rerun_relative_path": RERUN_RELATIVE.as_posix(),
        },
        "sources": {"fixture_source": _baseline_source(source)},
    }
    _write_json(root / BASELINE_RELATIVE, baseline)
    return baseline, audit


class DataFoundationBaselineVerifierTests(unittest.TestCase):
    def _verify(self, root: Path) -> dict[str, object]:
        evidence = GitSourceEvidence(
            FIXTURE_COMMIT,
            tuple(FIXTURE_SOURCE_PATHS),
            FIXTURE_TREE_HASH,
        )
        with (
            patch(
                "scripts.verify_data_foundation_baseline.verify_frozen_git_source_tree",
                return_value=evidence,
            ),
            patch(
                "scripts.verify_data_foundation_baseline.strict_workspace_source_tree_sha256",
                return_value=FIXTURE_TREE_HASH,
            ),
        ):
            return verify_data_foundation_baseline(
                root,
                baseline_path=BASELINE_RELATIVE,
                audit_path=AUDIT_RELATIVE,
            )

    def test_synthetic_lock_audit_descriptor_and_manifest_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, audit = _make_fixture(root)
            summary = self._verify(root)
            self.assertEqual(summary["source_count"], 1)
            self.assertFalse(summary["raw_artifacts_rehashed"])
            self.assertTrue(summary["workspace_source_matches_frozen"])
            self.assertEqual(summary["audit_payload_sha256"], audit["audit_payload_sha256"])
            sources = summary["sources"]
            self.assertIsInstance(sources, dict)
            self.assertEqual(set(sources), {"fixture_source"})
            rendered = json.dumps(summary, sort_keys=True)
            self.assertNotIn("task-0", rendered)
            self.assertNotIn(str(root), rendered)

    def test_full_source_mode_rebuilds_exact_summaries_and_marks_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, audit = _make_fixture(root)
            expected_source = copy.deepcopy(audit["sources"]["fixture_source"])
            loader = Mock(return_value=expected_source)
            evidence = GitSourceEvidence(
                FIXTURE_COMMIT,
                tuple(FIXTURE_SOURCE_PATHS),
                FIXTURE_TREE_HASH,
            )
            with (
                patch(
                    "scripts.verify_data_foundation_baseline.verify_frozen_git_source_tree",
                    return_value=evidence,
                ),
                patch(
                    "scripts.verify_data_foundation_baseline.strict_workspace_source_tree_sha256",
                    return_value=FIXTURE_TREE_HASH,
                ),
                patch.dict(
                    FULL_SOURCE_LOADERS,
                    {"fixture_source": loader},
                    clear=True,
                ),
            ):
                summary = verify_data_foundation_baseline(
                    root,
                    baseline_path=BASELINE_RELATIVE,
                    audit_path=AUDIT_RELATIVE,
                    full_source_verify=True,
                )
            self.assertTrue(summary["raw_artifacts_rehashed"])
            loader.assert_called_once_with(root.resolve())

            mismatched = copy.deepcopy(expected_source)
            mismatched["source_name"] = "drifted"
            loader = Mock(return_value=mismatched)
            with (
                patch(
                    "scripts.verify_data_foundation_baseline.verify_frozen_git_source_tree",
                    return_value=evidence,
                ),
                patch(
                    "scripts.verify_data_foundation_baseline.strict_workspace_source_tree_sha256",
                    return_value=FIXTURE_TREE_HASH,
                ),
                patch.dict(
                    FULL_SOURCE_LOADERS,
                    {"fixture_source": loader},
                    clear=True,
                ),
                self.assertRaisesRegex(DataFoundationAuditError, "full source verification mismatch"),
            ):
                verify_data_foundation_baseline(
                    root,
                    baseline_path=BASELINE_RELATIVE,
                    audit_path=AUDIT_RELATIVE,
                    full_source_verify=True,
                )

    def test_historical_mode_allows_normal_source_drift_but_strict_and_full_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            package = root / "src/token_prediction"
            package.mkdir(parents=True)
            (package / "__init__.py").write_bytes(b"\n")
            (package / "stage2_new.py").write_bytes(b"STAGE = 2\n")
            audit_script = root / "scripts/audit_data_foundation_v2.py"
            audit_script.parent.mkdir(parents=True)
            audit_script.write_bytes(b"# historical audit\n")
            evidence = GitSourceEvidence(
                FIXTURE_COMMIT,
                tuple(FIXTURE_SOURCE_PATHS),
                FIXTURE_TREE_HASH,
            )
            with patch(
                "scripts.verify_data_foundation_baseline.verify_frozen_git_source_tree",
                return_value=evidence,
            ):
                summary = verify_data_foundation_baseline(
                    root,
                    baseline_path=BASELINE_RELATIVE,
                    audit_path=AUDIT_RELATIVE,
                )
                self.assertFalse(summary["workspace_source_matches_frozen"])
                with self.assertRaisesRegex(
                    DataFoundationAuditError, "does not match the frozen implementation"
                ):
                    verify_data_foundation_baseline(
                        root,
                        baseline_path=BASELINE_RELATIVE,
                        audit_path=AUDIT_RELATIVE,
                        require_workspace_source_match=True,
                    )
                with self.assertRaisesRegex(
                    DataFoundationAuditError, "does not match the frozen implementation"
                ):
                    verify_data_foundation_baseline(
                        root,
                        baseline_path=BASELINE_RELATIVE,
                        audit_path=AUDIT_RELATIVE,
                        full_source_verify=True,
                    )

    def test_frozen_git_commit_object_and_all_42_blobs_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit, tree_hash, paths = _make_git_source_fixture(root)
            evidence = verify_frozen_git_source_tree(
                root,
                git_commit=commit,
                expected_source_tree_sha256=tree_hash,
                expected_blob_count=42,
            )
            self.assertEqual(evidence.git_commit, commit)
            self.assertEqual(evidence.paths, tuple(sorted(paths)))
            self.assertEqual(evidence.blob_count, 42)
            self.assertEqual(evidence.source_tree_sha256, tree_hash)

            with self.assertRaisesRegex(DataFoundationAuditError, "commit object"):
                verify_frozen_git_source_tree(
                    root,
                    git_commit="f" * 40,
                    expected_source_tree_sha256=tree_hash,
                    expected_blob_count=42,
                )

            blob_id = _git(root, "rev-parse", f"{commit}:src/token_prediction/fixture_00.py")
            with self.assertRaisesRegex(DataFoundationAuditError, "commit object"):
                verify_frozen_git_source_tree(
                    root,
                    git_commit=blob_id,
                    expected_source_tree_sha256=tree_hash,
                    expected_blob_count=42,
                )

            changed = root / "src/token_prediction/fixture_00.py"
            changed.write_bytes(b"VALUE = 'drift'\n")
            _git(root, "add", "--", "src/token_prediction/fixture_00.py")
            _git(root, "commit", "--quiet", "-m", "blob drift")
            drift_commit = _git(root, "rev-parse", "HEAD")
            with self.assertRaisesRegex(DataFoundationAuditError, "blob source tree hash"):
                verify_frozen_git_source_tree(
                    root,
                    git_commit=drift_commit,
                    expected_source_tree_sha256=tree_hash,
                    expected_blob_count=42,
                )

    def test_baseline_and_audit_extra_keys_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            for location in ("baseline", "baseline source", "audit", "audit dataset"):
                with self.subTest(location=location):
                    candidate_baseline = copy.deepcopy(baseline)
                    candidate_audit = copy.deepcopy(audit)
                    if location == "baseline":
                        candidate_baseline["unexpected"] = True
                    elif location == "baseline source":
                        candidate_baseline["sources"]["fixture_source"]["unexpected"] = True
                    elif location == "audit":
                        candidate_audit["unexpected"] = True
                        _refresh_audit_payload(candidate_audit)
                    else:
                        candidate_audit["sources"]["fixture_source"]["dataset"][
                            "unexpected"
                        ] = True
                        _refresh_audit_payload(candidate_audit)
                    _refresh_baseline_audit_pin(root, candidate_baseline, candidate_audit)
                    with self.assertRaisesRegex(DataFoundationAuditError, "keys"):
                        self._verify(root)
            _refresh_baseline_audit_pin(root, baseline, audit)

    def test_audit_file_and_payload_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            audit_path = root / AUDIT_RELATIVE
            audit_path.write_bytes(audit_path.read_bytes() + b" ")
            with self.assertRaisesRegex(DataFoundationAuditError, "byte size|SHA-256"):
                self._verify(root)

            tampered = copy.deepcopy(audit)
            tampered["source_count"] = 2
            _refresh_baseline_audit_pin(root, baseline, tampered)
            with self.assertRaisesRegex(DataFoundationAuditError, "payload SHA-256"):
                self._verify(root)

    def test_rerun_is_independently_required_hashed_and_byte_compared(self) -> None:
        cases = ("missing", "tampered", "primary-alias", "hardlink-alias")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline, audit = _make_fixture(root)
                rerun_path = root / RERUN_RELATIVE
                if case == "missing":
                    rerun_path.unlink()
                elif case == "tampered":
                    rerun_path.write_bytes(rerun_path.read_bytes() + b" ")
                elif case == "primary-alias":
                    production = baseline["production_audit"]
                    production["rerun_relative_path"] = AUDIT_RELATIVE.as_posix()
                    _write_json(root / BASELINE_RELATIVE, baseline)
                else:
                    rerun_path.unlink()
                    try:
                        os.link(root / AUDIT_RELATIVE, rerun_path)
                    except OSError as exc:  # pragma: no cover - host policy dependent
                        self.skipTest(f"hardlink creation is unavailable: {exc}")
                with self.assertRaises(DataFoundationAuditError):
                    self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            rerun_path = root / RERUN_RELATIVE
            rerun_path.write_bytes(b"different-but-repinned")
            rerun_bytes = rerun_path.read_bytes()
            production = baseline["production_audit"]
            production["rerun_bytes"] = len(rerun_bytes)
            production["rerun_file_sha256"] = _sha256_bytes(rerun_bytes)
            _write_json(root / BASELINE_RELATIVE, baseline)
            with self.assertRaisesRegex(DataFoundationAuditError, "bytes differ"):
                self._verify(root)

    def test_descriptor_and_manifest_actual_bytes_are_verified(self) -> None:
        for target, expected in (
            ("configs/source_descriptors/fixture.json", "SHA-256"),
            ("workspace/fixtures/manifest.json", "SHA-256|byte size"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _make_fixture(root)
                (root / target).write_bytes(b"tampered")
                with self.assertRaisesRegex(DataFoundationAuditError, expected):
                    self._verify(root)

    def test_capability_matrix_must_be_derived_from_actual_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            decisions = audit["sources"]["fixture_source"]["capability_decision_matrix"]
            decisions[0]["reason"] = "plausible-but-not-derived"
            _refresh_audit_payload(audit)
            _refresh_baseline_audit_pin(root, baseline, audit)
            with self.assertRaisesRegex(
                DataFoundationAuditError, "not derived from the tracked descriptor"
            ):
                self._verify(root)

    def test_dataset_cell_margins_and_unavailable_cells_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            dataset = audit["sources"]["fixture_source"]["dataset"]
            cells = dataset["by_position_target"]
            donor = next(
                cell
                for cell in cells
                if cell["status_counts"]["observed"] > 0
            )
            receiver = next(
                cell
                for cell in cells
                if cell["target"] == donor["target"]
                and cell["position"] != donor["position"]
            )
            donor["row_count"] -= 1
            donor["status_counts"]["observed"] -= 1
            receiver["row_count"] += 1
            receiver["status_counts"]["observed"] += 1
            _refresh_audit_payload(audit)
            _refresh_baseline_audit_pin(root, baseline, audit)
            with self.assertRaisesRegex(DataFoundationAuditError, "cell margins"):
                self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            source = audit["sources"]["fixture_source"]
            unavailable = next(
                decision
                for decision in source["capability_decision_matrix"]
                if not decision["available"]
            )
            dataset = source["dataset"]
            cell = next(
                item
                for item in dataset["by_position_target"]
                if item["position"] == unavailable["position"]
                and item["target"] == unavailable["target"]
            )
            cell["row_count"] += 1
            cell["status_counts"]["observed"] += 1
            dataset["row_count"] += 1
            dataset["status_counts"]["observed"] += 1
            position = next(
                item
                for item in dataset["by_position"]
                if item["position"] == unavailable["position"]
            )
            position["row_count"] += 1
            position["status_counts"]["observed"] += 1
            target = next(
                item
                for item in dataset["by_target"]
                if item["target"] == unavailable["target"]
            )
            target["row_count"] += 1
            target["status_counts"]["observed"] += 1
            _refresh_audit_payload(audit)
            _refresh_baseline_audit_pin(root, baseline, audit)
            with self.assertRaisesRegex(DataFoundationAuditError, "unavailable cell"):
                self._verify(root)

    def test_locked_source_dataset_status_and_identity_values_are_exact(self) -> None:
        cases = (
            ("dataset_id", "0" * 64),
            ("row_count", 999),
            (
                "dataset_status_counts",
                {"censored": 0, "invalid": 0, "missing": 0, "observed": 999},
            ),
            ("task_count", 999),
            ("condition_count", 999),
            ("source_id", "other-source"),
            ("revision", "other-revision"),
            ("manifest_sha256", "3" * 64),
            ("capability_contract_hash", "4" * 64),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline, audit = _make_fixture(root)
                baseline["sources"]["fixture_source"][field] = value
                _refresh_baseline_audit_pin(root, baseline, audit)
                with self.assertRaises(DataFoundationAuditError):
                    self._verify(root)

    def test_implementation_commit_tree_and_policy_must_close(self) -> None:
        for field, value in (
            ("git_commit", "5" * 40),
            ("source_blob_count", 41),
            ("source_tree_sha256", "6" * 64),
            ("git_source_binding_policy", "unsupported"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline, audit = _make_fixture(root)
                baseline["implementation"][field] = value
                _refresh_baseline_audit_pin(root, baseline, audit)
                with self.assertRaises(DataFoundationAuditError):
                    self._verify(root)

    def test_unsafe_and_mismatched_paths_are_rejected(self) -> None:
        unsafe = (
            "../audit.json",
            "C:/audit.json",
            "/tmp/audit.json",
            "a\\b.json",
            " workspace/data_foundation/audit.json",
            "workspace/data_foundation/audit.json ",
            "workspace/data_foundation/\x00audit.json",
        )
        for value in unsafe:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline, audit = _make_fixture(root)
                baseline["production_audit"]["relative_path"] = value
                _refresh_baseline_audit_pin(root, baseline, audit)
                with self.assertRaises(DataFoundationAuditError):
                    self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            with self.assertRaisesRegex(DataFoundationAuditError, "canonical relative"):
                verify_data_foundation_baseline(
                    root,
                    baseline_path=(root / BASELINE_RELATIVE).resolve(),
                    audit_path=AUDIT_RELATIVE,
                )

    def test_privacy_blacklist_and_malicious_reason_or_path_fail_closed(self) -> None:
        for key in (
            "point_id",
            "event_id",
            "logical_call_id",
            "attempt_id",
            "raw_ref",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                DataFoundationAuditError, "forbidden row-level identity key"
            ):
                _assert_privacy_safe({key: "secret"}, label="fixture")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            audit["sources"]["fixture_source"]["capability_decision_matrix"][0][
                "reason"
            ] = "C:/private/raw-event.json"
            _refresh_audit_payload(audit)
            _refresh_baseline_audit_pin(root, baseline, audit)
            with self.assertRaisesRegex(DataFoundationAuditError, "absolute local path"):
                self._verify(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, audit = _make_fixture(root)
            audit["sources"]["fixture_source"]["artifacts"]["manifest"][
                "path"
            ] = "../private/manifest.json"
            _refresh_audit_payload(audit)
            _refresh_baseline_audit_pin(root, baseline, audit)
            with self.assertRaisesRegex(DataFoundationAuditError, "canonical relative"):
                self._verify(root)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junction_component_is_rejected_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            alias = root / "config-alias"
            result = subprocess.run(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(alias),
                    str(root / "configs"),
                ],
                capture_output=True,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                self.skipTest("junction creation is unavailable on this host")
            with self.assertRaisesRegex(
                DataFoundationAuditError, "junctions|reparse points"
            ):
                verify_data_foundation_baseline(
                    root,
                    baseline_path=Path("config-alias/data_foundation_v2_baseline.json"),
                    audit_path=AUDIT_RELATIVE,
                )

    @unittest.skipUnless(os.name == "nt", "Windows source junction test")
    def test_workspace_source_tree_rejects_junction_inside_src(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/token_prediction"
            package.mkdir(parents=True)
            (package / "__init__.py").write_bytes(b"\n")
            audit_script = root / "scripts/audit_data_foundation_v2.py"
            audit_script.parent.mkdir(parents=True)
            audit_script.write_bytes(b"\n")
            junction_target = root / "junction-target"
            junction_target.mkdir()
            (junction_target / "hidden.py").write_bytes(b"VALUE = 1\n")
            junction = package / "linked"
            result = subprocess.run(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(junction),
                    str(junction_target),
                ],
                capture_output=True,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                self.skipTest("junction creation is unavailable on this host")
            with self.assertRaisesRegex(
                DataFoundationAuditError, "junction|reparse point"
            ):
                strict_workspace_source_tree_sha256(
                    root,
                )

    def test_symlink_for_baseline_or_descriptor_is_rejected(self) -> None:
        for target in ("baseline", "descriptor"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _make_fixture(root)
                if target == "baseline":
                    path = root / BASELINE_RELATIVE
                else:
                    path = root / "configs/source_descriptors/fixture.json"
                real = path.with_name(f"{path.name}.real")
                path.replace(real)
                try:
                    path.symlink_to(real.name)
                except OSError as exc:  # pragma: no cover - host policy dependent
                    self.skipTest(f"symlink creation is unavailable: {exc}")
                with self.assertRaisesRegex(DataFoundationAuditError, "symlink"):
                    self._verify(root)


if __name__ == "__main__":
    unittest.main()
