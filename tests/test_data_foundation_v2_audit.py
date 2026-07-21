from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_data_foundation_v2 import (
    BAGEN_COMBINED_AUDIT_SOURCE_ID,
    SPEND_INVENTORY_SOURCE_ID,
    ArtifactEvidence,
    DataFoundationAuditError,
    _assert_aggregate_safe,
    _canonical_relative_path,
    _require_source_id,
    _strict_json_loads,
    atomic_write_json,
    build_data_foundation_audit,
    build_source_audit,
    load_source_descriptor,
    resolve_git_commit,
    source_tree_sha256,
    verify_audit_payload,
    verify_file,
    verify_git_source_binding,
)
from tests.helpers import make_two_call_trajectory
from token_prediction.collection import BagenSwebenchReader, OpenHandsArchiveReader
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor


FIXTURE_GIT_COMMIT = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor() -> SourceDescriptor:
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
    return SourceDescriptor(
        source_id="fixture-source",
        revision="fixture-revision",
        manifest_path="workspace/fixtures/manifest.json",
        manifest_sha256="a" * 64,
        capabilities=capabilities,
    )


def _artifacts() -> dict[str, ArtifactEvidence]:
    return {
        "descriptor": ArtifactEvidence(
            path="configs/source_descriptors/fixture.json",
            bytes=123,
            sha256="b" * 64,
        ),
        "raw_trajectories": ArtifactEvidence(
            path="workspace/fixtures/raw",
            bytes=456,
            sha256="c" * 64,
            file_count=1,
            sha256_kind="framed_file_index_v1",
        ),
    }


def _source_summary() -> dict[str, object]:
    return build_source_audit(
        source_name="fixture_source",
        trajectories=(make_two_call_trajectory(0),),
        descriptor=_descriptor(),
        artifacts=_artifacts(),
    )


class DataFoundationV2AuditTests(unittest.TestCase):
    def test_artifact_source_ids_are_distinct_from_reader_source_ids(self) -> None:
        self.assertNotEqual(
            BAGEN_COMBINED_AUDIT_SOURCE_ID,
            BagenSwebenchReader.source_id,
        )
        self.assertNotEqual(
            SPEND_INVENTORY_SOURCE_ID,
            OpenHandsArchiveReader.source_id,
        )
        accepted = (
            (
                {"source_id": BAGEN_COMBINED_AUDIT_SOURCE_ID},
                BAGEN_COMBINED_AUDIT_SOURCE_ID,
                "BAGEN combined audit",
            ),
            (
                {"source_id": BagenSwebenchReader.source_id},
                BagenSwebenchReader.source_id,
                "BAGEN family audit",
            ),
            (
                {"source_id": SPEND_INVENTORY_SOURCE_ID},
                SPEND_INVENTORY_SOURCE_ID,
                "Spend inventory",
            ),
            (
                {"source_id": OpenHandsArchiveReader.source_id},
                OpenHandsArchiveReader.source_id,
                "Spend descriptor",
            ),
        )
        for value, expected, label in accepted:
            with self.subTest(label=label):
                _require_source_id(value, expected=expected, label=label)

        crossed = (
            (
                {"source_id": BagenSwebenchReader.source_id},
                BAGEN_COMBINED_AUDIT_SOURCE_ID,
                "BAGEN combined audit",
            ),
            (
                {"source_id": OpenHandsArchiveReader.source_id},
                SPEND_INVENTORY_SOURCE_ID,
                "Spend inventory",
            ),
        )
        for value, expected, label in crossed:
            with self.subTest(label=f"crossed {label}"), self.assertRaisesRegex(
                DataFoundationAuditError, "source_id mismatch"
            ):
                _require_source_id(value, expected=expected, label=label)

    def test_pure_single_source_audit_is_deterministic_and_aggregate_only(self) -> None:
        first_summary = _source_summary()
        second_summary = _source_summary()
        self.assertEqual(first_summary, second_summary)

        first = build_data_foundation_audit(
            {"fixture_source": first_summary},
            git_commit=FIXTURE_GIT_COMMIT,
            source_tree_sha256="d" * 64,
            runtime={"python_implementation": "CPython", "python_version": "3.11.0"},
        )
        second = build_data_foundation_audit(
            {"fixture_source": second_summary},
            git_commit=FIXTURE_GIT_COMMIT,
            source_tree_sha256="d" * 64,
            runtime={"python_version": "3.11.0", "python_implementation": "CPython"},
        )
        self.assertEqual(first, second)
        verify_audit_payload(first)
        self.assertEqual(first["source_count"], 1)
        self.assertEqual(
            first["implementation"]["git_commit"],  # type: ignore[index]
            FIXTURE_GIT_COMMIT,
        )
        self.assertEqual(
            first["implementation"][  # type: ignore[index]
                "git_source_binding_policy"
            ],
            "tracked_clean_head_blob_tree_v1",
        )
        self.assertEqual(
            first_summary["identity_counts"],
            {
                "task_count": 1,
                "trajectory_count": 1,
                "run_count": 1,
                "condition_count": 1,
            },
        )
        rendered = json.dumps(first, sort_keys=True, allow_nan=False)
        self.assertNotIn("task-0", rendered)
        self.assertNotIn("task0-run0", rendered)
        self.assertNotIn(str(Path.cwd().resolve()), rendered)

    def test_capability_matrix_gates_proxy_targets_and_counts_new_targets(self) -> None:
        summary = _source_summary()
        decisions = {
            (item["position"], item["target"]): item
            for item in summary["capability_decision_matrix"]  # type: ignore[index]
        }
        self.assertEqual(len(decisions), 5 * 8)
        self.assertFalse(
            decisions[
                ("task_pre", "task_unknown_remaining_tokens")
            ]["available"]
        )
        self.assertEqual(
            decisions[
                ("task_pre", "task_unknown_remaining_tokens")
            ]["missing_observables"],
            ["request_local_count"],
        )
        self.assertFalse(
            decisions[
                ("call_pre", "call_unknown_billable_tokens")
            ]["available"]
        )
        self.assertTrue(
            decisions[
                ("task_pre", "task_provider_accounted_remaining_tokens")
            ]["available"]
        )
        self.assertTrue(
            decisions[("call_pre", "call_billable_total_tokens")]["available"]
        )

        cells = {
            (item["position"], item["target"]): item
            for item in summary["dataset"]["by_position_target"]  # type: ignore[index]
        }
        self.assertEqual(len(cells), 5 * 8)
        self.assertEqual(
            cells[("task_pre", "task_unknown_remaining_tokens")]["row_count"],
            0,
        )
        self.assertEqual(
            cells[("call_pre", "call_unknown_billable_tokens")]["row_count"],
            0,
        )
        self.assertEqual(
            cells[
                ("task_pre", "task_provider_accounted_remaining_tokens")
            ]["row_count"],
            1,
        )
        self.assertEqual(
            cells[
                ("task_update", "task_provider_accounted_remaining_tokens")
            ]["row_count"],
            1,
        )
        self.assertEqual(
            cells[("call_pre", "call_billable_total_tokens")]["row_count"],
            2,
        )

    def test_descriptor_sha_and_canonical_schema_fail_closed_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor_path = root / "configs" / "fixture.json"
            descriptor_path.parent.mkdir(parents=True)
            descriptor_path.write_text(
                json.dumps(_descriptor().to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            expected_sha = _sha256(descriptor_path)
            loaded, evidence = load_source_descriptor(
                root,
                "configs/fixture.json",
                expected_sha256=expected_sha,
            )
            self.assertEqual(loaded, _descriptor())
            self.assertEqual(evidence.sha256, expected_sha)

            payload = _descriptor().to_dict()
            payload["revision"] = "tampered"
            descriptor_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataFoundationAuditError, "SHA-256"):
                load_source_descriptor(
                    root,
                    "configs/fixture.json",
                    expected_sha256=expected_sha,
                )

    def test_raw_manifest_and_archive_file_pins_reject_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("raw.traj.json", "manifest.jsonl", "archive.tar.gz"):
                with self.subTest(name=name):
                    path = root / "workspace" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"frozen-{name}".encode())
                    expected_sha = _sha256(path)
                    verify_file(
                        root,
                        f"workspace/{name}",
                        expected_sha256=expected_sha,
                        expected_bytes=path.stat().st_size,
                        label=name,
                    )
                    path.write_bytes(b"tampered")
                    with self.assertRaisesRegex(
                        DataFoundationAuditError, "SHA-256|byte size"
                    ):
                        verify_file(
                            root,
                            f"workspace/{name}",
                            expected_sha256=expected_sha,
                            label=name,
                        )

    def test_strict_json_and_path_validation_reject_unsafe_inputs(self) -> None:
        for document in (
            '{"value": 1, "value": 2}',
            '{"value": NaN}',
            '{"value": 1e999}',
        ):
            with self.subTest(document=document), self.assertRaises(
                DataFoundationAuditError
            ):
                _strict_json_loads(document, label="fixture")
        for path in ("../escape.json", "C:/escape.json", "/tmp/escape.json", "a\\b"):
            with self.subTest(path=path), self.assertRaises(
                DataFoundationAuditError
            ):
                _canonical_relative_path(path, label="fixture path")
        for value in ("C:\\private\\audit.json", "/home/private/audit.json"):
            with self.subTest(value=value), self.assertRaises(
                DataFoundationAuditError
            ):
                _assert_aggregate_safe({"artifact": value})

    def test_payload_tamper_and_output_overwrite_fail_closed(self) -> None:
        audit = build_data_foundation_audit(
            {"fixture_source": _source_summary()},
            git_commit=FIXTURE_GIT_COMMIT,
            source_tree_sha256="e" * 64,
            runtime={"python_version": "3.11.0"},
        )
        tampered = dict(audit)
        tampered["source_count"] = 2
        with self.assertRaisesRegex(DataFoundationAuditError, "does not match"):
            verify_audit_payload(tampered)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            atomic_write_json(output, audit)
            frozen_bytes = output.read_bytes()
            with self.assertRaisesRegex(DataFoundationAuditError, "already exists"):
                atomic_write_json(output, {"different": True})
            self.assertEqual(output.read_bytes(), frozen_bytes)
            atomic_write_json(output, {"different": True}, force=True)
            self.assertEqual(json.loads(output.read_text()), {"different": True})

            invalid_output = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                atomic_write_json(invalid_output, {"value": math.nan})
            self.assertFalse(invalid_output.exists())

    def test_git_commit_resolver_verifies_worktree_root_and_commit_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            calls: list[tuple[str, ...]] = []

            def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(tuple(command))
                if command[-1] == "--show-toplevel":
                    stdout = f"{root}\n"
                elif command[-1] == "HEAD^{commit}":
                    stdout = f"{FIXTURE_GIT_COMMIT}\n"
                else:
                    return subprocess.CompletedProcess(command, 1, "", "")
                return subprocess.CompletedProcess(command, 0, stdout, "")

            commit = resolve_git_commit(
                root,
                git_executable="fixture-git",
                runner=runner,
            )
            self.assertEqual(commit, FIXTURE_GIT_COMMIT)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][-2:], ("rev-parse", "--show-toplevel"))
            self.assertEqual(
                calls[1][-3:],
                ("rev-parse", "--verify", "HEAD^{commit}"),
            )

    def test_git_commit_binding_fails_closed_on_unresolvable_or_invalid_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            def failed_runner(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 1, "", "failure")

            with self.assertRaisesRegex(DataFoundationAuditError, "could not resolve"):
                resolve_git_commit(
                    root,
                    git_executable="fixture-git",
                    runner=failed_runner,
                )

            def invalid_head_runner(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                stdout = f"{root}\n" if command[-1] == "--show-toplevel" else "not-a-commit\n"
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with self.assertRaisesRegex(DataFoundationAuditError, "Git object id"):
                resolve_git_commit(
                    root,
                    git_executable="fixture-git",
                    runner=invalid_head_runner,
                )

            with self.assertRaisesRegex(DataFoundationAuditError, "Git object id"):
                build_data_foundation_audit(
                    {"fixture_source": _source_summary()},
                    git_commit="not-a-commit",
                    source_tree_sha256="e" * 64,
                    runtime={"python_version": "3.11.0"},
                )

    def test_git_source_binding_matches_clean_tracked_head_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            blobs = {
                "scripts/audit_data_foundation_v2.py": b"print('audit')\n",
                "src/token_prediction/__init__.py": b"\n",
                "src/token_prediction/module.py": b"VALUE = 1\n",
            }
            for relative_path, content in blobs.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            workspace_hash = source_tree_sha256(root)
            calls: list[tuple[str, ...]] = []

            def runner(command: list[str], **kwargs: object):
                calls.append(tuple(command))
                arguments = command[3:]
                if arguments == ["rev-parse", "--show-toplevel"]:
                    stdout: str | bytes = f"{root}\n"
                elif arguments == ["rev-parse", "--verify", "HEAD^{commit}"]:
                    stdout = f"{FIXTURE_GIT_COMMIT}\n"
                elif arguments[:1] == ["diff"]:
                    stdout = ""
                elif arguments[:3] == ["ls-files", "--others", "--exclude-standard"]:
                    stdout = ""
                elif arguments[:4] == ["ls-tree", "-r", "-z", "--name-only"]:
                    stdout = "".join(f"{path}\0" for path in sorted(blobs))
                elif arguments[:2] == ["cat-file", "blob"]:
                    relative_path = arguments[2].split(":", 1)[1]
                    stdout = blobs[relative_path]
                else:
                    return subprocess.CompletedProcess(command, 1, "", "")
                stderr: str | bytes = "" if kwargs.get("text") else b""
                return subprocess.CompletedProcess(command, 0, stdout, stderr)

            verify_git_source_binding(
                root,
                git_commit=FIXTURE_GIT_COMMIT,
                workspace_source_tree_sha256=workspace_hash,
                git_executable="fixture-git",
                runner=runner,
            )
            cat_file_calls = [call for call in calls if "cat-file" in call]
            self.assertEqual(len(cat_file_calls), len(blobs))

    def test_git_source_binding_rejects_all_relevant_dirty_states_and_blob_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            blobs = {
                "scripts/audit_data_foundation_v2.py": b"print('audit')\n",
                "src/token_prediction/__init__.py": b"\n",
            }
            for relative_path, content in blobs.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            workspace_hash = source_tree_sha256(root)

            def make_runner(
                *,
                staged: str = "",
                unstaged: str = "",
                untracked: str = "",
                head_blobs: dict[str, bytes] | None = None,
            ):
                resolved_blobs = head_blobs or blobs

                def runner(command: list[str], **kwargs: object):
                    arguments = command[3:]
                    if arguments == ["rev-parse", "--show-toplevel"]:
                        stdout: str | bytes = f"{root}\n"
                    elif arguments == ["rev-parse", "--verify", "HEAD^{commit}"]:
                        stdout = f"{FIXTURE_GIT_COMMIT}\n"
                    elif arguments[:2] == ["diff", "--cached"]:
                        stdout = staged
                    elif arguments[:1] == ["diff"]:
                        stdout = unstaged
                    elif arguments[:3] == [
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                    ]:
                        stdout = untracked
                    elif arguments[:4] == ["ls-tree", "-r", "-z", "--name-only"]:
                        stdout = "".join(
                            f"{path}\0" for path in sorted(resolved_blobs)
                        )
                    elif arguments[:2] == ["cat-file", "blob"]:
                        relative_path = arguments[2].split(":", 1)[1]
                        stdout = resolved_blobs[relative_path]
                    else:
                        return subprocess.CompletedProcess(command, 1, "", "")
                    stderr: str | bytes = "" if kwargs.get("text") else b""
                    return subprocess.CompletedProcess(command, 0, stdout, stderr)

                return runner

            cases = (
                (
                    "staged",
                    make_runner(staged="src/token_prediction/__init__.py\0"),
                    "staged changes",
                ),
                (
                    "unstaged",
                    make_runner(unstaged="src/token_prediction/__init__.py\0"),
                    "unstaged changes",
                ),
                (
                    "untracked",
                    make_runner(untracked="src/token_prediction/new.py\0"),
                    "untracked files",
                ),
                (
                    "head blob drift",
                    make_runner(
                        head_blobs={
                            **blobs,
                            "src/token_prediction/__init__.py": b"changed\n",
                        }
                    ),
                    "HEAD blob source tree hash",
                ),
                (
                    "workspace path not tracked by head",
                    make_runner(
                        head_blobs={
                            "scripts/audit_data_foundation_v2.py": blobs[
                                "scripts/audit_data_foundation_v2.py"
                            ]
                        }
                    ),
                    "not all tracked by HEAD",
                ),
            )
            for name, runner, error in cases:
                with self.subTest(name=name), self.assertRaisesRegex(
                    DataFoundationAuditError, error
                ):
                    verify_git_source_binding(
                        root,
                        git_commit=FIXTURE_GIT_COMMIT,
                        workspace_source_tree_sha256=workspace_hash,
                        git_executable="fixture-git",
                        runner=runner,
                    )


if __name__ == "__main__":
    unittest.main()
