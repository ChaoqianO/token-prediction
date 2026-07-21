from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts import run_data_foundation_baseline as baseline
from tests.helpers import make_two_call_trajectory
from token_prediction.collection import BagenSwebenchReader, OpenHandsArchiveReader
from token_prediction.contracts import SourceDescriptor
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
    build_capability_supervised_dataset,
)
from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory


def _descriptor(source_id: str, manifest_path: str, capabilities: object) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=source_id,
        revision="a" * 40,
        manifest_path=manifest_path,
        manifest_sha256="b" * 64,
        capabilities=capabilities,  # type: ignore[arg-type]
    )


def _trajectory(task_index: int, run_index: int = 0, condition_id: str | None = None) -> Trajectory:
    trajectory = make_two_call_trajectory(task_index, run_index)
    totals = [
        event.payload["usage"]
        for event in trajectory.events
        if event.event_type == EventType.API_COMPLETED
    ]
    task_usage = {
        "input_tokens": sum(item["input_tokens"] for item in totals),
        "output_tokens": sum(item["output_tokens"] for item in totals),
    }
    task_usage["total_tokens"] = task_usage["input_tokens"] + task_usage["output_tokens"]
    started = trajectory.events[0]
    if condition_id is not None:
        started = started.with_payload({**started.payload, "condition_id": condition_id})
    terminal = trajectory.events[-1].with_payload(
        {**trajectory.events[-1].payload, "usage": task_usage}
    )
    return Trajectory.from_events((started, *trajectory.events[1:-1], terminal))


def _synthetic_dataset(task_count: int = 40) -> SupervisedDataset:
    descriptor = _descriptor(
        BagenSwebenchReader.source_id,
        "workspace/synthetic/manifest.jsonl",
        BagenSwebenchReader.capabilities,
    )
    return build_capability_supervised_dataset(
        (_trajectory(index) for index in range(task_count)), descriptor
    )


def _frozen_synthetic_datasets() -> tuple[SupervisedDataset, SupervisedDataset]:
    bagen_descriptor = _descriptor(
        BagenSwebenchReader.source_id,
        "workspace/synthetic/bagen-manifest.jsonl",
        BagenSwebenchReader.capabilities,
    )
    bagen_trajectories = []
    for run_index, condition_id in enumerate(
        sorted(baseline.FROZEN_BAGEN_ESTIMABLE_CONDITIONS)
    ):
        bagen_trajectories.extend(
            _trajectory(task_index, run_index, condition_id)
            for task_index in range(40)
        )
    for run_index, condition_id in enumerate(
        sorted(baseline.FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS), start=5
    ):
        count = 3 if condition_id.endswith("a166") else 5
        bagen_trajectories.extend(
            _trajectory(task_index, run_index, condition_id)
            for task_index in range(count)
        )
    bagen = build_capability_supervised_dataset(bagen_trajectories, bagen_descriptor)

    spend_descriptor = _descriptor(
        OpenHandsArchiveReader.source_id,
        "workspace/synthetic/spend-inventory.json",
        OpenHandsArchiveReader.capabilities,
    )
    spend_condition = next(iter(baseline.FROZEN_SPEND_CONDITIONS))
    spend = build_capability_supervised_dataset(
        (_trajectory(index, 20, spend_condition) for index in range(40)),
        spend_descriptor,
    )
    return bagen, spend


def _source_lock(
    name: str,
    dataset: SupervisedDataset,
    descriptor: SourceDescriptor,
) -> baseline.SourceLock:
    return baseline.SourceLock(
        name=name,
        descriptor_path=f"configs/{name}.json",
        descriptor_file_sha256="c" * 64,
        descriptor=descriptor,
        manifest_path=descriptor.manifest_path,
        manifest_sha256=descriptor.manifest_sha256,
        dataset_id=dataset.dataset_id,
        dataset_row_count=len(dataset.rows),
        raw_artifact_path=f"workspace/{name}/raw.bin",
        raw_artifact_sha256="d" * 64,
        raw_artifact_sha256_kind=(
            "framed_file_index_v1" if name == "bagen_swebench" else "file_bytes"
        ),
        raw_artifact_bytes=123,
    )


def _lock_context(
    bagen_dataset: SupervisedDataset,
    spend_dataset: SupervisedDataset | None = None,
) -> baseline.LockContext:
    spend_dataset = spend_dataset or bagen_dataset
    bagen_descriptor = _descriptor(
        BagenSwebenchReader.source_id,
        "workspace/synthetic/bagen-manifest.jsonl",
        BagenSwebenchReader.capabilities,
    )
    spend_descriptor = _descriptor(
        OpenHandsArchiveReader.source_id,
        "workspace/synthetic/spend-inventory.json",
        OpenHandsArchiveReader.capabilities,
    )
    return baseline.LockContext(
        baseline_lock_path="configs/data_foundation_v2_baseline.json",
        baseline_lock_file_sha256="e" * 64,
        audit_path="workspace/data_foundation/data_foundation_v2_audit.json",
        audit_file_sha256="f" * 64,
        audit_payload_sha256="1" * 64,
        audit_git_commit="2" * 40,
        audit_source_tree_sha256="3" * 64,
        sources={
            "bagen_swebench": _source_lock(
                "bagen_swebench", bagen_dataset, bagen_descriptor
            ),
            "spend_openhands": _source_lock(
                "spend_openhands", spend_dataset, spend_descriptor
            ),
        },
    )


def _build_results_for_datasets(
    bagen_dataset: SupervisedDataset,
    spend_dataset: SupervisedDataset,
) -> tuple[dict[str, object], dict[str, bytes]]:
    lock = _lock_context(bagen_dataset, spend_dataset)
    code = baseline.CodeBinding(
        git_commit="4" * 40,
        source_tree_sha256="5" * 64,
        paths=(
            "scripts/run_data_foundation_baseline.py",
            "src/token_prediction/__init__.py",
        ),
    )
    return baseline.build_results(
        bagen_dataset=bagen_dataset,
        spend_dataset=spend_dataset,
        lock_context=lock,
        code_binding=code,
        audit_compatible_source_tree_hash="3" * 64,
        tracked_control_tree_hash="6" * 64,
    )


def _build_synthetic_results() -> tuple[dict[str, object], dict[str, bytes]]:
    return _build_results_for_datasets(*_frozen_synthetic_datasets())


def _rehash_results(results: dict[str, object]) -> None:
    results.pop("results_payload_sha256", None)
    results["results_payload_sha256"] = baseline._semantic_sha256(results)


class DataFoundationBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _synthetic_dataset()
        cls.holdout = baseline.make_holdout_plan(cls.dataset)
        cls.development = baseline.development_dataset(cls.dataset, cls.holdout)

    def test_v3_identity_and_output_path_are_paired(self) -> None:
        self.assertEqual(
            baseline.BASELINE_ID,
            "data_foundation_empirical_development_v3",
        )
        self.assertEqual(
            baseline.DEFAULT_OUTPUT,
            "workspace/data_foundation/baselines/empirical-development-v3",
        )

    def test_stable_holdout_depends_only_on_task_identity(self) -> None:
        holdout_task = next(iter(self.holdout.final_holdout_tasks))
        changed_rows = []
        for row in self.dataset.rows:
            if row.point.task_id == holdout_task:
                changed_rows.append(
                    replace(
                        row,
                        point=row.point.with_features(
                            {**row.point.features, "task_char_count": 987_654_321}
                        ),
                        label=(row.label + 123_456 if row.label is not None else None),
                    )
                )
            else:
                changed_rows.append(row)
        perturbed = replace(
            self.dataset,
            dataset_id="f" * 64,
            rows=tuple(changed_rows),
        )
        second = baseline.make_holdout_plan(perturbed)

        self.assertEqual(second.development_tasks, self.holdout.development_tasks)
        self.assertEqual(second.final_holdout_tasks, self.holdout.final_holdout_tasks)
        self.assertEqual(second.assignment_digest, self.holdout.assignment_digest)
        original_development = baseline.development_dataset(self.dataset, self.holdout)
        perturbed_development = baseline.development_dataset(perturbed, second)
        self.assertEqual(original_development, perturbed_development)

        condition = next(
            iter({row.point.condition_id for row in original_development.rows})
        )
        original_result, original_bundles = baseline.run_development_cell(
            original_development,
            source_name="bagen_swebench",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition,
            split_seed=baseline.SPLIT_SEEDS[0],
        )
        perturbed_result, perturbed_bundles = baseline.run_development_cell(
            perturbed_development,
            source_name="bagen_swebench",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition,
            split_seed=baseline.SPLIT_SEEDS[0],
        )
        self.assertEqual(original_result, perturbed_result)
        self.assertEqual(original_bundles, perturbed_bundles)

    def test_development_identity_changes_with_development_labels(self) -> None:
        development_task = next(iter(self.holdout.development_tasks))
        changed_rows = tuple(
            replace(row, label=int(row.label or 0) + 1)
            if row.point.task_id == development_task
            else row
            for row in self.dataset.rows
        )
        perturbed = replace(
            self.dataset,
            dataset_id="e" * 64,
            rows=changed_rows,
        )
        second = baseline.make_holdout_plan(perturbed)
        self.assertEqual(second.assignment_digest, self.holdout.assignment_digest)
        self.assertNotEqual(
            second.development_dataset_id,
            self.holdout.development_dataset_id,
        )

    def test_build_results_ignores_final_holdout_labels_and_statuses(self) -> None:
        bagen_dataset, spend_dataset = _frozen_synthetic_datasets()
        original_results, original_bundles = _build_results_for_datasets(
            bagen_dataset, spend_dataset
        )
        plan = baseline.make_holdout_plan(bagen_dataset)
        perturbed = replace(
            bagen_dataset,
            dataset_id="9" * 64,
            rows=tuple(
                replace(
                    row,
                    label=None,
                    status=LabelStatus.MISSING,
                    invalid_reason="final_holdout_redacted_for_regression_test",
                )
                if row.point.task_id in plan.final_holdout_tasks
                else row
                for row in bagen_dataset.rows
            ),
        )
        perturbed_plan = baseline.make_holdout_plan(perturbed)
        perturbed_results, perturbed_bundles = _build_results_for_datasets(
            perturbed, spend_dataset
        )

        self.assertEqual(
            perturbed_plan.development_dataset_id,
            plan.development_dataset_id,
        )
        self.assertEqual(perturbed_results["cells"], original_results["cells"])
        self.assertEqual(
            perturbed_results["not_estimable_conditions"],
            original_results["not_estimable_conditions"],
        )
        self.assertEqual(
            perturbed_results["condition_gate_policy"],
            original_results["condition_gate_policy"],
        )
        self.assertEqual(perturbed_bundles, original_bundles)

    def test_test_fold_label_does_not_change_its_prediction(self) -> None:
        condition = next(iter({row.point.condition_id for row in self.development.rows}))
        first, _ = baseline.run_development_cell(
            self.development,
            source_name="bagen_swebench",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition,
            split_seed=baseline.SPLIT_SEEDS[0],
        )
        split_plan = baseline.make_baseline_split_plan(
            self.development.task_ids,
            dataset_id=self.development.dataset_id,
            seed=baseline.SPLIT_SEEDS[0],
        )
        condition_tasks = {
            row.point.task_id
            for row in self.development.rows
            if row.eligible
            and row.point.position == PredictionPosition.TASK_UPDATE
            and row.point.target
            == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            and row.point.condition_id == condition
        }
        chosen_task = sorted(split_plan.partition(0).test_tasks & condition_tasks)[0]
        chosen_hash = baseline._holdout_task_digest(chosen_task)
        changed = replace(
            self.development,
            rows=tuple(
                replace(row, label=row.label + 999_999)
                if row.point.task_id == chosen_task and row.label is not None
                else row
                for row in self.development.rows
            ),
        )
        second, _ = baseline.run_development_cell(
            changed,
            source_name="bagen_swebench",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition,
            split_seed=baseline.SPLIT_SEEDS[0],
        )
        first_prediction = next(
            item for item in first["predictions"] if item["task_id_sha256"] == chosen_hash
        )
        second_prediction = next(
            item for item in second["predictions"] if item["task_id_sha256"] == chosen_hash
        )
        self.assertEqual(first_prediction["fold"], second_prediction["fold"])
        self.assertEqual(first_prediction["forecast"], second_prediction["forecast"])

    def test_final_holdout_is_absent_from_fit_score_and_prediction_payloads(self) -> None:
        results, _ = _build_synthetic_results()
        rendered = baseline._canonical_bytes(results)
        holdout_hashes = {
            baseline._holdout_task_digest(task)
            for task in self.holdout.final_holdout_tasks
        }
        predicted_hashes: set[str] = set()
        assigned_hashes: set[str] = set()
        for cell in results["cells"]:  # type: ignore[index]
            for seed in cell["seed_results"]:
                predicted_hashes.update(
                    record["task_id_sha256"] for record in seed["predictions"]
                )
                assigned_hashes.update(
                    record["task_id_sha256"] for record in seed["split_assignments"]
                )
        self.assertFalse(holdout_hashes & predicted_hashes)
        self.assertFalse(holdout_hashes & assigned_hashes)
        self.assertFalse(results["final_holdout_evaluated"])
        self.assertEqual(results["final_holdout_prediction_count"], 0)
        self.assertFalse(
            results["final_holdout_target_values_used_for_fit_calibration_scoring"]
        )
        for task_id in self.holdout.final_holdout_tasks:
            self.assertNotIn(task_id.encode("utf-8"), rendered)

    def test_empirical_bundle_is_strict_and_reloads_exact_predictions(self) -> None:
        condition = next(iter({row.point.condition_id for row in self.development.rows}))
        result, bundles = baseline.run_development_cell(
            self.development,
            source_name="bagen_swebench",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition,
            split_seed=baseline.SPLIT_SEEDS[0],
        )
        record = result["predictions"][0]
        document, loaded = baseline.load_empirical_bundle_bytes(
            bundles[record["bundle_path"]]
        )
        self.assertEqual(
            baseline._forecast_dict(loaded.predict(record["point_id_sha256"])),
            record["forecast"],
        )

        tampered = json.loads(json.dumps(document))
        tampered["estimator"]["unexpected"] = True
        tampered.pop("bundle_payload_sha256")
        tampered["bundle_payload_sha256"] = baseline._semantic_sha256(tampered)
        with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "not exact"):
            baseline.load_empirical_bundle_bytes(baseline._canonical_bytes(tampered))

    def test_artifact_is_immutable_closed_and_replays_every_prediction(self) -> None:
        results, bundles = _build_synthetic_results()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = baseline.publish_artifact(
                root,
                "workspace/data_foundation/baselines/synthetic",
                results=results,
                bundles=bundles,
                pre_publish_check=lambda: None,
            )
            manifest = baseline.verify_artifact(artifact)
            self.assertRegex(manifest["artifact_id"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "overwrite"):
                baseline.publish_artifact(
                    root,
                    "workspace/data_foundation/baselines/synthetic",
                    results=results,
                    bundles=bundles,
                    pre_publish_check=lambda: None,
                )
            (artifact / "extra.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "extra"):
                baseline.verify_artifact(artifact)

    def test_invalid_temporary_artifact_is_never_published(self) -> None:
        results, bundles = _build_synthetic_results()
        corrupt = dict(bundles)
        corrupt[next(iter(corrupt))] = b"{}\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_relative = "workspace/data_foundation/baselines/invalid"
            output = root / output_relative
            with self.assertRaises(baseline.DataFoundationBaselineError):
                baseline.publish_artifact(
                    root,
                    output_relative,
                    results=results,
                    bundles=corrupt,
                    pre_publish_check=lambda: None,
                )
            self.assertFalse(output.exists())
            parent = output.parent
            self.assertFalse(
                parent.exists()
                and any(parent.glob(f".{output.name}.tmp-*"))
            )

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special nodes are unavailable")
    def test_canonical_artifact_verifier_rejects_special_nodes(self) -> None:
        results, bundles = _build_synthetic_results()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = baseline.publish_artifact(
                Path(temporary),
                "workspace/data_foundation/baselines/special-nodes",
                results=results,
                bundles=bundles,
                pre_publish_check=lambda: None,
            )
            fifo = artifact / "unexpected.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError,
                "only regular files and directories",
            ):
                baseline.verify_artifact(artifact)
            fifo.unlink()

            if hasattr(socket, "AF_UNIX"):
                unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    unix_socket.bind(str(artifact / "unexpected.sock"))
                    with self.assertRaisesRegex(
                        baseline.DataFoundationBaselineError,
                        "only regular files and directories",
                    ):
                        baseline.verify_artifact(artifact)
                finally:
                    unix_socket.close()

    def test_bundle_cross_link_with_identical_forecast_is_rejected(self) -> None:
        results, bundles = _build_synthetic_results()
        changed_results = json.loads(json.dumps(results))
        record = changed_results["cells"][0]["seed_results"][0]["predictions"][0]
        original_document = json.loads(
            bundles[record["bundle_path"]].decode("utf-8")
        )
        original_document["identity"]["fold"] = (
            original_document["identity"]["fold"] + 1
        ) % baseline.FOLDS
        original_document.pop("bundle_payload_sha256")
        original_document["bundle_payload_sha256"] = baseline._semantic_sha256(
            original_document
        )
        cross_link_path = "bundles/cross-link/same-forecast.json"
        changed_bundles = dict(bundles)
        changed_bundles[cross_link_path] = baseline._canonical_bytes(original_document)
        record["bundle_path"] = cross_link_path
        record["bundle_payload_sha256"] = original_document["bundle_payload_sha256"]
        changed_results["bundle_count"] += 1
        _rehash_results(changed_results)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "workspace/data_foundation/baselines/cross-link"
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "identities differ"
            ):
                baseline.publish_artifact(
                    root,
                    "workspace/data_foundation/baselines/cross-link",
                    results=changed_results,
                    bundles=changed_bundles,
                    pre_publish_check=lambda: None,
                )
            self.assertFalse(output.exists())

    def test_metrics_tamper_with_rehashed_results_is_rejected(self) -> None:
        results, bundles = _build_synthetic_results()
        changed_results = json.loads(json.dumps(results))
        metrics = changed_results["cells"][0]["seed_results"][0]["metrics"]
        metrics["mae"] += 1.0
        _rehash_results(changed_results)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "metrics do not replay"
            ):
                baseline.publish_artifact(
                    root,
                    "workspace/data_foundation/baselines/metrics-tamper",
                    results=changed_results,
                    bundles=bundles,
                    pre_publish_check=lambda: None,
                )

    def test_split_rotation_and_weight_tampering_are_rejected(self) -> None:
        results, bundles = _build_synthetic_results()
        changed_split = json.loads(json.dumps(results))
        partition = changed_split["cells"][0]["seed_results"][0]["partitions"][0]
        partition["train_task_id_sha256"], partition["validation_task_id_sha256"] = (
            partition["validation_task_id_sha256"],
            partition["train_task_id_sha256"],
        )
        _rehash_results(changed_split)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "partition rotation"
            ):
                baseline.publish_artifact(
                    Path(temporary),
                    "workspace/data_foundation/baselines/split-tamper",
                    results=changed_split,
                    bundles=bundles,
                    pre_publish_check=lambda: None,
                )

        changed_weight = json.loads(json.dumps(results))
        record = changed_weight["cells"][0]["seed_results"][0]["predictions"][0]
        record["sample_weight"] *= 100
        _rehash_results(changed_weight)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "weights do not replay"
            ):
                baseline.publish_artifact(
                    Path(temporary),
                    "workspace/data_foundation/baselines/weight-tamper",
                    results=changed_weight,
                    bundles=bundles,
                    pre_publish_check=lambda: None,
                )

    def test_frozen_condition_cells_and_not_estimable_gates_are_exact(self) -> None:
        results, bundles = _build_synthetic_results()
        gates = results["not_estimable_conditions"]
        self.assertEqual(
            {gate["condition_id"] for gate in gates},
            baseline.FROZEN_BAGEN_NOT_ESTIMABLE_CONDITIONS,
        )
        self.assertTrue(
            all(
                gate["prediction_count"] == 0
                and gate["bundle_count"] == 0
                and gate["target_values_used_for_fit_calibration_scoring"] is False
                and "metrics" not in gate
                and "predictions" not in gate
                for gate in gates
            )
        )
        self.assertNotIn(
            "required_final_holdout_task_count",
            results["condition_gate_policy"],
        )
        self.assertTrue(
            all(
                "final_holdout_task_count" not in gate
                and "final_holdout_task_set_sha256" not in gate
                and "required_final_holdout_task_count" not in gate
                for gate in gates
            )
        )

        deleted = json.loads(json.dumps(results))
        removed_cell = next(
            cell for cell in deleted["cells"] if cell["source_name"] == "bagen_swebench"
        )
        deleted["cells"].remove(removed_cell)
        removed_bundle_paths = {
            record["bundle_path"]
            for seed in removed_cell["seed_results"]
            for record in seed["predictions"]
        }
        reduced_bundles = {
            path: payload
            for path, payload in bundles.items()
            if path not in removed_bundle_paths
        }
        deleted["bundle_count"] = len(reduced_bundles)
        _rehash_results(deleted)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "estimable condition set"
            ):
                baseline.publish_artifact(
                    Path(temporary),
                    "workspace/data_foundation/baselines/deleted-cell",
                    results=deleted,
                    bundles=reduced_bundles,
                    pre_publish_check=lambda: None,
                )

        gate_tamper = json.loads(json.dumps(results))
        gate_tamper["not_estimable_conditions"][0]["metrics"] = {"mae": 0.0}
        _rehash_results(gate_tamper)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "gate is invalid"
            ):
                baseline.publish_artifact(
                    Path(temporary),
                    "workspace/data_foundation/baselines/gate-tamper",
                    results=gate_tamper,
                    bundles=bundles,
                    pre_publish_check=lambda: None,
                )

    def test_paths_reject_whitespace_and_linked_ancestors(self) -> None:
        with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "canonical"):
            baseline._safe_relative(" workspace/data.json", label="fixture")
        with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "canonical"):
            baseline._safe_relative("workspace/data.json ", label="fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "linked"
            try:
                os.symlink(target, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable on this platform")
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "symlink|junction|reparse"
            ):
                baseline._repo_path(root, "linked/file.json", label="fixture")

    def test_current_runner_commit_may_differ_from_frozen_audit_commit(self) -> None:
        results, _ = _build_synthetic_results()
        binding = results["source_binding"]
        self.assertEqual(binding["git_commit"], "4" * 40)
        self.assertEqual(binding["data_foundation_audit_git_commit"], "2" * 40)
        self.assertNotEqual(
            binding["git_commit"], binding["data_foundation_audit_git_commit"]
        )

    def test_git_binding_rejects_dirty_runner_or_src(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "token_prediction" / "__init__.py"
            runner = root / baseline.RUNNER_RELATIVE
            source.parent.mkdir(parents=True)
            runner.parent.mkdir(parents=True)
            controls = (
                root / "configs" / "data_foundation_v2_baseline.json",
                root / "configs" / "source_descriptors" / "bagen.json",
            )
            for control in controls:
                control.parent.mkdir(parents=True, exist_ok=True)
                control.write_bytes(b"{}\n")
            (root / ".gitattributes").write_bytes(b"*.py text eol=lf\n")
            source.write_bytes(b"VALUE = 1\n")
            runner.write_bytes(b"VALUE = 2\n")
            git = baseline._git_executable()
            for arguments in (
                ("init", "-q"),
                ("config", "user.email", "ci@example.invalid"),
                ("config", "user.name", "CI"),
                (
                    "add",
                    ".gitattributes",
                    "src",
                    "configs",
                    baseline.RUNNER_RELATIVE,
                ),
                ("commit", "-q", "-m", "fixture"),
            ):
                subprocess.run((git, "-C", str(root), *arguments), check=True)
            binding = baseline.capture_code_binding(root)
            self.assertRegex(binding.git_commit, r"^[0-9a-f]{40}$")
            control_relatives = tuple(
                control.relative_to(root).as_posix() for control in controls
            )
            control_hash = baseline.tracked_control_tree_sha256(
                root, control_relatives, git_commit=binding.git_commit
            )
            self.assertRegex(control_hash, r"^[0-9a-f]{64}$")
            controls[0].write_bytes(b'{"dirty":true}\n')
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "tracked-clean"
            ):
                baseline.tracked_control_tree_sha256(
                    root, control_relatives, git_commit=binding.git_commit
                )
            source.write_bytes(b"VALUE = 3\n")
            with self.assertRaisesRegex(baseline.DataFoundationBaselineError, "clean"):
                baseline.capture_code_binding(root)

    def test_execution_origin_must_match_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        baseline.verify_execution_origin(repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "does not originate"
            ):
                baseline.verify_execution_origin(Path(temporary))


class DataFoundationBaselineLockTests(unittest.TestCase):
    def _write_lock_fixture(self, root: Path) -> tuple[Path, list[Path]]:
        def write_json(relative: str, value: object) -> tuple[Path, str, int]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = baseline._canonical_bytes(value)
            path.write_bytes(payload)
            return path, baseline._sha256_bytes(payload), len(payload)

        raw_paths: list[Path] = []
        family_records: list[dict[str, object]] = []
        raw_index: list[tuple[str, int, str]] = []
        manifest_lines: list[str] = []
        for index in range(5):
            raw_relative = (
                f"workspace/external/bagen/origin/family-{index}/task-{index}.traj.json"
            )
            raw_path = root / raw_relative
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_payload = f"{index:02d}".encode("ascii")
            raw_path.write_bytes(raw_payload)
            raw_paths.append(raw_path)
            raw_sha = baseline._sha256_bytes(raw_payload)
            raw_index.append((raw_relative, len(raw_payload), raw_sha))
            local_path = f"task-{index}.traj.json"
            family_audit = {
                "raw_files": [
                    {"path": local_path, "bytes": len(raw_payload), "sha256": raw_sha}
                ],
                "source_hashes": {local_path: raw_sha},
            }
            audit_relative = f"workspace/external/bagen/audits/family-{index}.json"
            _, audit_sha, audit_bytes = write_json(audit_relative, family_audit)
            family_records.append(
                {
                    "local_relative_root": (
                        f"workspace/external/bagen/origin/family-{index}"
                    ),
                    "audit_path": audit_relative,
                    "audit_sha256": audit_sha,
                    "audit_bytes": audit_bytes,
                }
            )
            manifest_lines.append(
                json.dumps(
                    {
                        "path": f"origin/family-{index}/{local_path}",
                        "size_bytes": len(raw_payload),
                    },
                    sort_keys=True,
                )
            )
        manifest_path = root / "workspace/external/bagen/manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        manifest_sha = baseline._sha256_file(manifest_path)

        combined = {"families": family_records}
        combined["audit_payload_sha256"] = baseline._semantic_sha256(combined)
        combined_path, combined_sha, combined_bytes = write_json(
            "workspace/external/bagen/combined.json", combined
        )
        framed_sha, raw_bytes = baseline._framed_file_index_sha256(raw_index)
        spend_inventory_path, spend_manifest_sha, spend_manifest_bytes = write_json(
            "workspace/external/spend/inventory.json", {"fixture": True}
        )

        bagen_descriptor = SourceDescriptor(
            source_id=BagenSwebenchReader.source_id,
            revision="a" * 40,
            manifest_path=manifest_path.relative_to(root).as_posix(),
            manifest_sha256=manifest_sha,
            capabilities=BagenSwebenchReader.capabilities,
        )
        spend_descriptor = SourceDescriptor(
            source_id=OpenHandsArchiveReader.source_id,
            revision="b" * 40,
            manifest_path=spend_inventory_path.relative_to(root).as_posix(),
            manifest_sha256=spend_manifest_sha,
            capabilities=OpenHandsArchiveReader.capabilities,
        )
        descriptor_records: dict[str, tuple[SourceDescriptor, str, str, int]] = {}
        for name, descriptor in (
            ("bagen_swebench", bagen_descriptor),
            ("spend_openhands", spend_descriptor),
        ):
            relative = f"configs/{name}.json"
            _, sha, size = write_json(relative, descriptor.to_dict())
            descriptor_records[name] = (descriptor, relative, sha, size)

        source_payloads: dict[str, dict[str, object]] = {}
        lock_sources: dict[str, dict[str, object]] = {}
        for name, descriptor in (
            ("bagen_swebench", bagen_descriptor),
            ("spend_openhands", spend_descriptor),
        ):
            _, descriptor_relative, descriptor_sha, descriptor_bytes = descriptor_records[name]
            dataset_id = ("6" if name == "bagen_swebench" else "7") * 64
            row_count = 5 if name == "bagen_swebench" else 8
            if name == "bagen_swebench":
                artifacts = {
                    "descriptor": {
                        "path": descriptor_relative,
                        "sha256": descriptor_sha,
                        "bytes": descriptor_bytes,
                        "file_count": 1,
                    },
                    "manifest": {
                        "path": descriptor.manifest_path,
                        "sha256": descriptor.manifest_sha256,
                        "bytes": manifest_path.stat().st_size,
                        "file_count": 1,
                    },
                    "combined_audit": {
                        "path": combined_path.relative_to(root).as_posix(),
                        "sha256": combined_sha,
                        "bytes": combined_bytes,
                        "file_count": 1,
                    },
                    "raw_trajectories": {
                        "path": "workspace/external/bagen/origin",
                        "sha256": framed_sha,
                        "bytes": raw_bytes,
                        "file_count": 5,
                        "sha256_kind": "framed_file_index_v1",
                    },
                }
            else:
                artifacts = {
                    "descriptor": {
                        "path": descriptor_relative,
                        "sha256": descriptor_sha,
                        "bytes": descriptor_bytes,
                        "file_count": 1,
                    },
                    "inventory": {
                        "path": descriptor.manifest_path,
                        "sha256": descriptor.manifest_sha256,
                        "bytes": spend_manifest_bytes,
                        "file_count": 1,
                    },
                    "archive": {
                        "path": "workspace/external/spend/archive.tar.gz",
                        "sha256": "8" * 64,
                        "bytes": 123,
                        "file_count": 1,
                        "sha256_kind": "file_bytes",
                    },
                }
            source_payloads[name] = {
                "source_descriptor": descriptor.to_dict(),
                "artifacts": artifacts,
                "dataset": {"dataset_id": dataset_id, "row_count": row_count},
            }
            lock_sources[name] = {
                "descriptor_file_sha256": descriptor_sha,
                "source_descriptor_hash": descriptor.descriptor_hash,
                "capability_contract_hash": descriptor.capabilities.contract_hash,
                "manifest_sha256": descriptor.manifest_sha256,
                "dataset_id": dataset_id,
                "row_count": row_count,
            }

        audit = {
            "dataset_schema_version": 2,
            "implementation": {
                "git_commit": "9" * 40,
                "source_tree_sha256": "a" * 64,
            },
            "sources": source_payloads,
        }
        audit["audit_payload_sha256"] = baseline._semantic_sha256(audit)
        audit_path, audit_sha, audit_bytes = write_json(
            "workspace/data_foundation/audit.json", audit
        )
        lock = {
            "baseline_schema_version": 1,
            "baseline_type": "data_foundation_v2",
            "implementation": {
                "git_commit": "9" * 40,
                "source_tree_sha256": "a" * 64,
            },
            "production_audit": {
                "relative_path": audit_path.relative_to(root).as_posix(),
                "file_sha256": audit_sha,
                "audit_payload_sha256": audit["audit_payload_sha256"],
                "bytes": audit_bytes,
            },
            "sources": lock_sources,
        }
        lock_path, _, _ = write_json("configs/data_foundation_v2_baseline.json", lock)
        return lock_path, raw_paths

    def test_lock_closes_active_readers_and_same_size_raw_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, raw_paths = self._write_lock_fixture(root)
            context = baseline.load_lock_context(
                root, lock_path.relative_to(root).as_posix()
            )
            selected = baseline._load_bagen_manifest(
                root, context.sources["bagen_swebench"]
            )
            self.assertEqual(set(selected), set(raw_paths))
            victim = raw_paths[0]
            victim.write_bytes(b"xx")
            self.assertEqual(victim.stat().st_size, 2)
            with self.assertRaisesRegex(
                baseline.DataFoundationBaselineError, "SHA-256"
            ):
                baseline._load_bagen_manifest(
                    root, context.sources["bagen_swebench"]
                )


if __name__ == "__main__":
    unittest.main()
