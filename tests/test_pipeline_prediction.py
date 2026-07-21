from __future__ import annotations

import importlib.util
import hashlib
import json
import platform
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.config import ConfiguredExperiment, load_config
from token_prediction.collection import CodexTurnMetadata, CodexTurnReader
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    build_supervised_dataset,
)
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    build_development_protocol,
    verify_development_audit_document,
)
from token_prediction.estimators import (
    EstimatorRegistry,
    RunContext,
    TokenForecast,
    load_lightgbm_bundle,
)
from token_prediction.experiment import (
    CandidateRole,
    CandidateSpec,
    ExperimentSpec,
)
from token_prediction.features import FULL_FEATURE_SET, FeatureSet
from token_prediction.lineage import ArtifactVerificationError, verify_artifact
from token_prediction.pipeline import (
    _experiment_runtime_versions,
    load_trajectory,
    run_configured_experiments,
    run_development_experiments,
)

from tests.helpers import make_two_call_trajectory, write_trajectory


ROOT = Path(__file__).resolve().parents[1]
CODEX_FIXTURE = ROOT / "tests" / "fixtures" / "codex_turn_events.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v2_config(
    base_config: object,
    root: Path,
    event_paths: list[Path],
    *,
    observables: frozenset[Observable] | None = None,
) -> object:
    source_id = "canonical_fixture_v2"
    revision = "fixture-revision-1"
    manifest_path = root / "manifests" / "canonical-source.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_schema_version": 1,
        "source_id": source_id,
        "revision": revision,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(event_paths)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capabilities = SourceCapabilities(
        source_id=source_id,
        source="test",
        observables=observables
        or frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.REQUEST_LOCAL_COUNT,
                Observable.TASK_TERMINATION,
                Observable.TASK_USAGE,
            }
        ),
    )
    descriptor = SourceDescriptor(
        source_id=source_id,
        revision=revision,
        manifest_path="manifests/canonical-source.json",
        manifest_sha256=_sha256(manifest_path),
        capabilities=capabilities,
    )
    descriptor_path = root / "configs" / "source_descriptors" / "fixture.json"
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        json.dumps(descriptor.to_dict(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replace(
        base_config,
        source_path=root / "configs" / "test.toml",
        source_hash="schema-v2-test-config",
        schema_version=2,
        source_descriptor=descriptor,
        source_descriptor_path="configs/source_descriptors/fixture.json",
        source_descriptor_file_sha256=_sha256(descriptor_path),
        canonical_manifest_path="manifests/canonical-source.json",
        canonical_manifest_sha256=_sha256(manifest_path),
    )


class PredictionPipelineSmokeTests(unittest.TestCase):
    def test_neural_runtime_versions_are_run_bound_and_missing_fails_closed(self) -> None:
        spec = ExperimentSpec(
            experiment_id="independent-mlp-runtime",
            position=PredictionPosition.TASK_PRE,
            target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            candidates=(
                CandidateSpec(
                    "independent-mlp",
                    "independent_mlp",
                    FULL_FEATURE_SET,
                ),
            ),
        )
        installed = {
            "token-prediction": "0.1.0",
            "numpy": "2.1.3",
            "torch": "2.6.0",
            "safetensors": "0.5.3",
        }
        with (
            patch(
                "token_prediction.pipeline._installed_version",
                side_effect=lambda distribution: installed.get(
                    distribution, "not-installed"
                ),
            ),
            patch(
                "token_prediction.pipeline._module_version",
                side_effect=lambda distribution, _module: installed.get(
                    distribution, "not-installed"
                ),
            ),
        ):
            runtime = _experiment_runtime_versions((spec,))
        self.assertEqual(runtime["numpy_version"], "2.1.3")
        self.assertEqual(runtime["torch_version"], "2.6.0")
        self.assertEqual(runtime["safetensors_version"], "0.5.3")

        installed["safetensors"] = "not-installed"
        with patch(
            "token_prediction.pipeline._module_version",
            side_effect=lambda distribution, _module: installed.get(
                distribution, "not-installed"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "safetensors_version"):
                _experiment_runtime_versions((spec,))

    def test_schema_v1_is_verification_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "task.jsonl"
            write_trajectory(path, make_two_call_trajectory(0, 0))
            with self.assertRaisesRegex(ValueError, "verification-only"):
                run_configured_experiments(
                    load_config(ROOT / "configs" / "mvp.toml"),
                    [path],
                    output_dir=root / "experiment",
                )
            self.assertFalse((root / "experiment").exists())

    def test_schema_v2_rejects_non_frozen_fold_or_seed_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "task.jsonl"
            write_trajectory(path, make_two_call_trajectory(0, 0))
            config = _v2_config(
                load_config(ROOT / "configs" / "mvp.toml"),
                root,
                [path],
            )
            with self.assertRaisesRegex(ValueError, "exactly five outer folds"):
                run_configured_experiments(
                    replace(config, folds=4),
                    [path],
                    output_dir=root / "bad-folds",
                )
            with self.assertRaisesRegex(ValueError, "frozen development protocol"):
                run_configured_experiments(
                    replace(config, seed=STAGE_SPLIT_SEEDS[1]),
                    [path],
                    output_dir=root / "bad-seed",
                )

    def test_config_drives_real_fit_predict_evaluate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                for run in range(2):
                    path = root / f"task-{task}-run-{run}.jsonl"
                    write_trajectory(path, make_two_call_trajectory(task, run))
                    event_paths.append(path)
            output = root / "experiment"
            config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            summary = run_configured_experiments(
                config,
                event_paths,
                output_dir=output,
            )
            self.assertEqual(summary.experiment_count, 3)
            self.assertEqual(summary.candidate_run_count, 18)
            manifest = verify_artifact(output)
            self.assertEqual(manifest.artifact_id, summary.artifact_id)
            self.assertIn("code_hash", manifest.metadata)
            self.assertIn("source_hashes", manifest.metadata)
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(predictions)
            self.assertTrue(all("prediction" in row and "point_id" in row for row in predictions))
            self.assertTrue(all("label" not in row and "y_true" not in row for row in predictions))
            self.assertEqual(
                {row["split_seed"] for row in predictions},
                set(STAGE_SPLIT_SEEDS),
            )
            assert config.source_descriptor is not None
            parent_dataset = build_capability_supervised_dataset(
                (load_trajectory(path) for path in event_paths),
                config.source_descriptor,
            )
            protocol = build_development_protocol(parent_dataset)
            self.assertEqual(summary.dataset_id, protocol.development_dataset.dataset_id)
            self.assertFalse({row["task_id"] for row in predictions} & protocol.final_holdout_tasks)
            verify_development_audit_document(
                json.loads((output / "split.json").read_text(encoding="utf-8"))
            )
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(metrics),
                {
                    "task_pre_unknown_remaining",
                    "task_update_unknown_remaining",
                    "call_pre_unknown_billable",
                },
            )
            for candidates in metrics.values():
                for candidate in candidates.values():
                    self.assertEqual(
                        set(candidate["split_seed_results"]),
                        {str(seed) for seed in STAGE_SPLIT_SEEDS},
                    )
                    for seed_result in candidate["split_seed_results"].values():
                        self.assertEqual(
                            set(seed_result["fold_metrics"]),
                            {"0", "1", "2", "3", "4"},
                        )
            with patch(
                "token_prediction.pipeline.ExperimentRunner.run",
                side_effect=AssertionError("cache hit retrained an estimator"),
            ):
                repeated = run_configured_experiments(
                    config,
                    event_paths,
                    output_dir=output,
                )
            self.assertEqual(repeated.artifact_id, summary.artifact_id)

    def test_reusable_development_runner_never_exposes_final_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task, 0))
                event_paths.append(path)
            config = _v2_config(
                load_config(ROOT / "configs" / "mvp.toml"),
                root,
                event_paths,
            )
            assert config.source_descriptor is not None
            parent_dataset = build_capability_supervised_dataset(
                (load_trajectory(path) for path in event_paths),
                config.source_descriptor,
            )
            protocol = build_development_protocol(parent_dataset)
            seen_tasks: set[str] = set()

            class Session:
                def predict(self, point):
                    seen_tasks.add(point.task_id)
                    return TokenForecast(
                        point.point_id,
                        point.target,
                        1.0,
                        1.0,
                        1.0,
                    )

                def observe(self, transition):
                    del transition

            class Fitted:
                estimator_id = "protocol_spy"

                def start(self, context):
                    del context
                    return Session()

            class SpyEstimator:
                estimator_id = "protocol_spy"

                def fit(self, train, validation, context):
                    del context
                    seen_tasks.update(
                        example.point.task_id for example in (*train.examples, *validation.examples)
                    )
                    return Fitted()

            registry = EstimatorRegistry()
            registry.register("protocol_spy", lambda params: SpyEstimator())
            condition_id = parent_dataset.rows[0].point.condition_id
            execution = run_development_experiments(
                parent_dataset,
                (
                    ExperimentSpec(
                        experiment_id="protocol-spy",
                        position=PredictionPosition.TASK_PRE,
                        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                        candidates=(
                            CandidateSpec(
                                "protocol-spy",
                                "protocol_spy",
                                FULL_FEATURE_SET,
                            ),
                        ),
                        calibrator_id="none",
                        condition_id=condition_id,
                    ),
                ),
                source_provenance={
                    "source_descriptor": config.source_descriptor.to_dict(),
                    "source_descriptor_hash": (config.source_descriptor.descriptor_hash),
                    "code_hash": "d" * 64,
                    "runtime_versions": {
                        "python_version": platform.python_version(),
                        "token_prediction_version": TOKEN_PREDICTION_VERSION,
                    },
                },
                protocol=protocol,
                registry=registry,
            )
            self.assertEqual(
                tuple(result.split_seed for result in execution.seed_results),
                STAGE_SPLIT_SEEDS,
            )
            self.assertEqual(seen_tasks, protocol.development_dataset.task_ids)
            self.assertFalse(seen_tasks & protocol.final_holdout_tasks)

    def test_codex_turn_aggregate_reaches_task_prediction_without_fake_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            reader = CodexTurnReader()
            for task in range(25):
                trajectory = reader.read(
                    CODEX_FIXTURE,
                    CodexTurnMetadata(
                        task_id=f"codex-task-{task}",
                        task_tokens=20 + task,
                        model_id="gpt-fixture",
                        reasoning_effort="medium",
                        started_at="2026-07-21T00:00:00+00:00",
                        finished_at="2026-07-21T00:00:01+00:00",
                    ),
                )
                path = root / f"codex-task-{task}.jsonl"
                write_trajectory(path, trajectory)
                paths.append(path)
            output = root / "codex-experiment"
            config = _v2_config(
                load_config(ROOT / "configs" / "codex_task_mvp.toml"),
                root,
                paths,
                observables=frozenset({Observable.TASK_TERMINATION, Observable.TASK_USAGE}),
            )
            summary = run_configured_experiments(
                config,
                paths,
                output_dir=output,
            )
            self.assertEqual(summary.experiment_count, 1)
            predictions = (output / "predictions.jsonl").read_text(encoding="utf-8")
            self.assertIn("task_total_accounted_tokens", predictions)
            self.assertNotIn("call_unknown_billable_tokens", predictions)

    def test_config_v2_binds_source_capabilities_into_dataset_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task, 0))
                event_paths.append(path)
            config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            assert config.source_descriptor is not None
            descriptor = config.source_descriptor
            output = root / "experiment-v2"
            summary = run_configured_experiments(config, event_paths, output_dir=output)
            manifest = verify_artifact(output)
            self.assertEqual(manifest.metadata["config_schema_version"], 2)
            self.assertEqual(manifest.metadata["dataset_schema_version"], 2)
            self.assertEqual(
                manifest.metadata["capability_contract_hash"],
                descriptor.capabilities.contract_hash,
            )
            dataset_summary = json.loads(
                (output / "dataset_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(dataset_summary["schema_version"], 2)
            self.assertEqual(summary.dataset_id, manifest.metadata["dataset_id"])

    def test_effective_candidate_change_cannot_reuse_existing_run_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task, 0))
                event_paths.append(path)
            config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            output = root / "experiment"
            first = run_configured_experiments(config, event_paths, output_dir=output)
            changed_candidate = replace(
                config.candidates[0],
                params={**config.candidates[0].params, "alpha": 0.20},
            )
            changed = replace(
                config,
                candidates=(changed_candidate, *config.candidates[1:]),
            )
            with patch(
                "token_prediction.pipeline.ExperimentRunner.run",
                side_effect=AssertionError("cache collision reached training"),
            ):
                with self.assertRaisesRegex(ValueError, "existing experiment artifact"):
                    run_configured_experiments(
                        changed,
                        event_paths,
                        output_dir=output,
                    )
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(verify_artifact(output).metadata["run_id"], first.run_id)

    def test_source_change_during_training_refuses_artifact_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task, 0))
                event_paths.append(path)
            output = root / "experiment"
            config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            with patch(
                "token_prediction.pipeline._source_tree_hash",
                side_effect=("a" * 64, "b" * 64),
            ):
                with self.assertRaisesRegex(RuntimeError, "source tree changed"):
                    run_configured_experiments(
                        config,
                        event_paths,
                        output_dir=output,
                    )
            self.assertFalse(output.exists())

    def test_v2_rejects_untracked_capability_claim_and_input_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task, 0))
                event_paths.append(path)
            config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            assert config.source_descriptor is not None
            untrusted = replace(
                config.source_descriptor,
                capabilities=SourceCapabilities(
                    source_id=config.source_descriptor.source_id,
                    source="untracked-claim",
                    observables=config.source_descriptor.capabilities.observables,
                ),
            )
            with self.assertRaisesRegex(ValueError, "differs from tracked"):
                run_configured_experiments(
                    replace(config, source_descriptor=untrusted),
                    event_paths,
                    output_dir=root / "untrusted",
                )

            event_paths[0].write_bytes(event_paths[0].read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "size or SHA-256"):
                run_configured_experiments(
                    config,
                    event_paths,
                    output_dir=root / "tampered",
                )
            self.assertFalse((root / "untrusted").exists())
            self.assertFalse((root / "tampered").exists())

    def test_v2_rejects_in_root_symlink_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-event.jsonl"
            write_trajectory(target, make_two_call_trajectory(0, 0))
            link = root / "linked-event.jsonl"
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink creation is unavailable: {exc}")
            config = _v2_config(
                load_config(ROOT / "configs" / "mvp.toml"),
                root,
                [link],
            )
            with self.assertRaisesRegex(ValueError, "symlinks.*reparse points"):
                run_configured_experiments(
                    config,
                    [link],
                    output_dir=root / "experiment",
                )

    def test_v2_rejects_event_beneath_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_parent = root / "real-events"
            real_parent.mkdir()
            target = real_parent / "event.jsonl"
            write_trajectory(target, make_two_call_trajectory(0, 0))
            linked_parent = root / "linked-events"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")
            linked_event = linked_parent / target.name
            config = _v2_config(
                load_config(ROOT / "configs" / "mvp.toml"),
                root,
                [linked_event],
            )
            with self.assertRaisesRegex(ValueError, "symlinks.*reparse points"):
                run_configured_experiments(
                    config,
                    [linked_event],
                    output_dir=root / "experiment",
                )

    def test_v2_rejects_whitespace_in_canonical_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "event.jsonl"
            write_trajectory(event, make_two_call_trajectory(0, 0))
            config = _v2_config(
                load_config(ROOT / "configs" / "mvp.toml"),
                root,
                [event],
            )
            unsafe = replace(
                config,
                canonical_manifest_path=f" {config.canonical_manifest_path}",
            )
            with self.assertRaisesRegex(ValueError, "canonical relative POSIX"):
                run_configured_experiments(
                    unsafe,
                    [event],
                    output_dir=root / "experiment",
                )

    @unittest.skipUnless(
        importlib.util.find_spec("lightgbm") is not None
        and importlib.util.find_spec("numpy") is not None,
        "LightGBM estimator extra is not installed",
    )
    def test_lightgbm_fold_audit_artifacts_and_runtime_versions_are_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(25):
                for run in range(2):
                    path = root / f"task-{task}-run-{run}.jsonl"
                    write_trajectory(path, make_two_call_trajectory(task, run))
                    event_paths.append(path)

            feature_set = FeatureSet(
                "request_length",
                include_all=False,
                include_features=frozenset({"current_request_tokens_local"}),
            )
            candidate = CandidateSpec(
                "lightgbm",
                "lightgbm_quantile",
                feature_set,
                params={
                    "num_boost_round": 8,
                    "early_stopping_rounds": 2,
                    "learning_rate": 0.2,
                    "num_leaves": 3,
                    "min_data_in_leaf": 1,
                },
                role=CandidateRole.MODEL,
            )
            base_config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            config = replace(
                base_config,
                source_hash="lightgbm-fold-artifact-test",
                candidates=(candidate,),
                experiments=(
                    ConfiguredExperiment(
                        experiment_id="lightgbm-audit",
                        position=PredictionPosition.CALL_PRE,
                        target=PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                        candidate_ids=("lightgbm",),
                        required_features=frozenset({"current_request_tokens_local"}),
                        condition_id=None,
                    ),
                ),
            )
            output = root / "lightgbm-experiment"
            summary = run_configured_experiments(
                config,
                event_paths,
                output_dir=output,
            )
            manifest = verify_artifact(output)
            self.assertEqual(manifest.artifact_id, summary.artifact_id)
            self.assertNotEqual(manifest.metadata["lightgbm_version"], "not-installed")
            self.assertNotEqual(manifest.metadata["numpy_version"], "not-installed")

            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            seed_metrics = metrics["lightgbm-audit"]["lightgbm"]["split_seed_results"]
            self.assertEqual(
                set(seed_metrics),
                {str(seed) for seed in STAGE_SPLIT_SEEDS},
            )
            for seed_result in seed_metrics.values():
                self.assertEqual(
                    set(seed_result["fold_metrics"]),
                    {"0", "1", "2", "3", "4"},
                )
            published_predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            point_by_id = {
                row.point.point_id: row.point
                for row in build_supervised_dataset(
                    load_trajectory(path) for path in event_paths
                ).rows
            }
            selected_seed = STAGE_SPLIT_SEEDS[0]
            for fold in range(5):
                fold_dir = (
                    output
                    / "fold_artifacts"
                    / "lightgbm-audit"
                    / "lightgbm"
                    / f"seed_{selected_seed}"
                    / f"fold_{fold}"
                )
                self.assertTrue((fold_dir / "encoder.json").is_file())
                fit_report = json.loads((fold_dir / "fit_report.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    fit_report["lightgbm_version"],
                    manifest.metadata["lightgbm_version"],
                )
                self.assertTrue(fit_report["quantiles"])
                importance = [
                    json.loads(line)
                    for line in (fold_dir / "feature_importance.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                ]
                self.assertTrue(importance)
                self.assertTrue(
                    all(
                        row["source_feature_name"] == "current_request_tokens_local"
                        for row in importance
                    )
                )
                for quantile in ("q05", "q50", "q95"):
                    self.assertTrue((fold_dir / f"{quantile}.model.txt").is_file())
                loaded = load_lightgbm_bundle(fold_dir / "bundle")
                self.assertEqual(loaded.dataset_id, summary.dataset_id)
                self.assertEqual(loaded.position, PredictionPosition.CALL_PRE)
                self.assertEqual(
                    loaded.target,
                    PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                )
                self.assertEqual(len(loaded.allowed_condition_ids), 1)
                expected = next(
                    row
                    for row in published_predictions
                    if row["candidate_id"] == "lightgbm"
                    and row["split_seed"] == selected_seed
                    and row["fold"] == fold
                )
                raw_point = point_by_id[expected["point_id"]]
                selected = raw_point.with_features(feature_set.select(raw_point.features))
                forecast = loaded.start(
                    RunContext(
                        selected.task_id,
                        selected.trajectory_id,
                        selected.run_id,
                    )
                ).predict(selected)
                self.assertEqual(
                    (forecast.raw_lower, forecast.raw_point, forecast.raw_upper),
                    (
                        expected["raw_lower"],
                        expected["raw_prediction"],
                        expected["raw_upper"],
                    ),
                )

            nested_manifest = (
                output
                / "fold_artifacts"
                / "lightgbm-audit"
                / "lightgbm"
                / f"seed_{selected_seed}"
                / "fold_0"
                / "bundle"
                / "manifest.json"
            )
            nested_manifest.write_bytes(nested_manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ArtifactVerificationError, "checksum"):
                verify_artifact(output)

    def test_unsafe_experiment_id_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task in range(5):
                path = root / f"task-{task}.jsonl"
                write_trajectory(path, make_two_call_trajectory(task))
                event_paths.append(path)
            base_config = _v2_config(load_config(ROOT / "configs" / "mvp.toml"), root, event_paths)
            malicious = replace(base_config.experiments[0], experiment_id="../outside")
            config = replace(
                base_config,
                source_hash="unsafe-id-test",
                experiments=(malicious,),
            )
            with patch(
                "token_prediction.pipeline.ExperimentRunner.run",
                side_effect=AssertionError("unsafe path reached training"),
            ):
                with self.assertRaisesRegex(ValueError, "unsafe experiment_id"):
                    run_configured_experiments(
                        config,
                        event_paths,
                        output_dir=root / "experiment",
                    )
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
