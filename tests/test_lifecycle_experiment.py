from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

import token_prediction.lifecycle_bundle as lifecycle_bundle_module
from token_prediction import __version__ as TOKEN_PREDICTION_VERSION
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.crossfit import (
    POINT_ONLY_SEED_POLICY_HASH,
    POINT_ONLY_SEED_POLICY_ID,
    SEED_POLICY_ID,
)
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    assign_inner_task_folds,
    build_capability_supervised_dataset,
    build_lifecycle_slice,
    lifecycle_scored_hash,
    make_task_split_plan,
)
from token_prediction.development import OuterInnerPlan
from token_prediction.estimators import (
    EstimatorRegistry,
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    builtin_registry,
)
from token_prediction.estimators.cross_position_deduct import (
    CrossPositionDeductEstimator,
)
from token_prediction.experiment import (
    CandidateGraph,
    CandidateSpec,
    ExperimentRunner,
    ExperimentSpec,
)
from token_prediction.features import FULL_FEATURE_SET, NO_FEATURES
from token_prediction.lifecycle_experiment import (
    TASK_LIFECYCLE_SCHEMA_ID,
    run_lifecycle_candidate_cv,
)
from token_prediction.lifecycle_bundle import (
    LifecycleBundleError,
    load_lifecycle_bundle,
    validate_source_provenance,
)

from tests.helpers import make_two_call_trajectory


TARGET = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
HAS_NEURAL = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors")
)


def _without_latency(forecast: TokenForecast) -> TokenForecast:
    return replace(forecast, latency_ms=0.0)


def _descriptor() -> SourceDescriptor:
    return SourceDescriptor(
        source_id="lifecycle-cv-fixture",
        revision="revision-1",
        manifest_path="workspace/manifests/lifecycle-cv-fixture.json",
        manifest_sha256="c" * 64,
        capabilities=SourceCapabilities(
            source_id="lifecycle-cv-fixture",
            observables=frozenset(
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.TASK_TERMINATION,
                    Observable.TASK_USAGE,
                }
            ),
        ),
    )


@dataclass
class _LeakGuardSession:
    target: PredictionTarget
    fit_tasks: frozenset[str]

    def predict(self, point: object) -> TokenForecast:
        task_id = getattr(point, "task_id")
        if task_id in self.fit_tasks:
            raise AssertionError("an OOF Task-pre point entered its initializer fit set")
        return TokenForecast(
            point_id=getattr(point, "point_id"),
            target=self.target,
            lower=400.0,
            point=500.0,
            upper=600.0,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _LeakGuardFitted:
    estimator_id: str
    target: PredictionTarget
    fit_tasks: frozenset[str]
    lower: float = 400.0
    point: float = 500.0
    upper: float = 600.0

    def start(self, context: RunContext) -> _LeakGuardSession:
        del context
        return _LeakGuardSession(self.target, self.fit_tasks)

    def bundle_files(self) -> Mapping[str, bytes]:
        # Deliberately excludes raw task identities. Partition membership is
        # carried only as salted task digests by the lifecycle composite bundle.
        return {"model.json": b'{"kind":"leak-guard","value":500}'}


class _LeakGuardInitializer:
    estimator_id = "empirical_quantile"

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _LeakGuardFitted:
        del context
        train_tasks = frozenset(example.point.task_id for example in train.examples)
        validation_tasks = frozenset(example.point.task_id for example in validation.examples)
        if not train_tasks or not validation_tasks or train_tasks & validation_tasks:
            raise AssertionError("initializer fit/validation partition is invalid")
        return _LeakGuardFitted(self.estimator_id, train.target, train_tasks)


class _SeedAuditedCrossPositionDeduct(CrossPositionDeductEstimator):
    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> object:
        train_sequences = tuple(train.lifecycle_sequences or ())
        validation_sequences = tuple(validation.lifecycle_sequences or ())
        if not train_sequences or not validation_sequences:
            raise AssertionError("updater did not receive lifecycle sequences")
        for sequence in (*train_sequences, *validation_sequences):
            first = sequence.steps[0]
            if first.label is not None or first.status != LabelStatus.MISSING:
                raise AssertionError("updater could read the true Task-pre label")
            if first.invalid_reason != "redacted_task_pre_label":
                raise AssertionError("Task-pre redaction was not explicit")
        train_bundle_counts = {
            len(sequence.session_seed.component_bundle_hashes) for sequence in train_sequences
        }
        validation_bundle_counts = {
            len(sequence.session_seed.component_bundle_hashes) for sequence in validation_sequences
        }
        if len(train_bundle_counts) != 1 or len(validation_bundle_counts) != 1:
            raise AssertionError("seed producer cardinality is inconsistent")
        if max(train_bundle_counts) >= min(validation_bundle_counts):
            raise AssertionError("outer-train did not receive OOF-only initializer seeds")
        return super().fit(train, validation, context)


def _registry() -> EstimatorRegistry:
    registry = EstimatorRegistry()
    registry.register(
        "empirical_quantile",
        lambda params: _LeakGuardInitializer(),
    )
    registry.register(
        "cross_position_deduct",
        lambda params: _SeedAuditedCrossPositionDeduct(**params),
    )
    return registry


def _candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="lifecycle-cross-position-deduct",
        estimator_id="cross_position_deduct",
        feature_set=FULL_FEATURE_SET,
        graph=CandidateGraph(
            initializer_estimator_id="empirical_quantile",
            updater_estimator_id="cross_position_deduct",
            lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
            seed_policy_id=SEED_POLICY_ID,
            inner_split_policy_id=INNER_FOLD_POLICY_ID,
        ),
    )


def _empirical_candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="empirical-seeded-cross-position-deduct",
        estimator_id="cross_position_deduct",
        feature_set=FULL_FEATURE_SET,
        graph=CandidateGraph(
            initializer_estimator_id="empirical_quantile",
            updater_estimator_id="cross_position_deduct",
            lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
            seed_policy_id=SEED_POLICY_ID,
            inner_split_policy_id=INNER_FOLD_POLICY_ID,
        ),
    )


def _inner_plans(split_plan: SplitPlan) -> dict[int, OuterInnerPlan]:
    return {
        outer_fold: OuterInnerPlan(
            split_seed=split_plan.seed,
            outer_test_fold=outer_fold,
            outer_split_plan_id=split_plan.split_plan_id,
            assignment=assign_inner_task_folds(
                split_plan.partition(outer_fold).train_tasks,
                seed=split_plan.seed,
            ),
        )
        for outer_fold in range(split_plan.folds)
    }


def _source_provenance() -> dict[str, object]:
    descriptor = _descriptor()
    return {
        "source_descriptor": descriptor.to_dict(),
        "source_descriptor_hash": descriptor.descriptor_hash,
        "code_hash": "d" * 64,
        "runtime_versions": {
            "python_version": platform.python_version(),
            "token_prediction_version": TOKEN_PREDICTION_VERSION,
        },
    }


class SourceProvenanceValidationTests(unittest.TestCase):
    def test_source_provenance_lifecycle_capability_gate_is_explicit(self) -> None:
        descriptor = SourceDescriptor(
            source_id="point-only-fixture",
            revision="revision-1",
            manifest_path="workspace/manifests/point-only-fixture.json",
            manifest_sha256="f" * 64,
            capabilities=SourceCapabilities(
                source_id="point-only-fixture",
                observables=frozenset(
                    {Observable.TASK_TERMINATION, Observable.TASK_USAGE}
                ),
            ),
        )
        provenance = {
            "source_descriptor": descriptor.to_dict(),
            "source_descriptor_hash": descriptor.descriptor_hash,
            "code_hash": "d" * 64,
            "runtime_versions": {
                "python_version": platform.python_version(),
                "token_prediction_version": TOKEN_PREDICTION_VERSION,
            },
        }

        validated = validate_source_provenance(
            provenance,
            source_descriptor_hash=descriptor.descriptor_hash,
            capability_contract_hash=descriptor.capabilities.contract_hash,
        )
        self.assertEqual(validated["source_descriptor"], descriptor.to_dict())
        with self.assertRaisesRegex(
            LifecycleBundleError,
            "cannot produce the lifecycle target",
        ):
            validate_source_provenance(
                provenance,
                source_descriptor_hash=descriptor.descriptor_hash,
                capability_contract_hash=descriptor.capabilities.contract_hash,
                require_lifecycle_capabilities=True,
            )


class LifecycleExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        trajectories = tuple(make_two_call_trajectory(task) for task in range(15))
        cls.dataset = build_capability_supervised_dataset(
            trajectories,
            _descriptor(),
        )
        cls.lifecycle = build_lifecycle_slice(cls.dataset, target=TARGET)
        cls.split_plan = make_task_split_plan(
            cls.lifecycle.task_ids,
            dataset_id=cls.dataset.dataset_id,
            folds=5,
            seed=20260719,
        )
        cls.inner_plans = _inner_plans(cls.split_plan)
        cls.source_provenance = _source_provenance()
        cls.candidate = _candidate()
        cls.result = run_lifecycle_candidate_cv(
            cls.dataset,
            cls.lifecycle,
            cls.split_plan,
            cls.candidate,
            _registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=cls.split_plan.seed,
            inner_plans=cls.inner_plans,
            source_provenance=cls.source_provenance,
        )
        cls.empirical_result = run_lifecycle_candidate_cv(
            cls.dataset,
            cls.lifecycle,
            cls.split_plan,
            _empirical_candidate(),
            builtin_registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=cls.split_plan.seed,
            inner_plans=cls.inner_plans,
            source_provenance=cls.source_provenance,
        )

    def test_nested_cv_predicts_exact_task_update_cohort_once(self) -> None:
        eligible = self.dataset.select(
            PredictionPosition.TASK_UPDATE,
            TARGET,
            condition_id=self.lifecycle.condition_id,
        )
        actual = [record.point_id for record in self.result.predictions]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {row.point.point_id for row in eligible.rows})
        self.assertTrue(all(record.target == TARGET for record in self.result.predictions))
        self.assertEqual(self.result.position, PredictionPosition.TASK_UPDATE)
        self.assertEqual(len(self.result.fold_artifacts), 5)

    def test_builtin_empirical_initializer_runs_cross_position_deduct(self) -> None:
        self.assertEqual(
            {record.point_id for record in self.empirical_result.predictions},
            {step.point.point_id for step in self.lifecycle.scored_steps},
        )
        self.assertEqual(len(self.empirical_result.fold_artifacts), 5)

    def test_point_only_seed_policy_bundle_reloads_calibrated_trajectories(self) -> None:
        primary = _empirical_candidate()
        candidate = replace(
            primary,
            candidate_id="point-only-seeded-cross-position-deduct",
            graph=replace(
                primary.graph,
                seed_policy_id=POINT_ONLY_SEED_POLICY_ID,
            ),
        )
        result = run_lifecycle_candidate_cv(
            self.dataset,
            self.lifecycle,
            self.split_plan,
            candidate,
            builtin_registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=self.split_plan.seed,
            inner_plans=self.inner_plans,
            source_provenance=self.source_provenance,
        )
        self.assertEqual(len(result.fold_artifacts), 5)
        for artifact in result.fold_artifacts:
            bundle = dict(artifact.bundle_files or {})
            manifest = json.loads(bundle["manifest.json"])
            self.assertEqual(
                manifest["seed_policy_id"],
                POINT_ONLY_SEED_POLICY_ID,
            )
            self.assertEqual(
                manifest["seed_policy_hash"],
                POINT_ONLY_SEED_POLICY_HASH,
            )
            loaded = load_lifecycle_bundle(
                bundle,
                expected_source_provenance=self.source_provenance,
            )
            test_tasks = self.split_plan.partition(artifact.fold).test_tasks
            sequences = tuple(
                sequence
                for sequence in self.lifecycle.sequences
                if sequence.task_id in test_tasks
            )
            seeds = loaded.external_seeds(sequences)
            self.assertTrue(
                all(
                    seed.forecast.lower
                    == seed.forecast.point
                    == seed.forecast.upper
                    for seed in seeds.values()
                )
            )
            replay = loaded.run_calibrated(sequences)
            expected = {
                record.point_id: _without_latency(record.forecast)
                for record in result.predictions
                if record.fold == artifact.fold
            }
            actual = {
                prediction.step.point.point_id: _without_latency(
                    prediction.forecast
                )
                for run in replay
                for prediction in run.scored_predictions
            }
            self.assertEqual(actual, expected)

    @unittest.skipUnless(HAS_NEURAL, "requires token-prediction[neural]")
    def test_gru_lifecycle_bundle_reloads_complete_calibrated_trajectories(self) -> None:
        candidate = CandidateSpec(
            candidate_id="gru-residual-small",
            estimator_id="gru_residual",
            feature_set=FULL_FEATURE_SET,
            params={
                "transition_dim": 8,
                "hidden_dim": 8,
                "residual_head_dim": 8,
                "max_epochs": 2,
                "patience": 1,
            },
            graph=CandidateGraph(
                initializer_estimator_id="empirical_quantile",
                updater_estimator_id="gru_residual",
                lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
                seed_policy_id=SEED_POLICY_ID,
                inner_split_policy_id=INNER_FOLD_POLICY_ID,
            ),
        )
        result = run_lifecycle_candidate_cv(
            self.dataset,
            self.lifecycle,
            self.split_plan,
            candidate,
            builtin_registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=self.split_plan.seed,
            inner_plans=self.inner_plans,
            source_provenance=self.source_provenance,
        )
        self.assertEqual(
            {record.point_id for record in result.predictions},
            {step.point.point_id for step in self.lifecycle.scored_steps},
        )
        for artifact in result.fold_artifacts:
            loaded = load_lifecycle_bundle(
                dict(artifact.bundle_files or {}),
                expected_source_provenance=self.source_provenance,
            )
            test_tasks = self.split_plan.partition(artifact.fold).test_tasks
            sequences = tuple(
                sequence
                for sequence in self.lifecycle.sequences
                if sequence.task_id in test_tasks
            )
            replay = loaded.run_calibrated(sequences)
            expected = {
                record.point_id: _without_latency(record.forecast)
                for record in result.predictions
                if record.fold == artifact.fold
            }
            actual = {
                prediction.step.point.point_id: _without_latency(prediction.forecast)
                for run in replay
                for prediction in run.scored_predictions
            }
            self.assertEqual(actual, expected)

    def test_experiment_runner_routes_mixed_point_and_lifecycle_candidates(self) -> None:
        lifecycle_candidate = CandidateSpec(
            candidate_id="runner-cross-position-deduct",
            estimator_id="cross_position_deduct",
            feature_set=FULL_FEATURE_SET,
            graph=CandidateGraph(
                initializer_estimator_id="empirical_quantile",
                updater_estimator_id="cross_position_deduct",
                lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
                seed_policy_id=SEED_POLICY_ID,
                inner_split_policy_id=INNER_FOLD_POLICY_ID,
            ),
        )
        point_candidate = CandidateSpec(
            candidate_id="runner-empirical-point",
            estimator_id="empirical_quantile",
            feature_set=FULL_FEATURE_SET,
        )
        results = ExperimentRunner(builtin_registry()).run(
            self.dataset,
            self.split_plan,
            ExperimentSpec(
                experiment_id="mixed-runner",
                position=PredictionPosition.TASK_UPDATE,
                target=TARGET,
                candidates=(point_candidate, lifecycle_candidate),
                calibrator_id="none",
                condition_id=self.lifecycle.condition_id,
            ),
            seed=self.split_plan.seed,
            source_provenance=self.source_provenance,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {record.point_id for record in results[0].predictions},
            {record.point_id for record in results[1].predictions},
        )

    def test_experiment_runner_routes_only_the_selected_condition_task_subset(self) -> None:
        dataset = build_capability_supervised_dataset(
            (make_two_call_trajectory(task) for task in range(30)),
            _descriptor(),
        )
        split_plan = make_task_split_plan(
            dataset.task_ids,
            dataset_id=dataset.dataset_id,
            folds=5,
            seed=20260719,
        )
        condition_tasks: set[str] = set()
        for fold in range(5):
            fold_tasks = sorted(
                task for task, assigned in split_plan.assignments if assigned == fold
            )
            condition_tasks.update(fold_tasks[:3])
        condition_id = "condition:selected-family"
        conditioned = replace(
            dataset,
            rows=tuple(
                replace(row, point=replace(row.point, condition_id=condition_id))
                if row.point.task_id in condition_tasks
                else replace(
                    row,
                    point=replace(row.point, condition_id="condition:other-family"),
                )
                for row in dataset.rows
            ),
        )
        with patch(
            "token_prediction.lifecycle_experiment.run_lifecycle_candidate_cv",
            side_effect=RuntimeError("routing inspected"),
        ) as mocked:
            with self.assertRaisesRegex(RuntimeError, "routing inspected"):
                ExperimentRunner(builtin_registry()).run(
                    conditioned,
                    split_plan,
                    ExperimentSpec(
                        experiment_id="condition-routing",
                        position=PredictionPosition.TASK_UPDATE,
                        target=TARGET,
                        candidates=(_candidate(),),
                        calibrator_id="none",
                        condition_id=condition_id,
                    ),
                    seed=split_plan.seed,
                    source_provenance=_source_provenance(),
                )
        lifecycle_slice = mocked.call_args.args[1]
        self.assertEqual(lifecycle_slice.task_ids, frozenset(condition_tasks))
        self.assertNotEqual(
            lifecycle_slice.task_ids,
            frozenset(task for task, _fold in split_plan.assignments),
        )
        inner_plans = mocked.call_args.kwargs["inner_plans"]
        self.assertEqual(set(inner_plans), set(range(5)))
        for outer_fold, inner_plan in inner_plans.items():
            self.assertEqual(
                inner_plan.assignment.task_ids,
                split_plan.partition(outer_fold).train_tasks,
            )

    def test_sparse_condition_composite_bundles_reload_against_global_inner_plan(self) -> None:
        dataset = build_capability_supervised_dataset(
            (make_two_call_trajectory(task) for task in range(30)),
            _descriptor(),
        )
        split_plan = make_task_split_plan(
            dataset.task_ids,
            dataset_id=dataset.dataset_id,
            folds=5,
            seed=20260719,
        )
        excluded_task = sorted(dataset.task_ids)[0]
        condition_id = "condition:sparse-family"
        conditioned = replace(
            dataset,
            rows=tuple(
                replace(
                    row,
                    point=replace(
                        row.point,
                        condition_id=(
                            "condition:other-family"
                            if row.point.task_id == excluded_task
                            else condition_id
                        ),
                    ),
                )
                for row in dataset.rows
            ),
        )
        lifecycle = build_lifecycle_slice(
            conditioned,
            target=TARGET,
            condition_id=condition_id,
        )
        result = run_lifecycle_candidate_cv(
            conditioned,
            lifecycle,
            split_plan,
            _empirical_candidate(),
            builtin_registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=split_plan.seed,
            inner_plans=_inner_plans(split_plan),
            source_provenance=_source_provenance(),
        )
        for artifact in result.fold_artifacts:
            loaded = load_lifecycle_bundle(
                dict(artifact.bundle_files or {}),
                expected_source_provenance=_source_provenance(),
            )
            test_tasks = split_plan.partition(artifact.fold).test_tasks & lifecycle.task_ids
            sequence = next(
                item for item in lifecycle.sequences if item.task_id in test_tasks
            )
            replay = loaded.run_calibrated((sequence,))[0]
            expected = {
                record.point_id: _without_latency(record.forecast)
                for record in result.predictions
                if record.fold == artifact.fold
            }
            self.assertEqual(
                [_without_latency(item.forecast) for item in replay.predictions],
                [expected[item.step.point.point_id] for item in replay.predictions],
            )

    def test_oof_guard_and_composite_bundle_publish_no_raw_task_ids(self) -> None:
        # setUpClass completing proves every inner OOF prediction was made by a
        # component whose fit set did not contain that task.
        for artifact in self.result.fold_artifacts:
            self.assertIsNotNone(artifact.bundle_files)
            bundle = dict(artifact.bundle_files or {})
            self.assertIn("manifest.json", bundle)
            self.assertIn("manifest.sha256", bundle)
            self.assertIn("calibrator.json", bundle)
            published = b"\n".join(bundle.values())
            for task_id in self.lifecycle.task_ids:
                self.assertNotIn(task_id.encode("utf-8"), published)

    def test_updater_cannot_inspect_task_pre_label(self) -> None:
        # _SeedAuditedCrossPositionDeduct reads the complete updater-facing
        # lifecycle object and raises unless Task-pre is genuinely redacted.
        self.assertTrue(self.result.predictions)

    def test_nonreloadable_lifecycle_dag_fails_before_training(self) -> None:
        unsupported = replace(
            _empirical_candidate(),
            candidate_id="unsupported-lifecycle-dag",
            graph=CandidateGraph(
                initializer_estimator_id="opaque_initializer",
                updater_estimator_id="cross_position_deduct",
                lifecycle_schema_id=TASK_LIFECYCLE_SCHEMA_ID,
                seed_policy_id=SEED_POLICY_ID,
                inner_split_policy_id=INNER_FOLD_POLICY_ID,
            ),
        )
        with self.assertRaisesRegex(ValueError, "reloadable"):
            run_lifecycle_candidate_cv(
                self.dataset,
                self.lifecycle,
                self.split_plan,
                unsupported,
                builtin_registry(),
                alpha=0.10,
                calibrator_id="none",
                seed=self.split_plan.seed,
                inner_plans=self.inner_plans,
                source_provenance=self.source_provenance,
            )

    def test_frozen_inner_plan_identity_is_consumed_without_reassignment(self) -> None:
        for artifact in self.empirical_result.fold_artifacts:
            bundle = dict(artifact.bundle_files or {})
            manifest = json.loads(bundle["manifest.json"])
            self.assertEqual(
                manifest["inner_split_id"],
                self.inner_plans[artifact.fold].assignment.assignment_id,
            )

        outer_fold = 0
        outer_tasks = self.split_plan.partition(outer_fold).train_tasks
        reduced = outer_tasks - {next(iter(outer_tasks))}
        invalid = dict(self.inner_plans)
        invalid[outer_fold] = OuterInnerPlan(
            split_seed=self.split_plan.seed,
            outer_test_fold=outer_fold,
            outer_split_plan_id=self.split_plan.split_plan_id,
            assignment=assign_inner_task_folds(
                reduced,
                seed=self.split_plan.seed,
            ),
        )
        with self.assertRaisesRegex(ValueError, "full outer-train"):
            run_lifecycle_candidate_cv(
                self.dataset,
                self.lifecycle,
                self.split_plan,
                _empirical_candidate(),
                builtin_registry(),
                alpha=0.10,
                calibrator_id="none",
                seed=self.split_plan.seed,
                inner_plans=invalid,
                source_provenance=self.source_provenance,
            )

    def test_composite_restart_replays_calibrated_unscored_context(self) -> None:
        fold = 0
        artifact = self.empirical_result.fold_artifacts[fold]
        bundle = dict(artifact.bundle_files or {})
        loaded = load_lifecycle_bundle(bundle)
        manifest = loaded.manifest
        descriptor = _descriptor()
        self.assertEqual(manifest["source_descriptor"], descriptor.to_dict())
        self.assertEqual(manifest["dataset_schema_version"], self.dataset.schema_version)
        self.assertIn("feature_schema_version", manifest)
        self.assertIn("lifecycle_schema_version", manifest)
        self.assertEqual(manifest["code_hash"], "d" * 64)
        self.assertEqual(len(loaded.initializers), 5)

        test_tasks = self.split_plan.partition(fold).test_tasks
        original = next(
            sequence for sequence in self.lifecycle.sequences if sequence.task_id in test_tasks
        )
        steps = list(original.steps)
        steps[1] = replace(
            steps[1],
            loss_mask=False,
            score_mask=False,
            sample_weight=0.0,
        )
        sequence = replace(
            original,
            steps=tuple(steps),
            scored_hash=lifecycle_scored_hash(original.context_hash, steps),
        )
        offline = loaded.run_calibrated((sequence,), runtime_mode="offline")[0]
        shadow = loaded.run_calibrated((sequence,), runtime_mode="shadow")[0]
        self.assertEqual(len(offline.predictions), len(sequence.steps) - 1)
        self.assertFalse(offline.predictions[0].step.score_mask)
        self.assertEqual(
            [_without_latency(item.forecast) for item in offline.predictions],
            [_without_latency(item.forecast) for item in shadow.predictions],
        )
        self.assertTrue(all(item.forecast.latency_ms > 0 for item in offline.predictions))
        self.assertTrue(all(item.forecast.latency_ms > 0 for item in shadow.predictions))
        expected = {
            record.point_id: _without_latency(record.forecast)
            for record in self.empirical_result.predictions
            if record.fold == fold
        }
        self.assertEqual(
            [_without_latency(item.forecast) for item in offline.predictions],
            [expected[item.step.point.point_id] for item in offline.predictions],
        )

    def test_restricted_feature_bundle_preserves_task_pre_protocol_counters(self) -> None:
        candidate = replace(
            _empirical_candidate(),
            candidate_id="restricted-empirical-seeded-deduct",
            feature_set=NO_FEATURES,
        )
        result = run_lifecycle_candidate_cv(
            self.dataset,
            self.lifecycle,
            self.split_plan,
            candidate,
            builtin_registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=self.split_plan.seed,
            inner_plans=self.inner_plans,
            source_provenance=self.source_provenance,
        )
        artifact = result.fold_artifacts[0]
        loaded = load_lifecycle_bundle(dict(artifact.bundle_files or {}))
        test_tasks = self.split_plan.partition(0).test_tasks
        sequence = next(
            item for item in self.lifecycle.sequences if item.task_id in test_tasks
        )
        seed = loaded.external_seeds((sequence,))[sequence.steps[0].point.point_id]
        self.assertEqual(
            set(seed.task_pre_point.features),
            {
                "missing_usage_attempts",
                "cumulative_provider_input_tokens",
                "cumulative_provider_output_tokens",
            },
        )
        replay = loaded.run_calibrated((sequence,))[0]
        expected = {
            record.point_id: _without_latency(record.forecast)
            for record in result.predictions
            if record.fold == 0
        }
        self.assertEqual(
            [_without_latency(item.forecast) for item in replay.predictions],
            [expected[item.step.point.point_id] for item in replay.predictions],
        )

    def test_composite_loader_rejects_unknown_tampered_extra_and_symlink(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        unknown = dict(bundle)
        unknown_manifest = json.loads(unknown["manifest.json"])
        unknown_manifest["candidate_graph"]["initializer_estimator_id"] = "unknown"
        unknown_bytes = json.dumps(
            unknown_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        unknown["manifest.json"] = unknown_bytes
        unknown["manifest.sha256"] = (
            hashlib.sha256(unknown_bytes).hexdigest() + "\n"
        ).encode("ascii")
        with self.assertRaisesRegex(LifecycleBundleError, "unsupported lifecycle"):
            load_lifecycle_bundle(unknown)

        extra = {**bundle, "extra.bin": b"unexpected"}
        with self.assertRaisesRegex(LifecycleBundleError, "extra files"):
            load_lifecycle_bundle(extra)
        tampered = dict(bundle)
        tampered["calibrator.json"] += b" "
        with self.assertRaisesRegex(LifecycleBundleError, "checksum mismatch"):
            load_lifecycle_bundle(tampered)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            for name, payload in bundle.items():
                destination = root.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            outside = Path(temporary) / "outside.json"
            outside.write_bytes(bundle["calibrator.json"])
            linked = root / "calibrator.json"
            linked.unlink()
            try:
                linked.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(LifecycleBundleError, "symlink/reparse"):
                load_lifecycle_bundle(root)

    def test_composite_loader_enforces_runtime_and_expected_provenance(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        loaded = load_lifecycle_bundle(
            bundle,
            expected_source_provenance=self.source_provenance,
        )
        self.assertEqual(loaded.manifest["code_hash"], self.source_provenance["code_hash"])

        incompatible = dict(bundle)
        manifest = json.loads(incompatible["manifest.json"])
        manifest["runtime_versions"]["python_version"] = "0.0.0-impossible"
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        incompatible["manifest.json"] = manifest_bytes
        incompatible["manifest.sha256"] = (
            hashlib.sha256(manifest_bytes).hexdigest() + "\n"
        ).encode("ascii")
        with self.assertRaisesRegex(LifecycleBundleError, "Python.*incompatible"):
            load_lifecycle_bundle(incompatible)

        unexpected = {**self.source_provenance, "code_hash": "0" * 64}
        with self.assertRaisesRegex(LifecycleBundleError, "expected provenance"):
            load_lifecycle_bundle(
                bundle,
                expected_source_provenance=unexpected,
            )

    def test_composite_loader_rejects_resigned_unknown_dataset_or_feature_schema(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        for field in ("dataset_schema_version", "feature_schema_version"):
            tampered = dict(bundle)
            manifest = json.loads(tampered["manifest.json"])
            manifest[field] = 999
            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            tampered["manifest.json"] = manifest_bytes
            tampered["manifest.sha256"] = (
                hashlib.sha256(manifest_bytes).hexdigest() + "\n"
            ).encode("ascii")
            with self.subTest(field=field):
                with self.assertRaisesRegex(LifecycleBundleError, "unsupported lifecycle"):
                    load_lifecycle_bundle(tampered)

    def test_composite_loader_rechecks_capability_and_input_contract(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})

        insufficient_descriptor = replace(
            _descriptor(),
            capabilities=SourceCapabilities(
                source_id=_descriptor().source_id,
                observables=frozenset({Observable.TASK_USAGE}),
            ),
        )
        insufficient = dict(bundle)
        manifest = json.loads(insufficient["manifest.json"])
        manifest["source_descriptor"] = insufficient_descriptor.to_dict()
        manifest["source_descriptor_hash"] = insufficient_descriptor.descriptor_hash
        manifest["capability_contract_hash"] = (
            insufficient_descriptor.capabilities.contract_hash
        )
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        insufficient["manifest.json"] = manifest_bytes
        insufficient["manifest.sha256"] = (
            hashlib.sha256(manifest_bytes).hexdigest() + "\n"
        ).encode("ascii")
        with self.assertRaisesRegex(LifecycleBundleError, "cannot produce"):
            load_lifecycle_bundle(insufficient)

        inconsistent = dict(bundle)
        manifest = json.loads(inconsistent["manifest.json"])
        manifest["input_contract_hash"] = "0" * 64
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        inconsistent["manifest.json"] = manifest_bytes
        inconsistent["manifest.sha256"] = (
            hashlib.sha256(manifest_bytes).hexdigest() + "\n"
        ).encode("ascii")
        with self.assertRaisesRegex(LifecycleBundleError, "input contract"):
            load_lifecycle_bundle(inconsistent)

    def test_composite_loader_rejects_resigned_overlapping_inner_partition(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        manifest = json.loads(bundle["manifest.json"])
        summary = manifest["initializer_components"][0]
        old_hash = summary["component_hash"]
        old_prefix = f"components/{old_hash}/"
        descriptor_name = f"{old_prefix}component.json"
        descriptor = json.loads(bundle[descriptor_name])
        partitions = descriptor["task_partitions_sha256"]
        partitions["fit"] = sorted({*partitions["fit"], partitions["holdout"][0]})

        semantic = dict(descriptor)
        semantic.pop("component_hash")
        semantic_bytes = json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        new_hash = hashlib.sha256(semantic_bytes).hexdigest()
        descriptor["component_hash"] = new_hash
        descriptor_bytes = json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        new_prefix = f"components/{new_hash}/"
        resigned: dict[str, bytes] = {}
        for name, payload in bundle.items():
            if name in {"manifest.json", "manifest.sha256"}:
                continue
            if name.startswith(old_prefix):
                suffix = name.removeprefix(old_prefix)
                resigned[f"{new_prefix}{suffix}"] = (
                    descriptor_bytes if suffix == "component.json" else payload
                )
            else:
                resigned[name] = payload
        summary["component_hash"] = new_hash
        summary["bundle_hashes"] = sorted(
            hashlib.sha256(payload).hexdigest()
            for name, payload in resigned.items()
            if name.startswith(new_prefix)
        )
        manifest["files"] = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(resigned.items())
        }
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        resigned["manifest.json"] = manifest_bytes
        resigned["manifest.sha256"] = (
            hashlib.sha256(manifest_bytes).hexdigest() + "\n"
        ).encode("ascii")

        with self.assertRaisesRegex(LifecycleBundleError, "partitions overlap"):
            load_lifecycle_bundle(resigned)

    def test_composite_loader_rejects_file_swapped_to_same_byte_symlink(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            for name, payload in bundle.items():
                destination = root.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            outside = Path(temporary) / "outside.json"
            outside.write_bytes(bundle["calibrator.json"])
            probe = root / "symlink-probe"
            try:
                probe.symlink_to(outside)
                probe.unlink()
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            victim = root / "calibrator.json"
            original_open = os.open
            swapped = False

            def racing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
                nonlocal swapped
                if not swapped and Path(path) == victim:
                    victim.unlink()
                    victim.symlink_to(outside)
                    swapped = True
                return original_open(path, flags, *args, **kwargs)

            with patch("token_prediction.lifecycle_bundle.os.open", side_effect=racing_open):
                with self.assertRaisesRegex(
                    LifecycleBundleError,
                    "cannot open|changed identity|resolved through a link|regular non-link",
                ):
                    load_lifecycle_bundle(root)
            self.assertTrue(swapped)

    def test_composite_loader_binds_scan_metadata_before_reading(self) -> None:
        bundle = dict(self.empirical_result.fold_artifacts[0].bundle_files or {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            for name, payload in bundle.items():
                destination = root.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)

            victim = root / "calibrator.json"
            original_read = lifecycle_bundle_module._read_regular_file
            rewritten = False

            def racing_read(
                path: Path,
                *,
                expected_metadata: os.stat_result,
                maximum_bytes: int,
                description: str,
            ) -> bytes:
                nonlocal rewritten
                if not rewritten and path == victim:
                    payload = victim.read_bytes()
                    victim.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                    os.utime(
                        victim,
                        ns=(
                            expected_metadata.st_atime_ns,
                            expected_metadata.st_mtime_ns + 2_000_000_000,
                        ),
                    )
                    rewritten = True
                return original_read(
                    path,
                    expected_metadata=expected_metadata,
                    maximum_bytes=maximum_bytes,
                    description=description,
                )

            with patch(
                "token_prediction.lifecycle_bundle._read_regular_file",
                side_effect=racing_read,
            ):
                with self.assertRaisesRegex(LifecycleBundleError, "changed before reading"):
                    load_lifecycle_bundle(root)
            self.assertTrue(rewritten)

    def test_test_suffix_label_mutation_cannot_change_that_fold_predictions(self) -> None:
        fold = 0
        test_tasks = self.split_plan.partition(fold).test_tasks
        row = next(
            item
            for item in self.dataset.rows
            if item.point.position == PredictionPosition.TASK_UPDATE
            and item.point.target == TARGET
            and item.point.task_id in test_tasks
        )
        changed = replace(row, label=int(row.label or 0) + 10_000)
        mutated_dataset = replace(
            self.dataset,
            rows=tuple(
                changed if item.point.point_id == row.point.point_id else item
                for item in self.dataset.rows
            ),
        )
        mutated_lifecycle = build_lifecycle_slice(mutated_dataset, target=TARGET)
        self.assertEqual(self.lifecycle.context_hash, mutated_lifecycle.context_hash)
        self.assertEqual(self.lifecycle.scored_hash, mutated_lifecycle.scored_hash)

        mutated_result = run_lifecycle_candidate_cv(
            mutated_dataset,
            mutated_lifecycle,
            self.split_plan,
            self.candidate,
            _registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=self.split_plan.seed,
            inner_plans=self.inner_plans,
            source_provenance=self.source_provenance,
        )
        baseline_fold = {
            record.point_id: _without_latency(record.forecast)
            for record in self.result.predictions
            if record.fold == fold
        }
        mutated_fold = {
            record.point_id: _without_latency(record.forecast)
            for record in mutated_result.predictions
            if record.fold == fold
        }
        self.assertEqual(baseline_fold, mutated_fold)
        baseline_bundle = self.result.fold_artifacts[fold].bundle_files
        mutated_bundle = mutated_result.fold_artifacts[fold].bundle_files
        self.assertEqual(baseline_bundle, mutated_bundle)

    def test_rebuilt_outer_test_label_cannot_change_that_fold_seed_identity(self) -> None:
        fold = 0
        test_tasks = self.split_plan.partition(fold).test_tasks
        row = next(
            item
            for item in self.dataset.rows
            if item.point.position == PredictionPosition.TASK_UPDATE
            and item.point.target == TARGET
            and item.point.task_id in test_tasks
        )
        changed = replace(row, label=int(row.label or 0) + 10_000)
        mutated_dataset = replace(
            self.dataset,
            dataset_id="9" * 64,
            rows=tuple(
                changed if item.point.point_id == row.point.point_id else item
                for item in self.dataset.rows
            ),
        )
        mutated_lifecycle = build_lifecycle_slice(mutated_dataset, target=TARGET)
        mutated_split = make_task_split_plan(
            mutated_lifecycle.task_ids,
            dataset_id=mutated_dataset.dataset_id,
            folds=5,
            seed=self.split_plan.seed,
        )
        self.assertEqual(self.split_plan.assignments, mutated_split.assignments)
        mutated_result = run_lifecycle_candidate_cv(
            mutated_dataset,
            mutated_lifecycle,
            mutated_split,
            self.candidate,
            _registry(),
            alpha=0.10,
            calibrator_id="task_max_conformal",
            seed=mutated_split.seed,
            inner_plans=_inner_plans(mutated_split),
            source_provenance=self.source_provenance,
        )
        baseline_forecasts = {
            record.point_id: _without_latency(record.forecast)
            for record in self.result.predictions
            if record.fold == fold
        }
        mutated_forecasts = {
            record.point_id: _without_latency(record.forecast)
            for record in mutated_result.predictions
            if record.fold == fold
        }
        self.assertEqual(baseline_forecasts, mutated_forecasts)
        baseline_manifest = json.loads(
            dict(self.result.fold_artifacts[fold].bundle_files or {})["manifest.json"]
        )
        mutated_manifest = json.loads(
            dict(mutated_result.fold_artifacts[fold].bundle_files or {})["manifest.json"]
        )
        self.assertEqual(
            baseline_manifest["initializer_hash"],
            mutated_manifest["initializer_hash"],
        )
        self.assertEqual(
            baseline_manifest["initializer_components"],
            mutated_manifest["initializer_components"],
        )
        self.assertEqual(
            baseline_manifest["seed_set_hash"],
            mutated_manifest["seed_set_hash"],
        )

    def test_lifecycle_score_mask_must_match_public_eligibility(self) -> None:
        omitted_task = next(iter(self.lifecycle.task_ids))
        masked = build_lifecycle_slice(
            self.dataset,
            target=TARGET,
            scored_task_ids=self.lifecycle.task_ids - {omitted_task},
        )
        with self.assertRaisesRegex(ValueError, "scored cohort"):
            run_lifecycle_candidate_cv(
                self.dataset,
                masked,
                self.split_plan,
                self.candidate,
                _registry(),
                alpha=0.10,
                calibrator_id="none",
                seed=self.split_plan.seed,
                inner_plans=self.inner_plans,
                source_provenance=self.source_provenance,
            )


if __name__ == "__main__":
    unittest.main()
