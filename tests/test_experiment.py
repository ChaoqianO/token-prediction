from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from types import MappingProxyType

from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
    make_task_split_plan,
)
from token_prediction.estimators import (
    EstimatorRegistry,
    ObservedTransition,
    RunContext,
    TokenForecast,
    builtin_registry,
)
from token_prediction.experiment import (
    AblationAxis,
    AblationSpec,
    CandidateRole,
    CandidateSpec,
    ExperimentRunner,
    ExperimentSpec,
    FoldArtifact,
    compare_candidate_results,
    _transition_spend,
    validate_ablation_specs,
)
from token_prediction.features import FULL_FEATURE_SET, NO_FEATURES, FeatureGroup, FeatureSet

from tests.helpers import make_two_call_trajectory


class ExperimentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = build_supervised_dataset(
            make_two_call_trajectory(task, run)
            for task in range(5)
            for run in range(2)
        )
        self.split = make_task_split_plan(
            self.dataset.task_ids,
            dataset_id=self.dataset.dataset_id,
            folds=5,
            seed=13,
        )

    def test_baselines_share_exact_cohort_split_and_metrics(self) -> None:
        candidates = (
            CandidateSpec(
                "empirical",
                "empirical_quantile",
                NO_FEATURES,
                role=CandidateRole.BASELINE,
            ),
            CandidateSpec(
                "length",
                "length_only",
                FeatureSet(
                    "length",
                    include_all=False,
                    include_features=frozenset({"current_request_tokens_local"}),
                ),
                role=CandidateRole.BASELINE,
            ),
        )
        results = ExperimentRunner(builtin_registry()).run(
            self.dataset,
            self.split,
            ExperimentSpec(
                "call-pre",
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                candidates,
            ),
            seed=13,
        )
        compare_candidate_results(results)
        point_sets = [set(record.point_id for record in result.predictions) for result in results]
        self.assertEqual(point_sets[0], point_sets[1])
        self.assertEqual(len(point_sets[0]), 20)
        for result in results:
            self.assertEqual(result.split_plan_id, self.split.split_plan_id)
            self.assertEqual(result.metrics["n_points"], 20)
            self.assertEqual(result.metrics["n_tasks"], 5)
            self.assertAlmostEqual(float(result.metrics["weight_sum"]), 5.0)
            self.assertEqual(set(result.fold_metrics), set(range(self.split.folds)))
            self.assertTrue(
                all(metrics["n_tasks"] == 1 for metrics in result.fold_metrics.values())
            )
            self.assertEqual(result.fold_artifacts, ())

    def test_ablation_may_change_only_declared_axis(self) -> None:
        without_history = FeatureSet(
            "without_history",
            exclude_groups=frozenset({FeatureGroup.G1}),
        )
        reference = CandidateSpec("full", "empirical_quantile", FULL_FEATURE_SET)
        valid = CandidateSpec(
            "without-history",
            "empirical_quantile",
            without_history,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                "full",
                AblationAxis.FEATURE_SET,
                frozenset({"feature_set"}),
            ),
        )
        validate_ablation_specs((reference, valid))
        invalid = CandidateSpec(
            "invalid",
            "length_only",
            without_history,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                "full",
                AblationAxis.FEATURE_SET,
                frozenset({"feature_set"}),
            ),
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_ablation_specs((reference, invalid))

    def test_registry_extension_predicts_without_test_labels(self) -> None:
        audit: dict[str, list[str]] = {"fit": [], "predict": []}

        @dataclass
        class Session:
            target: PredictionTarget

            def predict(self, point):
                self_outer.assertFalse(hasattr(point, "label"))
                audit["predict"].append(point.point_id)
                return TokenForecast(point.point_id, self.target, 1.0, 1.0, 1.0)

            def observe(self, transition: ObservedTransition) -> None:
                del transition

        @dataclass
        class Fitted:
            estimator_id: str
            target: PredictionTarget

            def start(self, context: RunContext):
                del context
                return Session(self.target)

        class SpyEstimator:
            estimator_id = "spy"

            def fit(self, train, validation, context):
                del validation, context
                audit["fit"].extend(example.point.point_id for example in train.examples)
                return Fitted(self.estimator_id, train.target)

        self_outer = self
        registry = EstimatorRegistry()
        registry.register("spy", lambda params: SpyEstimator())
        results = ExperimentRunner(registry).run(
            self.dataset,
            self.split,
            ExperimentSpec(
                "spy",
                PredictionPosition.TASK_PRE,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                (CandidateSpec("spy", "spy", FULL_FEATURE_SET),),
                calibrator_id="none",
            ),
            seed=1,
        )
        self.assertEqual(len(results[0].predictions), 10)
        self.assertTrue(audit["fit"])
        self.assertTrue(audit["predict"])

    def test_transition_spend_recovers_after_an_earlier_missing_attempt(self) -> None:
        template = next(
            row.point
            for row in self.dataset.rows
            if row.point.position == PredictionPosition.TASK_UPDATE
        )
        previous = replace(
            template,
            point_id="previous",
            features={
                "missing_usage_attempts": 1,
                "cumulative_provider_input_tokens": 100,
                "cumulative_provider_output_tokens": 20,
            },
        )
        current = replace(
            template,
            point_id="current",
            cutoff_event_seq=template.cutoff_event_seq + 1,
            features={
                "missing_usage_attempts": 1,
                "cumulative_provider_input_tokens": 130,
                "cumulative_provider_output_tokens": 25,
            },
        )
        self.assertEqual(_transition_spend(previous, current), 35)

        newly_missing = replace(
            current,
            features={
                "missing_usage_attempts": 2,
                "cumulative_provider_input_tokens": 140,
                "cumulative_provider_output_tokens": 26,
            },
        )
        self.assertIsNone(_transition_spend(previous, newly_missing))

        decreasing_missing = replace(
            current,
            features={
                "missing_usage_attempts": 0,
                "cumulative_provider_input_tokens": 130,
                "cumulative_provider_output_tokens": 25,
            },
        )
        with self.assertRaisesRegex(ValueError, "missing usage"):
            _transition_spend(previous, decreasing_missing)

    def test_optional_fitted_audit_interfaces_are_collected_per_fold(self) -> None:
        @dataclass(frozen=True)
        class FitReport:
            fold_name: str
            parameters: object

        @dataclass(frozen=True)
        class Importance:
            source_feature_name: str
            gain: float

        class Encoder:
            def to_dict(self):
                return MappingProxyType(
                    {
                        "schema_version": 1,
                        "columns": ({"name": "task_tokens"},),
                    }
                )

        @dataclass
        class Session:
            target: PredictionTarget

            def predict(self, point):
                return TokenForecast(point.point_id, self.target, 1.0, 1.0, 1.0)

            def observe(self, transition: ObservedTransition) -> None:
                del transition

        @dataclass
        class Fitted:
            estimator_id: str
            target: PredictionTarget
            fit_report: FitReport
            encoder: Encoder

            def start(self, context: RunContext):
                del context
                return Session(self.target)

            def source_feature_importance(self):
                return (Importance("task_tokens", 2.5),)

            def model_strings(self):
                return MappingProxyType({"q50": "model text"})

            def bundle_files(self):
                return MappingProxyType({"manifest.json": b"{}"})

        class AuditedEstimator:
            estimator_id = "audited"

            def fit(self, train, validation, context):
                del validation
                return Fitted(
                    self.estimator_id,
                    train.target,
                    FitReport(
                        f"fold-{context.fold}",
                        MappingProxyType({"deterministic": True}),
                    ),
                    Encoder(),
                )

        registry = EstimatorRegistry()
        registry.register("audited", lambda params: AuditedEstimator())
        result = ExperimentRunner(registry).run(
            self.dataset,
            self.split,
            ExperimentSpec(
                "audited",
                PredictionPosition.TASK_PRE,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                (CandidateSpec("audited", "audited", FULL_FEATURE_SET),),
                calibrator_id="none",
            ),
            seed=3,
        )[0]

        self.assertEqual(len(result.fold_artifacts), self.split.folds)
        for fold, artifact in enumerate(result.fold_artifacts):
            self.assertEqual(artifact.fold, fold)
            self.assertEqual(artifact.encoder["schema_version"], 1)
            self.assertEqual(artifact.fit_report["fold_name"], f"fold-{fold}")
            self.assertEqual(
                artifact.fit_report["parameters"], {"deterministic": True}
            )
            self.assertEqual(
                artifact.feature_importance[0]["source_feature_name"], "task_tokens"
            )
            self.assertEqual(artifact.model_strings["q50"], "model text")
            self.assertEqual(artifact.bundle_files["manifest.json"], b"{}")

    def test_fold_artifact_rejects_unsafe_bundle_payloads(self) -> None:
        for name in ("", ".", "..", "../manifest.json", "a/b", "a\\b"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    FoldArtifact(fold=0, bundle_files={name: b"payload"})
        with self.assertRaises(TypeError):
            FoldArtifact(fold=0, bundle_files={"manifest.json": "not-bytes"})


if __name__ == "__main__":
    unittest.main()
