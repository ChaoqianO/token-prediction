from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from token_prediction.crossfit import (
    SUPPORTED_SEED_POLICY_IDS,
    CrossfitSeedSet,
    InitializerComponent,
    generate_crossfit_seeds,
)
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    LIFECYCLE_SCHEMA_VERSION,
    DatasetRow,
    DatasetSlice,
    InnerFoldPartition,
    LabelStatus,
    LifecycleSequence,
    LifecycleSlice,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    lifecycle_scored_hash,
    point_input_semantic,
    supported_input_contract_hashes_from_capability,
)
from token_prediction.development import OuterInnerPlan
from token_prediction.estimators import (
    EstimatorRegistry,
    FitCheckpoint,
    FitContext,
    ObservedTransition,
    SessionSeed,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.gru_bundle import GRU_BUNDLE_FORMAT
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    IdentityCalibrator,
    ScoredForecast,
    TaskMaxConformalCalibrator,
    evaluate_forecasts,
    evaluate_task_forecasts,
)
from token_prediction.experiment import (
    CandidateResult,
    CandidateSpec,
    FoldArtifact,
    PredictionRecord,
    _collect_fold_artifact,
    _json_compatible,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, FeatureSet
from token_prediction.lifecycle import run_lifecycle_batch
from token_prediction.lifecycle_bundle import (
    CROSS_POSITION_DEDUCT_FORMAT,
    EMPIRICAL_INITIALIZER_FORMAT,
    LIFECYCLE_COMPONENT_SCHEMA_VERSION,
    LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION,
    OPAQUE_AUDIT_FORMAT,
    feature_set_document,
    load_lifecycle_bundle,
    validate_source_provenance,
)


TASK_LIFECYCLE_SCHEMA_ID = f"task_lifecycle_v{LIFECYCLE_SCHEMA_VERSION}"
_PROTOCOL_FEATURES = frozenset(
    {
        "missing_usage_attempts",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_hash(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _task_pseudonym(task_id: str, *, split_plan_id: str) -> str:
    return hashlib.sha256(
        f"lifecycle-audit-v1\0{split_plan_id}\0{task_id}".encode("utf-8")
    ).hexdigest()


def _pseudonymous_tasks(
    tasks: Sequence[str] | frozenset[str],
    *,
    split_plan_id: str,
) -> tuple[str, ...]:
    return tuple(sorted(_task_pseudonym(task, split_plan_id=split_plan_id) for task in tasks))


def _derived_fit_seed(seed: int, *, outer_fold: int, inner_fold: int | None) -> int:
    digest = hashlib.sha256(
        f"lifecycle-fit-v1\0{seed}\0{outer_fold}\0{inner_fold}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _strict_selected_point(point: PredictionPoint, feature_set: FeatureSet) -> PredictionPoint:
    """Select candidate inputs without consulting status, labels, or suffix events."""

    return point.with_features(feature_set.select(point.features))


def _lifecycle_selected_point(
    point: PredictionPoint,
    feature_set: FeatureSet,
) -> PredictionPoint:
    """Keep declared model features plus the mechanical lifecycle contract fields."""

    selected = feature_set.select(point.features)
    for name in _PROTOCOL_FEATURES:
        if name in point.features:
            selected[name] = point.features[name]
    return point.with_features(selected)


def _selected_sequence(
    sequence: LifecycleSequence,
    feature_set: FeatureSet,
) -> LifecycleSequence:
    selected_steps = tuple(
        replace(step, point=_lifecycle_selected_point(step.point, feature_set))
        for step in sequence.steps
    )
    # The Task-pre target is represented exclusively by the OOF SessionSeed.
    # Masking a label is not sufficient: a learned updater must not be able to
    # inspect the true full-task target through lifecycle_sequences.
    steps = (
        replace(
            selected_steps[0],
            label=None,
            status=LabelStatus.MISSING,
            invalid_reason="redacted_task_pre_label",
        ),
        *selected_steps[1:],
    )
    context_hash = _semantic_hash(
        {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "input_contract_hash": sequence.input_contract_hash,
            "task_id": sequence.task_id,
            "trajectory_id": sequence.trajectory_id,
            "run_id": sequence.run_id,
            "condition_id": sequence.condition_id,
            "target": sequence.target.value,
            "points": [point_input_semantic(step.point) for step in steps],
        }
    )
    return replace(
        sequence,
        steps=steps,
        context_hash=context_hash,
        scored_hash=lifecycle_scored_hash(context_hash, steps),
    )


@dataclass
class _SelectedSession:
    delegate: Any
    feature_set: FeatureSet

    def predict(self, point: PredictionPoint) -> TokenForecast:
        return self.delegate.predict(_strict_selected_point(point, self.feature_set))

    def observe(self, transition: ObservedTransition) -> None:
        self.delegate.observe(transition)


@dataclass(frozen=True)
class _SelectedFitted:
    """Restrict initializer inference to the same label-free feature view as fit."""

    estimator_id: str
    delegate: Any
    feature_set: FeatureSet

    def start(self, context: Any) -> _SelectedSession:
        return _SelectedSession(self.delegate.start(context), self.feature_set)


@dataclass(frozen=True)
class SeededLifecycleTrainingSequence:
    """A validated lifecycle sequence paired with its label-free OOF seed."""

    sequence: LifecycleSequence
    session_seed: SessionSeed

    def __post_init__(self) -> None:
        first = self.sequence.steps[0].point
        seed_point = self.session_seed.task_pre_point
        if (
            first.point_id != seed_point.point_id
            or first.task_id != seed_point.task_id
            or first.trajectory_id != seed_point.trajectory_id
            or first.run_id != seed_point.run_id
            or first.condition_id != seed_point.condition_id
            or first.target != seed_point.target
        ):
            raise ValueError("updater lifecycle sequence does not match its SessionSeed")

    @property
    def seed(self) -> SessionSeed:
        return self.session_seed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sequence, name)


def _make_initializer_view(
    dataset_slice: DatasetSlice,
    rows: Sequence[DatasetRow],
    weights: Mapping[str, float],
    feature_set: FeatureSet,
    *,
    dataset_id: str,
    partition_name: str,
) -> TrainingView:
    examples = tuple(
        TrainingExample(
            point=_strict_selected_point(row.point, feature_set),
            target_value=float(row.label),
            sample_weight=weights[row.point.point_id],
        )
        for row in sorted(rows, key=lambda item: item.point.point_id)
        if row.label is not None
    )
    if not examples:
        raise ValueError(f"initializer {partition_name} partition is empty")
    return TrainingView(
        dataset_id=dataset_id,
        position=PredictionPosition.TASK_PRE,
        target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        examples=examples,
        input_contract_hash=dataset_slice.input_contract_hash,
    )


def _initializer_training_scope_id(
    dataset_slice: DatasetSlice,
    *,
    tasks: frozenset[str],
    outer_assignment_id: str,
    outer_fold: int,
) -> str:
    rows = tuple(
        sorted(
            (row for row in dataset_slice.rows if row.point.task_id in tasks),
            key=lambda row: row.point.point_id,
        )
    )
    if not rows:
        raise ValueError("initializer training scope has no observed Task-pre rows")
    return _semantic_hash(
        {
            "scope_schema_version": 1,
            "dataset_schema_version": dataset_slice.dataset_schema_version,
            "capability_contract_hash": dataset_slice.capability_contract_hash,
            "input_contract_hash": dataset_slice.input_contract_hash,
            "outer_assignment_id": outer_assignment_id,
            "outer_fold": outer_fold,
            "rows": [
                {
                    "point": point_input_semantic(row.point),
                    "label": row.label,
                    "status": row.status.value,
                    "invalid_reason": row.invalid_reason,
                }
                for row in rows
            ],
        }
    )


def _make_updater_view(
    *,
    dataset_id: str,
    target: PredictionTarget,
    sequences: Sequence[LifecycleSequence],
    seeds: Mapping[str, SessionSeed],
    feature_set: FeatureSet,
    partition_name: str,
    input_contract_hash: str,
) -> TrainingView:
    if not sequences:
        raise ValueError(f"updater {partition_name} lifecycle partition is empty")
    selected_sequences = tuple(
        _selected_sequence(sequence, feature_set)
        for sequence in sorted(
            sequences,
            key=lambda item: (item.task_id, item.run_id, item.trajectory_id),
        )
    )
    expected_seed_ids = {sequence.steps[0].point.point_id for sequence in selected_sequences}
    if set(seeds) != expected_seed_ids:
        raise ValueError(f"updater {partition_name} seeds do not exactly cover lifecycle sequences")
    seeded_sequences = tuple(
        SeededLifecycleTrainingSequence(
            sequence,
            seeds[sequence.steps[0].point.point_id],
        )
        for sequence in selected_sequences
    )
    examples = tuple(
        TrainingExample(
            point=step.point,
            target_value=float(step.label),
            sample_weight=step.sample_weight,
        )
        for sequence in seeded_sequences
        for step in sequence.steps[1:]
        if step.loss_mask and step.label is not None
    )
    if not examples:
        raise ValueError(f"updater {partition_name} scored partition is empty")
    return TrainingView(
        dataset_id=dataset_id,
        position=PredictionPosition.TASK_UPDATE,
        target=target,
        examples=examples,
        lifecycle_sequences=seeded_sequences,
        input_contract_hash=input_contract_hash,
    )


def _artifact_payload_files(
    fitted: Any,
    *,
    fold: int,
    calibrator: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[FoldArtifact | None, dict[str, bytes]]:
    artifact = _collect_fold_artifact(
        fitted,
        fold=fold,
        calibrator=calibrator,
        provenance=provenance,
    )
    files: dict[str, bytes] = {}
    if artifact is not None:
        if artifact.bundle_files:
            files.update(
                {f"bundle/{name}": payload for name, payload in artifact.bundle_files.items()}
            )
        if artifact.encoder is not None:
            files["encoder.json"] = _canonical_json_bytes(dict(artifact.encoder))
        if artifact.fit_report is not None:
            files["fit-report.json"] = _canonical_json_bytes(dict(artifact.fit_report))
        if artifact.feature_importance is not None:
            files["feature-importance.json"] = _canonical_json_bytes(
                [dict(record) for record in artifact.feature_importance]
            )
        if artifact.model_strings is not None:
            files["model-strings.json"] = _canonical_json_bytes(dict(artifact.model_strings))
    if not files:
        converted = _json_compatible(fitted, context="fitted lifecycle component")
        files["fitted-state.json"] = _canonical_json_bytes(converted)
    return artifact, files


def _empirical_initializer_payload(
    fitted: Any,
) -> tuple[str, FoldArtifact | None, dict[str, bytes]]:
    if getattr(fitted, "estimator_id", None) != "empirical_quantile":
        raise TypeError("empirical initializer state has the wrong estimator_id")
    target = getattr(fitted, "target", None)
    values = tuple(getattr(fitted, name, None) for name in ("lower", "point", "upper"))
    if not isinstance(target, PredictionTarget) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise TypeError("empirical initializer state is not portable")
    lower, point, upper = (float(value) for value in values)
    if not 0 <= lower <= point <= upper:
        raise ValueError("empirical initializer quantiles are invalid")
    state = {
        "state_schema_version": 1,
        "estimator_id": "empirical_quantile",
        "target": target.value,
        "lower": lower,
        "point": point,
        "upper": upper,
    }
    return EMPIRICAL_INITIALIZER_FORMAT, None, {"state.json": _canonical_json_bytes(state)}


def _cross_position_updater_payload(
    fitted: Any,
) -> tuple[str, FoldArtifact | None, dict[str, bytes]]:
    expected = {
        "estimator_id": "cross_position_deduct",
        "dataset_id": getattr(fitted, "dataset_id", None),
        "target": getattr(fitted, "target", None),
        "condition_id": getattr(fitted, "condition_id", None),
        "input_contract_hash": getattr(fitted, "input_contract_hash", None),
    }
    if getattr(fitted, "estimator_id", None) != expected["estimator_id"]:
        raise TypeError("cross-position updater state has the wrong estimator_id")
    if not isinstance(expected["target"], PredictionTarget):
        raise TypeError("cross-position updater target is invalid")
    for name in ("dataset_id", "condition_id", "input_contract_hash"):
        value = expected[name]
        if not isinstance(value, str) or not value:
            raise TypeError(f"cross-position updater {name} is invalid")
    state = {
        "state_schema_version": 1,
        "estimator_id": expected["estimator_id"],
        "dataset_id": expected["dataset_id"],
        "target": expected["target"].value,
        "condition_id": expected["condition_id"],
        "input_contract_hash": expected["input_contract_hash"],
    }
    return CROSS_POSITION_DEDUCT_FORMAT, None, {"state.json": _canonical_json_bytes(state)}


def _initializer_payload_files(
    fitted: Any,
    *,
    estimator_id: str,
    fold: int,
    calibrator: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[str, FoldArtifact | None, dict[str, bytes]]:
    if estimator_id == "empirical_quantile":
        return _empirical_initializer_payload(fitted)
    artifact, files = _artifact_payload_files(
        fitted,
        fold=fold,
        calibrator=calibrator,
        provenance=provenance,
    )
    return OPAQUE_AUDIT_FORMAT, artifact, files


def _updater_payload_files(
    fitted: Any,
    *,
    estimator_id: str,
    fold: int,
    calibrator: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[str, FoldArtifact | None, dict[str, bytes]]:
    if estimator_id == "cross_position_deduct":
        return _cross_position_updater_payload(fitted)
    artifact, files = _artifact_payload_files(
        fitted,
        fold=fold,
        calibrator=calibrator,
        provenance=provenance,
    )
    if estimator_id == "gru_residual":
        standalone = {
            name.removeprefix("bundle/"): payload
            for name, payload in files.items()
            if name.startswith("bundle/")
        }
        try:
            manifest = json.loads(standalone["manifest.json"].decode("utf-8"))
            component_path = manifest["component"]["path"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TypeError("GRU updater bundle manifest is invalid") from exc
        if not isinstance(component_path, str) or not component_path.startswith("components/"):
            raise TypeError("GRU updater bundle component path is invalid")
        expected = {
            "manifest.json",
            "manifest.sha256",
            "calibrator.json",
            f"{component_path}/component.json",
            f"{component_path}/encoder.json",
            f"{component_path}/architecture.json",
            f"{component_path}/weights.safetensors",
        }
        if set(standalone) != expected:
            raise TypeError("GRU updater did not produce a reloadable neural bundle")
        embedded = {
            "gru/manifest.json": standalone["manifest.json"],
            "gru/manifest.sha256": standalone["manifest.sha256"],
            "gru/calibrator.json": standalone["calibrator.json"],
            "gru/component.json": standalone[f"{component_path}/component.json"],
            "gru/encoder.json": standalone[f"{component_path}/encoder.json"],
            "gru/architecture.json": standalone[f"{component_path}/architecture.json"],
            "gru/weights.safetensors": standalone[f"{component_path}/weights.safetensors"],
        }
        return GRU_BUNDLE_FORMAT, artifact, embedded
    return OPAQUE_AUDIT_FORMAT, artifact, files


@dataclass(frozen=True)
class _PackedInitializer:
    component: InitializerComponent
    files: Mapping[str, bytes]
    document: Mapping[str, Any]


def _pack_initializer(
    fitted: Any,
    *,
    initializer_id: str,
    initializer_hash: str,
    inner_split_id: str,
    inner_assignments: Sequence[tuple[str, int]],
    outer_fold: int,
    partition: InnerFoldPartition,
    feature_set: FeatureSet,
    split_plan_id: str,
    alpha: float,
) -> _PackedInitializer:
    inner_mapping = tuple(
        sorted(
            (
                _task_pseudonym(task, split_plan_id=split_plan_id),
                inner_fold,
            )
            for task, inner_fold in inner_assignments
        )
    )
    inner_assignment_hash = _semantic_hash(inner_mapping)
    pseudonymous_partitions = {
        "fit": _pseudonymous_tasks(
            partition.initializer_fit_tasks,
            split_plan_id=split_plan_id,
        ),
        "validation": _pseudonymous_tasks(
            partition.validation_tasks,
            split_plan_id=split_plan_id,
        ),
        "holdout": _pseudonymous_tasks(
            partition.holdout_tasks,
            split_plan_id=split_plan_id,
        ),
    }
    no_calibration = IdentityCalibrator(alpha=alpha).fit(()).to_dict()
    serialization_format, _artifact, model_files = _initializer_payload_files(
        fitted,
        estimator_id=initializer_id,
        fold=partition.holdout_fold,
        calibrator=no_calibration,
        provenance={
            "role": "inner_initializer",
            "initializer_id": initializer_id,
            "initializer_hash": initializer_hash,
            "inner_split_id": inner_split_id,
            "inner_task_assignments_sha256": inner_assignment_hash,
            "outer_fold": outer_fold,
            "inner_fold": partition.holdout_fold,
            "feature_set_hash": feature_set.content_hash,
            "task_partitions_sha256": pseudonymous_partitions,
            "calibration": "none",
        },
    )
    model_hashes = {name: _sha256_bytes(payload) for name, payload in sorted(model_files.items())}
    semantic: dict[str, Any] = {
        "component_schema_version": LIFECYCLE_COMPONENT_SCHEMA_VERSION,
        "role": "inner_initializer",
        "estimator_id": initializer_id,
        "serialization_format": serialization_format,
        "initializer_hash": initializer_hash,
        "inner_split_id": inner_split_id,
        "inner_task_assignments_sha256": inner_assignment_hash,
        "outer_fold": outer_fold,
        "inner_fold": partition.holdout_fold,
        "feature_set_hash": feature_set.content_hash,
        "task_partitions_sha256": pseudonymous_partitions,
        "model_files": model_hashes,
        "calibration": "none",
    }
    component_hash = _semantic_hash(semantic)
    document = {**semantic, "component_hash": component_hash}
    component_document = _canonical_json_bytes(document)
    component_files = {
        "component.json": component_document,
        **{f"model/{name}": payload for name, payload in model_files.items()},
    }
    bundle_hashes = tuple(sorted(_sha256_bytes(payload) for payload in component_files.values()))
    component = InitializerComponent(
        inner_fold=partition.holdout_fold,
        component_id=(f"{initializer_id}:outer-{outer_fold}:inner-{partition.holdout_fold}"),
        component_hash=component_hash,
        bundle_hashes=bundle_hashes,
        fit_tasks=partition.initializer_fit_tasks,
        validation_tasks=partition.validation_tasks,
        holdout_tasks=partition.holdout_tasks,
        fitted=_SelectedFitted(initializer_id, fitted, feature_set),
    )
    return _PackedInitializer(component, component_files, document)


def _partition_sequences(
    lifecycle_slice: LifecycleSlice,
    tasks: frozenset[str],
) -> tuple[LifecycleSequence, ...]:
    return tuple(sequence for sequence in lifecycle_slice.sequences if sequence.task_id in tasks)


def _scored_from_runs(runs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(prediction for run in runs for prediction in run.scored_predictions)


def _fit_calibrator(
    calibrator_id: str,
    *,
    alpha: float,
    examples: Sequence[CalibrationExample],
) -> Any:
    if calibrator_id == "task_max_conformal":
        return TaskMaxConformalCalibrator(alpha=alpha).fit(examples)
    if calibrator_id == "none":
        return IdentityCalibrator(alpha=alpha).fit(examples)
    raise ValueError(f"unknown calibrator {calibrator_id!r}")


def _pack_composite_artifact(
    fitted_updater: Any,
    *,
    fold: int,
    candidate: CandidateSpec,
    dataset_id: str,
    dataset_schema_version: int,
    split_plan: SplitPlan,
    eligibility_hash: str,
    lifecycle_slice: LifecycleSlice,
    alpha: float,
    calibrator_id: str,
    calibrator_document: Mapping[str, Any],
    initializer_hash: str,
    inner_split_id: str,
    inner_assignments: Sequence[tuple[str, int]],
    packed_initializers: Sequence[_PackedInitializer],
    seed_set: CrossfitSeedSet,
    outer_partitions: Mapping[str, frozenset[str]],
    source_provenance: Mapping[str, Any],
) -> FoldArtifact:
    pseudonymous_outer = {
        name: _pseudonymous_tasks(tasks, split_plan_id=split_plan.assignment_id)
        for name, tasks in sorted(outer_partitions.items())
    }
    updater_serialization, updater_artifact, updater_model_files = _updater_payload_files(
        fitted_updater,
        estimator_id=candidate.estimator_id,
        fold=fold,
        calibrator=calibrator_document,
        provenance={
            "role": "lifecycle_updater",
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.content_hash,
            "candidate_graph": candidate.graph.to_dict(),
            "dataset_id": dataset_id,
            "split_plan_id": split_plan.split_plan_id,
            "eligibility_hash": eligibility_hash,
            "lifecycle_context_hash": lifecycle_slice.context_hash,
            "lifecycle_scored_hash": lifecycle_slice.scored_hash,
            "outer_fold": fold,
            "outer_task_partitions_sha256": pseudonymous_outer,
            "initializer_hash": initializer_hash,
            "inner_split_id": inner_split_id,
            "seed_set_hash": seed_set.content_hash,
            "interval_alpha": alpha,
            "calibrator_id": calibrator_id,
        },
    )
    updater_model_hashes = {
        name: _sha256_bytes(payload) for name, payload in sorted(updater_model_files.items())
    }
    updater_semantic: dict[str, Any] = {
        "component_schema_version": LIFECYCLE_COMPONENT_SCHEMA_VERSION,
        "role": "lifecycle_updater",
        "estimator_id": candidate.estimator_id,
        "serialization_format": updater_serialization,
        "candidate_hash": candidate.content_hash,
        "outer_fold": fold,
        "model_files": updater_model_hashes,
    }
    updater_hash = _semantic_hash(updater_semantic)
    updater_document = _canonical_json_bytes({**updater_semantic, "component_hash": updater_hash})

    bundle: dict[str, bytes] = {
        "calibrator.json": _canonical_json_bytes(dict(calibrator_document)),
        f"components/{updater_hash}/component.json": updater_document,
    }
    bundle.update(
        {
            f"components/{updater_hash}/model/{name}": payload
            for name, payload in updater_model_files.items()
        }
    )
    initializer_summaries: list[dict[str, Any]] = []
    for packed in sorted(
        packed_initializers,
        key=lambda item: item.component.inner_fold,
    ):
        component_hash = packed.component.component_hash
        bundle.update(
            {
                f"components/{component_hash}/{name}": payload
                for name, payload in packed.files.items()
            }
        )
        initializer_summaries.append(
            {
                "inner_fold": packed.component.inner_fold,
                "component_hash": component_hash,
                "bundle_hashes": packed.component.bundle_hashes,
            }
        )

    inner_mapping = tuple(
        sorted(
            (
                _task_pseudonym(task, split_plan_id=split_plan.assignment_id),
                inner_fold,
            )
            for task, inner_fold in inner_assignments
        )
    )
    file_hashes = {name: _sha256_bytes(payload) for name, payload in sorted(bundle.items())}
    manifest = {
        "bundle_schema_version": LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION,
        "bundle_kind": "lifecycle_composite",
        "candidate_id": candidate.candidate_id,
        "candidate_hash": candidate.content_hash,
        "candidate_graph": candidate.graph.to_dict(),
        "dataset_id": dataset_id,
        "dataset_schema_version": dataset_schema_version,
        "source_descriptor": source_provenance["source_descriptor"],
        "source_descriptor_hash": source_provenance["source_descriptor_hash"],
        "capability_contract_hash": lifecycle_slice.capability_contract_hash,
        "code_hash": source_provenance["code_hash"],
        "runtime_versions": source_provenance["runtime_versions"],
        "split_plan_id": split_plan.split_plan_id,
        "eligibility_hash": eligibility_hash,
        "position": PredictionPosition.TASK_UPDATE.value,
        "target": lifecycle_slice.target.value,
        "condition_id": lifecycle_slice.condition_id,
        "outer_fold": fold,
        "outer_task_partitions_sha256": pseudonymous_outer,
        "feature_set": dict(feature_set_document(candidate.feature_set)),
        "feature_set_hash": candidate.feature_set.content_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "protocol_features": sorted(_PROTOCOL_FEATURES),
        "lifecycle_schema_id": candidate.graph.lifecycle_schema_id,
        "lifecycle_schema_version": lifecycle_slice.schema_version,
        "lifecycle_weighting_id": lifecycle_slice.weighting_id,
        "lifecycle_context_hash": lifecycle_slice.context_hash,
        "lifecycle_scored_hash": lifecycle_slice.scored_hash,
        "input_contract_hash": lifecycle_slice.input_contract_hash,
        "initializer_hash": initializer_hash,
        "initializer_components": initializer_summaries,
        "updater_component_hash": updater_hash,
        "inner_split_policy_id": candidate.graph.inner_split_policy_id,
        "inner_split_id": inner_split_id,
        "inner_task_assignments_sha256": _semantic_hash(inner_mapping),
        "inner_task_assignments": [
            {"task_pseudonym": task, "fold": inner_fold} for task, inner_fold in inner_mapping
        ],
        "seed_policy_id": seed_set.seed_policy_id,
        "seed_policy_hash": seed_set.seed_policy_hash,
        "seed_set_hash": seed_set.content_hash,
        "calibrator_id": calibrator_id,
        "interval_alpha": alpha,
        "files": file_hashes,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    bundle["manifest.json"] = manifest_bytes
    bundle["manifest.sha256"] = f"{_sha256_bytes(manifest_bytes)}\n".encode("ascii")
    return FoldArtifact(
        fold=fold,
        encoder=(None if updater_artifact is None else updater_artifact.encoder),
        fit_report=(None if updater_artifact is None else updater_artifact.fit_report),
        feature_importance=(
            None if updater_artifact is None else updater_artifact.feature_importance
        ),
        model_strings=(None if updater_artifact is None else updater_artifact.model_strings),
        bundle_files=bundle,
        calibrator=calibrator_document,
        provenance=manifest,
    )


def _validate_inputs(
    dataset: SupervisedDataset,
    lifecycle_slice: LifecycleSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    *,
    alpha: float,
    seed: int,
) -> tuple[DatasetSlice, dict[str, float]]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if split_plan.dataset_id != dataset.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    if split_plan.folds != 5:
        raise ValueError("lifecycle CV requires exactly five outer folds")
    if split_plan.seed != seed:
        raise ValueError("lifecycle model seed must match the outer split seed")
    if lifecycle_slice.dataset_id != dataset.dataset_id:
        raise ValueError("lifecycle slice belongs to another dataset")
    if (
        dataset.source_descriptor_hash != lifecycle_slice.source_descriptor_hash
        or dataset.capability_contract_hash != lifecycle_slice.capability_contract_hash
        or dataset.input_contract_hash != lifecycle_slice.input_contract_hash
    ):
        raise ValueError("lifecycle slice provenance differs from its dataset")
    allowed_input_contract_hashes = supported_input_contract_hashes_from_capability(
        str(lifecycle_slice.capability_contract_hash)
    )
    if lifecycle_slice.input_contract_hash not in allowed_input_contract_hashes:
        raise ValueError("lifecycle input contract differs from its capability contract")
    if lifecycle_slice.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
        raise ValueError("lifecycle CV requires the provider-accounted Task target")
    if candidate.graph is None or not candidate.graph.is_lifecycle:
        raise ValueError("lifecycle CV requires an initializer/updater candidate graph")
    if candidate.graph.lifecycle_schema_id != TASK_LIFECYCLE_SCHEMA_ID:
        raise ValueError("candidate lifecycle schema does not match the runtime schema")
    if candidate.graph.seed_policy_id not in SUPPORTED_SEED_POLICY_IDS:
        raise ValueError("candidate seed policy is not supported by the crossfit runtime")
    if candidate.graph.inner_split_policy_id != INNER_FOLD_POLICY_ID:
        raise ValueError("candidate inner split policy does not match the runtime")
    allowed_updaters = frozenset({"cross_position_deduct", "gru_residual"})
    if (
        candidate.graph.initializer_estimator_id != "empirical_quantile"
        or candidate.graph.updater_estimator_id not in allowed_updaters
        or candidate.estimator_id != candidate.graph.updater_estimator_id
    ):
        raise ValueError(
            "lifecycle CV requires a reloadable empirical_quantile initializer "
            "and an explicitly supported updater DAG"
        )
    if any(
        sequence.steps[0].loss_mask
        or sequence.steps[0].score_mask
        or sequence.steps[0].sample_weight != 0
        for sequence in lifecycle_slice.sequences
    ):
        raise ValueError("Task-pre must remain unscored lifecycle context")
    split_plan.validate_tasks(lifecycle_slice.task_ids, require_exact=False)

    update_slice = dataset.select(
        PredictionPosition.TASK_UPDATE,
        lifecycle_slice.target,
        condition_id=lifecycle_slice.condition_id,
    )
    if not update_slice.rows:
        raise ValueError("lifecycle experiment has no eligible Task-update points")
    expected = {row.point.point_id for row in update_slice.rows}
    scored = {step.point.point_id for step in lifecycle_slice.scored_steps}
    if scored != expected:
        raise ValueError("lifecycle scored cohort differs from Task-update eligibility")
    weights = {
        weighted.row.point.point_id: weighted.sample_weight
        for weighted in update_slice.weighted_rows()
    }
    for step in lifecycle_slice.scored_steps:
        if not math.isclose(
            step.sample_weight,
            weights[step.point.point_id],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("lifecycle weights differ from the eligible point cohort")
    return update_slice, weights


def _validated_inner_plans(
    split_plan: SplitPlan,
    inner_plans: Mapping[int, OuterInnerPlan],
) -> dict[int, OuterInnerPlan]:
    if not isinstance(inner_plans, Mapping):
        raise TypeError("inner_plans must map every outer fold to an OuterInnerPlan")
    expected_folds = set(range(split_plan.folds))
    if set(inner_plans) != expected_folds:
        raise ValueError("inner_plans must exactly cover every outer fold")
    validated: dict[int, OuterInnerPlan] = {}
    for outer_fold in range(split_plan.folds):
        plan = inner_plans[outer_fold]
        if not isinstance(plan, OuterInnerPlan):
            raise TypeError("inner_plans values must be OuterInnerPlan instances")
        if plan.outer_test_fold != outer_fold:
            raise ValueError("inner plan is indexed by the wrong outer fold")
        if plan.split_seed != split_plan.seed:
            raise ValueError("inner plan seed differs from the outer split seed")
        if plan.outer_split_plan_id != split_plan.split_plan_id:
            raise ValueError("inner plan belongs to another outer split plan")
        expected_tasks = split_plan.partition(outer_fold).train_tasks
        if plan.assignment.task_ids != expected_tasks:
            raise ValueError("inner plan task universe must equal the full outer-train partition")
        validated[outer_fold] = plan
    return validated


def _condition_inner_partition(
    plan: OuterInnerPlan,
    *,
    inner_fold: int,
    condition_train_tasks: frozenset[str],
) -> InnerFoldPartition:
    """Project a frozen assignment without changing any surviving task's fold."""

    frozen = plan.assignment.partition(inner_fold)
    return InnerFoldPartition(
        holdout_fold=inner_fold,
        initializer_fit_tasks=(frozen.initializer_fit_tasks & condition_train_tasks),
        validation_tasks=frozen.validation_tasks & condition_train_tasks,
        holdout_tasks=frozen.holdout_tasks & condition_train_tasks,
    )


def run_lifecycle_candidate_cv(
    dataset: SupervisedDataset,
    lifecycle_slice: LifecycleSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    registry: EstimatorRegistry,
    *,
    alpha: float,
    calibrator_id: str,
    seed: int,
    inner_plans: Mapping[int, OuterInnerPlan],
    source_provenance: Mapping[str, Any],
    fit_checkpoint_factory: Callable[[int], FitCheckpoint | None] | None = None,
) -> CandidateResult:
    """Run leakage-free outer CV for a Task-pre initializer -> Task-update updater DAG."""

    if calibrator_id not in {"task_max_conformal", "none"}:
        raise ValueError(f"unknown calibrator {calibrator_id!r}")
    update_slice, expected_weights = _validate_inputs(
        dataset,
        lifecycle_slice,
        split_plan,
        candidate,
        alpha=alpha,
        seed=seed,
    )
    validated_inner_plans = _validated_inner_plans(split_plan, inner_plans)
    validated_source_provenance = validate_source_provenance(
        source_provenance,
        source_descriptor_hash=lifecycle_slice.source_descriptor_hash,
        capability_contract_hash=lifecycle_slice.capability_contract_hash,
        require_lifecycle_capabilities=True,
    )
    pre_slice = dataset.select(
        PredictionPosition.TASK_PRE,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        condition_id=lifecycle_slice.condition_id,
    )
    pre_weights = {
        weighted.row.point.point_id: weighted.sample_weight
        for weighted in pre_slice.weighted_rows()
    }
    graph = candidate.graph
    if graph is None:  # CandidateSpec normalizes this; retain a static type guard.
        raise AssertionError("candidate graph was not normalized")

    all_records: list[PredictionRecord] = []
    all_scored: list[ScoredForecast] = []
    fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
    fold_artifacts: list[FoldArtifact] = []

    for outer_fold in range(split_plan.folds):
        partition = split_plan.partition(outer_fold)
        actual = lifecycle_slice.task_ids
        outer_tasks = {
            "train": actual & partition.train_tasks,
            "validation": actual & partition.validation_tasks,
            "calibration": actual & partition.calibration_tasks,
            "test": actual & partition.test_tasks,
        }
        if any(not tasks for tasks in outer_tasks.values()):
            raise ValueError(f"outer fold {outer_fold} has an empty lifecycle partition")
        inner_plan = validated_inner_plans[outer_fold]
        inner_assignment = inner_plan.assignment
        initializer_scope_id = _initializer_training_scope_id(
            pre_slice,
            tasks=outer_tasks["train"],
            outer_assignment_id=split_plan.assignment_id,
            outer_fold=outer_fold,
        )
        initializer_hash = _semantic_hash(
            {
                "initializer_estimator_id": graph.initializer_estimator_id,
                "initializer_params": candidate.initializer_params,
                "feature_set_hash": candidate.feature_set.content_hash,
                "initializer_training_scope_id": initializer_scope_id,
                "outer_assignment_id": split_plan.assignment_id,
                "outer_fold": outer_fold,
                "inner_split_id": inner_assignment.assignment_id,
                "inner_split_policy_id": graph.inner_split_policy_id,
                "seed_policy_id": graph.seed_policy_id,
                "model_seed": seed,
                "interval_alpha": alpha,
                "calibration": "none",
            }
        )

        packed_initializers: list[_PackedInitializer] = []
        for inner_fold in range(inner_assignment.folds):
            inner_partition = _condition_inner_partition(
                inner_plan,
                inner_fold=inner_fold,
                condition_train_tasks=outer_tasks["train"],
            )
            fit_rows = tuple(
                row
                for row in pre_slice.rows
                if row.point.task_id in inner_partition.initializer_fit_tasks
            )
            validation_rows = tuple(
                row
                for row in pre_slice.rows
                if row.point.task_id in inner_partition.validation_tasks
            )
            initializer = registry.create(
                graph.initializer_estimator_id,
                candidate.initializer_params,
            )
            fitted_initializer = initializer.fit(
                _make_initializer_view(
                    pre_slice,
                    fit_rows,
                    pre_weights,
                    candidate.feature_set,
                    dataset_id=initializer_scope_id,
                    partition_name=(f"outer-{outer_fold}/inner-{inner_fold}/fit"),
                ),
                _make_initializer_view(
                    pre_slice,
                    validation_rows,
                    pre_weights,
                    candidate.feature_set,
                    dataset_id=initializer_scope_id,
                    partition_name=(f"outer-{outer_fold}/inner-{inner_fold}/validation"),
                ),
                FitContext(
                    seed=_derived_fit_seed(
                        seed,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                    ),
                    fold=inner_fold,
                    interval_alpha=alpha,
                ),
            )
            packed_initializers.append(
                _pack_initializer(
                    fitted_initializer,
                    initializer_id=graph.initializer_estimator_id,
                    initializer_hash=initializer_hash,
                    inner_split_id=inner_assignment.assignment_id,
                    inner_assignments=inner_assignment.assignments,
                    outer_fold=outer_fold,
                    partition=inner_partition,
                    feature_set=candidate.feature_set,
                    split_plan_id=split_plan.assignment_id,
                    alpha=alpha,
                )
            )

        components = tuple(packed.component for packed in packed_initializers)
        external_tasks = frozenset(
            task for role in ("validation", "calibration", "test") for task in outer_tasks[role]
        )
        task_pre_points = tuple(
            _lifecycle_selected_point(sequence.steps[0].point, candidate.feature_set)
            for sequence in lifecycle_slice.sequences
            if sequence.task_id in outer_tasks["train"] | external_tasks
        )
        seed_set = generate_crossfit_seeds(
            task_pre_points,
            components,
            dataset_id=initializer_scope_id,
            input_contract_hash=lifecycle_slice.input_contract_hash,
            initializer_id=graph.initializer_estimator_id,
            initializer_hash=initializer_hash,
            inner_split_id=inner_assignment.assignment_id,
            oof_tasks=outer_tasks["train"],
            external_tasks=external_tasks,
            seed_policy_id=graph.seed_policy_id,
        )

        train_sequences = _partition_sequences(
            lifecycle_slice,
            outer_tasks["train"],
        )
        validation_sequences = _partition_sequences(
            lifecycle_slice,
            outer_tasks["validation"],
        )
        train_seed_ids = {sequence.steps[0].point.point_id for sequence in train_sequences}
        validation_seed_ids = {
            sequence.steps[0].point.point_id for sequence in validation_sequences
        }
        updater = registry.create(candidate.estimator_id, candidate.params)
        fitted_updater = updater.fit(
            _make_updater_view(
                dataset_id=dataset.dataset_id,
                target=lifecycle_slice.target,
                sequences=train_sequences,
                seeds={point_id: seed_set.by_point_id[point_id] for point_id in train_seed_ids},
                feature_set=candidate.feature_set,
                partition_name=f"outer-{outer_fold}/train",
                input_contract_hash=lifecycle_slice.input_contract_hash,
            ),
            _make_updater_view(
                dataset_id=dataset.dataset_id,
                target=lifecycle_slice.target,
                sequences=validation_sequences,
                seeds={
                    point_id: seed_set.by_point_id[point_id] for point_id in validation_seed_ids
                },
                feature_set=candidate.feature_set,
                partition_name=f"outer-{outer_fold}/validation",
                input_contract_hash=lifecycle_slice.input_contract_hash,
            ),
            FitContext(
                seed=_derived_fit_seed(
                    seed,
                    outer_fold=outer_fold,
                    inner_fold=None,
                ),
                fold=outer_fold,
                interval_alpha=alpha,
                checkpoint=(
                    fit_checkpoint_factory(outer_fold)
                    if fit_checkpoint_factory is not None
                    else None
                ),
            ),
        )

        calibration_sequences = _partition_sequences(
            lifecycle_slice,
            outer_tasks["calibration"],
        )
        calibration_seed_ids = {
            sequence.steps[0].point.point_id for sequence in calibration_sequences
        }
        calibration_runs = run_lifecycle_batch(
            fitted_updater,
            calibration_sequences,
            {point_id: seed_set.by_point_id[point_id] for point_id in calibration_seed_ids},
            select_point=lambda point: _lifecycle_selected_point(
                point,
                candidate.feature_set,
            ),
        )
        calibration_predictions = _scored_from_runs(calibration_runs)
        if not calibration_predictions:
            raise ValueError(f"outer fold {outer_fold} calibration score set is empty")
        calibration_examples = tuple(
            CalibrationExample(
                task_id=prediction.step.point.task_id,
                forecast=prediction.forecast,
                target_value=float(prediction.step.label),
            )
            for prediction in calibration_predictions
            if prediction.step.label is not None
        )
        calibrator = _fit_calibrator(
            calibrator_id,
            alpha=alpha,
            examples=calibration_examples,
        )
        calibrator_document = calibrator.to_dict()

        fold_artifact = _pack_composite_artifact(
            fitted_updater,
            fold=outer_fold,
            candidate=candidate,
            dataset_id=dataset.dataset_id,
            dataset_schema_version=dataset.schema_version,
            split_plan=split_plan,
            eligibility_hash=update_slice.eligibility_hash,
            lifecycle_slice=lifecycle_slice,
            alpha=alpha,
            calibrator_id=calibrator_id,
            calibrator_document=calibrator_document,
            initializer_hash=initializer_hash,
            inner_split_id=inner_assignment.assignment_id,
            inner_assignments=inner_assignment.assignments,
            packed_initializers=packed_initializers,
            seed_set=seed_set,
            outer_partitions=outer_tasks,
            source_provenance=validated_source_provenance,
        )

        test_sequences = _partition_sequences(lifecycle_slice, outer_tasks["test"])
        test_seed_ids = {sequence.steps[0].point.point_id for sequence in test_sequences}
        test_runs = run_lifecycle_batch(
            fitted_updater,
            test_sequences,
            {point_id: seed_set.by_point_id[point_id] for point_id in test_seed_ids},
            select_point=lambda point: _lifecycle_selected_point(
                point,
                candidate.feature_set,
            ),
        )
        test_predictions = _scored_from_runs(test_runs)
        if not test_predictions:
            raise ValueError(f"outer fold {outer_fold} test score set is empty")
        reloaded = load_lifecycle_bundle(
            dict(fold_artifact.bundle_files or {}),
            expected_source_provenance=validated_source_provenance,
        )
        replayed_runs = reloaded.run_calibrated(test_sequences)
        if len(replayed_runs) != len(test_runs):
            raise ValueError("lifecycle bundle reload changed the test trajectory count")
        for original_run, replayed_run in zip(test_runs, replayed_runs):
            expected_forecasts = tuple(
                calibrator.transform(item.forecast) for item in original_run.predictions
            )
            actual_forecasts = tuple(item.forecast for item in replayed_run.predictions)
            for expected, actual in zip(
                expected_forecasts,
                actual_forecasts,
                strict=True,
            ):
                if replace(expected, latency_ms=actual.latency_ms) != actual:
                    raise ValueError("lifecycle bundle reload changed a calibrated test trajectory")
        fold_artifacts.append(fold_artifact)
        fold_scored: list[ScoredForecast] = []
        for prediction in test_predictions:
            step = prediction.step
            if step.label is None:
                raise AssertionError("scored lifecycle prediction has no label")
            point_id = step.point.point_id
            forecast = calibrator.transform(prediction.forecast)
            record = PredictionRecord(
                candidate_id=candidate.candidate_id,
                point_id=point_id,
                task_id=step.point.task_id,
                trajectory_id=step.point.trajectory_id,
                condition_id=step.point.condition_id,
                fold=outer_fold,
                target=step.point.target,
                forecast=forecast,
                sample_weight=expected_weights[point_id],
            )
            scored = ScoredForecast(
                task_id=step.point.task_id,
                trajectory_id=step.point.trajectory_id,
                forecast=forecast,
                target_value=float(step.label),
                sample_weight=expected_weights[point_id],
            )
            all_records.append(record)
            fold_scored.append(scored)
            all_scored.append(scored)
        fold_metrics[outer_fold] = evaluate_forecasts(fold_scored, alpha=alpha)

    expected_point_ids = {row.point.point_id for row in update_slice.rows}
    actual_point_ids = [record.point_id for record in all_records]
    if (
        len(actual_point_ids) != len(set(actual_point_ids))
        or set(actual_point_ids) != expected_point_ids
    ):
        raise ValueError(
            "lifecycle cross-validation must predict every eligible update exactly once"
        )
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        dataset_id=dataset.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        eligibility_hash=update_slice.eligibility_hash,
        position=PredictionPosition.TASK_UPDATE,
        target=lifecycle_slice.target,
        condition_id=lifecycle_slice.condition_id,
        calibrator_id=calibrator_id,
        alpha=alpha,
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(sorted(all_records, key=lambda record: record.point_id)),
        metrics=evaluate_forecasts(all_scored, alpha=alpha),
        fold_metrics=fold_metrics,
        task_metrics={
            task_id: task_metric.to_dict()
            for task_id, task_metric in evaluate_task_forecasts(
                all_scored,
                alpha=alpha,
            ).items()
        },
        fold_artifacts=tuple(fold_artifacts),
    )


__all__ = [
    "LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION",
    "SeededLifecycleTrainingSequence",
    "TASK_LIFECYCLE_SCHEMA_ID",
    "run_lifecycle_candidate_cv",
]
