from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_prediction.contracts import (
    SourceDescriptor,
    canonical_input_path,
    resolve_canonical_input_file,
)
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.experiment import (
    AblationAxis,
    AblationSpec,
    CandidateRole,
    CandidateSpec,
    ExperimentSpec,
)
from token_prediction.features import FeatureGroup, FeatureSet


def _reject_unknown(payload: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {', '.join(unknown)}")


def _sha256_digest(value: object, *, context: str) -> str:
    digest = str(value or "").strip()
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in source descriptor: {key!r}")
        result[key] = value
    return result


def _load_source_descriptor(
    project_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> SourceDescriptor:
    path = resolve_canonical_input_file(
        project_root,
        relative_path,
        context="source descriptor file",
    )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source descriptor file SHA-256 does not match")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in source descriptor: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source descriptor JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("source descriptor file must contain one JSON object")
    return SourceDescriptor.from_dict(value)


@dataclass(frozen=True)
class ConfiguredExperiment:
    experiment_id: str
    position: PredictionPosition
    target: PredictionTarget
    candidate_ids: tuple[str, ...]
    required_features: frozenset[str]
    condition_id: str | None


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    source_hash: str
    schema_version: int
    workspace: Path
    seed: int
    collection_source: str
    source_descriptor: SourceDescriptor | None
    source_descriptor_path: str | None
    source_descriptor_file_sha256: str | None
    canonical_manifest_path: str | None
    canonical_manifest_sha256: str | None
    split_unit: str
    folds: int
    alpha: float
    calibrator_id: str
    feature_sets: dict[str, FeatureSet]
    candidates: tuple[CandidateSpec, ...]
    experiments: tuple[ConfiguredExperiment, ...]

    def experiment_specs(self) -> tuple[ExperimentSpec, ...]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return tuple(
            ExperimentSpec(
                experiment_id=experiment.experiment_id,
                position=experiment.position,
                target=experiment.target,
                candidates=tuple(by_id[candidate_id] for candidate_id in experiment.candidate_ids),
                alpha=self.alpha,
                calibrator_id=self.calibrator_id,
                required_features=experiment.required_features,
                condition_id=experiment.condition_id,
            )
            for experiment in self.experiments
        )


def _strings(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return frozenset(str(item).strip() for item in value if str(item).strip())


def _feature_set(feature_set_id: str, value: Any) -> FeatureSet:
    payload = dict(value or {})
    _reject_unknown(
        payload,
        {
            "include_all",
            "include_groups",
            "exclude_groups",
            "include_subgroups",
            "exclude_subgroups",
            "include_features",
            "exclude_features",
        },
        f"feature_sets.{feature_set_id}",
    )
    return FeatureSet(
        feature_set_id=feature_set_id,
        include_all=bool(payload.get("include_all", True)),
        include_groups=frozenset(
            FeatureGroup(item) for item in _strings(payload.get("include_groups"))
        ),
        exclude_groups=frozenset(
            FeatureGroup(item) for item in _strings(payload.get("exclude_groups"))
        ),
        include_subgroups=_strings(payload.get("include_subgroups")),
        exclude_subgroups=_strings(payload.get("exclude_subgroups")),
        include_features=_strings(payload.get("include_features")),
        exclude_features=_strings(payload.get("exclude_features")),
    )


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    _reject_unknown(
        payload,
        {
            "schema_version",
            "workspace",
            "seed",
            "collection",
            "split",
            "interval",
            "feature_sets",
            "candidates",
            "experiments",
        },
        "top-level",
    )
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("config schema_version must be an integer")
    if schema_version not in {1, 2}:
        raise ValueError("config schema_version must be 1 or 2")
    collection = dict(payload.get("collection") or {})
    split = dict(payload.get("split") or {})
    interval = dict(payload.get("interval") or {})
    _reject_unknown(
        collection,
        {
            "source",
            "descriptor_path",
            "descriptor_sha256",
            "canonical_manifest_path",
            "canonical_manifest_sha256",
        },
        "collection",
    )
    _reject_unknown(split, {"unit", "folds"}, "split")
    _reject_unknown(interval, {"alpha", "calibrator"}, "interval")
    collection_source = str(collection.get("source") or "").strip()
    if not collection_source:
        raise ValueError("collection.source is required")
    descriptor_fields = {"descriptor_path", "descriptor_sha256"}
    canonical_manifest_fields = {
        "canonical_manifest_path",
        "canonical_manifest_sha256",
    }
    supplied_descriptor_fields = descriptor_fields & set(collection)
    supplied_canonical_manifest_fields = canonical_manifest_fields & set(collection)
    if schema_version == 1:
        if supplied_descriptor_fields or supplied_canonical_manifest_fields:
            raise ValueError(
                "config schema_version=1 cannot declare v2 source manifests"
            )
        source_descriptor = None
        source_descriptor_path = None
        source_descriptor_file_sha256 = None
        canonical_manifest_path = None
        canonical_manifest_sha256 = None
    else:
        missing_descriptor_fields = sorted(descriptor_fields - supplied_descriptor_fields)
        if missing_descriptor_fields:
            raise ValueError(
                "config schema_version=2 collection is missing: "
                + ", ".join(missing_descriptor_fields)
            )
        missing_canonical_manifest_fields = sorted(
            canonical_manifest_fields - supplied_canonical_manifest_fields
        )
        if missing_canonical_manifest_fields:
            raise ValueError(
                "config schema_version=2 collection is missing: "
                + ", ".join(missing_canonical_manifest_fields)
            )
        project_root = source.parent.parent
        source_descriptor_path = canonical_input_path(
            collection.get("descriptor_path"),
            context="collection.descriptor_path",
        )
        source_descriptor_file_sha256 = _sha256_digest(
            collection.get("descriptor_sha256"),
            context="collection.descriptor_sha256",
        )
        source_descriptor = _load_source_descriptor(
            project_root,
            source_descriptor_path,
            source_descriptor_file_sha256,
        )
        canonical_manifest_path = canonical_input_path(
            collection.get("canonical_manifest_path"),
            context="collection.canonical_manifest_path",
        )
        canonical_manifest_sha256 = _sha256_digest(
            collection.get("canonical_manifest_sha256"),
            context="collection.canonical_manifest_sha256",
        )
    workspace_value = Path(str(payload.get("workspace") or "workspace"))
    workspace = (
        workspace_value
        if workspace_value.is_absolute()
        else source.parent.parent / workspace_value
    )
    folds = int(split.get("folds") or 0)
    if folds < 4:
        raise ValueError("split.folds must be at least 4")
    split_unit = str(split.get("unit") or "").strip()
    if split_unit != "task_id":
        raise ValueError("the MVP only permits task_id grouped splits")
    alpha = float(interval.get("alpha") or 0.10)
    if not 0 < alpha < 1:
        raise ValueError("interval.alpha must be in (0, 1)")
    calibrator_id = str(interval.get("calibrator") or "").strip()
    if calibrator_id not in {"task_max_conformal", "none"}:
        raise ValueError("unsupported interval calibrator")

    feature_payload = payload.get("feature_sets")
    if not isinstance(feature_payload, dict) or not feature_payload:
        raise ValueError("at least one feature_sets table is required")
    feature_sets = {
        str(name): _feature_set(str(name), value)
        for name, value in feature_payload.items()
    }

    candidate_payloads = payload.get("candidates")
    if not isinstance(candidate_payloads, list) or not candidate_payloads:
        raise ValueError("at least one [[candidates]] entry is required")
    candidates: list[CandidateSpec] = []
    for value in candidate_payloads:
        item = dict(value or {})
        _reject_unknown(
            item,
            {"id", "estimator", "feature_set", "role", "params", "ablation"},
            "candidate",
        )
        candidate_id = str(item.get("id") or "").strip()
        feature_set_id = str(item.get("feature_set") or "").strip()
        if feature_set_id not in feature_sets:
            raise ValueError(
                f"candidate {candidate_id!r} references unknown feature set {feature_set_id!r}"
            )
        role = CandidateRole(str(item.get("role") or CandidateRole.MODEL.value))
        ablation_payload = item.get("ablation")
        ablation = None
        if ablation_payload is not None:
            ablation_item = dict(ablation_payload)
            _reject_unknown(
                ablation_item,
                {"reference_candidate_id", "axis", "allowed_config_paths"},
                f"candidate {candidate_id!r} ablation",
            )
            ablation = AblationSpec(
                reference_candidate_id=str(
                    ablation_item.get("reference_candidate_id") or ""
                ).strip(),
                axis=AblationAxis(str(ablation_item.get("axis") or "")),
                allowed_config_paths=_strings(
                    ablation_item.get("allowed_config_paths")
                ),
            )
        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                estimator_id=str(item.get("estimator") or "").strip(),
                feature_set=feature_sets[feature_set_id],
                params=dict(item.get("params") or {}),
                role=role,
                ablation=ablation,
            )
        )
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate ids must be unique")

    experiment_payloads = payload.get("experiments")
    if not isinstance(experiment_payloads, list) or not experiment_payloads:
        raise ValueError("at least one [[experiments]] entry is required")
    experiments: list[ConfiguredExperiment] = []
    for value in experiment_payloads:
        item = dict(value or {})
        _reject_unknown(
            item,
            {
                "id",
                "position",
                "target",
                "candidates",
                "required_features",
                "condition_id",
            },
            "experiment",
        )
        ids = tuple(str(entry) for entry in item.get("candidates") or ())
        unknown = set(ids) - candidate_ids
        if unknown:
            raise ValueError(f"experiment references unknown candidates: {sorted(unknown)}")
        experiments.append(
            ConfiguredExperiment(
                experiment_id=str(item.get("id") or "").strip(),
                position=PredictionPosition(str(item.get("position") or "")),
                target=PredictionTarget(str(item.get("target") or "")),
                candidate_ids=ids,
                required_features=_strings(item.get("required_features")),
                condition_id=(
                    str(item.get("condition_id")).strip()
                    if item.get("condition_id") is not None
                    else None
                ),
            )
        )
    if len({experiment.experiment_id for experiment in experiments}) != len(experiments):
        raise ValueError("experiment ids must be unique")
    return ProjectConfig(
        source_path=source,
        source_hash=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        workspace=workspace.resolve(),
        seed=int(payload.get("seed") or 0),
        collection_source=collection_source,
        source_descriptor=source_descriptor,
        source_descriptor_path=source_descriptor_path,
        source_descriptor_file_sha256=source_descriptor_file_sha256,
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest_sha256=canonical_manifest_sha256,
        split_unit=split_unit,
        folds=folds,
        alpha=alpha,
        calibrator_id=calibrator_id,
        feature_sets=feature_sets,
        candidates=tuple(candidates),
        experiments=tuple(experiments),
    )
