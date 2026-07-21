from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.config import ProjectConfig
from token_prediction.contracts import (
    CanonicalEvent,
    SourceDescriptor,
    canonical_input_path,
    resolve_canonical_input_file,
)
from token_prediction.dataset import (
    SupervisedDataset,
    build_capability_supervised_dataset,
    build_prediction_labels,
)
from token_prediction.development import (
    OUTER_FOLDS,
    STAGE_SPLIT_SEEDS,
    DevelopmentProtocol,
    build_development_protocol,
)
from token_prediction.estimators import EstimatorRegistry, builtin_registry
from token_prediction.experiment import (
    CandidateResult,
    ExperimentRunner,
    ExperimentSpec,
    FoldArtifact,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, replay_feature_snapshots
from token_prediction.lineage import publish_artifact, verify_artifact
from token_prediction.lifecycle_bundle import validate_source_provenance
from token_prediction.trajectory import Trajectory


@dataclass(frozen=True)
class ExperimentRunSummary:
    run_id: str
    output_dir: Path
    dataset_id: str
    split_plan_id: str
    artifact_id: str
    experiment_count: int
    candidate_run_count: int


@dataclass(frozen=True)
class SeedDevelopmentResults:
    split_seed: int
    split_plan_id: str
    result_groups: tuple[tuple[CandidateResult, ...], ...]

    def __post_init__(self) -> None:
        if self.split_seed not in STAGE_SPLIT_SEEDS:
            raise ValueError("development result uses an unapproved split seed")
        if not self.split_plan_id:
            raise ValueError("development result must bind a split plan")


@dataclass(frozen=True)
class DevelopmentExperimentResults:
    protocol: DevelopmentProtocol
    seed_results: tuple[SeedDevelopmentResults, ...]

    def __post_init__(self) -> None:
        if tuple(result.split_seed for result in self.seed_results) != STAGE_SPLIT_SEEDS:
            raise ValueError("development execution requires all three frozen split seeds")
        expected = tuple(plan.split_plan_id for plan in self.protocol.outer_plans)
        if tuple(result.split_plan_id for result in self.seed_results) != expected:
            raise ValueError("development results differ from the protocol split plans")

    @property
    def audit_document(self) -> dict[str, object]:
        return self.protocol.to_audit_document()


_SAFE_ARTIFACT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _safe_artifact_component(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not _SAFE_ARTIFACT_COMPONENT.fullmatch(value)
    ):
        raise ValueError(f"unsafe {label} for artifact path: {value!r}")
    return value


def _validate_artifact_identifiers(specs: Sequence[ExperimentSpec]) -> None:
    for spec in specs:
        _safe_artifact_component(spec.experiment_id, label="experiment_id")
        for candidate in spec.candidates:
            _safe_artifact_component(candidate.candidate_id, label="candidate_id")


def _effective_experiment_semantics(
    specs: Sequence[ExperimentSpec],
) -> list[dict[str, object]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "position": spec.position.value,
            "target": spec.target.value,
            "condition_id": spec.condition_id,
            "alpha": spec.alpha,
            "calibrator_id": spec.calibrator_id,
            "required_features": sorted(spec.required_features),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "content_hash": candidate.content_hash,
                    "role": candidate.role.value,
                    "ablation": (
                        {
                            "reference_candidate_id": (candidate.ablation.reference_candidate_id),
                            "axis": candidate.ablation.axis.value,
                            "allowed_config_paths": sorted(candidate.ablation.allowed_config_paths),
                        }
                        if candidate.ablation is not None
                        else None
                    ),
                }
                for candidate in spec.candidates
            ],
        }
        for spec in specs
    ]


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _module_version(distribution: str, module_name: str) -> str:
    """Return the version of the module that will actually execute."""

    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return "not-installed"
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value.strip():
        return _installed_version(distribution)
    return value.strip()


def _lightgbm_runtime_versions(specs: Sequence[ExperimentSpec]) -> dict[str, str]:
    if not any(
        candidate.estimator_id == "lightgbm_quantile"
        for spec in specs
        for candidate in spec.candidates
    ):
        return {}
    return {
        "lightgbm_version": _module_version("lightgbm", "lightgbm"),
        "numpy_version": _module_version("numpy", "numpy"),
    }


def _neural_runtime_versions(specs: Sequence[ExperimentSpec]) -> dict[str, str]:
    uses_independent_mlp = any(
        "independent_mlp"
        in {
            candidate.estimator_id,
            candidate.graph.initializer_estimator_id,
            candidate.graph.updater_estimator_id,
        }
        for spec in specs
        for candidate in spec.candidates
    )
    if not uses_independent_mlp:
        return {}
    versions = {
        "numpy_version": _module_version("numpy", "numpy"),
        "torch_version": _module_version("torch", "torch"),
        "safetensors_version": _module_version("safetensors", "safetensors"),
    }
    missing = sorted(name for name, value in versions.items() if value == "not-installed")
    if missing:
        raise RuntimeError(
            "Independent MLP runtime identity is incomplete; missing distributions: "
            + ", ".join(missing)
        )
    return versions


def _experiment_runtime_versions(
    specs: Sequence[ExperimentSpec],
) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "token_prediction_version": _installed_version("token-prediction"),
        **_lightgbm_runtime_versions(specs),
        **_neural_runtime_versions(specs),
    }


def run_development_experiments(
    dataset: SupervisedDataset,
    specs: Sequence[ExperimentSpec],
    *,
    source_provenance: Mapping[str, object],
    protocol: DevelopmentProtocol | None = None,
    registry: EstimatorRegistry | None = None,
) -> DevelopmentExperimentResults:
    """Run the frozen three-seed protocol for an already constructed dataset."""

    resolved_protocol = protocol or build_development_protocol(dataset)
    if (
        resolved_protocol.parent_dataset_id != dataset.dataset_id
        or resolved_protocol.parent_schema_version != dataset.schema_version
        or resolved_protocol.parent_source_descriptor_hash != dataset.source_descriptor_hash
        or resolved_protocol.parent_capability_contract_hash != dataset.capability_contract_hash
        or resolved_protocol.parent_input_contract_hash != dataset.input_contract_hash
    ):
        raise ValueError("development protocol belongs to another parent dataset")
    if (
        resolved_protocol.development_dataset.source_descriptor_hash is None
        or resolved_protocol.development_dataset.capability_contract_hash is None
    ):
        raise ValueError("development execution requires source/capability provenance")
    validated_source_provenance = validate_source_provenance(
        source_provenance,
        source_descriptor_hash=(resolved_protocol.development_dataset.source_descriptor_hash),
        capability_contract_hash=(resolved_protocol.development_dataset.capability_contract_hash),
    )
    runner = ExperimentRunner(registry or builtin_registry())
    seed_results: list[SeedDevelopmentResults] = []
    for nested_plan in resolved_protocol.outer_inner_plans:
        result_groups = tuple(
            runner.run(
                resolved_protocol.development_dataset,
                nested_plan.outer_plan,
                spec,
                seed=nested_plan.split_seed,
                development_protocol=resolved_protocol,
                source_provenance=validated_source_provenance,
            )
            for spec in specs
        )
        seed_results.append(
            SeedDevelopmentResults(
                split_seed=nested_plan.split_seed,
                split_plan_id=nested_plan.outer_plan.split_plan_id,
                result_groups=result_groups,
            )
        )
    return DevelopmentExperimentResults(
        protocol=resolved_protocol,
        seed_results=tuple(seed_results),
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_fold_artifacts(
    root: Path,
    *,
    experiment_id: str,
    candidate_id: str,
    split_seed: int,
    artifacts: Sequence[FoldArtifact],
) -> None:
    experiment_component = _safe_artifact_component(experiment_id, label="experiment_id")
    candidate_component = _safe_artifact_component(candidate_id, label="candidate_id")
    for artifact in artifacts:
        if not artifact.has_payload:
            continue
        fold_dir = (
            root
            / "fold_artifacts"
            / experiment_component
            / candidate_component
            / f"seed_{split_seed}"
            / f"fold_{artifact.fold}"
        )
        fold_dir.mkdir(parents=True, exist_ok=False)
        if artifact.encoder is not None:
            _write_json(fold_dir / "encoder.json", artifact.encoder)
        if artifact.fit_report is not None:
            _write_json(fold_dir / "fit_report.json", artifact.fit_report)
        if artifact.calibrator is not None:
            _write_json(fold_dir / "calibrator.json", artifact.calibrator)
        if artifact.provenance is not None:
            _write_json(fold_dir / "provenance.json", artifact.provenance)
        if artifact.feature_importance is not None:
            lines = [
                json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
                for record in artifact.feature_importance
            ]
            (fold_dir / "feature_importance.jsonl").write_text(
                ("\n".join(lines) + "\n") if lines else "",
                encoding="utf-8",
            )
        if artifact.model_strings is not None:
            for model_name, model_text in sorted(artifact.model_strings.items()):
                component = _safe_artifact_component(model_name, label="model_strings() key")
                (fold_dir / f"{component}.model.txt").write_text(model_text, encoding="utf-8")
        if artifact.bundle_files is not None:
            bundle_dir = fold_dir / "bundle"
            bundle_dir.mkdir()
            for filename, payload in sorted(artifact.bundle_files.items()):
                relative = PurePosixPath(filename)
                destination = bundle_dir.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    destination.resolve().relative_to(bundle_dir.resolve())
                except ValueError as exc:  # defensive parity with FoldArtifact
                    raise ValueError("bundle file escaped its fold artifact root") from exc
                destination.write_bytes(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _resolve_event_paths(
    project_root: Path,
    event_paths: Iterable[str | Path],
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for raw_path in event_paths:
        rendered = str(raw_path)
        if rendered != rendered.strip():
            raise ValueError("event path must not have leading or trailing whitespace")
        absolute = Path(os.path.abspath(rendered))
        try:
            relative = absolute.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("event path escapes the project root") from exc
        resolved.append(
            resolve_canonical_input_file(
                project_root,
                relative.as_posix(),
                context="event path",
            )
        )
    return tuple(resolved)


def _verify_v2_source_manifest(
    config: ProjectConfig,
    source_paths: Sequence[Path],
) -> dict[Path, tuple[str, int, str]]:
    descriptor = config.source_descriptor
    if descriptor is None:
        raise ValueError("config schema_version=2 requires a source descriptor")
    project_root = config.source_path.resolve().parent.parent
    if config.source_descriptor_path is None or config.source_descriptor_file_sha256 is None:
        raise ValueError("config schema_version=2 requires a tracked source descriptor")
    descriptor_path = resolve_canonical_input_file(
        project_root,
        config.source_descriptor_path,
        context="tracked source descriptor",
    )
    if _sha256_file(descriptor_path) != config.source_descriptor_file_sha256:
        raise ValueError("tracked source descriptor SHA-256 does not match")
    try:
        descriptor_payload = json.loads(
            descriptor_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in source descriptor: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tracked source descriptor: {exc}") from exc
    if not isinstance(descriptor_payload, dict):
        raise ValueError("tracked source descriptor must be a JSON object")
    if SourceDescriptor.from_dict(descriptor_payload) != descriptor:
        raise ValueError("configured source descriptor differs from tracked descriptor")
    descriptor_manifest_path = resolve_canonical_input_file(
        project_root,
        descriptor.manifest_path,
        context="source descriptor manifest",
    )
    if _sha256_file(descriptor_manifest_path) != descriptor.manifest_sha256:
        raise ValueError("source descriptor manifest SHA-256 does not match")
    if config.canonical_manifest_path is None or config.canonical_manifest_sha256 is None:
        raise ValueError("config schema_version=2 requires a canonical input manifest")
    manifest_path = resolve_canonical_input_file(
        project_root,
        config.canonical_manifest_path,
        context="canonical source manifest",
    )
    if _sha256_file(manifest_path) != config.canonical_manifest_sha256:
        raise ValueError("canonical source manifest SHA-256 does not match")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in source manifest: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid canonical source manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("canonical source manifest must be a JSON object")
    allowed = {"manifest_schema_version", "source_id", "revision", "files"}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(
            f"canonical source manifest keys differ; missing={missing}, unknown={unknown}"
        )
    if (
        isinstance(payload["manifest_schema_version"], bool)
        or payload["manifest_schema_version"] != 1
    ):
        raise ValueError("canonical source manifest schema version must be 1")
    if payload["source_id"] != descriptor.source_id:
        raise ValueError("canonical source manifest source_id does not match descriptor")
    if payload["revision"] != descriptor.revision:
        raise ValueError("canonical source manifest revision does not match descriptor")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("canonical source manifest files must be a non-empty list")
    declared: dict[Path, tuple[str, int, str]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("canonical source manifest file entries have an invalid schema")
        relative = canonical_input_path(
            item["path"],
            context="source manifest file path",
        )
        path = resolve_canonical_input_file(
            project_root,
            relative,
            context="canonical source manifest input",
        )
        if path in declared:
            raise ValueError("canonical source manifest contains a duplicate file path")
        byte_count = item["bytes"]
        digest = item["sha256"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("canonical source manifest bytes must be a non-negative integer")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("canonical source manifest SHA-256 is invalid")
        declared[path] = (relative, byte_count, digest)
    actual_paths = set(source_paths)
    if len(actual_paths) != len(source_paths):
        raise ValueError("event paths must not contain duplicates")
    if actual_paths != set(declared):
        raise ValueError("event paths do not exactly match the canonical source manifest")
    for path, (_, expected_bytes, expected_hash) in declared.items():
        if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_hash:
            raise ValueError("canonical source manifest input size or SHA-256 does not match")
    return declared


def _source_tree_hash() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_jsonl_event_bytes(raw: bytes, *, description: str) -> list[CanonicalEvent]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"invalid UTF-8 in {description}: {exc}") from exc
    events: list[CanonicalEvent] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant in canonical event: {value}")
                ),
            )
            if not isinstance(payload, dict):
                raise ValueError("canonical event must be a JSON object")
            events.append(CanonicalEvent.from_dict(payload))
        except Exception as exc:
            raise ValueError(f"invalid event on line {line_number}: {exc}") from exc
    return events


def load_jsonl_events(path: str | Path) -> list[CanonicalEvent]:
    source = Path(path)
    return _load_jsonl_event_bytes(source.read_bytes(), description=source.name)


def load_trajectory(path: str | Path) -> Trajectory:
    return Trajectory.from_events(load_jsonl_events(path))


def inspect_replay(path: str | Path) -> dict[str, object]:
    trajectory = load_trajectory(path)
    snapshots = replay_feature_snapshots(trajectory.events)
    labels = build_prediction_labels(trajectory.events)
    return {
        "events": len(trajectory.events),
        "task_id": trajectory.task_id,
        "prediction_points": len(snapshots),
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
        "labels": [asdict(label) for label in labels],
    }


def _prediction_dict(
    result: object,
    record: object,
    *,
    split_seed: int,
) -> dict[str, object]:
    forecast = getattr(record, "forecast")
    return {
        "candidate_id": getattr(result, "candidate_id"),
        "candidate_hash": getattr(result, "candidate_hash"),
        "dataset_id": getattr(result, "dataset_id"),
        "split_plan_id": getattr(result, "split_plan_id"),
        "split_seed": split_seed,
        "eligibility_hash": getattr(result, "eligibility_hash"),
        "point_id": getattr(record, "point_id"),
        "task_id": getattr(record, "task_id"),
        "trajectory_id": getattr(record, "trajectory_id"),
        "condition_id": getattr(record, "condition_id"),
        "fold": getattr(record, "fold"),
        "target": getattr(record, "target").value,
        "lower": forecast.lower,
        "prediction": forecast.point,
        "upper": forecast.upper,
        "raw_lower": forecast.raw_lower,
        "raw_prediction": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
        "quantiles_crossed_before_repair": (
            bool(forecast.raw_lower > forecast.raw_point or forecast.raw_point > forecast.raw_upper)
            if forecast.raw_lower is not None
            and forecast.raw_point is not None
            and forecast.raw_upper is not None
            else None
        ),
        "latency_ms": forecast.latency_ms,
        "overhead_input_tokens": forecast.overhead_input_tokens,
        "overhead_output_tokens": forecast.overhead_output_tokens,
        "sample_weight": getattr(record, "sample_weight"),
    }


def run_configured_experiments(
    config: ProjectConfig,
    event_paths: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
) -> ExperimentRunSummary:
    if config.schema_version == 1:
        raise ValueError(
            "config schema_version=1 is verification-only; new experiment runs require v2"
        )
    if config.folds != OUTER_FOLDS:
        raise ValueError("schema v2 production runs require exactly five outer folds")
    if config.seed != STAGE_SPLIT_SEEDS[0]:
        raise ValueError("schema v2 seed must declare the frozen development protocol anchor")
    if config.collection_source != "canonical_jsonl":
        raise ValueError(
            "configured experiments currently require collection.source='canonical_jsonl'"
        )
    specs = config.experiment_specs()
    _validate_artifact_identifiers(specs)
    project_root = config.source_path.resolve().parent.parent
    source_paths = _resolve_event_paths(project_root, event_paths)
    declared_sources = _verify_v2_source_manifest(config, source_paths)
    trajectories_list: list[Trajectory] = []
    source_facts: list[dict[str, object]] = []
    for path in source_paths:
        raw = path.read_bytes()
        relative, expected_bytes, expected_hash = declared_sources[path]
        observed_hash = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_bytes or observed_hash != expected_hash:
            raise ValueError("canonical source changed while it was being loaded")
        trajectories_list.append(
            Trajectory.from_events(_load_jsonl_event_bytes(raw, description=relative))
        )
        source_facts.append({"path": relative, "bytes": len(raw), "sha256": observed_hash})
    trajectories = tuple(trajectories_list)
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if config.source_descriptor is None:
        raise ValueError("config schema_version=2 requires a source descriptor")
    parent_dataset = build_capability_supervised_dataset(
        trajectories,
        config.source_descriptor,
    )
    protocol = build_development_protocol(parent_dataset)
    dataset = protocol.development_dataset
    runtime_versions = _experiment_runtime_versions(specs)
    code_hash = _source_tree_hash()
    source_provenance: dict[str, object] = {
        "source_descriptor": config.source_descriptor.to_dict(),
        "source_descriptor_hash": config.source_descriptor.descriptor_hash,
        "code_hash": code_hash,
        "runtime_versions": runtime_versions,
    }
    run_semantic = {
        "config_hash": config.source_hash,
        "code_hash": code_hash,
        "sources": sorted(source_facts, key=lambda item: str(item["path"])),
        "source_hashes": sorted(str(item["sha256"]) for item in source_facts),
        "source_descriptor_file_sha256": config.source_descriptor_file_sha256,
        "canonical_manifest_sha256": config.canonical_manifest_sha256,
        "dataset_id": dataset.dataset_id,
        "parent_dataset_id": parent_dataset.dataset_id,
        "development_protocol_id": protocol.protocol_id,
        "split_plan_id": protocol.protocol_id,
        "split_plan_ids": [plan.split_plan_id for plan in protocol.outer_plans],
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "permanent_holdout_plan_id": protocol.holdout_plan.holdout_plan_id,
        "permanent_holdout_assignment_id": protocol.holdout_plan.assignment_id,
        "final_holdout_task_count": len(protocol.final_holdout_tasks),
        "config_schema_version": config.schema_version,
        "dataset_schema_version": dataset.schema_version,
        "source_descriptor_hash": dataset.source_descriptor_hash,
        "capability_contract_hash": dataset.capability_contract_hash,
        "input_contract_hash": dataset.input_contract_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "experiments": _effective_experiment_semantics(specs),
        **runtime_versions,
    }
    run_id = hashlib.sha256(
        json.dumps(run_semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else config.workspace / "experiments" / run_id
    )
    if (directory / "_SUCCESS").exists():
        manifest = verify_artifact(directory)
        existing_run_id = manifest.metadata.get("run_id")
        if existing_run_id != run_id:
            raise ValueError(
                f"existing experiment artifact has run_id {existing_run_id!r}, expected {run_id!r}"
            )
        return ExperimentRunSummary(
            run_id=run_id,
            output_dir=directory,
            dataset_id=dataset.dataset_id,
            split_plan_id=protocol.protocol_id,
            artifact_id=manifest.artifact_id,
            experiment_count=len(specs),
            candidate_run_count=(
                len(STAGE_SPLIT_SEEDS) * sum(len(spec.candidates) for spec in specs)
            ),
        )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"experiment output is not empty: {directory}")
    execution = run_development_experiments(
        parent_dataset,
        specs,
        source_provenance=source_provenance,
        protocol=protocol,
    )
    try:
        current_declared_sources = _verify_v2_source_manifest(config, source_paths)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "source manifests changed while the experiment was running; "
            "refusing to publish a mixed-input artifact"
        ) from exc
    if current_declared_sources != declared_sources:
        raise RuntimeError("source manifest membership changed while the experiment was running")
    for path, (relative, expected_bytes, expected_hash) in declared_sources.items():
        try:
            current_path = resolve_canonical_input_file(
                project_root,
                relative,
                context="canonical source manifest input",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "canonical source changed while the experiment was running; "
                "refusing to publish a mixed-input artifact"
            ) from exc
        if (
            current_path != path
            or current_path.stat().st_size != expected_bytes
            or _sha256_file(current_path) != expected_hash
        ):
            raise RuntimeError(
                "canonical source changed while the experiment was running; "
                "refusing to publish a mixed-input artifact"
            )
    if _source_tree_hash() != run_semantic["code_hash"]:
        raise RuntimeError(
            "source tree changed while the experiment was running; "
            "refusing to publish a mixed-code artifact"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "parent_dataset_id": parent_dataset.dataset_id,
                "development_protocol_id": protocol.protocol_id,
                "schema_version": dataset.schema_version,
                "rows": len(dataset.rows),
                "eligible_rows": sum(row.eligible for row in dataset.rows),
                "tasks": len(dataset.task_ids),
                "final_holdout_tasks": len(protocol.final_holdout_tasks),
                "status_counts": {
                    status: sum(row.status.value == status for row in dataset.rows)
                    for status in sorted({row.status.value for row in dataset.rows})
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "split.json").write_text(
        json.dumps(
            execution.audit_document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_lines: list[str] = []
    metrics: dict[str, object] = {}
    for spec_index, spec in enumerate(specs):
        candidate_metrics: dict[str, object] = {}
        for candidate in spec.candidates:
            seed_metrics: dict[str, object] = {}
            for seed_result in execution.seed_results:
                results = seed_result.result_groups[spec_index]
                result = next(
                    item for item in results if item.candidate_id == candidate.candidate_id
                )
                seed_metrics[str(seed_result.split_seed)] = {
                    "split_plan_id": seed_result.split_plan_id,
                    "comparability_key": result.comparability_key,
                    "metrics": dict(result.metrics),
                    "fold_metrics": {
                        str(fold): dict(fold_values)
                        for fold, fold_values in result.fold_metrics.items()
                    },
                }
                _write_fold_artifacts(
                    directory,
                    experiment_id=spec.experiment_id,
                    candidate_id=result.candidate_id,
                    split_seed=seed_result.split_seed,
                    artifacts=result.fold_artifacts,
                )
                prediction_lines.extend(
                    json.dumps(
                        _prediction_dict(
                            result,
                            record,
                            split_seed=seed_result.split_seed,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    for record in result.predictions
                )
            candidate_metrics[candidate.candidate_id] = {
                "candidate_hash": candidate.content_hash,
                "split_seed_results": seed_metrics,
            }
        metrics[spec.experiment_id] = candidate_metrics
    (directory / "predictions.jsonl").write_text(
        "\n".join(prediction_lines) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = publish_artifact(
        directory,
        stage_name="experiment",
        metadata={
            **run_semantic,
            "run_id": run_id,
        },
    )
    return ExperimentRunSummary(
        run_id=run_id,
        output_dir=directory,
        dataset_id=dataset.dataset_id,
        split_plan_id=protocol.protocol_id,
        artifact_id=manifest.artifact_id,
        experiment_count=len(specs),
        candidate_run_count=(len(STAGE_SPLIT_SEEDS) * sum(len(spec.candidates) for spec in specs)),
    )
