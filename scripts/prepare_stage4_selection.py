"""Freeze the exact Stage 4 model selection without opening the final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.development import STAGE_SPLIT_SEEDS, build_development_protocol
from token_prediction.estimators import (
    FitContext,
    RunContext,
    TrainingExample,
    TrainingView,
    builtin_registry,
)
from token_prediction.evaluation import (
    CalibrationExample,
    TaskMaxConformalCalibrator,
    paired_task_metric_bootstrap,
)
from token_prediction.experiment import run_candidate_cv
from token_prediction.final_ensemble import (
    EmpiricalFoldState,
    canonical_json_bytes,
    semantic_sha256,
)
from token_prediction.lineage import (
    ArtifactManifest,
    publish_artifact,
    sha256_file,
    verify_artifact,
)
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.stage4_matrix import build_stage4_matrix

if __package__:
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from scripts.run_stage2_experiments import load_stage2_source
    from scripts.run_stage3_experiments import verify_stage3_results_document
    from scripts.run_stage4_experiments import (
        prediction_projection_sha256,
        verify_stage4_results_document,
    )
else:  # pragma: no cover - production CLI invocation
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from run_stage2_experiments import load_stage2_source
    from run_stage3_experiments import verify_stage3_results_document
    from run_stage4_experiments import (
        prediction_projection_sha256,
        verify_stage4_results_document,
    )


SELECTION_SCHEMA_VERSION = 1
SELECTION_ARTIFACT_SCHEMA_VERSION = 1
SELECTION_STAGE_NAME = "stage4_frozen_selection"
SELECTION_POLICY_ID = "stage4_development_only_stability_guard_v1"
SELECTION_CODE_POLICY_ID = "stage4_selection_code_tree_v1"
SELECTION_ENSEMBLE_POLICY_ID = "development_three_seed_five_fold_mean_v1"
SELECTION_REPLACEMENT_POLICY_ID = "all_three_task_bootstrap_ci_upper_below_zero_v1"
SELECTION_HOLDOUT_PROTOCOL_ID = "sealed_until_selection_commit_and_tag_v1"
SELECTION_RUNNER_RELATIVE = "scripts/prepare_stage4_selection.py"
DEFAULT_OUTPUT_ROOT = "workspace/stage4/selection"
ALLOWED_OUTPUT_PREFIX = "workspace/stage4/selection/"
EXPECTED_FOLDS = 5


@dataclass(frozen=True)
class SourceArtifactSpec:
    key: str
    source_name: str
    stage: str
    path: str
    artifact_id: str
    results_payload_sha256: str
    run_id: str
    source_commit: str
    source_tag: str


SOURCE_ARTIFACTS = (
    SourceArtifactSpec(
        "stage4_spend_aggregate",
        "spend_aggregate",
        "stage4",
        "workspace/stage4/runs/s4-d4802b015313e00f8fc5",
        "854f8019c304f1c6d9bf57fbd7c49f808a83b2d197ac6a55fed44c4385da5867",
        "8f0c9c07005823bc158b2f870c05c167615fcba093e75afeb94895385f983950",
        "d4802b015313e00f8fc549e3",
        "1a5994de859cfcd88528fe7b77cc37167dd8f657",
        "stage4-artifact-source-v1",
    ),
    SourceArtifactSpec(
        "stage4_bagen_sokoban",
        "bagen_sokoban",
        "stage4",
        "workspace/stage4/runs/s4-90f52b7baa3f3d17d3f3",
        "ba87b8f7ea809d6386a75f0743dbd066edbcb6aed76bc607ec50559a1223805a",
        "bcd2f5f4899a59b27a511f8ea9fbcb989722ead93f5ca3220c32697477ffcad6",
        "90f52b7baa3f3d17d3f3f069",
        "1a5994de859cfcd88528fe7b77cc37167dd8f657",
        "stage4-artifact-source-v1",
    ),
    SourceArtifactSpec(
        "stage4_bagen_swebench",
        "bagen_swebench",
        "stage4",
        "workspace/stage4/runs/s4-1adc466729682d4f2b85",
        "3fcdd70e9fc23102a672db7fe2ada70a82175ab90666904733313d1497ce6f28",
        "cdec8bf1c396334d99bac17dacbd7ceae730a466a9abf16b9949a1223999b8ff",
        "1adc466729682d4f2b857061",
        "1a5994de859cfcd88528fe7b77cc37167dd8f657",
        "stage4-artifact-source-v1",
    ),
    SourceArtifactSpec(
        "stage4_spend_openhands",
        "spend_openhands",
        "stage4",
        "workspace/stage4/runs/s4-6a50dc3d21cb7926ccfb",
        "7034f4578c4378fbd77eedbe7ccdb32bf0e7fd6b6410b897a958f3c94f4eedce",
        "f2f54887ac5103d1de8bff2a47ad5e90a7456a243a283acd45bea19472ae39eb",
        "6a50dc3d21cb7926ccfba7a6",
        "1a5994de859cfcd88528fe7b77cc37167dd8f657",
        "stage4-artifact-source-v1",
    ),
    SourceArtifactSpec(
        "stage3_spend_openhands",
        "spend_openhands",
        "stage3",
        "workspace/stage3/runs/s3-6c57b8ef3acc736cceea",
        "145b6b17ae6b67f07803845c3c4f06c1f321a5e5dad40e63380806f8e07d9f59",
        "ec14a98f607576c64687d41d21a3c9e07a22639fa14c68065c769958624ac31c",
        "6c57b8ef3acc736cceea2608",
        "d3767c135c255d3803195573130f8bb0aefe0d67",
        "stage3-artifact-source-v1",
    ),
)


class Stage4SelectionError(RuntimeError):
    """The development-only final selection cannot be frozen safely."""


@dataclass(frozen=True)
class LoadedSourceArtifact:
    spec: SourceArtifactSpec
    root: Path
    manifest: ArtifactManifest
    results: Mapping[str, Any]


@dataclass(frozen=True)
class SelectionSummary:
    selection_id: str
    run_id: str
    output_dir: Path
    artifact_id: str
    selection_payload_sha256: str
    cell_count: int
    ensemble_member_count: int
    final_holdout_evaluated: bool


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage4SelectionError("JSON document contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Stage4SelectionError(f"JSON document contains non-finite value {value}")


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4SelectionError(f"{description} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise Stage4SelectionError(f"{description} must be an object")
    return value


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage4SelectionError(f"Git command failed: {message}")
    return completed.stdout


def _selection_code_paths(root: Path) -> tuple[str, ...]:
    required = {
        SELECTION_RUNNER_RELATIVE,
        "scripts/run_stage2_experiments.py",
        "scripts/run_stage3_experiments.py",
        "scripts/run_stage4_experiments.py",
        "configs/data_foundation_prediction_baseline.json",
    }
    raw = _git(
        root,
        "ls-files",
        "-z",
        "--",
        "src/token_prediction",
        *sorted(required),
    )
    paths = tuple(
        sorted(
            {
                item.decode("utf-8", errors="strict")
                for item in raw.split(b"\0")
                if item
            }
        )
    )
    if not required <= set(paths) or not any(
        path.startswith("src/token_prediction/") and path.endswith(".py")
        for path in paths
    ):
        raise Stage4SelectionError("selection source closure is incomplete at HEAD")
    return paths


def _code_binding(root: Path) -> Mapping[str, object]:
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    paths = _selection_code_paths(root)
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise Stage4SelectionError("selection runner and package must be clean at HEAD")
    digest = hashlib.sha256(f"{SELECTION_CODE_POLICY_ID}\0".encode("ascii"))
    for relative in paths:
        workspace = _repo_path(root, relative, label="selection code path")
        if not workspace.is_file() or _is_link_or_reparse(workspace):
            raise Stage4SelectionError("selection code closure contains an unsafe file")
        payload = workspace.read_bytes()
        committed = _git(root, "show", f"{commit}:{relative}")
        if payload != committed:
            raise Stage4SelectionError("selection workspace differs from committed source")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "policy_id": SELECTION_CODE_POLICY_ID,
        "git_commit": commit,
        "code_tree_sha256": digest.hexdigest(),
        "paths": list(paths),
    }


def _verify_runner_origin(root: Path) -> None:
    expected = _repo_path(root, SELECTION_RUNNER_RELATIVE, label="selection runner")
    actual = Path(__file__)
    if _is_link_or_reparse(actual) or actual.resolve() != expected.resolve():
        raise Stage4SelectionError("executing selection runner is outside repository_root")


def _source_document(spec: SourceArtifactSpec) -> dict[str, object]:
    return asdict(spec)


def _verify_source_tag(root: Path, spec: SourceArtifactSpec) -> None:
    actual = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{spec.source_tag}^{{commit}}",
    ).decode("ascii").strip()
    if actual != spec.source_commit:
        raise Stage4SelectionError(f"{spec.source_tag} does not point to its source commit")


def _load_source_artifact(root: Path, spec: SourceArtifactSpec) -> LoadedSourceArtifact:
    relative = _safe_relative(spec.path, label=f"{spec.key} artifact path")
    source_root = _repo_path(root, relative, label=f"{spec.key} artifact")
    manifest = verify_artifact(source_root)
    expected_stage = "stage4_development_source" if spec.stage == "stage4" else (
        "stage3_development_source"
    )
    if (
        manifest.artifact_id != spec.artifact_id
        or manifest.stage_name != expected_stage
        or manifest.metadata.get("run_id") != spec.run_id
        or manifest.metadata.get("results_payload_sha256")
        != spec.results_payload_sha256
    ):
        raise Stage4SelectionError(f"{spec.key} artifact identity differs from the lock")
    results = _load_json(source_root / "results.json", description=f"{spec.key} results")
    payload_hash = (
        verify_stage4_results_document(results)
        if spec.stage == "stage4"
        else verify_stage3_results_document(results)
    )
    if (
        payload_hash != spec.results_payload_sha256
        or results.get("run_id") != spec.run_id
        or results.get("final_holdout")
        != {
            "evaluated": False,
            "prediction_count": 0,
            "selection_claim": "none",
            "target_values_used_for_fit_calibration_scoring": False,
        }
    ):
        raise Stage4SelectionError(f"{spec.key} results are not sealed development evidence")
    code = results.get("code_binding")
    if not isinstance(code, Mapping) or code.get("git_commit") != spec.source_commit:
        raise Stage4SelectionError(f"{spec.key} source commit differs from the lock")
    _verify_source_tag(root, spec)
    return LoadedSourceArtifact(spec, source_root, manifest, results)


def _experiment(
    source: LoadedSourceArtifact,
    *,
    position: str,
    target: str,
    suffix: str | None = None,
    calibrator_id: str = "task_max_conformal",
) -> Mapping[str, Any]:
    experiments = source.results.get("experiments")
    if not isinstance(experiments, list):
        raise Stage4SelectionError("source experiments are invalid")
    matches = [
        value
        for value in experiments
        if isinstance(value, Mapping)
        and value.get("position") == position
        and value.get("target") == target
        and value.get("calibrator_id") == calibrator_id
        and (suffix is None or str(value.get("experiment_id", "")).endswith(suffix))
    ]
    if len(matches) != 1:
        raise Stage4SelectionError(
            f"{source.spec.key} does not contain one selected {position}/{target} experiment"
        )
    return matches[0]


def _candidate(
    experiment: Mapping[str, Any],
    candidate_id: str,
) -> Mapping[str, Any]:
    values = experiment.get("candidates")
    if not isinstance(values, list):
        raise Stage4SelectionError("experiment candidates are invalid")
    matches = [
        value
        for value in values
        if isinstance(value, Mapping) and value.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise Stage4SelectionError(
            f"experiment does not contain candidate {candidate_id!r} exactly once"
        )
    return matches[0]


def _mean_mae(candidate: Mapping[str, Any]) -> float:
    try:
        value = float(candidate["cross_seed_metrics"]["mae"]["mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage4SelectionError("candidate cross-seed MAE is invalid") from exc
    if not value >= 0:
        raise Stage4SelectionError("candidate cross-seed MAE must be non-negative")
    return value


def _lowest_mae_candidate(experiment: Mapping[str, Any]) -> str:
    values = experiment.get("candidates")
    if not isinstance(values, list) or not values:
        raise Stage4SelectionError("method experiment candidates are invalid")
    ordered = sorted(
        (
            (_mean_mae(value), str(value.get("candidate_id", "")))
            for value in values
            if isinstance(value, Mapping)
        )
    )
    if not ordered or not ordered[0][1]:
        raise Stage4SelectionError("method experiment has no identified candidate")
    return ordered[0][1]


def _task_metric_mapping(seed_result: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    values = seed_result.get("task_metrics")
    if not isinstance(values, list) or not values:
        raise Stage4SelectionError("seed result task metrics are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise Stage4SelectionError("task metric record is invalid")
        task = value.get("task_pseudonym")
        if not isinstance(task, str) or not task or task in result:
            raise Stage4SelectionError("task metric pseudonym is invalid")
        result[task] = {
            key: item for key, item in value.items() if key != "task_pseudonym"
        }
    return result


def _paired_stage3_evidence(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> list[dict[str, object]]:
    candidate_seeds = {
        int(value["split_seed"]): value
        for value in candidate["seed_results"]
        if isinstance(value, Mapping)
    }
    reference_seeds = {
        int(value["split_seed"]): value
        for value in reference["seed_results"]
        if isinstance(value, Mapping)
    }
    if set(candidate_seeds) != set(STAGE_SPLIT_SEEDS) or set(reference_seeds) != set(
        STAGE_SPLIT_SEEDS
    ):
        raise Stage4SelectionError("Stage 3 comparison does not contain all split seeds")
    documents: list[dict[str, object]] = []
    for split_seed in STAGE_SPLIT_SEEDS:
        candidate_result = SimpleNamespace(
            candidate_id=candidate["candidate_id"],
            comparability_key=("stage3-spend-final-selection", str(split_seed)),
            task_metrics=_task_metric_mapping(candidate_seeds[split_seed]),
        )
        reference_result = SimpleNamespace(
            candidate_id=reference["candidate_id"],
            comparability_key=("stage3-spend-final-selection", str(split_seed)),
            task_metrics=_task_metric_mapping(reference_seeds[split_seed]),
        )
        bootstrap_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"{SELECTION_REPLACEMENT_POLICY_ID}\0{split_seed}\0"
                    f"{candidate['candidate_id']}\0{reference['candidate_id']}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        comparison = paired_task_metric_bootstrap(
            candidate_result,
            reference_result,
            iterations=10_000,
            seed=bootstrap_seed,
        )
        documents.append({"split_seed": split_seed, **asdict(comparison)})
    return documents


def _replacement_guard(experiment: Mapping[str, Any]) -> dict[str, object]:
    values = experiment.get("candidates")
    if not isinstance(values, list):
        raise Stage4SelectionError("feature-ablation candidates are invalid")
    candidates: list[dict[str, object]] = []
    for value in values:
        if (
            not isinstance(value, Mapping)
            or value.get("candidate_id") in {"empirical", "lightgbm_history"}
        ):
            continue
        seed_results = value.get("seed_results")
        if not isinstance(seed_results, list) or len(seed_results) != len(STAGE_SPLIT_SEEDS):
            raise Stage4SelectionError("ablation candidate seed evidence is incomplete")
        comparisons = []
        for seed_result in seed_results:
            if not isinstance(seed_result, Mapping):
                raise Stage4SelectionError("ablation seed result is invalid")
            comparison = seed_result.get("paired_vs_reference")
            if not isinstance(comparison, Mapping):
                raise Stage4SelectionError("ablation lacks paired reference evidence")
            comparisons.append(
                {
                    "split_seed": int(seed_result["split_seed"]),
                    "mae_delta": float(comparison["mae_delta"]),
                    "mae_delta_ci_lower": float(comparison["mae_delta_ci_lower"]),
                    "mae_delta_ci_upper": float(comparison["mae_delta_ci_upper"]),
                }
            )
        qualified = all(item["mae_delta_ci_upper"] < 0 for item in comparisons)
        candidates.append(
            {
                "candidate_id": value["candidate_id"],
                "cross_seed_mae": _mean_mae(value),
                "qualified_replacement": qualified,
                "comparisons": comparisons,
            }
        )
    if any(bool(value["qualified_replacement"]) for value in candidates):
        raise Stage4SelectionError("a feature/method ablation qualifies to replace history")
    return {
        "policy_id": SELECTION_REPLACEMENT_POLICY_ID,
        "reference_candidate_id": "lightgbm_history",
        "candidates": candidates,
        "selected_candidate_id": "lightgbm_history",
    }


def _member_file_projection(
    source: LoadedSourceArtifact,
    *,
    prefix: str,
) -> tuple[str, list[dict[str, str]]]:
    normalized = PurePosixPath(prefix).as_posix().rstrip("/") + "/"
    files = [
        {"path": path.removeprefix(normalized), "sha256": digest}
        for path, digest in sorted(source.manifest.files.items())
        if path.startswith(normalized)
    ]
    if not files:
        raise Stage4SelectionError(f"bundle member prefix has no files: {prefix}")
    return semantic_sha256(files), files


def _fold_members(
    source: LoadedSourceArtifact,
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    bundle_kind: str,
) -> list[dict[str, object]]:
    experiment_key = str(experiment["artifact_key"])
    candidate_key = str(candidate["artifact_key"])
    seed_results = candidate.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != len(STAGE_SPLIT_SEEDS):
        raise Stage4SelectionError("selected candidate seed results are incomplete")
    members: list[dict[str, object]] = []
    for seed_result in sorted(seed_results, key=lambda value: int(value["split_seed"])):
        split_seed = int(seed_result["split_seed"])
        if split_seed not in STAGE_SPLIT_SEEDS or seed_result.get(
            "fold_artifact_count"
        ) != EXPECTED_FOLDS:
            raise Stage4SelectionError("selected candidate fold evidence is incomplete")
        split_plan_id = str(seed_result["split_plan_id"])
        for fold in range(EXPECTED_FOLDS):
            fold_prefix = (
                f"fold_artifacts/{experiment_key}/{candidate_key}/"
                f"seed_{split_seed}/fold_{fold}"
            )
            bundle_relative = f"{fold_prefix}/bundle"
            bundle_root = source.root.joinpath(*PurePosixPath(bundle_relative).parts)
            calibrator_relative = f"{fold_prefix}/calibrator.json"
            provenance_relative = f"{fold_prefix}/provenance.json"
            calibrator_path = source.root.joinpath(
                *PurePosixPath(calibrator_relative).parts
            )
            provenance_path = source.root.joinpath(
                *PurePosixPath(provenance_relative).parts
            )
            if bundle_kind == "lightgbm":
                from token_prediction.estimators.lightgbm_bundle import (
                    load_lightgbm_bundle,
                )

                loaded = load_lightgbm_bundle(bundle_root)
                if (
                    loaded.target.value != experiment["target"]
                    or loaded.position.value != experiment["position"]
                    or loaded.allowed_condition_ids
                    != (str(experiment["condition_id"]),)
                ):
                    raise Stage4SelectionError("LightGBM bundle scope differs from selection")
            elif bundle_kind == "lifecycle":
                loaded_lifecycle = load_lifecycle_bundle(bundle_root)
                manifest = loaded_lifecycle.manifest
                if (
                    manifest["candidate_id"] != candidate["candidate_id"]
                    or manifest["candidate_hash"] != candidate["candidate_hash"]
                    or manifest["target"] != experiment["target"]
                    or manifest["condition_id"] != experiment["condition_id"]
                    or manifest["outer_fold"] != fold
                    or manifest["split_plan_id"] != split_plan_id
                ):
                    raise Stage4SelectionError("lifecycle bundle scope differs from selection")
            else:
                raise Stage4SelectionError("unsupported selected bundle kind")
            calibrator = _load_json(
                calibrator_path,
                description="selected fold calibrator",
            )
            if (
                calibrator.get("calibrator_id") != experiment["calibrator_id"]
                or float(calibrator.get("interval_alpha", -1)) != float(
                    experiment["alpha"]
                )
            ):
                raise Stage4SelectionError("selected fold calibrator differs from experiment")
            provenance = _load_json(
                provenance_path,
                description="selected fold provenance",
            )
            expected_provenance = {
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "condition_id": experiment["condition_id"],
                "position": experiment["position"],
                "target": experiment["target"],
                "fold": fold,
                "split_plan_id": split_plan_id,
                "calibrator_id": experiment["calibrator_id"],
            }
            if any(provenance.get(key) != value for key, value in expected_provenance.items()):
                raise Stage4SelectionError("selected fold provenance differs from experiment")
            bundle_tree_sha256, bundle_files = _member_file_projection(
                source,
                prefix=bundle_relative,
            )
            member = {
                "origin": source.spec.key,
                "bundle_kind": bundle_kind,
                "split_seed": split_seed,
                "split_plan_id": split_plan_id,
                "fold": fold,
                "bundle_path": f"{source.spec.path}/{bundle_relative}",
                "bundle_tree_sha256": bundle_tree_sha256,
                "bundle_file_count": len(bundle_files),
                "calibrator_path": f"{source.spec.path}/{calibrator_relative}",
                "calibrator_sha256": sha256_file(calibrator_path),
                "provenance_path": f"{source.spec.path}/{provenance_relative}",
                "provenance_sha256": sha256_file(provenance_path),
            }
            member["member_sha256"] = semantic_sha256(member)
            members.append(member)
    if len(members) != len(STAGE_SPLIT_SEEDS) * EXPECTED_FOLDS:
        raise Stage4SelectionError("selected ensemble does not contain 15 fold members")
    return members


def _candidate_evidence(experiment: Mapping[str, Any]) -> list[dict[str, object]]:
    values = experiment.get("candidates")
    if not isinstance(values, list):
        raise Stage4SelectionError("candidate evidence is invalid")
    return [
        {
            "candidate_id": value["candidate_id"],
            "candidate_hash": value["candidate_hash"],
            "cross_seed_mae": _mean_mae(value),
        }
        for value in values
        if isinstance(value, Mapping)
    ]


def _cell_document(
    source: LoadedSourceArtifact,
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    bundle_kind: str,
    selection_reason: str,
    guard: Mapping[str, object] | None = None,
    members: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    resolved_members = members or _fold_members(
        source,
        experiment,
        candidate,
        bundle_kind=bundle_kind,
    )
    identity = {
        "source_name": source.spec.source_name,
        "source_id": source.results["source"]["source_id"],
        "condition_id": experiment["condition_id"],
        "position": experiment["position"],
        "target": experiment["target"],
    }
    return {
        "cell_id": semantic_sha256(identity),
        **identity,
        "selected_stage": source.spec.stage,
        "selected_artifact_key": source.spec.key,
        "experiment_id": experiment["experiment_id"],
        "experiment_artifact_key": experiment["artifact_key"],
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "candidate_artifact_key": candidate["artifact_key"],
        "estimator_id": candidate["estimator_id"],
        "feature_set_id": candidate["feature_set_id"],
        "feature_set_hash": candidate["feature_set_hash"],
        "calibrator_id": experiment["calibrator_id"],
        "alpha": experiment["alpha"],
        "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
        "ensemble_member_count": len(resolved_members),
        "members": resolved_members,
        "development_evidence": {
            "selected_cross_seed_mae": _mean_mae(candidate),
            "candidates": _candidate_evidence(experiment),
            "selection_reason": selection_reason,
            "replacement_guard": dict(guard) if guard is not None else None,
        },
    }


def _training_view(
    dataset_slice: Any,
    rows: Sequence[Any],
    weights: Mapping[str, float],
    feature_set: Any,
) -> TrainingView:
    examples = tuple(
        TrainingExample(
            point=row.point.with_features(feature_set.select(row.point.features)),
            target_value=float(row.label),
            sample_weight=weights[row.point.point_id],
        )
        for row in rows
        if row.label is not None
    )
    return TrainingView(
        dataset_id=dataset_slice.dataset_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        examples=examples,
        input_contract_hash=dataset_slice.input_contract_hash,
    )


def _fit_empirical_members(
    root: Path,
    temporary: Path,
    frozen_source: LoadedSourceArtifact,
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, object]]:
    lock_context = load_lock_context(root, DEFAULT_BASELINE_LOCK)
    loaded = load_stage2_source(root, lock_context, source_name="spend_aggregate")
    protocol = build_development_protocol(loaded.derived_dataset)
    if (
        protocol.protocol_id
        != frozen_source.results["development_protocol"]["protocol_id"]
        or protocol.development_dataset.dataset_id
        != frozen_source.results["dataset"]["development_dataset_id"]
    ):
        raise Stage4SelectionError("aggregate development protocol changed before selection")
    matrix = build_stage4_matrix(
        protocol,
        source_id=loaded.source_lock.descriptor.source_id,
        capabilities=loaded.source_lock.descriptor.capabilities,
    )
    specs = [
        spec
        for spec in matrix.experiments
        if spec.experiment_id == experiment["experiment_id"]
    ]
    if len(specs) != 1:
        raise Stage4SelectionError("aggregate selected experiment is absent from the matrix")
    spec = specs[0]
    candidates = [
        value
        for value in spec.candidates
        if value.candidate_id == candidate["candidate_id"]
    ]
    if len(candidates) != 1 or candidates[0].content_hash != candidate["candidate_hash"]:
        raise Stage4SelectionError("aggregate empirical candidate changed before selection")
    candidate_spec = candidates[0]
    dataset_slice = protocol.development_dataset.select(
        spec.position,
        spec.target,
        required_features=spec.required_features,
        condition_id=spec.condition_id,
    )
    frozen_seed_results = {
        int(value["split_seed"]): value
        for value in candidate["seed_results"]
        if isinstance(value, Mapping)
    }
    registry = builtin_registry()
    weights = {
        value.row.point.point_id: value.sample_weight
        for value in dataset_slice.weighted_rows()
    }
    members: list[dict[str, object]] = []
    for split_plan in protocol.outer_plans:
        parity_result = run_candidate_cv(
            dataset_slice,
            split_plan,
            candidate_spec,
            registry,
            alpha=spec.alpha,
            calibrator_id=spec.calibrator_id,
            seed=split_plan.seed,
        )
        if prediction_projection_sha256(parity_result) != frozen_seed_results[
            split_plan.seed
        ]["prediction_projection_sha256"]:
            raise Stage4SelectionError(
                "aggregate empirical development prediction parity failed"
            )
        for fold in range(split_plan.folds):
            partition = split_plan.partition(fold)
            train_rows = [
                row
                for row in dataset_slice.rows
                if row.point.task_id in partition.train_tasks
            ]
            validation_rows = [
                row
                for row in dataset_slice.rows
                if row.point.task_id in partition.validation_tasks
            ]
            calibration_rows = [
                row
                for row in dataset_slice.rows
                if row.point.task_id in partition.calibration_tasks
            ]
            if not train_rows or not validation_rows or not calibration_rows:
                raise Stage4SelectionError("aggregate empirical fold partition is empty")
            fitted = registry.create(
                candidate_spec.estimator_id,
                candidate_spec.params,
            ).fit(
                _training_view(
                    dataset_slice,
                    train_rows,
                    weights,
                    candidate_spec.feature_set,
                ),
                _training_view(
                    dataset_slice,
                    validation_rows,
                    weights,
                    candidate_spec.feature_set,
                ),
                FitContext(
                    seed=split_plan.seed,
                    fold=fold,
                    interval_alpha=spec.alpha,
                ),
            )
            calibration_examples: list[CalibrationExample] = []
            for row in calibration_rows:
                point = row.point.with_features(
                    candidate_spec.feature_set.select(row.point.features)
                )
                session = fitted.start(
                    RunContext(
                        point.task_id,
                        point.trajectory_id,
                        point.run_id,
                        dataset_id=dataset_slice.dataset_id,
                        condition_id=dataset_slice.condition_id,
                        target=dataset_slice.target,
                        input_contract_hash=dataset_slice.input_contract_hash,
                    )
                )
                if row.label is None:
                    raise Stage4SelectionError("eligible calibration row lacks a label")
                calibration_examples.append(
                    CalibrationExample(
                        row.point.task_id,
                        session.predict(point),
                        float(row.label),
                    )
                )
            calibrator = TaskMaxConformalCalibrator(alpha=spec.alpha).fit(
                calibration_examples
            )
            state = EmpiricalFoldState(
                target=dataset_slice.target,
                lower=float(fitted.lower),
                point=float(fitted.point),
                upper=float(fitted.upper),
                calibrator=calibrator,
                development_dataset_id=dataset_slice.dataset_id,
                split_plan_id=split_plan.split_plan_id,
                split_seed=split_plan.seed,
                fold=fold,
            )
            relative = (
                f"empirical/{experiment['artifact_key']}/{candidate['artifact_key']}/"
                f"seed_{split_plan.seed}/fold_{fold}/state.json"
            )
            destination = temporary.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(canonical_json_bytes(state.to_dict()) + b"\n")
            reloaded = EmpiricalFoldState.load(destination)
            if reloaded != state:
                raise Stage4SelectionError("aggregate empirical state reload changed values")
            member = {
                "origin": "selection_artifact",
                "bundle_kind": "empirical_json",
                "split_seed": split_plan.seed,
                "split_plan_id": split_plan.split_plan_id,
                "fold": fold,
                "state_path": relative,
                "state_sha256": sha256_file(destination),
            }
            member["member_sha256"] = semantic_sha256(member)
            members.append(member)
    if len(members) != len(STAGE_SPLIT_SEEDS) * EXPECTED_FOLDS:
        raise Stage4SelectionError("aggregate empirical ensemble does not contain 15 members")
    return members


def _build_cells(
    root: Path,
    temporary: Path,
    sources: Mapping[str, LoadedSourceArtifact],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    aggregate = sources["stage4_spend_aggregate"]
    aggregate_experiment = _experiment(
        aggregate,
        position=PredictionPosition.TASK_LAUNCH.value,
        target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value,
        suffix="-method",
    )
    if _lowest_mae_candidate(aggregate_experiment) != "empirical":
        raise Stage4SelectionError("aggregate empirical is no longer the lowest-MAE method")
    aggregate_candidate = _candidate(aggregate_experiment, "empirical")
    empirical_members = _fit_empirical_members(
        root,
        temporary,
        aggregate,
        aggregate_experiment,
        aggregate_candidate,
    )
    cells.append(
        _cell_document(
            aggregate,
            aggregate_experiment,
            aggregate_candidate,
            bundle_kind="empirical_json",
            selection_reason="lowest_cross_seed_development_mae",
            members=empirical_members,
        )
    )

    for source_key in ("stage4_bagen_sokoban", "stage4_bagen_swebench"):
        source = sources[source_key]
        task_experiments = [
            value
            for value in source.results["experiments"]
            if value["position"] == PredictionPosition.TASK_UPDATE.value
            and value["target"]
            == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value
            and value["calibrator_id"] == "task_max_conformal"
            and str(value["experiment_id"]).endswith("-feature-ablation")
        ]
        for experiment in task_experiments:
            selected = _candidate(experiment, "lightgbm_history")
            guard = _replacement_guard(experiment)
            cells.append(
                _cell_document(
                    source,
                    experiment,
                    selected,
                    bundle_kind="lightgbm",
                    selection_reason="locked_full_history_no_stable_single_axis_replacement",
                    guard=guard,
                )
            )
        call_experiments = [
            value
            for value in source.results["experiments"]
            if value["position"] == PredictionPosition.CALL_PRE.value
            and value["calibrator_id"] == "task_max_conformal"
            and str(value["experiment_id"]).endswith("-method")
        ]
        for experiment in call_experiments:
            if _lowest_mae_candidate(experiment) != "lightgbm_history":
                raise Stage4SelectionError("a BAGEN Call-pre method now beats LightGBM")
            selected = _candidate(experiment, "lightgbm_history")
            cells.append(
                _cell_document(
                    source,
                    experiment,
                    selected,
                    bundle_kind="lightgbm",
                    selection_reason="lowest_cross_seed_development_mae",
                )
            )

    spend_stage4 = sources["stage4_spend_openhands"]
    for experiment in spend_stage4.results["experiments"]:
        if (
            experiment["position"] == PredictionPosition.CALL_PRE.value
            and experiment["calibrator_id"] == "task_max_conformal"
            and str(experiment["experiment_id"]).endswith("-method")
        ):
            if _lowest_mae_candidate(experiment) != "lightgbm_history":
                raise Stage4SelectionError("Spend Call-pre LightGBM is not lowest-MAE")
            selected = _candidate(experiment, "lightgbm_history")
            cells.append(
                _cell_document(
                    spend_stage4,
                    experiment,
                    selected,
                    bundle_kind="lightgbm",
                    selection_reason="lowest_cross_seed_development_mae",
                )
            )

    spend_stage3 = sources["stage3_spend_openhands"]
    lifecycle_experiment = _experiment(
        spend_stage3,
        position=PredictionPosition.TASK_UPDATE.value,
        target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS.value,
    )
    selected_gru = _candidate(lifecycle_experiment, "gru_no_recurrence")
    reference_lightgbm = _candidate(lifecycle_experiment, "lightgbm_history")
    paired = _paired_stage3_evidence(selected_gru, reference_lightgbm)
    if not all(float(value["mae_delta_ci_upper"]) < 0 for value in paired):
        raise Stage4SelectionError("Spend GRU does not pass the three-seed stability guard")
    guard = {
        "policy_id": SELECTION_REPLACEMENT_POLICY_ID,
        "reference_candidate_id": "lightgbm_history",
        "candidate_id": "gru_no_recurrence",
        "qualified_replacement": True,
        "comparisons": paired,
    }
    cells.append(
        _cell_document(
            spend_stage3,
            lifecycle_experiment,
            selected_gru,
            bundle_kind="lifecycle",
            selection_reason="gru_replaces_lightgbm_with_all_three_paired_ci_upper_below_zero",
            guard=guard,
        )
    )

    cells.sort(
        key=lambda value: (
            str(value["source_name"]),
            str(value["condition_id"]),
            str(value["position"]),
            str(value["target"]),
        )
    )
    ids = [str(value["cell_id"]) for value in cells]
    if len(cells) != 29 or len(ids) != len(set(ids)):
        raise Stage4SelectionError("final selection must contain 29 unique cells")
    return cells


def _selection_document(
    *,
    code_binding: Mapping[str, object],
    sources: Mapping[str, LoadedSourceArtifact],
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source_documents = []
    for key in sorted(sources):
        source = sources[key]
        matrix = source.results.get("matrix")
        if not isinstance(matrix, Mapping):
            raise Stage4SelectionError("source matrix evidence is invalid")
        source_documents.append(
            {
                **_source_document(source.spec),
                "matrix_id": matrix.get("matrix_id"),
                "development_protocol_id": source.results["development_protocol"][
                    "protocol_id"
                ],
                "derived_dataset_id": source.results["dataset"]["derived_dataset_id"],
                "development_dataset_id": source.results["dataset"][
                    "development_dataset_id"
                ],
                "gate_projection_sha256": semantic_sha256(matrix.get("gates")),
                "telemetry_projection_sha256": semantic_sha256(
                    matrix.get("telemetry_decisions")
                ),
            }
        )
    base: dict[str, object] = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "policy_id": SELECTION_POLICY_ID,
        "replacement_policy_id": SELECTION_REPLACEMENT_POLICY_ID,
        "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
        "code_binding": dict(code_binding),
        "source_artifacts": source_documents,
        "cells": list(cells),
        "summary": {
            "source_artifact_count": len(source_documents),
            "cell_count": len(cells),
            "ensemble_member_count": sum(
                int(value["ensemble_member_count"]) for value in cells
            ),
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "outer_folds": EXPECTED_FOLDS,
        },
        "final_holdout": {
            "protocol_id": SELECTION_HOLDOUT_PROTOCOL_ID,
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
        },
    }
    base["selection_id"] = semantic_sha256(base)
    return base


def verify_selection_document(value: Mapping[str, Any]) -> str:
    expected = {
        "selection_schema_version",
        "policy_id",
        "replacement_policy_id",
        "ensemble_policy_id",
        "code_binding",
        "source_artifacts",
        "cells",
        "summary",
        "final_holdout",
        "selection_id",
        "selection_payload_sha256",
    }
    if set(value) != expected:
        raise Stage4SelectionError("selection document has missing or extra fields")
    if (
        value["selection_schema_version"] != SELECTION_SCHEMA_VERSION
        or value["policy_id"] != SELECTION_POLICY_ID
        or value["replacement_policy_id"] != SELECTION_REPLACEMENT_POLICY_ID
        or value["ensemble_policy_id"] != SELECTION_ENSEMBLE_POLICY_ID
    ):
        raise Stage4SelectionError("selection policy identity is invalid")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != 29:
        raise Stage4SelectionError("selection document must contain 29 cells")
    member_count = 0
    cell_ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise Stage4SelectionError("selection cell is invalid")
        cell_id = cell.get("cell_id")
        members = cell.get("members")
        if (
            not isinstance(cell_id, str)
            or len(cell_id) != 64
            or cell_id in cell_ids
            or not isinstance(members, list)
            or len(members) != 15
            or cell.get("ensemble_member_count") != 15
        ):
            raise Stage4SelectionError("selection cell identity or ensemble is invalid")
        cell_ids.add(cell_id)
        coordinates = {
            "source_name": cell.get("source_name"),
            "source_id": cell.get("source_id"),
            "condition_id": cell.get("condition_id"),
            "position": cell.get("position"),
            "target": cell.get("target"),
        }
        if semantic_sha256(coordinates) != cell_id:
            raise Stage4SelectionError("selection cell id does not match its coordinates")
        member_pairs = {
            (member.get("split_seed"), member.get("fold"))
            for member in members
            if isinstance(member, Mapping)
        }
        if member_pairs != {
            (seed, fold) for seed in STAGE_SPLIT_SEEDS for fold in range(EXPECTED_FOLDS)
        }:
            raise Stage4SelectionError("selection cell members do not cover 3x5 folds")
        member_count += len(members)
    summary = value["summary"]
    if (
        not isinstance(summary, Mapping)
        or summary.get("source_artifact_count") != len(SOURCE_ARTIFACTS)
        or summary.get("cell_count") != 29
        or summary.get("ensemble_member_count") != member_count
        or summary.get("split_seeds") != list(STAGE_SPLIT_SEEDS)
        or summary.get("outer_folds") != EXPECTED_FOLDS
    ):
        raise Stage4SelectionError("selection summary does not close over cells")
    if value["final_holdout"] != {
        "protocol_id": SELECTION_HOLDOUT_PROTOCOL_ID,
        "evaluated": False,
        "prediction_count": 0,
        "target_values_used_for_fit_calibration_scoring": False,
    }:
        raise Stage4SelectionError("selection document opened the final holdout")
    without_hashes = dict(value)
    declared_payload = without_hashes.pop("selection_payload_sha256")
    declared_selection_id = without_hashes.pop("selection_id")
    if semantic_sha256(without_hashes) != declared_selection_id:
        raise Stage4SelectionError("selection id does not match its semantics")
    payload = dict(value)
    payload.pop("selection_payload_sha256")
    actual_payload = semantic_sha256(payload)
    if declared_payload != actual_payload:
        raise Stage4SelectionError("selection payload checksum does not match")
    return actual_payload


def _safe_output_root(root: Path, output_root: str) -> tuple[str, Path]:
    relative = _safe_relative(output_root, label="selection output root").rstrip("/")
    canonical = f"{relative}/"
    if not canonical.startswith(ALLOWED_OUTPUT_PREFIX):
        raise Stage4SelectionError("selection output root is outside its workspace")
    output = _repo_path(root, relative, label="selection output root")
    if output.exists() and _is_link_or_reparse(output):
        raise Stage4SelectionError("selection output root is unsafe")
    return relative, output


def _existing_summary(output: Path, *, expected_run_id: str) -> SelectionSummary:
    manifest = verify_artifact(output)
    document = _load_json(output / "selection.json", description="selection document")
    payload_hash = verify_selection_document(document)
    if (
        manifest.stage_name != SELECTION_STAGE_NAME
        or manifest.schema_version != SELECTION_ARTIFACT_SCHEMA_VERSION
        or manifest.metadata.get("run_id") != expected_run_id
        or manifest.metadata.get("selection_id") != document["selection_id"]
        or manifest.metadata.get("selection_payload_sha256") != payload_hash
    ):
        raise Stage4SelectionError("existing selection artifact has another identity")
    summary = document["summary"]
    return SelectionSummary(
        selection_id=str(document["selection_id"]),
        run_id=expected_run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        selection_payload_sha256=payload_hash,
        cell_count=int(summary["cell_count"]),
        ensemble_member_count=int(summary["ensemble_member_count"]),
        final_holdout_evaluated=False,
    )


def prepare_stage4_selection(
    *,
    repository_root: str | Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> SelectionSummary:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage4SelectionError("repository root must not be linked or reparse-backed")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise Stage4SelectionError("repository root is not a directory")
    _verify_runner_origin(root)
    _relative_output, output_parent = _safe_output_root(root, output_root)
    code_binding = _code_binding(root)
    loaded_sources = {
        spec.key: _load_source_artifact(root, spec) for spec in SOURCE_ARTIFACTS
    }
    semantic = {
        "policy_id": SELECTION_POLICY_ID,
        "code_binding": code_binding,
        "source_artifacts": [_source_document(spec) for spec in SOURCE_ARTIFACTS],
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "outer_folds": EXPECTED_FOLDS,
    }
    run_id = semantic_sha256(semantic)[:24]
    output = output_parent / f"s4sel-{run_id[:20]}"
    if output.exists():
        return _existing_summary(output, expected_run_id=run_id)
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".s4sel-", dir=output_parent))
    try:
        cells = _build_cells(root, temporary, loaded_sources)
        document = _selection_document(
            code_binding=code_binding,
            sources=loaded_sources,
            cells=cells,
        )
        document["selection_payload_sha256"] = semantic_sha256(document)
        payload_hash = verify_selection_document(document)
        (temporary / "selection.json").write_bytes(
            canonical_json_bytes(document) + b"\n"
        )
        if _code_binding(root) != code_binding:
            raise Stage4SelectionError("selection code changed during preparation")
        for spec in SOURCE_ARTIFACTS:
            if _load_source_artifact(root, spec).manifest.artifact_id != spec.artifact_id:
                raise Stage4SelectionError("development artifact changed during selection")
        manifest = publish_artifact(
            temporary,
            stage_name=SELECTION_STAGE_NAME,
            schema_version=SELECTION_ARTIFACT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": semantic,
                "selection_id": document["selection_id"],
                "selection_payload_sha256": payload_hash,
                "final_holdout_evaluated": False,
            },
        )
        if output.exists():
            raise Stage4SelectionError("selection artifact destination appeared")
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage4SelectionError("published selection artifact failed verification")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return SelectionSummary(
        selection_id=str(document["selection_id"]),
        run_id=run_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        selection_payload_sha256=payload_hash,
        cell_count=len(cells),
        ensemble_member_count=sum(
            int(value["ensemble_member_count"]) for value in cells
        ),
        final_holdout_evaluated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the Stage 4 development-only final selection."
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = prepare_stage4_selection(
            repository_root=args.repository_root,
            output_root=args.output_root,
        )
    except (OSError, TypeError, ValueError, Stage4SelectionError) as exc:
        raise SystemExit(f"Stage 4 selection failed: {exc}") from exc
    print(
        json.dumps(
            {
                **asdict(summary),
                "output_dir": summary.output_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
