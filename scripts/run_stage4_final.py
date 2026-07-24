"""Evaluate the frozen Stage 4 selection on the permanent holdout exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.dataset import (
    DatasetSlice,
    LifecycleSequence,
    PredictionPosition,
    PredictionTarget,
    build_lifecycle_slice,
)
from token_prediction.development import build_development_protocol
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    FittedExpansionCalibrator,
    ScoredForecast,
    evaluate_budget_scenarios,
    evaluate_forecasts,
    evaluate_progress_checkpoints,
    evaluate_same_task_run_variance,
    evaluate_task_forecasts,
    evaluate_termination_strata,
)
from token_prediction.final_ensemble import (
    FINAL_ENSEMBLE_POLICY_ID,
    EmpiricalFoldState,
    canonical_json_bytes,
    ensemble_prediction_maps,
    final_holdout_dataset_id,
    final_task_pseudonym,
    predict_point_rows,
    semantic_sha256,
)
from token_prediction.lifecycle import (
    LifecyclePrediction,
    LifecycleRun,
)
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import publish_artifact, sha256_file, verify_artifact
from token_prediction.stage3_matrix import STAGE3_BUDGET_THRESHOLDS

if __package__:
    from scripts.prepare_stage4_selection import (
        SELECTION_ENSEMBLE_POLICY_ID,
        SELECTION_POLICY_ID,
        SOURCE_ARTIFACTS,
        Stage4SelectionError,
        verify_selection_document,
    )
    from scripts.run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from scripts.run_stage2_experiments import load_stage2_source
else:  # pragma: no cover - production CLI invocation
    from prepare_stage4_selection import (
        SELECTION_ENSEMBLE_POLICY_ID,
        SELECTION_POLICY_ID,
        SOURCE_ARTIFACTS,
        Stage4SelectionError,
        verify_selection_document,
    )
    from run_data_foundation_baseline import (
        DEFAULT_BASELINE_LOCK,
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
        load_lock_context,
    )
    from run_stage2_experiments import load_stage2_source


SELECTION_LOCK_SCHEMA_VERSION = 1
SELECTION_LOCK_POLICY_ID = "stage4_final_selection_lock_v1"
SELECTION_TAG = "stage4-final-selection-v1"
DEFAULT_SELECTION_LOCK = "configs/stage4_selection.json"
FINAL_RESULTS_SCHEMA_VERSION = 1
FINAL_ARTIFACT_SCHEMA_VERSION = 1
FINAL_STAGE_NAME = "stage4_final_holdout"
FINAL_RUN_POLICY_ID = "stage4_single_open_resumable_final_holdout_v1"
FINAL_SCORE_PROJECTION_ID = "stage4_final_scored_projection_v1"
FINAL_COHORT_PROJECTION_ID = "stage4_final_cohort_projection_v1"
FINAL_CHECKPOINT_SCHEMA_VERSION = 1
FINAL_LEDGER_SCHEMA_VERSION = 1
FINAL_RUNNER_RELATIVE = "scripts/run_stage4_final.py"
DEFAULT_OUTPUT_ROOT = "workspace/stage4/final"
ALLOWED_OUTPUT_PREFIX = "workspace/stage4/final/"
DEFAULT_CHECKPOINT_ROOT = "workspace/stage4/final-checkpoints"
ALLOWED_CHECKPOINT_PREFIX = "workspace/stage4/final-checkpoints/"


class Stage4FinalError(RuntimeError):
    """The one-time final holdout evaluation cannot continue safely."""


@dataclass(frozen=True)
class SelectionLockContext:
    path: str
    sha256: str
    document: Mapping[str, Any]
    selection_root: Path
    selection_manifest_id: str
    selection: Mapping[str, Any]
    selection_commit: str


@dataclass(frozen=True)
class FinalSummary:
    run_id: str
    selection_id: str
    output_dir: Path
    artifact_id: str
    results_payload_sha256: str
    cell_count: int
    prediction_count: int
    final_holdout_evaluated: bool


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage4FinalError("JSON document contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Stage4FinalError(f"JSON document contains non-finite value {value}")


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4FinalError(f"{description} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise Stage4FinalError(f"{description} must be an object")
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
        raise Stage4FinalError(f"Git command failed: {message}")
    return completed.stdout


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage4FinalError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_selection_lock_document(value: Mapping[str, Any]) -> None:
    expected = {
        "selection_lock_schema_version",
        "policy_id",
        "selection_tag",
        "selection_artifact",
        "source_artifacts",
        "protocol",
    }
    if set(value) != expected:
        raise Stage4FinalError("selection lock has missing or extra fields")
    if (
        value["selection_lock_schema_version"] != SELECTION_LOCK_SCHEMA_VERSION
        or value["policy_id"] != SELECTION_LOCK_POLICY_ID
        or value["selection_tag"] != SELECTION_TAG
    ):
        raise Stage4FinalError("selection lock policy identity is invalid")
    artifact = value["selection_artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "artifact_id",
        "run_id",
        "selection_id",
        "selection_payload_sha256",
        "selection_code_commit",
        "selection_code_tree_sha256",
    }:
        raise Stage4FinalError("selection artifact lock is invalid")
    for name in (
        "artifact_id",
        "selection_id",
        "selection_payload_sha256",
        "selection_code_tree_sha256",
    ):
        _sha256(artifact[name], name=f"selection artifact {name}")
    code_commit = artifact["selection_code_commit"]
    if (
        not isinstance(code_commit, str)
        or len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
    ):
        raise Stage4FinalError("selection artifact code commit is invalid")
    _safe_relative(artifact["path"], label="selection artifact path")
    sources = value["source_artifacts"]
    expected_sources = [asdict(item) for item in SOURCE_ARTIFACTS]
    if sources != expected_sources:
        raise Stage4FinalError("selection lock source artifacts differ from frozen inventory")
    protocol = value["protocol"]
    if protocol != {
        "selection_policy_id": SELECTION_POLICY_ID,
        "ensemble_policy_id": SELECTION_ENSEMBLE_POLICY_ID,
        "final_holdout_evaluation_count": 1,
        "refit_selected_learned_models": False,
        "calibration_application_count": 1,
        "resume_policy_id": FINAL_RUN_POLICY_ID,
    }:
        raise Stage4FinalError("selection lock final protocol is invalid")


def load_selection_lock(
    root: Path,
    lock_path: str = DEFAULT_SELECTION_LOCK,
    *,
    require_head_at_tag: bool,
) -> SelectionLockContext:
    relative = _safe_relative(lock_path, label="selection lock path")
    path = _repo_path(root, relative, label="selection lock")
    document = _load_json(path, description="selection lock")
    _validate_selection_lock_document(document)
    selection_commit = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{SELECTION_TAG}^{{commit}}",
    ).decode("ascii").strip()
    tagged_lock = _git(root, "show", f"{selection_commit}:{relative}")
    if tagged_lock != path.read_bytes():
        raise Stage4FinalError("selection lock differs from the frozen selection tag")
    if require_head_at_tag:
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        if head != selection_commit:
            raise Stage4FinalError("final holdout may run only at the frozen selection tag")
        status = _git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            relative,
            "src/token_prediction",
            FINAL_RUNNER_RELATIVE,
            "scripts/prepare_stage4_selection.py",
        )
        if status:
            raise Stage4FinalError("final evaluation code or selection lock is dirty")
    artifact_lock = document["selection_artifact"]
    selection_relative = _safe_relative(
        artifact_lock["path"],
        label="selection artifact path",
    )
    selection_root = _repo_path(root, selection_relative, label="selection artifact")
    manifest = verify_artifact(selection_root)
    if (
        manifest.stage_name != "stage4_frozen_selection"
        or manifest.schema_version != 1
        or manifest.artifact_id != artifact_lock["artifact_id"]
        or manifest.metadata.get("run_id") != artifact_lock["run_id"]
        or manifest.metadata.get("selection_id") != artifact_lock["selection_id"]
        or manifest.metadata.get("selection_payload_sha256")
        != artifact_lock["selection_payload_sha256"]
    ):
        raise Stage4FinalError("selection artifact differs from the tracked lock")
    selection = _load_json(
        selection_root / "selection.json",
        description="selection artifact document",
    )
    try:
        payload_hash = verify_selection_document(selection)
    except Stage4SelectionError as exc:
        raise Stage4FinalError("selection artifact document is invalid") from exc
    code = selection.get("code_binding")
    if (
        payload_hash != artifact_lock["selection_payload_sha256"]
        or selection.get("selection_id") != artifact_lock["selection_id"]
        or not isinstance(code, Mapping)
        or code.get("git_commit") != artifact_lock["selection_code_commit"]
        or code.get("code_tree_sha256") != artifact_lock["selection_code_tree_sha256"]
    ):
        raise Stage4FinalError("selection artifact payload differs from the lock")
    for source_spec in SOURCE_ARTIFACTS:
        source_relative = _safe_relative(
            source_spec.path,
            label=f"{source_spec.key} artifact path",
        )
        source_manifest = verify_artifact(
            _repo_path(root, source_relative, label=f"{source_spec.key} artifact")
        )
        if (
            source_manifest.artifact_id != source_spec.artifact_id
            or source_manifest.metadata.get("run_id") != source_spec.run_id
            or source_manifest.metadata.get("results_payload_sha256")
            != source_spec.results_payload_sha256
        ):
            raise Stage4FinalError(
                f"{source_spec.key} development artifact differs from final selection"
            )
    return SelectionLockContext(
        path=relative,
        sha256=sha256_file(path),
        document=document,
        selection_root=selection_root,
        selection_manifest_id=manifest.artifact_id,
        selection=selection,
        selection_commit=selection_commit,
    )


def _safe_workspace_root(
    root: Path,
    raw: str,
    *,
    prefix: str,
    label: str,
) -> tuple[str, Path]:
    relative = _safe_relative(raw, label=label).rstrip("/")
    canonical = f"{relative}/"
    if not canonical.startswith(prefix):
        raise Stage4FinalError(f"{label} is outside its allowed workspace")
    destination = _repo_path(root, relative, label=label)
    if destination.exists() and _is_link_or_reparse(destination):
        raise Stage4FinalError(f"{label} is unsafe")
    return relative, destination


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _ledger_document(
    *,
    run_id: str,
    selection_id: str,
    status: str,
    completed_cells: Sequence[str],
    final_artifact_id: str | None = None,
) -> dict[str, object]:
    if status not in {"started", "published"}:
        raise Stage4FinalError("final ledger status is invalid")
    return {
        "ledger_schema_version": FINAL_LEDGER_SCHEMA_VERSION,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "run_id": run_id,
        "selection_id": selection_id,
        "status": status,
        "completed_cell_ids": sorted(set(completed_cells)),
        "final_artifact_id": final_artifact_id,
    }


def _load_or_create_ledger(
    path: Path,
    *,
    run_id: str,
    selection_id: str,
) -> Mapping[str, Any]:
    if path.exists():
        ledger = _load_json(path, description="final holdout ledger")
        expected_keys = {
            "ledger_schema_version",
            "run_policy_id",
            "run_id",
            "selection_id",
            "status",
            "completed_cell_ids",
            "final_artifact_id",
        }
        if (
            set(ledger) != expected_keys
            or ledger.get("ledger_schema_version") != FINAL_LEDGER_SCHEMA_VERSION
            or ledger.get("run_policy_id") != FINAL_RUN_POLICY_ID
            or ledger.get("run_id") != run_id
            or ledger.get("selection_id") != selection_id
            or ledger.get("status") not in {"started", "published"}
            or not isinstance(ledger.get("completed_cell_ids"), list)
        ):
            raise Stage4FinalError("final holdout ledger identity is invalid")
        return ledger
    ledger = _ledger_document(
        run_id=run_id,
        selection_id=selection_id,
        status="started",
        completed_cells=(),
    )
    _atomic_json(path, ledger)
    return ledger


def _cell_checkpoint_path(checkpoint_root: Path, cell_id: str) -> Path:
    _sha256(cell_id, name="final cell id")
    return checkpoint_root / "cells" / f"{cell_id}.json"


def _verify_cell_checkpoint(
    value: Mapping[str, Any],
    *,
    selection_id: str,
    cell_id: str,
) -> str:
    expected = {
        "checkpoint_schema_version",
        "run_policy_id",
        "selection_id",
        "cell_id",
        "source_name",
        "source_id",
        "condition_id",
        "position",
        "target",
        "candidate_id",
        "candidate_hash",
        "calibrator_id",
        "alpha",
        "final_dataset",
        "model_execution",
        "metrics",
        "task_metrics",
        "diagnostics",
        "prediction_projection_id",
        "prediction_projection_sha256",
        "cohort_projection_id",
        "cohort_projection_sha256",
        "prediction_count",
        "checkpoint_payload_sha256",
    }
    if set(value) != expected:
        raise Stage4FinalError("final cell checkpoint has missing or extra fields")
    if (
        value["checkpoint_schema_version"] != FINAL_CHECKPOINT_SCHEMA_VERSION
        or value["run_policy_id"] != FINAL_RUN_POLICY_ID
        or value["selection_id"] != selection_id
        or value["cell_id"] != cell_id
    ):
        raise Stage4FinalError("final cell checkpoint identity is invalid")
    payload = dict(value)
    declared = payload.pop("checkpoint_payload_sha256")
    actual = semantic_sha256(payload)
    if declared != actual:
        raise Stage4FinalError("final cell checkpoint checksum does not match")
    return actual


def _member_integrity(member: Mapping[str, Any]) -> None:
    declared = member.get("member_sha256")
    payload = dict(member)
    payload.pop("member_sha256", None)
    if declared != semantic_sha256(payload):
        raise Stage4FinalError("selected ensemble member checksum does not match")


def _load_calibrator(root: Path, relative: str) -> FittedExpansionCalibrator:
    safe = _safe_relative(relative, label="selected calibrator path")
    value = _load_json(
        _repo_path(root, safe, label="selected calibrator"),
        description="selected calibrator",
    )
    try:
        return FittedExpansionCalibrator.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise Stage4FinalError("selected calibrator is invalid") from exc


def _verify_member_auxiliary_files(
    root: Path,
    member: Mapping[str, Any],
) -> None:
    for role in ("calibrator", "provenance"):
        relative = _safe_relative(
            member[f"{role}_path"],
            label=f"selected {role} path",
        )
        path = _repo_path(root, relative, label=f"selected {role}")
        if sha256_file(path) != member[f"{role}_sha256"]:
            raise Stage4FinalError(f"selected {role} differs from frozen selection")


def _point_cell_rows(
    dataset: Any,
    *,
    final_tasks: frozenset[str],
    final_dataset_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
    condition_id: str,
) -> DatasetSlice:
    rows = tuple(
        sorted(
            (
                row
                for row in dataset.rows
                if row.eligible
                and row.point.task_id in final_tasks
                and row.point.position == position
                and row.point.target == target
                and row.point.condition_id == condition_id
            ),
            key=lambda row: row.point.point_id,
        )
    )
    if not rows:
        raise Stage4FinalError("selected final point cell is empty")
    eligibility_hash = hashlib.sha256(
        json.dumps(
            [row.point.point_id for row in rows],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return DatasetSlice(
        dataset_id=final_dataset_id,
        position=position,
        target=target,
        condition_id=condition_id,
        rows=rows,
        eligibility_hash=eligibility_hash,
        input_contract_hash=dataset.input_contract_hash,
        dataset_schema_version=dataset.schema_version,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
    )


def _lightgbm_member_predictions(
    root: Path,
    member: Mapping[str, Any],
    rows: DatasetSlice,
) -> Mapping[str, TokenForecast]:
    from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle

    _member_integrity(member)
    bundle_relative = _safe_relative(member["bundle_path"], label="selected bundle path")
    bundle_root = _repo_path(root, bundle_relative, label="selected LightGBM bundle")
    if member["bundle_tree_sha256"] != _directory_projection_sha256(bundle_root):
        raise Stage4FinalError("selected LightGBM bundle tree differs from selection")
    _verify_member_auxiliary_files(root, member)
    calibrator_relative = _safe_relative(
        member["calibrator_path"],
        label="selected calibrator path",
    )
    calibrator_path = _repo_path(root, calibrator_relative, label="selected calibrator")
    if sha256_file(calibrator_path) != member["calibrator_sha256"]:
        raise Stage4FinalError("selected calibrator differs from selection")
    fitted = load_lightgbm_bundle(bundle_root)
    raw = predict_point_rows(
        fitted,
        rows.rows,
        dataset_id=rows.dataset_id,
        input_contract_hash=rows.input_contract_hash,
    )
    calibrator = _load_calibrator(root, calibrator_relative)
    return {point_id: calibrator.transform(forecast) for point_id, forecast in raw.items()}


def _directory_projection_sha256(directory: Path) -> str:
    files: list[dict[str, str]] = []
    if not directory.is_dir() or _is_link_or_reparse(directory):
        raise Stage4FinalError("selected bundle directory is unsafe")
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            if _is_link_or_reparse(path):
                raise Stage4FinalError("selected bundle contains an unsafe directory")
            continue
        if not path.is_file() or _is_link_or_reparse(path):
            raise Stage4FinalError("selected bundle contains an unsafe file")
        relative = path.relative_to(directory).as_posix()
        files.append({"path": relative, "sha256": sha256_file(path)})
    if not files:
        raise Stage4FinalError("selected bundle directory is empty")
    return semantic_sha256(files)


def _empirical_member_predictions(
    selection_root: Path,
    member: Mapping[str, Any],
    rows: DatasetSlice,
) -> Mapping[str, TokenForecast]:
    _member_integrity(member)
    relative = _safe_relative(member["state_path"], label="empirical state path")
    path = _repo_path(selection_root, relative, label="empirical state")
    if sha256_file(path) != member["state_sha256"]:
        raise Stage4FinalError("empirical state differs from frozen selection")
    state = EmpiricalFoldState.load(path)
    return {row.point.point_id: state.predict(row.point) for row in rows.rows}


def _lifecycle_member_runs(
    root: Path,
    member: Mapping[str, Any],
    sequences: Sequence[LifecycleSequence],
) -> tuple[LifecycleRun, ...]:
    _member_integrity(member)
    relative = _safe_relative(member["bundle_path"], label="lifecycle bundle path")
    bundle_root = _repo_path(root, relative, label="selected lifecycle bundle")
    if member["bundle_tree_sha256"] != _directory_projection_sha256(bundle_root):
        raise Stage4FinalError("selected lifecycle bundle tree differs from selection")
    _verify_member_auxiliary_files(root, member)
    loaded = load_lifecycle_bundle(bundle_root)
    training_dataset_id = str(loaded.manifest["dataset_id"])
    adapted = tuple(
        replace(sequence, dataset_id=training_dataset_id) for sequence in sequences
    )
    return loaded.run_calibrated(adapted)


def _ensemble_lifecycle_runs(
    members: Sequence[tuple[LifecycleRun, ...]],
) -> tuple[LifecycleRun, ...]:
    if not members:
        raise Stage4FinalError("lifecycle ensemble has no members")
    run_count = len(members[0])
    if any(len(value) != run_count for value in members[1:]):
        raise Stage4FinalError("lifecycle ensemble run cohorts differ")
    ensembled: list[LifecycleRun] = []
    for run_index in range(run_count):
        runs = [value[run_index] for value in members]
        point_sequences = [
            tuple(item.step.point.point_id for item in run.predictions) for run in runs
        ]
        if any(value != point_sequences[0] for value in point_sequences[1:]):
            raise Stage4FinalError("lifecycle ensemble point order differs")
        predictions = tuple(
            LifecyclePrediction(
                runs[0].predictions[index].step,
                ensemble_prediction_maps(
                    tuple(
                        {run.predictions[index].forecast.point_id: run.predictions[index].forecast}
                        for run in runs
                    )
                )[runs[0].predictions[index].forecast.point_id],
                runs[0].predictions[index].transition,
            )
            for index in range(len(runs[0].predictions))
        )
        ensembled.append(
            LifecycleRun(
                runs[0].sequence,
                "offline",
                runs[0].seed,
                predictions,
            )
        )
    return tuple(ensembled)


def _scored_projection(
    rows: Sequence[ScoredForecast],
) -> tuple[str, str]:
    score_digest = hashlib.sha256(f"{FINAL_SCORE_PROJECTION_ID}\0".encode("ascii"))
    cohort_digest = hashlib.sha256(f"{FINAL_COHORT_PROJECTION_ID}\0".encode("ascii"))
    for row in sorted(rows, key=lambda item: item.forecast.point_id):
        forecast = row.forecast
        score = {
            "point_id": forecast.point_id,
            "task_id": row.task_id,
            "trajectory_id": row.trajectory_id,
            "target": forecast.target.value,
            "target_value": row.target_value,
            "sample_weight": row.sample_weight,
            "lower": forecast.lower,
            "point": forecast.point,
            "upper": forecast.upper,
            "raw_lower": forecast.raw_lower,
            "raw_point": forecast.raw_point,
            "raw_upper": forecast.raw_upper,
        }
        cohort = {
            "point_id": forecast.point_id,
            "task_id": row.task_id,
            "trajectory_id": row.trajectory_id,
            "target": forecast.target.value,
            "sample_weight": row.sample_weight,
        }
        for digest, value in ((score_digest, score), (cohort_digest, cohort)):
            payload = canonical_json_bytes(value)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return score_digest.hexdigest(), cohort_digest.hexdigest()


def _task_metric_projection(
    rows: Sequence[ScoredForecast],
    *,
    alpha: float,
    final_dataset_id: str,
) -> list[dict[str, object]]:
    metrics = evaluate_task_forecasts(rows, alpha=alpha)
    return sorted(
        (
            {
                "task_pseudonym": final_task_pseudonym(
                    task_id,
                    final_dataset_id=final_dataset_id,
                ),
                **value.to_dict(),
            }
            for task_id, value in metrics.items()
        ),
        key=lambda value: str(value["task_pseudonym"]),
    )


def _point_cell_checkpoint(
    root: Path,
    selection_root: Path,
    selection_id: str,
    cell: Mapping[str, Any],
    dataset: Any,
    final_tasks: frozenset[str],
    final_dataset_id: str,
) -> dict[str, object]:
    position = PredictionPosition(str(cell["position"]))
    target = PredictionTarget(str(cell["target"]))
    dataset_slice = _point_cell_rows(
        dataset,
        final_tasks=final_tasks,
        final_dataset_id=final_dataset_id,
        position=position,
        target=target,
        condition_id=str(cell["condition_id"]),
    )
    members = cell["members"]
    prediction_members = []
    for member in members:
        if member["bundle_kind"] == "lightgbm":
            prediction_members.append(
                _lightgbm_member_predictions(root, member, dataset_slice)
            )
        elif member["bundle_kind"] == "empirical_json":
            prediction_members.append(
                _empirical_member_predictions(selection_root, member, dataset_slice)
            )
        else:
            raise Stage4FinalError("point cell uses an unsupported selected bundle")
    predictions = ensemble_prediction_maps(tuple(prediction_members))
    weights = {
        value.row.point.point_id: value.sample_weight
        for value in dataset_slice.weighted_rows()
    }
    scored = tuple(
        ScoredForecast(
            task_id=row.point.task_id,
            trajectory_id=row.point.trajectory_id,
            forecast=predictions[row.point.point_id],
            target_value=float(row.label),
            sample_weight=weights[row.point.point_id],
        )
        for row in dataset_slice.rows
        if row.label is not None
    )
    projection, cohort = _scored_projection(scored)
    checkpoint: dict[str, object] = {
        "checkpoint_schema_version": FINAL_CHECKPOINT_SCHEMA_VERSION,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_id": selection_id,
        "cell_id": cell["cell_id"],
        "source_name": cell["source_name"],
        "source_id": cell["source_id"],
        "condition_id": cell["condition_id"],
        "position": cell["position"],
        "target": cell["target"],
        "candidate_id": cell["candidate_id"],
        "candidate_hash": cell["candidate_hash"],
        "calibrator_id": cell["calibrator_id"],
        "alpha": cell["alpha"],
        "final_dataset": {
            "dataset_id": final_dataset_id,
            "parent_dataset_id": dataset.dataset_id,
            "task_count": len({row.task_id for row in scored}),
            "trajectory_count": len({row.trajectory_id for row in scored}),
            "scored_point_count": len(scored),
        },
        "model_execution": {
            "ensemble_policy_id": FINAL_ENSEMBLE_POLICY_ID,
            "member_count": len(members),
            "member_projection_sha256": semantic_sha256(
                [member["member_sha256"] for member in members]
            ),
            "execution_mode": "strict_loaded_bundle_only",
            "refit": False,
            "calibration_application_count": 1,
        },
        "metrics": evaluate_forecasts(scored, alpha=float(cell["alpha"])),
        "task_metrics": _task_metric_projection(
            scored,
            alpha=float(cell["alpha"]),
            final_dataset_id=final_dataset_id,
        ),
        "diagnostics": {"status": "point_cell"},
        "prediction_projection_id": FINAL_SCORE_PROJECTION_ID,
        "prediction_projection_sha256": projection,
        "cohort_projection_id": FINAL_COHORT_PROJECTION_ID,
        "cohort_projection_sha256": cohort,
        "prediction_count": len(scored),
    }
    checkpoint["checkpoint_payload_sha256"] = semantic_sha256(checkpoint)
    return checkpoint


def _lifecycle_cell_checkpoint(
    root: Path,
    selection_id: str,
    cell: Mapping[str, Any],
    dataset: Any,
    final_tasks: frozenset[str],
    final_dataset_id: str,
) -> dict[str, object]:
    target = PredictionTarget(str(cell["target"]))
    lifecycle = build_lifecycle_slice(
        dataset,
        target=target,
        condition_id=str(cell["condition_id"]),
        task_ids=final_tasks,
        scored_task_ids=final_tasks,
    )
    member_runs = [
        _lifecycle_member_runs(root, member, lifecycle.sequences)
        for member in cell["members"]
    ]
    runs = _ensemble_lifecycle_runs(member_runs)
    scored = tuple(
        ScoredForecast(
            task_id=prediction.step.point.task_id,
            trajectory_id=prediction.step.point.trajectory_id,
            forecast=prediction.forecast,
            target_value=float(prediction.step.label),
            sample_weight=prediction.step.sample_weight,
        )
        for run in runs
        for prediction in run.scored_predictions
        if prediction.step.label is not None
    )
    if not scored:
        raise Stage4FinalError("final lifecycle cell has no scored predictions")
    projection, cohort = _scored_projection(scored)
    checkpoint: dict[str, object] = {
        "checkpoint_schema_version": FINAL_CHECKPOINT_SCHEMA_VERSION,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_id": selection_id,
        "cell_id": cell["cell_id"],
        "source_name": cell["source_name"],
        "source_id": cell["source_id"],
        "condition_id": cell["condition_id"],
        "position": cell["position"],
        "target": cell["target"],
        "candidate_id": cell["candidate_id"],
        "candidate_hash": cell["candidate_hash"],
        "calibrator_id": cell["calibrator_id"],
        "alpha": cell["alpha"],
        "final_dataset": {
            "dataset_id": final_dataset_id,
            "parent_dataset_id": dataset.dataset_id,
            "task_count": len({row.task_id for row in scored}),
            "trajectory_count": len({row.trajectory_id for row in scored}),
            "scored_point_count": len(scored),
            "context_boundary_count": sum(
                len(run.predictions) for run in runs
            ),
            "unscored_context_boundary_count": sum(
                len(run.predictions) - len(run.scored_predictions) for run in runs
            ),
        },
        "model_execution": {
            "ensemble_policy_id": FINAL_ENSEMBLE_POLICY_ID,
            "member_count": len(cell["members"]),
            "member_projection_sha256": semantic_sha256(
                [member["member_sha256"] for member in cell["members"]]
            ),
            "execution_mode": "strict_loaded_calibrated_full_trajectory_only",
            "refit": False,
            "calibration_application_count": 1,
        },
        "metrics": evaluate_forecasts(scored, alpha=float(cell["alpha"])),
        "task_metrics": _task_metric_projection(
            scored,
            alpha=float(cell["alpha"]),
            final_dataset_id=final_dataset_id,
        ),
        "diagnostics": {
            "status": "complete_calibrated_trajectory_replay",
            "run_count": len(runs),
            "progress": evaluate_progress_checkpoints(
                runs,
                alpha=float(cell["alpha"]),
            ),
            "termination": evaluate_termination_strata(
                runs,
                alpha=float(cell["alpha"]),
            ),
            "run_variance": evaluate_same_task_run_variance(runs),
            "budget": evaluate_budget_scenarios(
                scored,
                budgets=STAGE3_BUDGET_THRESHOLDS,
            ),
        },
        "prediction_projection_id": FINAL_SCORE_PROJECTION_ID,
        "prediction_projection_sha256": projection,
        "cohort_projection_id": FINAL_COHORT_PROJECTION_ID,
        "cohort_projection_sha256": cohort,
        "prediction_count": len(scored),
    }
    checkpoint["checkpoint_payload_sha256"] = semantic_sha256(checkpoint)
    return checkpoint


def _source_expected_dataset(
    selection: Mapping[str, Any],
    source_name: str,
) -> tuple[str, str]:
    matches = [
        value
        for value in selection["source_artifacts"]
        if value["source_name"] == source_name
    ]
    if not matches:
        raise Stage4FinalError("selection lacks source dataset evidence")
    derived = {value["derived_dataset_id"] for value in matches}
    protocols = {value["development_protocol_id"] for value in matches}
    if len(derived) != 1 or len(protocols) != 1:
        raise Stage4FinalError("selection source dataset evidence is inconsistent")
    return next(iter(derived)), next(iter(protocols))


def _evaluate_missing_cells(
    root: Path,
    lock: SelectionLockContext,
    checkpoint_root: Path,
    ledger_path: Path,
) -> list[Mapping[str, Any]]:
    selection_id = str(lock.selection["selection_id"])
    cells = list(lock.selection["cells"])
    completed: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        checkpoint_path = _cell_checkpoint_path(checkpoint_root, cell_id)
        if checkpoint_path.exists():
            value = _load_json(
                checkpoint_path,
                description=f"final cell checkpoint {cell_id}",
            )
            _verify_cell_checkpoint(
                value,
                selection_id=selection_id,
                cell_id=cell_id,
            )
            completed[cell_id] = value
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        if cell["cell_id"] not in completed:
            by_source.setdefault(str(cell["source_name"]), []).append(cell)
    for source_name in sorted(by_source):
        lock_context = load_lock_context(root, DEFAULT_BASELINE_LOCK)
        loaded = load_stage2_source(root, lock_context, source_name=source_name)
        protocol = build_development_protocol(loaded.derived_dataset)
        expected_dataset, expected_protocol = _source_expected_dataset(
            lock.selection,
            source_name,
        )
        if (
            loaded.derived_dataset.dataset_id != expected_dataset
            or protocol.protocol_id != expected_protocol
        ):
            raise Stage4FinalError("source dataset changed after final selection")
        final_tasks = protocol.final_holdout_tasks
        final_dataset_id = final_holdout_dataset_id(
            parent_dataset_id=loaded.derived_dataset.dataset_id,
            holdout_plan_id=protocol.holdout_plan.holdout_plan_id,
            task_ids=final_tasks,
        )
        for cell in sorted(by_source[source_name], key=lambda value: value["cell_id"]):
            if cell["selected_artifact_key"] == "stage3_spend_openhands":
                checkpoint = _lifecycle_cell_checkpoint(
                    root,
                    selection_id,
                    cell,
                    loaded.derived_dataset,
                    final_tasks,
                    final_dataset_id,
                )
            else:
                checkpoint = _point_cell_checkpoint(
                    root,
                    lock.selection_root,
                    selection_id,
                    cell,
                    loaded.derived_dataset,
                    final_tasks,
                    final_dataset_id,
                )
            checkpoint_path = _cell_checkpoint_path(
                checkpoint_root,
                str(cell["cell_id"]),
            )
            _atomic_json(checkpoint_path, checkpoint)
            reloaded = _load_json(
                checkpoint_path,
                description="new final cell checkpoint",
            )
            _verify_cell_checkpoint(
                reloaded,
                selection_id=selection_id,
                cell_id=str(cell["cell_id"]),
            )
            completed[str(cell["cell_id"])] = reloaded
            _atomic_json(
                ledger_path,
                _ledger_document(
                    run_id=checkpoint_root.name,
                    selection_id=selection_id,
                    status="started",
                    completed_cells=tuple(completed),
                ),
            )
    if set(completed) != {str(cell["cell_id"]) for cell in cells}:
        raise Stage4FinalError("final holdout checkpoints do not cover every selected cell")
    return [completed[str(cell["cell_id"])] for cell in cells]


def _results_document(
    *,
    run_id: str,
    lock: SelectionLockContext,
    cells: Sequence[Mapping[str, Any]],
    selection_code_binding: Mapping[str, object],
) -> dict[str, object]:
    prediction_count = sum(int(value["prediction_count"]) for value in cells)
    datasets: dict[str, dict[str, object]] = {}
    for value in cells:
        source_name = str(value["source_name"])
        final_dataset = value["final_dataset"]
        current = datasets.setdefault(
            source_name,
            {
                "source_name": source_name,
                "source_id": value["source_id"],
                "dataset_id": final_dataset["dataset_id"],
                "parent_dataset_id": final_dataset["parent_dataset_id"],
                "task_count": final_dataset["task_count"],
            },
        )
        if (
            current["dataset_id"] != final_dataset["dataset_id"]
            or current["parent_dataset_id"] != final_dataset["parent_dataset_id"]
        ):
            raise Stage4FinalError("final source cells use inconsistent holdout datasets")
    base: dict[str, object] = {
        "results_schema_version": FINAL_RESULTS_SCHEMA_VERSION,
        "stage_name": FINAL_STAGE_NAME,
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "run_id": run_id,
        "selection": {
            "selection_id": lock.selection["selection_id"],
            "selection_artifact_id": lock.selection_manifest_id,
            "selection_payload_sha256": lock.selection[
                "selection_payload_sha256"
            ],
            "selection_lock_path": lock.path,
            "selection_lock_sha256": lock.sha256,
            "selection_tag": SELECTION_TAG,
            "selection_commit": lock.selection_commit,
        },
        "evaluation_code_binding": dict(selection_code_binding),
        "datasets": [datasets[key] for key in sorted(datasets)],
        "cells": list(cells),
        "summary": {
            "source_count": len(datasets),
            "cell_count": len(cells),
            "ensemble_member_count": sum(
                int(value["model_execution"]["member_count"]) for value in cells
            ),
            "prediction_count": prediction_count,
        },
        "final_holdout": {
            "evaluated": True,
            "evaluation_count": 1,
            "prediction_count": prediction_count,
            "target_values_used_for_fit": False,
            "target_values_used_for_calibration": False,
            "target_values_used_for_scoring": True,
            "model_selection_after_open": False,
        },
    }
    base["results_payload_sha256"] = semantic_sha256(base)
    return base


def verify_final_results_document(value: Mapping[str, Any]) -> str:
    expected = {
        "results_schema_version",
        "stage_name",
        "run_policy_id",
        "run_id",
        "selection",
        "evaluation_code_binding",
        "datasets",
        "cells",
        "summary",
        "final_holdout",
        "results_payload_sha256",
    }
    if set(value) != expected:
        raise Stage4FinalError("final results have missing or extra fields")
    if (
        value["results_schema_version"] != FINAL_RESULTS_SCHEMA_VERSION
        or value["stage_name"] != FINAL_STAGE_NAME
        or value["run_policy_id"] != FINAL_RUN_POLICY_ID
    ):
        raise Stage4FinalError("final results policy identity is invalid")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != 29:
        raise Stage4FinalError("final results must contain 29 cells")
    prediction_count = 0
    ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise Stage4FinalError("final result cell is invalid")
        cell_id = str(cell.get("cell_id", ""))
        if cell_id in ids:
            raise Stage4FinalError("final results repeat a cell")
        ids.add(cell_id)
        _verify_cell_checkpoint(
            cell,
            selection_id=str(value["selection"]["selection_id"]),
            cell_id=cell_id,
        )
        prediction_count += int(cell["prediction_count"])
    summary = value["summary"]
    if (
        not isinstance(summary, Mapping)
        or summary.get("source_count") != 4
        or summary.get("cell_count") != 29
        or summary.get("ensemble_member_count") != 435
        or summary.get("prediction_count") != prediction_count
    ):
        raise Stage4FinalError("final result summary does not close over cells")
    if value["final_holdout"] != {
        "evaluated": True,
        "evaluation_count": 1,
        "prediction_count": prediction_count,
        "target_values_used_for_fit": False,
        "target_values_used_for_calibration": False,
        "target_values_used_for_scoring": True,
        "model_selection_after_open": False,
    }:
        raise Stage4FinalError("final holdout protocol is invalid")
    payload = dict(value)
    declared = payload.pop("results_payload_sha256")
    actual = semantic_sha256(payload)
    if declared != actual:
        raise Stage4FinalError("final results checksum does not match")
    return actual


def _evaluation_code_binding(root: Path) -> Mapping[str, object]:
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    raw = _git(
        root,
        "ls-files",
        "-z",
        "--",
        "src/token_prediction",
        FINAL_RUNNER_RELATIVE,
        "scripts/prepare_stage4_selection.py",
        DEFAULT_SELECTION_LOCK,
    )
    paths = tuple(
        sorted(
            item.decode("utf-8", errors="strict")
            for item in raw.split(b"\0")
            if item
        )
    )
    required = {
        FINAL_RUNNER_RELATIVE,
        "scripts/prepare_stage4_selection.py",
        DEFAULT_SELECTION_LOCK,
    }
    if not required <= set(paths):
        raise Stage4FinalError("final evaluation code closure is incomplete")
    digest = hashlib.sha256(b"stage4-final-evaluation-code-tree-v1\0")
    for relative in paths:
        path = _repo_path(root, relative, label="final evaluation code path")
        payload = path.read_bytes()
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise Stage4FinalError("final evaluation workspace differs from HEAD")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "policy_id": "stage4_final_evaluation_code_tree_v1",
        "git_commit": commit,
        "code_tree_sha256": digest.hexdigest(),
        "paths": list(paths),
    }


def _existing_final_summary(
    output: Path,
    *,
    run_id: str,
    selection_id: str,
) -> FinalSummary:
    manifest = verify_artifact(output)
    results = _load_json(output / "results.json", description="final results")
    payload_hash = verify_final_results_document(results)
    if (
        manifest.stage_name != FINAL_STAGE_NAME
        or manifest.schema_version != FINAL_ARTIFACT_SCHEMA_VERSION
        or manifest.metadata.get("run_id") != run_id
        or manifest.metadata.get("selection_id") != selection_id
        or manifest.metadata.get("results_payload_sha256") != payload_hash
    ):
        raise Stage4FinalError("existing final artifact has another identity")
    return FinalSummary(
        run_id=run_id,
        selection_id=selection_id,
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=payload_hash,
        cell_count=int(results["summary"]["cell_count"]),
        prediction_count=int(results["summary"]["prediction_count"]),
        final_holdout_evaluated=True,
    )


def run_stage4_final(
    *,
    repository_root: str | Path,
    selection_lock: str = DEFAULT_SELECTION_LOCK,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
) -> FinalSummary:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage4FinalError("repository root must not be linked or reparse-backed")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise Stage4FinalError("repository root is not a directory")
    expected_runner = _repo_path(root, FINAL_RUNNER_RELATIVE, label="final runner")
    if Path(__file__).resolve() != expected_runner.resolve() or _is_link_or_reparse(
        Path(__file__)
    ):
        raise Stage4FinalError("executing final runner is outside repository_root")
    _output_relative, output_parent = _safe_workspace_root(
        root,
        output_root,
        prefix=ALLOWED_OUTPUT_PREFIX,
        label="final output root",
    )
    _checkpoint_relative, checkpoint_parent = _safe_workspace_root(
        root,
        checkpoint_root,
        prefix=ALLOWED_CHECKPOINT_PREFIX,
        label="final checkpoint root",
    )
    lock = load_selection_lock(
        root,
        selection_lock,
        require_head_at_tag=True,
    )
    code_binding = _evaluation_code_binding(root)
    run_semantic = {
        "run_policy_id": FINAL_RUN_POLICY_ID,
        "selection_id": lock.selection["selection_id"],
        "selection_artifact_id": lock.selection_manifest_id,
        "selection_lock_sha256": lock.sha256,
        "selection_commit": lock.selection_commit,
        "evaluation_code_binding": code_binding,
    }
    run_id = semantic_sha256(run_semantic)[:24]
    output = output_parent / f"s4final-{run_id[:20]}"
    checkpoint = checkpoint_parent / run_id
    ledger_path = checkpoint / "ledger.json"
    if output.exists():
        return _existing_final_summary(
            output,
            run_id=run_id,
            selection_id=str(lock.selection["selection_id"]),
        )
    checkpoint.mkdir(parents=True, exist_ok=True)
    ledger = _load_or_create_ledger(
        ledger_path,
        run_id=run_id,
        selection_id=str(lock.selection["selection_id"]),
    )
    if ledger["status"] == "published":
        raise Stage4FinalError("ledger says published but final artifact is missing")
    started = time.time()
    cell_results = _evaluate_missing_cells(root, lock, checkpoint, ledger_path)
    results = _results_document(
        run_id=run_id,
        lock=lock,
        cells=cell_results,
        selection_code_binding=code_binding,
    )
    payload_hash = verify_final_results_document(results)
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".s4final-", dir=output_parent))
    try:
        (temporary / "results.json").write_bytes(
            canonical_json_bytes(results) + b"\n"
        )
        (temporary / "selection-lock.json").write_bytes(
            canonical_json_bytes(lock.document) + b"\n"
        )
        manifest = publish_artifact(
            temporary,
            stage_name=FINAL_STAGE_NAME,
            schema_version=FINAL_ARTIFACT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "selection_id": lock.selection["selection_id"],
                "results_payload_sha256": payload_hash,
                "final_holdout_evaluated": True,
                "evaluation_count": 1,
                "elapsed_seconds": time.time() - started,
            },
        )
        if output.exists():
            raise Stage4FinalError("final artifact destination appeared")
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage4FinalError("published final artifact failed verification")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _atomic_json(
        ledger_path,
        _ledger_document(
            run_id=run_id,
            selection_id=str(lock.selection["selection_id"]),
            status="published",
            completed_cells=[str(value["cell_id"]) for value in cell_results],
            final_artifact_id=manifest.artifact_id,
        ),
    )
    return FinalSummary(
        run_id=run_id,
        selection_id=str(lock.selection["selection_id"]),
        output_dir=output,
        artifact_id=manifest.artifact_id,
        results_payload_sha256=payload_hash,
        cell_count=len(cell_results),
        prediction_count=int(results["summary"]["prediction_count"]),
        final_holdout_evaluated=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single frozen Stage 4 final-holdout evaluation."
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--selection-lock", default=DEFAULT_SELECTION_LOCK)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_stage4_final(
            repository_root=args.repository_root,
            selection_lock=args.selection_lock,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
        )
    except (OSError, TypeError, ValueError, Stage4FinalError) as exc:
        raise SystemExit(f"Stage 4 final evaluation failed: {exc}") from exc
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
