"""Freeze the Stage 4 development-only completion supplement.

This command deliberately accepts only aggregate development artifacts below
``workspace/stage4/runs`` plus one completion-diagnostics artifact.  It never
opens the permanent final holdout artifact, its ledger, source data, or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.lineage import (
    ArtifactVerificationError,
    sha256_file,
    verify_artifact,
)

if __package__:
    from scripts.run_stage4_experiments import (
        Stage4ExperimentError,
        _framed_code_hash,
        verify_stage4_results_document,
    )
    from scripts.verify_stage4_release import (
        Stage4ReleaseError,
        _validate_release_document as _validate_parent_release_document,
    )
else:  # pragma: no cover - production CLI invocation
    from run_stage4_experiments import (
        Stage4ExperimentError,
        _framed_code_hash,
        verify_stage4_results_document,
    )
    from verify_stage4_release import (
        Stage4ReleaseError,
        _validate_release_document as _validate_parent_release_document,
    )


ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCHEMA_VERSION = 1
RELEASE_STAGE_NAME = "stage4_development_completion_supplement"
RELEASE_POLICY_ID = "stage4_development_only_completion_release_v1"
SOURCE_BINDING_POLICY_ID = "stage4_completion_source_code_tree_v1"
EXPECTED_SOURCE_TAG = "stage4-completion-source-v1"
EXPECTED_DIAGNOSTICS_SOURCE_TAG = "stage4-completion-diagnostics-source-v1"
DIAGNOSTICS_RUNNER_RELATIVE = "scripts/run_stage4_completion_diagnostics.py"
EXPECTED_SOURCE_COMMIT = "c1ac2484f44ed65705cdd00eba7b70a739a3ac0b"
EXPECTED_CODE_TREE_SHA256 = (
    "6418545afa08a39df1797486e4c845063c2de13b29f20c81500933fad2201757"
)
EXPECTED_MATRIX_SCHEMA_VERSION = 2
EXPECTED_MATRIX_POLICY_ID = "stage4_single_axis_condition_position_target_matrix_v2"
EXPECTED_SPLIT_SEEDS = (20260719, 20260720, 20260721)
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 5
EXPECTED_FINAL_HOLDOUT = {
    "evaluated": False,
    "prediction_count": 0,
    "target_values_used_for_fit_calibration_scoring": False,
    "selection_claim": "none",
}
EXPECTED_PARENT_LOCK = "configs/stage4_release.json"
EXPECTED_PARENT_STAGE_NAME = "stage4_final_holdout"
EXPECTED_PARENT_POLICY_ID = "stage4_commit_bound_single_final_holdout_release_v1"
EXPECTED_PARENT_TAG = "stage4-final-release-v1"
EXPECTED_PARENT_FINAL_EVALUATION_COUNT = 1
EXPECTED_PARENT_FINAL_PREDICTION_COUNT = 86_335
EXPECTED_REPORT = "docs/stage-4-completion-supplement.md"
DEFAULT_OUTPUT = "configs/stage4_completion_release.json"
DEVELOPMENT_RUNS_PREFIX = PurePosixPath("workspace/stage4/runs")
MLP_CANDIDATE_ID = "mlp_history"
RAW_SEED_CANDIDATE_ID = "cross_position_deduct_raw_repaired_oof_seed"
POINT_ONLY_SEED_CANDIDATE_ID = "cross_position_deduct_point_only_oof_seed"
EXPECTED_TOTAL_EXPERIMENT_COUNT = 52
EXPECTED_TOTAL_CANDIDATE_SEED_RUN_COUNT = 477
EXPECTED_TOTAL_RELOADABLE_BUNDLE_FOLD_COUNT = 1_950
EXPECTED_CALL_PRE_MLP_CELL_COUNT = 21
EXPECTED_CALL_PRE_MLP_BUNDLE_FOLD_COUNT = 315
EXPECTED_SEED_POLICY_CELL_COUNT = 7
EXPECTED_SEED_POLICY_BUNDLE_FOLD_COUNT = 210
DIAGNOSTICS_PREFIX = PurePosixPath("workspace/stage4/completion_diagnostics")
DIAGNOSTICS_RESULTS_SCHEMA_VERSION = 1
DIAGNOSTICS_STAGE_NAME = "stage4_completion_diagnostics"
DIAGNOSTICS_POLICY_ID = "stage4_completion_development_lifecycle_replay_v1"
EXPECTED_DIAGNOSTICS_BOUND_SOURCE_COUNT = 4
EXPECTED_DIAGNOSTICS_LIFECYCLE_SOURCE_COUNT = 3
EXPECTED_DIAGNOSTICS_LIFECYCLE_CONDITION_COUNT = 7
EXPECTED_DIAGNOSTICS_CANDIDATE_COUNT = 2
EXPECTED_DIAGNOSTICS_CANDIDATE_CELL_COUNT = 14
EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT = 42
EXPECTED_DIAGNOSTICS_BUNDLE_COUNT = 210
RUN_VARIANCE_ID = "same_task_run_mae_variance_v1"
RUN_DISPERSION_EXTENSION_ID = "same_task_run_mae_iqr_max_minus_min_v1"
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SourceExpectation:
    source_name: str
    source_id: str
    experiment_count: int
    candidate_seed_run_count: int
    reloadable_bundle_fold_count: int


SOURCE_EXPECTATIONS = (
    SourceExpectation(
        "spend_aggregate",
        "spend_your_money_aggregate_v1",
        3,
        15,
        60,
    ),
    SourceExpectation(
        "bagen_sokoban",
        "bagen_sokoban_dialogues_v1",
        7,
        66,
        270,
    ),
    SourceExpectation(
        "bagen_swebench",
        "bagen_swebench_traj_v2",
        35,
        330,
        1_350,
    ),
    SourceExpectation(
        "spend_openhands",
        "openhands_archive_trajectory_v3",
        7,
        66,
        270,
    ),
)


class Stage4CompletionFreezeError(RuntimeError):
    """The completion supplement cannot be frozen safely."""


@dataclass(frozen=True)
class ArtifactAudit:
    record: Mapping[str, object]
    reloadable_bundle_fold_count: int
    call_pre_mlp_cell_count: int
    call_pre_mlp_bundle_fold_count: int
    seed_policy_cell_count: int
    seed_policy_bundle_fold_count: int


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage4CompletionFreezeError("JSON document contains duplicate keys")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise Stage4CompletionFreezeError(f"JSON document contains non-finite value {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage4CompletionFreezeError(
            "release metadata is not finite canonical JSON"
        ) from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def _regular_bytes(path: Path, *, maximum_bytes: int, description: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise Stage4CompletionFreezeError(f"{description} cannot be inspected") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise Stage4CompletionFreezeError(
            f"{description} must be a regular non-link file"
        )
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise Stage4CompletionFreezeError(f"{description} exceeds its size limit")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise Stage4CompletionFreezeError(f"{description} cannot be read") from exc
    if (
        len(payload) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise Stage4CompletionFreezeError(f"{description} changed while being read")
    return payload


def _json_object(payload: bytes, *, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage4CompletionFreezeError(
            f"{description} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise Stage4CompletionFreezeError(f"{description} must be a JSON object")
    return value


def _mapping(value: object, *, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise Stage4CompletionFreezeError(f"{description} must be a JSON object")
    return value


def _list(value: object, *, description: str) -> list[object]:
    if not isinstance(value, list):
        raise Stage4CompletionFreezeError(f"{description} must be a JSON array")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage4CompletionFreezeError(f"{description} must be a non-empty string")
    return value


def _sha256(value: object, *, description: str) -> str:
    text = _text(value, description=description)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise Stage4CompletionFreezeError(
            f"{description} must be a lowercase SHA-256"
        )
    return text


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage4CompletionFreezeError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _safe_relative(value: str | os.PathLike[str], *, description: str) -> str:
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\\" in text:
        raise Stage4CompletionFreezeError(
            f"{description} must be a forward-slash repository-relative path"
        )
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise Stage4CompletionFreezeError(
            f"{description} must be a safe repository-relative path"
        )
    return path.as_posix()


def _bound_repo_path(
    root: Path,
    value: str | os.PathLike[str],
    *,
    description: str,
    expected_prefix: PurePosixPath | None = None,
) -> tuple[str, Path]:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise Stage4CompletionFreezeError(
            f"{description} must exist within the repository"
        ) from exc
    canonical = PurePosixPath(*relative.parts)
    if expected_prefix is not None and (
        canonical == expected_prefix
        or not canonical.is_relative_to(expected_prefix)
    ):
        raise Stage4CompletionFreezeError(
            f"{description} must be below {expected_prefix.as_posix()}"
        )
    return canonical.as_posix(), resolved


def _tag_commit(root: Path, tag: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage4CompletionFreezeError(
            f"source tag cannot be resolved: {message}"
        )
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise Stage4CompletionFreezeError("source tag commit is not ASCII") from exc


def _git_file(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage4CompletionFreezeError(
            f"diagnostics code path is absent from its source commit: {message}"
        )
    return completed.stdout


def _commit(value: object, *, description: str) -> str:
    text = _text(value, description=description)
    if (
        len(text) != 40
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise Stage4CompletionFreezeError(
            f"{description} must be a lowercase full Git commit"
        )
    return text


def _audit_diagnostics_code_binding(
    root: Path,
    value: object,
    *,
    source_tag: str,
) -> Mapping[str, object]:
    binding = _mapping(value, description="completion diagnostics code binding")
    if set(binding) != {"git_commit", "code_tree_sha256", "code_paths"}:
        raise Stage4CompletionFreezeError(
            "completion diagnostics code binding keys differ"
        )
    if source_tag != EXPECTED_DIAGNOSTICS_SOURCE_TAG:
        raise Stage4CompletionFreezeError(
            "completion diagnostics source tag identity differs"
        )
    git_commit = _commit(
        binding.get("git_commit"),
        description="completion diagnostics code commit",
    )
    code_tree_sha256 = _sha256(
        binding.get("code_tree_sha256"),
        description="completion diagnostics code tree",
    )
    raw_paths = _list(
        binding.get("code_paths"),
        description="completion diagnostics code paths",
    )
    paths = [
        _safe_relative(
            _text(path, description="completion diagnostics code path"),
            description="completion diagnostics code path",
        )
        for path in raw_paths
    ]
    required = {
        DIAGNOSTICS_RUNNER_RELATIVE,
        "src/token_prediction/evaluation/stratification.py",
        "src/token_prediction/lifecycle_bundle.py",
        "src/token_prediction/lifecycle.py",
    }
    if paths != sorted(set(paths)) or not required <= set(paths):
        raise Stage4CompletionFreezeError(
            "completion diagnostics code closure is incomplete or unordered"
        )
    if _tag_commit(root, source_tag) != git_commit:
        raise Stage4CompletionFreezeError(
            "completion diagnostics source tag points elsewhere"
        )
    items = [(relative, _git_file(root, git_commit, relative)) for relative in paths]
    if _framed_code_hash(items) != code_tree_sha256:
        raise Stage4CompletionFreezeError(
            "completion diagnostics code tree does not close"
        )
    return {
        "git_commit": git_commit,
        "code_tree_sha256": code_tree_sha256,
        "code_paths": paths,
        "source_tag": source_tag,
    }


def _candidate_seed_audit(
    experiments: Sequence[object],
) -> tuple[int, int, int, int, int, int]:
    candidate_seed_count = 0
    reloadable_bundle_fold_count = 0
    call_pre_mlp_cell_count = 0
    call_pre_mlp_bundle_fold_count = 0
    seed_policy_cell_count = 0
    seed_policy_bundle_fold_count = 0
    experiment_ids: set[str] = set()
    for experiment_index, raw_experiment in enumerate(experiments):
        experiment = _mapping(
            raw_experiment,
            description=f"experiments[{experiment_index}]",
        )
        experiment_id = _text(
            experiment.get("experiment_id"),
            description=f"experiments[{experiment_index}].experiment_id",
        )
        if experiment_id in experiment_ids:
            raise Stage4CompletionFreezeError("experiment ids must be unique")
        experiment_ids.add(experiment_id)
        candidates = _list(
            experiment.get("candidates"),
            description=f"{experiment_id}.candidates",
        )
        candidate_ids: set[str] = set()
        candidate_bundle_counts: dict[str, int] = {}
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _mapping(
                raw_candidate,
                description=f"{experiment_id}.candidates[{candidate_index}]",
            )
            candidate_id = _text(
                candidate.get("candidate_id"),
                description=f"{experiment_id} candidate id",
            )
            if candidate_id in candidate_ids:
                raise Stage4CompletionFreezeError(
                    f"{experiment_id} candidate ids must be unique"
                )
            candidate_ids.add(candidate_id)
            seed_results = _list(
                candidate.get("seed_results"),
                description=f"{experiment_id}/{candidate_id}.seed_results",
            )
            observed_seeds: list[int] = []
            candidate_fold_count = 0
            for seed_index, raw_seed in enumerate(seed_results):
                seed = _mapping(
                    raw_seed,
                    description=(
                        f"{experiment_id}/{candidate_id}.seed_results[{seed_index}]"
                    ),
                )
                observed_seeds.append(
                    _integer(
                        seed.get("split_seed"),
                        description=f"{experiment_id}/{candidate_id} split seed",
                    )
                )
                folds = _list(
                    seed.get("reloadable_bundle_folds"),
                    description=(
                        f"{experiment_id}/{candidate_id} reloadable bundle folds"
                    ),
                )
                fold_artifact_count = _integer(
                    seed.get("fold_artifact_count"),
                    description=f"{experiment_id}/{candidate_id} fold artifact count",
                )
                if fold_artifact_count != len(folds):
                    raise Stage4CompletionFreezeError(
                        f"{experiment_id}/{candidate_id} fold count differs"
                    )
                if len(folds) not in {0, EXPECTED_OUTER_FOLDS}:
                    raise Stage4CompletionFreezeError(
                        f"{experiment_id}/{candidate_id} bundle folds are incomplete"
                    )
                candidate_fold_count += len(folds)
            if tuple(observed_seeds) != EXPECTED_SPLIT_SEEDS:
                raise Stage4CompletionFreezeError(
                    f"{experiment_id}/{candidate_id} split seeds differ"
                )
            candidate_seed_count += len(seed_results)
            candidate_bundle_counts[candidate_id] = candidate_fold_count
            reloadable_bundle_fold_count += candidate_fold_count

        if (
            experiment.get("position") == "call_pre"
            and MLP_CANDIDATE_ID in candidate_ids
        ):
            call_pre_mlp_cell_count += 1
            call_pre_mlp_bundle_fold_count += candidate_bundle_counts[
                MLP_CANDIDATE_ID
            ]
        if {
            RAW_SEED_CANDIDATE_ID,
            POINT_ONLY_SEED_CANDIDATE_ID,
        } <= candidate_ids:
            seed_policy_cell_count += 1
            seed_policy_bundle_fold_count += (
                candidate_bundle_counts[RAW_SEED_CANDIDATE_ID]
                + candidate_bundle_counts[POINT_ONLY_SEED_CANDIDATE_ID]
            )
    return (
        candidate_seed_count,
        reloadable_bundle_fold_count,
        call_pre_mlp_cell_count,
        call_pre_mlp_bundle_fold_count,
        seed_policy_cell_count,
        seed_policy_bundle_fold_count,
    )


def _audit_training_artifact(
    root: Path,
    raw_path: str | os.PathLike[str],
    expectation: SourceExpectation,
) -> ArtifactAudit:
    relative, artifact_root = _bound_repo_path(
        root,
        raw_path,
        description=f"{expectation.source_name} artifact",
        expected_prefix=DEVELOPMENT_RUNS_PREFIX,
    )
    try:
        manifest = verify_artifact(artifact_root)
    except (ArtifactVerificationError, OSError) as exc:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} artifact verification failed"
        ) from exc
    if manifest.stage_name != "stage4_development_source":
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} artifact stage is invalid"
        )
    if not manifest.files:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} artifact manifest is empty"
        )
    results = _json_object(
        _regular_bytes(
            artifact_root / "results.json",
            maximum_bytes=MAX_JSON_BYTES,
            description=f"{expectation.source_name} results",
        ),
        description=f"{expectation.source_name} results",
    )
    try:
        results_payload_sha256 = verify_stage4_results_document(results)
    except (Stage4ExperimentError, TypeError, ValueError) as exc:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} results verification failed"
        ) from exc
    if results.get("final_holdout") != EXPECTED_FINAL_HOLDOUT:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} artifact is not development-only"
        )
    source = _mapping(
        results.get("source"),
        description=f"{expectation.source_name} source",
    )
    if (
        source.get("source_name") != expectation.source_name
        or source.get("source_id") != expectation.source_id
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} source identity differs"
        )
    code = _mapping(
        results.get("code_binding"),
        description=f"{expectation.source_name} code binding",
    )
    if (
        code.get("git_commit") != EXPECTED_SOURCE_COMMIT
        or code.get("code_tree_sha256") != EXPECTED_CODE_TREE_SHA256
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} code binding differs from c1ac248"
        )
    matrix = _mapping(
        results.get("matrix"),
        description=f"{expectation.source_name} matrix",
    )
    expected_matrix_keys = {
        "schema_version",
        "policy_id",
        "source_id",
        "capability_contract_hash",
        "development_protocol_id",
        "plans",
        "gates",
        "telemetry_decisions",
        "safety_invariants",
        "matrix_id",
    }
    if set(matrix) != expected_matrix_keys:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} matrix keys differ"
        )
    if (
        matrix.get("schema_version") != EXPECTED_MATRIX_SCHEMA_VERSION
        or matrix.get("policy_id") != EXPECTED_MATRIX_POLICY_ID
        or matrix.get("source_id") != expectation.source_id
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} matrix policy identity differs"
        )
    matrix_id = _sha256(
        matrix.get("matrix_id"),
        description=f"{expectation.source_name} matrix id",
    )
    matrix_semantic = dict(matrix)
    matrix_semantic.pop("matrix_id")
    if _semantic_sha256(matrix_semantic) != matrix_id:
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} matrix id does not close"
        )
    plans = _list(
        matrix.get("plans"),
        description=f"{expectation.source_name} matrix plans",
    )
    experiments = _list(
        results.get("experiments"),
        description=f"{expectation.source_name} experiments",
    )
    if len(plans) != expectation.experiment_count or len(experiments) != len(plans):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} experiment matrix count differs"
        )
    plan_ids = [
        _text(
            _mapping(plan, description="matrix plan").get("spec", {}).get(
                "experiment_id"
            )
            if isinstance(_mapping(plan, description="matrix plan").get("spec"), Mapping)
            else None,
            description=f"{expectation.source_name} matrix experiment id",
        )
        for plan in plans
    ]
    experiment_ids = [
        _text(
            _mapping(experiment, description="experiment").get("experiment_id"),
            description=f"{expectation.source_name} result experiment id",
        )
        for experiment in experiments
    ]
    if len(set(plan_ids)) != len(plan_ids) or set(plan_ids) != set(experiment_ids):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} matrix/result experiments differ"
        )
    (
        candidate_seed_run_count,
        reloadable_bundle_fold_count,
        call_pre_mlp_cell_count,
        call_pre_mlp_bundle_fold_count,
        seed_policy_cell_count,
        seed_policy_bundle_fold_count,
    ) = _candidate_seed_audit(experiments)
    summary = _mapping(
        results.get("summary"),
        description=f"{expectation.source_name} summary",
    )
    if (
        summary.get("experiment_count") != expectation.experiment_count
        or summary.get("candidate_seed_run_count")
        != expectation.candidate_seed_run_count
        or summary.get("split_seeds") != list(EXPECTED_SPLIT_SEEDS)
        or summary.get("outer_folds") != EXPECTED_OUTER_FOLDS
        or summary.get("inner_folds") != EXPECTED_INNER_FOLDS
        or candidate_seed_run_count != expectation.candidate_seed_run_count
        or reloadable_bundle_fold_count
        != expectation.reloadable_bundle_fold_count
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} summary or bundle count differs"
        )
    run_id = _text(
        results.get("run_id"),
        description=f"{expectation.source_name} run id",
    )
    if (
        len(run_id) < 20
        or any(character not in "0123456789abcdef" for character in run_id)
        or relative.rsplit("/", 1)[-1] != f"s4-{run_id[:20]}"
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} artifact path and run id differ"
        )
    if (
        manifest.metadata.get("run_id") != run_id
        or manifest.metadata.get("results_payload_sha256")
        != results_payload_sha256
    ):
        raise Stage4CompletionFreezeError(
            f"{expectation.source_name} manifest/results metadata differs"
        )
    record = {
        "source_name": expectation.source_name,
        "source_id": expectation.source_id,
        "path": relative,
        "artifact_id": manifest.artifact_id,
        "run_id": run_id,
        "results_payload_sha256": results_payload_sha256,
        "manifest_sha256": sha256_file(artifact_root / "manifest.json"),
        "matrix_id": matrix_id,
        "experiment_count": expectation.experiment_count,
        "candidate_seed_run_count": expectation.candidate_seed_run_count,
        "manifest_file_count": len(manifest.files),
    }
    return ArtifactAudit(
        record=record,
        reloadable_bundle_fold_count=reloadable_bundle_fold_count,
        call_pre_mlp_cell_count=call_pre_mlp_cell_count,
        call_pre_mlp_bundle_fold_count=call_pre_mlp_bundle_fold_count,
        seed_policy_cell_count=seed_policy_cell_count,
        seed_policy_bundle_fold_count=seed_policy_bundle_fold_count,
    )


def _diagnostics_coverage(value: object) -> Mapping[str, object]:
    coverage = _mapping(value, description="completion diagnostics coverage")
    expected_keys = {
        "bound_source_artifact_count",
        "lifecycle_source_count",
        "lifecycle_condition_count",
        "lifecycle_candidate_count",
        "lifecycle_candidate_cell_count",
        "lifecycle_candidate_seed_count",
        "lifecycle_bundle_count",
        "replayed_run_count",
        "scored_run_count",
        "scored_boundary_count",
        "unscored_context_boundary_count",
    }
    if set(coverage) != expected_keys:
        raise Stage4CompletionFreezeError("diagnostics coverage keys differ")
    fixed = {
        "bound_source_artifact_count": EXPECTED_DIAGNOSTICS_BOUND_SOURCE_COUNT,
        "lifecycle_source_count": EXPECTED_DIAGNOSTICS_LIFECYCLE_SOURCE_COUNT,
        "lifecycle_condition_count": (
            EXPECTED_DIAGNOSTICS_LIFECYCLE_CONDITION_COUNT
        ),
        "lifecycle_candidate_count": EXPECTED_DIAGNOSTICS_CANDIDATE_COUNT,
        "lifecycle_candidate_cell_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_CELL_COUNT
        ),
        "lifecycle_candidate_seed_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
        ),
        "lifecycle_bundle_count": EXPECTED_DIAGNOSTICS_BUNDLE_COUNT,
    }
    if any(coverage.get(key) != expected for key, expected in fixed.items()):
        raise Stage4CompletionFreezeError(
            "diagnostics fixed coverage differs from the completion protocol"
        )
    replayed = _integer(
        coverage.get("replayed_run_count"),
        description="diagnostics replayed run count",
        minimum=1,
    )
    scored = _integer(
        coverage.get("scored_run_count"),
        description="diagnostics scored run count",
        minimum=1,
    )
    if scored > replayed:
        raise Stage4CompletionFreezeError(
            "diagnostics scored run count exceeds replayed run count"
        )
    _integer(
        coverage.get("scored_boundary_count"),
        description="diagnostics scored boundary count",
        minimum=1,
    )
    _integer(
        coverage.get("unscored_context_boundary_count"),
        description="diagnostics unscored context boundary count",
    )
    return dict(coverage)


def _verify_run_dispersion(value: object) -> int:
    dispersion = _mapping(value, description="completion run dispersion")
    required = {
        "run_variance_id",
        "run_dispersion_extension_id",
        "n_tasks",
        "n_scored_runs",
        "n_repeated_tasks",
        "status",
        "mean_within_task_run_mae_variance",
        "median_within_task_run_mae_variance",
        "max_within_task_run_mae_variance",
        "mean_within_task_run_mae_iqr",
        "median_within_task_run_mae_iqr",
        "max_within_task_run_mae_iqr",
        "mean_within_task_run_mae_max_minus_min",
        "median_within_task_run_mae_max_minus_min",
        "max_within_task_run_mae_max_minus_min",
    }
    if set(dispersion) != required:
        raise Stage4CompletionFreezeError("completion run dispersion keys differ")
    if (
        dispersion.get("run_variance_id") != RUN_VARIANCE_ID
        or dispersion.get("run_dispersion_extension_id")
        != RUN_DISPERSION_EXTENSION_ID
        or dispersion.get("status") not in {"estimable", "not_estimable"}
    ):
        raise Stage4CompletionFreezeError(
            "completion run dispersion extension is missing or invalid"
        )
    for key in ("n_tasks", "n_scored_runs", "n_repeated_tasks"):
        _integer(
            dispersion.get(key),
            description=f"completion run dispersion {key}",
        )
    for key in required - {
        "run_variance_id",
        "run_dispersion_extension_id",
        "n_tasks",
        "n_scored_runs",
        "n_repeated_tasks",
        "status",
    }:
        number = dispersion.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0
        ):
            raise Stage4CompletionFreezeError(
                f"completion run dispersion {key} is invalid"
            )
    return int(dispersion["n_scored_runs"])


def _audit_diagnostics_artifact(
    root: Path,
    raw_path: str | os.PathLike[str],
    *,
    source_artifact_ids: Mapping[str, str],
    diagnostics_source_tag: str,
) -> Mapping[str, object]:
    relative, artifact_root = _bound_repo_path(
        root,
        raw_path,
        description="completion diagnostics artifact",
        expected_prefix=DIAGNOSTICS_PREFIX,
    )
    if not relative.startswith(f"{DIAGNOSTICS_PREFIX.as_posix()}/s4diag-"):
        raise Stage4CompletionFreezeError(
            "completion diagnostics path does not use its immutable run prefix"
        )
    try:
        manifest = verify_artifact(artifact_root)
    except (ArtifactVerificationError, OSError) as exc:
        raise Stage4CompletionFreezeError(
            "completion diagnostics artifact verification failed"
        ) from exc
    if (
        manifest.stage_name != DIAGNOSTICS_STAGE_NAME
        or manifest.schema_version != DIAGNOSTICS_RESULTS_SCHEMA_VERSION
        or not manifest.files
    ):
        raise Stage4CompletionFreezeError(
            "completion diagnostics manifest identity differs"
        )
    results = _json_object(
        _regular_bytes(
            artifact_root / "results.json",
            maximum_bytes=MAX_JSON_BYTES,
            description="completion diagnostics results",
        ),
        description="completion diagnostics results",
    )
    expected_result_keys = {
        "results_schema_version",
        "stage_name",
        "policy_id",
        "source_binding",
        "diagnostics_code_binding",
        "source_artifacts",
        "coverage",
        "bundle_inventory",
        "diagnostics",
        "final_holdout",
        "results_payload_sha256",
    }
    if set(results) != expected_result_keys:
        raise Stage4CompletionFreezeError(
            "completion diagnostics results keys differ"
        )
    if (
        results.get("results_schema_version")
        != DIAGNOSTICS_RESULTS_SCHEMA_VERSION
        or results.get("stage_name") != DIAGNOSTICS_STAGE_NAME
        or results.get("policy_id") != DIAGNOSTICS_POLICY_ID
        or results.get("final_holdout") != EXPECTED_FINAL_HOLDOUT
    ):
        raise Stage4CompletionFreezeError(
            "completion diagnostics opened final data or has another identity"
        )
    declared = _sha256(
        results.get("results_payload_sha256"),
        description="completion diagnostics results payload",
    )
    semantic = dict(results)
    semantic.pop("results_payload_sha256")
    if _semantic_sha256(semantic) != declared:
        raise Stage4CompletionFreezeError(
            "completion diagnostics results payload does not close"
        )
    training_source_binding = _mapping(
        results.get("source_binding"),
        description="completion diagnostics source binding",
    )
    if set(training_source_binding) != {"git_commit", "code_tree_sha256"} or dict(
        training_source_binding
    ) != {
        "git_commit": EXPECTED_SOURCE_COMMIT,
        "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    }:
        raise Stage4CompletionFreezeError(
            "completion diagnostics source binding differs from c1ac248"
        )
    diagnostics_code_binding = _audit_diagnostics_code_binding(
        root,
        results.get("diagnostics_code_binding"),
        source_tag=diagnostics_source_tag,
    )
    source_documents = _list(
        results.get("source_artifacts"),
        description="completion diagnostics source artifacts",
    )
    observed_source_names: list[str] = []
    observed_source_ids: dict[str, str] = {}
    for index, raw_source in enumerate(source_documents):
        source = _mapping(
            raw_source,
            description=f"completion diagnostics source_artifacts[{index}]",
        )
        source_name = _text(
            source.get("source_name"),
            description="completion diagnostics source name",
        )
        artifact_id = _sha256(
            source.get("artifact_id"),
            description=f"completion diagnostics {source_name} artifact id",
        )
        observed_source_names.append(source_name)
        observed_source_ids[source_name] = artifact_id
    expected_names = sorted(item.source_name for item in SOURCE_EXPECTATIONS)
    if (
        observed_source_names != expected_names
        or observed_source_ids != dict(source_artifact_ids)
    ):
        raise Stage4CompletionFreezeError(
            "completion diagnostics do not bind the four training artifacts"
        )
    coverage = _diagnostics_coverage(results.get("coverage"))
    inventory = _list(
        results.get("bundle_inventory"),
        description="completion diagnostics bundle inventory",
    )
    diagnostics = _list(
        results.get("diagnostics"),
        description="completion diagnostics records",
    )
    if (
        len(inventory) != EXPECTED_DIAGNOSTICS_BUNDLE_COUNT
        or len(diagnostics) != EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
    ):
        raise Stage4CompletionFreezeError(
            "completion diagnostics inventory or record count differs"
        )
    scored_run_count = 0
    for index, raw_record in enumerate(diagnostics):
        record = _mapping(
            raw_record,
            description=f"completion diagnostics records[{index}]",
        )
        scored_run_count += _verify_run_dispersion(record.get("run_variance"))
    if scored_run_count != coverage["scored_run_count"]:
        raise Stage4CompletionFreezeError(
            "completion run dispersion scored-run coverage does not close"
        )
    expected_metadata_keys = {
        "run_id",
        "run_semantic",
        "results_payload_sha256",
        "source_git_commit",
        "source_code_tree_sha256",
        "diagnostics_code_binding",
        "source_artifact_ids",
        "coverage",
        "diagnostics_runner_sha256",
    }
    result_diagnostics_code = {
        key: diagnostics_code_binding[key]
        for key in ("git_commit", "code_tree_sha256", "code_paths")
    }
    run_id = _text(
        manifest.metadata.get("run_id"),
        description="completion diagnostics run id",
    )
    run_semantic = _mapping(
        manifest.metadata.get("run_semantic"),
        description="completion diagnostics run semantic",
    )
    runner_sha256 = _sha256(
        manifest.metadata.get("diagnostics_runner_sha256"),
        description="completion diagnostics runner SHA-256",
    )
    if (
        set(manifest.metadata) != expected_metadata_keys
        or len(run_id) < 20
        or any(character not in "0123456789abcdef" for character in run_id)
        or relative.rsplit("/", 1)[-1] != f"s4diag-{run_id[:20]}"
        or manifest.metadata.get("results_payload_sha256") != declared
        or manifest.metadata.get("source_git_commit") != EXPECTED_SOURCE_COMMIT
        or manifest.metadata.get("source_code_tree_sha256")
        != EXPECTED_CODE_TREE_SHA256
        or manifest.metadata.get("diagnostics_code_binding")
        != result_diagnostics_code
        or manifest.metadata.get("source_artifact_ids")
        != [source_artifact_ids[name] for name in expected_names]
        or manifest.metadata.get("coverage") != dict(coverage)
        or run_semantic.get("results_schema_version")
        != DIAGNOSTICS_RESULTS_SCHEMA_VERSION
        or run_semantic.get("policy_id") != DIAGNOSTICS_POLICY_ID
        or run_semantic.get("source_binding") != dict(training_source_binding)
        or run_semantic.get("diagnostics_code_binding")
        != result_diagnostics_code
        or run_semantic.get("diagnostics_runner_sha256") != runner_sha256
        or run_semantic.get("final_holdout") != EXPECTED_FINAL_HOLDOUT
    ):
        raise Stage4CompletionFreezeError(
            "completion diagnostics manifest metadata differs"
        )
    return {
        "path": relative,
        "artifact_id": manifest.artifact_id,
        "manifest_sha256": sha256_file(artifact_root / "manifest.json"),
        "results_payload_sha256": declared,
        "training_source_commit": EXPECTED_SOURCE_COMMIT,
        "diagnostics_code_binding": dict(diagnostics_code_binding),
        "source_artifact_ids": dict(source_artifact_ids),
        "coverage": dict(coverage),
    }


def _protocol_with_diagnostics(
    training_protocol: Mapping[str, object],
    diagnostics_artifact: Mapping[str, object],
) -> Mapping[str, object]:
    coverage = _diagnostics_coverage(diagnostics_artifact.get("coverage"))
    return {
        **dict(training_protocol),
        "diagnostics_artifact_count": 1,
        "diagnostics_bound_source_artifact_count": coverage[
            "bound_source_artifact_count"
        ],
        "diagnostics_lifecycle_source_count": coverage[
            "lifecycle_source_count"
        ],
        "diagnostics_lifecycle_condition_count": coverage[
            "lifecycle_condition_count"
        ],
        "diagnostics_candidate_count": coverage["lifecycle_candidate_count"],
        "diagnostics_candidate_cell_count": coverage[
            "lifecycle_candidate_cell_count"
        ],
        "diagnostics_candidate_seed_count": coverage[
            "lifecycle_candidate_seed_count"
        ],
        "diagnostics_bundle_count": coverage["lifecycle_bundle_count"],
    }


def _parent_release_binding(
    root: Path,
    raw_path: str | os.PathLike[str],
) -> Mapping[str, object]:
    relative, path = _bound_repo_path(
        root,
        raw_path,
        description="parent Stage 4 release lock",
    )
    if relative != EXPECTED_PARENT_LOCK:
        raise Stage4CompletionFreezeError("parent Stage 4 release lock path differs")
    payload = _regular_bytes(
        path,
        maximum_bytes=MAX_JSON_BYTES,
        description="parent Stage 4 release lock",
    )
    document = _json_object(payload, description="parent Stage 4 release lock")
    try:
        _validate_parent_release_document(document)
    except Stage4ReleaseError as exc:
        raise Stage4CompletionFreezeError(
            "parent Stage 4 release lock is invalid"
        ) from exc
    final_artifact = _mapping(
        document.get("final_artifact"),
        description="parent final artifact",
    )
    protocol = _mapping(
        document.get("protocol"),
        description="parent final protocol",
    )
    remote = _mapping(
        document.get("remote_controls"),
        description="parent final remote controls",
    )
    if (
        document.get("stage_name") != EXPECTED_PARENT_STAGE_NAME
        or document.get("policy_id") != EXPECTED_PARENT_POLICY_ID
        or protocol.get("final_holdout_evaluation_count")
        != EXPECTED_PARENT_FINAL_EVALUATION_COUNT
        or protocol.get("final_holdout_prediction_count")
        != EXPECTED_PARENT_FINAL_PREDICTION_COUNT
        or remote.get("final_release_tag") != EXPECTED_PARENT_TAG
    ):
        raise Stage4CompletionFreezeError("parent Stage 4 release identity differs")
    return {
        "lock_path": relative,
        "lock_sha256": hashlib.sha256(payload).hexdigest(),
        "final_release_tag": EXPECTED_PARENT_TAG,
        "final_artifact_id": _sha256(
            final_artifact.get("artifact_id"),
            description="parent final artifact id",
        ),
        "final_holdout_evaluation_count": EXPECTED_PARENT_FINAL_EVALUATION_COUNT,
        "final_holdout_prediction_count": EXPECTED_PARENT_FINAL_PREDICTION_COUNT,
    }


def _report_binding(
    root: Path,
    raw_path: str | os.PathLike[str],
) -> Mapping[str, object]:
    relative, path = _bound_repo_path(
        root,
        raw_path,
        description="completion supplement report",
    )
    if relative != EXPECTED_REPORT:
        raise Stage4CompletionFreezeError("completion report path differs")
    payload = _regular_bytes(
        path,
        maximum_bytes=MAX_REPORT_BYTES,
        description="completion supplement report",
    )
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}


def _training_protocol(audits: Sequence[ArtifactAudit]) -> Mapping[str, object]:
    protocol = {
        "development_only": True,
        "artifact_count": len(audits),
        "experiment_count": sum(
            int(audit.record["experiment_count"]) for audit in audits
        ),
        "candidate_seed_run_count": sum(
            int(audit.record["candidate_seed_run_count"]) for audit in audits
        ),
        "reloadable_bundle_fold_count": sum(
            audit.reloadable_bundle_fold_count for audit in audits
        ),
        "call_pre_mlp_cell_count": sum(
            audit.call_pre_mlp_cell_count for audit in audits
        ),
        "call_pre_mlp_bundle_fold_count": sum(
            audit.call_pre_mlp_bundle_fold_count for audit in audits
        ),
        "seed_policy_cell_count": sum(
            audit.seed_policy_cell_count for audit in audits
        ),
        "seed_policy_bundle_fold_count": sum(
            audit.seed_policy_bundle_fold_count for audit in audits
        ),
        "split_seeds": list(EXPECTED_SPLIT_SEEDS),
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "inner_folds": EXPECTED_INNER_FOLDS,
        "final_holdout_evaluated": False,
        "final_holdout_prediction_count": 0,
        "final_holdout_target_values_used_for_fit_calibration_scoring": False,
        "final_holdout_selection_claim": "none",
    }
    expected = {
        **protocol,
        "artifact_count": 4,
        "experiment_count": EXPECTED_TOTAL_EXPERIMENT_COUNT,
        "candidate_seed_run_count": EXPECTED_TOTAL_CANDIDATE_SEED_RUN_COUNT,
        "reloadable_bundle_fold_count": (
            EXPECTED_TOTAL_RELOADABLE_BUNDLE_FOLD_COUNT
        ),
        "call_pre_mlp_cell_count": EXPECTED_CALL_PRE_MLP_CELL_COUNT,
        "call_pre_mlp_bundle_fold_count": (
            EXPECTED_CALL_PRE_MLP_BUNDLE_FOLD_COUNT
        ),
        "seed_policy_cell_count": EXPECTED_SEED_POLICY_CELL_COUNT,
        "seed_policy_bundle_fold_count": EXPECTED_SEED_POLICY_BUNDLE_FOLD_COUNT,
    }
    if protocol != expected:
        raise Stage4CompletionFreezeError(
            "completion training artifact totals differ from the frozen matrix"
        )
    return protocol


def build_release_document(
    *,
    source_tag: str,
    source_tag_commit: str,
    parent_final_release: Mapping[str, object],
    artifacts: Sequence[ArtifactAudit],
    diagnostics_artifact: Mapping[str, object],
    protocol: Mapping[str, object],
    report: Mapping[str, object],
) -> Mapping[str, object]:
    """Build the deterministic lock after every external input was audited."""

    if source_tag != EXPECTED_SOURCE_TAG or source_tag_commit != EXPECTED_SOURCE_COMMIT:
        raise Stage4CompletionFreezeError("completion source tag identity differs")
    return {
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "stage_name": RELEASE_STAGE_NAME,
        "policy_id": RELEASE_POLICY_ID,
        "source_binding": {
            "policy_id": SOURCE_BINDING_POLICY_ID,
            "git_commit": EXPECTED_SOURCE_COMMIT,
            "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
            "source_tag": EXPECTED_SOURCE_TAG,
        },
        "parent_final_release": dict(parent_final_release),
        "artifacts": [dict(audit.record) for audit in artifacts],
        "diagnostics_artifact": dict(diagnostics_artifact),
        "protocol": dict(protocol),
        "report": dict(report),
    }


def _write_release(path: Path, document: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        existing = _regular_bytes(
            path,
            maximum_bytes=MAX_JSON_BYTES,
            description="existing completion release lock",
        )
        if existing != payload:
            raise Stage4CompletionFreezeError(
                "existing completion release lock is immutable and differs"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def freeze_completion_release(
    *,
    repository_root: str | os.PathLike[str],
    artifact_paths: Sequence[str | os.PathLike[str]],
    diagnostics_artifact_path: str | os.PathLike[str],
    source_tag: str,
    diagnostics_source_tag: str,
    parent_release_path: str | os.PathLike[str] = EXPECTED_PARENT_LOCK,
    report_path: str | os.PathLike[str] = EXPECTED_REPORT,
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT,
) -> Mapping[str, object]:
    """Audit all development-only inputs and write one immutable release lock."""

    root = Path(repository_root)
    if len(artifact_paths) != len(SOURCE_EXPECTATIONS):
        raise Stage4CompletionFreezeError("exactly four training artifacts are required")
    tag_commit = _tag_commit(root, source_tag)
    if tag_commit != EXPECTED_SOURCE_COMMIT:
        raise Stage4CompletionFreezeError("source tag does not point to c1ac248")
    audits = tuple(
        _audit_training_artifact(root, raw_path, expectation)
        for raw_path, expectation in zip(
            artifact_paths,
            SOURCE_EXPECTATIONS,
            strict=True,
        )
    )
    artifact_paths_seen = [str(audit.record["path"]) for audit in audits]
    artifact_ids_seen = [str(audit.record["artifact_id"]) for audit in audits]
    if (
        len(set(artifact_paths_seen)) != len(artifact_paths_seen)
        or len(set(artifact_ids_seen)) != len(artifact_ids_seen)
    ):
        raise Stage4CompletionFreezeError(
            "training artifact paths and ids must be unique"
        )
    protocol = _training_protocol(audits)
    source_artifact_ids = {
        str(audit.record["source_name"]): str(audit.record["artifact_id"])
        for audit in audits
    }
    diagnostics_artifact = _audit_diagnostics_artifact(
        root,
        diagnostics_artifact_path,
        source_artifact_ids=source_artifact_ids,
        diagnostics_source_tag=diagnostics_source_tag,
    )
    protocol = _protocol_with_diagnostics(protocol, diagnostics_artifact)
    document = build_release_document(
        source_tag=source_tag,
        source_tag_commit=tag_commit,
        parent_final_release=_parent_release_binding(root, parent_release_path),
        artifacts=audits,
        diagnostics_artifact=diagnostics_artifact,
        protocol=protocol,
        report=_report_binding(root, report_path),
    )
    output_relative = _safe_relative(output_path, description="completion lock output")
    output = root.joinpath(*PurePosixPath(output_relative).parts)
    _write_release(output, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs=4,
        metavar="ARTIFACT",
        help=(
            "four development run directories in canonical source order: "
            "spend_aggregate, bagen_sokoban, bagen_swebench, spend_openhands"
        ),
    )
    parser.add_argument(
        "--diagnostics-artifact",
        required=True,
        help="development-only completion diagnostics artifact",
    )
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--diagnostics-source-tag", required=True)
    parser.add_argument("--parent-release", default=EXPECTED_PARENT_LOCK)
    parser.add_argument("--report", default=EXPECTED_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--repository-root", default=str(ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    document = freeze_completion_release(
        repository_root=arguments.repository_root,
        artifact_paths=arguments.artifacts,
        diagnostics_artifact_path=arguments.diagnostics_artifact,
        source_tag=arguments.source_tag,
        diagnostics_source_tag=arguments.diagnostics_source_tag,
        parent_release_path=arguments.parent_release,
        report_path=arguments.report,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "output": arguments.output,
                "artifact_count": len(document["artifacts"]),
                "diagnostics_artifact_id": document["diagnostics_artifact"][
                    "artifact_id"
                ],
                "final_holdout_evaluated": document["protocol"][
                    "final_holdout_evaluated"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
