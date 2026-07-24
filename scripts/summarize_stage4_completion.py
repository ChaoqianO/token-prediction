"""Summarize Stage 4 completion experiments without opening the final holdout.

The command accepts exactly four Stage 4 development artifact directories by
default.  A future completion release lock can be supplied instead when it
contains a ``development_artifacts`` (or
``supplementary_development_artifacts``) list of safe repository-relative run
paths.

Only ``manifest.json``, ``results.json``, and ``_SUCCESS`` are read from each
development artifact.  Fold artifacts, source data, final artifacts, and final
labels are deliberately outside this tool's input surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from token_prediction.lineage import verify_artifact


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_RUNS_ROOT = ROOT / "workspace" / "stage4" / "runs"
DIAGNOSTICS_ROOT = ROOT / "workspace" / "stage4" / "completion_diagnostics"
DEVELOPMENT_STAGE_NAME = "stage4_development_source"
RESULTS_STAGE_NAME = DEVELOPMENT_STAGE_NAME
SUMMARY_SCHEMA_VERSION = 2
SUMMARY_POLICY_ID = "stage4_completion_development_only_summary_v1"
FINAL_HOLDOUT_POLICY_ID = "never_open_final_holdout_or_source_labels_v1"
REPLACEMENT_RULE_ID = (
    "all_seed_paired_bootstrap_mae_delta_ci_upper_below_zero_v1"
)
RAW_SEED_CANDIDATE_ID = "cross_position_deduct_raw_repaired_oof_seed"
POINT_ONLY_SEED_CANDIDATE_ID = "cross_position_deduct_point_only_oof_seed"
MLP_CANDIDATE_ID = "mlp_history"
LIGHTGBM_CANDIDATE_ID = "lightgbm_history"
RAW_SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_quantile_mean_v1"
POINT_ONLY_SEED_POLICY_ID = (
    "inner_oof_uncalibrated_repaired_point_only_mean_v1"
)
FROZEN_SPLIT_SEEDS = (20260719, 20260720, 20260721)
MAX_RESULTS_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
LOCK_ARTIFACT_KEYS = (
    "development_artifacts",
    "supplementary_development_artifacts",
)
INTERVAL_RESERVE_FIELDS = (
    "interval_diagnostics_id",
    "interval_below_truth_rate",
    "interval_above_truth_rate",
    "target_exceeds_upper_rate",
    "mean_extra_reserved_tokens",
    "raw_interval_below_truth_rate",
    "raw_interval_above_truth_rate",
    "raw_target_exceeds_upper_rate",
    "raw_mean_extra_reserved_tokens",
)
RUN_DISPERSION_FIELDS = (
    "run_dispersion_extension_id",
    "mean_within_task_run_mae_iqr",
    "median_within_task_run_mae_iqr",
    "max_within_task_run_mae_iqr",
    "mean_within_task_run_mae_max_minus_min",
    "median_within_task_run_mae_max_minus_min",
    "max_within_task_run_mae_max_minus_min",
)
EXPECTED_FINAL_HOLDOUT = {
    "evaluated": False,
    "prediction_count": 0,
    "target_values_used_for_fit_calibration_scoring": False,
    "selection_claim": "none",
}
EXPECTED_RESULTS_KEYS = {
    "results_schema_version",
    "stage_name",
    "run_policy_id",
    "artifact_layout_id",
    "checkpoint_policy_id",
    "run_id",
    "source",
    "data_foundation",
    "code_binding",
    "runtime_versions",
    "dataset",
    "development_protocol",
    "matrix",
    "experiments",
    "matched_coverage_calibration",
    "paired_same_task_across_conditions",
    "summary",
    "final_holdout",
    "results_payload_sha256",
}


class CompletionSummaryError(RuntimeError):
    """Completion artifacts cannot be summarized without violating the contract."""


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    expected_artifact_id: str | None = None
    expected_results_payload_sha256: str | None = None


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    artifact_id: str
    results_payload_sha256: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class LoadedDiagnosticsArtifact:
    path: Path
    artifact_id: str
    results_payload_sha256: str
    document: Mapping[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise CompletionSummaryError("metadata is not finite canonical JSON") from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompletionSummaryError(f"{name} must be a non-empty string")
    return value


def _required_sha256(value: object, *, name: str) -> str:
    text = _required_string(value, name=name)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise CompletionSummaryError(f"{name} must be a lowercase SHA-256")
    return text


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CompletionSummaryError(f"{name} must be a JSON object")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CompletionSummaryError(f"{name} must be a JSON array")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompletionSummaryError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CompletionSummaryError(f"{name} must be finite")
    return number


def _read_regular_bytes(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    if path.is_symlink():
        raise CompletionSummaryError(f"{name} cannot be a symlink")
    try:
        before = path.stat()
    except OSError as exc:
        raise CompletionSummaryError(f"{name} cannot be inspected") from exc
    if not path.is_file():
        raise CompletionSummaryError(f"{name} must be a regular file")
    if before.st_size > maximum_bytes:
        raise CompletionSummaryError(f"{name} exceeds the size limit")
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise CompletionSummaryError(f"{name} cannot be read") from exc
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise CompletionSummaryError(f"{name} changed while being read")
    return payload


def _json_object(payload: bytes, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionSummaryError(f"{name} is not valid UTF-8 JSON") from exc
    return _mapping(value, name=name)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _development_artifact_directory(
    path: Path,
    *,
    development_runs_root: Path,
) -> Path:
    candidate = path.parent if path.name == "results.json" else path
    try:
        resolved_root = development_runs_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CompletionSummaryError("development artifact path does not exist") from exc
    if resolved == resolved_root or not _is_within(resolved, resolved_root):
        raise CompletionSummaryError(
            "artifact must be below the Stage 4 development runs root; "
            "final artifacts are never opened"
        )
    if not resolved.is_dir():
        raise CompletionSummaryError("development artifact path must be a directory")
    return resolved


def _safe_lock_artifact_path(value: object, *, repo_root: Path) -> Path:
    text = _required_string(value, name="release-lock artifact path")
    if "\\" in text:
        raise CompletionSummaryError(
            "release-lock artifact paths must use forward slashes"
        )
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise CompletionSummaryError(
            "release-lock artifact paths must be safe repository-relative paths"
        )
    expected_prefix = PurePosixPath("workspace/stage4/runs")
    if pure == expected_prefix or not pure.is_relative_to(expected_prefix):
        raise CompletionSummaryError(
            "release-lock artifacts must be below workspace/stage4/runs"
        )
    return repo_root.joinpath(*pure.parts)


def _release_lock_references(
    path: Path,
    *,
    repo_root: Path,
) -> list[ArtifactReference]:
    document = _json_object(
        _read_regular_bytes(
            path,
            maximum_bytes=MAX_METADATA_BYTES,
            name="completion release lock",
        ),
        name="completion release lock",
    )
    present = [key for key in LOCK_ARTIFACT_KEYS if key in document]
    if len(present) != 1:
        raise CompletionSummaryError(
            "completion release lock must contain exactly one development artifact list"
        )
    entries = _list(document[present[0]], name=present[0])
    if not entries:
        raise CompletionSummaryError("completion release lock has no development artifacts")
    references: list[ArtifactReference] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            relative = entry
            expected_artifact_id = None
            expected_results = None
        else:
            item = _mapping(entry, name=f"{present[0]}[{index}]")
            relative = item.get("path")
            expected_artifact_id = (
                _required_sha256(
                    item["artifact_id"],
                    name=f"{present[0]}[{index}].artifact_id",
                )
                if "artifact_id" in item
                else None
            )
            expected_results = (
                _required_sha256(
                    item["results_payload_sha256"],
                    name=f"{present[0]}[{index}].results_payload_sha256",
                )
                if "results_payload_sha256" in item
                else None
            )
        references.append(
            ArtifactReference(
                path=_safe_lock_artifact_path(relative, repo_root=repo_root),
                expected_artifact_id=expected_artifact_id,
                expected_results_payload_sha256=expected_results,
            )
        )
    return references


def resolve_artifact_references(
    inputs: Sequence[str | os.PathLike[str]],
    *,
    repo_root: Path = ROOT,
    development_runs_root: Path | None = None,
) -> tuple[ArtifactReference, ...]:
    """Resolve direct run directories or an explicit completion release lock."""

    if not inputs:
        raise CompletionSummaryError("at least one input is required")
    runs_root = development_runs_root or repo_root / "workspace" / "stage4" / "runs"
    references: list[ArtifactReference] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_dir() or path.name == "results.json":
            references.append(
                ArtifactReference(
                    _development_artifact_directory(
                        path,
                        development_runs_root=runs_root,
                    )
                )
            )
            continue
        if path.suffix.casefold() != ".json":
            raise CompletionSummaryError(
                "inputs must be development run directories, results.json files, "
                "or completion release locks"
            )
        references.extend(_release_lock_references(path, repo_root=repo_root))

    resolved: list[ArtifactReference] = []
    seen: set[Path] = set()
    for reference in references:
        directory = _development_artifact_directory(
            reference.path,
            development_runs_root=runs_root,
        )
        if directory in seen:
            raise CompletionSummaryError(
                f"duplicate development artifact input: {directory}"
            )
        seen.add(directory)
        resolved.append(
            ArtifactReference(
                directory,
                reference.expected_artifact_id,
                reference.expected_results_payload_sha256,
            )
        )
    return tuple(resolved)


def load_development_artifact(reference: ArtifactReference) -> LoadedArtifact:
    """Load only aggregate-safe metadata from one development artifact."""

    manifest_payload = _read_regular_bytes(
        reference.path / "manifest.json",
        maximum_bytes=MAX_METADATA_BYTES,
        name="development artifact manifest",
    )
    manifest = _json_object(manifest_payload, name="development artifact manifest")
    if set(manifest) != {
        "artifact_id",
        "stage_name",
        "schema_version",
        "files",
        "metadata",
    }:
        raise CompletionSummaryError("development artifact manifest keys differ")
    if manifest["stage_name"] != DEVELOPMENT_STAGE_NAME:
        raise CompletionSummaryError(
            "only Stage 4 development source artifacts may be summarized"
        )
    artifact_id = _required_sha256(
        manifest["artifact_id"], name="development artifact id"
    )
    files = _mapping(manifest["files"], name="development artifact files")
    results_file_hash = _required_sha256(
        files.get("results.json"), name="manifest results.json SHA-256"
    )
    metadata = _mapping(manifest["metadata"], name="development artifact metadata")
    manifest_semantic = {
        "stage_name": manifest["stage_name"],
        "schema_version": manifest["schema_version"],
        "files": dict(files),
        "metadata": dict(metadata),
    }
    if _semantic_sha256(manifest_semantic) != artifact_id:
        raise CompletionSummaryError("development artifact semantic id does not close")

    success = _read_regular_bytes(
        reference.path / "_SUCCESS",
        maximum_bytes=256,
        name="development artifact success marker",
    )
    try:
        success_id = success.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CompletionSummaryError("success marker is not ASCII") from exc
    if success_id != artifact_id:
        raise CompletionSummaryError("success marker does not match artifact id")

    results_payload = _read_regular_bytes(
        reference.path / "results.json",
        maximum_bytes=MAX_RESULTS_BYTES,
        name="Stage 4 development results",
    )
    if hashlib.sha256(results_payload).hexdigest() != results_file_hash:
        raise CompletionSummaryError("results.json does not match the artifact manifest")
    results = _json_object(results_payload, name="Stage 4 development results")
    if set(results) != EXPECTED_RESULTS_KEYS:
        raise CompletionSummaryError("Stage 4 development results keys differ")
    if results["stage_name"] != RESULTS_STAGE_NAME:
        raise CompletionSummaryError("results are not a Stage 4 development document")
    if results["final_holdout"] != EXPECTED_FINAL_HOLDOUT:
        raise CompletionSummaryError(
            "development results claim final-holdout access; refusing to summarize"
        )
    results_digest = _required_sha256(
        results["results_payload_sha256"],
        name="Stage 4 results payload SHA-256",
    )
    digest_document = dict(results)
    digest_document.pop("results_payload_sha256")
    if _semantic_sha256(digest_document) != results_digest:
        raise CompletionSummaryError("Stage 4 results payload SHA-256 does not close")
    if metadata.get("results_payload_sha256") != results_digest:
        raise CompletionSummaryError(
            "artifact metadata and results payload SHA-256 differ"
        )
    if metadata.get("run_id") != results.get("run_id"):
        raise CompletionSummaryError("artifact metadata and results run id differ")
    if (
        reference.expected_artifact_id is not None
        and reference.expected_artifact_id != artifact_id
    ):
        raise CompletionSummaryError("release lock artifact id differs")
    if (
        reference.expected_results_payload_sha256 is not None
        and reference.expected_results_payload_sha256 != results_digest
    ):
        raise CompletionSummaryError("release lock results payload SHA-256 differs")
    return LoadedArtifact(reference.path, artifact_id, results_digest, results)


def load_completion_diagnostics_artifact(
    path: str | os.PathLike[str],
    *,
    repo_root: Path = ROOT,
    diagnostics_root: Path | None = None,
) -> LoadedDiagnosticsArtifact:
    """Load and fully verify the optional immutable diagnostics supplement."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    root = diagnostics_root or repo_root / "workspace" / "stage4" / (
        "completion_diagnostics"
    )
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CompletionSummaryError(
            "completion diagnostics artifact path does not exist"
        ) from exc
    if resolved == resolved_root or not _is_within(resolved, resolved_root):
        raise CompletionSummaryError(
            "diagnostics supplement must be below the completion diagnostics root"
        )
    try:
        manifest = verify_artifact(resolved)
    except Exception as exc:
        raise CompletionSummaryError(
            "completion diagnostics artifact verification failed"
        ) from exc
    if (
        manifest.stage_name != "stage4_completion_diagnostics"
        or manifest.schema_version != 1
    ):
        raise CompletionSummaryError(
            "input is not a Stage 4 completion diagnostics artifact"
        )
    payload = _read_regular_bytes(
        resolved / "results.json",
        maximum_bytes=MAX_RESULTS_BYTES,
        name="completion diagnostics results",
    )
    document = _json_object(payload, name="completion diagnostics results")
    try:
        if __package__:
            from scripts.run_stage4_completion_diagnostics import (
                verify_diagnostics_results_document,
            )
        else:  # pragma: no cover - direct production CLI invocation
            from run_stage4_completion_diagnostics import (
                verify_diagnostics_results_document,
            )

        results_digest = verify_diagnostics_results_document(document)
    except Exception as exc:
        raise CompletionSummaryError(
            "completion diagnostics results failed verification"
        ) from exc
    if (
        manifest.files.get("results.json")
        != hashlib.sha256(payload).hexdigest()
        or manifest.metadata.get("results_payload_sha256") != results_digest
    ):
        raise CompletionSummaryError(
            "completion diagnostics manifest/results binding differs"
        )
    return LoadedDiagnosticsArtifact(
        resolved,
        manifest.artifact_id,
        results_digest,
        document,
    )


def _seed_results(candidate: Mapping[str, object], *, name: str) -> list[Mapping[str, object]]:
    values = _list(candidate.get("seed_results"), name=f"{name}.seed_results")
    resolved = [
        _mapping(value, name=f"{name}.seed_results[{index}]")
        for index, value in enumerate(values)
    ]
    seeds = tuple(value.get("split_seed") for value in resolved)
    if seeds != FROZEN_SPLIT_SEEDS:
        raise CompletionSummaryError(
            f"{name} must contain the three frozen split seeds in order"
        )
    return resolved


def _candidate_map(
    experiment: Mapping[str, object],
    *,
    name: str,
) -> dict[str, Mapping[str, object]]:
    candidates = _list(experiment.get("candidates"), name=f"{name}.candidates")
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, name=f"{name}.candidates[{index}]")
        candidate_id = _required_string(
            candidate.get("candidate_id"),
            name=f"{name}.candidates[{index}].candidate_id",
        )
        if candidate_id in result:
            raise CompletionSummaryError(f"{name} has duplicate candidate ids")
        result[candidate_id] = candidate
    return result


def _comparison_seed_row(
    *,
    candidate_seed: Mapping[str, object],
    reference_seed: Mapping[str, object],
    candidate_id: str,
    reference_id: str,
    name: str,
) -> dict[str, object]:
    if candidate_seed.get("split_seed") != reference_seed.get("split_seed"):
        raise CompletionSummaryError(f"{name} seed pairing differs")
    paired = _mapping(
        candidate_seed.get("paired_vs_reference"),
        name=f"{name}.paired_vs_reference",
    )
    if (
        paired.get("candidate_id") != candidate_id
        or paired.get("reference_id") != reference_id
    ):
        raise CompletionSummaryError(f"{name} paired candidate identities differ")
    candidate_mae = _finite_number(
        paired.get("candidate_mae"), name=f"{name}.candidate_mae"
    )
    reference_mae = _finite_number(
        paired.get("reference_mae"), name=f"{name}.reference_mae"
    )
    delta = _finite_number(paired.get("mae_delta"), name=f"{name}.mae_delta")
    lower = _finite_number(
        paired.get("mae_delta_ci_lower"), name=f"{name}.mae_delta_ci_lower"
    )
    upper = _finite_number(
        paired.get("mae_delta_ci_upper"), name=f"{name}.mae_delta_ci_upper"
    )
    win_probability = _finite_number(
        paired.get("candidate_win_probability"),
        name=f"{name}.candidate_win_probability",
    )
    if not 0 <= win_probability <= 1 or lower > upper:
        raise CompletionSummaryError(f"{name} paired bootstrap values are invalid")
    if not math.isclose(
        delta,
        candidate_mae - reference_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise CompletionSummaryError(f"{name} paired MAE delta does not close")
    candidate_metrics = _mapping(
        candidate_seed.get("metrics"), name=f"{name}.candidate metrics"
    )
    reference_metrics = _mapping(
        reference_seed.get("metrics"), name=f"{name}.reference metrics"
    )
    if not math.isclose(
        _finite_number(candidate_metrics.get("mae"), name=f"{name}.candidate metrics.mae"),
        candidate_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ) or not math.isclose(
        _finite_number(reference_metrics.get("mae"), name=f"{name}.reference metrics.mae"),
        reference_mae,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise CompletionSummaryError(f"{name} paired MAE differs from candidate metrics")
    if upper < 0:
        bootstrap_outcome = "candidate_supported"
    elif lower > 0:
        bootstrap_outcome = "reference_supported"
    else:
        bootstrap_outcome = "inconclusive"
    return {
        "split_seed": candidate_seed["split_seed"],
        "candidate_mae": candidate_mae,
        "reference_mae": reference_mae,
        "mae_delta": delta,
        "mae_winner": (
            "candidate" if delta < 0 else "reference" if delta > 0 else "tie"
        ),
        "mae_delta_ci_lower": lower,
        "mae_delta_ci_upper": upper,
        "candidate_win_probability": win_probability,
        "bootstrap_outcome": bootstrap_outcome,
    }


def _ablation(
    candidate: Mapping[str, object],
    *,
    reference_id: str,
    axis: str,
    allowed_paths: set[str] | None,
    name: str,
) -> None:
    ablation = _mapping(candidate.get("ablation"), name=f"{name}.ablation")
    if (
        ablation.get("reference_candidate_id") != reference_id
        or ablation.get("axis") != axis
    ):
        raise CompletionSummaryError(f"{name} ablation identity differs")
    if allowed_paths is not None:
        paths = _list(
            ablation.get("allowed_config_paths"),
            name=f"{name}.ablation.allowed_config_paths",
        )
        if set(paths) != allowed_paths:
            raise CompletionSummaryError(f"{name} ablation paths differ")


def _graph_seed_policy(candidate: Mapping[str, object], *, name: str) -> str:
    graph = _mapping(candidate.get("candidate_graph"), name=f"{name}.candidate_graph")
    return _required_string(
        graph.get("seed_policy_id"), name=f"{name}.candidate_graph.seed_policy_id"
    )


def _comparison_document(
    *,
    artifact: LoadedArtifact,
    experiment: Mapping[str, object],
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    candidate_id: str,
    reference_id: str,
    name: str,
) -> dict[str, object]:
    candidate_seeds = _seed_results(candidate, name=f"{name}.{candidate_id}")
    reference_seeds = _seed_results(reference, name=f"{name}.{reference_id}")
    seed_rows = [
        _comparison_seed_row(
            candidate_seed=candidate_seed,
            reference_seed=reference_seed,
            candidate_id=candidate_id,
            reference_id=reference_id,
            name=f"{name}.seed[{candidate_seed['split_seed']}]",
        )
        for candidate_seed, reference_seed in zip(
            candidate_seeds, reference_seeds, strict=True
        )
    ]
    source = _mapping(artifact.document.get("source"), name="artifact source")
    return {
        "artifact_id": artifact.artifact_id,
        "run_id": artifact.document["run_id"],
        "source_name": _required_string(
            source.get("source_name"), name="source.source_name"
        ),
        "experiment_id": _required_string(
            experiment.get("experiment_id"), name=f"{name}.experiment_id"
        ),
        "position": _required_string(
            experiment.get("position"), name=f"{name}.position"
        ),
        "target": _required_string(experiment.get("target"), name=f"{name}.target"),
        "condition_id": _required_string(
            experiment.get("condition_id"), name=f"{name}.condition_id"
        ),
        "candidate_id": candidate_id,
        "reference_candidate_id": reference_id,
        "seed_results": seed_rows,
        "seed_outcomes": {
            "candidate_supported": sum(
                row["bootstrap_outcome"] == "candidate_supported" for row in seed_rows
            ),
            "reference_supported": sum(
                row["bootstrap_outcome"] == "reference_supported" for row in seed_rows
            ),
            "inconclusive": sum(
                row["bootstrap_outcome"] == "inconclusive" for row in seed_rows
            ),
        },
    }


def _call_pre_comparisons(artifacts: Sequence[LoadedArtifact]) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    seen_cells: set[tuple[str, str, str, str]] = set()
    for artifact in artifacts:
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for index, raw in enumerate(experiments):
            experiment = _mapping(raw, name=f"experiments[{index}]")
            if experiment.get("position") != "call_pre":
                continue
            candidates = _candidate_map(experiment, name=f"experiments[{index}]")
            if MLP_CANDIDATE_ID not in candidates:
                continue
            if LIGHTGBM_CANDIDATE_ID not in candidates:
                raise CompletionSummaryError(
                    "Call-pre MLP experiment lacks the LightGBM reference"
                )
            candidate = candidates[MLP_CANDIDATE_ID]
            _ablation(
                candidate,
                reference_id=LIGHTGBM_CANDIDATE_ID,
                axis="method",
                allowed_paths=None,
                name=f"experiments[{index}].{MLP_CANDIDATE_ID}",
            )
            document = _comparison_document(
                artifact=artifact,
                experiment=experiment,
                candidate=candidate,
                reference=candidates[LIGHTGBM_CANDIDATE_ID],
                candidate_id=MLP_CANDIDATE_ID,
                reference_id=LIGHTGBM_CANDIDATE_ID,
                name=f"experiments[{index}]",
            )
            cell = (
                str(document["source_name"]),
                str(document["condition_id"]),
                str(document["position"]),
                str(document["target"]),
            )
            if cell in seen_cells:
                raise CompletionSummaryError("duplicate Call-pre comparison cell")
            seen_cells.add(cell)
            documents.append(document)
    return sorted(
        documents,
        key=lambda item: (
            str(item["source_name"]),
            str(item["condition_id"]),
            str(item["target"]),
        ),
    )


def _seed_policy_comparisons(
    artifacts: Sequence[LoadedArtifact],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    documents: list[dict[str, object]] = []
    seen_conditions: set[tuple[str, str]] = set()
    for artifact in artifacts:
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for index, raw in enumerate(experiments):
            experiment = _mapping(raw, name=f"experiments[{index}]")
            candidates = _candidate_map(experiment, name=f"experiments[{index}]")
            if POINT_ONLY_SEED_CANDIDATE_ID not in candidates:
                continue
            if RAW_SEED_CANDIDATE_ID not in candidates:
                raise CompletionSummaryError(
                    "point-only seed experiment lacks the raw repaired reference"
                )
            candidate = candidates[POINT_ONLY_SEED_CANDIDATE_ID]
            reference = candidates[RAW_SEED_CANDIDATE_ID]
            _ablation(
                candidate,
                reference_id=RAW_SEED_CANDIDATE_ID,
                axis="seed_policy",
                allowed_paths={"graph.seed_policy_id"},
                name=f"experiments[{index}].{POINT_ONLY_SEED_CANDIDATE_ID}",
            )
            if (
                _graph_seed_policy(candidate, name=POINT_ONLY_SEED_CANDIDATE_ID)
                != POINT_ONLY_SEED_POLICY_ID
                or _graph_seed_policy(reference, name=RAW_SEED_CANDIDATE_ID)
                != RAW_SEED_POLICY_ID
            ):
                raise CompletionSummaryError("seed-policy candidate graph differs")
            document = _comparison_document(
                artifact=artifact,
                experiment=experiment,
                candidate=candidate,
                reference=reference,
                candidate_id=POINT_ONLY_SEED_CANDIDATE_ID,
                reference_id=RAW_SEED_CANDIDATE_ID,
                name=f"experiments[{index}]",
            )
            all_upper_below_zero = all(
                float(row["mae_delta_ci_upper"]) < 0
                for row in document["seed_results"]  # type: ignore[union-attr]
            )
            document["replacement_rule"] = {
                "rule_id": REPLACEMENT_RULE_ID,
                "required_seed_count": len(FROZEN_SPLIT_SEEDS),
                "all_seed_bootstrap_upper_below_zero": all_upper_below_zero,
                "replace_reference": all_upper_below_zero,
            }
            cell = (
                str(document["source_name"]),
                str(document["condition_id"]),
            )
            if cell in seen_conditions:
                raise CompletionSummaryError("duplicate seed-policy condition")
            seen_conditions.add(cell)
            documents.append(document)
    documents.sort(
        key=lambda item: (str(item["source_name"]), str(item["condition_id"]))
    )
    passing = sum(
        bool(item["replacement_rule"]["replace_reference"])  # type: ignore[index]
        for item in documents
    )
    overall = {
        "rule_id": REPLACEMENT_RULE_ID,
        "condition_count": len(documents),
        "passing_condition_count": passing,
        "all_conditions_pass": bool(documents) and passing == len(documents),
        "decision": (
            "replace_raw_repaired_reference"
            if documents and passing == len(documents)
            else "retain_raw_repaired_reference"
        ),
    }
    return documents, overall


def _find_run_dispersion(
    value: object,
    *,
    path: str,
) -> list[tuple[str, Mapping[str, object]]]:
    found: list[tuple[str, Mapping[str, object]]] = []
    if isinstance(value, Mapping):
        if "run_dispersion_extension_id" in value:
            found.append((path, _mapping(value, name=path)))
        for key, item in value.items():
            if isinstance(key, str):
                found.extend(
                    _find_run_dispersion(item, path=f"{path}.{key}")
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _find_run_dispersion(item, path=f"{path}[{index}]")
            )
    return found


def _is_lifecycle_candidate(candidate: Mapping[str, object]) -> bool:
    graph = _mapping(candidate.get("candidate_graph"), name="candidate graph")
    return graph.get("initializer_estimator_id") != "none"


def _diagnostics_index(
    artifacts: Sequence[LoadedArtifact],
    diagnostics: LoadedDiagnosticsArtifact | None,
) -> dict[tuple[str, str, str, int], Mapping[str, object]]:
    if diagnostics is None:
        return {}
    expected_artifacts: dict[str, tuple[str, str]] = {}
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        source_name = _required_string(
            source.get("source_name"), name="source.source_name"
        )
        expected_artifacts[source_name] = (
            artifact.artifact_id,
            artifact.results_payload_sha256,
        )
    declared: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(
        _list(
            diagnostics.document.get("source_artifacts"),
            name="diagnostics source_artifacts",
        )
    ):
        item = _mapping(raw, name=f"diagnostics source_artifacts[{index}]")
        declared[
            _required_string(item.get("source_name"), name="diagnostics source name")
        ] = (
            _required_sha256(
                item.get("artifact_id"), name="diagnostics source artifact id"
            ),
            _required_sha256(
                item.get("results_payload_sha256"),
                name="diagnostics source results hash",
            ),
        )
    if declared != expected_artifacts:
        raise CompletionSummaryError(
            "diagnostics supplement is bound to another source artifact set"
        )
    index: dict[tuple[str, str, str, int], Mapping[str, object]] = {}
    for item_index, raw in enumerate(
        _list(
            diagnostics.document.get("diagnostics"),
            name="completion diagnostics",
        )
    ):
        item = _mapping(raw, name=f"completion diagnostics[{item_index}]")
        seed = item.get("split_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CompletionSummaryError("completion diagnostic split seed is invalid")
        key = (
            _required_string(item.get("source_name"), name="diagnostic source"),
            _required_string(item.get("experiment_id"), name="diagnostic experiment"),
            _required_string(item.get("candidate_id"), name="diagnostic candidate"),
            seed,
        )
        if key in index:
            raise CompletionSummaryError("completion diagnostics repeat an identity")
        index[key] = item
    return index


def _metric_coverage(
    artifacts: Sequence[LoadedArtifact],
    diagnostics: LoadedDiagnosticsArtifact | None = None,
) -> dict[str, object]:
    diagnostics_by_key = _diagnostics_index(artifacts, diagnostics)
    consumed_diagnostics: set[tuple[str, str, str, int]] = set()
    artifact_rows: list[dict[str, object]] = []
    total_seed_results = 0
    interval_complete = 0
    total_lifecycle_seed_results = 0
    run_dispersion_complete = 0
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        source_name = _required_string(
            source.get("source_name"), name="source.source_name"
        )
        interval_missing: list[dict[str, object]] = []
        run_missing: list[dict[str, object]] = []
        artifact_seed_results = 0
        artifact_interval_complete = 0
        artifact_lifecycle_seed_results = 0
        artifact_run_complete = 0
        experiments = _list(
            artifact.document.get("experiments"), name="artifact experiments"
        )
        for experiment_index, raw_experiment in enumerate(experiments):
            experiment = _mapping(
                raw_experiment, name=f"experiments[{experiment_index}]"
            )
            experiment_id = _required_string(
                experiment.get("experiment_id"),
                name=f"experiments[{experiment_index}].experiment_id",
            )
            candidates = _candidate_map(
                experiment, name=f"experiments[{experiment_index}]"
            )
            for candidate_id, candidate in candidates.items():
                lifecycle = _is_lifecycle_candidate(candidate)
                for seed in _seed_results(
                    candidate, name=f"{experiment_id}.{candidate_id}"
                ):
                    artifact_seed_results += 1
                    metrics = _mapping(
                        seed.get("metrics"),
                        name=f"{experiment_id}.{candidate_id}.metrics",
                    )
                    missing_interval = [
                        field for field in INTERVAL_RESERVE_FIELDS if field not in metrics
                    ]
                    if missing_interval:
                        interval_missing.append(
                            {
                                "experiment_id": experiment_id,
                                "candidate_id": candidate_id,
                                "split_seed": seed["split_seed"],
                                "missing_fields": missing_interval,
                            }
                        )
                    else:
                        artifact_interval_complete += 1
                    if lifecycle:
                        artifact_lifecycle_seed_results += 1
                        found = _find_run_dispersion(seed, path="seed_result")
                        diagnostic_key = (
                            source_name,
                            experiment_id,
                            candidate_id,
                            int(seed["split_seed"]),
                        )
                        supplement = diagnostics_by_key.get(diagnostic_key)
                        if supplement is not None:
                            supplement_variance = _mapping(
                                supplement.get("run_variance"),
                                name="completion diagnostic run variance",
                            )
                            if found and found[0][1] != supplement_variance:
                                raise CompletionSummaryError(
                                    "source and supplement run dispersion differ"
                                )
                            if not found:
                                found = [
                                    (
                                        "completion_diagnostics.run_variance",
                                        supplement_variance,
                                    )
                                ]
                            consumed_diagnostics.add(diagnostic_key)
                        if len(found) > 1:
                            raise CompletionSummaryError(
                                "lifecycle seed contains multiple run-dispersion documents"
                            )
                        missing_run = (
                            list(RUN_DISPERSION_FIELDS)
                            if not found
                            else [
                                field
                                for field in RUN_DISPERSION_FIELDS
                                if field not in found[0][1]
                            ]
                        )
                        if missing_run:
                            run_missing.append(
                                {
                                    "experiment_id": experiment_id,
                                    "candidate_id": candidate_id,
                                    "split_seed": seed["split_seed"],
                                    "document_path": found[0][0] if found else None,
                                    "missing_fields": missing_run,
                                }
                            )
                        else:
                            artifact_run_complete += 1
        total_seed_results += artifact_seed_results
        interval_complete += artifact_interval_complete
        total_lifecycle_seed_results += artifact_lifecycle_seed_results
        run_dispersion_complete += artifact_run_complete
        artifact_rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "run_id": artifact.document["run_id"],
                "source_name": source_name,
                "candidate_seed_result_count": artifact_seed_results,
                "interval_reserve_complete_count": artifact_interval_complete,
                "interval_reserve_missing": interval_missing,
                "lifecycle_candidate_seed_result_count": (
                    artifact_lifecycle_seed_results
                ),
                "run_dispersion_complete_count": artifact_run_complete,
                "run_dispersion_missing": run_missing,
            }
        )
    if consumed_diagnostics != set(diagnostics_by_key):
        extra = sorted(set(diagnostics_by_key) - consumed_diagnostics)
        raise CompletionSummaryError(
            f"completion diagnostics do not match lifecycle seed results: {extra[:3]}"
        )
    return {
        "interval_reserve": {
            "required_fields": list(INTERVAL_RESERVE_FIELDS),
            "complete_count": interval_complete,
            "expected_count": total_seed_results,
            "complete": interval_complete == total_seed_results,
        },
        "repeated_run_dispersion": {
            "required_fields": list(RUN_DISPERSION_FIELDS),
            "complete_count": run_dispersion_complete,
            "expected_lifecycle_count": total_lifecycle_seed_results,
            "complete": run_dispersion_complete == total_lifecycle_seed_results,
            "supplement_artifact_id": (
                diagnostics.artifact_id if diagnostics is not None else None
            ),
        },
        "artifacts": artifact_rows,
    }


def build_completion_summary(
    artifacts: Sequence[LoadedArtifact],
    *,
    diagnostics: LoadedDiagnosticsArtifact | None = None,
) -> dict[str, object]:
    """Build the machine-readable completion comparison summary."""

    if not artifacts:
        raise CompletionSummaryError("cannot summarize an empty artifact set")
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise CompletionSummaryError("development artifact ids must be unique")
    call_pre = _call_pre_comparisons(artifacts)
    seed_policy, replacement = _seed_policy_comparisons(artifacts)
    coverage = _metric_coverage(artifacts, diagnostics)
    sources = []
    for artifact in artifacts:
        source = _mapping(artifact.document.get("source"), name="artifact source")
        sources.append(
            {
                "source_name": _required_string(
                    source.get("source_name"), name="source.source_name"
                ),
                "run_id": artifact.document["run_id"],
                "artifact_id": artifact.artifact_id,
                "results_payload_sha256": artifact.results_payload_sha256,
            }
        )
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "policy_id": SUMMARY_POLICY_ID,
        "final_holdout_access": {
            "policy_id": FINAL_HOLDOUT_POLICY_ID,
            "allowed": False,
            "accessed": False,
            "artifact_count_checked": len(artifacts),
            "all_development_documents_closed": True,
            "statement": (
                "Only aggregate-safe Stage 4 development results were read; "
                "final artifacts, source data, and final labels were not opened."
            ),
        },
        "sources": sorted(sources, key=lambda item: str(item["source_name"])),
        "call_pre_mlp_vs_lightgbm": call_pre,
        "seed_policy_point_only_vs_raw_repaired": seed_policy,
        "seed_policy_frozen_replacement_rule": replacement,
        "metric_coverage": coverage,
        "diagnostics_supplement": (
            {
                "artifact_id": diagnostics.artifact_id,
                "results_payload_sha256": diagnostics.results_payload_sha256,
            }
            if diagnostics is not None
            else None
        ),
        "completion_status": {
            "has_call_pre_comparisons": bool(call_pre),
            "has_seed_policy_comparisons": bool(seed_policy),
            "metric_coverage_complete": bool(
                coverage["interval_reserve"]["complete"]  # type: ignore[index]
                and coverage["repeated_run_dispersion"]["complete"]  # type: ignore[index]
            ),
        },
    }


def _compact_seed_rows(rows: Iterable[Mapping[str, object]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            (
                f"{row['split_seed']}: delta={float(row['mae_delta']):.3f}, "
                f"CI=[{float(row['mae_delta_ci_lower']):.3f},"
                f"{float(row['mae_delta_ci_upper']):.3f}], "
                f"{row['bootstrap_outcome']}"
            )
        )
    return "<br>".join(parts)


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render a compact human-readable view of the machine summary."""

    sources = _list(summary.get("sources"), name="summary.sources")
    call_pre = _list(
        summary.get("call_pre_mlp_vs_lightgbm"),
        name="summary.call_pre_mlp_vs_lightgbm",
    )
    seed_policy = _list(
        summary.get("seed_policy_point_only_vs_raw_repaired"),
        name="summary.seed_policy_point_only_vs_raw_repaired",
    )
    replacement = _mapping(
        summary.get("seed_policy_frozen_replacement_rule"),
        name="summary.seed_policy_frozen_replacement_rule",
    )
    coverage = _mapping(
        summary.get("metric_coverage"), name="summary.metric_coverage"
    )
    interval = _mapping(
        coverage.get("interval_reserve"), name="metric_coverage.interval_reserve"
    )
    dispersion = _mapping(
        coverage.get("repeated_run_dispersion"),
        name="metric_coverage.repeated_run_dispersion",
    )
    lines = [
        "# Stage 4 completion development summary",
        "",
        f"- Development artifacts: {len(sources)}",
        (
            "- Final holdout: not accessed; final artifacts, source data, and "
            "final labels were not opened."
        ),
        "",
        "## Call-pre MLP vs LightGBM",
        "",
    ]
    if call_pre:
        lines.extend(
            [
                "| Source | Condition | Target | Per-seed paired MAE result |",
                "|---|---|---|---|",
            ]
        )
        for raw in call_pre:
            row = _mapping(raw, name="Call-pre comparison")
            lines.append(
                "| {source} | `{condition}` | `{target}` | {seeds} |".format(
                    source=row["source_name"],
                    condition=row["condition_id"],
                    target=row["target"],
                    seeds=_compact_seed_rows(
                        _mapping(item, name="Call-pre seed")
                        for item in _list(
                            row.get("seed_results"),
                            name="Call-pre comparison.seed_results",
                        )
                    ),
                )
            )
    else:
        lines.append("No Call-pre MLP comparison was present.")
    lines.extend(["", "## Seed policy", ""])
    if seed_policy:
        lines.extend(
            [
                "| Source | Condition | Per-seed point-only minus raw-repaired MAE | Rule |",
                "|---|---|---|---|",
            ]
        )
        for raw in seed_policy:
            row = _mapping(raw, name="seed-policy comparison")
            rule = _mapping(
                row.get("replacement_rule"),
                name="seed-policy comparison.replacement_rule",
            )
            lines.append(
                "| {source} | `{condition}` | {seeds} | {decision} |".format(
                    source=row["source_name"],
                    condition=row["condition_id"],
                    seeds=_compact_seed_rows(
                        _mapping(item, name="seed-policy seed")
                        for item in _list(
                            row.get("seed_results"),
                            name="seed-policy comparison.seed_results",
                        )
                    ),
                    decision=(
                        "replace" if rule["replace_reference"] else "retain reference"
                    ),
                )
            )
    else:
        lines.append("No seed-policy comparison was present.")
    lines.extend(
        [
            "",
            (
                f"Frozen replacement rule: `{replacement['decision']}` "
                f"({replacement['passing_condition_count']}/"
                f"{replacement['condition_count']} conditions passed)."
            ),
            "",
            "## Metric-field coverage",
            "",
            "| Group | Complete / expected | Status |",
            "|---|---:|---|",
            (
                f"| Interval/reserve | {interval['complete_count']} / "
                f"{interval['expected_count']} | "
                f"{'complete' if interval['complete'] else 'incomplete'} |"
            ),
            (
                f"| Repeated-run dispersion (lifecycle only) | "
                f"{dispersion['complete_count']} / "
                f"{dispersion['expected_lifecycle_count']} | "
                f"{'complete' if dispersion['complete'] else 'incomplete'} |"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize four Stage 4 development artifacts without opening "
            "the final holdout."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "development artifact directories/results.json files, or one "
            "completion release lock"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="stdout format (the command never writes artifact or report files)",
    )
    parser.add_argument(
        "--diagnostics-artifact",
        help=(
            "optional immutable completion diagnostics artifact used to close "
            "lifecycle repeated-run coverage"
        ),
    )
    parser.add_argument(
        "--expected-artifacts",
        type=int,
        default=4,
        help="required distinct development artifact count; use 0 to disable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        references = resolve_artifact_references(args.inputs)
        if args.expected_artifacts < 0:
            raise CompletionSummaryError("--expected-artifacts cannot be negative")
        if args.expected_artifacts and len(references) != args.expected_artifacts:
            raise CompletionSummaryError(
                f"expected {args.expected_artifacts} development artifacts, "
                f"found {len(references)}"
            )
        artifacts = tuple(load_development_artifact(item) for item in references)
        diagnostics = (
            load_completion_diagnostics_artifact(args.diagnostics_artifact)
            if args.diagnostics_artifact
            else None
        )
        summary = build_completion_summary(
            artifacts,
            diagnostics=diagnostics,
        )
    except CompletionSummaryError as exc:
        print(f"completion summary failed: {exc}", file=sys.stderr)
        return 2
    if args.format == "markdown":
        print(render_markdown(summary))
    else:
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
