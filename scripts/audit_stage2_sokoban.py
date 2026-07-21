"""Publish aggregate-only Stage 2 compatibility evidence for BAGEN Sokoban."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from token_prediction.collection import BagenSokobanMetadata, BagenSokobanReader
from token_prediction.crossfit import SEED_POLICY_ID
from token_prediction.dataset import (
    LabelStatus,
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    build_lifecycle_slice,
)
from token_prediction.development import build_development_protocol
from token_prediction.estimators import (
    FittedCrossPositionDeduct,
    SessionSeed,
    TokenForecast,
)
from token_prediction.lifecycle import run_lifecycle_sequence
from token_prediction.lineage import publish_artifact, verify_artifact

if __package__:
    from scripts.run_stage2_experiments import (
        DEFAULT_OUTPUT_ROOT,
        Stage2ExperimentError,
        _assert_aggregate_safe,
        _is_link_or_reparse,
        _load_auxiliary_source_lock,
        _output_key,
        _required_sha256,
        _safe_output_root,
        _semantic_sha256,
        capture_stage2_code_binding,
    )
    from scripts.verify_stage1_baseline import verify_stage1
else:  # pragma: no cover - exercised by the production CLI
    from run_stage2_experiments import (
        DEFAULT_OUTPUT_ROOT,
        Stage2ExperimentError,
        _assert_aggregate_safe,
        _is_link_or_reparse,
        _load_auxiliary_source_lock,
        _output_key,
        _required_sha256,
        _safe_output_root,
        _semantic_sha256,
        capture_stage2_code_binding,
    )
    from verify_stage1_baseline import verify_stage1


SOKOBAN_AUDIT_SCHEMA_VERSION = 2
SOKOBAN_AUDIT_STAGE_NAME = "stage2_bagen_sokoban_compatibility"
SOKOBAN_AUDIT_POLICY_ID = "stage2_sokoban_regression_lifecycle_estimability_v2"
DEFAULT_STAGE1_ARTIFACT = "workspace/experiments/lightgbm_preliminary/c52866a7e251768726fd"
EXPECTED_STAGE1_ARTIFACT_ID = (
    "d26969603582ff590a6193234e17d39e6f0697a8e36e08a559549d0a45597afe"
)


@dataclass(frozen=True)
class SokobanAuditSummary:
    run_id: str
    output_dir: Path
    artifact_id: str
    results_payload_sha256: str


def _task_set_sha256(tasks: set[str]) -> str:
    return _semantic_sha256(sorted(tasks))


def build_sokoban_audit_results(
    root: Path,
    *,
    stage1_artifact: str,
) -> tuple[dict[str, object], object, tuple[Path, ...]]:
    source, paths = _load_auxiliary_source_lock(root, source_name="bagen_sokoban")
    raw = paths[0]
    trajectories = BagenSokobanReader().read_all(
        raw,
        BagenSokobanMetadata(reasoning_effort="low"),
    )
    dataset = build_capability_supervised_dataset(trajectories, source.descriptor)
    protocol = build_development_protocol(dataset)
    conditions = sorted({row.point.condition_id for row in dataset.rows})
    status_counts = {
        status.value: sum(row.status == status for row in dataset.rows)
        for status in LabelStatus
    }
    lifecycle_counts = {
        "condition_count": len(conditions),
        "sequence_count": 0,
        "step_count": 0,
        "scored_step_count": 0,
        "unscored_step_count": 0,
        "offline_shadow_prediction_count": 0,
    }
    parity_digests: list[str] = []
    for condition_id in conditions:
        target_rows = tuple(
            row
            for row in dataset.rows
            if row.point.condition_id == condition_id
            and row.point.target
            == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            and row.point.position
            in {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE}
        )
        if not target_rows:
            continue
        lifecycle = build_lifecycle_slice(
            dataset,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            condition_id=condition_id,
        )
        lifecycle_counts["sequence_count"] += len(lifecycle.sequences)
        lifecycle_counts["step_count"] += sum(
            len(sequence.steps) for sequence in lifecycle.sequences
        )
        lifecycle_counts["scored_step_count"] += sum(
            step.score_mask
            for sequence in lifecycle.sequences
            for step in sequence.steps
        )
        lifecycle_counts["unscored_step_count"] += sum(
            not step.score_mask
            for sequence in lifecycle.sequences
            for step in sequence.steps
        )
        sequence = next(
            item for item in lifecycle.sequences if len(item.steps) > 1
        )
        task_pre = sequence.steps[0].point
        seed_forecast = TokenForecast(
            point_id=task_pre.point_id,
            target=task_pre.target,
            lower=750_000.0,
            point=1_000_000.0,
            upper=1_250_000.0,
            raw_lower=750_000.0,
            raw_point=1_000_000.0,
            raw_upper=1_250_000.0,
        )
        seed = SessionSeed(
            task_pre_point=task_pre,
            forecast=seed_forecast,
            initializer_id="compatibility_fixed_forecast",
            initializer_hash=_semantic_sha256(
                {"policy_id": "sokoban_driver_compatibility_seed_v1"}
            ),
            inner_split_id=protocol.protocol_id,
            component_bundle_hashes=(
                _semantic_sha256({"component": "fixed_compatibility_seed"}),
            ),
            seed_policy_id=SEED_POLICY_ID,
            seed_policy_hash=_semantic_sha256(
                {"seed_policy_id": SEED_POLICY_ID, "mode": "compatibility_only"}
            ),
        )
        fitted = FittedCrossPositionDeduct(
            estimator_id="cross_position_deduct",
            dataset_id=sequence.dataset_id,
            target=sequence.target,
            condition_id=sequence.condition_id,
            input_contract_hash=sequence.input_contract_hash,
        )
        offline = run_lifecycle_sequence(fitted, sequence, seed, runtime_mode="offline")
        shadow = run_lifecycle_sequence(fitted, sequence, seed, runtime_mode="shadow")
        offline_projection = [
            {
                "forecast": asdict(item.forecast),
                "transition": asdict(item.transition),
            }
            for item in offline.predictions
        ]
        shadow_projection = [
            {
                "forecast": asdict(item.forecast),
                "transition": asdict(item.transition),
            }
            for item in shadow.predictions
        ]
        if offline_projection != shadow_projection:
            raise Stage2ExperimentError("Sokoban offline/shadow lifecycle parity failed")
        lifecycle_counts["offline_shadow_prediction_count"] += len(offline.predictions)
        parity_digests.append(_semantic_sha256(offline_projection))

    stage1 = verify_stage1(
        root / stage1_artifact,
        raw,
        repository_root=root,
        expected_artifact_id=EXPECTED_STAGE1_ARTIFACT_ID,
        expected_bundles=20,
        expected_parity=992,
        discover_commit=True,
    )
    results: dict[str, object] = {
        "audit_schema_version": SOKOBAN_AUDIT_SCHEMA_VERSION,
        "stage_name": SOKOBAN_AUDIT_STAGE_NAME,
        "policy_id": SOKOBAN_AUDIT_POLICY_ID,
        "source": {
            "source_id": source.descriptor.source_id,
            "revision": source.descriptor.revision,
            "source_descriptor_hash": source.descriptor.descriptor_hash,
            "capability_contract_hash": source.descriptor.capabilities.contract_hash,
            "manifest_path": source.manifest_path,
            "manifest_sha256": source.manifest_sha256,
            "raw_artifact_sha256": source.raw_artifact_sha256,
        },
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "row_count": len(dataset.rows),
            "trajectory_count": len(trajectories),
            "task_count": len(dataset.task_ids),
            "task_set_sha256": _task_set_sha256(set(dataset.task_ids)),
            "condition_count": len(conditions),
            "status_counts": status_counts,
        },
        "lifecycle_compatibility": {
            **lifecycle_counts,
            "offline_shadow_exact": True,
            "prediction_projection_sha256": _semantic_sha256(sorted(parity_digests)),
        },
        "development_gate": {
            "status": "estimable",
            "reason": "nested_five_fold_protocol_available",
            "protocol_id": protocol.protocol_id,
            "development_task_count": len(protocol.development_dataset.task_ids),
            "final_holdout_task_count": len(protocol.final_holdout_tasks),
            "outer_folds": 5,
            "inner_folds": 5,
            "folds_reduced": False,
            "target_values_used_for_gate": False,
        },
        "stage1_regression": {
            "artifact_id": stage1.artifact_id,
            "artifact_manifest_sha256": stage1.artifact_manifest_sha256,
            "bundle_count": stage1.bundle_count,
            "parity_record_count": stage1.parity_record_count,
            "parity_mismatch_count": stage1.parity_mismatch_count,
            "parity_sha256": stage1.parity_sha256,
            "historical_source_binding_status": stage1.source_binding_status,
        },
        "final_holdout": {
            "evaluated": False,
            "prediction_count": 0,
            "target_values_used_for_fit_calibration_scoring": False,
            "selection_claim": "none",
        },
    }
    _assert_aggregate_safe(results)
    results["results_payload_sha256"] = _semantic_sha256(results)
    return results, source, paths


def verify_sokoban_audit_results(value: Mapping[str, object]) -> str:
    required = {
        "audit_schema_version",
        "stage_name",
        "policy_id",
        "source",
        "dataset",
        "lifecycle_compatibility",
        "development_gate",
        "stage1_regression",
        "final_holdout",
        "results_payload_sha256",
    }
    if set(value) != required:
        raise Stage2ExperimentError("Sokoban audit results keys do not match")
    if (
        value["audit_schema_version"] != SOKOBAN_AUDIT_SCHEMA_VERSION
        or value["stage_name"] != SOKOBAN_AUDIT_STAGE_NAME
        or value["policy_id"] != SOKOBAN_AUDIT_POLICY_ID
    ):
        raise Stage2ExperimentError("Sokoban audit result identity is invalid")
    _assert_aggregate_safe(value)
    source = value["source"]
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "source_id",
            "revision",
            "source_descriptor_hash",
            "capability_contract_hash",
            "manifest_path",
            "manifest_sha256",
            "raw_artifact_sha256",
        }
        or source["source_id"] != "bagen_sokoban_dialogues_v1"
        or source["manifest_path"] != "configs/stage2_auxiliary_sources.json"
    ):
        raise Stage2ExperimentError("Sokoban audit source identity is invalid")
    for field in (
        "revision",
        "source_descriptor_hash",
        "capability_contract_hash",
        "manifest_sha256",
        "raw_artifact_sha256",
    ):
        _required_sha256(source[field], name=f"Sokoban source {field}")

    dataset = value["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "dataset_id",
        "row_count",
        "trajectory_count",
        "task_count",
        "task_set_sha256",
        "condition_count",
        "status_counts",
    }:
        raise Stage2ExperimentError("Sokoban audit dataset summary is invalid")
    _required_sha256(dataset["dataset_id"], name="Sokoban dataset id")
    _required_sha256(dataset["task_set_sha256"], name="Sokoban task set")
    for field in ("row_count", "trajectory_count", "task_count", "condition_count"):
        count = dataset[field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise Stage2ExperimentError("Sokoban audit dataset counts are invalid")
    status_counts = dataset["status_counts"]
    if not isinstance(status_counts, Mapping) or set(status_counts) != {
        status.value for status in LabelStatus
    }:
        raise Stage2ExperimentError("Sokoban audit status counts are invalid")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in status_counts.values()
    ) or sum(status_counts.values()) != dataset["row_count"]:
        raise Stage2ExperimentError("Sokoban audit status counts do not close")
    if (
        dataset["trajectory_count"] != dataset["task_count"]
        or dataset["condition_count"] != 1
    ):
        raise Stage2ExperimentError("Sokoban audit cohort counts are invalid")

    lifecycle = value["lifecycle_compatibility"]
    if not isinstance(lifecycle, Mapping) or set(lifecycle) != {
        "condition_count",
        "sequence_count",
        "step_count",
        "scored_step_count",
        "unscored_step_count",
        "offline_shadow_prediction_count",
        "offline_shadow_exact",
        "prediction_projection_sha256",
    }:
        raise Stage2ExperimentError("Sokoban lifecycle evidence is invalid")
    for field in (
        "condition_count",
        "sequence_count",
        "step_count",
        "scored_step_count",
        "unscored_step_count",
        "offline_shadow_prediction_count",
    ):
        count = lifecycle[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise Stage2ExperimentError("Sokoban lifecycle counts are invalid")
    if (
        lifecycle["condition_count"] != dataset["condition_count"]
        or lifecycle["sequence_count"] != dataset["task_count"]
        or lifecycle["step_count"]
        != lifecycle["scored_step_count"] + lifecycle["unscored_step_count"]
        or lifecycle["offline_shadow_prediction_count"] <= 0
        or lifecycle["offline_shadow_exact"] is not True
    ):
        raise Stage2ExperimentError("Sokoban lifecycle parity does not close")
    _required_sha256(
        lifecycle["prediction_projection_sha256"],
        name="Sokoban lifecycle prediction projection",
    )
    holdout = value["final_holdout"]
    if not isinstance(holdout, Mapping) or holdout != {
        "evaluated": False,
        "prediction_count": 0,
        "target_values_used_for_fit_calibration_scoring": False,
        "selection_claim": "none",
    }:
        raise Stage2ExperimentError("Sokoban audit final holdout is not sealed")
    gate = value["development_gate"]
    if (
        not isinstance(gate, Mapping)
        or set(gate)
        != {
            "status",
            "reason",
            "protocol_id",
            "development_task_count",
            "final_holdout_task_count",
            "outer_folds",
            "inner_folds",
            "folds_reduced",
            "target_values_used_for_gate",
        }
        or gate["status"] != "estimable"
        or gate["reason"] != "nested_five_fold_protocol_available"
        or gate["outer_folds"] != 5
        or gate["inner_folds"] != 5
        or gate["folds_reduced"] is not False
        or gate["target_values_used_for_gate"] is not False
        or isinstance(gate["development_task_count"], bool)
        or not isinstance(gate["development_task_count"], int)
        or gate["development_task_count"] < 15
        or isinstance(gate["final_holdout_task_count"], bool)
        or not isinstance(gate["final_holdout_task_count"], int)
        or gate["final_holdout_task_count"] < 1
        or gate["development_task_count"] + gate["final_holdout_task_count"]
        != dataset["task_count"]
    ):
        raise Stage2ExperimentError("Sokoban development estimability is invalid")
    _required_sha256(gate["protocol_id"], name="Sokoban development protocol")

    stage1 = value["stage1_regression"]
    if (
        not isinstance(stage1, Mapping)
        or set(stage1)
        != {
            "artifact_id",
            "artifact_manifest_sha256",
            "bundle_count",
            "parity_record_count",
            "parity_mismatch_count",
            "parity_sha256",
            "historical_source_binding_status",
        }
        or stage1["artifact_id"] != EXPECTED_STAGE1_ARTIFACT_ID
        or stage1["bundle_count"] != 20
        or stage1["parity_record_count"] != 992
        or stage1["parity_mismatch_count"] != 0
        or stage1["historical_source_binding_status"]
        not in {"bound", "unrecoverable"}
    ):
        raise Stage2ExperimentError("Sokoban Stage 1 regression evidence is invalid")
    for field in ("artifact_manifest_sha256", "parity_sha256"):
        _required_sha256(stage1[field], name=f"Sokoban Stage 1 {field}")
    expected = dict(value)
    declared = expected.pop("results_payload_sha256")
    if not isinstance(declared, str) or _semantic_sha256(expected) != declared:
        raise Stage2ExperimentError("Sokoban audit results payload does not close")
    return declared


def run_sokoban_audit(
    *,
    repository_root: str | Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    stage1_artifact: str = DEFAULT_STAGE1_ARTIFACT,
) -> SokobanAuditSummary:
    root = Path(repository_root).resolve()
    _canonical_output, output_parent = _safe_output_root(root, output_root)
    code = capture_stage2_code_binding(root)
    results, source, raw_paths = build_sokoban_audit_results(
        root,
        stage1_artifact=stage1_artifact,
    )
    run_semantic = {
        "policy_id": SOKOBAN_AUDIT_POLICY_ID,
        "source_descriptor_hash": source.descriptor.descriptor_hash,
        "raw_artifact_sha256": source.raw_artifact_sha256,
        "dataset_id": results["dataset"]["dataset_id"],
        "stage1_artifact_id": results["stage1_regression"]["artifact_id"],
        "git_commit": code.git_commit,
        "code_tree_sha256": code.code_tree_sha256,
    }
    run_id = _semantic_sha256(run_semantic)[:24]
    output = output_parent / _output_key(run_id)
    if output.exists():
        manifest = verify_artifact(output)
        document = json.loads((output / "results.json").read_text(encoding="utf-8"))
        payload_hash = verify_sokoban_audit_results(document)
        if manifest.metadata.get("run_semantic") != run_semantic:
            raise Stage2ExperimentError("existing Sokoban audit has another identity")
        return SokobanAuditSummary(run_id, output, manifest.artifact_id, payload_hash)

    output_parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(output_parent):
        raise Stage2ExperimentError("Stage 2 output parent is unsafe")
    temporary = Path(tempfile.mkdtemp(prefix=".s2-", dir=output_parent))
    try:
        (temporary / "results.json").write_bytes(
            json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        if capture_stage2_code_binding(root) != code:
            raise Stage2ExperimentError("Stage 2 code changed during Sokoban audit")
        current_source, current_paths = _load_auxiliary_source_lock(
            root,
            source_name="bagen_sokoban",
        )
        if current_source != source or current_paths != raw_paths:
            raise Stage2ExperimentError("Sokoban source changed during audit")
        manifest = publish_artifact(
            temporary,
            stage_name=SOKOBAN_AUDIT_STAGE_NAME,
            schema_version=SOKOBAN_AUDIT_SCHEMA_VERSION,
            metadata={
                "run_id": run_id,
                "run_semantic": run_semantic,
                "results_payload_sha256": results["results_payload_sha256"],
            },
        )
        os.replace(temporary, output)
        if verify_artifact(output) != manifest:
            raise Stage2ExperimentError("published Sokoban audit failed verification")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return SokobanAuditSummary(
        run_id,
        output,
        manifest.artifact_id,
        str(results["results_payload_sha256"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage 2 BAGEN Sokoban compatibility")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-artifact", default=DEFAULT_STAGE1_ARTIFACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_sokoban_audit(
        repository_root=args.repository_root,
        output_root=args.output_root,
        stage1_artifact=args.stage1_artifact,
    )
    print(
        json.dumps(
            {**asdict(summary), "output_dir": summary.output_dir.as_posix()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
