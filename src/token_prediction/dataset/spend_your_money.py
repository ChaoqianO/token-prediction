from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .schema import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)


SPEND_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweBenchTaskMetadata:
    instance_id: str
    repo: str
    problem_statement: str
    difficulty: str | None = None


@dataclass(frozen=True)
class SpendYourMoneyImport:
    dataset: SupervisedDataset
    model_key: str
    task_count: int
    source_csv_sha256: str
    metadata_sha256: str
    target_definition: str = "rounded_mean_input_plus_mean_output_tokens"


def load_swebench_verified_metadata(
    parquet_path: str | Path,
) -> dict[str, SweBenchTaskMetadata]:
    """Read only the causal task metadata needed by the aggregate pilot."""

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - optional base-only install
        raise RuntimeError(
            "SWE-bench parquet ingestion requires pyarrow; "
            "install token-prediction[data]"
        ) from exc
    source = Path(parquet_path).resolve()
    table = pq.read_table(
        source,
        columns=["instance_id", "repo", "problem_statement", "difficulty"],
    )
    result: dict[str, SweBenchTaskMetadata] = {}
    for row in table.to_pylist():
        item = SweBenchTaskMetadata(
            instance_id=str(row.get("instance_id") or "").strip(),
            repo=str(row.get("repo") or "").strip(),
            problem_statement=str(row.get("problem_statement") or ""),
            difficulty=(
                str(row.get("difficulty")).strip()
                if row.get("difficulty") is not None
                else None
            ),
        )
        if not item.instance_id or not item.repo or not item.problem_statement:
            raise ValueError("SWE-bench metadata contains an incomplete task")
        if item.instance_id in result:
            raise ValueError(f"duplicate SWE-bench instance_id {item.instance_id!r}")
        result[item.instance_id] = item
    if not result:
        raise ValueError("SWE-bench metadata is empty")
    return result


def build_spend_your_money_dataset(
    aggregate_csv_path: str | Path,
    task_metadata: Mapping[str, SweBenchTaskMetadata],
    *,
    model_key: str,
    model_id: str | None = None,
    agent_id: str = "openhands",
    condition_id: str | None = None,
    metadata_sha256: str | None = None,
) -> SpendYourMoneyImport:
    """Build a Task-launch dataset from the paper's four-run aggregate CSV.

    The source contains one row per SWE-bench task and separate columns per
    model.  This importer intentionally selects exactly one model condition;
    it never treats the eight model-task rows as independent task samples.
    Self-estimation columns are not exposed as model features.
    """

    source = Path(aggregate_csv_path).resolve()
    csv_hash = _sha256_file(source)
    input_column = f"{model_key}_gt_input_token_avg"
    output_column = f"{model_key}_gt_output_token_avg"
    predicted_input_column = f"{model_key}_predicted_avg_input"
    predicted_output_column = f"{model_key}_predicted_avg_output"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            "problem_id",
            input_column,
            output_column,
            predicted_input_column,
            predicted_output_column,
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"aggregate CSV is missing columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("aggregate CSV is empty")

    metadata = dict(task_metadata)
    ids = [str(row.get("problem_id") or "").strip() for row in raw_rows]
    if any(not value for value in ids):
        raise ValueError("aggregate CSV contains an empty problem_id")
    if len(ids) != len(set(ids)):
        raise ValueError("aggregate CSV contains duplicate problem_id values")
    missing_metadata = sorted(set(ids) - set(metadata))
    if missing_metadata:
        raise ValueError(
            f"missing SWE-bench metadata for {len(missing_metadata)} task(s): "
            f"{missing_metadata[:3]}"
        )

    resolved_model_id = model_id or model_key
    resolved_condition = condition_id or (
        "condition:spend-your-money:"
        + _semantic_hash(
            {
                "model_key": model_key,
                "model_id": resolved_model_id,
                "agent_id": agent_id,
                "target": "mean_billed_total_tokens",
            }
        )[:20]
    )
    rows: list[DatasetRow] = []
    labels: dict[str, int] = {}
    for raw in raw_rows:
        problem_id = str(raw["problem_id"]).strip()
        task = metadata[problem_id]
        mean_input = _finite_non_negative_float(raw[input_column], input_column)
        mean_output = _finite_non_negative_float(raw[output_column], output_column)
        predicted_input = _finite_non_negative_float(
            raw[predicted_input_column], predicted_input_column
        )
        predicted_output = _finite_non_negative_float(
            raw[predicted_output_column], predicted_output_column
        )
        label = int(round(mean_input + mean_output))
        labels[problem_id] = label
        text = task.problem_statement
        task_id = f"swebench:{problem_id}"
        event_id = f"spend-your-money:{model_key}:{problem_id}:aggregate"
        point = PredictionPoint(
            point_id=(
                f"{event_id}:{PredictionPosition.TASK_LAUNCH.value}:"
                f"{PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value}"
            ),
            source_event_id=event_id,
            task_id=task_id,
            trajectory_id=f"spend-your-money:{model_key}:{problem_id}",
            run_id="four-run-aggregate",
            prediction_context_id=f"{task_id}:initial",
            condition_id=resolved_condition,
            logical_call_id=None,
            attempt_id=None,
            cutoff_event_seq=0,
            position=PredictionPosition.TASK_LAUNCH,
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            features={
                "task_char_count": len(text),
                "task_word_count": len(re.findall(r"\S+", text)),
                "task_line_count": max(1, len(text.splitlines())),
                "task_code_fence_count": text.count("```") // 2,
                "repo_id": task.repo,
                "model_id": resolved_model_id,
                "agent_id": agent_id,
                "llm_self_estimated_total_tokens": predicted_input + predicted_output,
            },
            known_offset_tokens=0,
        )
        rows.append(DatasetRow(point=point, label=label, status=LabelStatus.OBSERVED))

    resolved_metadata_hash = metadata_sha256 or _semantic_hash(
        {
            key: {
                "repo": metadata[key].repo,
                "problem_statement": metadata[key].problem_statement,
            }
            for key in sorted(set(ids))
        }
    )
    dataset_semantic = {
        "schema_version": SPEND_DATASET_SCHEMA_VERSION,
        "source_csv_sha256": csv_hash,
        "metadata_sha256": resolved_metadata_hash,
        "model_key": model_key,
        "condition_id": resolved_condition,
        "labels": labels,
    }
    dataset_id = "spend-your-money:" + _semantic_hash(dataset_semantic)
    dataset = SupervisedDataset(
        dataset_id=dataset_id,
        rows=tuple(sorted(rows, key=lambda item: item.point.point_id)),
    )
    return SpendYourMoneyImport(
        dataset=dataset,
        model_key=model_key,
        task_count=len(rows),
        source_csv_sha256=csv_hash,
        metadata_sha256=resolved_metadata_hash,
    )


def spend_import_to_dict(value: SpendYourMoneyImport) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "dataset_id": value.dataset.dataset_id,
            "model_key": value.model_key,
            "task_count": value.task_count,
            "source_csv_sha256": value.source_csv_sha256,
            "metadata_sha256": value.metadata_sha256,
            "target_definition": value.target_definition,
        }
    )


def _finite_non_negative_float(value: Any, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} contains a non-numeric value") from exc
    if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"column {column!r} must be finite and non-negative")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
