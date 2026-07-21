from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from token_prediction.features.reducer import FeatureValue


DATASET_SCHEMA_VERSION = 1
CAPABILITY_DATASET_SCHEMA_VERSION = 2


class PredictionPosition(StrEnum):
    TASK_LAUNCH = "task_launch"
    TASK_PRE = "task_pre"
    TASK_UPDATE = "task_update"
    CALL_PRE = "call_pre"
    CALL_UPDATE = "call_update"


class PredictionTarget(StrEnum):
    TASK_TOTAL_ACCOUNTED_TOKENS = "task_total_accounted_tokens"
    TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS = (
        "task_provider_accounted_remaining_tokens"
    )
    TASK_UNKNOWN_REMAINING_TOKENS = "task_unknown_remaining_tokens"
    CALL_BILLABLE_TOTAL_TOKENS = "call_billable_total_tokens"
    CALL_UNKNOWN_BILLABLE_TOKENS = "call_unknown_billable_tokens"
    CALL_BILLABLE_OUTPUT_TOKENS = "call_billable_output_tokens"
    CALL_FINAL_RESPONSE_OUTPUT_TOKENS = "call_final_response_output_tokens"
    CALL_REMAINING_OUTPUT_TOKENS = "call_remaining_output_tokens"


class LabelStatus(StrEnum):
    OBSERVED = "observed"
    CENSORED = "censored"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True)
class PredictionPoint:
    point_id: str
    source_event_id: str
    task_id: str
    trajectory_id: str
    run_id: str
    prediction_context_id: str
    condition_id: str
    logical_call_id: str | None
    attempt_id: str | None
    cutoff_event_seq: int
    position: PredictionPosition
    target: PredictionTarget
    features: Mapping[str, FeatureValue]
    known_offset_tokens: int | None

    def __post_init__(self) -> None:
        for name in (
            "point_id",
            "source_event_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "prediction_context_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff must be non-negative")
        if self.position in {PredictionPosition.CALL_PRE, PredictionPosition.CALL_UPDATE}:
            if not self.logical_call_id:
                raise ValueError("Call prediction points require logical_call_id")
        if self.known_offset_tokens is not None and self.known_offset_tokens < 0:
            raise ValueError("known token offset must be non-negative or missing")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def with_features(self, features: Mapping[str, FeatureValue]) -> "PredictionPoint":
        return replace(self, features=dict(features))


@dataclass(frozen=True)
class DatasetRow:
    point: PredictionPoint
    label: int | None
    status: LabelStatus
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        if self.status == LabelStatus.OBSERVED:
            if self.label is None or self.label < 0:
                raise ValueError("observed labels must be non-negative")
        elif self.label is not None:
            raise ValueError("non-observed labels must be None")

    @property
    def eligible(self) -> bool:
        return self.status == LabelStatus.OBSERVED


@dataclass(frozen=True)
class WeightedRow:
    row: DatasetRow
    sample_weight: float


@dataclass(frozen=True)
class DatasetSlice:
    dataset_id: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    rows: tuple[DatasetRow, ...]
    eligibility_hash: str
    weighting_id: str = "task_run_point_equal_v1"

    def weighted_rows(self) -> tuple[WeightedRow, ...]:
        by_task_trajectory: dict[str, dict[str, list[DatasetRow]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in self.rows:
            by_task_trajectory[row.point.task_id][row.point.trajectory_id].append(row)
        weighted: list[WeightedRow] = []
        for task_id in sorted(by_task_trajectory):
            trajectories = by_task_trajectory[task_id]
            run_count = len(trajectories)
            for trajectory_id in sorted(trajectories):
                points = trajectories[trajectory_id]
                weight = 1.0 / (run_count * len(points))
                weighted.extend(WeightedRow(row=row, sample_weight=weight) for row in points)
        return tuple(sorted(weighted, key=lambda item: item.row.point.point_id))


@dataclass(frozen=True)
class SupervisedDataset:
    dataset_id: str
    rows: tuple[DatasetRow, ...]
    schema_version: int = DATASET_SCHEMA_VERSION
    source_descriptor_hash: str | None = None
    capability_contract_hash: str | None = None

    def select(
        self,
        position: PredictionPosition,
        target: PredictionTarget,
        *,
        required_features: frozenset[str] = frozenset(),
        condition_id: str | None = None,
    ) -> DatasetSlice:
        rows = tuple(
            sorted(
                (
                    row
                    for row in self.rows
                    if row.eligible
                    and row.point.position == position
                    and row.point.target == target
                    and (condition_id is None or row.point.condition_id == condition_id)
                    and all(row.point.features.get(name) is not None for name in required_features)
                ),
                key=lambda row: row.point.point_id,
            )
        )
        encoded = json.dumps(
            [row.point.point_id for row in rows], separators=(",", ":")
        ).encode()
        conditions = {row.point.condition_id for row in rows}
        if len(conditions) > 1:
            raise ValueError(
                "experiment cell mixes execution conditions; select one condition_id"
            )
        resolved_condition = next(iter(conditions), condition_id or "condition:empty")
        return DatasetSlice(
            dataset_id=self.dataset_id,
            position=position,
            target=target,
            condition_id=resolved_condition,
            rows=rows,
            eligibility_hash=hashlib.sha256(encoded).hexdigest(),
        )

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(row.point.task_id for row in self.rows)
