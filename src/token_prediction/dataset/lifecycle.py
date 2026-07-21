from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from token_prediction.dataset.points import (
    PredictionPointSet,
    point_input_semantic,
    prediction_input_contract_hash_from_capability,
)
from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)

LIFECYCLE_SCHEMA_VERSION = 1
LIFECYCLE_WEIGHTING_ID = "task_run_point_equal_observed_v1"
_TASK_LIFECYCLE_TARGETS = frozenset(
    {
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def lifecycle_scored_hash(
    context_hash: str,
    steps: Iterable[LifecycleStep],
) -> str:
    """Bind masks and weights for every updater loss/score point."""

    _require_sha256(context_hash, name="context_hash")
    resolved = tuple(steps)
    return _canonical_sha256(
        {
            "context_hash": context_hash,
            "masked_steps": [
                {
                    "point_id": step.point.point_id,
                    "loss_mask": step.loss_mask,
                    "score_mask": step.score_mask,
                    "sample_weight": step.sample_weight,
                }
                for step in resolved
                if step.loss_mask or step.score_mask
            ],
            "weighting_id": LIFECYCLE_WEIGHTING_ID,
        }
    )


def _sequence_context_semantic(
    *,
    input_contract_hash: str,
    task_id: str,
    trajectory_id: str,
    run_id: str,
    condition_id: str,
    target: PredictionTarget,
    points: Iterable[PredictionPoint],
) -> dict[str, object]:
    return {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "input_contract_hash": input_contract_hash,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "run_id": run_id,
        "condition_id": condition_id,
        "target": target.value,
        "points": [point_input_semantic(point) for point in points],
    }


@dataclass(frozen=True)
class LifecycleStep:
    point: PredictionPoint
    label: int | None
    status: LabelStatus
    invalid_reason: str
    loss_mask: bool
    score_mask: bool
    sample_weight: float

    def __post_init__(self) -> None:
        if self.point.position not in {
            PredictionPosition.TASK_PRE,
            PredictionPosition.TASK_UPDATE,
        }:
            raise ValueError("lifecycle steps must be Task-pre or Task-update points")
        if self.status == LabelStatus.INVALID:
            raise ValueError("invalid rows cannot be lifecycle context")
        if self.status == LabelStatus.OBSERVED:
            if self.label is None or self.label < 0:
                raise ValueError("observed lifecycle labels must be non-negative")
        elif self.label is not None:
            raise ValueError("unobserved lifecycle labels must be missing")
        if (self.loss_mask or self.score_mask) and self.status != LabelStatus.OBSERVED:
            raise ValueError("only observed lifecycle steps may be scored")
        if not math.isfinite(self.sample_weight) or self.sample_weight < 0:
            raise ValueError("lifecycle sample weight must be finite and non-negative")
        if self.loss_mask or self.score_mask:
            if self.sample_weight <= 0:
                raise ValueError("scored lifecycle steps require a positive weight")
        elif self.sample_weight != 0:
            raise ValueError("unscored lifecycle context must have zero weight")

    @property
    def is_observed(self) -> bool:
        return self.status == LabelStatus.OBSERVED

    @property
    def is_context_only(self) -> bool:
        return not self.loss_mask and not self.score_mask


@dataclass(frozen=True)
class LifecycleSequence:
    dataset_id: str
    input_contract_hash: str
    task_id: str
    trajectory_id: str
    run_id: str
    condition_id: str
    target: PredictionTarget
    steps: tuple[LifecycleStep, ...]
    context_hash: str
    scored_hash: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle schema version")
        for name in (
            "dataset_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        for name in ("input_contract_hash", "context_hash", "scored_hash"):
            _require_sha256(getattr(self, name), name=name)
        if self.target not in _TASK_LIFECYCLE_TARGETS:
            raise ValueError("lifecycle target must be a Task remaining target")
        if not self.steps:
            raise ValueError("lifecycle sequence is empty")
        if self.steps[0].point.position != PredictionPosition.TASK_PRE:
            raise ValueError("lifecycle sequence must start at Task-pre")
        if any(step.point.position != PredictionPosition.TASK_UPDATE for step in self.steps[1:]):
            raise ValueError("only the first lifecycle step may be Task-pre")
        cutoffs = [step.point.cutoff_event_seq for step in self.steps]
        if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
            raise ValueError("lifecycle points must have strictly increasing cutoffs")
        for identity_name in (
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
        ):
            if any(
                getattr(step.point, identity_name) != getattr(self, identity_name)
                for step in self.steps
            ):
                raise ValueError(f"lifecycle {identity_name} is inconsistent")
        if any(step.point.target != self.target for step in self.steps):
            raise ValueError("lifecycle sequence mixes targets")
        if any(step.point.attempt_id is not None for step in self.steps):
            raise ValueError("Task lifecycle points cannot carry attempt identity")
        if self.target == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS and any(
            step.point.known_offset_tokens != 0 for step in self.steps
        ):
            raise ValueError("provider-accounted lifecycle offsets must be zero")
        point_ids = [step.point.point_id for step in self.steps]
        source_ids = [step.point.source_event_id for step in self.steps]
        call_ids = [step.point.logical_call_id for step in self.steps]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("lifecycle point ids must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("lifecycle source boundary ids must be unique")
        if any(call_id is None for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
            raise ValueError("lifecycle logical call ids must be present and unique")
        expected_context = _canonical_sha256(
            _sequence_context_semantic(
                input_contract_hash=self.input_contract_hash,
                task_id=self.task_id,
                trajectory_id=self.trajectory_id,
                run_id=self.run_id,
                condition_id=self.condition_id,
                target=self.target,
                points=(step.point for step in self.steps),
            )
        )
        if self.context_hash != expected_context:
            raise ValueError("lifecycle sequence context_hash does not match steps")
        expected_scored = lifecycle_scored_hash(self.context_hash, self.steps)
        if self.scored_hash != expected_scored:
            raise ValueError("lifecycle sequence scored_hash does not match masks")


@dataclass(frozen=True)
class LifecycleSlice:
    dataset_id: str
    input_contract_hash: str
    source_descriptor_hash: str
    capability_contract_hash: str
    condition_id: str
    target: PredictionTarget
    sequences: tuple[LifecycleSequence, ...]
    context_hash: str
    scored_hash: str
    weighting_id: str = LIFECYCLE_WEIGHTING_ID
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle schema version")
        if self.weighting_id != LIFECYCLE_WEIGHTING_ID:
            raise ValueError("unsupported lifecycle weighting policy")
        for name in (
            "dataset_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        for name in (
            "input_contract_hash",
            "source_descriptor_hash",
            "capability_contract_hash",
            "context_hash",
            "scored_hash",
        ):
            _require_sha256(getattr(self, name), name=name)
        if not self.sequences:
            raise ValueError("lifecycle slice is empty")
        identities = [
            (sequence.task_id, sequence.run_id, sequence.trajectory_id)
            for sequence in self.sequences
        ]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise ValueError("lifecycle sequences must have unique canonical identities")
        if any(
            sequence.dataset_id != self.dataset_id
            or sequence.input_contract_hash != self.input_contract_hash
            or sequence.condition_id != self.condition_id
            or sequence.target != self.target
            for sequence in self.sequences
        ):
            raise ValueError("lifecycle slice identity is inconsistent")
        by_task_run: dict[str, dict[str, list[LifecycleStep]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sequence in self.sequences:
            masked = [step for step in sequence.steps if step.loss_mask or step.score_mask]
            if masked:
                by_task_run[sequence.task_id][sequence.run_id].extend(masked)
        for runs in by_task_run.values():
            run_count = len(runs)
            for masked_steps in runs.values():
                expected_weight = 1.0 / (run_count * len(masked_steps))
                if any(
                    not math.isclose(
                        step.sample_weight,
                        expected_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    for step in masked_steps
                ):
                    raise ValueError("lifecycle task/run/point weights are inconsistent")
        expected_context = _canonical_sha256(
            {
                "lifecycle_schema_version": self.schema_version,
                "input_contract_hash": self.input_contract_hash,
                "source_descriptor_hash": self.source_descriptor_hash,
                "capability_contract_hash": self.capability_contract_hash,
                "condition_id": self.condition_id,
                "target": self.target.value,
                "sequence_context_hashes": [sequence.context_hash for sequence in self.sequences],
            }
        )
        if self.context_hash != expected_context:
            raise ValueError("lifecycle slice context_hash does not match sequences")
        expected_scored = _canonical_sha256(
            {
                "context_hash": self.context_hash,
                "weighting_id": self.weighting_id,
                "sequence_scored_hashes": [sequence.scored_hash for sequence in self.sequences],
            }
        )
        if self.scored_hash != expected_scored:
            raise ValueError("lifecycle slice scored_hash does not match sequences")

    @property
    def steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for sequence in self.sequences for step in sequence.steps)

    @property
    def scored_steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for step in self.steps if step.score_mask)

    @property
    def loss_steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for step in self.steps if step.loss_mask)

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(sequence.task_id for sequence in self.sequences)


def _fallback_input_contract_hash(
    dataset: SupervisedDataset,
) -> str:
    if dataset.input_contract_hash is not None:
        return dataset.input_contract_hash
    if dataset.source_descriptor_hash is None or dataset.capability_contract_hash is None:
        raise ValueError("lifecycle dataset is missing source/capability identity")
    return prediction_input_contract_hash_from_capability(
        capability_contract_hash=dataset.capability_contract_hash,
    )


def _validate_point_set_join(
    rows: tuple[DatasetRow, ...],
    point_set: PredictionPointSet,
    *,
    target: PredictionTarget,
    condition_id: str,
    task_ids: frozenset[str],
) -> None:
    expected = {
        point.point_id: point
        for point in point_set.points
        if point.target == target
        and point.position in {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE}
        and point.condition_id == condition_id
        and point.task_id in task_ids
    }
    if set(expected) != {row.point.point_id for row in rows}:
        raise ValueError("lifecycle rows do not have an exact prefix point join")
    for row in rows:
        if row.point != expected[row.point.point_id]:
            raise ValueError("lifecycle supervised point differs from prefix point input")


def lifecycle_condition_task_ids(
    dataset: SupervisedDataset,
    *,
    target: PredictionTarget,
    condition_id: str | None,
) -> frozenset[str]:
    """Return the label-free task universe actually present in one lifecycle cell."""

    if target not in _TASK_LIFECYCLE_TARGETS:
        raise ValueError("lifecycle target must be a Task remaining target")
    matching_rows = tuple(
        row
        for row in dataset.rows
        if row.point.target == target
        and row.point.position in {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE}
        and (condition_id is None or row.point.condition_id == condition_id)
    )
    if not matching_rows:
        raise ValueError("lifecycle condition has no matching prediction boundaries")
    conditions = {row.point.condition_id for row in matching_rows}
    if len(conditions) != 1:
        raise ValueError("lifecycle task selection must resolve exactly one condition")
    return frozenset(row.point.task_id for row in matching_rows)


def build_lifecycle_slice(
    dataset: SupervisedDataset,
    *,
    target: PredictionTarget,
    condition_id: str | None = None,
    task_ids: Iterable[str] | None = None,
    scored_task_ids: Iterable[str] | None = None,
    point_set: PredictionPointSet | None = None,
) -> LifecycleSlice:
    """Build Task-pre through Task-update sequences without dropping context.

    Missing and censored values remain in sequence order with both masks clear.
    Structurally invalid rows fail the whole trajectory instead of becoming
    context.  Weights are equal by task, then run, then observed scored point.
    """

    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise ValueError("lifecycle sequences require a capability dataset")
    if dataset.source_descriptor_hash is None or dataset.capability_contract_hash is None:
        raise ValueError("lifecycle dataset is missing source/capability identity")
    if target not in _TASK_LIFECYCLE_TARGETS:
        raise ValueError("lifecycle target must be a Task remaining target")

    condition_tasks = lifecycle_condition_task_ids(
        dataset,
        target=target,
        condition_id=condition_id,
    )
    requested_tasks = (
        None
        if task_ids is None
        else frozenset(str(task).strip() for task in task_ids if str(task).strip())
    )
    if requested_tasks is not None and not requested_tasks <= condition_tasks:
        raise ValueError("requested lifecycle tasks are absent from the selected condition")
    rows = tuple(
        row
        for row in dataset.rows
        if row.point.target == target
        and row.point.position in {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE}
        and (condition_id is None or row.point.condition_id == condition_id)
        and (requested_tasks is None or row.point.task_id in requested_tasks)
    )
    if not rows:
        raise ValueError("lifecycle slice has no matching prediction boundaries")
    point_ids = [row.point.point_id for row in rows]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("lifecycle dataset repeats point_id values")
    present_tasks = frozenset(row.point.task_id for row in rows)
    if requested_tasks is not None and requested_tasks != present_tasks:
        raise ValueError("requested lifecycle task set is not fully represented")
    conditions = {row.point.condition_id for row in rows}
    if len(conditions) != 1:
        raise ValueError("lifecycle slice must select exactly one condition")
    resolved_condition = next(iter(conditions))
    if any(row.status == LabelStatus.INVALID for row in rows):
        raise ValueError("invalid trajectory rows cannot be used as lifecycle context")

    input_contract_hash = (
        point_set.input_contract_hash
        if point_set is not None
        else _fallback_input_contract_hash(dataset)
    )
    if point_set is not None:
        if (
            point_set.source_descriptor_hash != dataset.source_descriptor_hash
            or point_set.capability_contract_hash != dataset.capability_contract_hash
        ):
            raise ValueError("prefix point set belongs to another capability contract")
        _validate_point_set_join(
            rows,
            point_set,
            target=target,
            condition_id=resolved_condition,
            task_ids=present_tasks,
        )

    selected_scored_tasks = (
        present_tasks
        if scored_task_ids is None
        else frozenset(str(task).strip() for task in scored_task_ids if str(task).strip())
    )
    if not selected_scored_tasks <= present_tasks:
        raise ValueError("scored task set is not a subset of lifecycle context")

    grouped: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.point.trajectory_id].append(row)
    ordered_rows: list[tuple[str, str, str, tuple[DatasetRow, ...]]] = []
    seen_task_runs: set[tuple[str, str]] = set()
    for trajectory_id, items in grouped.items():
        current = tuple(
            sorted(items, key=lambda row: (row.point.cutoff_event_seq, row.point.point_id))
        )
        first = current[0].point
        identity = (first.task_id, first.run_id)
        if identity in seen_task_runs:
            raise ValueError("a task/run identity maps to multiple trajectories")
        seen_task_runs.add(identity)
        ordered_rows.append((first.task_id, first.run_id, trajectory_id, current))
    ordered_rows.sort(key=lambda item: item[:3])

    scored_counts_by_task_run: dict[tuple[str, str], int] = {}
    scored_runs_by_task: dict[str, set[str]] = defaultdict(set)
    for task_id, run_id, _trajectory_id, items in ordered_rows:
        count = sum(
            row.status == LabelStatus.OBSERVED
            and row.point.position == PredictionPosition.TASK_UPDATE
            and task_id in selected_scored_tasks
            for row in items
        )
        if count:
            scored_counts_by_task_run[(task_id, run_id)] = count
            scored_runs_by_task[task_id].add(run_id)

    sequences: list[LifecycleSequence] = []
    for task_id, run_id, trajectory_id, items in ordered_rows:
        run_count = len(scored_runs_by_task.get(task_id, ()))
        point_count = scored_counts_by_task_run.get((task_id, run_id), 0)
        steps: list[LifecycleStep] = []
        for row in items:
            scored = (
                row.status == LabelStatus.OBSERVED
                and row.point.position == PredictionPosition.TASK_UPDATE
                and task_id in selected_scored_tasks
            )
            weight = 1.0 / (run_count * point_count) if scored else 0.0
            steps.append(
                LifecycleStep(
                    point=row.point,
                    label=row.label,
                    status=row.status,
                    invalid_reason=row.invalid_reason,
                    loss_mask=scored,
                    score_mask=scored,
                    sample_weight=weight,
                )
            )
        step_tuple = tuple(steps)
        context_hash = _canonical_sha256(
            _sequence_context_semantic(
                input_contract_hash=input_contract_hash,
                task_id=task_id,
                trajectory_id=trajectory_id,
                run_id=run_id,
                condition_id=resolved_condition,
                target=target,
                points=(step.point for step in step_tuple),
            )
        )
        scored_hash = lifecycle_scored_hash(context_hash, step_tuple)
        sequences.append(
            LifecycleSequence(
                dataset_id=dataset.dataset_id,
                input_contract_hash=input_contract_hash,
                task_id=task_id,
                trajectory_id=trajectory_id,
                run_id=run_id,
                condition_id=resolved_condition,
                target=target,
                steps=step_tuple,
                context_hash=context_hash,
                scored_hash=scored_hash,
            )
        )

    resolved_sequences = tuple(sequences)
    context_hash = _canonical_sha256(
        {
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "input_contract_hash": input_contract_hash,
            "source_descriptor_hash": dataset.source_descriptor_hash,
            "capability_contract_hash": dataset.capability_contract_hash,
            "condition_id": resolved_condition,
            "target": target.value,
            "sequence_context_hashes": [sequence.context_hash for sequence in resolved_sequences],
        }
    )
    scored_hash = _canonical_sha256(
        {
            "context_hash": context_hash,
            "weighting_id": LIFECYCLE_WEIGHTING_ID,
            "sequence_scored_hashes": [sequence.scored_hash for sequence in resolved_sequences],
        }
    )
    return LifecycleSlice(
        dataset_id=dataset.dataset_id,
        input_contract_hash=input_contract_hash,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
        condition_id=resolved_condition,
        target=target,
        sequences=resolved_sequences,
        context_hash=context_hash,
        scored_hash=scored_hash,
    )
