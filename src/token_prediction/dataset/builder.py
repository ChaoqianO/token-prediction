from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Iterable, Mapping

from token_prediction.contracts import EventType, Observable, SourceDescriptor
from token_prediction.dataset.capabilities import decide_target_capability
from token_prediction.dataset.labels import (
    build_generation_labels,
    build_prediction_labels,
    build_task_aggregate_label,
)
from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    DatasetRow,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, replay_feature_snapshots
from token_prediction.features.reducer import FeatureValue
from token_prediction.trajectory import Trajectory


V2_EXCLUDED_LOCAL_FEATURES = frozenset(
    {
        "current_request_tokens_local",
        "request_delta_tokens",
        "context_utilization",
    }
)

_V2_CAPABILITY_CELLS = (
    (PredictionPosition.TASK_LAUNCH, PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS),
    (
        PredictionPosition.TASK_PRE,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
    ),
    (
        PredictionPosition.TASK_UPDATE,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
    ),
    (PredictionPosition.TASK_PRE, PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS),
    (PredictionPosition.TASK_UPDATE, PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS),
    (PredictionPosition.CALL_PRE, PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS),
    (PredictionPosition.CALL_PRE, PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS),
    (PredictionPosition.CALL_PRE, PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS),
    (
        PredictionPosition.CALL_PRE,
        PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
    ),
    (PredictionPosition.CALL_UPDATE, PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS),
)


def _point_id(
    source_event_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
) -> str:
    return f"{source_event_id}:{position.value}:{target.value}"


def _context_id(trajectory: Trajectory, event_payload: dict[str, object], call_id: str) -> str:
    explicit = str(
        event_payload.get("prediction_context_id")
        or event_payload.get("state_id")
        or event_payload.get("request_hash")
        or ""
    ).strip()
    return explicit or f"{trajectory.prediction_context_id}:call:{call_id}"


def _task_launch_row(trajectory: Trajectory) -> DatasetRow:
    snapshots = [
        snapshot
        for snapshot in replay_feature_snapshots(
            trajectory.events, include_task_started=True
        )
        if snapshot.boundary_type == EventType.TASK_STARTED
    ]
    if len(snapshots) != 1:
        raise ValueError("trajectory requires exactly one task launch feature snapshot")
    snapshot = snapshots[0]
    label = build_task_aggregate_label(trajectory.events)
    if label.point_event_id != snapshot.point_event_id:
        raise ValueError("task launch snapshot and label point ids differ")
    point = PredictionPoint(
        point_id=_point_id(
            snapshot.point_event_id,
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ),
        source_event_id=snapshot.point_event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=trajectory.prediction_context_id,
        condition_id=trajectory.condition_id,
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=snapshot.visibility_cutoff_event_seq,
        position=PredictionPosition.TASK_LAUNCH,
        target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        features=snapshot.values,
        known_offset_tokens=0,
    )
    return DatasetRow(
        point=point,
        label=label.total_accounted_tokens.value,
        status=label.total_accounted_tokens.status,
        invalid_reason=label.total_accounted_tokens.reason,
    )


def _request_rows(trajectory: Trajectory) -> list[DatasetRow]:
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(trajectory.events)
        if snapshot.boundary_type == EventType.REQUEST_BUILT
    }
    labels = {label.point_event_id: label for label in build_prediction_labels(trajectory.events)}
    request_events = [
        event for event in trajectory.events if event.event_type == EventType.REQUEST_BUILT
    ]
    request_ids = {event.event_id for event in request_events}
    if set(snapshots) != set(labels) or set(labels) != request_ids:
        raise ValueError("request snapshots and labels do not form a one-to-one point-id join")

    rows: list[DatasetRow] = []
    for request_index, event in enumerate(request_events):
        snapshot = snapshots[event.event_id]
        label = labels[event.event_id]
        request_tokens = snapshot.values.get("current_request_tokens_local")
        known_offset = int(request_tokens) if isinstance(request_tokens, int) else None
        task_position = (
            PredictionPosition.TASK_PRE
            if request_index == 0
            else PredictionPosition.TASK_UPDATE
        )
        task_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                task_position,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=(
                trajectory.prediction_context_id
                if request_index == 0
                else _context_id(trajectory, event.payload, str(event.logical_call_id))
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=task_position,
            target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=task_point,
                label=label.task_unknown_remaining.value,
                status=label.task_unknown_remaining.status,
                invalid_reason=label.task_unknown_remaining.reason,
            )
        )

        call_unknown_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=call_unknown_point,
                label=label.call_unknown_billable.value,
                status=label.call_unknown_billable.status,
                invalid_reason=label.call_unknown_billable.reason,
            )
        )

        call_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=call_point,
                label=label.call_billable_output.value,
                status=label.call_billable_output.status,
                invalid_reason=label.call_billable_output.reason,
            )
        )
        final_response_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=final_response_point,
                label=label.final_response_output.value,
                status=label.final_response_output.status,
                invalid_reason=label.final_response_output.reason,
            )
        )
    return rows


def _generation_rows(trajectory: Trajectory) -> list[DatasetRow]:
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(trajectory.events)
        if snapshot.boundary_type == EventType.GENERATION_CHECKPOINT
    }
    labels = {label.point_event_id: label for label in build_generation_labels(trajectory.events)}
    if set(snapshots) != set(labels):
        raise ValueError("generation snapshots and labels do not form a one-to-one point-id join")
    rows: list[DatasetRow] = []
    for event in (
        item for item in trajectory.events if item.event_type == EventType.GENERATION_CHECKPOINT
    ):
        snapshot = snapshots[event.event_id]
        label = labels[event.event_id]
        point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_UPDATE,
                PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=str(event.attempt_id),
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_UPDATE,
            target=PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=0,
        )
        rows.append(
            DatasetRow(
                point=point,
                label=label.remaining_output.value,
                status=label.remaining_output.status,
                invalid_reason=label.remaining_output.reason,
            )
        )
    return rows


def build_supervised_dataset(trajectories: Iterable[Trajectory]) -> SupervisedDataset:
    rows: list[DatasetRow] = []
    for trajectory in tuple(trajectories):
        rows.append(_task_launch_row(trajectory))
        rows.extend(_request_rows(trajectory))
        rows.extend(_generation_rows(trajectory))
    rows.sort(key=lambda row: row.point.point_id)
    if len({row.point.point_id for row in rows}) != len(rows):
        raise ValueError("dataset point_id values must be unique")
    semantic = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "rows": [
            {
                "point": {
                    "point_id": row.point.point_id,
                    "source_event_id": row.point.source_event_id,
                    "task_id": row.point.task_id,
                    "trajectory_id": row.point.trajectory_id,
                    "run_id": row.point.run_id,
                    "prediction_context_id": row.point.prediction_context_id,
                    "condition_id": row.point.condition_id,
                    "logical_call_id": row.point.logical_call_id,
                    "attempt_id": row.point.attempt_id,
                    "cutoff_event_seq": row.point.cutoff_event_seq,
                    "position": row.point.position.value,
                    "target": row.point.target.value,
                    "features": dict(row.point.features),
                    "known_offset_tokens": row.point.known_offset_tokens,
                },
                "label": row.label,
                "status": row.status.value,
                "invalid_reason": row.invalid_reason,
            }
            for row in rows
        ],
    }
    encoded = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return SupervisedDataset(
        dataset_id=hashlib.sha256(encoded).hexdigest(),
        rows=tuple(rows),
    )


def _v2_features(
    features: Mapping[str, FeatureValue],
    source_descriptor: SourceDescriptor,
) -> dict[str, FeatureValue]:
    resolved = dict(features)
    if Observable.REQUEST_LOCAL_COUNT in source_descriptor.capabilities.observables:
        return resolved
    return {
        name: value
        for name, value in resolved.items()
        if name not in V2_EXCLUDED_LOCAL_FEATURES
    }


def _v2_point(
    point: PredictionPoint,
    source_descriptor: SourceDescriptor,
) -> PredictionPoint:
    has_local_count = (
        Observable.REQUEST_LOCAL_COUNT
        in source_descriptor.capabilities.observables
    )
    return replace(
        point,
        features=_v2_features(dict(point.features), source_descriptor),
        known_offset_tokens=(
            point.known_offset_tokens
            if has_local_count
            and point.target
            in {
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            }
            else 0
        ),
    )


def _row_semantic(row: DatasetRow) -> dict[str, object]:
    return {
        "point": {
            "point_id": row.point.point_id,
            "source_event_id": row.point.source_event_id,
            "task_id": row.point.task_id,
            "trajectory_id": row.point.trajectory_id,
            "run_id": row.point.run_id,
            "prediction_context_id": row.point.prediction_context_id,
            "condition_id": row.point.condition_id,
            "logical_call_id": row.point.logical_call_id,
            "attempt_id": row.point.attempt_id,
            "cutoff_event_seq": row.point.cutoff_event_seq,
            "position": row.point.position.value,
            "target": row.point.target.value,
            "features": dict(row.point.features),
            "known_offset_tokens": row.point.known_offset_tokens,
        },
        "label": row.label,
        "status": row.status.value,
        "invalid_reason": row.invalid_reason,
    }


def _provider_accounted_request_rows(
    trajectory: Trajectory,
    descriptor: SourceDescriptor,
) -> list[DatasetRow]:
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(trajectory.events)
        if snapshot.boundary_type == EventType.REQUEST_BUILT
    }
    labels = {
        label.point_event_id: label
        for label in build_prediction_labels(trajectory.events)
    }
    request_events = [
        event
        for event in trajectory.events
        if event.event_type == EventType.REQUEST_BUILT
    ]
    request_ids = {event.event_id for event in request_events}
    if set(snapshots) != set(labels) or set(labels) != request_ids:
        raise ValueError("request snapshots and labels do not form a one-to-one point-id join")

    rows: list[DatasetRow] = []
    for request_index, event in enumerate(request_events):
        snapshot = snapshots[event.event_id]
        label = labels[event.event_id]
        task_position = (
            PredictionPosition.TASK_PRE
            if request_index == 0
            else PredictionPosition.TASK_UPDATE
        )
        task_target = PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        if decide_target_capability(
            descriptor.capabilities, task_position, task_target
        ).available:
            point = PredictionPoint(
                point_id=_point_id(event.event_id, task_position, task_target),
                source_event_id=event.event_id,
                task_id=trajectory.task_id,
                trajectory_id=trajectory.trajectory_id,
                run_id=trajectory.run_id,
                prediction_context_id=(
                    trajectory.prediction_context_id
                    if request_index == 0
                    else _context_id(
                        trajectory, event.payload, str(event.logical_call_id)
                    )
                ),
                condition_id=trajectory.condition_id,
                logical_call_id=str(event.logical_call_id),
                attempt_id=None,
                cutoff_event_seq=event.event_seq,
                position=task_position,
                target=task_target,
                features=_v2_features(snapshot.values, descriptor),
                known_offset_tokens=0,
            )
            task_value = label.task_provider_accounted_remaining
            rows.append(
                DatasetRow(
                    point=point,
                    label=task_value.value,
                    status=task_value.status,
                    invalid_reason=task_value.reason,
                )
            )

        call_target = PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS
        if decide_target_capability(
            descriptor.capabilities,
            PredictionPosition.CALL_PRE,
            call_target,
        ).available:
            call_point = PredictionPoint(
                point_id=_point_id(
                    event.event_id,
                    PredictionPosition.CALL_PRE,
                    call_target,
                ),
                source_event_id=event.event_id,
                task_id=trajectory.task_id,
                trajectory_id=trajectory.trajectory_id,
                run_id=trajectory.run_id,
                prediction_context_id=_context_id(
                    trajectory, event.payload, str(event.logical_call_id)
                ),
                condition_id=trajectory.condition_id,
                logical_call_id=str(event.logical_call_id),
                attempt_id=None,
                cutoff_event_seq=event.event_seq,
                position=PredictionPosition.CALL_PRE,
                target=call_target,
                features=_v2_features(snapshot.values, descriptor),
                known_offset_tokens=0,
            )
            call_value = label.call_billable_total
            rows.append(
                DatasetRow(
                    point=call_point,
                    label=call_value.value,
                    status=call_value.status,
                    invalid_reason=call_value.reason,
                )
            )
    return rows


def build_capability_supervised_dataset(
    trajectories: Iterable[Trajectory],
    source_descriptor: SourceDescriptor,
) -> SupervisedDataset:
    """Build schema-v2 rows strictly from the declared source capability contract.

    Unavailable targets are omitted and remain inspectable through
    :func:`decide_target_capability`. Local-count-derived point features and
    offsets are admitted only when the source explicitly declares a genuine
    local request count. Provider-accounted targets always keep a zero offset.
    """

    resolved_trajectories = tuple(trajectories)
    legacy = build_supervised_dataset(resolved_trajectories)
    rows = [
        replace(row, point=_v2_point(row.point, source_descriptor))
        for row in legacy.rows
        if decide_target_capability(
            source_descriptor.capabilities,
            row.point.position,
            row.point.target,
        ).available
    ]
    for trajectory in resolved_trajectories:
        rows.extend(_provider_accounted_request_rows(trajectory, source_descriptor))
    rows.sort(key=lambda row: row.point.point_id)
    if len({row.point.point_id for row in rows}) != len(rows):
        raise ValueError("dataset point_id values must be unique")

    decisions = [
        decide_target_capability(
            source_descriptor.capabilities,
            position,
            target,
        ).to_dict()
        for position, target in _V2_CAPABILITY_CELLS
    ]
    semantic = {
        "schema_version": CAPABILITY_DATASET_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_descriptor": source_descriptor.to_dict(),
        "capability_contract_hash": source_descriptor.capabilities.contract_hash,
        "capability_decisions": decisions,
        "excluded_local_features": (
            []
            if Observable.REQUEST_LOCAL_COUNT
            in source_descriptor.capabilities.observables
            else sorted(V2_EXCLUDED_LOCAL_FEATURES)
        ),
        "rows": [_row_semantic(row) for row in rows],
    }
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SupervisedDataset(
        dataset_id=hashlib.sha256(encoded).hexdigest(),
        rows=tuple(rows),
        schema_version=CAPABILITY_DATASET_SCHEMA_VERSION,
        source_descriptor_hash=source_descriptor.descriptor_hash,
        capability_contract_hash=source_descriptor.capabilities.contract_hash,
    )
