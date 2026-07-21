"""Prefix-only derived features that leave the frozen source dataset untouched."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Iterable, Mapping

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory

from .schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    SupervisedDataset,
)
from .points import prediction_input_contract_hash_from_capability


REQUEST_SHAPE_PROJECTION_ID = "request_boundary_shape_v1"
REQUEST_SHAPE_FEATURES = ("request_message_count", "request_content_chars")
AGGREGATE_TASK_SHAPE_PROJECTION_ID = "spend_aggregate_task_shape_v1"
AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID = (
    "spend_aggregate_task_launch_input_contract_v1"
)
AGGREGATE_TASK_SHAPE_FEATURES = (
    "agent_id",
    "llm_self_estimated_total_tokens",
    "model_id",
    "repo_id",
    "task_char_count",
    "task_code_fence_count",
    "task_line_count",
    "task_word_count",
)
_REQUEST_SHAPE_CONTRACT = {
    "projection_id": REQUEST_SHAPE_PROJECTION_ID,
    "visibility": "same_request_built_boundary",
    "source_payload_fields": list(REQUEST_SHAPE_FEATURES),
    "output_features": [
        {"name": "request_message_count", "dtype": "optional_non_negative_integer"},
        {"name": "request_content_chars", "dtype": "optional_non_negative_integer"},
    ],
    "forbidden_inputs": ["labels", "post_response_usage", "future_events"],
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def request_shape_input_contract_hash(base_input_contract_hash: str) -> str:
    if (
        not isinstance(base_input_contract_hash, str)
        or len(base_input_contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in base_input_contract_hash)
    ):
        raise ValueError("base_input_contract_hash must be a lowercase SHA-256 digest")
    return _canonical_sha256(
        {
            "base_input_contract_hash": base_input_contract_hash,
            "derived_feature_contract": _REQUEST_SHAPE_CONTRACT,
        }
    )


def aggregate_task_shape_input_contract_hash(
    capability_contract_hash: str,
) -> str:
    """Bind the audited aggregate-only Task-launch feature contract."""

    if (
        not isinstance(capability_contract_hash, str)
        or len(capability_contract_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capability_contract_hash
        )
    ):
        raise ValueError("capability_contract_hash must be a lowercase SHA-256 digest")
    return _canonical_sha256(
        {
            "policy_id": AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID,
            "capability_contract_hash": capability_contract_hash,
            "features": list(AGGREGATE_TASK_SHAPE_FEATURES),
        }
    )


def supported_input_contract_hashes_from_capability(
    capability_contract_hash: str,
    *,
    capabilities: SourceCapabilities | None = None,
) -> frozenset[str]:
    """Return the explicitly reviewed raw and derived point-contract hashes."""

    base = prediction_input_contract_hash_from_capability(
        capability_contract_hash=capability_contract_hash,
    )
    supported = {base, request_shape_input_contract_hash(base)}
    if capabilities is not None:
        if capabilities.contract_hash != capability_contract_hash:
            raise ValueError("capabilities do not match the declared contract hash")
        if Observable.TASK_AGGREGATE_USAGE in capabilities.observables:
            supported.add(
                aggregate_task_shape_input_contract_hash(capability_contract_hash)
            )
    return frozenset(supported)


def request_shape_projection_document() -> Mapping[str, Any]:
    """Return the observation-independent projection contract for provenance."""

    return json.loads(json.dumps(_REQUEST_SHAPE_CONTRACT, sort_keys=True))


def _optional_non_negative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or missing")
    return value


def augment_request_shape_features(
    dataset: SupervisedDataset,
    trajectories: Iterable[Trajectory],
) -> SupervisedDataset:
    """Create a derived dataset using only facts visible at ``REQUEST_BUILT``.

    The parent dataset and its frozen identity are never mutated.  The derived
    identity binds the parent plus the visible projection, while the input
    contract binds only schemas and capability policy, never observed values or
    labels.
    """

    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise ValueError("request-shape projection requires a capability dataset")
    if dataset.input_contract_hash is None:
        raise ValueError("source dataset is missing its input contract hash")
    resolved_trajectories = tuple(trajectories)
    if not resolved_trajectories:
        raise ValueError("request-shape projection requires trajectories")
    by_trajectory: dict[str, Trajectory] = {}
    events: dict[tuple[str, str], CanonicalEvent] = {}
    prediction_boundaries = {
        EventType.TASK_STARTED,
        EventType.REQUEST_BUILT,
        EventType.GENERATION_CHECKPOINT,
    }
    for trajectory in resolved_trajectories:
        if trajectory.trajectory_id in by_trajectory:
            raise ValueError("request-shape projection repeats trajectory_id")
        by_trajectory[trajectory.trajectory_id] = trajectory
        for event in trajectory.events:
            if event.event_type not in prediction_boundaries:
                continue
            key = (trajectory.trajectory_id, event.event_id)
            if key in events:
                raise ValueError("request-shape projection repeats an event identity")
            events[key] = event
    dataset_trajectories = {row.point.trajectory_id for row in dataset.rows}
    if dataset_trajectories != set(by_trajectory):
        raise ValueError("request-shape trajectories do not exactly cover the dataset")

    projected_records: list[dict[str, object]] = []
    rows: list[DatasetRow] = []
    for row in dataset.rows:
        point = row.point
        trajectory = by_trajectory[point.trajectory_id]
        if point.task_id != trajectory.task_id or point.run_id != trajectory.run_id:
            raise ValueError("request-shape point and trajectory identity differ")
        try:
            event = events[(point.trajectory_id, point.source_event_id)]
        except KeyError as exc:
            raise ValueError("request-shape point boundary is missing from trajectories") from exc
        if event.event_seq != point.cutoff_event_seq:
            raise ValueError("request-shape point cutoff differs from its boundary")
        if any(name in point.features for name in REQUEST_SHAPE_FEATURES):
            raise ValueError("request-shape features are already present")

        features = dict(point.features)
        record: dict[str, object] = {"point_id": point.point_id}
        if event.event_type == EventType.REQUEST_BUILT:
            payload = event.payload
            for name in REQUEST_SHAPE_FEATURES:
                value = _optional_non_negative_integer(payload.get(name), name=name)
                features[name] = value
                record[name] = value
        else:
            record["not_request_boundary"] = True
        projected_records.append(record)
        rows.append(replace(row, point=point.with_features(features)))

    input_contract_hash = request_shape_input_contract_hash(dataset.input_contract_hash)
    dataset_id = _canonical_sha256(
        {
            "derived_dataset_schema_version": 1,
            "parent_dataset_id": dataset.dataset_id,
            "projection_id": REQUEST_SHAPE_PROJECTION_ID,
            "input_contract_hash": input_contract_hash,
            "projection_records": sorted(
                projected_records, key=lambda record: str(record["point_id"])
            ),
        }
    )
    return SupervisedDataset(
        dataset_id=dataset_id,
        rows=tuple(rows),
        schema_version=dataset.schema_version,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
        input_contract_hash=input_contract_hash,
    )


__all__ = [
    "AGGREGATE_TASK_SHAPE_FEATURES",
    "AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID",
    "AGGREGATE_TASK_SHAPE_PROJECTION_ID",
    "REQUEST_SHAPE_FEATURES",
    "REQUEST_SHAPE_PROJECTION_ID",
    "aggregate_task_shape_input_contract_hash",
    "augment_request_shape_features",
    "request_shape_input_contract_hash",
    "request_shape_projection_document",
    "supported_input_contract_hashes_from_capability",
]
