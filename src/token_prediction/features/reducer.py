from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, TypeAlias

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage


FEATURE_SCHEMA_VERSION = 2
FeatureValue: TypeAlias = int | float | str | tuple[float, ...] | None


@dataclass(frozen=True)
class FeatureSnapshot:
    point_event_id: str
    trajectory_id: str
    logical_call_id: str | None
    attempt_id: str | None
    boundary_type: EventType
    visibility_cutoff_event_seq: int
    feature_schema_version: int
    values: dict[str, FeatureValue]
    feature_hash: str


@dataclass(frozen=True)
class FeatureState:
    """Prefix-causal feature state.

    Logical-call G1 features deliberately use two layers of state.  Attempt and
    tool events first accumulate in ``current_call_*`` fields.  The next
    ``REQUEST_BUILT`` is the only boundary that closes that call and publishes
    its history features.  Consequently, a generation checkpoint can observe
    already billed cumulative usage, but never mistakes a partial retry chain
    for a completed logical call.

    Definitions fixed by this reducer:

    * a call output is the sum of billable output from *all* of its attempts;
      it is unknown when any attempt has incomplete usage, no attempt
      terminates, or a started attempt has not terminated;
    * ``recent_generated_mean_3`` covers the most recent one-to-three closed
      calls and is unknown whenever that window contains an unknown output;
    * an error round is a closed logical call containing at least one
      ``API_FAILED`` or ``TOOL_FAILED`` event;
    * ``last_tool_type`` is the most recent visible tool type in the trajectory
      prefix; a later tool-free round does not erase that history;
    * one visible terminal tool event is one explicit action.  Its key uses
      ``action_hash``, ``action_name``, ``action``, then ``tool_name`` in that
      order. ``TOOL_STARTED`` is not counted as a second action;
    * ``repeated_action_count_3`` is ``window size - unique keys`` over the
      most recent one-to-three explicit actions.
    """

    trajectory_id: str
    max_source_event_seq: int = -1
    task_tokens: int | None = None
    max_steps: int | None = None
    model_id: str | None = None
    agent_id: str | None = None
    reasoning_effort: str | None = None
    completed_logical_call_ids: frozenset[str] = frozenset()
    current_logical_call_id: str | None = None
    completed_api_attempts: int = 0
    failed_api_attempts: int = 0
    completed_tool_calls: int = 0
    failed_tool_calls: int = 0
    known_usage_attempts: int = 0
    missing_usage_attempts: int = 0
    request_count: int = 0
    cumulative_provider_input_tokens: int = 0
    cumulative_provider_output_tokens: int = 0
    last_call_output_tokens: int | None = None
    recent_call_outputs: tuple[int | None, ...] = ()
    recent_generated_mean_3: float | None = None
    last_tool_type: str | None = None
    last_round_tool_error_count: int | None = None
    consecutive_error_rounds: int = 0
    recent_action_keys: tuple[str, ...] = ()
    current_call_started_attempt_ids: frozenset[str] = frozenset()
    current_call_terminal_attempt_ids: frozenset[str] = frozenset()
    current_call_output_tokens: int = 0
    current_call_has_unknown_usage: bool = False
    current_call_last_tool_type: str | None = None
    current_call_tool_error_count: int = 0
    current_call_has_error: bool = False
    previous_request_tokens_local: int | None = None
    current_request_tokens_local: int | None = None
    request_delta_tokens: int | None = None
    context_window: int | None = None
    generated_tokens_so_far: int | None = None
    stop_prob_mean_16: float | None = None
    next_token_entropy_mean_16: float | None = None
    hidden_state_projection: tuple[float, ...] | None = None

    def apply(self, event: CanonicalEvent) -> "FeatureState":
        if event.trajectory_id != self.trajectory_id:
            raise ValueError("cannot apply an event from another trajectory")
        if event.event_seq <= self.max_source_event_seq:
            raise ValueError("events must be reduced in strictly increasing order")

        state = replace(self, max_source_event_seq=event.event_seq)
        payload = event.payload

        if event.event_type == EventType.TASK_STARTED:
            return replace(
                state,
                task_tokens=_optional_non_negative_int(payload.get("task_tokens")),
                max_steps=_optional_non_negative_int(payload.get("max_steps")),
                model_id=_optional_text(payload.get("model_id")),
                agent_id=_optional_text(payload.get("agent_id")),
                reasoning_effort=_optional_text(payload.get("reasoning_effort")),
            )

        if event.event_type == EventType.REQUEST_BUILT:
            state = state._start_logical_call(event.logical_call_id)
            current = _optional_non_negative_int(payload.get("request_tokens_local"))
            previous = self.current_request_tokens_local if self.request_count > 0 else None
            delta = current - previous if current is not None and previous is not None else None
            return replace(
                state,
                request_count=self.request_count + 1,
                previous_request_tokens_local=previous,
                current_request_tokens_local=current,
                request_delta_tokens=delta,
                context_window=_optional_non_negative_int(payload.get("context_window")),
                generated_tokens_so_far=None,
                stop_prob_mean_16=None,
                next_token_entropy_mean_16=None,
                hidden_state_projection=None,
            )

        if event.event_type == EventType.API_ATTEMPT_STARTED:
            if event.logical_call_id != state.current_logical_call_id:
                return state
            return replace(
                state,
                current_call_started_attempt_ids=(
                    state.current_call_started_attempt_ids | {str(event.attempt_id)}
                ),
            )

        if event.event_type == EventType.GENERATION_CHECKPOINT:
            projection = payload.get("hidden_state_projection")
            return replace(
                state,
                generated_tokens_so_far=_optional_non_negative_int(
                    payload.get("generated_tokens_so_far")
                ),
                stop_prob_mean_16=_optional_float(payload.get("stop_prob_mean_16")),
                next_token_entropy_mean_16=_optional_float(
                    payload.get("next_token_entropy_mean_16")
                ),
                hidden_state_projection=(
                    tuple(float(value) for value in projection)
                    if isinstance(projection, list)
                    else None
                ),
            )

        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            usage = TokenUsage.from_mapping(payload.get("usage"))
            base = replace(
                state,
                completed_api_attempts=(
                    state.completed_api_attempts
                    + (1 if event.event_type == EventType.API_COMPLETED else 0)
                ),
                failed_api_attempts=(
                    state.failed_api_attempts
                    + (1 if event.event_type == EventType.API_FAILED else 0)
                ),
            )
            if event.logical_call_id == base.current_logical_call_id:
                base = replace(
                    base,
                    current_call_terminal_attempt_ids=(
                        base.current_call_terminal_attempt_ids | {str(event.attempt_id)}
                    ),
                    current_call_has_error=(
                        base.current_call_has_error
                        or event.event_type == EventType.API_FAILED
                    ),
                )
            if not usage.is_complete:
                updates: dict[str, Any] = {
                    "missing_usage_attempts": base.missing_usage_attempts + 1,
                }
                if event.logical_call_id == base.current_logical_call_id:
                    updates["current_call_has_unknown_usage"] = True
                return replace(
                    base,
                    **updates,
                )
            updates = {
                "known_usage_attempts": base.known_usage_attempts + 1,
                "cumulative_provider_input_tokens": (
                    base.cumulative_provider_input_tokens + int(usage.input_tokens or 0)
                ),
                "cumulative_provider_output_tokens": (
                    base.cumulative_provider_output_tokens + int(usage.output_tokens or 0)
                ),
            }
            if event.logical_call_id == base.current_logical_call_id:
                updates["current_call_output_tokens"] = (
                    base.current_call_output_tokens + int(usage.output_tokens or 0)
                )
            return replace(
                base,
                **updates,
            )

        if event.event_type in {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
        }:
            return state._apply_tool_event(event)
        return state

    def _start_logical_call(self, logical_call_id: str | None) -> "FeatureState":
        """Close the preceding call, publish its G1 facts, and reset scratch state."""

        if self.current_logical_call_id is None:
            return replace(
                self,
                current_logical_call_id=logical_call_id,
                current_call_started_attempt_ids=frozenset(),
                current_call_terminal_attempt_ids=frozenset(),
                current_call_output_tokens=0,
                current_call_has_unknown_usage=False,
                current_call_last_tool_type=None,
                current_call_tool_error_count=0,
                current_call_has_error=False,
            )

        attempts_are_complete = (
            bool(self.current_call_terminal_attempt_ids)
            and self.current_call_started_attempt_ids
            == self.current_call_terminal_attempt_ids
            and not self.current_call_has_unknown_usage
        )
        closed_output = self.current_call_output_tokens if attempts_are_complete else None
        recent_outputs = (*self.recent_call_outputs, closed_output)[-3:]
        recent_mean = (
            sum(value for value in recent_outputs if value is not None)
            / len(recent_outputs)
            if recent_outputs and all(value is not None for value in recent_outputs)
            else None
        )
        consecutive_errors = (
            self.consecutive_error_rounds + 1 if self.current_call_has_error else 0
        )
        return replace(
            self,
            completed_logical_call_ids=(
                self.completed_logical_call_ids | {self.current_logical_call_id}
            ),
            current_logical_call_id=logical_call_id,
            last_call_output_tokens=closed_output,
            recent_call_outputs=recent_outputs,
            recent_generated_mean_3=recent_mean,
            last_tool_type=(
                self.current_call_last_tool_type
                if self.current_call_last_tool_type is not None
                else self.last_tool_type
            ),
            last_round_tool_error_count=self.current_call_tool_error_count,
            consecutive_error_rounds=consecutive_errors,
            current_call_started_attempt_ids=frozenset(),
            current_call_terminal_attempt_ids=frozenset(),
            current_call_output_tokens=0,
            current_call_has_unknown_usage=False,
            current_call_last_tool_type=None,
            current_call_tool_error_count=0,
            current_call_has_error=False,
        )

    def _apply_tool_event(self, event: CanonicalEvent) -> "FeatureState":
        """Apply visible tool facts without counting start/terminal pairs twice."""

        payload = event.payload
        updates: dict[str, Any] = {}
        tool_type = _first_optional_text(payload, "tool_name", "tool_type")
        belongs_to_current_call = event.logical_call_id == self.current_logical_call_id
        if belongs_to_current_call and tool_type is not None:
            updates["current_call_last_tool_type"] = tool_type

        if event.event_type == EventType.TOOL_COMPLETED:
            updates["completed_tool_calls"] = self.completed_tool_calls + 1
        elif event.event_type == EventType.TOOL_FAILED:
            updates["failed_tool_calls"] = self.failed_tool_calls + 1
            if belongs_to_current_call:
                updates["current_call_tool_error_count"] = (
                    self.current_call_tool_error_count + 1
                )
                updates["current_call_has_error"] = True

        # A terminal tool event represents one observable action.  TOOL_STARTED
        # may repeat the same payload and is intentionally excluded.
        if event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
            action_key = _explicit_action_key(payload)
            if action_key is not None:
                updates["recent_action_keys"] = (*self.recent_action_keys, action_key)[-3:]
        return replace(self, **updates) if updates else self

    def snapshot(self, boundary: CanonicalEvent) -> FeatureSnapshot:
        if self.max_source_event_seq < 0:
            raise ValueError("cannot snapshot an empty state")
        if boundary.event_seq != self.max_source_event_seq:
            raise ValueError("snapshot boundary must be the most recently applied event")
        if boundary.event_type != EventType.TASK_STARTED and not boundary.logical_call_id:
            raise ValueError("Call/request prediction boundary requires logical_call_id")
        context_utilization = (
            self.current_request_tokens_local / self.context_window
            if self.current_request_tokens_local is not None
            and self.context_window is not None
            and self.context_window > 0
            else None
        )
        values: dict[str, FeatureValue] = {
            "task_tokens": self.task_tokens,
            "max_steps": self.max_steps,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "reasoning_effort": self.reasoning_effort,
            "completed_call_count": len(self.completed_logical_call_ids),
            "completed_api_attempts": self.completed_api_attempts,
            "failed_api_attempts": self.failed_api_attempts,
            "completed_tool_calls": self.completed_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "known_usage_attempts": self.known_usage_attempts,
            "missing_usage_attempts": self.missing_usage_attempts,
            "request_count": self.request_count,
            "cumulative_provider_input_tokens": self.cumulative_provider_input_tokens,
            "cumulative_provider_output_tokens": self.cumulative_provider_output_tokens,
            "last_call_output_tokens": self.last_call_output_tokens,
            "recent_generated_mean_3": self.recent_generated_mean_3,
            "last_tool_type": self.last_tool_type,
            "last_round_tool_error_count": self.last_round_tool_error_count,
            "consecutive_error_rounds": self.consecutive_error_rounds,
            "repeated_action_count_3": (
                len(self.recent_action_keys) - len(set(self.recent_action_keys))
            ),
            "current_request_tokens_local": self.current_request_tokens_local,
            "request_delta_tokens": self.request_delta_tokens,
            "context_utilization": context_utilization,
        }
        if boundary.event_type == EventType.GENERATION_CHECKPOINT:
            values.update(
                {
                    "generated_tokens_so_far": self.generated_tokens_so_far,
                    "stop_prob_mean_16": self.stop_prob_mean_16,
                    "next_token_entropy_mean_16": self.next_token_entropy_mean_16,
                    "hidden_state_projection": self.hidden_state_projection,
                }
            )
        semantic = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "boundary_type": boundary.event_type.value,
            "values": values,
        }
        encoded = json.dumps(
            semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return FeatureSnapshot(
            point_event_id=boundary.event_id,
            trajectory_id=self.trajectory_id,
            logical_call_id=boundary.logical_call_id,
            attempt_id=boundary.attempt_id,
            boundary_type=boundary.event_type,
            visibility_cutoff_event_seq=self.max_source_event_seq,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            values=values,
            feature_hash=hashlib.sha256(encoded).hexdigest(),
        )


def replay_feature_snapshots(
    events: Iterable[CanonicalEvent],
    *,
    include_task_started: bool = False,
) -> list[FeatureSnapshot]:
    ordered = sorted(events, key=lambda event: event.event_seq)
    if not ordered:
        return []
    trajectory_ids = {event.trajectory_id for event in ordered}
    if len(trajectory_ids) != 1:
        raise ValueError("replay_feature_snapshots expects one trajectory")
    state = FeatureState(trajectory_id=ordered[0].trajectory_id)
    snapshots: list[FeatureSnapshot] = []
    for event in ordered:
        state = state.apply(event)
        if event.event_type in {EventType.REQUEST_BUILT, EventType.GENERATION_CHECKPOINT} or (
            include_task_started and event.event_type == EventType.TASK_STARTED
        ):
            snapshots.append(state.snapshot(event))
    return snapshots


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counts must be non-negative integers")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_optional_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        text = _optional_text(payload.get(key))
        if text is not None:
            return text
    return None


def _explicit_action_key(payload: dict[str, Any]) -> str | None:
    """Return the highest-fidelity explicit action identity in fixed priority order."""

    for key in ("action_hash", "action_name", "action", "tool_name"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            rendered = value.strip()
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        else:
            rendered = str(value).strip()
        if rendered:
            return rendered
    return None
