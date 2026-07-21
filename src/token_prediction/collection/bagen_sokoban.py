from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory


class BagenSokobanSchemaError(ValueError):
    """Raised when a preserved BAGEN rollout cannot be normalized honestly."""


@dataclass(frozen=True)
class BagenSokobanMetadata:
    benchmark_id: str = "bagen-sokoban"
    agent_id: str = "bagen-coord-sokoban"
    reasoning_effort: str | None = None
    condition_id: str | None = None


class BagenSokobanReader:
    """Normalize a BAGEN Sokoban dialogue JSON into canonical trajectories.

    One BAGEN file contains many rollouts.  Repeated rollouts of the same
    puzzle are grouped by a hash of the *initial state*, not by rollout index.

    BAGEN records provider input usage after a call, but not a separately
    computed local request count.  The canonical ``request_tokens_local``
    field therefore remains missing.  Provider input usage is retained only
    as an explicitly post-response audit field on the attempt terminal event.
    This reader deliberately does not advertise the ``REQUEST_LOCAL_COUNT``
    capability.
    """

    source_id = "bagen_sokoban_dialogues_v1"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset(
            {
                Observable.TASK_USAGE,
                Observable.CALL_USAGE,
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_MESSAGES,
                Observable.TOOL_EVENTS,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
            }
        ),
        source="declared",
    )

    def read_all(
        self,
        location: str | Path,
        metadata: BagenSokobanMetadata | None = None,
    ) -> tuple[Trajectory, ...]:
        source = Path(location).resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BagenSokobanSchemaError(f"cannot read BAGEN JSON: {exc}") from exc
        if not isinstance(payload, list) or not payload:
            raise BagenSokobanSchemaError("BAGEN dialogue file must be a non-empty list")
        resolved = metadata or BagenSokobanMetadata()
        trajectories = tuple(
            self._normalize_rollout(source, index, item, resolved)
            for index, item in enumerate(payload)
        )
        trajectory_ids = [item.trajectory_id for item in trajectories]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise BagenSokobanSchemaError("normalized trajectory ids are not unique")
        return trajectories

    def _normalize_rollout(
        self,
        source: Path,
        rollout_index: int,
        value: Any,
        metadata: BagenSokobanMetadata,
    ) -> Trajectory:
        if not isinstance(value, Mapping):
            raise BagenSokobanSchemaError(f"rollout {rollout_index} is not an object")
        rollout = dict(value)
        initial_state = _required_text(rollout, "initial_state", rollout_index)
        turns = rollout.get("turns")
        if not isinstance(turns, list) or not turns:
            raise BagenSokobanSchemaError(f"rollout {rollout_index} has no turns")

        state_semantic = {
            "benchmark_id": metadata.benchmark_id,
            "tag": str(rollout.get("tag") or "CoordSokoban"),
            "initial_state": initial_state.strip(),
        }
        state_hash = _semantic_hash(state_semantic)
        task_id = f"{metadata.benchmark_id}:task:{state_hash[:20]}"
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"{metadata.benchmark_id}:trajectory:{absolute_id}:{rollout_index}:"
            f"{state_hash[:12]}"
        )

        model_ids = {
            str(interaction.get("model") or "").strip()
            for turn in turns
            if isinstance(turn, Mapping)
            for interaction in (turn.get("api_interactions") or [])
            if isinstance(interaction, Mapping)
            and str(interaction.get("model") or "").strip()
        }
        if len(model_ids) > 1:
            raise BagenSokobanSchemaError(
                f"rollout {rollout_index} mixes models: {sorted(model_ids)}"
            )
        model_id = next(iter(model_ids), "unknown-model")
        condition_id = metadata.condition_id or (
            "condition:"
            + _semantic_hash(
                {
                    "benchmark_id": metadata.benchmark_id,
                    "agent_id": metadata.agent_id,
                    "model_id": model_id,
                    "reasoning_effort": metadata.reasoning_effort,
                }
            )[:20]
        )

        first_turn = turns[0]
        if not isinstance(first_turn, Mapping):
            raise BagenSokobanSchemaError(f"rollout {rollout_index} turn 0 is not an object")
        events: list[CanonicalEvent] = []
        base_time = datetime(2000, 1, 1, tzinfo=UTC)

        def append(
            event_type: EventType,
            *,
            payload: Mapping[str, Any] | None = None,
            logical_call_id: str | None = None,
            attempt_id: str | None = None,
            raw_ref: str,
        ) -> None:
            seq = len(events)
            events.append(
                CanonicalEvent.create(
                    schema_version=1,
                    event_id=f"{trajectory_id}:event:{seq}",
                    trajectory_id=trajectory_id,
                    event_seq=seq,
                    event_type=event_type,
                    occurred_at=(base_time + timedelta(microseconds=seq)).isoformat(),
                    logical_call_id=logical_call_id,
                    attempt_id=attempt_id,
                    raw_ref=raw_ref,
                    payload=payload or {},
                )
            )

        append(
            EventType.TASK_STARTED,
            payload={
                "task_id": task_id,
                "run_id": trajectory_id,
                "prediction_context_id": f"{task_id}:initial",
                "condition_id": condition_id,
                "task_tokens": None,
                "task_tokens_source": "missing",
                "model_id": model_id,
                "agent_id": metadata.agent_id,
                "reasoning_effort": metadata.reasoning_effort,
                "source_env_id": rollout.get("env_id"),
                "source_absolute_env_id": rollout.get("absolute_env_id"),
                "initial_state_hash": state_hash,
            },
            raw_ref=f"{source.name}#/{rollout_index}",
        )

        summed_input = 0
        summed_output = 0
        for turn_index, turn_value in enumerate(turns):
            if not isinstance(turn_value, Mapping):
                raise BagenSokobanSchemaError(
                    f"rollout {rollout_index} turn {turn_index} is not an object"
                )
            turn = dict(turn_value)
            call_id = f"{trajectory_id}:call:{turn_index}"
            turn_ref = f"{source.name}#/{rollout_index}/turns/{turn_index}"
            append(
                EventType.REQUEST_BUILT,
                logical_call_id=call_id,
                raw_ref=turn_ref,
                payload={
                    "request_tokens_local": None,
                    "request_token_count_source": "missing",
                    "prediction_context_id": f"{task_id}:turn:{turn_index}",
                    "source_turn_idx": turn.get("turn_idx", turn_index + 1),
                },
            )

            interactions = turn.get("api_interactions")
            if not isinstance(interactions, list) or not interactions:
                raise BagenSokobanSchemaError(
                    f"rollout {rollout_index} turn {turn_index} has no api_interactions"
                )
            turn_input = 0
            turn_output = 0
            for interaction_index, interaction_value in enumerate(interactions):
                if not isinstance(interaction_value, Mapping):
                    raise BagenSokobanSchemaError(
                        f"rollout {rollout_index} turn {turn_index} interaction "
                        f"{interaction_index} is not an object"
                    )
                interaction = dict(interaction_value)
                attempt_number = interaction.get("attempt", interaction_index + 1)
                attempt_id = f"{call_id}:attempt:{attempt_number}:{interaction_index}"
                interaction_ref = f"{turn_ref}/api_interactions/{interaction_index}"
                append(
                    EventType.API_ATTEMPT_STARTED,
                    logical_call_id=call_id,
                    attempt_id=attempt_id,
                    raw_ref=interaction_ref,
                    payload={
                        "provider": interaction.get("provider"),
                        "model": interaction.get("model"),
                        "request_id": interaction.get("request_id"),
                    },
                )
                usage = _usage_mapping(interaction)
                provider_input_audit = _provider_input_post_response_audit(
                    interaction
                )
                if usage is not None:
                    turn_input += int(usage["input_tokens"])
                    turn_output += int(usage["output_tokens"])
                terminal_type = (
                    EventType.API_COMPLETED
                    if bool(interaction.get("success"))
                    else EventType.API_FAILED
                )
                append(
                    terminal_type,
                    logical_call_id=call_id,
                    attempt_id=attempt_id,
                    raw_ref=interaction_ref,
                    payload={
                        "usage": usage,
                        "error": interaction.get("error"),
                        "error_type": interaction.get("error_type"),
                        "status_code": interaction.get("status_code"),
                        "retryable": interaction.get("retryable"),
                        "provider_input_tokens_post_response_audit": (
                            provider_input_audit
                        ),
                        "provider_input_tokens_post_response_audit_source": (
                            "provider_response_usage"
                            if provider_input_audit is not None
                            else "missing"
                        ),
                    },
                )

            _validate_optional_total(
                rollout_index,
                turn_index,
                "api_input_tokens",
                turn.get("api_input_tokens"),
                turn_input,
            )
            _validate_optional_total(
                rollout_index,
                turn_index,
                "api_output_tokens",
                turn.get("api_output_tokens"),
                turn_output,
            )
            summed_input += turn_input
            summed_output += turn_output

            actions = turn.get("actions") or []
            if not isinstance(actions, list):
                raise BagenSokobanSchemaError(
                    f"rollout {rollout_index} turn {turn_index} actions is not a list"
                )
            action_names = turn.get("action_names") or []
            for action_index, action in enumerate(actions):
                append(
                    EventType.TOOL_COMPLETED,
                    logical_call_id=call_id,
                    raw_ref=f"{turn_ref}/actions/{action_index}",
                    payload={
                        "tool_name": "sokoban_action",
                        "action": action,
                        "action_name": (
                            action_names[action_index]
                            if isinstance(action_names, list)
                            and action_index < len(action_names)
                            else None
                        ),
                    },
                )

        task_usage = _rollout_usage(rollout, rollout_index)
        if (
            summed_input != task_usage["input_tokens"]
            or summed_output != task_usage["output_tokens"]
        ):
            raise BagenSokobanSchemaError(
                f"rollout {rollout_index} aggregate usage disagrees with interactions"
            )
        last_turn = turns[-1]
        rollout_success = (
            bool(last_turn.get("success"))
            if isinstance(last_turn, Mapping)
            else False
        )
        append(
            EventType.TASK_FINISHED,
            payload={
                "outcome": "success" if rollout_success else "unsolved",
                "reason": "agent_finished",
                "usage": task_usage,
            },
            raw_ref=f"{source.name}#/{rollout_index}",
        )
        return Trajectory.from_events(events)


def _required_text(value: Mapping[str, Any], key: str, rollout_index: int) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise BagenSokobanSchemaError(f"rollout {rollout_index} requires {key}")
    return text


def _semantic_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provider_input_post_response_audit(
    interaction: Mapping[str, Any],
) -> int | None:
    raw_value = interaction.get("input_tokens")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise BagenSokobanSchemaError(
            "provider input token audit must be an integer"
        ) from exc
    if value < 0:
        raise BagenSokobanSchemaError(
            "provider input token audit must be non-negative"
        )
    return value


def _usage_mapping(interaction: Mapping[str, Any]) -> dict[str, int] | None:
    input_tokens = interaction.get("input_tokens")
    output_tokens = interaction.get("output_tokens")
    total_tokens = interaction.get("total_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    parsed_input = int(input_tokens)
    parsed_output = int(output_tokens)
    parsed_total = (
        int(total_tokens)
        if total_tokens is not None
        else parsed_input + parsed_output
    )
    if min(parsed_input, parsed_output, parsed_total) < 0:
        raise BagenSokobanSchemaError("usage token counts must be non-negative")
    if parsed_input + parsed_output != parsed_total:
        raise BagenSokobanSchemaError("interaction token total is inconsistent")
    return {
        "input_tokens": parsed_input,
        "output_tokens": parsed_output,
        "total_tokens": parsed_total,
    }


def _rollout_usage(rollout: Mapping[str, Any], rollout_index: int) -> dict[str, int]:
    try:
        input_tokens = int(rollout["api_input_tokens"])
        output_tokens = int(rollout["api_output_tokens"])
        total_tokens = int(rollout["api_total_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BagenSokobanSchemaError(
            f"rollout {rollout_index} requires aggregate token usage"
        ) from exc
    if min(input_tokens, output_tokens, total_tokens) < 0:
        raise BagenSokobanSchemaError("rollout usage must be non-negative")
    if input_tokens + output_tokens != total_tokens:
        raise BagenSokobanSchemaError(
            f"rollout {rollout_index} aggregate token total is inconsistent"
        )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _validate_optional_total(
    rollout_index: int,
    turn_index: int,
    name: str,
    raw_value: Any,
    computed: int,
) -> None:
    if raw_value is None:
        if computed != 0:
            raise BagenSokobanSchemaError(
                f"rollout {rollout_index} turn {turn_index} omits {name} "
                "despite observed interaction usage"
            )
        return
    if int(raw_value) != computed:
        raise BagenSokobanSchemaError(
            f"rollout {rollout_index} turn {turn_index} {name} mismatch"
        )
