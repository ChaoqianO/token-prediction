from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .reducer import FeatureValue


class FeatureGroup(StrEnum):
    G0 = "g0"
    G1 = "g1"
    G2 = "g2"
    G3 = "g3"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: FeatureGroup
    subgroup: str
    dtype: str


class FeatureCatalog:
    def __init__(self, specs: tuple[FeatureSpec, ...]) -> None:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("feature names must be unique")
        self._by_name = by_name

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"feature {name!r} is not registered") from exc

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)


DEFAULT_FEATURE_CATALOG = FeatureCatalog(
    (
        FeatureSpec("task_tokens", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_char_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_word_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_line_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_code_fence_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_embedding", FeatureGroup.G0, "task_text", "vector"),
        FeatureSpec("repo_id", FeatureGroup.G0, "task_metadata", "category"),
        FeatureSpec(
            "llm_self_estimated_total_tokens",
            FeatureGroup.G0,
            "baseline_prediction",
            "numeric",
        ),
        FeatureSpec("max_steps", FeatureGroup.G0, "agent_config", "numeric"),
        FeatureSpec("model_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("agent_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("reasoning_effort", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec(
            "similar_task_total_tokens_median",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_total_tokens_iqr",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_call_count_median",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_mean_similarity",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec("completed_call_count", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("completed_api_attempts", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("failed_api_attempts", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("completed_tool_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("failed_tool_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("known_usage_attempts", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("missing_usage_attempts", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("request_count", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("step_progress_ratio", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec(
            "cumulative_provider_input_tokens", FeatureGroup.G1, "progress_tokens", "numeric"
        ),
        FeatureSpec(
            "cumulative_provider_output_tokens", FeatureGroup.G1, "progress_tokens", "numeric"
        ),
        FeatureSpec("last_call_output_tokens", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("recent_generated_mean_3", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("last_tool_type", FeatureGroup.G1, "tools_errors", "category"),
        FeatureSpec(
            "last_round_tool_error_count", FeatureGroup.G1, "tools_errors", "numeric"
        ),
        FeatureSpec("consecutive_error_rounds", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("repeated_action_count_3", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec(
            "current_request_tokens_local", FeatureGroup.G2, "request_context", "numeric"
        ),
        FeatureSpec("request_delta_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("context_utilization", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("new_message_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("new_tool_output_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("reused_context_ratio", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("new_context_embedding", FeatureGroup.G2, "context_text", "vector"),
        FeatureSpec("generated_tokens_so_far", FeatureGroup.G3, "generation_progress", "numeric"),
        FeatureSpec("stop_prob_mean_16", FeatureGroup.G3, "entropy_stop", "numeric"),
        FeatureSpec(
            "next_token_entropy_mean_16", FeatureGroup.G3, "entropy_stop", "numeric"
        ),
        FeatureSpec(
            "hidden_state_projection", FeatureGroup.G3, "hidden_state", "vector"
        ),
    )
)


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    include_all: bool = True
    include_groups: frozenset[FeatureGroup] = frozenset()
    exclude_groups: frozenset[FeatureGroup] = frozenset()
    include_subgroups: frozenset[str] = frozenset()
    exclude_subgroups: frozenset[str] = frozenset()
    include_features: frozenset[str] = frozenset()
    exclude_features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if self.include_groups & self.exclude_groups:
            raise ValueError("a feature group cannot be both included and excluded")
        if self.include_features & self.exclude_features:
            raise ValueError("a feature cannot be both included and excluded")

    @property
    def content_hash(self) -> str:
        payload = {
            "include_all": self.include_all,
            "include_groups": sorted(value.value for value in self.include_groups),
            "exclude_groups": sorted(value.value for value in self.exclude_groups),
            "include_subgroups": sorted(self.include_subgroups),
            "exclude_subgroups": sorted(self.exclude_subgroups),
            "include_features": sorted(self.include_features),
            "exclude_features": sorted(self.exclude_features),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def select(
        self,
        values: Mapping[str, FeatureValue],
        *,
        catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
    ) -> dict[str, FeatureValue]:
        unknown = set(values) - catalog.names
        if unknown:
            raise KeyError(f"unregistered features: {', '.join(sorted(unknown))}")
        selected: dict[str, FeatureValue] = {}
        explicit_includes = bool(
            self.include_groups or self.include_subgroups or self.include_features
        )
        for name, value in values.items():
            spec = catalog.get(name)
            include = self.include_all and not explicit_includes
            include = include or spec.group in self.include_groups
            include = include or spec.subgroup in self.include_subgroups
            include = include or name in self.include_features
            if spec.group in self.exclude_groups:
                include = False
            if spec.subgroup in self.exclude_subgroups:
                include = False
            if name in self.exclude_features:
                include = False
            if include:
                selected[name] = value
        return selected


FULL_FEATURE_SET = FeatureSet("full")
NO_FEATURES = FeatureSet("none", include_all=False)
REQUEST_LENGTH_ONLY = FeatureSet(
    "request_length_only",
    include_all=False,
    include_features=frozenset({"current_request_tokens_local"}),
)
