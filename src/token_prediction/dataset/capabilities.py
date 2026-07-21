from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from token_prediction.contracts import Observable, SourceCapabilities, SourceRequirements
from token_prediction.dataset.schema import PredictionPosition, PredictionTarget


@dataclass(frozen=True)
class TargetCapabilityRequirement:
    positions: frozenset[PredictionPosition]
    requirements: SourceRequirements


def _requirement(
    positions: set[PredictionPosition],
    observables: set[Observable],
) -> TargetCapabilityRequirement:
    return TargetCapabilityRequirement(
        positions=frozenset(positions),
        requirements=SourceRequirements(observables=frozenset(observables)),
    )


TARGET_CAPABILITY_REQUIREMENTS: Mapping[
    PredictionTarget, TargetCapabilityRequirement
] = MappingProxyType(
    {
        PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS: _requirement(
            {PredictionPosition.TASK_LAUNCH},
            {Observable.TASK_USAGE, Observable.TASK_TERMINATION},
        ),
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS: _requirement(
            {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE},
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
            },
        ),
        PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS: _requirement(
            {PredictionPosition.TASK_PRE, PredictionPosition.TASK_UPDATE},
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.REQUEST_LOCAL_COUNT,
                Observable.TASK_TERMINATION,
            },
        ),
        PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS: _requirement(
            {PredictionPosition.CALL_PRE},
            {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
        ),
        PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS: _requirement(
            {PredictionPosition.CALL_PRE},
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.REQUEST_LOCAL_COUNT,
            },
        ),
        PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS: _requirement(
            {PredictionPosition.CALL_PRE},
            {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
        ),
        PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS: _requirement(
            {PredictionPosition.CALL_PRE},
            {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
        ),
        PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS: _requirement(
            {PredictionPosition.CALL_UPDATE},
            {
                Observable.ATTEMPT_USAGE,
                Observable.OUTPUT_DELTAS,
                Observable.REQUEST_BOUNDARIES,
            },
        ),
    }
)


@dataclass(frozen=True)
class CapabilityDecision:
    source_id: str
    position: PredictionPosition
    target: PredictionTarget
    capability_contract_hash: str
    required_observables: tuple[str, ...]
    missing_observables: tuple[str, ...]
    available: bool
    reason: str

    @property
    def gated(self) -> bool:
        return not self.available

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "position": self.position.value,
            "target": self.target.value,
            "capability_contract_hash": self.capability_contract_hash,
            "required_observables": list(self.required_observables),
            "missing_observables": list(self.missing_observables),
            "available": self.available,
            "reason": self.reason,
        }


def target_requirements(target: PredictionTarget | str) -> TargetCapabilityRequirement:
    resolved = PredictionTarget(target)
    return TARGET_CAPABILITY_REQUIREMENTS[resolved]


def decide_target_capability(
    capabilities: SourceCapabilities,
    position: PredictionPosition | str,
    target: PredictionTarget | str,
) -> CapabilityDecision:
    """Resolve target eligibility without ever inferring undeclared observables."""

    resolved_position = PredictionPosition(position)
    resolved_target = PredictionTarget(target)
    requirement = target_requirements(resolved_target)
    required = tuple(
        sorted(value.value for value in requirement.requirements.observables)
    )
    if resolved_position not in requirement.positions:
        return CapabilityDecision(
            source_id=capabilities.source_id,
            position=resolved_position,
            target=resolved_target,
            capability_contract_hash=capabilities.contract_hash,
            required_observables=required,
            missing_observables=(),
            available=False,
            reason="unsupported_position_target",
        )
    missing = capabilities.missing(requirement.requirements)
    return CapabilityDecision(
        source_id=capabilities.source_id,
        position=resolved_position,
        target=resolved_target,
        capability_contract_hash=capabilities.contract_hash,
        required_observables=required,
        missing_observables=missing,
        available=not missing,
        reason=("available" if not missing else f"missing_observables:{','.join(missing)}"),
    )
