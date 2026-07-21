from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from token_prediction.dataset.points import point_input_semantic
from token_prediction.dataset.schema import SupervisedDataset
from token_prediction.dataset.splits import (
    DEFAULT_FINAL_HOLDOUT_BUCKET_COUNT,
    DEFAULT_FINAL_HOLDOUT_BUCKET_THRESHOLD,
    DEFAULT_FINAL_HOLDOUT_SALT,
    INNER_FOLDS,
    InnerTaskFoldAssignment,
    PermanentHoldoutPlan,
    SplitPlan,
    assign_inner_task_folds,
    assign_permanent_task_holdout,
    assign_task_folds,
)


STAGE_SPLIT_SEEDS = (20260719, 20260720, 20260721)
OUTER_FOLDS = 5
DEVELOPMENT_PROTOCOL_SCHEMA_VERSION = 1
DEVELOPMENT_PROTOCOL_POLICY_ID = "permanent_holdout_three_seed_nested_cv_v1"
TASK_PSEUDONYM_POLICY_ID = "sha256_development_task_pseudonym_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _task_pseudonym(task_id: str) -> str:
    return hashlib.sha256(f"{TASK_PSEUDONYM_POLICY_ID}\0{task_id}".encode("utf-8")).hexdigest()


def _require_nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_nonempty_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, name=name)


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name=name)


def _development_dataset_id(
    dataset: SupervisedDataset,
    *,
    holdout_assignment_id: str,
    development_tasks: Iterable[str],
) -> str:
    """Bind the exact development projection without consulting holdout rows."""

    selected_tasks = frozenset(development_tasks)
    selected_rows = tuple(
        sorted(
            (row for row in dataset.rows if row.point.task_id in selected_tasks),
            key=lambda row: row.point.point_id,
        )
    )
    if frozenset(row.point.task_id for row in selected_rows) != selected_tasks:
        raise ValueError("development identity tasks do not match the projected rows")
    semantic = {
        "identity_schema_version": DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
        "dataset_schema_version": dataset.schema_version,
        "source_descriptor_hash": dataset.source_descriptor_hash,
        "capability_contract_hash": dataset.capability_contract_hash,
        "input_contract_hash": dataset.input_contract_hash,
        "holdout_assignment_id": holdout_assignment_id,
        "selected_task_pseudonyms": sorted(
            _task_pseudonym(task_id) for task_id in selected_tasks
        ),
        "rows": [
            {
                "point": point_input_semantic(row.point),
                "label": row.label,
                "status": row.status.value,
                "invalid_reason": row.invalid_reason,
            }
            for row in selected_rows
        ],
    }
    return _canonical_sha256(semantic)


def _protocol_identity(
    *,
    parent_schema_version: int,
    source_descriptor_hash: str | None,
    capability_contract_hash: str | None,
    input_contract_hash: str | None,
    development_dataset_id: str,
    holdout_assignment_id: str,
    outer_plans: Iterable[tuple[int, int, str, str]],
    inner_plans: Iterable[tuple[int, int, str]],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
        "policy_id": DEVELOPMENT_PROTOCOL_POLICY_ID,
        "development_contract": {
            "schema_version": parent_schema_version,
            "source_descriptor_hash": source_descriptor_hash,
            "capability_contract_hash": capability_contract_hash,
            "input_contract_hash": input_contract_hash,
        },
        "development_dataset_id": development_dataset_id,
        "holdout_assignment_id": holdout_assignment_id,
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "outer_plans": [
            {
                "seed": seed,
                "folds": folds,
                "split_plan_id": split_plan_id,
                "assignment_id": assignment_id,
            }
            for seed, folds, split_plan_id, assignment_id in outer_plans
        ],
        "inner_plans": [
            {
                "seed": seed,
                "outer_test_fold": outer_test_fold,
                "assignment_id": assignment_id,
            }
            for seed, outer_test_fold, assignment_id in inner_plans
        ],
    }


@dataclass(frozen=True)
class OuterInnerPlan:
    split_seed: int
    outer_test_fold: int
    outer_split_plan_id: str
    assignment: InnerTaskFoldAssignment

    def __post_init__(self) -> None:
        if self.split_seed not in STAGE_SPLIT_SEEDS:
            raise ValueError("inner plan uses an unapproved split seed")
        if not 0 <= self.outer_test_fold < OUTER_FOLDS:
            raise ValueError("inner plan outer test fold is out of range")
        if not self.outer_split_plan_id:
            raise ValueError("inner plan must bind an outer split plan")
        if self.assignment.seed != self.split_seed:
            raise ValueError("inner assignment seed differs from its outer split seed")
        if self.assignment.folds != INNER_FOLDS:
            raise ValueError("inner assignment must contain exactly five folds")


@dataclass(frozen=True)
class NestedDevelopmentPlan:
    """One frozen outer plan and its five prevalidated inner assignments."""

    outer_plan: SplitPlan
    inner_plans: tuple[OuterInnerPlan, ...]

    def __post_init__(self) -> None:
        if self.outer_plan.seed not in STAGE_SPLIT_SEEDS:
            raise ValueError("nested development plan uses an unapproved split seed")
        if len(self.inner_plans) != OUTER_FOLDS:
            raise ValueError("nested development plan requires five inner assignments")
        for outer_test_fold, inner_plan in enumerate(self.inner_plans):
            if (
                inner_plan.split_seed != self.outer_plan.seed
                or inner_plan.outer_test_fold != outer_test_fold
                or inner_plan.outer_split_plan_id != self.outer_plan.split_plan_id
            ):
                raise ValueError("inner assignment is not bound to its outer fold")

    @property
    def split_seed(self) -> int:
        return self.outer_plan.seed


@dataclass(frozen=True)
class DevelopmentProtocol:
    protocol_id: str
    parent_dataset_id: str
    parent_schema_version: int
    parent_source_descriptor_hash: str | None
    parent_capability_contract_hash: str | None
    parent_input_contract_hash: str | None
    holdout_plan: PermanentHoldoutPlan
    development_dataset: SupervisedDataset
    outer_plans: tuple[SplitPlan, ...]
    inner_plans: tuple[OuterInnerPlan, ...]

    def __post_init__(self) -> None:
        if not self.parent_dataset_id:
            raise ValueError("parent dataset id is required")
        if self.parent_schema_version < 1:
            raise ValueError("parent dataset schema version must be positive")
        if self.holdout_plan.dataset_id != self.parent_dataset_id:
            raise ValueError("holdout plan is not bound to the parent dataset")
        if self.development_dataset.schema_version != self.parent_schema_version:
            raise ValueError("development dataset changed the parent schema version")
        if self.development_dataset.source_descriptor_hash != self.parent_source_descriptor_hash:
            raise ValueError("development dataset changed the source descriptor hash")
        if (
            self.development_dataset.capability_contract_hash
            != self.parent_capability_contract_hash
        ):
            raise ValueError("development dataset changed the capability contract hash")
        if self.development_dataset.input_contract_hash != self.parent_input_contract_hash:
            raise ValueError("development dataset changed the input contract hash")

        development_tasks = self.holdout_plan.development_tasks
        final_holdout_tasks = self.holdout_plan.final_holdout_tasks
        if not development_tasks or not final_holdout_tasks:
            raise ValueError("development and final-holdout cohorts must both be non-empty")
        if development_tasks & final_holdout_tasks:
            raise ValueError("development and final-holdout tasks overlap")
        if self.development_dataset.task_ids != development_tasks:
            raise ValueError("development dataset is not an exact task projection")
        if self.development_dataset.task_ids & final_holdout_tasks:
            raise ValueError("final-holdout rows leaked into the development dataset")

        expected_dataset_id = _development_dataset_id(
            self.development_dataset,
            holdout_assignment_id=self.holdout_plan.assignment_id,
            development_tasks=development_tasks,
        )
        if self.development_dataset.dataset_id != expected_dataset_id:
            raise ValueError("development dataset id does not match its sealed identity")

        if tuple(plan.seed for plan in self.outer_plans) != STAGE_SPLIT_SEEDS:
            raise ValueError("development protocol requires the three frozen split seeds")
        if len(self.outer_plans) != len(STAGE_SPLIT_SEEDS):
            raise ValueError("development protocol requires exactly three outer plans")

        expected_inner_keys: set[tuple[int, int]] = set()
        outer_by_seed: dict[int, SplitPlan] = {}
        for plan in self.outer_plans:
            expected = assign_task_folds(
                development_tasks,
                folds=OUTER_FOLDS,
                seed=plan.seed,
            ).bind(self.development_dataset.dataset_id)
            if plan != expected:
                raise ValueError("outer split plan violates the frozen task-only policy")
            if plan.dataset_id != self.development_dataset.dataset_id:
                raise ValueError("outer split plan is not bound to the development dataset")
            plan.validate_tasks(development_tasks)
            plan_tasks = frozenset(task for task, _fold in plan.assignments)
            if plan_tasks & final_holdout_tasks:
                raise ValueError("final-holdout tasks leaked into an outer split")
            outer_by_seed[plan.seed] = plan
            expected_inner_keys.update((plan.seed, fold) for fold in range(OUTER_FOLDS))

        actual_inner_keys = {(plan.split_seed, plan.outer_test_fold) for plan in self.inner_plans}
        if len(actual_inner_keys) != len(self.inner_plans):
            raise ValueError("duplicate outer-fold inner plans are not allowed")
        if actual_inner_keys != expected_inner_keys:
            raise ValueError("every outer train partition requires exactly one inner plan")

        for inner_plan in self.inner_plans:
            outer_plan = outer_by_seed[inner_plan.split_seed]
            if inner_plan.outer_split_plan_id != outer_plan.split_plan_id:
                raise ValueError("inner plan is not bound to its outer split plan")
            outer_train = outer_plan.partition(inner_plan.outer_test_fold).train_tasks
            if outer_train & final_holdout_tasks:
                raise ValueError("final-holdout tasks leaked into an outer-train cohort")
            expected_assignment = assign_inner_task_folds(
                outer_train,
                seed=inner_plan.split_seed,
            )
            if inner_plan.assignment != expected_assignment:
                raise ValueError("inner plan violates the frozen five-fold task policy")
            inner_plan.assignment.validate_tasks(outer_train)
            if inner_plan.assignment.task_ids & final_holdout_tasks:
                raise ValueError("final-holdout tasks leaked into an inner split")
            for holdout_fold in range(INNER_FOLDS):
                partition = inner_plan.assignment.partition(holdout_fold)
                if not (
                    partition.initializer_fit_tasks
                    and partition.validation_tasks
                    and partition.holdout_tasks
                ):
                    raise ValueError("inner fit/validation/holdout partition is empty")
                if (
                    partition.initializer_fit_tasks
                    | partition.validation_tasks
                    | partition.holdout_tasks
                ) != outer_train:
                    raise ValueError("inner partitions do not cover the outer-train cohort")

        expected_protocol_id = _canonical_sha256(
            _protocol_identity(
                parent_schema_version=self.parent_schema_version,
                source_descriptor_hash=self.parent_source_descriptor_hash,
                capability_contract_hash=self.parent_capability_contract_hash,
                input_contract_hash=self.parent_input_contract_hash,
                development_dataset_id=self.development_dataset.dataset_id,
                holdout_assignment_id=self.holdout_plan.assignment_id,
                outer_plans=(
                    (
                        plan.seed,
                        plan.folds,
                        plan.split_plan_id,
                        plan.assignment_id,
                    )
                    for plan in self.outer_plans
                ),
                inner_plans=(
                    (
                        plan.split_seed,
                        plan.outer_test_fold,
                        plan.assignment.assignment_id,
                    )
                    for plan in self.inner_plans
                ),
            )
        )
        if self.protocol_id != expected_protocol_id:
            raise ValueError("development protocol id does not match its identity")

    @property
    def split_seeds(self) -> tuple[int, int, int]:
        return STAGE_SPLIT_SEEDS

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return self.holdout_plan.final_holdout_tasks

    @property
    def outer_inner_plans(self) -> tuple[NestedDevelopmentPlan, ...]:
        """Expose each production outer plan with its frozen inner OOF plans."""

        inner_by_key = {(plan.split_seed, plan.outer_test_fold): plan for plan in self.inner_plans}
        return tuple(
            NestedDevelopmentPlan(
                outer_plan=outer_plan,
                inner_plans=tuple(
                    inner_by_key[(outer_plan.seed, fold)] for fold in range(OUTER_FOLDS)
                ),
            )
            for outer_plan in self.outer_plans
        )

    def nested_plan_for(self, split_plan: SplitPlan) -> NestedDevelopmentPlan:
        """Resolve a production split by exact identity, failing closed otherwise."""

        matches = tuple(
            plan
            for plan in self.outer_inner_plans
            if plan.outer_plan.split_plan_id == split_plan.split_plan_id
        )
        if len(matches) != 1 or matches[0].outer_plan != split_plan:
            raise ValueError("split plan is not part of the development protocol")
        return matches[0]

    def to_audit_document(self) -> dict[str, object]:
        """Return a public, pseudonymized, checksum-protected audit projection."""

        task_stats: list[dict[str, object]] = []
        for task_id in sorted(
            self.development_dataset.task_ids,
            key=_task_pseudonym,
        ):
            task_rows = tuple(
                row for row in self.development_dataset.rows if row.point.task_id == task_id
            )
            task_stats.append(
                {
                    "task_pseudonym": _task_pseudonym(task_id),
                    "row_count": len(task_rows),
                    "run_count": len({row.point.run_id for row in task_rows}),
                    "condition_count": len({row.point.condition_id for row in task_rows}),
                }
            )

        outer_documents: list[dict[str, object]] = []
        inner_by_outer = {
            (plan.split_seed, plan.outer_test_fold): plan for plan in self.inner_plans
        }
        for outer in self.outer_plans:
            inner_documents: list[dict[str, object]] = []
            for outer_test_fold in range(OUTER_FOLDS):
                inner = inner_by_outer[(outer.seed, outer_test_fold)]
                inner_documents.append(
                    {
                        "outer_test_fold": outer_test_fold,
                        "assignment_id": inner.assignment.assignment_id,
                        "policy_id": inner.assignment.policy_id,
                        "folds": inner.assignment.folds,
                        "assignments": [
                            {
                                "task_pseudonym": _task_pseudonym(task_id),
                                "fold": fold,
                            }
                            for task_id, fold in sorted(
                                inner.assignment.assignments,
                                key=lambda item: _task_pseudonym(item[0]),
                            )
                        ],
                    }
                )
            outer_documents.append(
                {
                    "seed": outer.seed,
                    "folds": outer.folds,
                    "split_plan_id": outer.split_plan_id,
                    "assignment_id": outer.assignment_id,
                    "assignments": [
                        {
                            "task_pseudonym": _task_pseudonym(task_id),
                            "fold": fold,
                        }
                        for task_id, fold in sorted(
                            outer.assignments,
                            key=lambda item: _task_pseudonym(item[0]),
                        )
                    ],
                    "inner_plans": inner_documents,
                }
            )

        payload: dict[str, object] = {
            "audit_schema_version": DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
            "policy_id": DEVELOPMENT_PROTOCOL_POLICY_ID,
            "task_pseudonym_policy_id": TASK_PSEUDONYM_POLICY_ID,
            "protocol_id": self.protocol_id,
            "parent_dataset": {
                "dataset_id": self.parent_dataset_id,
                "schema_version": self.parent_schema_version,
                "source_descriptor_hash": self.parent_source_descriptor_hash,
                "capability_contract_hash": self.parent_capability_contract_hash,
                "input_contract_hash": self.parent_input_contract_hash,
            },
            "development_dataset": {
                "dataset_id": self.development_dataset.dataset_id,
                "schema_version": self.development_dataset.schema_version,
                "source_descriptor_hash": (self.development_dataset.source_descriptor_hash),
                "capability_contract_hash": (self.development_dataset.capability_contract_hash),
                "input_contract_hash": self.development_dataset.input_contract_hash,
                "row_count": len(self.development_dataset.rows),
                "task_count": len(task_stats),
                "tasks": task_stats,
            },
            "permanent_holdout": {
                "holdout_plan_id": self.holdout_plan.holdout_plan_id,
                "assignment_id": self.holdout_plan.assignment_id,
                "policy_id": self.holdout_plan.policy_id,
                "salt_sha256": hashlib.sha256(self.holdout_plan.salt.encode("utf-8")).hexdigest(),
                "bucket_count": self.holdout_plan.bucket_count,
                "final_holdout_bucket_threshold_exclusive": (
                    self.holdout_plan.final_holdout_bucket_threshold_exclusive
                ),
                "assignments": [
                    {
                        "task_pseudonym": _task_pseudonym(task_id),
                        "cohort": cohort,
                    }
                    for task_id, cohort in sorted(
                        self.holdout_plan.assignments,
                        key=lambda item: _task_pseudonym(item[0]),
                    )
                ],
            },
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "outer_plans": outer_documents,
        }
        payload["audit_sha256"] = _canonical_sha256(payload)
        return payload


def build_development_protocol(
    dataset: SupervisedDataset,
    *,
    holdout_salt: str = DEFAULT_FINAL_HOLDOUT_SALT,
    holdout_bucket_count: int = DEFAULT_FINAL_HOLDOUT_BUCKET_COUNT,
    final_holdout_bucket_threshold_exclusive: int = (DEFAULT_FINAL_HOLDOUT_BUCKET_THRESHOLD),
) -> DevelopmentProtocol:
    """Seal final-holdout tasks before deriving the three development CV plans."""

    if not dataset.dataset_id:
        raise ValueError("parent dataset id is required")
    if not dataset.rows:
        raise ValueError("parent dataset must contain rows")
    point_ids = [row.point.point_id for row in dataset.rows]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("parent dataset point ids must be unique")

    holdout_assignment = assign_permanent_task_holdout(
        dataset.task_ids,
        salt=holdout_salt,
        bucket_count=holdout_bucket_count,
        final_holdout_bucket_threshold_exclusive=(final_holdout_bucket_threshold_exclusive),
        minimum_development_tasks=15,
    )
    holdout_plan = holdout_assignment.bind(dataset.dataset_id)
    development_tasks = holdout_plan.development_tasks
    development_rows = tuple(
        sorted(
            (row for row in dataset.rows if row.point.task_id in development_tasks),
            key=lambda row: row.point.point_id,
        )
    )
    development_dataset = SupervisedDataset(
        dataset_id=_development_dataset_id(
            dataset,
            holdout_assignment_id=holdout_assignment.assignment_id,
            development_tasks=development_tasks,
        ),
        rows=development_rows,
        schema_version=dataset.schema_version,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
        input_contract_hash=dataset.input_contract_hash,
    )

    outer_plans: list[SplitPlan] = []
    inner_plans: list[OuterInnerPlan] = []
    for seed in STAGE_SPLIT_SEEDS:
        outer = assign_task_folds(
            development_tasks,
            folds=OUTER_FOLDS,
            seed=seed,
        ).bind(development_dataset.dataset_id)
        outer_plans.append(outer)
        for outer_test_fold in range(OUTER_FOLDS):
            outer_train = outer.partition(outer_test_fold).train_tasks
            inner_assignment = assign_inner_task_folds(outer_train, seed=seed)
            for inner_holdout_fold in range(INNER_FOLDS):
                partition = inner_assignment.partition(inner_holdout_fold)
                if not (
                    partition.initializer_fit_tasks
                    and partition.validation_tasks
                    and partition.holdout_tasks
                ):
                    raise ValueError("inner fit/validation/holdout partition is empty")
            inner_plans.append(
                OuterInnerPlan(
                    split_seed=seed,
                    outer_test_fold=outer_test_fold,
                    outer_split_plan_id=outer.split_plan_id,
                    assignment=inner_assignment,
                )
            )

    identity = _protocol_identity(
        parent_schema_version=dataset.schema_version,
        source_descriptor_hash=dataset.source_descriptor_hash,
        capability_contract_hash=dataset.capability_contract_hash,
        input_contract_hash=dataset.input_contract_hash,
        development_dataset_id=development_dataset.dataset_id,
        holdout_assignment_id=holdout_plan.assignment_id,
        outer_plans=(
            (plan.seed, plan.folds, plan.split_plan_id, plan.assignment_id) for plan in outer_plans
        ),
        inner_plans=(
            (
                plan.split_seed,
                plan.outer_test_fold,
                plan.assignment.assignment_id,
            )
            for plan in inner_plans
        ),
    )
    return DevelopmentProtocol(
        protocol_id=_canonical_sha256(identity),
        parent_dataset_id=dataset.dataset_id,
        parent_schema_version=dataset.schema_version,
        parent_source_descriptor_hash=dataset.source_descriptor_hash,
        parent_capability_contract_hash=dataset.capability_contract_hash,
        parent_input_contract_hash=dataset.input_contract_hash,
        holdout_plan=holdout_plan,
        development_dataset=development_dataset,
        outer_plans=tuple(outer_plans),
        inner_plans=tuple(inner_plans),
    )


def _parse_public_assignments(
    value: object,
    *,
    name: str,
    value_key: str,
    allowed_values: frozenset[object],
) -> tuple[tuple[str, object], ...]:
    entries = _require_sequence(value, name=name)
    parsed: list[tuple[str, object]] = []
    for index, item in enumerate(entries):
        entry = _require_mapping(item, name=f"{name}[{index}]")
        _require_exact_keys(
            entry,
            frozenset({"task_pseudonym", value_key}),
            name=f"{name}[{index}]",
        )
        pseudonym = _require_sha256(
            entry["task_pseudonym"],
            name=f"{name}[{index}].task_pseudonym",
        )
        resolved = entry[value_key]
        if resolved not in allowed_values:
            raise ValueError(f"{name}[{index}].{value_key} is invalid")
        parsed.append((pseudonym, resolved))
    if [task for task, _value in parsed] != sorted(task for task, _value in parsed):
        raise ValueError(f"{name} must use canonical task-pseudonym order")
    if len({task for task, _value in parsed}) != len(parsed):
        raise ValueError(f"{name} contains duplicate task pseudonyms")
    return tuple(parsed)


def verify_development_audit_document(document: Mapping[str, object]) -> None:
    """Verify checksum, public shape, nested split coverage, and holdout sealing."""

    root = _require_mapping(document, name="audit document")
    _require_exact_keys(
        root,
        frozenset(
            {
                "audit_schema_version",
                "policy_id",
                "task_pseudonym_policy_id",
                "protocol_id",
                "parent_dataset",
                "development_dataset",
                "permanent_holdout",
                "split_seeds",
                "outer_plans",
                "audit_sha256",
            }
        ),
        name="audit document",
    )
    supplied_audit_sha = _require_sha256(root["audit_sha256"], name="audit document.audit_sha256")
    unhashed = dict(root)
    del unhashed["audit_sha256"]
    if supplied_audit_sha != _canonical_sha256(unhashed):
        raise ValueError("development audit checksum mismatch")
    if root["audit_schema_version"] != DEVELOPMENT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported development audit schema version")
    if root["policy_id"] != DEVELOPMENT_PROTOCOL_POLICY_ID:
        raise ValueError("unsupported development protocol policy")
    if root["task_pseudonym_policy_id"] != TASK_PSEUDONYM_POLICY_ID:
        raise ValueError("unsupported task pseudonym policy")
    protocol_id = _require_sha256(root["protocol_id"], name="protocol_id")

    parent = _require_mapping(root["parent_dataset"], name="parent_dataset")
    _require_exact_keys(
        parent,
        frozenset(
            {
                "dataset_id",
                "schema_version",
                "source_descriptor_hash",
                "capability_contract_hash",
                "input_contract_hash",
            }
        ),
        name="parent_dataset",
    )
    parent_dataset_id = _require_nonempty_text(
        parent["dataset_id"], name="parent_dataset.dataset_id"
    )
    parent_schema_version = _require_int(
        parent["schema_version"], name="parent_dataset.schema_version"
    )
    if parent_schema_version < 1:
        raise ValueError("parent dataset schema version must be positive")
    source_descriptor_hash = _optional_sha256(
        parent["source_descriptor_hash"], name="parent_dataset.source_descriptor_hash"
    )
    capability_contract_hash = _optional_sha256(
        parent["capability_contract_hash"],
        name="parent_dataset.capability_contract_hash",
    )
    input_contract_hash = _optional_sha256(
        parent["input_contract_hash"],
        name="parent_dataset.input_contract_hash",
    )

    development = _require_mapping(root["development_dataset"], name="development_dataset")
    _require_exact_keys(
        development,
        frozenset(
            {
                "dataset_id",
                "schema_version",
                "source_descriptor_hash",
                "capability_contract_hash",
                "input_contract_hash",
                "row_count",
                "task_count",
                "tasks",
            }
        ),
        name="development_dataset",
    )
    development_dataset_id = _require_sha256(
        development["dataset_id"], name="development_dataset.dataset_id"
    )
    if development["schema_version"] != parent_schema_version:
        raise ValueError("development audit changed the parent schema version")
    if development["source_descriptor_hash"] != source_descriptor_hash:
        raise ValueError("development audit changed the source descriptor hash")
    if development["capability_contract_hash"] != capability_contract_hash:
        raise ValueError("development audit changed the capability contract hash")
    if development["input_contract_hash"] != input_contract_hash:
        raise ValueError("development audit changed the input contract hash")
    row_count = _require_int(development["row_count"], name="row_count")
    task_count = _require_int(development["task_count"], name="task_count")
    if row_count < 1 or task_count < 1:
        raise ValueError("development row and task counts must be positive")
    task_entries = _require_sequence(development["tasks"], name="development.tasks")
    development_tasks: list[str] = []
    summed_rows = 0
    for index, item in enumerate(task_entries):
        entry = _require_mapping(item, name=f"development.tasks[{index}]")
        _require_exact_keys(
            entry,
            frozenset({"task_pseudonym", "row_count", "run_count", "condition_count"}),
            name=f"development.tasks[{index}]",
        )
        development_tasks.append(
            _require_sha256(
                entry["task_pseudonym"],
                name=f"development.tasks[{index}].task_pseudonym",
            )
        )
        counts = tuple(
            _require_int(entry[key], name=f"development.tasks[{index}].{key}")
            for key in ("row_count", "run_count", "condition_count")
        )
        if any(count < 1 for count in counts):
            raise ValueError("development task counts must be positive")
        summed_rows += counts[0]
    if development_tasks != sorted(development_tasks):
        raise ValueError("development tasks must use canonical pseudonym order")
    if len(set(development_tasks)) != len(development_tasks):
        raise ValueError("development task pseudonyms must be unique")
    if task_count != len(development_tasks) or row_count != summed_rows:
        raise ValueError("development task or row count does not match its projection")
    development_task_set = frozenset(development_tasks)

    holdout = _require_mapping(root["permanent_holdout"], name="permanent_holdout")
    _require_exact_keys(
        holdout,
        frozenset(
            {
                "holdout_plan_id",
                "assignment_id",
                "policy_id",
                "salt_sha256",
                "bucket_count",
                "final_holdout_bucket_threshold_exclusive",
                "assignments",
            }
        ),
        name="permanent_holdout",
    )
    holdout_plan_id = _require_sha256(
        holdout["holdout_plan_id"], name="permanent_holdout.holdout_plan_id"
    )
    holdout_assignment_id = _require_sha256(
        holdout["assignment_id"], name="permanent_holdout.assignment_id"
    )
    expected_holdout_plan_id = _canonical_sha256(
        {
            "assignment_id": holdout_assignment_id,
            "dataset_id": parent_dataset_id,
        }
    )
    if holdout_plan_id != expected_holdout_plan_id:
        raise ValueError("permanent holdout plan id does not match the parent dataset")
    _require_nonempty_text(holdout["policy_id"], name="permanent_holdout.policy_id")
    _require_sha256(holdout["salt_sha256"], name="permanent_holdout.salt_sha256")
    bucket_count = _require_int(holdout["bucket_count"], name="permanent_holdout.bucket_count")
    threshold = _require_int(
        holdout["final_holdout_bucket_threshold_exclusive"],
        name="permanent_holdout.final_holdout_bucket_threshold_exclusive",
    )
    if bucket_count <= 1 or not 0 < threshold < bucket_count:
        raise ValueError("permanent holdout bucket settings are invalid")
    holdout_assignments = _parse_public_assignments(
        holdout["assignments"],
        name="permanent_holdout.assignments",
        value_key="cohort",
        allowed_values=frozenset({"development", "final_holdout"}),
    )
    projected_development_tasks = frozenset(
        task for task, cohort in holdout_assignments if cohort == "development"
    )
    final_holdout_tasks = frozenset(
        task for task, cohort in holdout_assignments if cohort == "final_holdout"
    )
    if projected_development_tasks != development_task_set:
        raise ValueError("development projection differs from the permanent assignment")
    if not final_holdout_tasks or development_task_set & final_holdout_tasks:
        raise ValueError("final holdout is empty or overlaps development tasks")

    seeds = tuple(
        _require_int(seed, name="split_seeds entry")
        for seed in _require_sequence(root["split_seeds"], name="split_seeds")
    )
    if seeds != STAGE_SPLIT_SEEDS:
        raise ValueError("audit does not use the three frozen split seeds")
    outer_entries = _require_sequence(root["outer_plans"], name="outer_plans")
    if len(outer_entries) != len(STAGE_SPLIT_SEEDS):
        raise ValueError("audit must contain exactly three outer plans")

    outer_identities: list[tuple[int, int, str, str]] = []
    inner_identities: list[tuple[int, int, str]] = []
    for outer_index, outer_item in enumerate(outer_entries):
        outer = _require_mapping(outer_item, name=f"outer_plans[{outer_index}]")
        _require_exact_keys(
            outer,
            frozenset(
                {
                    "seed",
                    "folds",
                    "split_plan_id",
                    "assignment_id",
                    "assignments",
                    "inner_plans",
                }
            ),
            name=f"outer_plans[{outer_index}]",
        )
        seed = _require_int(outer["seed"], name=f"outer_plans[{outer_index}].seed")
        if seed != STAGE_SPLIT_SEEDS[outer_index]:
            raise ValueError("outer plans are not in frozen seed order")
        folds = _require_int(outer["folds"], name=f"outer_plans[{outer_index}].folds")
        if folds != OUTER_FOLDS:
            raise ValueError("outer plan must contain exactly five folds")
        split_plan_id = _require_sha256(
            outer["split_plan_id"], name=f"outer_plans[{outer_index}].split_plan_id"
        )
        assignment_id = _require_sha256(
            outer["assignment_id"], name=f"outer_plans[{outer_index}].assignment_id"
        )
        assignments = _parse_public_assignments(
            outer["assignments"],
            name=f"outer_plans[{outer_index}].assignments",
            value_key="fold",
            allowed_values=frozenset(range(OUTER_FOLDS)),
        )
        if frozenset(task for task, _fold in assignments) != development_task_set:
            raise ValueError("outer task universe differs from development tasks")
        if {fold for _task, fold in assignments} != set(range(OUTER_FOLDS)):
            raise ValueError("an outer fold is empty")
        if frozenset(task for task, _fold in assignments) & final_holdout_tasks:
            raise ValueError("final-holdout task leaked into an outer plan")

        inner_entries = _require_sequence(
            outer["inner_plans"], name=f"outer_plans[{outer_index}].inner_plans"
        )
        if len(inner_entries) != OUTER_FOLDS:
            raise ValueError("every outer fold requires one inner plan")
        outer_mapping = dict(assignments)
        for inner_index, inner_item in enumerate(inner_entries):
            inner = _require_mapping(
                inner_item,
                name=f"outer_plans[{outer_index}].inner_plans[{inner_index}]",
            )
            _require_exact_keys(
                inner,
                frozenset(
                    {
                        "outer_test_fold",
                        "assignment_id",
                        "policy_id",
                        "folds",
                        "assignments",
                    }
                ),
                name=f"outer_plans[{outer_index}].inner_plans[{inner_index}]",
            )
            outer_test_fold = _require_int(inner["outer_test_fold"], name="inner outer_test_fold")
            if outer_test_fold != inner_index:
                raise ValueError("inner plans are not in outer-fold order")
            inner_assignment_id = _require_sha256(
                inner["assignment_id"], name="inner assignment_id"
            )
            _require_nonempty_text(inner["policy_id"], name="inner policy_id")
            if _require_int(inner["folds"], name="inner folds") != INNER_FOLDS:
                raise ValueError("inner plan must contain exactly five folds")
            inner_assignments = _parse_public_assignments(
                inner["assignments"],
                name=(f"outer_plans[{outer_index}].inner_plans[{inner_index}].assignments"),
                value_key="fold",
                allowed_values=frozenset(range(INNER_FOLDS)),
            )
            excluded_outer_folds = {
                outer_test_fold,
                (outer_test_fold + 1) % OUTER_FOLDS,
                (outer_test_fold + 2) % OUTER_FOLDS,
            }
            outer_train = frozenset(
                task for task, fold in outer_mapping.items() if fold not in excluded_outer_folds
            )
            if frozenset(task for task, _fold in inner_assignments) != outer_train:
                raise ValueError("inner task universe differs from outer-train tasks")
            if {fold for _task, fold in inner_assignments} != set(range(INNER_FOLDS)):
                raise ValueError("an inner fold is empty")
            inner_mapping = dict(inner_assignments)
            for inner_holdout_fold in range(INNER_FOLDS):
                holdout_tasks = {
                    task for task, fold in inner_mapping.items() if fold == inner_holdout_fold
                }
                validation_tasks = {
                    task
                    for task, fold in inner_mapping.items()
                    if fold == (inner_holdout_fold + 1) % INNER_FOLDS
                }
                fit_tasks = {
                    task
                    for task, fold in inner_mapping.items()
                    if fold
                    not in {
                        inner_holdout_fold,
                        (inner_holdout_fold + 1) % INNER_FOLDS,
                    }
                }
                if not holdout_tasks or not validation_tasks or not fit_tasks:
                    raise ValueError("inner fit/validation/holdout partition is empty")
            if outer_train & final_holdout_tasks:
                raise ValueError("final-holdout task leaked into an inner plan")
            inner_identities.append((seed, outer_test_fold, inner_assignment_id))
        outer_identities.append((seed, folds, split_plan_id, assignment_id))

    expected_protocol_id = _canonical_sha256(
        _protocol_identity(
            parent_schema_version=parent_schema_version,
            source_descriptor_hash=source_descriptor_hash,
            capability_contract_hash=capability_contract_hash,
            input_contract_hash=input_contract_hash,
            development_dataset_id=development_dataset_id,
            holdout_assignment_id=holdout_assignment_id,
            outer_plans=outer_identities,
            inner_plans=inner_identities,
        )
    )
    if protocol_id != expected_protocol_id:
        raise ValueError("development audit protocol id does not match its identity")
