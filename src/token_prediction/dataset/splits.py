from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping


PERMANENT_HOLDOUT_POLICY_ID = "stable_task_sha256_bucket_v1"
INNER_FOLD_POLICY_ID = "task_multicohort_balanced_inner_oof_v2"
OUTER_BALANCED_FOLD_POLICY_ID = "task_multicohort_balanced_outer_cv_v1"
DEFAULT_FINAL_HOLDOUT_SALT = "token-prediction/final-holdout/2026-07-21/v1"
DEFAULT_FINAL_HOLDOUT_BUCKET_COUNT = 10_000
DEFAULT_FINAL_HOLDOUT_BUCKET_THRESHOLD = 2_000
INNER_FOLDS = 5


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


def _holdout_evidence(
    tasks: Iterable[str],
    *,
    salt: str,
    bucket_count: int,
    threshold: int,
) -> tuple[list[dict[str, object]], tuple[tuple[str, str], ...]]:
    evidence: list[dict[str, object]] = []
    assignments: list[tuple[str, str]] = []
    for task in sorted(tasks):
        digest = hashlib.sha256(f"{salt}\0{task}".encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % bucket_count
        partition = "final_holdout" if bucket < threshold else "development"
        evidence.append(
            {"task_id_sha256": digest, "bucket": bucket, "cohort": partition}
        )
        assignments.append((task, partition))
    return evidence, tuple(assignments)


BalanceGroups = tuple[tuple[str, tuple[str, ...]], ...]


def _normalized_balance_groups(
    tasks: Iterable[str],
    groups: Mapping[str, Iterable[str]] | None,
    *,
    folds: int,
) -> BalanceGroups:
    task_set = frozenset(tasks)
    if groups is None:
        return ()
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw_group_id, raw_members in groups.items():
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError("balance group ids must be non-empty")
        members = tuple(
            sorted(
                {
                    str(task_id).strip()
                    for task_id in raw_members
                    if str(task_id).strip()
                }
            )
        )
        if not frozenset(members) <= task_set:
            raise ValueError(f"balance group {group_id!r} contains an unknown task")
        if len(members) < folds:
            raise ValueError(
                f"balance group {group_id!r} cannot cover all {folds} folds"
            )
        normalized.append((group_id, members))
    normalized.sort(key=lambda item: item[0])
    if len({group_id for group_id, _members in normalized}) != len(normalized):
        raise ValueError("balance group ids must be unique")
    return tuple(normalized)


def _balance_groups_sha256(groups: BalanceGroups) -> str:
    return _canonical_sha256(
        [
            {"group_id": group_id, "tasks": list(members)}
            for group_id, members in groups
        ]
    )


def _balanced_assignments(
    tasks: Iterable[str],
    *,
    folds: int,
    seed: int,
    policy_id: str,
    balance_groups: BalanceGroups,
) -> tuple[tuple[str, int], ...]:
    """Greedily balance multiple task cohorts with deterministic hash tie-breaks."""

    canonical_tasks = tuple(sorted(tasks))
    group_members = {
        group_id: frozenset(members) for group_id, members in balance_groups
    }
    memberships = {
        task: tuple(
            group_id
            for group_id, members in balance_groups
            if task in group_members[group_id]
        )
        for task in canonical_tasks
    }
    group_sizes = {group_id: len(members) for group_id, members in balance_groups}
    ranked = sorted(
        canonical_tasks,
        key=lambda task: (
            min((group_sizes[group_id] for group_id in memberships[task]), default=10**18),
            -len(memberships[task]),
            hashlib.sha256(
                f"{policy_id}\0{seed}\0task\0{task}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    fold_sizes = [0] * folds
    group_fold_counts = {
        group_id: [0] * folds for group_id, _members in balance_groups
    }
    assigned: dict[str, int] = {}
    for task in ranked:
        task_groups = memberships[task]

        def score(fold: int) -> tuple[int, int, int, str]:
            counts = [group_fold_counts[group_id][fold] for group_id in task_groups]
            return (
                max(counts, default=0),
                sum(counts),
                fold_sizes[fold],
                hashlib.sha256(
                    f"{policy_id}\0{seed}\0fold\0{task}\0{fold}".encode("utf-8")
                ).hexdigest(),
            )

        selected = min(range(folds), key=score)
        assigned[task] = selected
        fold_sizes[selected] += 1
        for group_id in task_groups:
            group_fold_counts[group_id][selected] += 1

    if any(count == 0 for count in fold_sizes):
        raise ValueError("balanced task assignment produced an empty fold")
    for group_id, counts in group_fold_counts.items():
        if any(count == 0 for count in counts):
            raise ValueError(
                f"balanced task assignment left group {group_id!r} empty in a fold"
            )
    return tuple(sorted(assigned.items()))


def _inner_assignments(
    tasks: Iterable[str],
    *,
    seed: int,
    balance_groups: BalanceGroups,
) -> tuple[tuple[str, int], ...]:
    return _balanced_assignments(
        tasks,
        folds=INNER_FOLDS,
        seed=seed,
        policy_id=INNER_FOLD_POLICY_ID,
        balance_groups=balance_groups,
    )


class Partition(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


@dataclass(frozen=True)
class FoldPartition:
    fold: int
    train_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    calibration_tasks: frozenset[str]
    test_tasks: frozenset[str]

    def __post_init__(self) -> None:
        partitions = (
            self.train_tasks,
            self.validation_tasks,
            self.calibration_tasks,
            self.test_tasks,
        )
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                if left & right:
                    raise ValueError("train/validation/calibration/test tasks overlap")


@dataclass(frozen=True)
class SplitPlan:
    split_plan_id: str
    assignment_id: str
    dataset_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]

    @property
    def task_to_fold(self) -> dict[str, int]:
        return dict(self.assignments)

    def partition(self, test_fold: int) -> FoldPartition:
        if not 0 <= test_fold < self.folds:
            raise ValueError("test fold is out of range")
        calibration_fold = (test_fold + 1) % self.folds
        validation_fold = (test_fold + 2) % self.folds
        mapping = self.task_to_fold
        test = frozenset(task for task, fold in mapping.items() if fold == test_fold)
        calibration = frozenset(
            task for task, fold in mapping.items() if fold == calibration_fold
        )
        validation = frozenset(
            task for task, fold in mapping.items() if fold == validation_fold
        )
        train = frozenset(
            task
            for task, fold in mapping.items()
            if fold not in {test_fold, calibration_fold, validation_fold}
        )
        return FoldPartition(test_fold, train, validation, calibration, test)

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(task_ids)
        planned = frozenset(task for task, _ in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"split task mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class TaskFoldAssignment:
    assignment_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _ in self.assignments)

    def bind(self, dataset_id: str) -> SplitPlan:
        semantic = {
            "assignment_id": self.assignment_id,
            "dataset_id": dataset_id,
        }
        encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        return SplitPlan(
            split_plan_id=hashlib.sha256(encoded).hexdigest(),
            assignment_id=self.assignment_id,
            dataset_id=dataset_id,
            folds=self.folds,
            seed=self.seed,
            assignments=self.assignments,
        )


@dataclass(frozen=True)
class PermanentHoldoutPlan:
    holdout_plan_id: str
    assignment_id: str
    dataset_id: str
    policy_id: str
    salt: str
    bucket_count: int
    final_holdout_bucket_threshold_exclusive: int
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        assignment = PermanentHoldoutAssignment(
            assignment_id=self.assignment_id,
            policy_id=self.policy_id,
            salt=self.salt,
            bucket_count=self.bucket_count,
            final_holdout_bucket_threshold_exclusive=(
                self.final_holdout_bucket_threshold_exclusive
            ),
            assignments=self.assignments,
        )
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id is required")
        expected = _canonical_sha256(
            {"assignment_id": assignment.assignment_id, "dataset_id": self.dataset_id}
        )
        if self.holdout_plan_id != expected:
            raise ValueError("permanent holdout plan id does not match its identity")

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "development"
        )

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "final_holdout"
        )

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(
            str(task_id).strip() for task_id in task_ids if str(task_id).strip()
        )
        planned = frozenset(task for task, _partition in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(
                f"permanent holdout task mismatch; missing={missing}, extra={extra}"
            )


@dataclass(frozen=True)
class PermanentHoldoutAssignment:
    assignment_id: str
    policy_id: str
    salt: str
    bucket_count: int
    final_holdout_bucket_threshold_exclusive: int
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.policy_id != PERMANENT_HOLDOUT_POLICY_ID:
            raise ValueError("unsupported permanent holdout policy")
        if not self.salt:
            raise ValueError("permanent holdout salt is required")
        if self.bucket_count <= 1:
            raise ValueError("permanent holdout bucket_count must exceed one")
        if not 0 < self.final_holdout_bucket_threshold_exclusive < self.bucket_count:
            raise ValueError("permanent holdout threshold must be inside bucket range")
        tasks = [task for task, _partition in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("permanent holdout assignments must use unique task order")
        partitions = {partition for _task, partition in self.assignments}
        if partitions != {"development", "final_holdout"}:
            raise ValueError("permanent holdout assignments require both partitions")
        evidence, expected_assignments = _holdout_evidence(
            tasks,
            salt=self.salt,
            bucket_count=self.bucket_count,
            threshold=self.final_holdout_bucket_threshold_exclusive,
        )
        if self.assignments != expected_assignments:
            raise ValueError("permanent holdout assignment violates task hash policy")
        if self.assignment_id != _canonical_sha256(evidence):
            raise ValueError("permanent holdout assignment id does not match mapping")

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "development"
        )

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "final_holdout"
        )

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _partition in self.assignments)

    def bind(self, dataset_id: str) -> PermanentHoldoutPlan:
        if not str(dataset_id).strip():
            raise ValueError("dataset_id is required")
        semantic = {
            "assignment_id": self.assignment_id,
            "dataset_id": dataset_id,
        }
        encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        return PermanentHoldoutPlan(
            holdout_plan_id=hashlib.sha256(encoded).hexdigest(),
            assignment_id=self.assignment_id,
            dataset_id=dataset_id,
            policy_id=self.policy_id,
            salt=self.salt,
            bucket_count=self.bucket_count,
            final_holdout_bucket_threshold_exclusive=(
                self.final_holdout_bucket_threshold_exclusive
            ),
            assignments=self.assignments,
        )


@dataclass(frozen=True)
class InnerFoldPartition:
    holdout_fold: int
    initializer_fit_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    holdout_tasks: frozenset[str]

    def __post_init__(self) -> None:
        if not 0 <= self.holdout_fold < INNER_FOLDS:
            raise ValueError("inner holdout fold is out of range")
        partitions = (
            self.initializer_fit_tasks,
            self.validation_tasks,
            self.holdout_tasks,
        )
        if any(not partition for partition in partitions):
            raise ValueError("inner fit/validation/holdout partitions must be non-empty")
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                if left & right:
                    raise ValueError("inner fit/validation/holdout tasks overlap")


@dataclass(frozen=True)
class InnerTaskFoldAssignment:
    assignment_id: str
    policy_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]
    balance_groups: BalanceGroups = ()

    def __post_init__(self) -> None:
        if self.policy_id != INNER_FOLD_POLICY_ID:
            raise ValueError("unsupported inner fold policy")
        if self.folds != INNER_FOLDS:
            raise ValueError("inner OOF assignment requires exactly five folds")
        tasks = [task for task, _fold in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("inner assignments must use unique canonical task order")
        fold_values = {fold for _task, fold in self.assignments}
        if fold_values != set(range(INNER_FOLDS)):
            raise ValueError("every inner fold must be non-empty")
        normalized_groups = _normalized_balance_groups(
            tasks,
            {group_id: members for group_id, members in self.balance_groups},
            folds=self.folds,
        )
        if normalized_groups != self.balance_groups:
            raise ValueError("inner balance groups must use canonical order")
        if self.assignments != _inner_assignments(
            tasks,
            seed=self.seed,
            balance_groups=normalized_groups,
        ):
            raise ValueError("inner assignment violates task hash policy")
        semantic = {
            "policy_id": INNER_FOLD_POLICY_ID,
            "folds": self.folds,
            "seed": self.seed,
            "balance_groups_sha256": _balance_groups_sha256(normalized_groups),
            "assignments": self.assignments,
        }
        if self.assignment_id != _canonical_sha256(semantic):
            raise ValueError("inner assignment id does not match mapping")

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _fold in self.assignments)

    @property
    def task_to_fold(self) -> dict[str, int]:
        return dict(self.assignments)

    def partition(self, holdout_fold: int) -> InnerFoldPartition:
        if not 0 <= holdout_fold < self.folds:
            raise ValueError("inner holdout fold is out of range")
        validation_fold = (holdout_fold + 1) % self.folds
        mapping = self.task_to_fold
        holdout = frozenset(
            task for task, fold in mapping.items() if fold == holdout_fold
        )
        validation = frozenset(
            task for task, fold in mapping.items() if fold == validation_fold
        )
        fit = frozenset(
            task
            for task, fold in mapping.items()
            if fold not in {holdout_fold, validation_fold}
        )
        return InnerFoldPartition(
            holdout_fold=holdout_fold,
            initializer_fit_tasks=fit,
            validation_tasks=validation,
            holdout_tasks=holdout,
        )

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(
            str(task_id).strip() for task_id in task_ids if str(task_id).strip()
        )
        planned = self.task_ids
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"inner split task mismatch; missing={missing}, extra={extra}")


def assign_task_folds(
    task_ids: Iterable[str],
    *,
    folds: int,
    seed: int,
    balance_groups: Mapping[str, Iterable[str]] | None = None,
) -> TaskFoldAssignment:
    tasks = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if folds < 4:
        raise ValueError(
            "at least four folds are required for train/validation/calibration/test isolation"
        )
    if len(tasks) < folds:
        raise ValueError("number of tasks must be at least the number of folds")
    normalized_groups = _normalized_balance_groups(
        tasks,
        balance_groups,
        folds=folds,
    )
    if normalized_groups:
        assignments = _balanced_assignments(
            tasks,
            folds=folds,
            seed=seed,
            policy_id=OUTER_BALANCED_FOLD_POLICY_ID,
            balance_groups=normalized_groups,
        )
        semantic = {
            "policy_id": OUTER_BALANCED_FOLD_POLICY_ID,
            "folds": folds,
            "seed": seed,
            "balance_groups_sha256": _balance_groups_sha256(normalized_groups),
            "assignments": assignments,
        }
    else:
        ranked = sorted(
            tasks,
            key=lambda task: hashlib.sha256(f"{seed}\0{task}".encode("utf-8")).hexdigest(),
        )
        assignments = tuple(
            sorted((task, index % folds) for index, task in enumerate(ranked))
        )
        semantic = {
            "folds": folds,
            "seed": seed,
            "assignments": assignments,
        }
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return TaskFoldAssignment(
        assignment_id=hashlib.sha256(encoded).hexdigest(),
        folds=folds,
        seed=seed,
        assignments=assignments,
    )


def assign_permanent_task_holdout(
    task_ids: Iterable[str],
    *,
    salt: str = DEFAULT_FINAL_HOLDOUT_SALT,
    bucket_count: int = DEFAULT_FINAL_HOLDOUT_BUCKET_COUNT,
    final_holdout_bucket_threshold_exclusive: int = (
        DEFAULT_FINAL_HOLDOUT_BUCKET_THRESHOLD
    ),
    minimum_development_tasks: int = INNER_FOLDS,
) -> PermanentHoldoutAssignment:
    """Freeze a task-only development/final-holdout assignment.

    Duplicate task IDs (for example multiple runs or model families) collapse to
    one hash decision, so all observations for the task always stay together.
    Labels, dataset IDs, split seeds, family IDs, and run IDs are not inputs.
    """

    tasks = sorted(
        {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
    )
    if not salt:
        raise ValueError("permanent holdout salt is required")
    if bucket_count <= 1:
        raise ValueError("permanent holdout bucket_count must exceed one")
    if not 0 < final_holdout_bucket_threshold_exclusive < bucket_count:
        raise ValueError("permanent holdout threshold must be inside bucket range")
    if minimum_development_tasks < INNER_FOLDS:
        raise ValueError("development cohort must support five outer folds")
    if len(tasks) < minimum_development_tasks + 1:
        raise ValueError("too few tasks for a permanent holdout")

    evidence, resolved = _holdout_evidence(
        tasks,
        salt=salt,
        bucket_count=bucket_count,
        threshold=final_holdout_bucket_threshold_exclusive,
    )
    development_count = sum(
        partition == "development" for _task, partition in resolved
    )
    holdout_count = sum(
        partition == "final_holdout" for _task, partition in resolved
    )
    if development_count < minimum_development_tasks or holdout_count < 1:
        raise ValueError("task hash produced an empty or undersized holdout partition")
    return PermanentHoldoutAssignment(
        assignment_id=_canonical_sha256(evidence),
        policy_id=PERMANENT_HOLDOUT_POLICY_ID,
        salt=salt,
        bucket_count=bucket_count,
        final_holdout_bucket_threshold_exclusive=(
            final_holdout_bucket_threshold_exclusive
        ),
        assignments=resolved,
    )


def assign_inner_task_folds(
    task_ids: Iterable[str],
    *,
    seed: int,
    folds: int = INNER_FOLDS,
    balance_groups: Mapping[str, Iterable[str]] | None = None,
) -> InnerTaskFoldAssignment:
    """Assign exactly five task folds for leakage-free initializer OOF seeds."""

    if folds != INNER_FOLDS:
        raise ValueError("inner OOF assignment requires exactly five folds")
    tasks = sorted(
        {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
    )
    if len(tasks) < folds:
        raise ValueError("number of inner tasks must be at least five")
    normalized_groups = _normalized_balance_groups(
        tasks,
        balance_groups,
        folds=folds,
    )
    assignments = _inner_assignments(
        tasks,
        seed=seed,
        balance_groups=normalized_groups,
    )
    semantic = {
        "policy_id": INNER_FOLD_POLICY_ID,
        "folds": folds,
        "seed": seed,
        "balance_groups_sha256": _balance_groups_sha256(normalized_groups),
        "assignments": assignments,
    }
    return InnerTaskFoldAssignment(
        assignment_id=_canonical_sha256(semantic),
        policy_id=INNER_FOLD_POLICY_ID,
        folds=folds,
        seed=seed,
        assignments=assignments,
        balance_groups=normalized_groups,
    )


def make_task_split_plan(
    task_ids: Iterable[str],
    *,
    dataset_id: str,
    folds: int,
    seed: int,
) -> SplitPlan:
    return assign_task_folds(task_ids, folds=folds, seed=seed).bind(dataset_id)
