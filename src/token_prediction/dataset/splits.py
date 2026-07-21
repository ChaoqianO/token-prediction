from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


PERMANENT_HOLDOUT_POLICY_ID = "stable_task_sha256_bucket_v1"
INNER_FOLD_POLICY_ID = "task_sha256_rank_round_robin_inner_oof_v1"
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


def _inner_assignments(tasks: Iterable[str], *, seed: int) -> tuple[tuple[str, int], ...]:
    ranked = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{INNER_FOLD_POLICY_ID}\0{seed}\0{task}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(
        sorted((task, index % INNER_FOLDS) for index, task in enumerate(ranked))
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
        if self.assignments != _inner_assignments(tasks, seed=self.seed):
            raise ValueError("inner assignment violates task hash policy")
        semantic = {
            "policy_id": INNER_FOLD_POLICY_ID,
            "folds": self.folds,
            "seed": self.seed,
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
) -> TaskFoldAssignment:
    tasks = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if folds < 4:
        raise ValueError(
            "at least four folds are required for train/validation/calibration/test isolation"
        )
    if len(tasks) < folds:
        raise ValueError("number of tasks must be at least the number of folds")
    ranked = sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{seed}\0{task}".encode("utf-8")).hexdigest(),
    )
    assignments = tuple(sorted((task, index % folds) for index, task in enumerate(ranked)))
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
) -> InnerTaskFoldAssignment:
    """Assign exactly five task folds for leakage-free initializer OOF seeds."""

    if folds != INNER_FOLDS:
        raise ValueError("inner OOF assignment requires exactly five folds")
    tasks = sorted(
        {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
    )
    if len(tasks) < folds:
        raise ValueError("number of inner tasks must be at least five")
    assignments = _inner_assignments(tasks, seed=seed)
    semantic = {
        "policy_id": INNER_FOLD_POLICY_ID,
        "folds": folds,
        "seed": seed,
        "assignments": assignments,
    }
    return InnerTaskFoldAssignment(
        assignment_id=_canonical_sha256(semantic),
        policy_id=INNER_FOLD_POLICY_ID,
        folds=folds,
        seed=seed,
        assignments=assignments,
    )


def make_task_split_plan(
    task_ids: Iterable[str],
    *,
    dataset_id: str,
    folds: int,
    seed: int,
) -> SplitPlan:
    return assign_task_folds(task_ids, folds=folds, seed=seed).bind(dataset_id)
