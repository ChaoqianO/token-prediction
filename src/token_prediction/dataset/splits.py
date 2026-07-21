from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


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


def make_task_split_plan(
    task_ids: Iterable[str],
    *,
    dataset_id: str,
    folds: int,
    seed: int,
) -> SplitPlan:
    return assign_task_folds(task_ids, folds=folds, seed=seed).bind(dataset_id)
