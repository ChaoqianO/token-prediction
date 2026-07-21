"""Shared calibration and evaluation for every candidate."""

from .calibration import (
    CalibrationExample,
    FittedCalibrator,
    IdentityCalibrator,
    IntervalCalibrator,
    TaskMaxConformalCalibrator,
)
from .comparison import PairedBootstrapComparison, paired_task_bootstrap
from .metrics import METRIC_SUITE_ID, ScoredForecast, evaluate_forecasts

__all__ = [
    "CalibrationExample",
    "FittedCalibrator",
    "IdentityCalibrator",
    "IntervalCalibrator",
    "METRIC_SUITE_ID",
    "PairedBootstrapComparison",
    "ScoredForecast",
    "TaskMaxConformalCalibrator",
    "evaluate_forecasts",
    "paired_task_bootstrap",
]
