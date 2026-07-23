"""Shared calibration and evaluation for every candidate."""

from .calibration import (
    CALIBRATOR_SCHEMA_VERSION,
    CalibrationExample,
    FittedExpansionCalibrator,
    FittedCalibrator,
    IdentityCalibrator,
    IntervalCalibrator,
    TaskMaxConformalCalibrator,
)
from .comparison import (
    PairedBootstrapComparison,
    paired_task_bootstrap,
    paired_task_metric_bootstrap,
)
from .budget import BUDGET_METRIC_SUITE_ID, evaluate_budget_scenarios
from .metrics import (
    METRIC_SUITE_ID,
    ScoredForecast,
    TaskForecastMetrics,
    evaluate_forecasts,
    evaluate_task_forecasts,
)
from .stratification import (
    DEFAULT_PROGRESS_CHECKPOINTS,
    PROGRESS_STRATIFICATION_ID,
    RUN_VARIANCE_ID,
    TERMINATION_STRATIFICATION_ID,
    evaluate_progress_checkpoints,
    evaluate_same_task_run_variance,
    evaluate_termination_strata,
)
from .stage4 import (
    STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID,
    STAGE4_MATCHED_COVERAGE_POLICY_ID,
    CrossConditionPairedComparison,
    MatchedCoverageComparison,
    assert_calibration_raw_prediction_parity,
    compare_matched_coverage,
    compare_same_tasks_across_conditions,
)

__all__ = [
    "CALIBRATOR_SCHEMA_VERSION",
    "BUDGET_METRIC_SUITE_ID",
    "CalibrationExample",
    "FittedExpansionCalibrator",
    "FittedCalibrator",
    "IdentityCalibrator",
    "IntervalCalibrator",
    "METRIC_SUITE_ID",
    "DEFAULT_PROGRESS_CHECKPOINTS",
    "PairedBootstrapComparison",
    "PROGRESS_STRATIFICATION_ID",
    "RUN_VARIANCE_ID",
    "ScoredForecast",
    "TaskForecastMetrics",
    "TaskMaxConformalCalibrator",
    "TERMINATION_STRATIFICATION_ID",
    "evaluate_budget_scenarios",
    "evaluate_forecasts",
    "evaluate_progress_checkpoints",
    "evaluate_same_task_run_variance",
    "evaluate_task_forecasts",
    "evaluate_termination_strata",
    "STAGE4_CROSS_CONDITION_PAIRING_POLICY_ID",
    "STAGE4_MATCHED_COVERAGE_POLICY_ID",
    "CrossConditionPairedComparison",
    "MatchedCoverageComparison",
    "assert_calibration_raw_prediction_parity",
    "compare_matched_coverage",
    "compare_same_tasks_across_conditions",
    "paired_task_bootstrap",
    "paired_task_metric_bootstrap",
]
