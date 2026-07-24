"""Verify the development-only Stage 4 completion supplement.

The verifier has two deliberately separate modes:

* ``--tracked-only`` byte-binds the tracked lock/report/release tooling/CI to
  the fixed completion release-control tag, then closes the immutable source
  commits and the already-frozen parent final release.
* full verification additionally verifies all four development artifacts,
  checks the exact experiment/candidate matrix, and independently loads every
  declared LightGBM, Independent MLP, and lifecycle bundle.

Neither mode opens the final artifact, rebuilds a final dataset, reads final
labels, predicts on the final cohort, or scores it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.contracts import SourceDescriptor
from token_prediction.checkpoint import _result_from_dict
from token_prediction.development import (
    STAGE_SPLIT_SEEDS,
    TASK_PSEUDONYM_POLICY_ID as DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID,
)
from token_prediction.evaluation import METRIC_SUITE_ID
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.estimators.neural_bundle import load_neural_bundle
from token_prediction.evaluation.stratification import (
    PROGRESS_STRATIFICATION_ID,
    RUN_DISPERSION_EXTENSION_ID,
    RUN_VARIANCE_ID,
    TERMINATION_STRATIFICATION_ID,
)
from token_prediction.lifecycle_bundle import load_lifecycle_bundle
from token_prediction.lineage import verify_artifact
from token_prediction.features import NO_FEATURES
from token_prediction.stage2_matrix import (
    BAGEN_SOKOBAN_SOURCE_ID,
    BAGEN_SOURCE_ID,
    SPEND_AGGREGATE_STRUCTURED_FEATURES,
    SPEND_AGGREGATE_TASK_CHARS,
    SPEND_AGGREGATE_SOURCE_ID,
    SPEND_SOURCE_ID,
    STAGE2_HISTORY_FEATURES,
    STAGE2_STRUCTURED_FEATURES,
)
from token_prediction.stage4_matrix import (
    FROZEN_STAGE4_SOURCE_CONDITIONS,
    STAGE4_CALL_PRE_TARGETS,
    STAGE4_G3_FEATURES,
    STAGE4_MATRIX_POLICY_ID,
    STAGE4_MATRIX_SCHEMA_VERSION,
    STAGE4_MISSING_MASK_INVARIANT_ID,
    STAGE4_NO_PROGRESS_FEATURES,
    STAGE4_NO_TOOLS_ERRORS_FEATURES,
    STAGE4_PRE_REQUEST_CHAR_MESSAGE_FEATURES,
    STAGE4_RETRIEVAL_HISTORY_FEATURES,
)

if __package__:
    from scripts.run_data_foundation_baseline import (
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from scripts.run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
    )
    from scripts.run_stage4_experiments import (
        STAGE4_ARTIFACT_SCHEMA_VERSION,
        STAGE4_ARTIFACT_LAYOUT_ID,
        STAGE4_RUNNER_RELATIVE,
        STAGE4_STAGE_NAME,
        STAGE4_TASK_PSEUDONYM_POLICY_ID,
        Stage4ExperimentError,
        _framed_code_hash,
        cohort_projection_sha256,
        prediction_projection_sha256,
        verify_stage4_results_document,
    )
    from scripts.verify_stage4_release import (
        MAX_BUNDLE_FILE_BYTES,
        Stage4ReleaseError,
        _regular_file as _release_regular_file,
        _validate_release_document as _validate_parent_release_document,
    )
    from scripts.summarize_stage4_completion import (
        ArtifactReference,
        CompletionSummaryError,
        build_completion_summary,
        load_completion_diagnostics_artifact,
        load_development_artifact,
        render_markdown,
    )
else:  # pragma: no cover - direct production CLI invocation
    from run_data_foundation_baseline import (
        _is_link_or_reparse,
        _repo_path,
        _safe_relative,
    )
    from run_stage2_experiments import (
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
    )
    from run_stage4_experiments import (
        STAGE4_ARTIFACT_SCHEMA_VERSION,
        STAGE4_ARTIFACT_LAYOUT_ID,
        STAGE4_RUNNER_RELATIVE,
        STAGE4_STAGE_NAME,
        STAGE4_TASK_PSEUDONYM_POLICY_ID,
        Stage4ExperimentError,
        _framed_code_hash,
        cohort_projection_sha256,
        prediction_projection_sha256,
        verify_stage4_results_document,
    )
    from verify_stage4_release import (
        MAX_BUNDLE_FILE_BYTES,
        Stage4ReleaseError,
        _regular_file as _release_regular_file,
        _validate_release_document as _validate_parent_release_document,
    )
    from summarize_stage4_completion import (
        ArtifactReference,
        CompletionSummaryError,
        build_completion_summary,
        load_completion_diagnostics_artifact,
        load_development_artifact,
        render_markdown,
    )


DEFAULT_RELEASE_LOCK = "configs/stage4_completion_release.json"
DEFAULT_REPORT = "docs/stage-4-completion-supplement.md"
PARENT_RELEASE_LOCK = "configs/stage4_release.json"
COMPLETION_RELEASE_TAG = "stage4-completion-release-v1"
RELEASE_CONTROL_PATHS = (
    ".github/workflows/ci.yml",
    DEFAULT_RELEASE_LOCK,
    DEFAULT_REPORT,
    "scripts/freeze_stage4_completion.py",
    "scripts/verify_stage4_completion_release.py",
    "scripts/verify_stage4_release.py",
)
RELEASE_SCHEMA_VERSION = 2
RELEASE_STAGE_NAME = "stage4_development_completion_supplement"
RELEASE_POLICY_ID = "stage4_development_only_completion_release_v1"
SOURCE_CODE_POLICY_ID = "stage4_completion_source_code_tree_v1"
SOURCE_TAG = "stage4-completion-source-v1"
DIAGNOSTICS_SOURCE_TAG = "stage4-completion-diagnostics-source-v1"
DIAGNOSTICS_RUNNER_RELATIVE = "scripts/run_stage4_completion_diagnostics.py"
DIAGNOSTICS_SUMMARIZER_RELATIVE = "scripts/summarize_stage4_completion.py"
PARENT_FINAL_TAG = "stage4-final-release-v1"
PARENT_FINAL_ARTIFACT_ID = (
    "a1f41e7a91d48677fea7b869835a297b5b6073f43d074ba295c727d6e6167287"
)

EXPECTED_ARTIFACT_COUNT = 4
EXPECTED_EXPERIMENT_COUNT = 52
EXPECTED_CANDIDATE_SEED_RUN_COUNT = 477
EXPECTED_RELOADABLE_BUNDLE_FOLD_COUNT = 1_950
EXPECTED_CALL_PRE_MLP_CELL_COUNT = 21
EXPECTED_CALL_PRE_MLP_BUNDLE_FOLD_COUNT = 315
EXPECTED_SEED_POLICY_CELL_COUNT = 7
EXPECTED_SEED_POLICY_BUNDLE_FOLD_COUNT = 210
EXPECTED_DIAGNOSTICS_ARTIFACT_COUNT = 1
EXPECTED_DIAGNOSTICS_BOUND_SOURCE_COUNT = 4
EXPECTED_DIAGNOSTICS_LIFECYCLE_SOURCE_COUNT = 3
EXPECTED_DIAGNOSTICS_LIFECYCLE_CONDITION_COUNT = 7
EXPECTED_DIAGNOSTICS_CANDIDATE_COUNT = 2
EXPECTED_DIAGNOSTICS_CANDIDATE_CELL_COUNT = 14
EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT = 42
EXPECTED_DIAGNOSTICS_BUNDLE_COUNT = 210
DIAGNOSTICS_RESULTS_SCHEMA_VERSION = 2
DIAGNOSTICS_STAGE_NAME = "stage4_completion_diagnostics"
DIAGNOSTICS_POLICY_ID = "stage4_completion_artifact_checkpoint_only_v2"
DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON = (
    "no_presealed_development_lifecycle_projection_v1"
)
DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS = [
    "progress",
    "run_variance_iqr_max_minus_min",
    "termination",
]
_DIAGNOSTIC_RECORD_KEYS = {
    "source_name",
    "condition_id",
    "experiment_id",
    "candidate_id",
    "candidate_hash",
    "split_seed",
    "split_plan_id",
    "bundle_folds",
    "bundle_projection_sha256",
    "checkpoint_parity",
    "lifecycle_metrics",
}
_CHECKPOINT_PARITY_KEYS = {
    "status",
    "checkpoint_artifact_id",
    "checkpoint_result_sha256",
    "prediction_count",
    "expected_prediction_count",
    "prediction_projection_sha256",
    "expected_prediction_projection_sha256",
    "cohort_projection_sha256",
    "expected_cohort_projection_sha256",
    "aggregate_metrics_projection_sha256",
    "expected_aggregate_metrics_projection_sha256",
    "development_cohort_status",
    "development_task_count",
    "development_task_projection_sha256",
}
_LIFECYCLE_METRICS_KEYS = {
    "status",
    "reason_code",
    "labels_present",
    "lifecycle_sequences_present",
    "unavailable_metrics",
    "historical_stage3_reference",
}
_DIAGNOSTICS_COVERAGE = {
    "bound_source_artifact_count": EXPECTED_DIAGNOSTICS_BOUND_SOURCE_COUNT,
    "lifecycle_source_count": EXPECTED_DIAGNOSTICS_LIFECYCLE_SOURCE_COUNT,
    "lifecycle_condition_count": (
        EXPECTED_DIAGNOSTICS_LIFECYCLE_CONDITION_COUNT
    ),
    "lifecycle_candidate_count": EXPECTED_DIAGNOSTICS_CANDIDATE_COUNT,
    "lifecycle_candidate_cell_count": (
        EXPECTED_DIAGNOSTICS_CANDIDATE_CELL_COUNT
    ),
    "lifecycle_candidate_seed_count": (
        EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
    ),
    "lifecycle_bundle_count": EXPECTED_DIAGNOSTICS_BUNDLE_COUNT,
    "checkpoint_verified_candidate_seed_count": (
        EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
    ),
    "lifecycle_replayed_candidate_seed_count": 0,
    "lifecycle_metrics_unavailable_candidate_seed_count": (
        EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
    ),
}
EXPECTED_FINAL_EVALUATION_COUNT = 1
EXPECTED_FINAL_PREDICTION_COUNT = 86_335
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 5

MAX_RELEASE_JSON_BYTES = 2 * 1024 * 1024
MAX_PARENT_RELEASE_BYTES = 2 * 1024 * 1024
MAX_RESULTS_JSON_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_JSON_BYTES = 128 * 1024 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_PROVENANCE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_FILE_COUNT = 10_000
MAX_BUNDLE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024

_EXPECTED_SOURCE_SCOPE = (
    ("spend_aggregate", SPEND_AGGREGATE_SOURCE_ID, 3, 15, 60),
    ("bagen_sokoban", BAGEN_SOKOBAN_SOURCE_ID, 7, 66, 270),
    ("bagen_swebench", BAGEN_SOURCE_ID, 35, 330, 1_350),
    ("spend_openhands", SPEND_SOURCE_ID, 7, 66, 270),
)
_EXPECTED_BY_SOURCE = {
    source_name: {
        "source_id": source_id,
        "experiment_count": experiment_count,
        "candidate_seed_run_count": candidate_seed_count,
        "reloadable_bundle_fold_count": bundle_fold_count,
    }
    for (
        source_name,
        source_id,
        experiment_count,
        candidate_seed_count,
        bundle_fold_count,
    ) in _EXPECTED_SOURCE_SCOPE
}
_EXPECTED_ELIGIBLE_CONDITIONS = {
    "spend_aggregate": FROZEN_STAGE4_SOURCE_CONDITIONS[
        SPEND_AGGREGATE_SOURCE_ID
    ],
    "bagen_sokoban": FROZEN_STAGE4_SOURCE_CONDITIONS[
        BAGEN_SOKOBAN_SOURCE_ID
    ],
    "bagen_swebench": frozenset(
        {
            "condition:54cb50fce273f0aa2d74",
            "condition:949ac3b7a342718cd505",
            "condition:d94078c05d91b0d58aee",
            "condition:dce86ced00dc11c77205",
            "condition:f95ae2a5e11682f6b7fc",
        }
    ),
    "spend_openhands": FROZEN_STAGE4_SOURCE_CONDITIONS[SPEND_SOURCE_ID],
}
_EXPLICIT_CODE_PATHS = frozenset(
    {
        STAGE4_RUNNER_RELATIVE,
        STAGE2_RUNNER_RELATIVE,
        STAGE2_METADATA_EXTRACTOR_RELATIVE,
        STAGE2_SOKOBAN_AUDITOR_RELATIVE,
        STAGE1_VERIFIER_RELATIVE,
        DATA_FOUNDATION_BASELINE_RELATIVE,
        STAGE2_AUXILIARY_MANIFEST_RELATIVE,
    }
)
_DIAGNOSTICS_DIRECT_CODE_PATHS = frozenset(
    {
        DIAGNOSTICS_RUNNER_RELATIVE,
        DIAGNOSTICS_SUMMARIZER_RELATIVE,
    }
)
_MISSING_MASK_INVARIANT = {
    "invariant_id": STAGE4_MISSING_MASK_INVARIANT_ID,
    "estimator_ids": ["gru_residual", "independent_mlp"],
    "required_behavior": (
        "neural_inputs_keep_explicit_missing_indicators_and_history_ablations_keep_"
        "missing_usage_attempts"
    ),
    "prohibited_ablation": "disable_or_remove_missing_telemetry_masks",
    "violation_action": "fail_closed",
}
_TASK_FEATURE_CANDIDATES = (
    "empirical",
    "lightgbm_history",
    "mlp_history",
    "lightgbm_without_progress",
    "lightgbm_without_tools_errors",
    "lightgbm_structured",
)
_CALL_PRE_CANDIDATES = (
    "empirical",
    "pre_request_char_message_length",
    "lightgbm_history",
    "mlp_history",
)
_SEED_POLICY_CANDIDATES = (
    "cross_position_deduct_raw_repaired_oof_seed",
    "cross_position_deduct_point_only_oof_seed",
)
_FROZEN_FEATURE_SETS = (
    NO_FEATURES,
    STAGE2_STRUCTURED_FEATURES,
    STAGE2_HISTORY_FEATURES,
    SPEND_AGGREGATE_STRUCTURED_FEATURES,
    SPEND_AGGREGATE_TASK_CHARS,
    STAGE4_NO_PROGRESS_FEATURES,
    STAGE4_NO_TOOLS_ERRORS_FEATURES,
    STAGE4_PRE_REQUEST_CHAR_MESSAGE_FEATURES,
    STAGE4_RETRIEVAL_HISTORY_FEATURES,
    STAGE4_G3_FEATURES,
)
_EXPECTED_FEATURE_HASHES = {
    feature_set.feature_set_id: feature_set.content_hash
    for feature_set in _FROZEN_FEATURE_SETS
}
_INTERVAL_METRIC_FIELDS = {
    "interval_diagnostics_id",
    "interval_below_truth_rate",
    "interval_above_truth_rate",
    "target_exceeds_upper_rate",
    "mean_extra_reserved_tokens",
    "raw_interval_below_truth_rate",
    "raw_interval_above_truth_rate",
    "raw_target_exceeds_upper_rate",
    "raw_mean_extra_reserved_tokens",
}
_FINAL_HOLDOUT_SENTINEL = {
    "evaluated": False,
    "prediction_count": 0,
    "target_values_used_for_fit_calibration_scoring": False,
    "selection_claim": "none",
}


class Stage4CompletionReleaseError(RuntimeError):
    """The Stage 4 completion supplement does not close safely."""


@dataclass(frozen=True)
class Stage4CompletionReleaseVerification:
    lock_path: str
    report_path: str
    source_commit: str
    code_tree_sha256: str
    verified_artifact_count: int
    verified_experiment_count: int
    verified_candidate_seed_run_count: int
    independently_loaded_bundle_count: int
    verified_diagnostics_artifact_count: int
    verified_diagnostics_record_count: int
    verified_diagnostics_bundle_count: int
    call_pre_mlp_cell_count: int
    call_pre_mlp_bundle_count: int
    seed_policy_cell_count: int
    seed_policy_bundle_count: int
    parent_final_holdout_evaluation_count: int
    parent_final_holdout_prediction_count: int
    final_source_opened: bool
    final_labels_read: bool


@dataclass(frozen=True)
class _Coverage:
    experiment_count: int = 0
    candidate_seed_run_count: int = 0
    reloadable_bundle_fold_count: int = 0
    call_pre_mlp_cell_count: int = 0
    call_pre_mlp_bundle_fold_count: int = 0
    seed_policy_cell_count: int = 0
    seed_policy_bundle_fold_count: int = 0

    def plus(self, other: _Coverage) -> _Coverage:
        return _Coverage(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion JSON contains duplicate keys"
            )
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise Stage4CompletionReleaseError(
        f"Stage 4 completion JSON contains non-finite value {value}"
    )


def _reject_non_finite(value: object, *, description: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage4CompletionReleaseError(
                f"{description} contains a non-finite number"
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, description=f"{description}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, description=f"{description}[{index}]")


def _regular_file(path: Path, *, maximum_bytes: int, description: str) -> bytes:
    try:
        return _release_regular_file(
            path,
            maximum_bytes=maximum_bytes,
            description=description,
        )
    except (OSError, Stage4ReleaseError) as exc:
        raise Stage4CompletionReleaseError(str(exc)) from exc


def _load_json(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> Mapping[str, Any]:
    payload = _regular_file(
        path,
        maximum_bytes=maximum_bytes,
        description=description,
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4CompletionReleaseError(
            f"{description} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise Stage4CompletionReleaseError(f"{description} must be a JSON object")
    _reject_non_finite(value, description=description)
    return value


def _exact(
    value: object,
    keys: set[str],
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Stage4CompletionReleaseError(f"{description} keys do not match")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage4CompletionReleaseError(
            f"{description} must be normalized non-empty text"
        )
    return value


def _integer(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage4CompletionReleaseError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage4CompletionReleaseError(
            f"{description} must be a lowercase SHA-256"
        )
    return value


def _commit(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage4CompletionReleaseError(
            f"{description} must be a lowercase Git commit"
        )
    return value


def _relative(value: object, *, description: str) -> str:
    try:
        return _safe_relative(value, label=description)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            f"{description} is not a safe relative path"
        ) from exc


def _path(root: Path, value: object, *, description: str) -> Path:
    relative = _relative(value, description=description)
    try:
        return _repo_path(root, relative, label=description)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(f"{description} is unsafe") from exc


def _git(
    root: Path,
    *arguments: str,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise Stage4CompletionReleaseError(f"Git command failed: {message}")
    if len(completed.stdout) > maximum_bytes:
        raise Stage4CompletionReleaseError(
            "Git command output exceeds its safe size limit"
        )
    return completed.stdout


def _validate_release_document(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "release_schema_version",
            "stage_name",
            "policy_id",
            "release_control",
            "source_binding",
            "parent_final_release",
            "artifacts",
            "diagnostics_artifact",
            "protocol",
            "report",
        },
        description="Stage 4 completion release",
    )
    if (
        value["release_schema_version"] != RELEASE_SCHEMA_VERSION
        or value["stage_name"] != RELEASE_STAGE_NAME
        or value["policy_id"] != RELEASE_POLICY_ID
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release identity is invalid"
        )

    release_control = _exact(
        value["release_control"],
        {"release_tag"},
        description="Stage 4 completion release control",
    )
    if release_control["release_tag"] != COMPLETION_RELEASE_TAG:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release-control tag is invalid"
        )

    source = _exact(
        value["source_binding"],
        {"policy_id", "git_commit", "code_tree_sha256", "source_tag"},
        description="Stage 4 completion source binding",
    )
    if source["policy_id"] != SOURCE_CODE_POLICY_ID:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion source policy is invalid"
        )
    _commit(source["git_commit"], description="Stage 4 completion source commit")
    _sha256(
        source["code_tree_sha256"],
        description="Stage 4 completion code tree",
    )
    if source["source_tag"] != SOURCE_TAG:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion source tag is invalid"
        )

    parent = _exact(
        value["parent_final_release"],
        {
            "lock_path",
            "lock_sha256",
            "final_release_tag",
            "final_artifact_id",
            "final_holdout_evaluation_count",
            "final_holdout_prediction_count",
        },
        description="Stage 4 parent final release",
    )
    if (
        parent["lock_path"] != PARENT_RELEASE_LOCK
        or parent["final_release_tag"] != PARENT_FINAL_TAG
        or parent["final_artifact_id"] != PARENT_FINAL_ARTIFACT_ID
        or parent["final_holdout_evaluation_count"]
        != EXPECTED_FINAL_EVALUATION_COUNT
        or parent["final_holdout_prediction_count"]
        != EXPECTED_FINAL_PREDICTION_COUNT
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 parent final release identity or count differs"
        )
    _sha256(parent["lock_sha256"], description="Stage 4 parent lock")

    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion requires exactly four artifacts"
        )
    names: list[str] = []
    for index, item in enumerate(artifacts):
        artifact = _exact(
            item,
            {
                "source_name",
                "source_id",
                "path",
                "artifact_id",
                "run_id",
                "results_payload_sha256",
                "manifest_sha256",
                "matrix_id",
                "experiment_count",
                "candidate_seed_run_count",
                "manifest_file_count",
            },
            description=f"Stage 4 completion artifact {index}",
        )
        source_name = _text(
            artifact["source_name"],
            description=f"Stage 4 completion artifact {index} source name",
        )
        names.append(source_name)
        expected = _EXPECTED_BY_SOURCE.get(source_name)
        if expected is None or any(
            artifact[key] != expected[key]
            for key in (
                "source_id",
                "experiment_count",
                "candidate_seed_run_count",
            )
        ):
            raise Stage4CompletionReleaseError(
                f"Stage 4 completion artifact {source_name!r} scope differs"
            )
        relative = _relative(
            artifact["path"],
            description=f"Stage 4 completion artifact {source_name} path",
        )
        if not relative.startswith("workspace/stage4/runs/s4-"):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion artifact path is outside the run root"
            )
        run_id = _text(
            artifact["run_id"],
            description=f"Stage 4 completion artifact {source_name} run id",
        )
        if (
            len(run_id) < 20
            or any(character not in "0123456789abcdef" for character in run_id)
            or relative.rsplit("/", 1)[-1] != f"s4-{run_id[:20]}"
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion artifact path and run id differ"
            )
        for field in (
            "artifact_id",
            "results_payload_sha256",
            "manifest_sha256",
            "matrix_id",
        ):
            _sha256(
                artifact[field],
                description=f"Stage 4 completion artifact {source_name} {field}",
            )
        _integer(
            artifact["manifest_file_count"],
            description=f"Stage 4 completion artifact {source_name} file count",
            minimum=1,
        )
    if tuple(names) != tuple(item[0] for item in _EXPECTED_SOURCE_SCOPE):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion artifacts are missing, duplicated, or out of order"
        )

    diagnostics = _exact(
        value["diagnostics_artifact"],
        {
            "path",
            "artifact_id",
            "manifest_sha256",
            "results_payload_sha256",
            "training_source_commit",
            "diagnostics_code_binding",
            "source_artifact_ids",
            "coverage",
        },
        description="Stage 4 completion diagnostics artifact",
    )
    diagnostics_relative = _relative(
        diagnostics["path"],
        description="Stage 4 completion diagnostics artifact path",
    )
    if not diagnostics_relative.startswith(
        "workspace/stage4/completion_diagnostics/s4diag-"
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics path is outside its frozen root"
        )
    for field in ("artifact_id", "manifest_sha256", "results_payload_sha256"):
        _sha256(
            diagnostics[field],
            description=f"Stage 4 completion diagnostics {field}",
        )
    if diagnostics["training_source_commit"] != source["git_commit"]:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics training source commit differs"
        )
    diagnostics_code = _exact(
        diagnostics["diagnostics_code_binding"],
        {"git_commit", "code_tree_sha256", "code_paths", "source_tag"},
        description="Stage 4 completion diagnostics code binding",
    )
    _commit(
        diagnostics_code["git_commit"],
        description="Stage 4 completion diagnostics code commit",
    )
    _sha256(
        diagnostics_code["code_tree_sha256"],
        description="Stage 4 completion diagnostics code tree",
    )
    if diagnostics_code["source_tag"] != DIAGNOSTICS_SOURCE_TAG:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics source tag differs"
        )
    code_paths = diagnostics_code["code_paths"]
    if (
        not isinstance(code_paths, list)
        or code_paths != sorted(set(code_paths))
        or DIAGNOSTICS_RUNNER_RELATIVE not in code_paths
        or any(
            not isinstance(path, str)
            or _relative(
                path,
                description="Stage 4 completion diagnostic code path",
            )
            != path
            for path in code_paths
        )
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics code paths are invalid"
        )
    source_artifact_ids = _exact(
        diagnostics["source_artifact_ids"],
        {item[0] for item in _EXPECTED_SOURCE_SCOPE},
        description="Stage 4 completion diagnostics source artifact ids",
    )
    locked_artifact_ids = {
        str(artifact["source_name"]): str(artifact["artifact_id"])
        for artifact in artifacts
    }
    if dict(source_artifact_ids) != locked_artifact_ids:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics source artifact ids differ"
        )
    diagnostics_coverage = _exact(
        diagnostics["coverage"],
        set(_DIAGNOSTICS_COVERAGE),
        description="Stage 4 completion diagnostics coverage",
    )
    if dict(diagnostics_coverage) != _DIAGNOSTICS_COVERAGE:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint-only diagnostics coverage differs"
        )

    protocol = _exact(
        value["protocol"],
        {
            "development_only",
            "artifact_count",
            "experiment_count",
            "candidate_seed_run_count",
            "reloadable_bundle_fold_count",
            "call_pre_mlp_cell_count",
            "call_pre_mlp_bundle_fold_count",
            "seed_policy_cell_count",
            "seed_policy_bundle_fold_count",
            "diagnostics_artifact_count",
            "diagnostics_bound_source_artifact_count",
            "diagnostics_lifecycle_source_count",
            "diagnostics_lifecycle_condition_count",
            "diagnostics_candidate_count",
            "diagnostics_candidate_cell_count",
            "diagnostics_candidate_seed_count",
            "diagnostics_bundle_count",
            "diagnostics_checkpoint_verified_candidate_seed_count",
            "diagnostics_lifecycle_replayed_candidate_seed_count",
            "diagnostics_lifecycle_metrics_unavailable_candidate_seed_count",
            "split_seeds",
            "outer_folds",
            "inner_folds",
            "final_holdout_evaluated",
            "final_holdout_prediction_count",
            "final_holdout_target_values_used_for_fit_calibration_scoring",
            "final_holdout_selection_claim",
        },
        description="Stage 4 completion protocol",
    )
    expected_protocol = {
        "development_only": True,
        "artifact_count": EXPECTED_ARTIFACT_COUNT,
        "experiment_count": EXPECTED_EXPERIMENT_COUNT,
        "candidate_seed_run_count": EXPECTED_CANDIDATE_SEED_RUN_COUNT,
        "reloadable_bundle_fold_count": EXPECTED_RELOADABLE_BUNDLE_FOLD_COUNT,
        "call_pre_mlp_cell_count": EXPECTED_CALL_PRE_MLP_CELL_COUNT,
        "call_pre_mlp_bundle_fold_count": (
            EXPECTED_CALL_PRE_MLP_BUNDLE_FOLD_COUNT
        ),
        "seed_policy_cell_count": EXPECTED_SEED_POLICY_CELL_COUNT,
        "seed_policy_bundle_fold_count": (
            EXPECTED_SEED_POLICY_BUNDLE_FOLD_COUNT
        ),
        "diagnostics_artifact_count": EXPECTED_DIAGNOSTICS_ARTIFACT_COUNT,
        "diagnostics_bound_source_artifact_count": (
            EXPECTED_DIAGNOSTICS_BOUND_SOURCE_COUNT
        ),
        "diagnostics_lifecycle_source_count": (
            EXPECTED_DIAGNOSTICS_LIFECYCLE_SOURCE_COUNT
        ),
        "diagnostics_lifecycle_condition_count": (
            EXPECTED_DIAGNOSTICS_LIFECYCLE_CONDITION_COUNT
        ),
        "diagnostics_candidate_count": EXPECTED_DIAGNOSTICS_CANDIDATE_COUNT,
        "diagnostics_candidate_cell_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_CELL_COUNT
        ),
        "diagnostics_candidate_seed_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
        ),
        "diagnostics_bundle_count": EXPECTED_DIAGNOSTICS_BUNDLE_COUNT,
        "diagnostics_checkpoint_verified_candidate_seed_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
        ),
        "diagnostics_lifecycle_replayed_candidate_seed_count": 0,
        "diagnostics_lifecycle_metrics_unavailable_candidate_seed_count": (
            EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
        ),
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "inner_folds": EXPECTED_INNER_FOLDS,
        "final_holdout_evaluated": False,
        "final_holdout_prediction_count": 0,
        "final_holdout_target_values_used_for_fit_calibration_scoring": False,
        "final_holdout_selection_claim": "none",
    }
    if dict(protocol) != expected_protocol:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion protocol differs from the frozen scope"
        )

    report = _exact(
        value["report"],
        {"path", "sha256"},
        description="Stage 4 completion report",
    )
    if report["path"] != DEFAULT_REPORT:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion report path is invalid"
        )
    _sha256(report["sha256"], description="Stage 4 completion report")


def _code_paths_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(_EXPLICIT_CODE_PATHS),
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage4CompletionReleaseError(
                "Git returned a non-UTF-8 Stage 4 completion code path"
            ) from exc
        relative = _relative(
            relative,
            description="Stage 4 completion historical code path",
        )
        if relative in _EXPLICIT_CODE_PATHS or (
            relative.startswith("src/token_prediction/")
            and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    if not _EXPLICIT_CODE_PATHS <= set(resolved) or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage4CompletionReleaseError(
            "source commit lacks the complete Stage 4 code set"
        )
    return resolved


def _code_binding_at_commit(root: Path, commit: str) -> Mapping[str, object]:
    resolved_commit = (
        _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    if resolved_commit != commit:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion source is not an exact commit"
        )
    paths = _code_paths_at_commit(root, commit)
    items = [
        (relative, _git(root, "show", f"{commit}:{relative}"))
        for relative in paths
    ]
    return {
        "git_commit": commit,
        "code_tree_sha256": _framed_code_hash(items),
        "code_paths": list(paths),
    }


def _diagnostics_code_paths_at_commit(
    root: Path,
    commit: str,
) -> tuple[str, ...]:
    """Derive the exact replay closure from the immutable diagnostics commit."""

    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        "src/token_prediction",
        *sorted(_EXPLICIT_CODE_PATHS | _DIAGNOSTICS_DIRECT_CODE_PATHS),
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Stage4CompletionReleaseError(
                "Git returned a non-UTF-8 diagnostics code path"
            ) from exc
        relative = _relative(
            relative,
            description="Stage 4 completion diagnostics historical code path",
        )
        if relative in (
            _EXPLICIT_CODE_PATHS | _DIAGNOSTICS_DIRECT_CODE_PATHS
        ) or (
            relative.startswith("src/token_prediction/")
            and relative.endswith(".py")
        ):
            paths.append(relative)
    resolved = tuple(sorted(set(paths)))
    required = _EXPLICIT_CODE_PATHS | _DIAGNOSTICS_DIRECT_CODE_PATHS
    if not required <= set(resolved) or not any(
        path.startswith("src/token_prediction/") for path in resolved
    ):
        raise Stage4CompletionReleaseError(
            "diagnostics commit lacks its complete execution closure"
        )
    return resolved


def _verify_diagnostics_code_binding(
    root: Path,
    value: Mapping[str, Any],
    *,
    require_git_clean: bool,
) -> Mapping[str, object]:
    commit = str(value["git_commit"])
    paths = tuple(str(path) for path in value["code_paths"])
    expected_paths = _diagnostics_code_paths_at_commit(root, commit)
    if paths != expected_paths:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics code closure differs from its tag"
        )
    tagged = (
        _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{value['source_tag']}^{{commit}}",
        )
        .decode("ascii")
        .strip()
    )
    if tagged != commit:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics source tag points elsewhere"
        )
    resolved = (
        _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    if resolved != commit:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics source is not an exact commit"
        )
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(root, "ls-files", "-z", "--", *paths).split(b"\0")
        if item
    }
    if tracked != set(paths):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics code paths must all be tracked"
        )
    if require_git_clean and _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--",
        *paths,
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics code paths must be clean"
        )
    items: list[tuple[str, bytes]] = []
    for relative in paths:
        committed = _git(
            root,
            "show",
            f"{commit}:{relative}",
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        )
        current = _regular_file(
            _path(
                root,
                relative,
                description="Stage 4 completion diagnostics code path",
            ),
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
            description=f"Stage 4 completion diagnostics code {relative}",
        )
        if current != committed:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostics code differs from source tag"
            )
        items.append((relative, committed))
    actual = {
        "git_commit": commit,
        "code_tree_sha256": _framed_code_hash(items),
        "code_paths": list(paths),
        "source_tag": value["source_tag"],
    }
    if actual != dict(value):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics code tree differs"
        )
    return actual


def _verify_parent_release(
    root: Path,
    release: Mapping[str, Any],
) -> Mapping[str, Any]:
    parent_binding = release["parent_final_release"]
    relative = str(parent_binding["lock_path"])
    path = _path(root, relative, description="Stage 4 parent final release lock")
    payload = _regular_file(
        path,
        maximum_bytes=MAX_PARENT_RELEASE_BYTES,
        description="Stage 4 parent final release lock",
    )
    if hashlib.sha256(payload).hexdigest() != parent_binding["lock_sha256"]:
        raise Stage4CompletionReleaseError(
            "Stage 4 parent final release lock SHA-256 differs"
        )
    parent = _load_json(
        path,
        maximum_bytes=MAX_PARENT_RELEASE_BYTES,
        description="Stage 4 parent final release lock",
    )
    try:
        _validate_parent_release_document(parent)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 parent final release schema is invalid"
        ) from exc
    if (
        parent["final_artifact"]["artifact_id"]
        != parent_binding["final_artifact_id"]
        or parent["protocol"]["final_holdout_evaluation_count"]
        != parent_binding["final_holdout_evaluation_count"]
        or parent["protocol"]["final_holdout_prediction_count"]
        != parent_binding["final_holdout_prediction_count"]
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 parent final release facts differ from the supplement"
        )
    tag = str(parent_binding["final_release_tag"])
    tagged_commit = (
        _git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    _commit(tagged_commit, description="Stage 4 parent final release tag commit")
    if (
        _git(
            root,
            "show",
            f"{tagged_commit}:{relative}",
            maximum_bytes=MAX_PARENT_RELEASE_BYTES,
        )
        != payload
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 parent final lock differs from its immutable tag"
        )
    return parent


def _release_control_size_limit(relative: str) -> int:
    if relative == DEFAULT_RELEASE_LOCK:
        return MAX_RELEASE_JSON_BYTES
    if relative == DEFAULT_REPORT:
        return MAX_REPORT_BYTES
    return MAX_BUNDLE_FILE_BYTES


def _verify_release_control_tag(
    root: Path,
    release: Mapping[str, Any],
    *,
    release_relative: str,
) -> str:
    """Byte-bind the formal release controls to the fixed protected tag."""

    if release_relative != DEFAULT_RELEASE_LOCK:
        raise Stage4CompletionReleaseError(
            "tracked-only verification requires the formal completion lock"
        )
    tag = str(release["release_control"]["release_tag"])
    if tag != COMPLETION_RELEASE_TAG:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release-control tag identity differs"
        )
    try:
        tagged_commit = (
            _git(
                root,
                "rev-parse",
                "--verify",
                f"refs/tags/{tag}^{{commit}}",
            )
            .decode("ascii")
            .strip()
        )
    except (Stage4CompletionReleaseError, UnicodeError) as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release-control tag is missing or invalid"
        ) from exc
    _commit(
        tagged_commit,
        description="Stage 4 completion release-control tag commit",
    )
    for relative in RELEASE_CONTROL_PATHS:
        maximum_bytes = _release_control_size_limit(relative)
        try:
            tagged_payload = _git(
                root,
                "show",
                f"{tagged_commit}:{relative}",
                maximum_bytes=maximum_bytes,
            )
        except Stage4CompletionReleaseError as exc:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion release-control tag omits "
                f"{relative}"
            ) from exc
        current_payload = _regular_file(
            _path(
                root,
                relative,
                description="Stage 4 completion release control",
            ),
            maximum_bytes=maximum_bytes,
            description=f"Stage 4 completion release control {relative}",
        )
        if current_payload != tagged_payload:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion release-control tag moved or current "
                f"bytes differ for {relative}"
            )
    return tagged_commit


def _verify_tracked_bindings(
    root: Path,
    release: Mapping[str, Any],
    *,
    release_relative: str,
    require_git_clean: bool,
    require_release_control_tag: bool,
) -> Mapping[str, object]:
    report_relative = str(release["report"]["path"])
    parent_relative = str(release["parent_final_release"]["lock_path"])
    if require_release_control_tag:
        if (
            release_relative != DEFAULT_RELEASE_LOCK
            or report_relative != DEFAULT_REPORT
        ):
            raise Stage4CompletionReleaseError(
                "tracked-only verification requires formal completion controls"
            )
        controls = tuple(
            dict.fromkeys((*RELEASE_CONTROL_PATHS, parent_relative))
        )
    else:
        controls = (release_relative, report_relative, parent_relative)
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(root, "ls-files", "-z", "--", *controls).split(b"\0")
        if item
    }
    if tracked != set(controls):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release controls must be tracked"
        )
    if require_git_clean and _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *controls,
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion release controls must be clean at HEAD"
        )
    if require_release_control_tag:
        _verify_release_control_tag(
            root,
            release,
            release_relative=release_relative,
        )

    source = release["source_binding"]
    commit = str(source["git_commit"])
    tagged = (
        _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{source['source_tag']}^{{commit}}",
        )
        .decode("ascii")
        .strip()
    )
    if tagged != commit:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion source tag points to another commit"
        )
    code = _code_binding_at_commit(root, commit)
    if (
        code["git_commit"] != source["git_commit"]
        or code["code_tree_sha256"] != source["code_tree_sha256"]
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion source code tree differs"
        )
    _verify_diagnostics_code_binding(
        root,
        release["diagnostics_artifact"]["diagnostics_code_binding"],
        require_git_clean=require_git_clean,
    )

    report_payload = _regular_file(
        _path(root, report_relative, description="Stage 4 completion report"),
        maximum_bytes=MAX_REPORT_BYTES,
        description="Stage 4 completion report",
    )
    if hashlib.sha256(report_payload).hexdigest() != release["report"]["sha256"]:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion report SHA-256 differs"
        )
    _verify_parent_release(root, release)
    return code


def _canonical_report_payload(
    root: Path,
    release: Mapping[str, Any],
) -> bytes:
    try:
        artifacts = tuple(
            load_development_artifact(
                ArtifactReference(
                    path=_path(
                        root,
                        item["path"],
                        description=(
                            "Stage 4 completion report source artifact"
                        ),
                    ),
                    expected_artifact_id=str(item["artifact_id"]),
                    expected_results_payload_sha256=str(
                        item["results_payload_sha256"]
                    ),
                )
            )
            for item in release["artifacts"]
        )
        diagnostics = load_completion_diagnostics_artifact(
            release["diagnostics_artifact"]["path"],
            repo_root=root,
            diagnostics_root=(
                root
                / "workspace"
                / "stage4"
                / "completion_diagnostics"
            ),
        )
        if (
            diagnostics.artifact_id
            != release["diagnostics_artifact"]["artifact_id"]
            or diagnostics.results_payload_sha256
            != release["diagnostics_artifact"]["results_payload_sha256"]
        ):
            raise Stage4CompletionReleaseError(
                "canonical report diagnostics binding differs"
            )
        summary = build_completion_summary(
            artifacts,
            diagnostics=diagnostics,
        )
        return (render_markdown(summary) + "\n").encode("utf-8")
    except Stage4CompletionReleaseError:
        raise
    except (CompletionSummaryError, OSError, TypeError, ValueError) as exc:
        raise Stage4CompletionReleaseError(
            "canonical completion report cannot be reconstructed"
        ) from exc


def _verify_report_semantics(
    root: Path,
    release: Mapping[str, Any],
) -> None:
    expected = _canonical_report_payload(root, release)
    actual = _regular_file(
        _path(
            root,
            release["report"]["path"],
            description="Stage 4 completion report",
        ),
        maximum_bytes=MAX_REPORT_BYTES,
        description="Stage 4 completion report",
    )
    if actual != expected:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion report differs from canonical artifact summary"
        )


def _artifact_key(value: object, *, prefix: str, description: str) -> str:
    text = _text(value, description=description)
    if (
        len(text) != 18
        or not text.startswith(f"{prefix}_")
        or any(character not in "0123456789abcdef" for character in text[2:])
    ):
        raise Stage4CompletionReleaseError(
            f"{description} is not a compact artifact key"
        )
    return text


def _expected_artifact_key(kind: str, identity: str) -> str:
    if kind not in {"e", "c"} or not identity:
        raise Stage4CompletionReleaseError("invalid artifact-key identity")
    digest = hashlib.sha256(
        f"{STAGE4_ARTIFACT_LAYOUT_ID}\0{kind}\0{identity}".encode("utf-8")
    ).hexdigest()
    return f"{kind}_{digest[:16]}"


def _verify_plan_candidate_hash(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    candidate = _exact(
        value,
        {
            "candidate_id",
            "candidate_hash",
            "estimator_id",
            "role",
            "feature_set_id",
            "feature_set_hash",
            "params",
            "initializer_params",
            "graph",
            "seed_policy_hash",
            "ablation",
        },
        description=description,
    )
    feature_set_id = _text(
        candidate["feature_set_id"],
        description=f"{description} feature set id",
    )
    expected_feature_hash = _EXPECTED_FEATURE_HASHES.get(feature_set_id)
    if (
        expected_feature_hash is None
        or _sha256(
            candidate["feature_set_hash"],
            description=f"{description} feature set hash",
        )
        != expected_feature_hash
    ):
        raise Stage4CompletionReleaseError(
            f"{description} feature set hash is not the frozen implementation"
        )
    params = candidate["params"]
    initializer_params = candidate["initializer_params"]
    graph = _exact(
        candidate["graph"],
        {
            "initializer_estimator_id",
            "updater_estimator_id",
            "lifecycle_schema_id",
            "seed_policy_id",
            "inner_split_policy_id",
        },
        description=f"{description} graph",
    )
    if not isinstance(params, Mapping) or not isinstance(
        initializer_params,
        Mapping,
    ):
        raise Stage4CompletionReleaseError(
            f"{description} parameters must be JSON objects"
        )
    expected_candidate_hash = _semantic_sha256(
        {
            "estimator_id": candidate["estimator_id"],
            "feature_set_hash": expected_feature_hash,
            "params": dict(params),
            "initializer_params": dict(initializer_params),
            "graph": dict(graph),
        }
    )
    if (
        _sha256(
            candidate["candidate_hash"],
            description=f"{description} candidate hash",
        )
        != expected_candidate_hash
        or graph["updater_estimator_id"] != candidate["estimator_id"]
    ):
        raise Stage4CompletionReleaseError(
            f"{description} candidate hash or graph does not close"
        )
    return candidate


def _candidate_projection(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_hash": candidate.get("candidate_hash"),
        "estimator_id": candidate.get("estimator_id"),
        "feature_set_id": candidate.get("feature_set_id"),
        "feature_set_hash": candidate.get("feature_set_hash"),
        "candidate_graph": candidate.get("candidate_graph"),
    }


def _plan_candidate_projection(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_hash": candidate.get("candidate_hash"),
        "estimator_id": candidate.get("estimator_id"),
        "feature_set_id": candidate.get("feature_set_id"),
        "feature_set_hash": candidate.get("feature_set_hash"),
        "candidate_graph": candidate.get("graph"),
    }


def _requires_reloadable_bundle(candidate: Mapping[str, Any]) -> bool:
    graph = candidate.get("candidate_graph")
    if not isinstance(graph, Mapping):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion candidate graph is invalid"
        )
    return (
        candidate.get("estimator_id")
        in {"independent_mlp", "lightgbm_quantile"}
        or graph.get("initializer_estimator_id") != "none"
    )


def _verify_interval_metrics(value: object, *, description: str) -> None:
    if not isinstance(value, Mapping) or not _INTERVAL_METRIC_FIELDS <= set(value):
        raise Stage4CompletionReleaseError(
            f"{description} lacks interval-tail or reserve metrics"
        )
    if value["interval_diagnostics_id"] != "weighted_interval_tail_and_reserve_v1":
        raise Stage4CompletionReleaseError(
            f"{description} interval diagnostics policy differs"
        )
    for key in _INTERVAL_METRIC_FIELDS - {"interval_diagnostics_id"}:
        metric = value[key]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise Stage4CompletionReleaseError(
                f"{description}.{key} must be finite"
            )


def _verify_matrix_binding(
    results: Mapping[str, Any],
    *,
    source_name: str,
) -> list[Mapping[str, Any]]:
    matrix = results.get("matrix")
    if not isinstance(matrix, Mapping):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion matrix must be an object"
        )
    expected_matrix_keys = {
        "schema_version",
        "policy_id",
        "source_id",
        "development_protocol_id",
        "capability_contract_hash",
        "minimum_development_tasks",
        "plans",
        "gates",
        "telemetry_decisions",
        "safety_invariants",
        "matrix_id",
    }
    if set(matrix) != expected_matrix_keys:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion matrix keys differ"
        )
    matrix_semantic = dict(matrix)
    matrix_id = _sha256(
        matrix_semantic.pop("matrix_id"),
        description="Stage 4 completion matrix id",
    )
    if (
        matrix.get("schema_version") != STAGE4_MATRIX_SCHEMA_VERSION
        or matrix.get("policy_id") != STAGE4_MATRIX_POLICY_ID
        or matrix.get("source_id") != results["source"]["source_id"]
        or _semantic_sha256(matrix_semantic) != matrix_id
        or (
            "matrix_id" in results["summary"]
            and matrix_id != results["summary"]["matrix_id"]
        )
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion matrix identity differs"
        )
    if matrix.get("safety_invariants") != [_MISSING_MASK_INVARIANT]:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion missing-mask safety invariant differs"
        )
    plans = matrix.get("plans")
    experiments = results.get("experiments")
    if not isinstance(plans, list) or not isinstance(experiments, list):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion plans and experiments must be lists"
        )
    plan_by_id: dict[str, Mapping[str, Any]] = {}
    for plan in plans:
        if not isinstance(plan, Mapping) or not isinstance(plan.get("spec"), Mapping):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion matrix plan is invalid"
            )
        experiment_id = _text(
            plan["spec"].get("experiment_id"),
            description="Stage 4 completion matrix experiment id",
        )
        if experiment_id in plan_by_id:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion matrix experiment id is duplicated"
            )
        plan_by_id[experiment_id] = plan
    if len(plan_by_id) != len(experiments):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion matrix/result cardinality differs"
        )
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion experiment is invalid"
            )
        experiment_id = _text(
            experiment.get("experiment_id"),
            description="Stage 4 completion experiment id",
        )
        plan = plan_by_id.get(experiment_id)
        if plan is None:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion result is absent from its matrix"
            )
        spec = plan["spec"]
        for result_key, plan_key in (
            ("position", "position"),
            ("target", "target"),
            ("condition_id", "condition_id"),
            ("alpha", "alpha"),
            ("calibrator_id", "calibrator_id"),
            ("plan_role", "role"),
            ("axis", "axis"),
            ("reference_experiment_id", "reference_experiment_id"),
            ("allowed_config_paths", "allowed_config_paths"),
        ):
            expected = plan.get(plan_key) if plan_key in plan else spec.get(plan_key)
            if experiment.get(result_key) != expected:
                raise Stage4CompletionReleaseError(
                    "Stage 4 completion matrix/result experiment binding differs"
                )
        plan_candidates = spec.get("candidates")
        result_candidates = experiment.get("candidates")
        if not isinstance(plan_candidates, list) or not isinstance(
            result_candidates, list
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion candidates must be lists"
            )
        verified_plan_candidates = [
            _verify_plan_candidate_hash(
                item,
                description=(
                    f"Stage 4 completion {experiment_id} matrix candidate "
                    f"{index}"
                ),
            )
            for index, item in enumerate(plan_candidates)
        ]
        if [
            _plan_candidate_projection(item)
            for item in verified_plan_candidates
        ] != [
            _candidate_projection(item) for item in result_candidates
        ]:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion matrix/result candidates differ"
            )
    return experiments


def _expected_candidate_sets(
    source_name: str,
    experiments: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    observed_conditions = {str(item.get("condition_id")) for item in experiments}
    if observed_conditions != set(_EXPECTED_ELIGIBLE_CONDITIONS[source_name]):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} conditions differ"
        )
    call_pre_cells = 0
    seed_policy_cells = 0
    if source_name == "spend_aggregate":
        signatures = sorted(
            (
                str(experiment.get("position")),
                str(experiment.get("target")),
                str(experiment.get("calibrator_id")),
                tuple(
                    str(candidate.get("candidate_id"))
                    for candidate in experiment.get("candidates", [])
                ),
            )
            for experiment in experiments
        )
        expected = sorted(
            [
                (
                    "task_launch",
                    "task_total_accounted_tokens",
                    "none",
                    ("lightgbm_structured",),
                ),
                (
                    "task_launch",
                    "task_total_accounted_tokens",
                    "task_max_conformal",
                    ("lightgbm_structured",),
                ),
                (
                    "task_launch",
                    "task_total_accounted_tokens",
                    "task_max_conformal",
                    (
                        "empirical",
                        "task_chars_length",
                        "lightgbm_structured",
                    ),
                ),
            ]
        )
        if signatures != expected:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion aggregate matrix coverage differs"
            )
        return call_pre_cells, seed_policy_cells

    for condition_id in sorted(observed_conditions):
        scoped = [
            experiment
            for experiment in experiments
            if experiment.get("condition_id") == condition_id
        ]
        if len(scoped) != 7:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion condition must contain seven experiments"
            )
        task_update = [
            experiment
            for experiment in scoped
            if experiment.get("position") == "task_update"
            and experiment.get("target")
            == "task_provider_accounted_remaining_tokens"
        ]
        candidate_sets = [
            tuple(
                str(candidate.get("candidate_id"))
                for candidate in experiment.get("candidates", [])
            )
            for experiment in task_update
        ]
        if sorted(candidate_sets) != sorted(
            [
                _TASK_FEATURE_CANDIDATES,
                ("lightgbm_history",),
                ("lightgbm_history",),
                _SEED_POLICY_CANDIDATES,
            ]
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion task-update candidate coverage differs"
            )
        seed_experiments = [
            experiment
            for experiment in task_update
            if tuple(
                candidate.get("candidate_id")
                for candidate in experiment.get("candidates", [])
            )
            == _SEED_POLICY_CANDIDATES
        ]
        if len(seed_experiments) != 1:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion seed-policy cell is missing"
            )
        seed_policy_cells += 1
        policies = [
            candidate["candidate_graph"]["seed_policy_id"]
            for candidate in seed_experiments[0]["candidates"]
        ]
        if policies != [
            "inner_oof_uncalibrated_repaired_quantile_mean_v1",
            "inner_oof_uncalibrated_repaired_point_only_mean_v1",
        ]:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion seed policies differ"
            )

        call_pre = [
            experiment
            for experiment in scoped
            if experiment.get("position") == "call_pre"
        ]
        if (
            {experiment.get("target") for experiment in call_pre}
            != {target.value for target in STAGE4_CALL_PRE_TARGETS}
            or len(call_pre) != len(STAGE4_CALL_PRE_TARGETS)
            or any(
                tuple(
                    candidate.get("candidate_id")
                    for candidate in experiment.get("candidates", [])
                )
                != _CALL_PRE_CANDIDATES
                for experiment in call_pre
            )
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion Call-pre MLP coverage differs"
            )
        call_pre_cells += len(call_pre)
    return call_pre_cells, seed_policy_cells


def _verify_result_coverage(
    results: Mapping[str, Any],
    *,
    source_name: str,
) -> _Coverage:
    experiments = _verify_matrix_binding(results, source_name=source_name)
    expectation = _EXPECTED_BY_SOURCE[source_name]
    if len(experiments) != expectation["experiment_count"]:
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} experiment count differs"
        )
    call_pre_cells, seed_policy_cells = _expected_candidate_sets(
        source_name,
        experiments,
    )
    candidate_seed_count = 0
    reloadable_folds = 0
    call_pre_mlp_folds = 0
    seed_policy_folds = 0
    for experiment in experiments:
        experiment_artifact_key = _artifact_key(
            experiment.get("artifact_key"),
            prefix="e",
            description="Stage 4 completion experiment artifact key",
        )
        if experiment_artifact_key != _expected_artifact_key(
            "e",
            str(experiment["experiment_id"]),
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion experiment artifact key does not close"
            )
        candidates = experiment.get("candidates")
        if not isinstance(candidates, list):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion experiment candidates are invalid"
            )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise Stage4CompletionReleaseError(
                    "Stage 4 completion candidate is invalid"
                )
            candidate_artifact_key = _artifact_key(
                candidate.get("artifact_key"),
                prefix="c",
                description="Stage 4 completion candidate artifact key",
            )
            candidate_hash = _sha256(
                candidate.get("candidate_hash"),
                description="Stage 4 completion candidate hash",
            )
            if candidate_artifact_key != _expected_artifact_key(
                "c",
                candidate_hash,
            ):
                raise Stage4CompletionReleaseError(
                    "Stage 4 completion candidate artifact key does not close"
                )
            seed_results = candidate.get("seed_results")
            if not isinstance(seed_results, list) or [
                result.get("split_seed")
                for result in seed_results
                if isinstance(result, Mapping)
            ] != list(STAGE_SPLIT_SEEDS):
                raise Stage4CompletionReleaseError(
                    "Stage 4 completion candidate split seeds differ"
                )
            candidate_seed_count += len(seed_results)
            reloadable = _requires_reloadable_bundle(candidate)
            for seed_result in seed_results:
                if not isinstance(seed_result, Mapping):
                    raise Stage4CompletionReleaseError(
                        "Stage 4 completion seed result is invalid"
                    )
                _sha256(
                    seed_result.get("split_plan_id"),
                    description="Stage 4 completion split plan",
                )
                _verify_interval_metrics(
                    seed_result.get("metrics"),
                    description="Stage 4 completion seed metrics",
                )
                fold_metrics = seed_result.get("fold_metrics")
                if not isinstance(fold_metrics, Mapping):
                    raise Stage4CompletionReleaseError(
                        "Stage 4 completion fold metrics are invalid"
                    )
                for metric in fold_metrics.values():
                    _verify_interval_metrics(
                        metric,
                        description="Stage 4 completion fold metrics",
                    )
                folds = seed_result.get("reloadable_bundle_folds")
                parity = seed_result.get("bundle_reload_parity")
                if reloadable:
                    if (
                        folds != list(range(EXPECTED_OUTER_FOLDS))
                        or seed_result.get("fold_artifact_count")
                        != EXPECTED_OUTER_FOLDS
                        or parity
                        != {
                            "fold_count": EXPECTED_OUTER_FOLDS,
                            "status": "exact_during_execution",
                        }
                    ):
                        raise Stage4CompletionReleaseError(
                            "Stage 4 completion reloadable fold declaration differs"
                        )
                    reloadable_folds += len(folds)
                    if (
                        experiment.get("position") == "call_pre"
                        and candidate.get("candidate_id") == "mlp_history"
                    ):
                        call_pre_mlp_folds += len(folds)
                    if candidate.get("candidate_id") in _SEED_POLICY_CANDIDATES:
                        seed_policy_folds += len(folds)
                elif (
                    folds != []
                    or seed_result.get("fold_artifact_count") != 0
                    or parity
                    != {
                        "fold_count": 0,
                        "status": "not_applicable_stateless",
                    }
                ):
                    raise Stage4CompletionReleaseError(
                        "Stage 4 completion stateless candidate declared a bundle"
                    )
    if (
        candidate_seed_count != expectation["candidate_seed_run_count"]
        or reloadable_folds != expectation["reloadable_bundle_fold_count"]
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} candidate/fold count differs"
        )
    return _Coverage(
        experiment_count=len(experiments),
        candidate_seed_run_count=candidate_seed_count,
        reloadable_bundle_fold_count=reloadable_folds,
        call_pre_mlp_cell_count=call_pre_cells,
        call_pre_mlp_bundle_fold_count=call_pre_mlp_folds,
        seed_policy_cell_count=seed_policy_cells,
        seed_policy_bundle_fold_count=seed_policy_folds,
    )


def _bundle_mapping(directory: Path) -> Mapping[str, bytes]:
    if _is_link_or_reparse(directory) or not directory.is_dir():
        raise Stage4CompletionReleaseError(
            "Stage 4 completion lifecycle bundle directory is unsafe"
        )
    files: dict[str, bytes] = {}
    total_bytes = 0
    for path in sorted(directory.rglob("*")):
        if _is_link_or_reparse(path):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion lifecycle bundle contains a link"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise Stage4CompletionReleaseError(
                "Stage 4 completion lifecycle bundle contains a special node"
            )
        relative = path.relative_to(directory).as_posix()
        payload = _regular_file(
            path,
            maximum_bytes=MAX_BUNDLE_FILE_BYTES,
            description=f"Stage 4 completion lifecycle bundle {relative}",
        )
        files[relative] = payload
        total_bytes += len(payload)
        if (
            len(files) > MAX_BUNDLE_FILE_COUNT
            or total_bytes > MAX_BUNDLE_TOTAL_BYTES
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion lifecycle bundle exceeds its safe limit"
            )
    if not files:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion lifecycle bundle is empty"
        )
    return files


def _require_exact_artifact_payloads(
    manifest_files: Mapping[str, str],
    expected_files: set[str],
    *,
    description: str,
) -> None:
    observed = set(manifest_files)
    if observed != expected_files:
        missing = sorted(expected_files - observed)
        extra = sorted(observed - expected_files)
        raise Stage4CompletionReleaseError(
            f"{description} payload topology differs; "
            f"missing={missing}, extra={extra}"
        )


def _verify_fold_provenance(
    fold_root: Path,
    *,
    results: Mapping[str, Any],
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    seed_result: Mapping[str, Any],
    fold: int,
) -> Mapping[str, Any]:
    provenance = _load_json(
        fold_root / "provenance.json",
        maximum_bytes=MAX_PROVENANCE_BYTES,
        description="Stage 4 completion fold provenance",
    )
    expected = {
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "candidate_graph": candidate["candidate_graph"],
        "dataset_id": results["dataset"]["development_dataset_id"],
        "condition_id": experiment["condition_id"],
        "source_descriptor_hash": results["source"]["source_descriptor_hash"],
        "capability_contract_hash": results["source"][
            "capability_contract_hash"
        ],
        "split_plan_id": seed_result["split_plan_id"],
        "feature_set_hash": candidate["feature_set_hash"],
        "position": experiment["position"],
        "target": experiment["target"],
        "calibrator_id": experiment["calibrator_id"],
        "interval_alpha": experiment["alpha"],
        "code_hash": results["code_binding"]["code_tree_sha256"],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion fold provenance differs from results"
        )
    graph = candidate["candidate_graph"]
    lifecycle = graph["initializer_estimator_id"] != "none"
    if not lifecycle:
        point_keys = {
            "bundle_role",
            "calibrator_id",
            "candidate_graph",
            "candidate_hash",
            "candidate_id",
            "capability_contract_hash",
            "code_hash",
            "condition_id",
            "dataset_id",
            "dataset_schema_version",
            "eligibility_hash",
            "feature_schema_version",
            "feature_set_hash",
            "fold",
            "input_contract_hash",
            "interval_alpha",
            "position",
            "source_descriptor_hash",
            "split_plan_id",
            "target",
        }
        if provenance.get("source_descriptor") is not None:
            point_keys.add("source_descriptor")
        if set(provenance) != point_keys or provenance.get(
            "bundle_role"
        ) != "point_model":
            raise Stage4CompletionReleaseError(
                "Stage 4 completion point-model provenance keys differ"
            )
        for key in (
            "capability_contract_hash",
            "eligibility_hash",
            "input_contract_hash",
            "source_descriptor_hash",
        ):
            _sha256(
                provenance.get(key),
                description=f"Stage 4 completion provenance {key}",
            )
        for key in ("dataset_schema_version", "feature_schema_version"):
            _integer(
                provenance.get(key),
                description=f"Stage 4 completion provenance {key}",
                minimum=1,
            )
    fold_field = "outer_fold" if lifecycle else "fold"
    if provenance.get(fold_field) != fold:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion fold provenance index differs"
        )
    _sha256(
        provenance.get("input_contract_hash"),
        description="Stage 4 completion fold input contract",
    )
    descriptor_document = provenance.get("source_descriptor")
    if descriptor_document is not None:
        if not isinstance(descriptor_document, Mapping):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion provenance source descriptor is invalid"
            )
        try:
            descriptor = SourceDescriptor.from_dict(descriptor_document)
        except (TypeError, ValueError) as exc:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion provenance source descriptor is invalid"
            ) from exc
        if (
            descriptor.source_id != results["source"]["source_id"]
            or descriptor.descriptor_hash
            != results["source"]["source_descriptor_hash"]
            or descriptor.capabilities.contract_hash
            != results["source"]["capability_contract_hash"]
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion provenance source descriptor differs"
            )
    return provenance


def _load_declared_bundles(
    artifact_root: Path,
    results: Mapping[str, Any],
    *,
    manifest_files: Mapping[str, str] | None = None,
) -> int:
    count = 0
    expected_files = {"results.json"}
    for experiment in results["experiments"]:
        experiment_key = _artifact_key(
            experiment["artifact_key"],
            prefix="e",
            description="Stage 4 completion experiment artifact key",
        )
        for candidate in experiment["candidates"]:
            if not _requires_reloadable_bundle(candidate):
                continue
            candidate_key = _artifact_key(
                candidate["artifact_key"],
                prefix="c",
                description="Stage 4 completion candidate artifact key",
            )
            graph = candidate["candidate_graph"]
            lifecycle = graph["initializer_estimator_id"] != "none"
            for seed_result in candidate["seed_results"]:
                seed = int(seed_result["split_seed"])
                for fold_value in seed_result["reloadable_bundle_folds"]:
                    fold = int(fold_value)
                    fold_root = (
                        artifact_root
                        / "fold_artifacts"
                        / experiment_key
                        / candidate_key
                        / f"seed_{seed}"
                        / f"fold_{fold}"
                    )
                    provenance = _verify_fold_provenance(
                        fold_root,
                        results=results,
                        experiment=experiment,
                        candidate=candidate,
                        seed_result=seed_result,
                        fold=fold,
                    )
                    bundle = fold_root / "bundle"
                    fold_relative = (
                        f"fold_artifacts/{experiment_key}/{candidate_key}/"
                        f"seed_{seed}/fold_{fold}"
                    )
                    if lifecycle:
                        sidecars = {
                            "calibrator.json",
                            "provenance.json",
                        }
                    elif candidate["estimator_id"] == "independent_mlp":
                        sidecars = {
                            "calibrator.json",
                            "encoder.json",
                            "fit_report.json",
                            "provenance.json",
                        }
                    elif candidate["estimator_id"] == "lightgbm_quantile":
                        sidecars = {
                            "calibrator.json",
                            "encoder.json",
                            "feature_importance.jsonl",
                            "fit_report.json",
                            "provenance.json",
                            "q05.model.txt",
                            "q50.model.txt",
                            "q95.model.txt",
                        }
                    else:
                        raise Stage4CompletionReleaseError(
                            "Stage 4 completion bundle estimator is unsupported"
                        )
                    expected_files.update(
                        f"{fold_relative}/{name}" for name in sidecars
                    )
                    bundle_files = _bundle_mapping(bundle)
                    try:
                        if lifecycle:
                            loaded = load_lifecycle_bundle(bundle_files)
                            if dict(loaded.manifest) != dict(provenance):
                                raise Stage4CompletionReleaseError(
                                    "Stage 4 completion lifecycle provenance differs"
                                )
                        elif candidate["estimator_id"] == "independent_mlp":
                            loaded = load_neural_bundle(bundle)
                            if dict(loaded.provenance or {}) != dict(provenance):
                                raise Stage4CompletionReleaseError(
                                    "Stage 4 completion MLP provenance differs"
                                )
                        elif candidate["estimator_id"] == "lightgbm_quantile":
                            loaded = load_lightgbm_bundle(bundle)
                            root_encoder = _load_json(
                                fold_root / "encoder.json",
                                maximum_bytes=MAX_PROVENANCE_BYTES,
                                description=(
                                    "Stage 4 completion LightGBM fold encoder"
                                ),
                            )
                            root_fit_report = _load_json(
                                fold_root / "fit_report.json",
                                maximum_bytes=MAX_PROVENANCE_BYTES,
                                description=(
                                    "Stage 4 completion LightGBM fit report"
                                ),
                            )
                            if (
                                loaded.dataset_id != provenance["dataset_id"]
                                or loaded.position.value
                                != provenance["position"]
                                or loaded.target.value != provenance["target"]
                                or loaded.allowed_condition_ids
                                != (provenance["condition_id"],)
                                or dict(root_encoder)
                                != loaded.encoder.to_dict()
                                or root_fit_report.get(
                                    "encoder_schema_hash"
                                )
                                != loaded.encoder.schema.content_hash
                                or loaded.fit_report.encoder_schema_hash
                                != loaded.encoder.schema.content_hash
                            ):
                                raise Stage4CompletionReleaseError(
                                    "Stage 4 completion LightGBM loaded scope "
                                    "or encoder differs"
                                )
                        else:
                            raise Stage4CompletionReleaseError(
                                "Stage 4 completion bundle estimator is unsupported"
                            )
                    except Stage4CompletionReleaseError:
                        raise
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        raise Stage4CompletionReleaseError(
                            "Stage 4 completion bundle failed independent loading"
                        ) from exc
                    expected_files.update(
                        f"{fold_relative}/bundle/{name}"
                        for name in bundle_files
                    )
                    count += 1
    if manifest_files is not None:
        _require_exact_artifact_payloads(
            manifest_files,
            expected_files,
            description="Stage 4 completion artifact",
        )
    return count


def _semantic_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics are not canonical JSON data"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _training_run_semantic_sha256(
    metadata: Mapping[str, Any],
    *,
    expected_run_id: object,
    expected_results_payload_sha256: object,
    description: str,
) -> str:
    run_id = _text(expected_run_id, description=f"{description} run id")
    payload_sha256 = _sha256(
        expected_results_payload_sha256,
        description=f"{description} results payload",
    )
    semantic = metadata.get("run_semantic")
    if (
        set(metadata)
        != {"run_id", "run_semantic", "results_payload_sha256"}
        or metadata.get("run_id") != run_id
        or metadata.get("results_payload_sha256") != payload_sha256
        or not isinstance(semantic, Mapping)
    ):
        raise Stage4CompletionReleaseError(
            f"{description} manifest metadata differs"
        )
    semantic_sha256 = _semantic_sha256(dict(semantic))
    if run_id != semantic_sha256[:24]:
        raise Stage4CompletionReleaseError(
            f"{description} run semantic SHA-256 differs"
        )
    return semantic_sha256


def _diagnostic_identity(value: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        value.get("source_name"),
        value.get("condition_id"),
        value.get("experiment_id"),
        value.get("candidate_id"),
        value.get("candidate_hash"),
        value.get("split_seed"),
        value.get("split_plan_id"),
    )


def _inventory_identity(value: Mapping[str, Any]) -> tuple[object, ...]:
    return (*_diagnostic_identity(value), value.get("fold"))


def _verify_shared_development_task_projection(
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    cells: dict[tuple[object, object, object], list[Mapping[str, Any]]] = {}
    for record in diagnostics:
        key = (
            record.get("source_name"),
            record.get("condition_id"),
            record.get("experiment_id"),
        )
        cells.setdefault(key, []).append(record)
    for cell in cells.values():
        if (
            len(cell) != len(_SEED_POLICY_CANDIDATES) * len(STAGE_SPLIT_SEEDS)
            or {record.get("candidate_id") for record in cell}
            != set(_SEED_POLICY_CANDIDATES)
            or {record.get("split_seed") for record in cell}
            != set(STAGE_SPLIT_SEEDS)
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion development task identity cell topology differs"
            )
        parities = [
            record.get("checkpoint_parity")
            for record in cell
            if isinstance(record.get("checkpoint_parity"), Mapping)
        ]
        if len(parities) != len(cell):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion development task identity parity is invalid"
            )
        task_counts = {parity.get("development_task_count") for parity in parities}
        task_projections = {
            parity.get("development_task_projection_sha256") for parity in parities
        }
        if len(task_counts) != 1 or len(task_projections) != 1:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion split-independent development task identity "
                "projection differs within a semantic cell"
            )


def _seed_aggregate_projection(
    seed_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    metrics = seed_result.get("metrics")
    fold_metrics = seed_result.get("fold_metrics")
    task_metrics = seed_result.get("task_metrics")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(fold_metrics, Mapping)
        or not isinstance(task_metrics, list)
        or any(not isinstance(item, Mapping) for item in task_metrics)
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion finalized seed metrics are invalid"
        )
    return {
        "metrics": dict(metrics),
        "fold_metrics": dict(fold_metrics),
        "task_metric_policy_id": STAGE4_TASK_PSEUDONYM_POLICY_ID,
        "task_metrics": [dict(item) for item in task_metrics],
    }


def _checkpoint_expectation(
    root: Path,
    *,
    results: Mapping[str, Any],
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    seed_result: Mapping[str, Any],
    source_provenance_hash: str,
    source_run_semantic_sha256: str,
) -> Mapping[str, Any]:
    comparability = seed_result.get("comparability_key")
    split_seed = seed_result.get("split_seed")
    source_run_semantic_sha256 = _sha256(
        source_run_semantic_sha256,
        description="Stage 4 completion source run semantic",
    )
    if (
        not isinstance(comparability, list)
        or len(comparability) != 9
        or isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint comparability key is invalid"
        )
    execution_key = {
        "experiment_id": experiment["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "dataset_id": comparability[0],
        "split_plan_id": seed_result["split_plan_id"],
        "split_seed": split_seed,
        "eligibility_hash": comparability[2],
        "position": experiment["position"],
        "target": experiment["target"],
        "condition_id": experiment["condition_id"],
        "calibrator_id": experiment["calibrator_id"],
        "alpha": experiment["alpha"],
        "source_provenance_hash": source_provenance_hash,
    }
    expected_comparability = [
        execution_key["dataset_id"],
        execution_key["split_plan_id"],
        execution_key["eligibility_hash"],
        execution_key["position"],
        execution_key["target"],
        execution_key["condition_id"],
        execution_key["calibrator_id"],
        str(execution_key["alpha"]),
        METRIC_SUITE_ID,
    ]
    if comparability != expected_comparability:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint comparability key differs from "
            "the execution identity"
        )
    execution_hash = _semantic_sha256(execution_key)
    run_id = _text(
        results.get("run_id"),
        description="Stage 4 completion checkpoint run id",
    )
    checkpoint = _path(
        root,
        (
            f"workspace/stage4/checkpoints/{run_id}/candidates/"
            f"{execution_hash}"
        ),
        description="Stage 4 completion candidate checkpoint",
    )
    try:
        manifest = verify_artifact(checkpoint)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion candidate checkpoint verification failed"
        ) from exc
    expected_checkpoint_metadata = {
        "candidate_execution_hash": execution_hash,
        "run_id": run_id,
        "run_semantic_sha256": source_run_semantic_sha256,
    }
    if (
        manifest.stage_name != "development_candidate_checkpoint"
        or manifest.schema_version != 1
        or set(manifest.files) != {"candidate_result.json"}
        or dict(manifest.metadata) != expected_checkpoint_metadata
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion candidate checkpoint identity differs"
        )
    wrapper = _load_json(
        checkpoint / "candidate_result.json",
        maximum_bytes=MAX_CHECKPOINT_JSON_BYTES,
        description="Stage 4 completion candidate checkpoint",
    )
    wrapper = _exact(
        wrapper,
        {
            "checkpoint_schema_version",
            "execution_key",
            "result",
            "result_sha256",
        },
        description="Stage 4 completion candidate checkpoint",
    )
    if (
        wrapper["checkpoint_schema_version"] != 1
        or wrapper["execution_key"] != execution_key
        or not isinstance(wrapper["result"], Mapping)
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion candidate checkpoint schema differs"
        )
    result_document = dict(wrapper["result"])
    result_sha256 = _sha256(
        wrapper["result_sha256"],
        description="Stage 4 completion checkpoint result",
    )
    if _semantic_sha256(result_document) != result_sha256:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint result checksum differs"
        )
    try:
        checkpoint_result = _result_from_dict(result_document)
    except Exception as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint result cannot be reconstructed"
        ) from exc
    expected_identity = {
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "dataset_id": comparability[0],
        "split_plan_id": seed_result["split_plan_id"],
        "eligibility_hash": comparability[2],
        "position": experiment["position"],
        "target": experiment["target"],
        "condition_id": experiment["condition_id"],
        "calibrator_id": experiment["calibrator_id"],
        "alpha": experiment["alpha"],
    }
    observed_identity = {
        "candidate_id": checkpoint_result.candidate_id,
        "candidate_hash": checkpoint_result.candidate_hash,
        "dataset_id": checkpoint_result.dataset_id,
        "split_plan_id": checkpoint_result.split_plan_id,
        "eligibility_hash": checkpoint_result.eligibility_hash,
        "position": checkpoint_result.position.value,
        "target": checkpoint_result.target.value,
        "condition_id": checkpoint_result.condition_id,
        "calibrator_id": checkpoint_result.calibrator_id,
        "alpha": checkpoint_result.alpha,
    }
    if observed_identity != expected_identity:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint result scope differs"
        )
    prediction_count = len(checkpoint_result.predictions)
    prediction_projection = prediction_projection_sha256(checkpoint_result)
    cohort_projection = cohort_projection_sha256(checkpoint_result)
    expected_count = _integer(
        seed_result.get("prediction_count"),
        description="Stage 4 completion finalized prediction count",
        minimum=1,
    )
    expected_prediction = _sha256(
        seed_result.get("prediction_projection_sha256"),
        description="Stage 4 completion finalized prediction projection",
    )
    expected_cohort = _sha256(
        seed_result.get("cohort_projection_sha256"),
        description="Stage 4 completion finalized cohort projection",
    )
    checkpoint_task_metrics = sorted(
        (
            {
                "task_pseudonym": hashlib.sha256(
                    (
                        f"{STAGE4_TASK_PSEUDONYM_POLICY_ID}\0"
                        f"{checkpoint_result.split_plan_id}\0{task_id}"
                    ).encode()
                ).hexdigest(),
                **dict(metrics),
            }
            for task_id, metrics in checkpoint_result.task_metrics.items()
        ),
        key=lambda item: str(item["task_pseudonym"]),
    )
    checkpoint_aggregate = {
        "metrics": dict(checkpoint_result.metrics),
        "fold_metrics": {
            str(fold): dict(metrics)
            for fold, metrics in checkpoint_result.fold_metrics.items()
        },
        "task_metric_policy_id": STAGE4_TASK_PSEUDONYM_POLICY_ID,
        "task_metrics": checkpoint_task_metrics,
    }
    expected_aggregate = _seed_aggregate_projection(seed_result)
    aggregate_projection = _semantic_sha256(checkpoint_aggregate)
    expected_aggregate_projection = _semantic_sha256(expected_aggregate)
    if (
        prediction_count != expected_count
        or prediction_projection != expected_prediction
        or cohort_projection != expected_cohort
        or aggregate_projection != expected_aggregate_projection
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint differs from its finalized seed"
        )
    protocol = results.get("development_protocol")
    if not isinstance(protocol, Mapping):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion development protocol is invalid"
        )
    permanent_holdout = protocol.get("permanent_holdout")
    if not isinstance(permanent_holdout, Mapping) or not isinstance(
        permanent_holdout.get("assignments"),
        list,
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion permanent holdout assignments are invalid"
        )
    assignments: dict[str, str] = {}
    for raw in permanent_holdout["assignments"]:
        assignment = _exact(
            raw,
            {"task_pseudonym", "cohort"},
            description="Stage 4 completion holdout assignment",
        )
        task_pseudonym = _sha256(
            assignment["task_pseudonym"],
            description="Stage 4 completion holdout task pseudonym",
        )
        cohort = _text(
            assignment["cohort"],
            description="Stage 4 completion holdout cohort",
        )
        if task_pseudonym in assignments or cohort not in {
            "development",
            "final_holdout",
        }:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion holdout assignment differs"
            )
        assignments[task_pseudonym] = cohort
    task_pseudonyms = sorted(
        {
            hashlib.sha256(
                (
                    f"{DEVELOPMENT_TASK_PSEUDONYM_POLICY_ID}\0"
                    f"{record.task_id}"
                ).encode()
            ).hexdigest()
            for record in checkpoint_result.predictions
        }
    )
    if not task_pseudonyms or any(
        assignments.get(task_pseudonym) != "development"
        for task_pseudonym in task_pseudonyms
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion checkpoint contains final or unassigned tasks"
        )
    development_projection = _semantic_sha256(
        [
            {
                "task_pseudonym": task_pseudonym,
                "cohort": assignments[task_pseudonym],
            }
            for task_pseudonym in task_pseudonyms
        ]
    )
    return {
        "status": "exact",
        "checkpoint_artifact_id": manifest.artifact_id,
        "checkpoint_result_sha256": result_sha256,
        "prediction_count": prediction_count,
        "expected_prediction_count": expected_count,
        "prediction_projection_sha256": prediction_projection,
        "expected_prediction_projection_sha256": expected_prediction,
        "cohort_projection_sha256": cohort_projection,
        "expected_cohort_projection_sha256": expected_cohort,
        "aggregate_metrics_projection_sha256": aggregate_projection,
        "expected_aggregate_metrics_projection_sha256": (
            expected_aggregate_projection
        ),
        "development_cohort_status": "development_only",
        "development_task_count": len(task_pseudonyms),
        "development_task_projection_sha256": development_projection,
    }


def _expected_diagnostics_scope(
    root: Path,
    release: Mapping[str, Any],
    training_results: Mapping[str, Mapping[str, Any]],
    training_run_semantic_sha256_by_source: Mapping[str, str],
) -> tuple[
    dict[tuple[object, ...], Mapping[str, Any]],
    dict[tuple[object, ...], Mapping[str, Any]],
]:
    expected_diagnostics: dict[tuple[object, ...], Mapping[str, Any]] = {}
    expected_inventory: dict[tuple[object, ...], Mapping[str, Any]] = {}
    locks = {
        str(item["source_name"]): item for item in release["artifacts"]
    }
    for source_name, results in training_results.items():
        if source_name == "spend_aggregate":
            continue
        source_run_semantic_sha256 = _sha256(
            training_run_semantic_sha256_by_source.get(source_name),
            description=(
                f"Stage 4 completion {source_name} source run semantic"
            ),
        )
        artifact_root = _path(
            root,
            locks[source_name]["path"],
            description=f"Stage 4 completion {source_name} artifact",
        )
        for experiment in results["experiments"]:
            experiment_key = str(experiment["artifact_key"])
            for candidate in experiment["candidates"]:
                if candidate["candidate_id"] not in _SEED_POLICY_CANDIDATES:
                    continue
                candidate_key = str(candidate["artifact_key"])
                for seed_result in candidate["seed_results"]:
                    base = {
                        "source_name": source_name,
                        "condition_id": experiment["condition_id"],
                        "experiment_id": experiment["experiment_id"],
                        "candidate_id": candidate["candidate_id"],
                        "candidate_hash": candidate["candidate_hash"],
                        "split_seed": seed_result["split_seed"],
                        "split_plan_id": seed_result["split_plan_id"],
                    }
                    identity = _diagnostic_identity(base)
                    if identity in expected_diagnostics:
                        raise Stage4CompletionReleaseError(
                            "Stage 4 completion diagnostic identity collided"
                        )
                    source_provenance_hashes: set[str] = set()
                    for fold in range(EXPECTED_OUTER_FOLDS):
                        bundle_relative = (
                            f"fold_artifacts/{experiment_key}/{candidate_key}/"
                            f"seed_{seed_result['split_seed']}/fold_{fold}/bundle"
                        )
                        bundle = artifact_root / Path(bundle_relative)
                        files = _bundle_mapping(bundle)
                        manifest_payload = files.get("manifest.json")
                        if manifest_payload is None:
                            raise Stage4CompletionReleaseError(
                                "Stage 4 completion lifecycle bundle lacks manifest"
                            )
                        try:
                            loaded_bundle = load_lifecycle_bundle(bundle)
                        except Exception as exc:
                            raise Stage4CompletionReleaseError(
                                "Stage 4 completion lifecycle bundle failed "
                                "independent safe loading"
                            ) from exc
                        bundle_manifest = loaded_bundle.manifest
                        source_provenance_hashes.add(
                            _semantic_sha256(
                                {
                                    "source_descriptor": bundle_manifest.get(
                                        "source_descriptor"
                                    ),
                                    "source_descriptor_hash": (
                                        bundle_manifest.get(
                                            "source_descriptor_hash"
                                        )
                                    ),
                                    "code_hash": bundle_manifest.get(
                                        "code_hash"
                                    ),
                                    "runtime_versions": results.get(
                                        "runtime_versions"
                                    ),
                                }
                            )
                        )
                        inventory = {
                            **base,
                            "fold": fold,
                            "bundle_relative_path": bundle_relative,
                            "bundle_manifest_sha256": hashlib.sha256(
                                manifest_payload
                            ).hexdigest(),
                            "bundle_file_count": len(files),
                            "load_status": "safe_loaded",
                        }
                        inventory_identity = _inventory_identity(inventory)
                        if inventory_identity in expected_inventory:
                            raise Stage4CompletionReleaseError(
                                "Stage 4 completion bundle inventory collided"
                            )
                        expected_inventory[inventory_identity] = inventory
                    if len(source_provenance_hashes) != 1:
                        raise Stage4CompletionReleaseError(
                            "Stage 4 completion lifecycle bundles disagree on "
                            "source provenance"
                        )
                    expected_diagnostics[identity] = {
                        **base,
                        "_checkpoint_parity": _checkpoint_expectation(
                            root,
                            results=results,
                            experiment=experiment,
                            candidate=candidate,
                            seed_result=seed_result,
                            source_provenance_hash=next(
                                iter(source_provenance_hashes)
                            ),
                            source_run_semantic_sha256=(
                                source_run_semantic_sha256
                            ),
                        ),
                    }
    if (
        len(expected_diagnostics) != EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
        or len(expected_inventory) != EXPECTED_DIAGNOSTICS_BUNDLE_COUNT
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion training artifacts lack diagnostic coverage"
        )
    return expected_diagnostics, expected_inventory


def _verify_progress_diagnostics(value: object) -> int:
    progress = _exact(
        value,
        {"stratification_id", "selection_policy", "strata"},
        description="Stage 4 completion progress diagnostics",
    )
    if (
        progress["stratification_id"] != PROGRESS_STRATIFICATION_ID
        or progress["selection_policy"]
        != "first_boundary_at_or_after_sequence_fraction_v1"
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion progress policy differs"
        )
    strata = _exact(
        progress["strata"],
        {"p25", "p50", "p75"},
        description="Stage 4 completion progress strata",
    )
    sequence_counts: set[int] = set()
    for key, checkpoint in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
        stratum = _exact(
            strata[key],
            {
                "checkpoint",
                "n_sequences",
                "n_selected_boundaries",
                "n_scored",
                "n_unscored",
                "metrics",
            },
            description=f"Stage 4 completion progress {key}",
        )
        if stratum["checkpoint"] != checkpoint:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion progress checkpoint differs"
            )
        count = _integer(
            stratum["n_sequences"],
            description=f"Stage 4 completion progress {key} sequence count",
            minimum=1,
        )
        sequence_counts.add(count)
        selected = _integer(
            stratum["n_selected_boundaries"],
            description=f"Stage 4 completion progress {key} selected count",
        )
        scored = _integer(
            stratum["n_scored"],
            description=f"Stage 4 completion progress {key} scored count",
        )
        unscored = _integer(
            stratum["n_unscored"],
            description=f"Stage 4 completion progress {key} unscored count",
        )
        if selected > count or scored > selected or scored + unscored != count:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion progress counts do not close"
            )
        if stratum["metrics"] is not None:
            _reject_non_finite(
                stratum["metrics"],
                description=f"Stage 4 completion progress {key} metrics",
            )
    if len(sequence_counts) != 1:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion progress sequence counts differ"
        )
    return next(iter(sequence_counts))


def _verify_termination_diagnostics(value: object) -> tuple[int, int, int]:
    termination = _exact(
        value,
        {"stratification_id", "strata"},
        description="Stage 4 completion termination diagnostics",
    )
    if termination["stratification_id"] != TERMINATION_STRATIFICATION_ID:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion termination policy differs"
        )
    strata = termination["strata"]
    if not isinstance(strata, Mapping) or not strata:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion termination strata are empty"
        )
    context_only = 0
    scored_boundaries = 0
    sequence_count = 0
    for name, raw in strata.items():
        _text(name, description="Stage 4 completion termination stratum")
        stratum = _exact(
            raw,
            {
                "n_sequences",
                "n_tasks",
                "n_update_boundaries",
                "n_scored",
                "n_context_only",
                "metrics",
            },
            description="Stage 4 completion termination stratum",
        )
        for key in (
            "n_sequences",
            "n_tasks",
            "n_update_boundaries",
            "n_scored",
            "n_context_only",
        ):
            _integer(
                stratum[key],
                description=f"Stage 4 completion termination {key}",
            )
        context_only += int(stratum["n_context_only"])
        scored_boundaries += int(stratum["n_scored"])
        sequence_count += int(stratum["n_sequences"])
        if stratum["metrics"] is not None:
            _reject_non_finite(
                stratum["metrics"],
                description="Stage 4 completion termination metrics",
            )
    return context_only, scored_boundaries, sequence_count


def _verify_run_dispersion(value: object) -> int:
    required = {
        "run_variance_id",
        "run_dispersion_extension_id",
        "n_tasks",
        "n_scored_runs",
        "n_repeated_tasks",
        "status",
        "mean_within_task_run_mae_variance",
        "median_within_task_run_mae_variance",
        "max_within_task_run_mae_variance",
        "mean_within_task_run_mae_iqr",
        "median_within_task_run_mae_iqr",
        "max_within_task_run_mae_iqr",
        "mean_within_task_run_mae_max_minus_min",
        "median_within_task_run_mae_max_minus_min",
        "max_within_task_run_mae_max_minus_min",
    }
    dispersion = _exact(
        value,
        required,
        description="Stage 4 completion run dispersion",
    )
    if (
        dispersion["run_variance_id"] != RUN_VARIANCE_ID
        or dispersion["run_dispersion_extension_id"]
        != RUN_DISPERSION_EXTENSION_ID
        or dispersion["status"] not in {"estimable", "not_estimable"}
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion run dispersion policy differs"
        )
    for key in ("n_tasks", "n_scored_runs", "n_repeated_tasks"):
        _integer(
            dispersion[key],
            description=f"Stage 4 completion run dispersion {key}",
        )
    for key in required - {
        "run_variance_id",
        "run_dispersion_extension_id",
        "n_tasks",
        "n_scored_runs",
        "n_repeated_tasks",
        "status",
    }:
        number = dispersion[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0
        ):
            raise Stage4CompletionReleaseError(
                f"Stage 4 completion run dispersion {key} is invalid"
            )
    return int(dispersion["n_scored_runs"])


def _verify_diagnostics_artifact(
    root: Path,
    release: Mapping[str, Any],
    training_results: Mapping[str, Mapping[str, Any]],
    training_run_semantic_sha256_by_source: Mapping[str, str],
) -> tuple[int, int]:
    locked = release["diagnostics_artifact"]
    artifact_root = _path(
        root,
        locked["path"],
        description="Stage 4 completion diagnostics artifact",
    )
    try:
        manifest = verify_artifact(artifact_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics artifact verification failed"
        ) from exc
    manifest_payload = _regular_file(
        artifact_root / "manifest.json",
        maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        description="Stage 4 completion diagnostics manifest",
    )
    if (
        manifest.artifact_id != locked["artifact_id"]
        or manifest.stage_name != DIAGNOSTICS_STAGE_NAME
        or manifest.schema_version != DIAGNOSTICS_RESULTS_SCHEMA_VERSION
        or set(manifest.files) != {"results.json"}
        or hashlib.sha256(manifest_payload).hexdigest()
        != locked["manifest_sha256"]
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics manifest binding or topology differs"
        )
    results = _load_json(
        artifact_root / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description="Stage 4 completion diagnostics results",
    )
    _exact(
        results,
        {
            "results_schema_version",
            "stage_name",
            "policy_id",
            "source_binding",
            "diagnostics_code_binding",
            "source_artifacts",
            "coverage",
            "bundle_inventory",
            "diagnostics",
            "final_holdout",
            "results_payload_sha256",
        },
        description="Stage 4 completion diagnostics results",
    )
    if (
        results["results_schema_version"] != DIAGNOSTICS_RESULTS_SCHEMA_VERSION
        or results["stage_name"] != DIAGNOSTICS_STAGE_NAME
        or results["policy_id"] != DIAGNOSTICS_POLICY_ID
        or results["final_holdout"] != _FINAL_HOLDOUT_SENTINEL
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics identity or final sentinel differs"
        )
    declared = _sha256(
        results["results_payload_sha256"],
        description="Stage 4 completion diagnostics results payload",
    )
    semantic = dict(results)
    semantic.pop("results_payload_sha256")
    if (
        _semantic_sha256(semantic) != declared
        or declared != locked["results_payload_sha256"]
        or manifest.metadata.get("results_payload_sha256") != declared
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics payload binding differs"
        )
    source_binding = _exact(
        results["source_binding"],
        {"git_commit", "code_tree_sha256"},
        description="Stage 4 completion diagnostics source binding",
    )
    if (
        source_binding["git_commit"] != locked["training_source_commit"]
        or source_binding["git_commit"]
        != release["source_binding"]["git_commit"]
        or source_binding["code_tree_sha256"]
        != release["source_binding"]["code_tree_sha256"]
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics source binding differs"
        )
    result_diagnostics_code = _exact(
        results["diagnostics_code_binding"],
        {"git_commit", "code_tree_sha256", "code_paths"},
        description="Stage 4 completion diagnostics executed code binding",
    )
    locked_diagnostics_code = locked["diagnostics_code_binding"]
    if dict(result_diagnostics_code) != {
        key: locked_diagnostics_code[key]
        for key in ("git_commit", "code_tree_sha256", "code_paths")
    }:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics executed code binding differs"
        )

    source_documents = results["source_artifacts"]
    if not isinstance(source_documents, list) or len(source_documents) != 4:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics must bind all four source artifacts"
        )
    locks = {
        str(item["source_name"]): item for item in release["artifacts"]
    }
    expected_source_order = sorted(locks)
    observed_source_order: list[str] = []
    for document in source_documents:
        source = _exact(
            document,
            {
                "source_name",
                "source_id",
                "run_id",
                "artifact_id",
                "results_payload_sha256",
                "matrix_id",
                "development_protocol_id",
                "lifecycle_status",
            },
            description="Stage 4 completion diagnostic source artifact",
        )
        source_name = str(source["source_name"])
        observed_source_order.append(source_name)
        training = training_results.get(source_name)
        lock = locks.get(source_name)
        if training is None or lock is None:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostics bind an unknown source"
            )
        expected_status = (
            "not_applicable_no_lifecycle"
            if source_name == "spend_aggregate"
            else "unavailable_no_presealed_replay_projection"
        )
        expected = {
            "source_name": source_name,
            "source_id": lock["source_id"],
            "run_id": lock["run_id"],
            "artifact_id": lock["artifact_id"],
            "results_payload_sha256": lock["results_payload_sha256"],
            "matrix_id": lock["matrix_id"],
            "development_protocol_id": training["development_protocol"][
                "protocol_id"
            ],
            "lifecycle_status": expected_status,
        }
        if dict(source) != expected:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic source binding differs"
            )
    if observed_source_order != expected_source_order:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostic sources are not canonical"
        )

    coverage = _exact(
        results["coverage"],
        set(locked["coverage"]),
        description="Stage 4 completion diagnostics result coverage",
    )
    if dict(coverage) != dict(locked["coverage"]):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostic coverage differs from its lock"
        )
    expected_diagnostics, expected_inventory = _expected_diagnostics_scope(
        root,
        release,
        training_results,
        training_run_semantic_sha256_by_source,
    )

    inventory = results["bundle_inventory"]
    if (
        not isinstance(inventory, list)
        or len(inventory) != EXPECTED_DIAGNOSTICS_BUNDLE_COUNT
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics bundle inventory count differs"
        )
    observed_inventory: dict[tuple[object, ...], Mapping[str, Any]] = {}
    for item in inventory:
        record = _exact(
            item,
            {
                "source_name",
                "condition_id",
                "experiment_id",
                "candidate_id",
                "candidate_hash",
                "split_seed",
                "split_plan_id",
                "fold",
                "bundle_relative_path",
                "bundle_manifest_sha256",
                "bundle_file_count",
                "load_status",
            },
            description="Stage 4 completion diagnostics bundle inventory",
        )
        identity = _inventory_identity(record)
        if identity in observed_inventory:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic bundle inventory is duplicated"
            )
        expected = expected_inventory.get(identity)
        if expected is None or dict(record) != dict(expected):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic bundle inventory differs"
            )
        observed_inventory[identity] = record
    if set(observed_inventory) != set(expected_inventory):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostic bundle inventory is incomplete"
        )

    diagnostics = results["diagnostics"]
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != EXPECTED_DIAGNOSTICS_CANDIDATE_SEED_COUNT
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostic record count differs"
        )
    observed_diagnostics: set[tuple[object, ...]] = set()
    checkpoint_verified = 0
    lifecycle_unavailable = 0
    for item in diagnostics:
        record = _exact(
            item,
            _DIAGNOSTIC_RECORD_KEYS,
            description="Stage 4 completion diagnostic record",
        )
        identity = _diagnostic_identity(record)
        if identity in observed_diagnostics or identity not in expected_diagnostics:
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic identity differs or is duplicated"
            )
        observed_diagnostics.add(identity)
        if record["bundle_folds"] != list(range(EXPECTED_OUTER_FOLDS)):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic bundle folds differ"
            )
        projection_items = [
            {
                "fold": fold,
                "bundle_manifest_sha256": expected_inventory[
                    (*identity, fold)
                ]["bundle_manifest_sha256"],
                "bundle_file_count": expected_inventory[
                    (*identity, fold)
                ]["bundle_file_count"],
            }
            for fold in range(EXPECTED_OUTER_FOLDS)
        ]
        if record["bundle_projection_sha256"] != _semantic_sha256(
            projection_items
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion diagnostic bundle projection differs"
            )
        parity = _exact(
            record["checkpoint_parity"],
            _CHECKPOINT_PARITY_KEYS,
            description="Stage 4 completion checkpoint parity",
        )
        expected_parity = expected_diagnostics[identity][
            "_checkpoint_parity"
        ]
        for key in _CHECKPOINT_PARITY_KEYS - {
            "status",
            "development_cohort_status",
            "prediction_count",
            "expected_prediction_count",
            "development_task_count",
        }:
            _sha256(
                parity[key],
                description=f"Stage 4 completion checkpoint {key}",
            )
        if (
            dict(parity) != dict(expected_parity)
            or parity["status"] != "exact"
            or parity["development_cohort_status"] != "development_only"
            or _integer(
                parity["prediction_count"],
                description="Stage 4 completion checkpoint prediction count",
                minimum=1,
            )
            != _integer(
                parity["expected_prediction_count"],
                description=(
                    "Stage 4 completion expected checkpoint prediction count"
                ),
                minimum=1,
            )
            or _integer(
                parity["development_task_count"],
                description="Stage 4 completion development task count",
                minimum=1,
            )
            <= 0
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion checkpoint parity differs from the "
                "training seed and independently reopened checkpoint"
            )
        lifecycle = _exact(
            record["lifecycle_metrics"],
            _LIFECYCLE_METRICS_KEYS,
            description="Stage 4 completion lifecycle metric availability",
        )
        if (
            lifecycle["status"] != "unavailable"
            or lifecycle["reason_code"]
            != DIAGNOSTICS_LIFECYCLE_UNAVAILABLE_REASON
            or lifecycle["labels_present"] is not False
            or lifecycle["lifecycle_sequences_present"] is not False
            or lifecycle["unavailable_metrics"]
            != DIAGNOSTICS_UNAVAILABLE_LIFECYCLE_METRICS
            or lifecycle["historical_stage3_reference"] is not None
        ):
            raise Stage4CompletionReleaseError(
                "Stage 4 completion lifecycle-unavailable declaration differs"
            )
        checkpoint_verified += 1
        lifecycle_unavailable += 1
    if set(expected_diagnostics) != observed_diagnostics:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics are incomplete"
        )
    _verify_shared_development_task_projection(diagnostics)
    if (
        coverage["checkpoint_verified_candidate_seed_count"]
        != checkpoint_verified
        or coverage["lifecycle_replayed_candidate_seed_count"] != 0
        or coverage[
            "lifecycle_metrics_unavailable_candidate_seed_count"
        ]
        != lifecycle_unavailable
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostic dynamic coverage does not close"
        )
    expected_metadata_keys = {
        "run_id",
        "run_semantic",
        "results_payload_sha256",
        "source_git_commit",
        "source_code_tree_sha256",
        "diagnostics_code_binding",
        "source_artifact_ids",
        "coverage",
        "diagnostics_runner_sha256",
    }
    if set(manifest.metadata) != expected_metadata_keys:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics manifest metadata keys differ"
        )
    run_id = _text(
        manifest.metadata["run_id"],
        description="Stage 4 completion diagnostics run id",
    )
    relative = str(locked["path"])
    if (
        len(run_id) < 20
        or any(character not in "0123456789abcdef" for character in run_id)
        or relative.rsplit("/", 1)[-1] != f"s4diag-{run_id[:20]}"
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics path and run id differ"
        )
    runner_relative = DIAGNOSTICS_RUNNER_RELATIVE
    runner_payload = _git(
        root,
        "show",
        (
            f"{locked_diagnostics_code['git_commit']}:"
            f"{runner_relative}"
        ),
        maximum_bytes=MAX_BUNDLE_FILE_BYTES,
    )
    runner_sha256 = hashlib.sha256(runner_payload).hexdigest()
    compact_sources = [
        {
            "source_name": item["source_name"],
            "artifact_id": item["artifact_id"],
            "results_payload_sha256": item["results_payload_sha256"],
        }
        for item in source_documents
    ]
    expected_run_semantic = {
        "results_schema_version": DIAGNOSTICS_RESULTS_SCHEMA_VERSION,
        "policy_id": DIAGNOSTICS_POLICY_ID,
        "source_binding": dict(source_binding),
        "diagnostics_code_binding": dict(result_diagnostics_code),
        "source_artifacts": compact_sources,
        "diagnostics_runner_sha256": runner_sha256,
        "final_holdout": dict(_FINAL_HOLDOUT_SENTINEL),
    }
    metadata_expected = {
        "run_id": run_id,
        "run_semantic": expected_run_semantic,
        "results_payload_sha256": declared,
        "source_git_commit": locked["training_source_commit"],
        "source_code_tree_sha256": source_binding["code_tree_sha256"],
        "diagnostics_code_binding": dict(result_diagnostics_code),
        "source_artifact_ids": [
            item["artifact_id"] for item in source_documents
        ],
        "coverage": dict(locked["coverage"]),
        "diagnostics_runner_sha256": runner_sha256,
    }
    if dict(manifest.metadata) != metadata_expected:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics manifest metadata differs"
        )
    return len(diagnostics), len(inventory)


def _verify_artifact(
    root: Path,
    release: Mapping[str, Any],
    artifact_lock: Mapping[str, Any],
    code_binding: Mapping[str, object],
) -> tuple[_Coverage, int, Mapping[str, Any], str]:
    source_name = str(artifact_lock["source_name"])
    artifact_root = _path(
        root,
        artifact_lock["path"],
        description=f"Stage 4 completion {source_name} artifact",
    )
    try:
        manifest = verify_artifact(artifact_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} artifact verification failed"
        ) from exc
    manifest_payload = _regular_file(
        artifact_root / "manifest.json",
        maximum_bytes=MAX_BUNDLE_FILE_BYTES,
        description=f"Stage 4 completion {source_name} manifest",
    )
    if (
        manifest.artifact_id != artifact_lock["artifact_id"]
        or manifest.stage_name != STAGE4_STAGE_NAME
        or manifest.schema_version != STAGE4_ARTIFACT_SCHEMA_VERSION
        or len(manifest.files) != artifact_lock["manifest_file_count"]
        or hashlib.sha256(manifest_payload).hexdigest()
        != artifact_lock["manifest_sha256"]
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} manifest binding differs"
        )
    results = _load_json(
        artifact_root / "results.json",
        maximum_bytes=MAX_RESULTS_JSON_BYTES,
        description=f"Stage 4 completion {source_name} results",
    )
    try:
        payload_hash = verify_stage4_results_document(results)
    except (Stage4ExperimentError, TypeError, ValueError, RuntimeError) as exc:
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} results schema is invalid"
        ) from exc
    if (
        payload_hash != artifact_lock["results_payload_sha256"]
        or manifest.metadata.get("results_payload_sha256") != payload_hash
        or results.get("run_id") != artifact_lock["run_id"]
        or results.get("final_holdout") != _FINAL_HOLDOUT_SENTINEL
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} result binding differs"
        )
    source = results.get("source")
    result_code = results.get("code_binding")
    if (
        not isinstance(source, Mapping)
        or source.get("source_name") != source_name
        or source.get("source_id") != artifact_lock["source_id"]
        or not isinstance(result_code, Mapping)
        or result_code.get("git_commit") != code_binding["git_commit"]
        or result_code.get("code_tree_sha256")
        != code_binding["code_tree_sha256"]
        or result_code.get("code_paths") != code_binding["code_paths"]
        or results.get("matrix", {}).get("matrix_id")
        != artifact_lock["matrix_id"]
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} source/code/matrix differs"
        )
    semantic = manifest.metadata.get("run_semantic")
    run_semantic_sha256 = _training_run_semantic_sha256(
        manifest.metadata,
        expected_run_id=artifact_lock["run_id"],
        expected_results_payload_sha256=payload_hash,
        description=f"Stage 4 completion {source_name}",
    )
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("source_name") != source_name
        or semantic.get("source_id") != artifact_lock["source_id"]
        or semantic.get("matrix_id") != artifact_lock["matrix_id"]
        or semantic.get("git_commit") != code_binding["git_commit"]
        or semantic.get("code_tree_sha256") != code_binding["code_tree_sha256"]
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} manifest semantic differs"
        )
    summary = results.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("experiment_count") != artifact_lock["experiment_count"]
        or summary.get("candidate_seed_run_count")
        != artifact_lock["candidate_seed_run_count"]
        or summary.get("split_seeds") != list(STAGE_SPLIT_SEEDS)
        or summary.get("outer_folds") != EXPECTED_OUTER_FOLDS
        or summary.get("inner_folds") != EXPECTED_INNER_FOLDS
    ):
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} summary differs"
        )
    coverage = _verify_result_coverage(results, source_name=source_name)
    loaded = _load_declared_bundles(
        artifact_root,
        results,
        manifest_files=manifest.files,
    )
    if loaded != coverage.reloadable_bundle_fold_count:
        raise Stage4CompletionReleaseError(
            f"Stage 4 completion {source_name} independently loaded count differs"
        )
    return coverage, loaded, results, run_semantic_sha256


def verify_stage4_completion_release(
    repository_root: str | Path,
    *,
    release_lock: str = DEFAULT_RELEASE_LOCK,
    tracked_only: bool = False,
    require_git_clean: bool = True,
) -> Stage4CompletionReleaseVerification:
    supplied_root = Path(repository_root)
    if _is_link_or_reparse(supplied_root):
        raise Stage4CompletionReleaseError(
            "repository root must not be linked or reparse-backed"
        )
    try:
        root = supplied_root.resolve(strict=True)
    except OSError as exc:
        raise Stage4CompletionReleaseError("repository root is missing") from exc
    if not root.is_dir():
        raise Stage4CompletionReleaseError(
            "repository root is not a directory"
        )
    release_relative = _relative(
        release_lock,
        description="Stage 4 completion release lock path",
    )
    release = _load_json(
        _path(
            root,
            release_relative,
            description="Stage 4 completion release lock",
        ),
        maximum_bytes=MAX_RELEASE_JSON_BYTES,
        description="Stage 4 completion release lock",
    )
    _validate_release_document(release)
    code = _verify_tracked_bindings(
        root,
        release,
        release_relative=release_relative,
        require_git_clean=require_git_clean,
        require_release_control_tag=tracked_only,
    )
    if tracked_only:
        return Stage4CompletionReleaseVerification(
            lock_path=release_relative,
            report_path=str(release["report"]["path"]),
            source_commit=str(code["git_commit"]),
            code_tree_sha256=str(code["code_tree_sha256"]),
            verified_artifact_count=0,
            verified_experiment_count=0,
            verified_candidate_seed_run_count=0,
            independently_loaded_bundle_count=0,
            verified_diagnostics_artifact_count=0,
            verified_diagnostics_record_count=0,
            verified_diagnostics_bundle_count=0,
            call_pre_mlp_cell_count=0,
            call_pre_mlp_bundle_count=0,
            seed_policy_cell_count=0,
            seed_policy_bundle_count=0,
            parent_final_holdout_evaluation_count=(
                EXPECTED_FINAL_EVALUATION_COUNT
            ),
            parent_final_holdout_prediction_count=(
                EXPECTED_FINAL_PREDICTION_COUNT
            ),
            final_source_opened=False,
            final_labels_read=False,
        )

    coverage = _Coverage()
    loaded_count = 0
    training_results: dict[str, Mapping[str, Any]] = {}
    training_run_semantic_sha256_by_source: dict[str, str] = {}
    for artifact in release["artifacts"]:
        (
            artifact_coverage,
            loaded,
            results,
            run_semantic_sha256,
        ) = _verify_artifact(
            root,
            release,
            artifact,
            code,
        )
        coverage = coverage.plus(artifact_coverage)
        loaded_count += loaded
        source_name = str(artifact["source_name"])
        training_results[source_name] = results
        training_run_semantic_sha256_by_source[source_name] = (
            run_semantic_sha256
        )
    protocol = release["protocol"]
    expected_totals = {
        "experiment_count": protocol["experiment_count"],
        "candidate_seed_run_count": protocol["candidate_seed_run_count"],
        "reloadable_bundle_fold_count": protocol[
            "reloadable_bundle_fold_count"
        ],
        "call_pre_mlp_cell_count": protocol["call_pre_mlp_cell_count"],
        "call_pre_mlp_bundle_fold_count": protocol[
            "call_pre_mlp_bundle_fold_count"
        ],
        "seed_policy_cell_count": protocol["seed_policy_cell_count"],
        "seed_policy_bundle_fold_count": protocol[
            "seed_policy_bundle_fold_count"
        ],
    }
    if any(
        getattr(coverage, key) != value
        for key, value in expected_totals.items()
    ) or loaded_count != protocol["reloadable_bundle_fold_count"]:
        raise Stage4CompletionReleaseError(
            "Stage 4 completion independently recomputed totals differ"
        )
    diagnostic_records, diagnostic_bundles = _verify_diagnostics_artifact(
        root,
        release,
        training_results,
        training_run_semantic_sha256_by_source,
    )
    if (
        diagnostic_records != protocol["diagnostics_candidate_seed_count"]
        or diagnostic_bundles != protocol["diagnostics_bundle_count"]
    ):
        raise Stage4CompletionReleaseError(
            "Stage 4 completion diagnostics totals differ from protocol"
        )
    _verify_report_semantics(root, release)
    return Stage4CompletionReleaseVerification(
        lock_path=release_relative,
        report_path=str(release["report"]["path"]),
        source_commit=str(code["git_commit"]),
        code_tree_sha256=str(code["code_tree_sha256"]),
        verified_artifact_count=EXPECTED_ARTIFACT_COUNT,
        verified_experiment_count=coverage.experiment_count,
        verified_candidate_seed_run_count=coverage.candidate_seed_run_count,
        independently_loaded_bundle_count=loaded_count,
        verified_diagnostics_artifact_count=1,
        verified_diagnostics_record_count=diagnostic_records,
        verified_diagnostics_bundle_count=diagnostic_bundles,
        call_pre_mlp_cell_count=coverage.call_pre_mlp_cell_count,
        call_pre_mlp_bundle_count=coverage.call_pre_mlp_bundle_fold_count,
        seed_policy_cell_count=coverage.seed_policy_cell_count,
        seed_policy_bundle_count=coverage.seed_policy_bundle_fold_count,
        parent_final_holdout_evaluation_count=EXPECTED_FINAL_EVALUATION_COUNT,
        parent_final_holdout_prediction_count=EXPECTED_FINAL_PREDICTION_COUNT,
        final_source_opened=False,
        final_labels_read=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the development-only Stage 4 completion release."
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--release-lock", default=DEFAULT_RELEASE_LOCK)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help=(
            "verify the tag-bound tracked release controls, source code, "
            "and parent final lock"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow tracked completion controls to differ from HEAD",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = verify_stage4_completion_release(
            args.repository_root,
            release_lock=args.release_lock,
            tracked_only=args.tracked_only,
            require_git_clean=not args.allow_dirty,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            f"Stage 4 completion release verification failed: {exc}"
        ) from exc
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
