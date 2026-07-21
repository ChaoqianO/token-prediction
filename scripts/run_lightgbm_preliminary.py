from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.collection import BagenSokobanMetadata, BagenSokobanReader
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
    assign_task_folds,
    build_spend_your_money_dataset,
    build_supervised_dataset,
    load_swebench_verified_metadata,
)
from token_prediction.estimators import builtin_registry
from token_prediction.evaluation import paired_task_bootstrap
from token_prediction.experiment import (
    AblationAxis,
    AblationSpec,
    CandidateResult,
    CandidateRole,
    CandidateSpec,
    ExperimentRunner,
    ExperimentSpec,
    FoldArtifact,
)
from token_prediction.features import FeatureGroup, FeatureSet
from token_prediction.lineage import publish_artifact, verify_artifact


PROTOCOL_VERSION = 2
PRIMARY_SEED = 20260719
DEFAULT_STABILITY_SEEDS = (20260719, 20260720, 20260721)
FOLDS = 5
ALPHA = 0.10
LIGHTGBM_PARAMS: dict[str, Any] = {
    "num_boost_round": 400,
    "early_stopping_rounds": 40,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "lambda_l2": 1.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen preliminary LightGBM experiments on two public datasets"
    )
    parser.add_argument("--bagen-json", required=True, type=Path)
    parser.add_argument("--spend-csv", required=True, type=Path)
    parser.add_argument("--swebench-parquet", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workspace/experiments/lightgbm_preliminary"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument(
        "--stability-seeds",
        default=",".join(str(value) for value in DEFAULT_STABILITY_SEEDS),
    )
    return parser


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("stability seeds must be comma-separated integers") from exc
    if not seeds:
        raise ValueError("at least one stability seed is required")
    if PRIMARY_SEED not in seeds:
        seeds = (PRIMARY_SEED, *seeds)
    return tuple(dict.fromkeys(seeds))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _spend_candidates() -> tuple[CandidateSpec, ...]:
    empty = FeatureSet("none", include_all=False)
    chars = FeatureSet(
        "task_chars",
        include_all=False,
        include_features=frozenset({"task_char_count"}),
    )
    self_estimate = FeatureSet(
        "self_estimate",
        include_all=False,
        include_features=frozenset({"llm_self_estimated_total_tokens"}),
    )
    shape = FeatureSet(
        "task_shape",
        include_all=False,
        include_features=frozenset(
            {
                "task_char_count",
                "task_word_count",
                "task_line_count",
                "task_code_fence_count",
            }
        ),
    )
    shape_repo = FeatureSet(
        "task_shape_repo",
        include_all=False,
        include_features=shape.include_features | {"repo_id"},
    )
    params = dict(LIGHTGBM_PARAMS)
    return (
        CandidateSpec(
            "empirical_quantile",
            "empirical_quantile",
            empty,
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "task_char_linear",
            "length_only",
            chars,
            params={"feature_name": "task_char_count"},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "llm_self_estimation",
            "direct_feature",
            self_estimate,
            params={"feature_name": "llm_self_estimated_total_tokens"},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_task_shape_repo",
            "lightgbm_quantile",
            shape_repo,
            params=params,
            role=CandidateRole.MODEL,
        ),
        CandidateSpec(
            "lightgbm_task_shape",
            "lightgbm_quantile",
            shape,
            params=params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_task_shape_repo",
                axis=AblationAxis.FEATURE_SET,
                allowed_config_paths=frozenset({"feature_set"}),
            ),
        ),
    )


def _bagen_candidates() -> tuple[CandidateSpec, ...]:
    empty = FeatureSet("none", include_all=False)
    request = FeatureSet(
        "request_length",
        include_all=False,
        include_features=frozenset({"current_request_tokens_local"}),
    )
    history = FeatureSet(
        "history_only",
        include_all=False,
        include_groups=frozenset({FeatureGroup.G1}),
    )
    history_request = FeatureSet(
        "history_request_proxy",
        include_all=False,
        include_groups=frozenset({FeatureGroup.G0, FeatureGroup.G1, FeatureGroup.G2}),
    )
    params = dict(LIGHTGBM_PARAMS)
    return (
        CandidateSpec(
            "empirical_quantile",
            "empirical_quantile",
            empty,
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "request_length_linear",
            "length_only",
            request,
            params={"feature_name": "current_request_tokens_local"},
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "deduct_only",
            "deduct_only",
            empty,
            role=CandidateRole.BASELINE,
        ),
        CandidateSpec(
            "lightgbm_history_request_proxy",
            "lightgbm_quantile",
            history_request,
            params=params,
            role=CandidateRole.MODEL,
        ),
        CandidateSpec(
            "lightgbm_history_only",
            "lightgbm_quantile",
            history,
            params=params,
            role=CandidateRole.ABLATION,
            ablation=AblationSpec(
                reference_candidate_id="lightgbm_history_request_proxy",
                axis=AblationAxis.FEATURE_SET,
                allowed_config_paths=frozenset({"feature_set"}),
            ),
        ),
    )


def _specs() -> tuple[ExperimentSpec, ExperimentSpec]:
    return (
        ExperimentSpec(
            experiment_id="spend_gpt52_task_launch",
            position=PredictionPosition.TASK_LAUNCH,
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            candidates=_spend_candidates(),
            alpha=ALPHA,
            calibrator_id="task_max_conformal",
        ),
        ExperimentSpec(
            experiment_id="bagen_codex_task_update",
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            candidates=_bagen_candidates(),
            alpha=ALPHA,
            calibrator_id="task_max_conformal",
        ),
    )


def _truth_by_point(dataset: SupervisedDataset, spec: ExperimentSpec) -> dict[str, float]:
    selected = dataset.select(spec.position, spec.target, condition_id=spec.condition_id)
    return {
        row.point.point_id: float(row.label)
        for row in selected.rows
        if row.label is not None
    }


def _result_map(results: Sequence[CandidateResult]) -> dict[str, CandidateResult]:
    return {result.candidate_id: result for result in results}


def _semantic_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("latency_")
    }


def _comparison_pairs(experiment_id: str) -> tuple[tuple[str, str], ...]:
    if experiment_id == "spend_gpt52_task_launch":
        return (
            ("lightgbm_task_shape_repo", "empirical_quantile"),
            ("lightgbm_task_shape", "empirical_quantile"),
            ("lightgbm_task_shape_repo", "lightgbm_task_shape"),
            ("llm_self_estimation", "empirical_quantile"),
        )
    return (
        ("deduct_only", "empirical_quantile"),
        ("deduct_only", "request_length_linear"),
        ("lightgbm_history_only", "deduct_only"),
        ("lightgbm_history_only", "empirical_quantile"),
        ("lightgbm_history_only", "request_length_linear"),
        ("lightgbm_history_request_proxy", "empirical_quantile"),
        ("lightgbm_history_request_proxy", "lightgbm_history_only"),
    )


def _run_one(
    dataset: SupervisedDataset,
    spec: ExperimentSpec,
    *,
    seed: int,
) -> tuple[CandidateResult, ...]:
    split = assign_task_folds(dataset.task_ids, folds=FOLDS, seed=seed).bind(
        dataset.dataset_id
    )
    return ExperimentRunner(builtin_registry()).run(dataset, split, spec, seed=seed)


def _summarize_seed(
    results: Sequence[CandidateResult],
    truth: Mapping[str, float],
    spec: ExperimentSpec,
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    by_id = _result_map(results)
    comparisons = {}
    for candidate_id, reference_id in _comparison_pairs(spec.experiment_id):
        comparison = paired_task_bootstrap(
            by_id[candidate_id],
            by_id[reference_id],
            truth,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        comparisons[f"{candidate_id}__vs__{reference_id}"] = asdict(comparison)
    return {
        "seed": seed,
        "candidates": {
            result.candidate_id: {
                "metrics": _semantic_metrics(result.metrics),
                "fold_metrics": {
                    str(fold): _semantic_metrics(metrics)
                    for fold, metrics in result.fold_metrics.items()
                },
            }
            for result in results
        },
        "paired_task_bootstrap": comparisons,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_fold_artifacts(
    root: Path,
    experiment_id: str,
    candidate_id: str,
    artifacts: Iterable[FoldArtifact],
) -> None:
    for artifact in artifacts:
        directory = root / "models" / experiment_id / candidate_id / f"fold_{artifact.fold}"
        directory.mkdir(parents=True, exist_ok=False)
        if artifact.encoder is not None:
            _write_json(directory / "encoder.json", dict(artifact.encoder))
        if artifact.fit_report is not None:
            _write_json(directory / "fit_report.json", dict(artifact.fit_report))
        if artifact.feature_importance is not None:
            (directory / "feature_importance.jsonl").write_text(
                "\n".join(
                    json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
                    for record in artifact.feature_importance
                )
                + "\n",
                encoding="utf-8",
            )
        for name, model in (artifact.model_strings or {}).items():
            if not name.replace("_", "").isalnum():
                raise ValueError(f"unsafe model artifact name: {name!r}")
            (directory / f"{name}.model.txt").write_text(model, encoding="utf-8")
        if artifact.bundle_files is not None:
            bundle_dir = directory / "bundle"
            bundle_dir.mkdir()
            for name, payload in sorted(artifact.bundle_files.items()):
                if (
                    not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\\" in name
                ):
                    raise ValueError(f"unsafe bundle artifact name: {name!r}")
                (bundle_dir / name).write_bytes(payload)


def _feature_importance_summary(results: Sequence[CandidateResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for result in results:
        records = [
            dict(record)
            for artifact in result.fold_artifacts
            for record in (artifact.feature_importance or ())
            if float(record["quantile"]) == 0.5
        ]
        if not records:
            continue
        by_feature: dict[str, list[float]] = {}
        for record in records:
            by_feature.setdefault(str(record["source_feature_name"]), []).append(
                float(record["normalized_gain"])
            )
        summary[result.candidate_id] = [
            {
                "source_feature_name": name,
                "mean_normalized_gain": sum(values) / len(values),
                "fold_count": len(values),
            }
            for name, values in sorted(
                by_feature.items(),
                key=lambda item: (-sum(item[1]) / len(item[1]), item[0]),
            )
        ]
    return summary


def _prediction_lines(
    experiment_id: str,
    results: Sequence[CandidateResult],
    truth: Mapping[str, float],
) -> list[str]:
    lines: list[str] = []
    for result in results:
        for record in result.predictions:
            forecast = record.forecast
            payload = {
                "experiment_id": experiment_id,
                "candidate_id": result.candidate_id,
                "point_id": record.point_id,
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "fold": record.fold,
                "truth": truth[record.point_id],
                "prediction": forecast.point,
                "lower": forecast.lower,
                "upper": forecast.upper,
                "raw_lower": forecast.raw_lower,
                "raw_prediction": forecast.raw_point,
                "raw_upper": forecast.raw_upper,
                "sample_weight": record.sample_weight,
            }
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return lines


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "mean": sum(ordered) / len(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    seeds = _parse_seeds(args.stability_seeds)
    paths = {
        "bagen_json": args.bagen_json.resolve(),
        "spend_csv": args.spend_csv.resolve(),
        "swebench_parquet": args.swebench_parquet.resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"missing input file(s): {missing}")

    project_root = Path(__file__).resolve().parents[1]
    source_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": seeds,
        "folds": FOLDS,
        "split_unit": "task_id",
        "partitions": ["train", "validation", "calibration", "test"],
        "interval_alpha": ALPHA,
        "calibrator": "task_max_conformal",
        "lightgbm_params": LIGHTGBM_PARAMS,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_unit": "task_id",
        "source_hashes": source_hashes,
        "code_hash": _code_hash(project_root),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
        },
    }
    run_id = _canonical_hash(protocol)[:20]
    output_dir = args.output_root.resolve() / run_id
    if (output_dir / "_SUCCESS").is_file():
        manifest = verify_artifact(output_dir)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                    "artifact_id": manifest.artifact_id,
                    "cached": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    metadata = load_swebench_verified_metadata(paths["swebench_parquet"])
    spend_import = build_spend_your_money_dataset(
        paths["spend_csv"],
        metadata,
        model_key="gpt52",
        model_id="gpt-5.2",
        metadata_sha256=source_hashes["swebench_parquet"],
    )
    bagen_trajectories = BagenSokobanReader().read_all(
        paths["bagen_json"],
        BagenSokobanMetadata(reasoning_effort="low"),
    )
    bagen_dataset = build_supervised_dataset(bagen_trajectories)
    spend_spec, bagen_spec = _specs()
    datasets = {
        spend_spec.experiment_id: spend_import.dataset,
        bagen_spec.experiment_id: bagen_dataset,
    }
    specs = {
        spend_spec.experiment_id: spend_spec,
        bagen_spec.experiment_id: bagen_spec,
    }
    truths = {
        experiment_id: _truth_by_point(dataset, specs[experiment_id])
        for experiment_id, dataset in datasets.items()
    }

    results_by_seed: dict[int, dict[str, tuple[CandidateResult, ...]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_results: dict[str, tuple[CandidateResult, ...]] = {}
        seed_summary: dict[str, Any] = {}
        for experiment_id, dataset in datasets.items():
            results = _run_one(dataset, specs[experiment_id], seed=seed)
            seed_results[experiment_id] = results
            seed_summary[experiment_id] = _summarize_seed(
                results,
                truths[experiment_id],
                specs[experiment_id],
                seed=seed,
                bootstrap_iterations=args.bootstrap_iterations,
            )
        results_by_seed[seed] = seed_results
        summaries[str(seed)] = seed_summary

    if _code_hash(project_root) != protocol["code_hash"]:
        raise RuntimeError(
            "source tree changed while the experiment was running; "
            "refusing to publish a mixed-code artifact"
        )

    primary = results_by_seed[PRIMARY_SEED]
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "stability_results.json", summaries)

    prediction_lines: list[str] = []
    importance: dict[str, Any] = {}
    for experiment_id, results in primary.items():
        prediction_lines.extend(
            _prediction_lines(experiment_id, results, truths[experiment_id])
        )
        importance[experiment_id] = _feature_importance_summary(results)
        for result in results:
            if result.fold_artifacts:
                _write_fold_artifacts(
                    output_dir,
                    experiment_id,
                    result.candidate_id,
                    result.fold_artifacts,
                )
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(prediction_lines) + "\n", encoding="utf-8"
    )
    _write_json(output_dir / "feature_importance.json", importance)

    bagen_slice = bagen_dataset.select(
        bagen_spec.position,
        bagen_spec.target,
    )
    dataset_summary = {
        "spend_gpt52_task_launch": {
            "dataset_id": spend_import.dataset.dataset_id,
            "tasks": len(spend_import.dataset.task_ids),
            "eligible_points": len(truths[spend_spec.experiment_id]),
            "target": "rounded four-run mean provider input + output tokens",
            "target_distribution": _distribution(
                list(truths[spend_spec.experiment_id].values())
            ),
            "condition": "GPT-5.2 + OpenHands only",
        },
        "bagen_codex_task_update": {
            "dataset_id": bagen_dataset.dataset_id,
            "raw_trajectories": len(bagen_trajectories),
            "raw_tasks": len(bagen_dataset.task_ids),
            "all_rows": len(bagen_dataset.rows),
            "eligible_points": len(bagen_slice.rows),
            "eligible_tasks": len({row.point.task_id for row in bagen_slice.rows}),
            "missing_or_invalid_rows": sum(
                not row.eligible for row in bagen_dataset.rows
            ),
            "target": "future billed tokens excluding the current request input proxy",
            "target_distribution": _distribution(
                list(truths[bagen_spec.experiment_id].values())
            ),
            "condition": "OpenAI GPT-5.2 Codex low-thinking + CoordSokoban",
        },
    }
    _write_json(output_dir / "dataset_summary.json", dataset_summary)
    _write_json(
        output_dir / "provenance.json",
        {
            "sources": {
                "spend_aggregate": {
                    "url": (
                        "https://github.com/LongjuBai/agent_token_consumption/"
                        "blob/master/all_models_averaged_predictions_new.csv"
                    ),
                    "sha256": source_hashes["spend_csv"],
                    "license_status": "not declared upstream; do not redistribute",
                },
                "swebench_verified": {
                    "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified",
                    "sha256": source_hashes["swebench_parquet"],
                },
                "bagen": {
                    "url": "https://huggingface.co/datasets/MLL-Lab/BAGEN",
                    "sha256": source_hashes["bagen_json"],
                    "license_status": "not declared upstream; do not redistribute",
                },
            },
            "leakage_guards": [
                "all folds grouped by task_id",
                "encoder vocabulary and vector layout fit on outer-train only",
                "early stopping uses validation only",
                "interval calibration uses calibration tasks only",
                "test labels are used only by evaluation and paired bootstrap",
                "BAGEN total_turns/final_state/success/actual_* fields are excluded",
                "SWE-bench gold patch/test patch fields are excluded",
                "LLM self-estimate is available only to its direct baseline feature set",
            ],
            "known_limitations": [
                "Spend target is a four-run mean, so it cannot evaluate run-level variance",
                "BAGEN request length uses provider input usage as a local-token proxy",
                "BAGEN missing per-attempt usage is never filled with zero",
                "gain importance is descriptive; paired feature-set ablation is primary evidence",
            ],
        },
    )
    manifest = publish_artifact(
        output_dir,
        stage_name="preliminary_lightgbm_experiment",
        metadata={"run_id": run_id, **protocol},
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "artifact_id": manifest.artifact_id,
                "cached": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
