# Token Prediction

Predict the remaining token consumption of an Agent using only information that
is visible at the prediction point.

The repository contains a zero-dependency core offline prediction pipeline plus
optional learned estimators. It can ingest multiple canonical trajectories,
build leakage-safe prediction rows, freeze task-grouped train/validation/
calibration/test folds, fit baselines and three-head quantile LightGBM, calibrate
intervals, compare candidates on the same cohort, and publish an immutable
experiment artifact with per-fold models and fit reports.

It does **not** yet launch live Codex collection. A saved Codex JSONL turn can
already be normalized into an honest Task-launch total-usage example. The
locally tested CLI does not expose verified internal request/Call boundaries,
so the code deliberately refuses to fabricate Call data. BAGEN Sokoban and the
full audited BAGEN SWE-bench and *How Do AI Agents Spend Your Money?* GPT-5.2
archives have dedicated offline readers. Schema-v2 builds are capability-gated
and bind tracked source descriptors plus byte-verified canonical manifests;
schema-v1 configurations remain verification-only for historical artifacts.

## Pipeline

```text
collection -> trajectory validation -> features/labels -> task split
           -> estimator sessions -> calibration/evaluation -> artifact
```

The important extension rule is simple: baselines, learned models, and
ablations all implement the same estimator/session contract and use the same
frozen point IDs, folds, weights, calibration, and metric code.

See:

- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Implementation roadmap](docs/roadmap.md)
- [Current engineering Stage 1 delivery](docs/stage-1-completion.md)
- [Preliminary LightGBM experiment](docs/preliminary-lightgbm.md)
- [Data Foundation verification](docs/data-foundation-report.md)

## Local tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The suite includes a 5-task × 2-run end-to-end experiment smoke test, target
censoring/retry checks, prefix-causality checks, common-cohort comparisons, and
artifact verification.

Inspect one canonical trajectory:

```powershell
python -m token_prediction.cli replay `
  --events tests/fixtures/two_call_events.jsonl
```

Historical schema-v1 experiment plans remain readable for artifact inspection,
but the runner deliberately refuses to train or publish from them. New runs
must use a schema-v2 plan that pins a tracked source descriptor and a canonical
input manifest (including every input path, byte count, and SHA-256):

```powershell
python -m token_prediction.cli experiment `
  --config path/to/schema-v2-plan.toml `
  --events path/to/run-1.jsonl path/to/run-2.jsonl
```

`configs/mvp.toml`, `configs/lightgbm_mvp.toml`, and
`configs/codex_task_mvp.toml` are retained historical schema-v1 plans. They are
not accepted as provenance for a new artifact.

Normalize a preserved raw turn first:

```powershell
python -m token_prediction.cli ingest codex-turn `
  --raw workspace/raw/codex-turn.jsonl `
  --output workspace/collections/codex-task.jsonl `
  --task-id swebench-task-id `
  --task-tokens 123 `
  --model-id gpt-model-id `
  --started-at 2026-07-21T00:00:00+08:00 `
  --finished-at 2026-07-21T00:05:00+08:00
```

The normalizer stores Task metadata, aggregate usage, raw hash and event-type
counts. It does not copy message text or turn item content into the canonical
trajectory.

At least as many distinct tasks as configured folds are required. Generated
collections, datasets, and experiments live under the ignored `workspace/`
tree.

## Preliminary LightGBM experiment

Install the optional estimator and data readers:

```powershell
python -m pip install ".[estimators,data]"
```

The frozen public-data pilot compares Empirical Quantile, task/request-length,
within-cell Deduct-only, LLM self-estimation, full LightGBM, and feature-set
ablations with task-grouped five-fold evaluation and task-clustered bootstrap:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_lightgbm_preliminary.py `
  --bagen-json workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json `
  --spend-csv workspace/external/spend_your_money/all_models_averaged_predictions_new.csv `
  --swebench-parquet workspace/external/spend_your_money/swe_bench_verified_test.parquet
```

The external raw files and all generated model artifacts remain under the
ignored `workspace/` tree. See the linked report for source hashes, leakage
guards, uncertainty intervals, limitations, and the exact observed results.

## Codex authentication

Authentication is delegated to the official Codex executable. Project code
never reads, returns, refreshes, or writes credential tokens:

```text
tp auth codex status
tp auth codex login
tp auth codex logout
```

`tp doctor codex` reports that live collection remains disabled until a
versioned raw JSONL reader has proven the observations required by a configured
prediction target.
