# Architecture

## The domain boundary

This repository is a token-consumption prediction system, not a generic Agent
framework. Its pipeline is therefore organized around trajectories and
prediction experiments:

```text
trajectory source
  -> canonical trajectory
  -> prefix-causal prediction points + target-specific labels
  -> frozen task split
  -> estimator sessions
  -> shared calibration and evaluation
  -> immutable experiment artifact
```

`collection/` replaces the former, overly broad `execution/` package. A replay
reader and a live Agent are both possible trajectory sources, but neither may
invent a request, Call, usage value, or checkpoint that its raw source did not
expose.

## Current tree

```text
src/token_prediction/
  collection/
    source.py          CollectionTask and TrajectorySource boundary
    replay.py          deterministic canonical JSONL ingestion
    codex_cli.py       official CLI auth delegation; no credential parsing
  trajectory.py        task/run identity and event lifecycle validation
  contracts.py         canonical events, usage ledger, observables
  recording/            append-only event storage and redaction
  features/
    reducer.py         online/offline prefix-causal raw facts
    selection.py       G0-G3 catalog and feature views
  dataset/
    capabilities.py    source × position × target fail-closed gates
    labels.py          independent status for every prediction target
    builder.py         v1 compatibility and capability-enforced v2 joins
    schema.py          points, rows, slices, target-specific weights
    splits.py          immutable task-grouped folds
    spend_your_money.py  one-condition aggregate benchmark importer
  estimators/
    base.py            common fit -> fitted -> session contract
    baselines.py       empirical, length-only, and direct-estimate baselines
    deduct.py          within-cell Task-update deduction state machine
    tabular_encoder.py train-fold-only numeric/category/vector encoding
    lightgbm.py        deterministic three-quantile LightGBM estimator
    lightgbm_bundle.py strict text-model bundle save/load and verification
    registry.py        explicit estimator extension point
  evaluation/
    calibration.py     task-separated conformal calibration
    metrics.py         one metric suite for every candidate
    comparison.py      paired task-clustered bootstrap
  experiment.py        common-cohort cross-validation and ablation guard
  pipeline.py          configuration-driven artifact orchestration
```

No empty OpenHands, MLP, GRU, or local-model package is created before its first
implementation exists. LightGBM now has a real estimator, fold-fitted encoder,
strict reloadable bundle, fit report, feature importance, and integration tests.

## Prediction positions and telemetry gates

The domain schema represents all four positions from the research design, plus
an earlier Task launch cell for sources that expose only aggregate run usage:

| Position | Boundary | Required observations | Current canonical support |
| --- | --- | --- | --- |
| `task_launch` | before starting the external Agent | task metadata and Task aggregate usage | Codex saved-JSONL and aggregate benchmark readers |
| `task_pre` | first `request_built` | request boundary, attempt usage, Task termination | provider-accounted target; local target gated separately |
| `task_update` | later `request_built` | prior usage, request boundary, Task termination | provider-accounted target; local target gated separately |
| `call_pre` | every `request_built` | request boundary and attempt usage | provider-accounted total/output/final targets |
| `call_update` | `generation_checkpoint` | output deltas and terminal Call usage | schema and builder only |

Support in this table means that correctly instrumented canonical trajectories
can flow through the dataset and experiment code. It does not mean every source
provides the necessary facts.

The locally tested Codex CLI JSONL surface exposes turn-level aggregate usage,
but not internal `request_built` boundaries or per-Call usage. The deterministic
`CodexTurnReader` therefore creates only `task_launch` examples. Live Codex
collection remains disabled until its raw collector is implemented. A future
verified CLI surface may add observations, but the current reader must never
parse item events into fictitious Calls.

`BagenSokobanReader` is an offline research adapter for preserved BAGEN
dialogues. It emits real request, attempt, usage, and tool boundaries and keeps
missing attempt usage unknown. BAGEN does not expose an independent local
request-token count, so provider input is retained only as a post-response audit
field and is not advertised as the `REQUEST_LOCAL_COUNT` source capability.

Capabilities are an independent set of `Observable` values, not a false total
ordering such as `run < call < attempt`. Schema-v2 builders fail closed when a
target's required set is absent. Their dataset identity binds the stable
`SourceDescriptor` and capability-contract hashes; schema v1 remains only for
frozen Stage 1 verification.

## Estimator interface

Every baseline and learned method follows the same lifecycle:

```text
TokenEstimator.fit(TrainingView, ValidationView, FitContext)
  -> FittedEstimator.start(RunContext)
  -> PredictionSession.observe(ObservedTransition)
  -> PredictionSession.predict(PredictionPoint)
  -> TokenForecast
```

This contract is intentionally session-based:

- empirical quantile, length-only, direct-feature, LightGBM, and independent MLP
  sessions ignore `observe`;
- the current within-cell Deduct-only cold-starts the first eligible Task-update
  from outer-train weighted quantiles, then stores the prior forecast and
  deducts newly observed spend;
- a GRU session updates hidden state from each visible transition;
- LLM self-estimation records its own input/output token overhead in the same
  `TokenForecast` schema;
- an online shadow predictor and offline replay use the same ordered session
  calls.

To add an estimator, implement the protocol and register one factory in an
`EstimatorRegistry`. The experiment runner, split logic, calibration, metrics,
and artifacts do not change.

## Fair comparison contract

`ExperimentRunner` fixes the following before any candidate is constructed:

- dataset ID and eligible `(point, target)` cohort;
- task-to-fold assignment;
- train/validation/calibration/test task partitions;
- sample weights, target, interval coverage, and metric suite.

The training type includes targets. A prediction point does not. Estimators
therefore cannot receive test labels through the prediction API. Every fold must
return exactly one forecast for every requested point; missing or duplicate
forecasts fail the run instead of changing the evaluation denominator.

Every trajectory and point also carries a `condition_id` derived from the
resolved Agent/model/reasoning/tool/context configuration. An experiment cell
may contain only one condition unless it explicitly selects one, so adding a new
target model cannot silently pool it with another model's baseline.

An ablation is a `CandidateSpec` with a reference and an explicit set of allowed
configuration paths. The resolved configurations are compared before training.
If a feature ablation also changes an estimator parameter, split, target, or
calibration setting, the experiment is rejected.

## Dependency direction

```text
contracts / trajectory
  <- collection, recording
  <- features, dataset
  <- estimators, evaluation
  <- experiment
  <- config, pipeline, cli
```

- feature reduction only consumes events at or before the point cutoff;
- labels may consume the complete trajectory but are never attached to a
  prediction input;
- fold-fitted retrieval, similar-task features, encoders, and scaling belong in
  estimator/transformer fit code, never the raw reducer;
- collection cannot import datasets or estimators;
- only the pipeline publishes generated artifacts.

## Persistence

Collection, dataset, split, candidate, prediction, and metric identities are
content-addressed or recorded in the experiment manifest. Published artifacts
are immutable and verification rejects changed, missing, and newly added files.
Credentials and local Codex runtime state remain outside artifact inputs.

LightGBM fold audit files and its strict deployable file set are deliberately
separate. `fold_N/bundle/` contains only files declared by the bundle manifest;
the parent fold directory contains feature importance and human-readable fit
reports. The outer artifact hashes nested bundle manifests and model files, so
both outer publication verification and inner bundle verification cover them.
