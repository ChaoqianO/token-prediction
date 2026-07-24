# Implementation roadmap

This file maps the research plan in `token prediction7.19.md` to executable
interfaces. “Interface exists” means a real test exercises it; it does not mean
the corresponding research result has already been produced.

## Engineering-stage mapping

The repository's later engineering delivery used PR-oriented stage names that
do not reuse the research-note numbering.  The authoritative mapping is:

| Engineering delivery | Research-roadmap coverage | Frozen report |
| --- | --- | --- |
| Data Foundation / PR-0 | Stage 1 source and target contracts, plus the real-data entry criteria needed by Stages 2 and 6 | [data-foundation-report.md](data-foundation-report.md) |
| Engineering Stage 2 / PR-1 | Stage 2 common experiment path, the cross-position part of Stage 4, and the Independent-MLP/quantile parts of Stage 5 | [stage-2-report.md](stage-2-report.md) |
| Engineering Stage 3 / PR-2 | Stage 4 recurrent lifecycle methods and the lifecycle diagnostics required by Stage 5 | [stage-3-report.md](stage-3-report.md) |
| Engineering Stage 4 / PR-3 | Stage 3 feature ablations, Stage 5 calibration ablations, and Stage 6 multi-condition/Call capability gates and final validation | [stage-4-final-report.md](stage-4-final-report.md) |

The delivery branches, PRs, CI state, commits, artifact identities, limitations,
and stage-entry decisions are recorded in
[engineering-stage-delivery.md](engineering-stage-delivery.md).  This separate
delivery ledger preserves the byte-frozen scientific reports and their release
hashes.

## Current executable baseline

The repository can now run:

```text
multiple canonical trajectories
  -> lifecycle validation
  -> causal features and target-specific labels
  -> deterministic task-grouped train/validation/calibration/test folds
  -> Empirical, Length-only, Direct-estimate, Deduct-only, and LightGBM sessions
  -> task-level conformal interval calibration
  -> one shared point/task/raw interval metric suite
  -> immutable predictions/metrics/audit/reloadable-model artifact
```

A Task-pre forecast can be propagated through the full Task-update lifecycle
with inner-OOF initialization.  Cross-position Deduct, Independent MLP, GRU
residual/no-recurrence/zero-residual models, safe neural bundles, fold-fitted
retrieval gates, multi-condition evaluation, and a shared offline/online-shadow
session driver are implemented.  Call-update and white-box G3 surfaces remain
fail-closed when genuine generation telemetry is absent.

A saved `codex exec --json` turn can also be normalized into an honest
`task_launch -> task_total_accounted_tokens` example. Its cached/reasoning usage
fields are preserved and no internal Calls are synthesized. The matching
baseline plan is `configs/codex_task_mvp.toml`.

It does not fabricate live Codex trajectory collection, live LLM
self-estimation, or white-box generation probes.  Those surfaces require
separately authorized execution and their declared observables.  The
public-data LightGBM pilot and its limitations are reported in
[preliminary-lightgbm.md](preliminary-lightgbm.md).

Two labels in the source research note should be read consistently in future
configs: LightGBM is B3 and Independent MLP is B4. The feature-importance note
currently says B2, and the method-ablation note says B3 Independent MLP; those
are numbering mistakes, not separate baselines.

## Research Stage 1: freeze the method contract

Implemented:

- typed `task_pre`, `task_update`, `call_pre`, and `call_update` positions;
- algebraically closed Task and Call unknown-cost targets, including retry input;
- explicit point IDs and target-specific observed/censored/missing status;
- local/provider offset identity and retry-aware billable targets;
- task-grouped folds, equal task weighting, causal feature checks;
- common-cohort estimator, calibration, metrics, and artifact contracts.

Closure status on the audited sources:

- versioned BAGEN and Spend normalizers, source manifests, canonical hashes,
  reconciliation reports, and capability contracts are frozen;
- every enabled position is backed by observed telemetry, while unavailable
  positions fail closed rather than receiving synthetic facts;
- provider/model/Agent condition identity and repeated-task grouping are part
  of the dataset and split contracts; and
- continuation-state experiments remain gated because the audited sources do
  not provide genuine resumable decoder state.

## Research Stage 2: first real MVP

1. **Capability-gated:** a separately authorized raw-first live Codex collector
   must preserve official JSONL unchanged; no paid live collection is claimed
   by the frozen public-data releases.
2. **Implemented:** the deterministic Codex turn reader declares only Task
   aggregate usage proven by the saved fixture.
3. **Implemented for audited data:** BAGEN and Spend sources use the common
   experiment path.  Spend supplies four grouped runs per task; no claim is
   made that they came from a new live Codex collection.
4. **Implemented:** `LightGBMQuantileEstimator` receives the same `TrainingView`,
   uses validation-only early stopping, and returns the same `TokenForecast` as
   every baseline. Encoder schema, model text, fit report, and gain are exported
   per fold; a strict checksum-verified text-model bundle is directly reloadable.
5. **Partially implemented:** a frozen external self-estimate can run through
   `DirectFeatureEstimator`. Live self-estimation still needs a source adapter,
   measured overhead fields, and explicit failure handling.

The entry criterion is satisfied: one experiment path deterministically
performs data selection, training, calibration, evaluation, and artifact
publication for every enabled candidate.

## Research Stage 3: feature selection

- **Implemented:** G0/G1/G2 facts enter only when their source observation is
  real.
- **Implemented and capability-gated:** similar-task retrieval is a fold-fitted
  transformer whose index contains outer-training tasks only; the frozen
  sources lack genuine task text, so the production matrix fails closed.
- **Implemented:** restricted and full feature sets use the same estimator,
  cohort, splits, calibration, and metric suite.
- **Implemented:** every group deletion is an ablation `CandidateSpec`, and the
  resolved-config guard proves that only `feature_set` changed.
- **Implemented:** retention decisions use paired task-level bootstrap
  confidence intervals; split gain remains diagnostic.

## Research Stage 4: dynamic methods

The session API is the extension point:

- **Implemented:** within-cell Deduct remains a distinct historical baseline.
- **Implemented:** cross-position Deduct starts from an inner-OOF Task-pre seed,
  applies the visible mechanical transition, and never sees the true total.
- **Implemented:** Independent MLP predicts each point from current visible
  facts using train-fold-only encoders and safe `safetensors` bundles.
- **Implemented:** GRU residual, no-recurrence, and exact zero-residual
  candidates consume the same visible transitions.
- **Implemented:** state-update ablations declare and validate the one allowed
  configuration path.

Offline replay and online shadow evaluation must call the same session methods
in the same order.

## Research Stage 5: interval prediction and ablations

- **Implemented:** quantile LightGBM/MLP heads return repaired
  lower/point/upper forecasts while preserving raw outputs.
- **Implemented:** raw crossing, coverage, interval score, and width are
  retained before calibration.
- **Implemented:** calibration tasks remain task-separated and calibration
  alternatives are compared at matched coverage.
- **Implemented/gated:** calibration ablations have exact config-diff guards;
  probe-interval ablations require genuine generation checkpoints.

## Research Stage 6: Call progress and model expansion

Call-progress experiments are enabled only for sources with real output deltas.
Entropy, stop probability, and hidden state additionally require verified
logprob/white-box observations. OpenHands and local-model readers are new
trajectory sources; they do not change datasets, estimators, or evaluation.

Call-pre is implemented for the three distinct billed-output, billed-total, and
final-response targets.  Call-update/G3 remains fail-closed on the audited
sources because real output deltas and white-box observations are absent.

Every target LLM is compared with its own Empirical Quantile, Length-only, and
Deduct-only baselines under the same Agent/model/config condition.  Primary
results remain condition-specific; matched-task cross-model analysis is
secondary.

## Adding a method without changing the pipeline

```python
registry.register("my_estimator", lambda params: MyEstimator(**params))
```

`MyEstimator.fit(train, validation, ...)` returns a fitted object whose `start(...)` creates a
session. The session implements `predict(point)` and `observe(transition)`. A
configuration entry then selects an existing `FeatureSet`. No baseline-specific,
model-specific, or ablation-specific branch is permitted in the pipeline.
