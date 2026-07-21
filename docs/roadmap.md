# Implementation roadmap

This file maps the research plan in `token prediction7.19.md` to executable
interfaces. “Interface exists” means a real test exercises it; it does not mean
the corresponding research result has already been produced.

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

A saved `codex exec --json` turn can also be normalized into an honest
`task_launch -> task_total_accounted_tokens` example. Its cached/reasoning usage
fields are preserved and no internal Calls are synthesized. The matching
baseline plan is `configs/codex_task_mvp.toml`.

It does not yet provide live Codex trajectory collection, live LLM
self-estimation with measured overhead, neural state updates, or white-box
generation probes. The public-data LightGBM pilot and its limitations are
reported in [preliminary-lightgbm.md](preliminary-lightgbm.md).

Two labels in the source research note should be read consistently in future
configs: LightGBM is B3 and Independent MLP is B4. The feature-importance note
currently says B2, and the method-ablation note says B3 Independent MLP; those
are numbering mistakes, not separate baselines.

## Stage 1: freeze the method contract

Implemented:

- typed `task_pre`, `task_update`, `call_pre`, and `call_update` positions;
- algebraically closed Task and Call unknown-cost targets, including retry input;
- explicit point IDs and target-specific observed/censored/missing status;
- local/provider offset identity and retry-aware billable targets;
- task-grouped folds, equal task weighting, causal feature checks;
- common-cohort estimator, calibration, metrics, and artifact contracts.

Still required before declaring Stage 1 complete on real data:

- a versioned raw-source normalizer and ledger reconciliation report;
- enough telemetry to construct every enabled position;
- provider/model/tokenizer/Agent condition identity;
- repeated-task and continuation-state identity audits.

## Stage 2: first real MVP

1. **Pending:** add a raw-first live Codex collector that preserves official
   JSONL unchanged.
2. **Implemented:** the deterministic Codex turn reader declares only Task
   aggregate usage proven by the saved fixture.
3. **Partially implemented:** public BAGEN and aggregate SWE-bench pilots use the
   common experiment path; repeated Task runs from our own Codex collector are
   still required.
4. **Implemented:** `LightGBMQuantileEstimator` receives the same `TrainingView`,
   uses validation-only early stopping, and returns the same `TokenForecast` as
   every baseline. Encoder schema, model text, fit report, and gain are exported
   per fold; a strict checksum-verified text-model bundle is directly reloadable.
5. **Partially implemented:** a frozen external self-estimate can run through
   `DirectFeatureEstimator`. Live self-estimation still needs a source adapter,
   measured overhead fields, and explicit failure handling.

Entry criterion: one configuration deterministically performs data selection,
training, calibration, evaluation, and artifact publication for every enabled
candidate.

## Stage 3: feature selection

- Add raw G0/G1/G2 facts only when their source observation is real.
- Implement similar-task retrieval as a fold-fitted transformer; its index may
  contain outer-training tasks only.
- Compare the same estimator family with restricted vs full feature sets.
- Express every group deletion as an ablation `CandidateSpec`; the resolved
  config guard must show that only `feature_set` changed.
- Use paired task-level bootstrap confidence intervals. Split gain may be a
  diagnostic, not the main causal evidence of feature value.

## Stage 4: dynamic methods

The existing session API is the extension point:

- **Partially implemented:** within-cell Deduct-only initializes from outer-train
  weighted quantiles and updates state in `observe`; it never sees the true
  total. A unified cross-position runner is still required to seed it from the
  same test trajectory's out-of-fold Task-pre forecast.
- Independent MLP ignores `observe` and predicts every point from current facts.
- GRU maintains a hidden state and consumes the same visible transitions.
- state-update ablations may alter only the declared update policy/config path.

Offline replay and online shadow evaluation must call the same session methods
in the same order.

## Stage 5: interval prediction and ablations

- add quantile LightGBM/MLP heads returning ordered lower/point/upper forecasts;
- preserve raw crossing and raw coverage before calibration;
- use task-separated calibration and compare interval score at matched coverage;
- add calibration and probe-interval ablation axes with exact config-diff guards.

## Stage 6: Call progress and model expansion

Call-progress experiments are enabled only for sources with real output deltas.
Entropy, stop probability, and hidden state additionally require verified
logprob/white-box observations. OpenHands and local-model readers are new
trajectory sources; they do not change datasets, estimators, or evaluation.

Every new target LLM is compared with its own Empirical Quantile, Length-only,
and Deduct-only baselines under the same Agent/model/config condition. Cross-model
pooled results are secondary unless token-accounting semantics are normalized.

## Adding a method without changing the pipeline

```python
registry.register("my_estimator", lambda params: MyEstimator(**params))
```

`MyEstimator.fit(train, validation, ...)` returns a fitted object whose `start(...)` creates a
session. The session implements `predict(point)` and `observe(transition)`. A
configuration entry then selects an existing `FeatureSet`. No baseline-specific,
model-specific, or ablation-specific branch is permitted in the pipeline.
