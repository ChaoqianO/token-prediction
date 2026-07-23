# Stage 3 development report

## Status

Stage 3 is complete on the sealed development cohorts. The implementation and
all production artifacts are bound to Git commit
`d3767c135c255d3803195573130f8bb0aefe0d67` and code-tree SHA-256
`18cb3fdc20df475f556683d3b1db7f83f549aa94442a39fe59225a084fdfba26`.
The permanent final holdout remains sealed: it was not used for fitting,
calibration, scoring, prediction, gating, or selection.

This stage delivered:

- one shared offline/shadow lifecycle driver with observe-then-predict ordering
  and measured per-prediction latency;
- deterministic training for the Independent MLP and GRU updater, with frozen
  inference models for portable replay and shadow execution;
- a GRU whose residuals are added to the mechanical cross-position Deduct
  forecast without teacher forcing;
- exact `residual_scale=0` reduction to cross-position Deduct and a separate
  no-recurrence ablation that changes only hidden-state carry;
- explicit progress, termination, repeated-run variance, and fixed token-budget
  evaluation;
- safe `safetensors` MLP and GRU components inside composite lifecycle bundles;
- atomic candidate and every-neural-epoch checkpoints containing current and
  best weights, full AdamW state, early-stopping history, and RNG state; and
- complete calibrated trajectory replay after bundle reload.

## Frozen protocol

The split seeds are `20260719`, `20260720`, and `20260721`. Development uses
five task-grouped outer folds and five task-grouped inner folds. All runs and
families belonging to a task remain grouped. Each outer-train updater receives
only inner-OOF Task-pre forecasts; outer validation, calibration, and test
receive the five inner models' ensemble forecast. Upstream quantiles are
non-negative and order-repaired but not conformal-calibrated. Task-update output
is calibrated exactly once with task-max conformal at alpha `0.10`.

Every candidate is evaluated on the same condition, target, cohort, weights,
folds, alpha, and metric suite. The frozen candidate set is Empirical,
cross-position Deduct, history LightGBM, history Independent MLP, GRU residual,
GRU without recurrence, and GRU with zero residual scale. Progress checkpoints
at 25%, 50%, and 75% are computed only by the evaluator from sequence order.
Budget decisions use the explicit thresholds 16,384, 32,768, 65,536, and
131,072 remaining provider-accounted tokens.

Each neural epoch is published atomically to a `safetensors` plus JSON
checkpoint. Restart resumes at the next epoch with optimizer, early-stopping,
and RNG state restored; completed candidate results are separately immutable
and identity-validated. Inference models are frozen before evaluation and
bundling, so training hardware is not part of the deployed lifecycle contract.

Missing or censored labels remain in lifecycle context with zero loss and score
masks. Missing usage-counter growth suppresses only the contaminated mechanical
deduction; later transitions recover when the counter is stable. Structurally
invalid trajectories are rejected rather than retained as context.

## Source coverage and artifacts

| Source | Development coverage | Experiments / gates | Candidate-seed runs | Artifact |
| --- | --- | ---: | ---: | --- |
| Spend aggregate | no request-boundary lifecycle; fail-closed | 0 / 1 | 0 | `workspace/stage3/runs/s3-0eadef35bf584fcb1ee7` |
| BAGEN Sokoban | 99 sequences; 649 update boundaries; 378 scored boundaries | 1 / 0 | 21 | `workspace/stage3/runs/s3-b0231bd18f6af59bb6e8` |
| BAGEN SWE | 229 condition-sequences; 8,091 update boundaries; 3,778 scored boundaries | 5 / 4 | 105 | `workspace/stage3/runs/s3-b35d4daecbeaa016a00a` |
| Spend full OpenHands | 1,586 sequences; 73,900 update boundaries; 71,360 scored boundaries | 1 / 0 | 21 | `workspace/stage3/runs/s3-6c57b8ef3acc736cceea` |

Spend aggregate is intentionally gate-only because aggregate Task-launch usage
does not expose request boundaries. BAGEN SWE conditions below the frozen
minimum of ten development tasks remain explicit gates and are not pooled into
the five primary model-family conditions.

## Development results

All values are mean MAE across the three frozen split seeds on the exact same
development cohort. MAE is in provider-accounted remaining tokens. These are
development results, not final-holdout estimates.

| Source / condition | Empirical | Cross-position Deduct | LightGBM history | MLP history | GRU residual | GRU no recurrence | GRU zero residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BAGEN Sokoban | 2,139 | 3,148 | **2,102** | 2,976 | 2,575 | 2,607 | 3,148 |
| BAGEN SWE Qwen 3 235B | 210,166 | 256,328 | **196,297** | 241,010 | 229,153 | 233,457 | 256,328 |
| BAGEN SWE GPT-5.2 Instant | 53,900 | 62,305 | **51,390** | 79,362 | 67,405 | 65,446 | 62,305 |
| BAGEN SWE Claude Opus 4.7 | 27,390 | 33,331 | **26,839** | 36,608 | 32,553 | 33,914 | 33,331 |
| BAGEN SWE Gemini 3.1 | 102,108 | 125,148 | **102,037** | 129,228 | 128,281 | 125,391 | 125,148 |
| BAGEN SWE Claude Sonnet 4.6 | 40,551 | 48,828 | **38,618** | 52,048 | 48,502 | 48,321 | 48,828 |
| Spend full OpenHands | 565,303 | 617,178 | 499,085 | 854,551 | 487,576 | **484,915** | 617,178 |

On Sokoban, the recurrent GRU lowers MAE from the mechanical Deduct base of
3,148.5 to 2,574.9, an 18.2% improvement. Recurrence contributes a further
31.9-token, or 1.2%, improvement over the otherwise identical no-recurrence
ablation. The learned residual is therefore useful, but history LightGBM remains
best at 2,101.8 MAE and Empirical remains lower at 2,139.0 MAE. Cross-source
interpretation is deferred until all frozen source artifacts close.

## Lifecycle diagnostics

Sokoban contains 98 observed-termination scored sequences and one unscored
missing-usage sequence. The missing sequence remains in context and contributes
no false Task-remaining MAE. The evaluator publishes per-candidate metrics at
25%, 50%, and 75% progress, termination strata, repeated-run MAE variance, and
all four budget scenarios for every seed.

Measured Sokoban mean p50 prediction latency is approximately 0.012 ms for
cross-position Deduct, 0.233 ms for LightGBM, 0.216 ms for Independent MLP, and
0.378 ms for the recurrent GRU. Latency measures only `session.predict`; state
observation, training, bundle loading, and serialization are excluded. Reload
parity intentionally ignores only this nondeterministic timing measurement and
requires every identity, raw/calibrated quantile, scope field, and overhead
counter to remain exact.

BAGEN SWE contributes 219 observed-termination sequences and 10 max-turns
censored sequences across the five primary model-family conditions. The 10
censored sequences remain context-only; their 2,417 update boundaries never
produce false Task-remaining scores. Mean p50 prediction latency across the
five conditions is approximately 0.006 ms for cross-position Deduct, 0.103 ms
for LightGBM, 0.124 ms for Independent MLP, and 0.228 ms for the recurrent GRU.

For the progress-stability gate, a checkpoint counts only when the GRU MAE is
lower than Deduct for all three split seeds. Qwen 3 235B passes only 75%;
GPT-5.2 Instant and Gemini 3.1 pass 50% and 75%; Claude Opus 4.7 and Claude
Sonnet 4.6 pass 25% and 50%. No BAGEN SWE condition passes all three progress
checkpoints, so the learned residual's benefit is not stable across task
progress.

Spend full contains 1,504 observed-termination sequences and 82 censored
`task_error` sequences. All 2,540 boundaries from the censored sequences remain
in lifecycle context and are unscored; the 71,360 observed-termination
boundaries form the scored cohort. All 397 development tasks have repeated
scored runs, so the evaluator publishes an estimable same-task run-variance
distribution rather than silently treating the four runs as independent tasks.

The Spend GRU beats Deduct at 25%, 50%, and 75% progress for every split seed.
Its mean p50 prediction latency is approximately 0.227 ms, compared with 0.117
ms for LightGBM, 0.124 ms for Independent MLP, and 0.005 ms for Deduct. The
fixed-budget tables for all four thresholds and all candidate-seed runs are
included in the artifact.

## Interpretation

The closed Sokoban result supports the residual formulation without supporting
a model-selection claim for GRU: it improves substantially over the exact
mechanical state transition, while remaining worse than both simpler point
comparators. Its mean 90% interval coverage is 90.44%, close to nominal;
LightGBM's 97.17% coverage is more conservative. The no-recurrence difference
is small enough that recurrence must be judged across the remaining conditions,
not from Sokoban alone. BAGEN SWE confirms that caution: GRU improves over
Deduct on Qwen 3 235B, Claude Opus 4.7, and Claude Sonnet 4.6, but regresses on
GPT-5.2 Instant and Gemini 3.1. Recurrence helps Qwen and Opus while hurting the
other three conditions. History LightGBM remains the lowest-MAE candidate in
all five BAGEN SWE conditions.

Spend full supplies the one condition where the learned residual also beats
history LightGBM: GRU lowers mean MAE from 499,085.3 to 487,576.4, a 2.3%
improvement, and lowers Deduct by 21.0%. The no-recurrence ablation improves a
further 2,661.7 tokens to 484,914.6 MAE. Thus visible current/history features
support a useful residual on Spend, but recurrent hidden-state carry does not.
Across the six BAGEN conditions plus Spend, there is no universal recurrent
advantage: history LightGBM wins every BAGEN condition, while no-recurrence
GRU wins Spend. Stage 4 therefore keeps condition-level reporting primary and
does not infer a pooled cross-source winner.

## Verification

The frozen release verifier requires:

- exact source, dataset, protocol, matrix, code-tree, artifact, and results
  identities;
- complete condition coverage, including fail-closed gates;
- positive measured prediction latency for every candidate and seed;
- exact zero-residual reduction to Deduct after removing latency only;
- exact Stage 2 numerical regression for Empirical, cross-position Deduct, and
  history LightGBM after removing only measured latency and run-local task
  pseudonyms;
- Stage 3 contract parity for history MLP, requiring the same estimator, feature
  set, candidate graph, splits, cohort projection, prediction count, task
  weights, alpha, and calibration contract;
- complete 25/50/75 progress, budget, and termination diagnostics;
- five reloadable folds for every declared bundle candidate;
- independent safe loading of every LightGBM, MLP, and lifecycle bundle, with
  each bundle checked against its source/code/dataset/condition/candidate/fold
  provenance; and
- a sealed permanent final holdout in every artifact.

The release closes four immutable artifacts containing seven experiments, one
fail-closed gate, 147 candidate-seed runs, 84 lifecycle candidate-seed runs,
420 exact lifecycle reload folds, 630 reloadable bundle folds, and 12,289
manifest-declared files. All 630 declared bundles load independently. The
Stage 2 regression compares 84 shared candidate-seed runs: Empirical,
cross-position Deduct, and history LightGBM remain numerically exact after
removing measured timing and run-local task pseudonyms; history MLP preserves
the Stage 3 runtime-scoped contract.

The immutable artifact IDs, hashes, paths, cardinalities, and report hash are
frozen in `configs/stage3_release.json`. The release can be rechecked with
`python scripts/verify_stage3_release.py`; CI uses `--tracked-only` to verify the
release controls and historical source tag without requiring private local
artifacts.
