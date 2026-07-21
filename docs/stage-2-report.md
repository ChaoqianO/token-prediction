# Stage 2 development report

## Status

Stage 2 is complete on the sealed development cohorts. The implementation and
all production artifacts are bound to Git commit
`ceecf9cf9cacb9e2aac395a6b29ad3cf130d1592` and code-tree SHA-256
`4e15c9f9b1c1eeeec14b1f22f8db74613591d3b4ecd14255018e9c035cf2c650`.
The permanent final holdout remains sealed in every artifact: it was not used
for fitting, calibration, scoring, prediction, gating, or selection.

This stage delivered:

- label-free prediction points separated from labels;
- Task-pre through Task-update lifecycle sequences with scored and unscored
  context masks;
- permanent task holdout plus three fixed five-fold outer plans and five-fold
  inner OOF initializer plans;
- reloadable cross-position Deduct with offline/shadow ordering parity;
- deterministic CPU Independent MLPs with train-only encoders, explicit
  missing masks, sample weights, and `safetensors` weights;
- config-v2 and immutable compact composite bundles; and
- task-clustered paired comparisons, latency metrics, and exact calibrated
  prediction reload checks.

## Frozen protocol

The split seeds are `20260719`, `20260720`, and `20260721`. All runs and model
families belonging to a task stay in the same permanent, outer, and inner
partition. Every outer-train updater receives only inner-OOF Task-pre seeds;
outer validation receives the five inner models' quantile ensemble. Upstream
quantiles are non-negative and order-repaired but not conformal-calibrated.
Task-update output is calibrated once with task-max conformal at alpha `0.10`.

Unscored missing or censored points remain in lifecycle order and update model
state, but have zero loss and score masks. Labels and suffix events are absent
from `PredictionPoint`, `SessionSeed`, feature encoders, and inference sessions.
The final holdout is assigned before matrix construction and has no prediction
records in this report.

## Source coverage and artifacts

| Source | Development coverage | Experiments | Candidate×seed runs | Artifact |
| --- | ---: | ---: | ---: | --- |
| Spend aggregate | 388 Task-launch tasks | 1 | 12 | `workspace/stage2/experiments/s2-4bfd11bb3e2cb1ea0367` |
| BAGEN Sokoban | 99 launch tasks; 98 scored update tasks | 2 | 36 | `workspace/stage2/experiments/s2-f913d50d680862b0a470` |
| BAGEN SWE, five primary conditions | 10 condition×position cells | 10 | 195 | `workspace/stage2/experiments/s2-25d25ae2874896dd85c1` |
| Spend full OpenHands | 397 tasks; 1,504 runs; 71,360 scored updates | 2 | 39 | `workspace/stage2/experiments/s2-1344b60559f72695ef85` |
| Sokoban compatibility audit | 128 trajectories and historical Stage 1 parity | audit | n/a | `workspace/stage2/experiments/s2-fe1659fe753f635ba4cf` |

The immutable artifact IDs, results hashes, dataset IDs, protocol IDs, matrix
IDs, exact file counts, and source descriptors are frozen in
`configs/stage2_release.json`. Independent verification read and hashed all
12,665 artifact files. Across the four experiment artifacts there are 282
candidate×seed runs. The 195 reloadable runs cover 975 fold bundles and all
reported exact calibrated prediction parity; the other 87 runs are stateless
or mechanical and correctly report reload as not applicable.

The separate Sokoban audit also reloads all 20 historical Stage 1 LightGBM
bundles and reproduces 992 raw predictions with zero mismatch. The historical
source commit remains unrecoverable, so this is regression evidence rather
than a claim that the old run can be retrained from the current repository.

## Development results

All values below are means across the three frozen split seeds on exactly the
same development cohort. MAE is in provider-accounted tokens. These are
development results, not final-holdout estimates.

### Task-launch

| Source / condition | Empirical MAE | Best LightGBM MAE | Best MLP MAE | Lowest MAE |
| --- | ---: | ---: | ---: | --- |
| Spend aggregate, four-run mean | 665,068 | 667,863 | 1,346,757 | Empirical |
| BAGEN Sokoban | 3,921 | 3,918 | 7,649 | LightGBM, negligible margin |
| BAGEN SWE Qwen 3 235B | 310,428 | 302,487 | 559,176 | LightGBM |
| BAGEN SWE GPT-5.2 Instant | 73,397 | 73,278 | 128,333 | LightGBM, negligible margin |
| BAGEN SWE Claude Opus 4.7 | 32,216 | 32,177 | 51,024 | LightGBM, negligible margin |
| BAGEN SWE Gemini 3.1 | 149,122 | 150,610 | 201,039 | Empirical |
| BAGEN SWE Claude Sonnet 4.6 | 57,583 | 57,249 | 81,038 | LightGBM, small margin |
| Spend full OpenHands | 750,500 | 750,230 | 1,377,774 | LightGBM, negligible margin |

The explicit Spend aggregate character-length baseline reached MAE 736,488,
worse than empirical. It is retained as an honestly named pre-request length
baseline and is not presented as token count.

### Task-update provider-accounted remaining tokens

| Source / condition | Empirical | Within-cell Deduct | Cross-position Deduct | Best LightGBM | Best MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BAGEN Sokoban | 2,139 | 2,874 | 3,148 | **2,102** history | 2,976 history |
| BAGEN SWE Qwen 3 235B | 210,166 | 245,295 | 256,328 | **196,297** history | 241,010 history |
| BAGEN SWE GPT-5.2 Instant | 53,900 | 62,984 | 62,305 | **51,390** history | 79,362 history |
| BAGEN SWE Claude Opus 4.7 | 27,390 | 31,997 | 33,331 | **26,839** history | 36,608 history |
| BAGEN SWE Gemini 3.1 | 102,108 | 116,744 | 125,148 | **101,873** structured | 129,228 history |
| BAGEN SWE Claude Sonnet 4.6 | 40,551 | 46,582 | 48,828 | **38,618** history | 52,048 history |
| Spend full OpenHands | 565,303 | 651,638 | 617,178 | **499,085** history | 854,551 history |

Spend full provides the clearest paired result. History LightGBM improves MAE
over empirical by 67,138, 64,940, and 66,576 tokens across the three seeds;
all task-clustered 95% bootstrap intervals exclude zero, with candidate win
probability 1.0. The corresponding intervals for Sokoban and most individual
BAGEN SWE conditions cross zero, so their small apparent improvements are not
treated as stable gains.

Nominal interval coverage is 90%. Task-update coverage is generally
conservative (roughly 94% to 99%), while several small Task-launch cohorts
under-cover. This supports retaining task-max conformal for the Stage 3
baseline and testing calibration alternatives only as controlled Stage 4
ablations.

## Interpretation

Independent MLP is operational, deterministic, safely reloadable, and free of
in-sample seed leakage, but it is a negative modeling result on every reported
primary cohort. The small per-condition task counts do not support the added
capacity under the current features. No pooling claim is made.

Mechanical cross-position Deduct is also not a point-MAE winner here. It is
still the required causal state transition and the exact zero-residual base for
Stage 3. Its value is semantic correctness and online/offline parity, not a
claim of standalone superiority.

History LightGBM is the strongest Stage 2 Task-update comparator, especially
on Spend full. Stage 3 must therefore report GRU residual results against both
history LightGBM and the exact cross-position Deduct base. The MLP remains a
documented negative control rather than being silently removed.

## Verification

Before publication, the repository passed:

- pytest: 438 passed, 3 skipped, 230 subtests;
- unittest: 441 passed, 3 skipped;
- Ruff;
- secret scan with zero findings; and
- wheel build, RECORD inspection, isolated install, import, and CLI smoke test; and
- real-source preflight for Spend aggregate and Sokoban.

After publication, every artifact manifest and results payload was independently
re-read, all source artifacts were bound to the same code-tree SHA-256, all
final-holdout sections were confirmed sealed, and all manifest file memberships
and hashes closed. `scripts/verify_stage2_release.py` reproduces these release
checks from `configs/stage2_release.json` without reading private task identities
or final-holdout labels.
