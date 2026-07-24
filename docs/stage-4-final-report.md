# Stage 4 final report

## Status

Stage 4 is complete through the single permanent-final-holdout evaluation. The
development-only selection was frozen before the holdout was opened:

- selection commit: `8b31e252d86e7288ce9d08a0829dc7f5b8bb5270`;
- selection tag: `stage4-final-selection-v1`;
- selection ID:
  `33e9b3fad0e97c271207994ef4cba5fc98d6078381f3232a5c05f45e3dc736ad`;
- selection artifact ID:
  `23d09c228ea9e5c438bfb37f799db96676fa7f815791ad4405bbdae9108e335d`;
- final run ID: `8d8bf46bd54e5ce4a8578418`; and
- final artifact ID:
  `a1f41e7a91d48677fea7b869835a297b5b6073f43d074ba295c727d6e6167287`.

The final artifact contains 29 condition/position/target cells, 435 frozen
ensemble members, and 86,335 scored predictions. Final target values were used
only for scoring. They were not used for fitting, calibration, gating,
selection, or post-open model changes. The one-time ledger is published and
closes over all 29 cells.

## Frozen development protocol

The permanent holdout was selected first by task hash. All runs and model
families for one task stayed together. Development then used five task-grouped
outer folds, five task-grouped inner folds, and split seeds `20260719`,
`20260720`, and `20260721`.

Task-update initializers propagated only non-negative, order-repaired,
uncalibrated inner-OOF forecasts. Calibration was applied once at the final
output. Each selected final cell is the mean of the exact 15 frozen members
from three split seeds and five outer folds; no selected learned model was
refit after selection.

Four immutable Stage 4 development artifacts contain 45 experiments, 372
candidate-seed runs, and 62 explicit gates:

| Source | Experiments | Candidate-seed runs | Gates |
| --- | ---: | ---: | ---: |
| Spend aggregate | 3 | 15 | 9 |
| BAGEN Sokoban | 6 | 51 | 7 |
| BAGEN SWE, five primary conditions | 30 | 255 | 39 |
| Spend full OpenHands | 6 | 51 | 7 |
| **Total** | **45** | **372** | **62** |

Primary results remain condition-specific. Cross-condition pooling was not
used for fitting or primary selection. Four additional BAGEN behavior
conditions had only three to five development tasks and therefore remained
explicitly gated below the frozen five-fold minimum.

## Development selection and ablations

The frozen selection contains 27 history LightGBM cells, one empirical
Task-launch cell, and one GRU-without-recurrence lifecycle cell:

- BAGEN Sokoban and all five primary BAGEN SWE conditions use history
  LightGBM for Task-update and all three Call-pre targets.
- Spend aggregate Task-launch uses the empirical fold estimator.
- Spend full Call-pre uses history LightGBM for all three targets.
- Spend full Task-update uses the Stage 3 GRU residual model with hidden-state
  carry disabled.

For each BAGEN Task-update condition, the structured-only, no-progress,
no-tool/error-history, and Independent-MLP alternatives were compared with the
locked history LightGBM reference using paired task bootstrap intervals at all
three split seeds. None satisfied the replacement rule requiring every 95%
confidence-interval upper bound to be below zero. The secondary matched-task
analysis across the five BAGEN families reached the same decision: no feature
deletion or MLP replacement was stable across all seeds.

On Spend full, GRU without recurrence beat history LightGBM for every split
seed. Its paired MAE deltas were approximately -10,521, -16,405, and -15,586
tokens, and all three 95% bootstrap intervals remained below zero. Recurrence
therefore was not retained merely because the model class supported it.

Uncalibrated intervals did not match task-simultaneous coverage within the
frozen 0.02 tolerance on any primary Task cell. Task-max conformal at
`alpha=0.10` was consequently retained. This is a coverage-matched calibration
decision, not an interval-width-only comparison.

Fold-fitted TF-IDF retrieval failed closed because the audited sources do not
provide the required genuine task-text observable. No surrogate text,
cross-fold vocabulary, or inferred embedding was introduced.

## Permanent final-holdout cohorts

| Source | Final tasks | Scored predictions |
| --- | ---: | ---: |
| BAGEN Sokoban | 29 | 625 |
| BAGEN SWE | 14 unique tasks across five condition cohorts | 7,974 |
| Spend aggregate | 112 | 112 |
| Spend full OpenHands | 103 tasks / 411 runs | 77,624 |
| **Total** |  | **86,335** |

Counts above sum predictions across the released cells; the same point can
appear under distinct target identities. In particular, billed-output and
final-response-output labels are numerically equal on these audited cohorts,
but remain separately identified and were never mixed during fitting,
calibration, or reporting. The BAGEN SWE holdout contains 14 canonical tasks;
individual scored cells contain 12 to 14 because condition availability,
missing usage, and censoring remain target-specific.

The immutable final artifact's source-level `datasets` summary records 13 for
BAGEN SWE because that field was populated from the first scored cell rather
than from the source holdout plan. This is a metadata-only semantic defect:
the dataset ID binds the correct holdout, every cell was filtered by the
14-task frozen set, and cell-level task and prediction counts are exact. The
tracked release contains a machine-readable amendment binding the artifact
value 13 to the authoritative 14 `final_holdout` assignments in the frozen
development protocol. The artifact is not rewritten and the final evaluation
is not rerun.

## Final Task results

MAE is in tokens. Point coverage and task-simultaneous coverage are for the
released 90% interval after the single calibration application.

| Source / condition | Selected candidate | Development MAE | Final MAE | Final points | Point coverage | Task coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BAGEN Sokoban | LightGBM history | 2,102 | 2,261 | 118 | 96.32% | 85.71% |
| BAGEN SWE Qwen 3 235B | LightGBM history | 196,297 | 347,445 | 425 | 88.60% | 83.33% |
| BAGEN SWE GPT-5.2 Instant | LightGBM history | 51,390 | 56,797 | 339 | 97.79% | 92.86% |
| BAGEN SWE Claude Opus 4.7 | LightGBM history | 26,839 | 40,101 | 103 | 94.64% | 92.86% |
| BAGEN SWE Gemini 3.1 | LightGBM history | 102,037 | 90,769 | 250 | 96.61% | 92.31% |
| BAGEN SWE Claude Sonnet 4.6 | LightGBM history | 38,618 | 23,705 | 161 | 100.00% | 100.00% |
| Spend aggregate Task-launch | Empirical | 665,068 | 619,041 | 112 | 89.29% | 89.29% |
| Spend full Task-update | GRU, no recurrence | 484,915 | 467,820 | 18,740 | 99.46% | 92.23% |

The holdout does not support a pooled BAGEN claim. Qwen and Opus exhibit
material development-to-final MAE increases, while Gemini and Sonnet improve.
This heterogeneity is why the protocol froze per-condition reporting and
prohibited post-open reselection. Qwen task-simultaneous coverage is also below
the nominal target on only 12 scored final tasks; the result is reported as
observed rather than repaired after opening the holdout.

Spend full confirms the development decision without further tuning:
no-recurrence GRU final MAE is 467,820 on 18,740 scored boundaries, with 99.46%
point coverage and 92.23% task-simultaneous coverage.

## Final Call-pre results

All released Call-pre cells use history LightGBM. The table reports final MAE
for billed output, billed total, and final response output respectively.

| Source / condition | Output MAE | Total MAE | Final-response MAE | Points per target |
| --- | ---: | ---: | ---: | ---: |
| BAGEN Sokoban | 155 | 154 | 155 | 169 |
| BAGEN SWE Qwen 3 235B | 60 | 1,609 | 60 | 942 |
| BAGEN SWE GPT-5.2 Instant | 55 | 421 | 55 | 353 |
| BAGEN SWE Claude Opus 4.7 | 73 | 797 | 73 | 185 |
| BAGEN SWE Gemini 3.1 | 102 | 2,327 | 102 | 442 |
| BAGEN SWE Claude Sonnet 4.6 | 51 | 1,379 | 51 | 310 |
| Spend full OpenHands | 374 | 1,374 | 374 | 19,628 |

Across these cells, final point coverage ranges from 95.54% to 100.00%.
Task-simultaneous coverage ranges from 85.71% to 100.00%. The intervals are
often conservative; no narrower calibration was substituted after final
coverage became visible.

## Spend lifecycle diagnostics

The shared offline/shadow lifecycle driver replayed 411 final Spend runs in
observe-then-predict order. It scored 392 observed-termination runs and 18,740
boundaries. Seventeen `task_error` runs contributed 477 context-only
boundaries, and two additional missing-label runs remained unscored. No
censored Task-remaining MAE was fabricated.

Final GRU-no-recurrence MAE decreases with lifecycle progress:

| Progress checkpoint | MAE |
| --- | ---: |
| 25% | 646,133 |
| 50% | 443,106 |
| 75% | 256,657 |

All 103 final Spend tasks have repeated scored runs, so run variance remains
estimable. For remaining-token overrun decisions, final accuracy / precision /
recall are 98.72% / 99.27% / 99.44% at 16,384 tokens; 97.50% / 98.22% /
99.25% at 32,768; 95.10% / 96.16% / 98.80% at 65,536; and 90.99% / 92.18% /
98.27% at 131,072. The calibrated intervals are conservative enough that
almost all budget decisions remain interval-uncertain; point-decision accuracy
must not be misrepresented as a definite interval decision. Actual overrun
prevalence is already 89.45% to 99.27% across these scenarios, so the high
accuracy values alone do not establish reliable automated termination.

## Telemetry and online surfaces

Task lifecycle, Call-pre, and the shared online-shadow interface are available
for the full trajectory sources because request boundaries and attempt usage
are observed. Aggregate Spend remains Task-launch only.

Call-update and the G3 generation-progress, entropy-stop, hidden-state, and
resumable-state surfaces remain fail-closed. The audited sources do not expose
true output deltas, log probabilities, hidden state, or resumable checkpoints.
The implementation publishes capability decisions and status reasons instead
of inferring those internal states from text. No paid live shadow run was
performed.

## Artifact and privacy verification

The final results payload SHA-256 is
`ee0a27184b10068720fdc194471df9268a0fb5bc94311692432bbe7a3ccd4b34`.
The artifact manifest, selection lock, selection payload, all source-artifact
bindings, 435 ensemble-member identities, calibrators, and bundle directory
hashes are release-locked.

The release verifier independently loads every selected LightGBM and lifecycle
bundle from safe non-pickle formats. It validates the 15-member execution
contract for every cell and the published 29-cell ledger without rereading or
rescoring final labels. Reload verification is therefore separate from the
one-time evaluation. The one-time evaluation itself generated every released
prediction through those strictly reloaded bundles; the release verifier does
not claim a second trajectory replay.

## Historical execution-control amendment

The frozen final runner's original code-tree field intrinsically binds 71
tracked paths, but its explicit path list omitted four imported experiment/data
loaders and five tracked data-control files. The final artifact is not
rewritten. Instead, the release lock carries a machine-readable provenance
amendment reconstructed from the frozen selection commit's Git blobs. The
amended executed closure binds 80 paths, with code-tree SHA-256
`ed201c1ba0d0ba475c13bb003d683fbb73ffa2604ae294a643f443ae9fc6d6ca`.
The verifier also proves that the runtime code and data controls, other than
the subsequently added selection lock itself, did not change between the
selection-construction commit and the tagged final-selection commit.

The contemporaneous execution was observed with a clean tracked worktree, one
final process (PID 4984), the canonical output and checkpoint roots, and one
published ledger covering 29 checkpoints. The current-state verifier confirms
that those roots still contain exactly one final artifact and one checkpoint
run. This is strong release evidence, but it is an explicit post-execution
provenance amendment: the historical runner did not itself hold an OS-level
exclusive lock or prove every imported module origin. Those properties cannot
be added retroactively without reopening the holdout, which is prohibited.
The released runner now rejects alternate roots, uses a fixed cross-process
lock and persistent tombstone, checks ledger/checkpoint agreement, binds the
full tracked execution closure, and refuses execution after the tracked
release exists. GitHub ruleset `19652329` additionally blocks update and
deletion of the four Stage 2-4 source/selection tags and the final release tag
with no bypass actors, so remote immutability is enforced rather than merely
checked after drift.

Public final results contain only run-local task pseudonyms. No raw task,
trajectory, run, or instance identity is present. Private source archives,
canonical joins, checkpoints, and model artifacts remain ignored workspace
data and are not committed. These values are pseudonyms rather than an
anonymity guarantee: because their inputs are public and unkeyed, an enumerable
task universe could permit dictionary re-identification.
