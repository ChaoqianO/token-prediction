# Engineering stage delivery ledger

This ledger supplies the Git/PR handoff fields for the PR-oriented engineering
stages.  The scientific reports remain byte-frozen because their hashes are
part of the corresponding release locks.  Stage numbering is mapped in
[roadmap.md](roadmap.md).

All four delivery PRs targeted `main`, used squash merge after the required
Python 3.11/3.12 quality and wheel checks were green, and were handled by the
repository's configured merge automation.  The current `main` commit is
`32712610eb0a6305bcd808d1977af6ee5f3237dd`; CI run `30062597198` completed
successfully for that exact commit.

## Data Foundation / PR-0

| Field | Frozen value |
| --- | --- |
| Branch | `pr-0-data-foundation` |
| PR | [#1](https://github.com/ChaoqianO/token-prediction/pull/1) |
| Merged | 2026-07-21; squash commit `4d08e715d9361fa634b26cfcc909a45b5cecd6b9` |
| Auto-merge status | Completed after required checks passed |
| Production source commits | canonical audit `3d3edf3fdbd870300260a690b48313ce225c33f0`; prediction runner `8609d4b1592579b0b5e36a3ca3dbb3f79e005217` |
| Protected source head | `9e00f6b025924b7c53038faad5e6e5775dcb8776` |
| Run/artifact | empirical development v3; artifact `f2e8587551d3077768e248a85c634a3a1d429baf3f5f5a5256d12e4c23302393` |
| Evidence | [data-foundation-report.md](data-foundation-report.md), `configs/data_foundation_v2_baseline.json`, `configs/data_foundation_prediction_baseline.json` |

The stage closed the BAGEN/Spend handoff, source and capability identities,
proxy-free targets, deterministic canonical builds, and a commit-bound
development baseline.  Historical Stage 1 inference remains exactly
reloadable, but its original source tree is unrecoverable and is not presented
as retrainable.  Stage 2 was allowed to begin only after the source audit,
reader tests, raw-data isolation, frozen hashes, baseline verifier, CI, and
independent data-contract review passed.

## Engineering Stage 2 / PR-1

| Field | Frozen value |
| --- | --- |
| Branch | `pr-1-stage-2` |
| PR | [#2](https://github.com/ChaoqianO/token-prediction/pull/2) |
| Merged | 2026-07-21; squash commit `9902df0ed36ca52e50e18b570bea94cbaecc4e3f` |
| Auto-merge status | Completed after required checks passed |
| Artifact source commit | `ceecf9cf9cacb9e2aac395a6b29ad3cf130d1592` |
| Run IDs | `4bfd11bb3e2cb1ea03674426`, `f913d50d680862b0a47089a4`, `25d25ae2874896dd85c150cd`, `1344b60559f72695ef85a5c7`, compatibility audit `fe1659fe753f635ba4cf6e37` |
| Release | 5 artifacts; 282 candidate-seed runs; 975 exact reload folds |
| Evidence | [stage-2-report.md](stage-2-report.md), `configs/stage2_release.json` |

This stage delivered label-free points, lifecycle masks, grouped permanent and
5×5 nested splits, inner-OOF initialization, cross-position Deduct, and safe
reloadable Independent MLP bundles.  Independent MLP and mechanical Deduct were
negative point-MAE results on the primary cohorts; those results were retained.
Stage 3 began only after leakage, lifecycle parity, deterministic reload,
three-seed, full-test, and immutable-artifact gates passed.

## Engineering Stage 3 / PR-2

| Field | Frozen value |
| --- | --- |
| Branch | `pr-2-stage-3` |
| PR | [#3](https://github.com/ChaoqianO/token-prediction/pull/3) |
| Merged | 2026-07-23; squash commit `895ea321c3caa5dd73a68ecd254542c60073e293` |
| Auto-merge status | Completed after required checks passed |
| Artifact source commit | `d3767c135c255d3803195573130f8bb0aefe0d67` |
| Run IDs | gate `0eadef35bf584fcb1ee7f198`; experiments `b0231bd18f6af59bb6e808d9`, `b35d4daecbeaa016a00ae0ba`, `6c57b8ef3acc736cceea2608` |
| Release | 4 artifacts; 147 candidate-seed runs; 420 exact lifecycle reload folds; 630 independently loaded bundles |
| Evidence | [stage-3-report.md](stage-3-report.md), `configs/stage3_release.json` |

This stage delivered the shared offline/shadow lifecycle driver, GRU residual,
no-recurrence and exact zero-residual paths, epoch checkpoints, and progress,
termination, budget, latency, and repeated-run diagnostics.  Recurrence was
not a universal improvement: history LightGBM remained best on BAGEN, while
the no-recurrence GRU won on Spend.  Stage 4 began only after complete
trajectory reload, no-teacher-forcing, zero-residual, same-cohort, three-seed,
termination-stratification, CI, and independent artifact/scientific reviews
passed.

## Engineering Stage 4 / PR-3

| Field | Frozen value |
| --- | --- |
| Branch | `pr-3-stage-4` |
| PR | [#4](https://github.com/ChaoqianO/token-prediction/pull/4) |
| Merged | 2026-07-24; squash commit `32712610eb0a6305bcd808d1977af6ee5f3237dd` |
| Auto-merge status | Completed after required checks passed |
| Selection commit/tag | `8b31e252d86e7288ce9d08a0829dc7f5b8bb5270`; `stage4-final-selection-v1` |
| Selection | run `feb2b40cb2cdf6387e26d430`; artifact `23d09c228ea9e5c438bfb37f799db96676fa7f815791ad4405bbdae9108e335d`; selection `33e9b3fad0e97c271207994ef4cba5fc98d6078381f3232a5c05f45e3dc736ad` |
| Final | run `8d8bf46bd54e5ce4a8578418`; artifact `a1f41e7a91d48677fea7b869835a297b5b6073f43d074ba295c727d6e6167287` |
| Final release tag | `stage4-final-release-v1` |
| Evidence | [stage-4-final-report.md](stage-4-final-report.md), `configs/stage4_selection.json`, `configs/stage4_release.json` |

Development-only selection used 372 candidate-seed runs and 62 explicit gates.
The permanent holdout was then evaluated exactly once: 29 cells, 435 ensemble
members, and 86,335 scored predictions.  All selected bundles reload, and the
one-time ledger closes all cells.  Condition heterogeneity, conservative
intervals, unavailable genuine task text, absent generation checkpoints/G3
observables, and the lack of a separately authorized paid live-shadow run are
reported as limitations rather than repaired after holdout opening.

There is no later engineering stage to enter.  The remaining completion
decision is whether the telemetry-dependent paid live-shadow surface is
accepted as gate-only or receives separate execution authorization.

## Repository controls

`main` requires the four Python 3.11/3.12 quality and wheel checks, applies the
rules to administrators, and rejects force pushes and deletion.  Ruleset
`19652329` prevents update or deletion of the five Stage 2–4 source,
selection, and final-release tags with no bypass actor.  Ruleset `19656197`
likewise prevents update or deletion of the Data Foundation source branch with
no bypass actor.  Raw data, archives, caches, checkpoints, learned artifacts,
credentials, and local state remain outside Git.
