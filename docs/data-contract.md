# Data contract

## Prediction points

A prediction point is a typed object independent from the event that created it.
Its identity contains the source event, position, and target. Feature/label joins
are always explicit by source event ID; list position is never a join key.

Request-boundary points use the final `request_built` event before an attempt.
Its `event_seq` is the visibility cutoff. A generation point uses a real
`generation_checkpoint`. Feature extraction may consume only the prefix through
that cutoff.

## Targets and the known offset

For logical Call `i`:

```text
call_billable_total_i =
    sum(provider_input_attempt + provider_output_attempt for every billed attempt)

provider_task_remaining_i =
    sum(call_billable_total_j for j >= i)

task_unknown_remaining_i =
    provider_task_remaining_i - request_tokens_local_pre_i
```

Schema v2 exposes `task_provider_accounted_remaining_tokens` directly with
`known_offset_tokens=0`. The separate
`task_unknown_remaining_tokens` target is emitted only when the source declares
a real pre-request `REQUEST_LOCAL_COUNT`; end-to-end budget evaluation then adds
back that exact local count. These targets are never renamed into one another.

Call output targets distinguish:

- `call_billable_total_tokens`: all billed attempt input and output with
  `known_offset_tokens=0`;
- `call_unknown_billable_tokens`: all billed attempt input/output minus the
  current local request count; adding the known offset exactly reconstructs the
  logical Call cost, including retry input;
- `call_billable_output_tokens`: output from every billed retry attempt;
- `final_response_output_tokens`: output from the final successful attempt.

Provider-telemetry datasets default to `call_billable_total_tokens`. The local
unknown target is available only for sources with a genuine pre-request count.
An output-length or final-response study may select the other targets, but must
not compare them as if they were identical.

## Independent target validity

Each target value has one status:

```text
observed | censored | missing | invalid
```

A timeout may censor Task remaining cost while completed Call labels stay
observed. Missing future usage invalidates Task suffix labels but does not erase
an already complete Call. Unknown abort reasons fail closed as censored; only
explicit natural termination classes produce exact Task labels.

## Availability

These values are different facts:

```text
request_tokens_local_pre       visible at request_built; feature/known offset
provider_input_tokens_post     visible after response; ledger fact
```

Provider usage, current output length, finish reason, future tools, final
outcome, total Calls, and true total tokens are forbidden as current features.
Missing request counts remain `None`; they are never converted to zero.

The BAGEN adapters record provider input usage only on the response terminal as
`provider_input_tokens_post_response_audit`. Their request boundary always has
`request_tokens_local=None`, and they do not advertise `REQUEST_LOCAL_COUNT`.
Consequently schema-v2 capability gating cannot create either legacy local
unknown target, its known offset, or a prefix token-length feature from this
proxy. Historical schema-v1 proxy experiments remain sensitivity evidence only.
Missing attempt usage still remains `None` and is never inferred from aggregate
totals.

Every schema-v2 dataset is bound to a `SourceDescriptor`: source ID and
revision, a relative manifest path and SHA-256, and the stable capability
contract hash. Target availability is decided from that contract before rows
are selected. The schema-v1 builder remains readable solely for frozen Stage 1
verification; new experiments use the capability-enforced builder.

The strongest automated leakage check is prefix invariance: two trajectories
with the same prefix and arbitrary different futures must yield byte-identical
features at the shared point.

Logical Calls are sequential in the current schema. All events for one Call
must be contiguous: a new `request_built` cannot occur while the active Call has
an unterminated attempt, and an older Call cannot emit a late terminal, tool, or
checkpoint event after the next request. A future concurrent-Agent source needs
a per-Call ledger and a schema revision; it cannot silently enter this reducer.

## Token ledger

- Raw provider usage is retained by the source adapter.
- Input and output form the accounted total.
- Cached input, cache-write input, and reasoning output remain separate because
  they may be subsets rather than additional charges.
- Failed attempts with reported usage are billable and enter both labels and
  the next visible cumulative-spend features.
- Tool output is later request input, not an independent token charge.
- Missing usage is unknown, never zero.
- A missing attempt makes only the transition in which the cumulative
  `missing_usage_attempts` counter increases unknown. If two later endpoints
  have the same counter, the earlier unknown amount cancels and the delta of
  cumulative known usage is observable again.
- Provider, resolved model, tokenizer, and normalization versions must be
  recorded before cross-provider experiments are enabled.

## Split, eligibility, and weighting

The split plan is created once from distinct `task_id` values and frozen before
candidate training. Every run, prefix, checkpoint, and continuation of a task
uses the same fold.

Evaluation cells are additionally conditioned on one `condition_id` (resolved
target model, Agent/version, reasoning and relevant runtime configuration).
Baselines are fitted within that condition rather than pooled across models.

For the current request-level targets, a row weight is:

```text
1 / (number_of_runs_for_task * number_of_points_in_trajectory)
```

Thus each task has total weight one within an experiment cell. Call-progress
data will add context/Call-level normalization before it is enabled, preventing
checkpoint-dense long Calls from dominating.

Eligibility is fixed before candidate creation. All methods receive identical
point IDs and sample weights. Prediction errors cannot be handled by dropping a
row.

The *Spend Your Money* aggregate importer creates exactly one Task-launch row
per SWE-bench task for one selected model/Agent condition. Its four-run mean
label is rounded to a token, so it evaluates expected task cost across tasks,
not run-level variance. Model-task combinations are never counted as independent
tasks, and the external self-estimate is isolated in a baseline-only feature
set.

## Calibration and evaluation

For each outer fold:

```text
test        = fold k
calibration = fold (k + 1) mod K
validation  = fold (k + 2) mod K
train       = all other folds
```

Validation tasks are available for model selection and early stopping but are
not used to calibrate intervals. The current calibrator takes a maximum
nonconformity score per calibration task,
then expands intervals using task-level split conformal calibration. Test labels
never reach estimator fitting or calibration.

Every candidate uses the same MAE, MedianAE, P90AE, WAPE, Pearson correlation,
bias, underestimation rate, point coverage, task-simultaneous coverage, interval
score, normalized width, raw quantile diagnostics, latency, and
prediction-overhead implementation. Raw quantile crossing is retained before
validity repair. Artifact comparison is rejected unless dataset, split,
eligibility, position, target, calibration, alpha, and metric suite all match.
