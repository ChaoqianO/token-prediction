# Trajectory data audit and handoff

Status: BAGEN SWE-bench and the Spend Your Money GPT-5.2 four-run archive are
frozen, fully audited, and reproducible. Both canonical trajectory adapters
have been built into supervised datasets twice with identical hashes.

All paths below are relative to the repository root. Raw archives,
trajectories, inventories, and derived datasets remain under `workspace/`,
which is ignored by Git. No raw prompt, response, task ID, message, tool
argument, evaluator artifact, or authentication material is reproduced here.

## 1. Source pins, download scope, and storage

### 1.1 BAGEN

- Hub repository: `MLL-Lab/BAGEN`
- Repository page: <https://huggingface.co/datasets/MLL-Lab/BAGEN>
- Resolved revision: `58189576e54b675fdd0e1d6c1c9f189c2992732f`
- Published manifest URL:
  <https://huggingface.co/datasets/MLL-Lab/BAGEN/resolve/main/manifest.jsonl>
- Pinned object form used by the downloader:
  `https://huggingface.co/datasets/MLL-Lab/BAGEN/resolve/58189576e54b675fdd0e1d6c1c9f189c2992732f/<path>`
- Local manifest: `workspace/external/bagen/manifest.jsonl`
- Manifest bytes: `228,164`
- Manifest SHA256:
  `f5900dead3a32ca303d500f123ee96b89e6797527cbb99fef0cd9beaf2a00071`
- Manifest Git-blob ETag (not a SHA256):
  `8a7f701692d90bf17b719220431c4b02ba14e780`
- Manifest summary: `workspace/external/bagen/manifest_summary.json`
- Manifest-summary SHA256:
  `97b602c1d1456bae407a442a4a1cb5267cb4a11e3aa38701a5962b3e4063c68b`

The manifest describes 445 files and 3,767,908,206 bytes. Its `origin/`
partition contains 393 files and 1,015,191,613 bytes; `estimation/` contains 52
files and 2,752,716,593 bytes. There are 316 individual `*.traj.json` files
totalling 263,785,722 bytes. The largest manifest object is the unneeded
1,085,402,003-byte SERA estimation JSON; it was not downloaded.

The selected five-family download is:

| Family | Local root | Downloaded scope | Files | Bytes | Trajectory audit |
| --- | --- | ---: | ---: | ---: | --- |
| GPT-5.2 Instant | `workspace/external/bagen/origin/swebench-origin-gpt5.2instant/` | complete family prefix | 69 | 9,295,671 | `workspace/external/bagen/audits/gpt5.2instant.json` |
| Claude Opus 4.7 | `workspace/external/bagen/origin/swebench-origin-claude-opus4.7/` | `*.traj.json` only | 64 | 6,706,297 | `workspace/external/bagen/audits/claude-opus4.7.json` |
| Claude Sonnet 4.6 | `workspace/external/bagen/origin/swebench-origin-claude-sonnet4.6/` | `*.traj.json` only | 64 | 10,239,153 | `workspace/external/bagen/audits/claude-sonnet4.6.json` |
| Qwen3 235B | `workspace/external/bagen/origin/swebench-origin-qwen3-235b/` | `*.traj.json` only | 60 | 34,362,584 | `workspace/external/bagen/audits/qwen3-235b.json` |
| Gemini 3.1 | `workspace/external/bagen/origin/swebench-origin-gemini3.1/` | `*.traj.json` only | 64 | 203,340,963 | `workspace/external/bagen/audits/gemini3.1.json` |

The five selected roots contain 321 raw files and 263,944,668 bytes. Including
the manifest, this branch downloaded 264,172,832 bytes. Every one of the 316
trajectory paths and sizes matches the pinned manifest. For each trajectory,
the audit's `raw_files` and `source_hashes` fields are the authoritative
per-file path/byte/SHA256 ledger; the task IDs and file names are intentionally
not duplicated in this document.

The five non-trajectory files included because GPT-5.2 Instant was downloaded
as a complete prefix are:

| Relative path below `workspace/external/bagen/` | Bytes | SHA256 |
| --- | ---: | --- |
| `origin/swebench-origin-gpt5.2instant/gpt52_instant/exit_statuses_1776975976.9557607.yaml` | 186 | `91a346133e434ae3be5e5914b94711e8dfea294300ae36cb03901ebc3b315caf` |
| `origin/swebench-origin-gpt5.2instant/gpt52_instant/exit_statuses_1776994471.2214096.yaml` | 1,692 | `819e86b43382552b788be275602597750a5909ddf563cddf3e6362735e29726a` |
| `origin/swebench-origin-gpt5.2instant/gpt52_instant/minisweagent.log` | 46,367 | `e8e28ab1837db8ba47397b2c5789e12156d1a47ca3b23403a72c00119abe2064` |
| `origin/swebench-origin-gpt5.2instant/gpt52_instant/preds.json` | 103,926 | `7a8033391d92a075aa7a0f0b7a7dd08bb2ff7f10e75581d7cf2a6f8276f594f5` |
| `origin/swebench-origin-gpt5.2instant/openai__gpt-5.2.gpt52_instant.json` | 6,775 | `df519f89790494fa147583884f3ccc193697a92681108ed95989dc4abd49f291` |

The audit-file hashes are:

| Audit | SHA256 |
| --- | --- |
| `claude-opus4.7.json` | `747946f905440cc02393fe229a429b538f85ff4f09e5ac7a25cb6d3ba6121c2d` |
| `claude-sonnet4.6.json` | `fbe5efa634bd6132c7ddefe64ce91268a164421471a1cbdee410dff52dc6458b` |
| `gemini3.1.json` | `9d39933cd41952a250b1c89968f94b3ec6dd4e6b462135d094a98d10cca1ec74` |
| `gpt5.2instant.json` | `a07a6d8c408ec6f1daa136c14606e7457b8e7339ebc1da7b7d7b9cef9a07430c` |
| `qwen3-235b.json` | `dc1c77394449996af48a8bf1dd0d32e3cc35ecfe08108b06a0e79c3f3141a3de` |

The cross-family freeze is
`workspace/external/bagen/combined_swebench_audit.json` (139,686 bytes; file
SHA256
`2d8f3abe10b526f80488554d672039c9f9bc81b31e230b7bb6b14c94b0ffaea5`;
payload SHA256
`f144f38fb7b7ab85c25ff859f6be59f5a9d316871a6ebfa5311ffc9a8730d17a`).
It is the authoritative task-to-family/run/condition mapping and verifies all
316 raw-file hashes against the five family audits and the pinned manifest.

Everything under `estimation/`, other BAGEN benchmarks/model families, and the
non-trajectory auxiliary files under the four B3 families remains
undownloaded. A previously present Sokoban source is retained separately at
`workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json` (15,557,656
bytes; SHA256
`c4f7c73c35b741b17093fd3136017baaa0069d2d5108d0792597801b162bad12`);
it is not part of the five-family SWE-bench freeze above.

### 1.2 Spend Your Money / OpenHands GPT-5.2

- Hub repository: `loong0814/openhands_trajectories`
- Repository page:
  <https://huggingface.co/datasets/loong0814/openhands_trajectories>
- Resolved revision: `fa9cbb063f770df596da95af24f7af3b8f595778`
- Pinned archive URL:
  <https://huggingface.co/datasets/loong0814/openhands_trajectories/resolve/fa9cbb063f770df596da95af24f7af3b8f595778/gpt_5.2_4runs.tar.gz>
- Local archive:
  `workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz`
- Compressed bytes: `2,908,192,516`
- Archive SHA256:
  `993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214`
- Xet ETag (not a SHA256):
  `5824153171526bdfb245b74fca532407cf68add02079b4fa0f7c1cf47ea1c1c8`
- Inventory: `workspace/external/spend_your_money/gpt_5.2_inventory.json`
- Inventory schema: `2`
- Inventory SHA256:
  `11735ca3ec625a21c57f72d8172cdc0e0e67fbc44ee99225366739e5dbfffc24`
- Canonical trajectory audit:
  `workspace/external/spend_your_money/gpt_5.2_trajectory_audit.json`
- Trajectory-audit bytes: `1,178,620`
- Trajectory-audit file SHA256:
  `6713db7b197899153cd8d783343dea7056aa2edcc6a43ba317498f38f6e0743c`
- Trajectory-audit payload SHA256:
  `4cd7f93d3ae7c0e6d49d65a7d5d640d409c8a5f3bd0708f8ae8f6da9d387571e`

The archive is not extracted. The streaming inventory saw 110,672 members,
106,730 regular files, and 13,536,029,869 logical uncompressed regular-file
bytes. The single compressed archive is the raw integrity unit; per-member
formal JSONL SHA256 values are recorded in the inventory, while the trajectory
audit records canonical per-trajectory hashes. No second archive or
extracted tree is retained.

Every task is present in exactly four runs at inventory level:

| Run ID | Archive-member root below `gpt_5.2_4runs/` | Tasks | Completion snapshots | Completion bytes | Task reports | Missing completion sets | Missing task reports |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_1` | `gpt-5.2_maxiter_500_N_v0.62.0-no-hint-run_1/` | 500 | 23,144 | 2,980,971,311 | 479 | 0 | 21 |
| `run_2` | `gpt-5.2_maxiter_500_N_v0.62.0-no-hint-run_2/` | 500 | 23,028 | 2,945,345,010 | 473 | 3 | 27 |
| `run_3` | `gpt-5.2_maxiter_500_N_v0.62.0-no-hint-run_3/` | 500 | 23,437 | 2,988,678,867 | 474 | 0 | 26 |
| `run_4` | `gpt-5.2_maxiter_500_N_v0.62.0-no-hint-run_4/` | 500 | 25,505 | 3,237,150,748 | 486 | 0 | 14 |

The four run-level formal-file SHA256 values are useful independent checks:

| Run | `output.jsonl` SHA256 | `output.swebench.jsonl` SHA256 |
| --- | --- | --- |
| `run_1` | `9287526de04694490d1b4cb6a79c7b8177eed3b9dafcf76c707fae14e998b14e` | `b966b2a66141374da26ae38f824bc884f55438ae235475927dc6817b4589a5d7` |
| `run_2` | `ee5cb1ece66ec9744e8e8682fcdd9522c6d8c169ad78026622f961549fa25285` | `161638590ba5018f4e487b9c0eac8387ff7050a8908ff3e378fd2937d6c7b72f` |
| `run_3` | `3706cc6a5ef55be243b9b9d28afb103b172ed35e64879aa802b5806806c3a3be` | `539078f4ee4a9221f0ba32afed79350219f2539837b78742263ca5fb546856f6` |
| `run_4` | `6e67abc55c426cb56cab5c0dfb743203a745d0f866f7f8f86fa03965d667f16c` | `80fb6d236a45db9c2948b07575fde57ba9c1633d9c218e524f56a0915cef909b` |

Two pre-existing aggregate sidecars remain separate from the archive and are
not substituted for trajectory telemetry:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `workspace/external/spend_your_money/all_models_averaged_predictions_new.csv` | 305,265 | `01839b0be787eeb00aec7661c0f7f7515e6c26b6cded8526b272e460510cde4b` |
| `workspace/external/spend_your_money/swe_bench_verified_test.parquet` | 2,096,679 | `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd` |

The archive transfer was allowed only after the E: drive exceeded the 20 GiB
free-space gate. At the final audit observation on 2026-07-21, E: had
276,505,001,984 bytes (257.515 GiB) free. This is an observation, not a future
space guarantee.

## 2. Schema and normalization

### 2.1 BAGEN `*.traj.json`

The verified top-level structure contains `instance_id`, `info`, `messages`,
and trajectory-format metadata. `info.config` preserves Agent, environment,
model, provider, version, and step-limit configuration. Messages use the
`system`, `user`, `assistant`, `tool`, and final `exit` roles.

The reader in `src/token_prediction/collection/bagen_swebench.py` parses one
top-level message at a time and does not materialize a large Gemini file with
`Path.read_text()` or `json.load()`. It emits one logical call and one attempt
for each assistant completion or recorded `FormatError` recovery interruption.
Provider response usage is attached to the matching terminal attempt. Tool
results are matched by call ID; because no trustworthy start timestamp exists,
only terminal `tool_completed`/`tool_failed` events are emitted. An exit can
close one pending final-submit tool.

`Submitted` is a real task finish. `LimitsExceeded` is tied to the configured
250-step limit and becomes `task_aborted(reason=max_turns)`, so task-suffix
labels are censored rather than completed. All canonical timestamps are
synthetic ordering timestamps and must not be used as latency measurements.

### 2.2 Spend OpenHands archive

The verified archive templates are:

```text
<wrapper>/<run>/llm_completions/<task_id>/<completion_file>.json
<wrapper>/<run>/output.jsonl
<wrapper>/<run>/output.swebench.jsonl
<wrapper>/<run>/eval_outputs/<task_id>/report.json
<wrapper>/<run>/report.json
```

Each completion object contains request `messages`, a current `response`,
`args`, `kwargs`, a top-level timestamp, and cost metadata. The response holds
one choice and current-response usage. Request snapshots repeat history; they
are not additive token ledgers. The reader sorts snapshots within a task by the
independently checked filename timestamp, top-level timestamp,
`response.created`, and response ID, then validates history-prefix extension.
Physical tar order is not treated as chronology.

The four `output.jsonl` files provide task history, accumulated metrics, task
errors, and explicit action/observation pairs. They contain 2,000 records. At
inventory level, the history/metrics/usage sections validate as 1,975 observed
and 25 censored records; those are formal-record validation statuses, not the
final prediction-label matrix. Tool failures are observable only where the
explicit history action/observation carries failure evidence. Completion-only
fallback tool results prove a terminal tool result but not success versus
failure.

Task-level `eval_outputs/<task_id>/report.json` and run-level `report.json` are
evaluator outcomes, not Agent lifecycle records. The inventory found 1,912
task reports, of which 1,257 are resolved. Evaluator-accuracy status is 1,912
observed, 79 missing, and 9 invalid. In particular, the 743 `false` values in
the derived `output.swebench.jsonl` `resolved` field are not 743 observed
failures: a derived false can represent a missing report, empty patch, or
evaluator error. Reports and termination must remain separate joins.

The inventory validated all formal JSONL schemas and a deterministic five-task
sample of completion/report schemas. It intentionally reports
`full_completion_schema_validated=false` and
`trajectory_ingestion_ready=false` because an inventory alone cannot close
those gates. The separate full canonical reader/audit closes trajectory
ingestion without rewriting that historical inventory result.

## 3. Identity and cross-source mapping

No raw task IDs are copied into this audit. Machine-readable mapping stays in
the ignored audit files.

### 3.1 BAGEN

- Unique `task_id`: 64, exactly path-validated against `instance_id`.
- Unique canonical `trajectory_id`: 316.
- Unique `run_id`: 316. A BAGEN source trajectory is one run, and its run ID is
  the stable relative source path beginning at `swebench-origin-*`.
- Unique `condition_id`: 9.
- Cross-family mapping: 60 tasks have five trajectories, one in every family;
  four tasks have four trajectories because Qwen contains 60 rather than 64
  tasks.
- Condition mapping: GPT-5.2 Instant, Opus, Sonnet, and Gemini each contain two
  behavior conditions; Qwen contains one. Conditions hash provider,
  configured model/family, Agent type/version, and behavior-relevant
  configuration. They are not outcome hashes.

The authoritative mappings are `task_trajectory_counts`,
`condition_task_counts`, `condition_count`, and `raw_files` in
`workspace/external/bagen/audits/*.json`. Experiments must keep all families
and runs for one task in the same task split, while fitting baselines within a
single condition.

### 3.2 Spend

Canonical identity is complete: 500 unique `task_id` values, four `run_id`
values, 2,000 unique `trajectory_id` values, and one `condition_id`
(`condition:b407e0d1ec34f386ebc4`) covering all 2,000 trajectories. Every task
maps one-to-one to exactly `run_1`, `run_2`, `run_3`, and `run_4`; each run has
500 trajectories under the same condition. The full task/run/trajectory join
is stored, not inferred by dividing an aggregate by four.

Three `run_2` task-runs have no completion snapshot. They still have formal
output records and source-reported zero accumulated usage. The reader emits
zero calls for those trajectories and accepts task usage zero only under the
audited `explicit_zero_call_task` rule; this is not zero imputation.

The canonical Spend mapping is in
`workspace/external/spend_your_money/gpt_5.2_trajectory_audit.json` under
`counts`, `run_ids`, `condition_counts`, and `task_run_mapping`.

## 4. Telemetry capability audit

Both readers declare the independent observables `task_usage`, `call_usage`,
`attempt_usage`, `request_messages`, and `tool_events`. Neither declares
`request_local_count` or output deltas. A declared observable means the source
can carry that fact; it does not mean every record is complete.

| Fact | BAGEN SWE-bench | Spend OpenHands |
| --- | --- | --- |
| `request_tokens_local` | No genuine local count. Provider input is copied only as a tagged `provider_input_proxy`; `REQUEST_LOCAL_COUNT` is not declared. | Always `None`; archive contains no independent local tokenizer count. |
| Attempt usage | 9,641 complete and 1,671 missing attempt-usage records. Missing remains unknown. | 95,114 complete, 0 missing, and 0 invalid current-response usage records. |
| Logical calls / attempts | 11,312 calls and 11,312 adapted attempts. | 95,114 stored completions become 95,114 logical calls and 95,114 terminal attempts. |
| Retry | 1,671 `FormatError` recovery calls under that exact audit definition. Within-call/provider-transport retry telemetry is unsupported; the recorded zero within-call count is not proof that providers never retried. | Provider-transport retry ledger is not preserved. A stored completion is not evidence about unseen retries. |
| Attempt errors | Recorded `FormatError` becomes `api_failed`; status code and usage may be missing. | All stored completions become `api_completed`; `api_failed=0`. The 228 preserved provider error envelopes remain completed-response metadata and are not promoted to transport failures or retries. |
| Tools | 9,825 terminal events; 1,524 explicit failures, primarily from return codes. No synthetic starts. | 88,992 starts, 79,731 completions, and 9,261 explicit failures. All 88,992 frozen terminal tool events have explicit failure observability; no failure-unobservable fallback was needed in this archive. |
| Task errors | Exit status and exception fields are distinct from tool/attempt errors. | 104 top-level `output.jsonl` task errors are distinct from API errors and evaluator failures. |
| Task termination | 303 `Submitted` finishes; 13 step-limit aborts (`max_turns`) and are censored. | 1,896 explicit finishes and 104 task-error aborts; all 2,000 lifecycle records are observed. Evaluator reports do not prove termination. |
| Generation checkpoint | None. No output delta is synthesized. | None. No streaming delta/checkpoint exists in the archive. |

Spend has 0 observed and 95,114 missing local request counts, and 0 generation
checkpoints. Task usage is complete for all 2,000 trajectories: 1,892 use the
current output-session aggregate directly, 83 combine that aggregate with
complete preserved completion extras, and 25 are derived from a complete
attempt sum. The direct-aggregate group includes 3 source-reported explicit
zero-call records. Those three must
have zero reported/accounted totals, zero completion snapshots, and
`output.metrics.accumulated_token_usage` as their source; the audit rejects a
defaulted or synthesized zero.

Restart/session evidence is retained separately: 108 trajectories contain
3,288 completion snapshots outside the current output metrics ledger; there
are 110 request-prefix resets, one repeated request snapshot, 115 responses not
materialized in the next request, and three reasoning-subset anomalies. These
facts affect session interpretation but are never backfilled into attempt
events.

## 5. Prediction-position × target capability and eligibility

The schema has five positions and six targets, a 30-cell Cartesian product.
The current builder intentionally emits only the seven typed cells below. The
other 23 combinations are unsupported schema combinations: they emit no rows,
and their absence is not an observed zero-token label.

### 5.1 Frozen BAGEN matrix

Only `observed` rows can enter ordinary supervised training. The three
`*_unknown_*` cells shown as proxy-only use the tagged provider-input proxy
and are eligible only for a separately named offline sensitivity analysis, not
for an online-local-count claim.

| Position | Target | Required evidence | Observed | Censored | Missing | Invalid | Eligibility |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `task_launch` | `task_total_accounted_tokens` | natural termination plus complete task usage | 232 | 13 | 71 | 0 | 232 exact rows |
| `task_pre` | `task_unknown_remaining_tokens` | natural termination, complete suffix usage, request count | 232 | 13 | 71 | 0 | proxy-only; not online eligible |
| `task_update` | `task_unknown_remaining_tokens` | natural termination, complete future suffix, request count | 5,484 | 3,046 | 2,466 | 0 | proxy-only; not online eligible |
| `call_pre` | `call_unknown_billable_tokens` | complete call usage and request count | 9,641 | 0 | 1,671 | 0 | proxy-only; not online eligible |
| `call_pre` | `call_billable_output_tokens` | complete usage for every adapted attempt in the call | 9,641 | 0 | 1,671 | 0 | exact observed subset |
| `call_pre` | `call_final_response_output_tokens` | complete usage on final successful response | 9,641 | 0 | 1,671 | 0 | exact observed subset |
| `call_update` | `call_remaining_output_tokens` | a real generation checkpoint plus terminal output usage | 0 rows | 0 | 0 | 0 | unsupported; no checkpoint |

Across all emitted BAGEN rows, the four-state total is 34,871 observed, 3,072
censored, 7,621 missing, and 0 invalid, over 45,564 rows. Reasons are
`max_turns` (3,072 rows), `missing_usage` (7,550), and
`missing_task_usage` (71). These are label-state counts, never replacement
numeric values.

### 5.2 Spend semantic matrix

These are full trajectory-audit counts, not inventory estimates. “Eligible”
means ordinary supervised training; a structurally emitted but all-missing or
all-censored cell remains ineligible.

| Position | Target | Rows | Observed | Censored | Missing | Invalid | Reasons | Eligible rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `task_launch` | `task_total_accounted_tokens` | 2,000 | 1,896 | 104 | 0 | 0 | `task_error=104` | 1,896 |
| `task_pre` | `task_unknown_remaining_tokens` | 1,997 | 0 | 101 | 1,896 | 0 | `task_error=101`; `missing_request_tokens_local=1,896` | 0 |
| `task_update` | `task_unknown_remaining_tokens` | 93,117 | 0 | 3,017 | 90,100 | 0 | `task_error=3,017`; `missing_request_tokens_local=90,100` | 0 |
| `call_pre` | `call_unknown_billable_tokens` | 95,114 | 0 | 0 | 95,114 | 0 | `missing_request_tokens_local=95,114` | 0 |
| `call_pre` | `call_billable_output_tokens` | 95,114 | 95,114 | 0 | 0 | 0 | none | 95,114 |
| `call_pre` | `call_final_response_output_tokens` | 95,114 | 95,114 | 0 | 0 | 0 | none | 95,114 |
| `call_update` | `call_remaining_output_tokens` | 0 | 0 | 0 | 0 | 0 | no generation checkpoints; no row emitted | 0 |

Across the 382,456 emitted Spend rows, the four-state total is 192,124
observed, 3,222 censored, 187,110 missing, and 0 invalid. All missing labels
are caused by the absent independent local request count; all censored labels
are task-error suffixes. The other 23 position × target combinations are
unsupported schema cells with zero rows, not observed zero-token labels.

## 6. Four-state policy and no-zero rule

Every target value is one of:

- `observed`: the exact numeric target closes from preserved evidence;
- `censored`: execution or logging ended such that the unobserved future may
  be nonzero, for example `max_turns`, timeout, provider ambiguity, or
  `logging_incomplete`;
- `missing`: the required field was not recorded, for example attempt usage,
  task usage, local request count, or evaluator report;
- `invalid`: preserved evidence contradicts the contract, for example a usage
  total mismatch, negative count, evaluator error, or contradictory report.

`censored`, `missing`, and `invalid` labels always carry `value=None`. They are
never converted to zero. A genuine observed zero remains distinguishable
because it has `status=observed`. Missing completion sets, missing reports,
empty patches, evaluator errors, and derived false flags are therefore not
unresolved-zero imputations.

Missing attempt usage invalidates only labels whose ledger spans that unknown
attempt. It does not erase an already complete, independent call. Likewise,
task termination may be censored while an earlier call-output label remains
observed.

## 7. Leakage blacklist and split discipline

At a prediction cutoff, features may consume only the canonical prefix through
that cutoff. The following are blacklisted as current features:

- current or future provider usage that becomes visible only after a response;
- current/future output length, reasoning tokens/content, finish reason, and
  response outcome;
- future tool calls, tool output, tool failure, task error, or retry outcome;
- final task outcome, evaluator `resolved`, report/test status, patch status,
  total calls, total task tokens, or any future suffix aggregate;
- `output.swebench.jsonl` derived report flags;
- aggregate self-estimates except in an explicitly isolated baseline-only
  feature set;
- raw task/message/tool IDs, raw content, file-system paths, and high-cardinality
  hashes that identify a future or label artifact.

Provider usage from prior completed calls is allowed as a visible historical
feature. Request content is reduced to causal, redacted summaries; labels are
joined by explicit event identity and never passed through the prediction API.
Prefix-invariance tests require byte-identical features for shared prefixes
with arbitrarily different futures.

All runs, model families, prefixes, and checkpoints belonging to the same
`task_id` must share one frozen fold. Candidate eligibility is fixed before
model creation. A model/Agent/tool configuration defines `condition_id`; an
experiment may not silently pool conditions or fit a baseline across them.

## 8. Canonical and dataset reproducibility

Set the source path once in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
```

Re-audit or fetch the pinned BAGEN selections:

```powershell
python scripts/audit_bagen_manifest.py workspace/external/bagen/manifest.jsonl workspace/external/bagen/manifest_summary.json
python scripts/download_bagen.py manifest --apply
python scripts/download_bagen.py origin/swebench-origin-gpt5.2instant/ --apply
python scripts/download_bagen.py origin/swebench-origin-claude-opus4.7/ --apply
python scripts/download_bagen.py origin/swebench-origin-claude-sonnet4.6/ --apply
python scripts/download_bagen.py origin/swebench-origin-qwen3-235b/ --apply
python scripts/download_bagen.py origin/swebench-origin-gemini3.1/ --apply
```

Rebuild the five deterministic BAGEN family audits:

```powershell
python scripts/audit_bagen_swebench.py workspace/external/bagen/origin/swebench-origin-gpt5.2instant workspace/external/bagen/audits/gpt5.2instant.json
python scripts/audit_bagen_swebench.py workspace/external/bagen/origin/swebench-origin-claude-opus4.7 workspace/external/bagen/audits/claude-opus4.7.json
python scripts/audit_bagen_swebench.py workspace/external/bagen/origin/swebench-origin-claude-sonnet4.6 workspace/external/bagen/audits/claude-sonnet4.6.json
python scripts/audit_bagen_swebench.py workspace/external/bagen/origin/swebench-origin-qwen3-235b workspace/external/bagen/audits/qwen3-235b.json
python scripts/audit_bagen_swebench.py workspace/external/bagen/origin/swebench-origin-gemini3.1 workspace/external/bagen/audits/gemini3.1.json
python scripts/audit_bagen_combined.py
```

The BAGEN rerun hashes and dataset IDs are:

| Family | Canonical trajectory-family SHA256 | Dataset ID | Rows |
| --- | --- | --- | ---: |
| Opus 4.7 | `cb7883b98d2794b4c5168149ddb4e9ead25447f5de76921e7199b71055a5f89b` | `0e043ebccd6037a123a15c595da2bc621fd4a1b1fe10c85bd06ff940da0d4a8b` | 5,548 |
| Sonnet 4.6 | `bf0d4d9d790746a362531f1fbdacba40637cfeeb4ea2bf3834ae39f722f0c5ef` | `cc3f77bb859689e4b544f833c5d32237c994ebd816930e6420258786395d1658` | 7,716 |
| Gemini 3.1 | `e44939a877720f35274811981ca29a9f5a2d6eb27a7971ccbdb4be547d53726c` | `8aa37b30a8b0dd9340d380c722c0de463182885583e14c6adbc9ca0a72f67565` | 10,832 |
| GPT-5.2 Instant | `407ba850846998ee1f4820c37b027674672cea832716c443abd3986970779f05` | `8f4f734332c9e3006ace31455ddccf7519fb8994dafc6e5b065c6a7205d50e32` | 5,904 |
| Qwen3 235B | `2d36f9a91d583bdf72712880b9fa3043e6df480c5bb33f67839f5bd1b0864b72` | `db977296560e7b34198f152f74720ebc0daa941f133b7490d8a660746048afd1` | 15,564 |

Each family audit parses and builds twice; `canonical_rerun_consistent=true`.
The combined five-family dataset contains 45,564 rows and has dataset ID
`c845574fd0c0e3da3b6a4d1782787d3d53a1b71db738314836f08419bcb57a60`.
Its canonical family-index SHA256 is
`93af71539f0c4a6991b59e6126e656cbda5add9b45bf4dcf79995cb61501d183`;
the canonical trajectory-index SHA256 is
`bf718b451c07475aff0e046cdea5e7149b500a84096d30731558469bfc71e4ae`.
It can be reproduced directly with the public reader and builder:

```powershell
@'
from pathlib import Path
from token_prediction.collection import BagenSwebenchReader
from token_prediction.dataset import build_supervised_dataset

root = Path("workspace/external/bagen/origin")
families = (
    "swebench-origin-claude-opus4.7",
    "swebench-origin-claude-sonnet4.6",
    "swebench-origin-gemini3.1",
    "swebench-origin-gpt5.2instant",
    "swebench-origin-qwen3-235b",
)
paths = sorted(path for family in families for path in (root / family).rglob("*.traj.json"))
reader = BagenSwebenchReader()
dataset = build_supervised_dataset(reader.read(path) for path in paths)
print(dataset.dataset_id, len(dataset.rows))
'@ | python -
```

Re-verify and inventory Spend, then build its canonical trajectory/dataset
audit without extracting the archive:

```powershell
python scripts/download_spend_archive.py --apply
python scripts/audit_openhands_archive.py --archive workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz --output workspace/external/spend_your_money/gpt_5.2_inventory.json --schema-sample-count 5
python scripts/audit_openhands_trajectory.py --archive workspace/external/spend_your_money/gpt_5.2_4runs.tar.gz --inventory workspace/external/spend_your_money/gpt_5.2_inventory.json --output workspace/external/spend_your_money/gpt_5.2_trajectory_audit.json
```

The Spend combined dataset contains 382,456 rows and has dataset ID
`da0284122fc8dc69297739ccb5a1e2ac826e8caafa4424fe3d53a7c011873a0a`.
Its canonical source-aggregate SHA256 is
`62f2f5150917ca426977232c6cb30c1cb956fd5efa0c2e9b3bbe31be9f5a9bd4`.
The inventory identity and archive identity both reconcile exactly.

| Run | Canonical trajectory aggregate SHA256 | Dataset ID | Rows |
| --- | --- | --- | ---: |
| `run_1` | `d73fbbaf7dcade5c2a948bdc2904ed7e694310719a8338daa98a59c97f161a94` | `44ff8d9db8218c98ae039b702fa26824a47609a7f6ae945a45e33523f6978179` | 93,076 |
| `run_2` | `566fa0903f1772695cee79e22cc9e8733ddf2c510745cd81293508c5ed349c26` | `01cc4558697124a4b6d4d9c034f495d71c7127c47f24433f3742bac6b252ede7` | 92,612 |
| `run_3` | `73b6aeb93292ff6a59f178bd0041e0849dc95c3fc1f5ddc26bb60e645c4cfc35` | `c9a9690426850a70cdeedf56548ba289b2cd2600fce1dcea7c3cdc177e9b2d1c` | 94,248 |
| `run_4` | `055e3a7684917b86ef79562063bfe930b631d54a65c62819888d8c9166b5ff39` | `3a1077215b0d7c426737324a0b12ce04d8e3d7dc44eda667c17ea56d1077f8d7` | 102,520 |

Two complete production executions took 717.2 and 691.6 seconds and produced
the same 1,178,620-byte audit file SHA256
`6713db7b197899153cd8d783343dea7056aa2edcc6a43ba317498f38f6e0743c`
and payload SHA256
`4cd7f93d3ae7c0e6d49d65a7d5d640d409c8a5f3bd0708f8ae8f6da9d387571e`.
The pinned reader SHA256 is
`67d719eb8182a8dd7339e1fa107300caaeafa46943a4b59c814812004e40cb07`;
the trajectory-audit script SHA256 is
`f0e7a6e2beae4470148b52455a4932c08c2257e30453b63d992fc99ec5c95747`.

The Spend hash definitions are fixed in the audit: canonical JSON of each
trajectory's `CanonicalEvent.to_dict()` sequence; an aggregate over
trajectory-ID/hash records sorted by trajectory ID; and builder-identical
dataset-row JSON externally sorted by `point_id`.

## 9. Files changed and verification

This repository has no initial Git commit, so `git status` cannot distinguish a
historical baseline. The trajectory-ingestion branch introduced or updated the
following in-scope files:

- `src/token_prediction/collection/bagen_swebench.py`
- `src/token_prediction/collection/openhands_trajectory.py`
- `src/token_prediction/collection/__init__.py`
- `scripts/audit_bagen_manifest.py`
- `scripts/download_bagen.py`
- `scripts/audit_bagen_swebench.py`
- `scripts/audit_bagen_combined.py`
- `scripts/download_spend_archive.py`
- `scripts/audit_openhands_archive.py`
- `scripts/audit_openhands_trajectory.py`
- `scripts/freeze_trajectory_handoff.py`
- `tests/test_bagen_swebench_reader.py`
- `tests/test_bagen_combined_audit.py`
- `tests/test_openhands_archive_audit.py`
- `tests/test_openhands_trajectory_reader.py`
- `tests/test_openhands_trajectory_audit.py`
- `tests/test_freeze_trajectory_handoff.py`
- `docs/trajectory-data-audit.md`

The final machine-readable mainline handoff is
`workspace/handoffs/trajectory_ingestion_summary.json` (1,843,445 bytes; file
SHA256
`a272b4645e32c53d02df860f3333e7c6f5ce993ffd64defb1fce54738355e189`;
payload SHA256
`f12c1b2d31efa9fbb4619a317db1520b5c4d9a61e023c7e3f904790dd7556e4c`).
It includes relative local paths, Hub pins, raw/file/archive hashes,
task-family/run mappings, all 60 source × position × target cells, four-state
reason counts, dataset/canonical IDs, code hashes, changed-file classes, test
results, and immediate/gated experiments. The production freeze command was
run twice with the same file and payload hashes:

```powershell
$env:PYTHONPATH = 'src'
python scripts/freeze_trajectory_handoff.py --ruff-status passed --ruff-result "All checks passed!" --pytest-status passed --pytest-result "190 passed, 79 subtests passed in 7.42s"
```

Final repository gates are:

```powershell
$env:PYTHONPATH = 'src'
python -m ruff check src tests scripts
python -m pytest -q
```

Final result: Ruff printed `All checks passed!`; Pytest reported `190 passed,
79 subtests passed in 7.42s`. The suite covers streaming/no-extraction
constraints, schema fail-closed behavior, path safety, deterministic
ordering/hashes, usage non-imputation, lifecycle separation, tool alignment,
prefix invariance, identity reconciliation, matrix closure, and handoff
tamper rejection.

No commit or push is part of this handoff.

## 10. Experiments that can start and experiments that remain gated

### Can start immediately from the frozen outputs

- Call-pre output prediction using `call_billable_output_tokens` or
  `call_final_response_output_tokens` on their 9,641 observed rows, with one
  condition per experiment cell and task-level grouped splits.
- Task-launch total-cost prediction on the 232 exact naturally terminated,
  complete-usage trajectories.
- Matched-task, cross-family descriptive comparisons over the 60 five-family
  tasks, without pooling condition-specific baselines.
- A separately named BAGEN provider-input-proxy sensitivity study for
  `task_unknown_remaining_tokens` or `call_unknown_billable_tokens`. Results
  must say “provider-input proxy” and cannot be presented as deployable online
  local-token evidence.
- Missingness/censoring analysis by family, `FormatError` recovery prevalence,
  and tool-failure association studies that do not use future outcomes as
  prediction features.
- Spend call-pre output/final-response prediction on the 95,114 observed rows
  for each exact target, preserving all four runs of a task in one fold and
  using declared task/run weighting.
- Spend task-launch total-accounted-token prediction on the 1,896 observed
  naturally finished trajectories; the 104 task-error trajectories remain
  censored and outside ordinary supervised training.
- Four-run within-task variability and repeatability analysis using the exact
  task-to-run mapping.
- Evaluator-outcome association analysis on the 1,912 observed evaluator
  labels, as an outcome analysis only. Missing and invalid evaluator records
  stay explicit, and evaluator fields remain blacklisted from token predictors.
- Cross-source robustness checks restricted to semantically identical,
  observed call-output targets and clearly separated conditions.

### Must remain gated

- Online `task_unknown_remaining_tokens` and
  `call_unknown_billable_tokens`: neither source has a genuine pre-request
  local tokenizer count; BAGEN has only a proxy and Spend has none.
- Retry-cost or provider-reliability modeling: neither source preserves a
  provider transport retry ledger. BAGEN recovery calls are not within-call
  retries.
- `call_update` remaining-output prediction, dynamic stopping, logprob, hidden
  state, or resumable-generation experiments: no generation checkpoints,
  deltas, logprobs, hidden state, or resumable state are present.
- Spend task-remaining training remains gated because no independent local
  request-token count exists. Evaluator reports cannot open this gate.
- Treating all derived `resolved=false` records as failures, dividing totals by
  four when a run is missing, or filling any absent telemetry with zero.
- Unqualified cross-source tool-failure comparisons until explicit-history and
  failure-unobservable fallback scopes are separated.
- Pooling providers, model variants, Agent/tool configurations, or tokenizer
  semantics into one baseline without a declared condition-normalization
  study.
- Redistribution or publication of raw archives/trajectories until licensing
  and dataset terms are independently cleared.

## 11. License and data-handling risk

Public Hub access is not evidence of redistribution permission. Neither the
download manifest nor this audit establishes a license grant for republishing
raw BAGEN or Spend trajectory content. Treat the license state as unresolved:
keep raw and derived record-level artifacts in ignored local workspace storage,
publish only necessary aggregate statistics/hashes, and obtain explicit legal
or owner confirmation before redistributing data or paper supplements.

The public-only downloaders disable implicit Hugging Face credentials and use
pinned HTTPS URLs. Authentication/configuration files and environment secrets
are outside the ingestion scope and must never be read into audits, copied into
artifacts, or included in logs.
