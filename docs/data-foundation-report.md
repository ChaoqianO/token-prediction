# Data foundation verification report

Status: PR-0 data-foundation evidence, frozen 2026-07-21. This report records
identities and verification boundaries; it does not redistribute raw data and
does not upgrade the historical Stage 1 pilot into a source-reproducible result.

## Evidence summary

The machine handoff is
`workspace/handoffs/trajectory_ingestion_summary.json` (1,843,445 bytes). Its
file SHA256 is
`a272b4645e32c53d02df860f3333e7c6f5ce993ffd64defb1fce54738355e189`
and its canonical payload SHA256 is
`f12c1b2d31efa9fbb4619a317db1520b5c4d9a61e023c7e3f904790dd7556e4c`.
The human-readable source audit, `docs/trajectory-data-audit.md`, has SHA256
`2b02dc5b29d115d8c55dab2831691441d2697379a9272a98bb5272d95f9d94a2`.

The production handoff records a successful final validation of `190 passed,
79 subtests passed` and Ruff `All checks passed!`. Those numbers identify the
frozen ingestion handoff run; they are not a prediction of the test count after
this PR adds further CI tests.

A separate read-only rehash covered 345 files and 3,201,394,694 bytes
(3.201 GB decimal), with zero byte-count or SHA256 mismatches. The inventory
comprised 316 BAGEN SWE trajectories, 5 BAGEN auxiliary files, 5 family audits,
the manifest and its summary, the combined audit, 5 BAGEN semantic-code pins,
the retained BAGEN Sokoban input, the Spend archive/inventory/trajectory audit,
4 Spend semantic-code pins, 2 Spend aggregate sidecars, and the final handoff.
This independent check did not read local credentials and its report contains
no absolute local paths or individual task identifiers.

## Pinned source and dataset matrix

| Source | Upstream revision | Raw/inventory anchor | Cohort | Frozen dataset | Canonical anchor |
| --- | --- | --- | ---: | --- | --- |
| BAGEN SWE-bench | `58189576e54b675fdd0e1d6c1c9f189c2992732f` | manifest SHA256 `f5900dead3a32ca303d500f123ee96b89e6797527cbb99fef0cd9beaf2a00071` | 316 trajectories / 64 tasks | 45,564 rows; `c845574fd0c0e3da3b6a4d1782787d3d53a1b71db738314836f08419bcb57a60` | family index `93af71539f0c4a6991b59e6126e656cbda5add9b45bf4dcf79995cb61501d183`; trajectory index `bf718b451c07475aff0e046cdea5e7149b500a84096d30731558469bfc71e4ae` |
| Spend Your Money / OpenHands GPT-5.2 | `fa9cbb063f770df596da95af24f7af3b8f595778` | archive SHA256 `993abcb55aae423f9067d5e6c8e1aeaccf83b9ce31474a215982686527934214` | 2,000 trajectories / 500 tasks | 382,456 rows; `da0284122fc8dc69297739ccb5a1e2ac826e8caafa4424fe3d53a7c011873a0a` | source aggregate `62f2f5150917ca426977232c6cb30c1cb956fd5efa0c2e9b3bbe31be9f5a9bd4` |

The BAGEN combined audit is 139,686 bytes with file SHA256
`2d8f3abe10b526f80488554d672039c9f9bc81b31e230b7bb6b14c94b0ffaea5`
and payload SHA256
`f144f38fb7b7ab85c25ff859f6be59f5a9d316871a6ebfa5311ffc9a8730d17a`.
Its five family identities are:

| Family | Canonical trajectory-family SHA256 | Dataset ID | Rows |
| --- | --- | --- | ---: |
| Claude Opus 4.7 | `cb7883b98d2794b4c5168149ddb4e9ead25447f5de76921e7199b71055a5f89b` | `0e043ebccd6037a123a15c595da2bc621fd4a1b1fe10c85bd06ff940da0d4a8b` | 5,548 |
| Claude Sonnet 4.6 | `bf0d4d9d790746a362531f1fbdacba40637cfeeb4ea2bf3834ae39f722f0c5ef` | `cc3f77bb859689e4b544f833c5d32237c994ebd816930e6420258786395d1658` | 7,716 |
| Gemini 3.1 | `e44939a877720f35274811981ca29a9f5a2d6eb27a7971ccbdb4be547d53726c` | `8aa37b30a8b0dd9340d380c722c0de463182885583e14c6adbc9ca0a72f67565` | 10,832 |
| GPT-5.2 Instant | `407ba850846998ee1f4820c37b027674672cea832716c443abd3986970779f05` | `8f4f734332c9e3006ace31455ddccf7519fb8994dafc6e5b065c6a7205d50e32` | 5,904 |
| Qwen3 235B | `2d36f9a91d583bdf72712880b9fa3043e6df480c5bb33f67839f5bd1b0864b72` | `db977296560e7b34198f152f74720ebc0daa941f133b7490d8a660746048afd1` | 15,564 |

The Spend archive is exactly 2,908,192,516 bytes. Its inventory is 6,624,209
bytes with SHA256
`11735ca3ec625a21c57f72d8172cdc0e0e67fbc44ee99225366739e5dbfffc24`.
The trajectory audit is 1,178,620 bytes with file SHA256
`6713db7b197899153cd8d783343dea7056aa2edcc6a43ba317498f38f6e0743c`
and payload SHA256
`4cd7f93d3ae7c0e6d49d65a7d5d640d409c8a5f3bd0708f8ae8f6da9d387571e`.
Run-level closure is:

| Run | Canonical trajectory aggregate SHA256 | Dataset ID | Rows |
| --- | --- | --- | ---: |
| `run_1` | `d73fbbaf7dcade5c2a948bdc2904ed7e694310719a8338daa98a59c97f161a94` | `44ff8d9db8218c98ae039b702fa26824a47609a7f6ae945a45e33523f6978179` | 93,076 |
| `run_2` | `566fa0903f1772695cee79e22cc9e8733ddf2c510745cd81293508c5ed349c26` | `01cc4558697124a4b6d4d9c034f495d71c7127c47f24433f3742bac6b252ede7` | 92,612 |
| `run_3` | `73b6aeb93292ff6a59f178bd0041e0849dc95c3fc1f5ddc26bb60e645c4cfc35` | `c9a9690426850a70cdeedf56548ba289b2cd2600fce1dcea7c3cdc177e9b2d1c` | 94,248 |
| `run_4` | `055e3a7684917b86ef79562063bfe930b631d54a65c62819888d8c9166b5ff39` | `3a1077215b0d7c426737324a0b12ce04d8e3d7dc44eda667c17ea56d1077f8d7` | 102,520 |

## Tracked source descriptors

New builds must use the tracked, reviewable source descriptors instead of a
free-form local path:

| Descriptor | Source ID | Revision / manifest binding | Capability contract hash |
| --- | --- | --- | --- |
| `configs/source_descriptors/bagen_swebench.json` | `bagen_swebench_traj_v2` | BAGEN revision above and manifest SHA256 `f5900dead3a32ca303d500f123ee96b89e6797527cbb99fef0cd9beaf2a00071` | `4f2ed89b0b0d2e31a8f8aa84a14d32f8eccb30c8bb1348fdb7427054ca038fc9` |
| `configs/source_descriptors/spend_openhands.json` | `openhands_archive_trajectory_v3` | Spend revision above and inventory SHA256 `11735ca3ec625a21c57f72d8172cdc0e0e67fbc44ee99225366739e5dbfffc24` | `44a63f678831715afbd1d106f657661ca8a47eb48c1dbd3f5331f4bb7ab12f2e` |

Schema-v2 dataset construction includes the entire source descriptor and its
capability-contract hash in the dataset semantic payload. The resulting
dataset ID therefore binds source identity, upstream revision/manifest,
declared observables, capability decisions, feature schema, and rows. Changing
the descriptor changes the dataset ID even when the row values happen to be
unchanged. The frozen IDs in the preceding tables remain the schema-v1 handoff
identities and must not be silently relabeled as schema-v2 IDs.

The source-ID versions distinguish normalization contracts from immutable raw
revisions. The historical BAGEN family audits and handoff retain
`bagen_swebench_traj_v1`, while the corrected proxy-free reader used by new
schema-v2 builds is `bagen_swebench_traj_v2`. Likewise, the frozen Spend
handoff retains `openhands_archive_trajectory_v2` and its pinned reader hash;
the causal route-identity correction is published as
`openhands_archive_trajectory_v3`. The old handoff, trajectory audits, source
IDs, and code pins are not rewritten. New descriptors keep the same upstream
revisions and raw manifest hashes, but their new source IDs and capability
contract hashes ensure that new dataset IDs cannot alias the historical
canonical semantics.

## Historical Stage 1 artifact

The retained preliminary artifact is
`workspace/experiments/lightgbm_preliminary/c52866a7e251768726fd`, with artifact
ID `d26969603582ff590a6193234e17d39e6f0697a8e36e08a559549d0a45597afe`.
Independent verification established:

- the outer artifact manifest closes over all 246 payload files and has SHA256
  `089bb98b607164c6641fd59d4878638155a690e8e50ea7ac0ea99ff8853ec146`;
- all 20 of 20 strict LightGBM bundles load (10 BAGEN and 10 Spend bundles);
- both BAGEN learned candidates reproduce all 992 published raw
  lower/point/upper predictions exactly, with zero mismatches; and
- the BAGEN input used for parity has SHA256
  `c4f7c73c35b741b17093fd3136017baaa0069d2d5108d0792597801b162bad12`.

That exact inference check uses the verifier-only
`legacy_proxy_projection_v1` compatibility projection. It recreates the
retired pilot's non-causal provider-input proxies: each request receives its
first observed post-response provider-input audit, while the task proxy uses
only the first logical call and remains missing if that call has no usage. The
projection is never used by a reader or schema-v2 dataset. Its 992-record
parity digest is
`dd1ee3b93fa791dffe8f0c5cd8ae2dbed8cd705cb47804f5552d7fc685108def`.

This is artifact and inference compatibility evidence, not end-to-end training
reproducibility. The protocol records code hash
`d03c979e6ac290089787456fee4258df073da014509812e3977d2fce28121fe8`,
but no commit in the available repository history reconstructs that exact
source set. The old source is therefore unrecoverable and the artifact must
remain explicitly “unbound”. Its metrics cannot be promoted as a
commit-reproducible baseline. Exact compatibility also does not make the
post-response proxy a valid online feature or constitute recovery of the old
source.

The reusable verifier makes this boundary executable:

```powershell
$env:PYTHONPATH = 'src'
python scripts/verify_stage1_baseline.py `
  --artifact workspace/experiments/lightgbm_preliminary/c52866a7e251768726fd `
  --bagen-json workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json `
  --expected-artifact-id d26969603582ff590a6193234e17d39e6f0697a8e36e08a559549d0a45597afe `
  --expected-bundles 20 `
  --expected-parity 992 `
  --discover-source-commit
```

For every new baseline, first commit the exact training source, run training
from that source, and bind the artifact's recorded code hash to that commit:

```powershell
$commit = git rev-parse HEAD
python scripts/verify_stage1_baseline.py `
  --artifact workspace/experiments/lightgbm_preliminary/<new-run> `
  --bagen-json workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json `
  --source-commit $commit `
  --write-baseline workspace/baselines/stage1/<new-run>.json
```

The command fails unless the commit's ordered `src/**/*.py` files plus
`scripts/run_lightgbm_preliminary.py` reproduce the artifact's protocol code
hash. The baseline file binds the full commit SHA, code hash, artifact ID,
artifact-manifest SHA256, BAGEN input SHA256, bundle count, and a digest of the
exact raw-prediction parity result. Later checks consume the frozen binding:

```powershell
python scripts/verify_stage1_baseline.py `
  --artifact workspace/experiments/lightgbm_preliminary/<new-run> `
  --bagen-json workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json `
  --baseline workspace/baselines/stage1/<new-run>.json
```

## CI matrix and local build gates

The GitHub Actions workflow uses only `pull_request`, `push`, and manual
dispatch events; it does not use `pull_request_target`. Workflow permissions
are limited to read-only repository contents. Third-party Actions are pinned
to full commit SHAs: checkout v6.0.2 at
`de0fac2e4500dabe0009e67214ff5f5447ce83dd` and setup-python v6.2.0 at
`a309ff8b426b58ec0e2a45f0f869d46889d02405`.

| Gate | Python 3.11 | Python 3.12 |
| --- | :---: | :---: |
| Install `.[dev,data,estimators]` | yes | yes |
| Repository-intended secret scan | yes | yes |
| Ruff lint (`check`, not formatter enforcement) | yes | yes |
| Targeted schema and prefix-causality tests | yes | yes |
| Synthetic artifact integrity and bundle prediction parity | yes | yes |
| Full Pytest suite | yes | yes |
| Full `unittest` discovery | yes | yes |
| Wheel build, closed-RECORD inspection, clean install and import | yes | yes |

Equivalent local gates are:

```powershell
python -m pip install ".[dev,data,estimators]"
python scripts/scan_secrets.py
python -m ruff check src tests scripts
python -m pytest -q
python -m unittest discover -s tests -v

python -m build --wheel
python scripts/check_distribution.py dist/token_prediction-0.1.0-py3-none-any.whl
```

The distribution check accepts one wheel, validates its `METADATA`, console
entry point, exact file set, and every SHA256/size entry in `RECORD`, and
rejects tests, workspace data, generated state, sensitive filenames,
symlinks, and unexpected top-level material. CI then installs that wheel with
no dependency or source-tree fallback into a fresh virtual environment and
imports both package metadata and the CLI.

## Data-handling boundary

Public availability is not a redistribution license. Raw trajectories,
archives, record-level derivatives, baseline bundles, and generated baseline
bindings stay under ignored `workspace/` storage. Only aggregate counts,
relative repository paths, revisions, and cryptographic identities appear in
this report. Publication or redistribution remains gated on independent
license and dataset-terms review.
