# 当前工程第一阶段交付

本文记录 2026-07-21 主线讨论中约定的第一阶段工程范围：

1. 把 Deduct-only 做成真实的在线状态 baseline；
2. 把 LightGBM 从“训练后能预测”推进到“可验证、可重载、可发布”；
3. 修正并补齐核心 G1 logical-call 特征；
4. 在真实 BAGEN cohort 上统一比较并发布不可变实验产物。

这不是对研究文档六阶段 roadmap 的越级宣称。原始 roadmap 的 Stage 1 还要求
项目自采数据覆盖全部预测位置、ledger reconciliation 和 repeated-run identity
审计；这些必须等 live Codex 轨迹到位后才能完成。

## 已完成的契约

### Deduct-only

`deduct_only` 已进入内置 estimator registry、`configs/lightgbm_mvp.toml` 和公共数据
实验脚本。它只支持：

```text
position = task_update
target   = task_unknown_remaining_tokens
```

同一 session 内的状态更新为：

```text
U_next = U_previous + previous_known_offset
                       - observed_spend
                       - current_known_offset
```

- 首个有效 update 使用 outer-train、task-weighted q(alpha/2)/q50/q(1-alpha/2)
  冷启动；
- 它不读取 test label，也不声称获得了 Task-pre 预测；
- transition、point 顺序、task/run/trajectory、position、target、condition 全部
  fail closed 校验；
- missing offset/spend 不补零，当前区间 carry forward；
- missing-attempt counter 只污染它增长的区间；前后 counter 相同时，历史缺失在
  cumulative delta 中抵消，后续可恢复扣减；
- 负数截断为 0，区间始终有序。

当前限制是明确的：ExperimentRunner 按单个 position 建立 cell，Task-update session
看不到同一轨迹的 Task-pre forecast。因此它是 within-cell baseline，不是最终论文中
完整的 cross-position B5。

### G1 logical-call 特征

`FEATURE_SCHEMA_VERSION = 2`，dataset identity 也显式包含 feature schema version。

- logical Call 在下一 `request_built` 才结算；generation checkpoint 不会把半个
  retry chain 当成已完成 Call；
- `last_call_output_tokens` 是所有 terminal attempts 的 billable output 之和；
  failed-with-usage 同样计费；
- 任一 attempt usage 不完整、无 terminal、或 started/terminal 集合不闭合时，
  Call output 为 `None`；
- `recent_generated_mean_3` 使用最近至多三个已结算 Call；窗口含未知值时为
  `None`；
- `last_tool_type` 是整个可见前缀中最近一次工具类型，空工具轮次不会抹掉它；
- error round 定义为含 `API_FAILED` 或 `TOOL_FAILED` 的 logical Call；
- `last_round_tool_error_count` 只统计上一 Call 的 `TOOL_FAILED`；
- `consecutive_error_rounds` 统计连续 error rounds；
- `repeated_action_count_3 = window_size - unique_action_keys`，窗口是最近至多三次
  terminal tool actions，action key 优先级为
  `action_hash > action_name > action > tool_name`；`TOOL_STARTED` 不重复计数。

当前 reducer 假设 Agent logical calls 串行。轨迹验证器已经把这个假设升级成显式
契约：新 request 开始后，旧 call 的 terminal/tool/checkpoint 事件不得迟到；新
request 也不能跨过仍未 terminal 的 active attempt。未来若支持并发 Agent，必须
引入 per-call ledger，而不是放宽验证后继续使用当前 reducer。

### LightGBM bundle

每个 fitted fold 可生成严格 bundle：

```text
fold_N/
  encoder.json
  fit_report.json
  feature_importance.jsonl
  q05.model.txt
  q50.model.txt
  q95.model.txt
  bundle/
    encoder.json
    manifest.json
    manifest.sha256
    model-<quantile-id>.txt
```

父目录保留人类审计材料；`bundle/` 是严格部署文件集，不能混入多余文件。

manifest 绑定：

- bundle/estimator/encoder schema version；
- dataset ID、prediction position、target、condition IDs；
- quantile value、无碰撞 quantile ID、best iteration；
- 每个模型和 encoder 的 SHA256；
- encoder content hash；
- LightGBM/NumPy 版本与 fit report。

加载器拒绝缺文件、多文件、symlink、重复 JSON key、schema 漂移、checksum 不符、
quantile/model 映射不符、encoder feature 不符和 major-version 不兼容。模型使用
LightGBM 文本格式，不使用 pickle。

标准 pipeline 通过通用的可选 `bundle_files() -> Mapping[str, bytes]` 接口收集
文件，核心没有 `if estimator == lightgbm` 分支。外层 artifact manifest 只排除
根目录自己的 `manifest.json` 和 `_SUCCESS`；嵌套 bundle manifest 也会被哈希。
训练结束、写产物之前还会再次计算 source-tree hash；若共享工作区在运行中被其他
任务修改，pipeline 会拒绝发布 mixed-code artifact。

`PredictionPoint` 和 `RunContext` 当前不携带 dataset ID，所以在线 session 能强制
验证 position/target/condition，但只能把 dataset ID 暴露为 fitted metadata，尚
不能在每次 `predict` 时核对。这是后续接口债务，不在本阶段伪装解决。

## 验证结果

- Ruff：通过；
- 全量自动化测试：`116 passed, 27 subtests passed`；
- BAGEN：128 条轨迹、496 个有效 Task-update 点、126 个有效 task；
- 三个 task-grouped split seed、每组五折、每个 paired comparison 10,000 次
  task bootstrap；
- 新 artifact：`c52866a7e251768726fd`；
- 20/20 fold bundles 加载成功；
- 10 组 BAGEN LightGBM fold raw 预测与发布结果精确一致。

完整数值、置信区间和负结果见
[第一阶段初步实验](preliminary-lightgbm.md)。

## 第一阶段之后的真实接口

下一阶段可以在不改现有数据/评测主干的前提下新增：

- cross-position Task lifecycle runner：用 outer-fold Task-pre 预测初始化 Deduct、
  GRU 等 Task-update 方法；
- `IndependentMLPQuantileEstimator`：忽略 `observe`，作为动态方法参照；
- `GRUTaskUpdateEstimator`：在同一 `observe -> predict` 顺序下维护 hidden state；
- 新模型/Agent 条件：继续由 `condition_id` 分 cell，各自重训 baseline；
- 新 feature group ablation：只改变 `feature_set`，由现有 config-diff guard 验证；
- live shadow predictor：复用完全相同的 session API，不另写一套离线更新逻辑。

第一优先级不是继续调 LightGBM，而是完成 cross-position seed 和项目自己的 Codex
重复轨迹采集。否则 Deduct/GRU 的初始化比较仍然不完整，公共数据上的信号也不能
升级为最终结论。
