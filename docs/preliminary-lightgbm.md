# 第一阶段初步实验（2026-07-21）

这轮实验验证的不是“模型已经可以部署”，而是下面四件更基础的事：

1. 同一套轨迹、因果特征、任务分组切分、校准和评测代码能否完整运行；
2. 修正 logical-call/retry 语义后，G1 执行历史是否仍有稳定信号；
3. Deduct-only 在同一 cohort 上究竟表现如何；
4. 每折 LightGBM 能否作为经过完整性校验的 bundle 重载并复现 raw 预测。

## 可复现产物

完整本地产物位于：

```text
workspace/experiments/lightgbm_preliminary/c52866a7e251768726fd
artifact_id = d26969603582ff590a6193234e17d39e6f0697a8e36e08a559549d0a45597afe
```

产物约 8.9 MiB，共 248 个文件，其中 246 个 payload 文件由外层 manifest
保护。它包含逐点 out-of-fold 预测、三个 split seed、每组 10,000 次
task-clustered paired bootstrap、特征增益、每折训练报告，以及 20 个独立
LightGBM bundle。

验收结果：

- 外层 artifact 校验通过；嵌套的 `bundle/manifest.json` 也在外层哈希中；
- 20/20 个 bundle 均能严格加载；
- BAGEN 的 2 个 LightGBM 候选 × 5 折，共 10 次重载预测，与发布的
  `raw_lower/raw_prediction/raw_upper` 逐浮点完全一致；
- bundle 使用 LightGBM 文本模型，不使用 pickle；缺文件、多文件、SHA 不符、
  encoder/quantile/scope 不符都会失败。
- 实验在训练前后各计算一次 source-tree hash；运行中源码变化会拒绝发布，防止
  共享工作区并发产生 mixed-code artifact。

## 固定协议

- 五折，按 `task_id` 分组，同一任务的全部 run 和 prefix 不跨折；
- 每个外层 fold 使用互斥的 train、validation、calibration、test task；
- 每个 task 在一个实验 cell 内总权重为 1；
- encoder 只在 outer-train fold 拟合；LightGBM 只用 validation early stopping；
- calibration/test 标签不进入 estimator fit；
- q05/q50/q95 三个独立 booster，CPU、单线程、确定性 seed；
- 400 rounds、early stopping 40、learning rate 0.03、15 leaves、
  `min_data_in_leaf=10`、`lambda_l2=1`；没有根据 test 结果调参；
- 主 seed 为 20260719，并完整复跑 20260720、20260721；
- 区间用独立 calibration task 上的 task-max split conformal 校准到名义 90%；
- 负的 paired-bootstrap MAE delta 表示候选优于参照。

本轮 G1 使用 feature schema v2：logical Call 只在下一次 `request_built` 结算；
retry 的所有计费 attempt output 求和；任一 attempt usage 缺失则该 Call output
未知。最近三 Call 均值、最近工具类型、上一轮工具错误、连续错误轮次和最近三次
显式 action 重复数都由同一前缀因果 reducer 产生。

## 数据与问题边界

### BAGEN OpenAI 5.2 Codex Sokoban：Task-update

- 128 条真实轨迹，动态 cell 有 496 个精确标签点，覆盖 126 个 task；
- 目标是当前 request 构造后仍未知的未来 billed tokens；
- 940 个跨位置标签因 attempt usage 缺失而保持 missing/invalid，没有补零；
- BAGEN 没有独立的本地 tokenizer 计数，request length 使用 provider-input proxy，
  因而 `history + request proxy` 只属于敏感性分析；
- 该数据衡量 observed cost-to-termination，不是 cost-to-success，也不是同一状态的
  随机 continuation 分布。

### How Do AI Agents Spend Your Money：Task-launch

- 500 个 SWE-bench Verified task，只选 GPT-5.2 + OpenHands 条件；
- 每个 task 一个样本，标签是四次真实运行平均 input + output tokens；
- gold patch、test patch、解决结果和 ground-truth 列不进入特征；
- 该 cell 衡量跨 task 的平均成本预测，不能估计同一 task 的 run-level 方差。

## 结果一：G1 历史信号仍然存在

主 seed 的 BAGEN out-of-fold 结果如下：

| Candidate | Overall MAE | 5-fold MAE mean ± sd | MedianAE | P90AE | WAPE | Pearson | 低估率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empirical Quantile | 2319.8 | 2319.7 ± 209.3 | 1661.0 | 5401.0 | 0.791 | 0.024 | 0.515 |
| Request-length linear | 2334.7 | 2336.2 ± 68.6 | 1637.4 | 4610.9 | 0.796 | 0.235 | 0.366 |
| Deduct-only（within-cell） | 2567.2 | 2565.3 ± 233.9 | 1646.0 | 6635.0 | 0.876 | 0.065 | 0.810 |
| LightGBM history + request proxy | 2171.7 | 2168.9 ± 244.6 | 1438.8 | 4750.6 | 0.741 | 0.287 | 0.442 |
| **LightGBM history-only** | **2118.3** | **2116.2 ± 207.7** | **1427.3** | **4769.7** | **0.723** | **0.327** | **0.439** |

history-only 相对 Empirical 的主 seed MAE 降低 201.5 tokens，即 8.7%。
task-paired bootstrap 的 95% CI 为 `[-308.2, -91.9]`，候选胜率 0.9998。
相对 Deduct-only 的 delta 为 -448.9，95% CI `[-619.8, -272.7]`。

三个重新分配的 split seed 都保持相同方向：

| Split seed | History-only vs Empirical | MAE delta 95% CI | Deduct vs Empirical | MAE delta 95% CI |
|---|---:|---:|---:|---:|
| 20260719 | 改善 8.7% | [-308.2, -91.9] | 变差 10.7% | [141.9, 354.9] |
| 20260720 | 改善 8.0% | [-300.1, -61.9] | 变差 8.9% | [93.4, 320.6] |
| 20260721 | 改善 6.3% | [-248.8, -42.1] | 变差 9.2% | [102.6, 329.0] |

三 seed 平均 MAE 为：Empirical 2330.3、request-length 2328.6、Deduct-only
2554.1、history-only 2151.7、history + request proxy 2147.8。history-only 对
Empirical 的平均相对改善为 7.7%。

加入 request proxy 的方向不稳定：三个 seed 相对 history-only 的 delta 分别为
+53.4、+3.7、-68.8 tokens。因此目前证据支持 G1 历史，但不支持把 BAGEN 的
provider-input proxy 当作已经验证的在线 request 特征。

主 seed、q50、五折平均 normalized split gain：

| History-only feature | Mean normalized gain |
|---|---:|
| completed_call_count | 0.206 |
| last_call_output_tokens | 0.201 |
| cumulative_provider_input_tokens | 0.151 |
| cumulative_provider_output_tokens | 0.135 |
| recent_generated_mean_3 | 0.117 |
| completed_tool_calls | 0.092 |
| failed_api_attempts | 0.059 |
| repeated_action_count_3 | 0.021 |

split gain 只描述当前树如何使用特征，不是因果贡献；主要证据仍是同 cohort 的
feature-set comparison 和 task-paired bootstrap。

## 结果二：当前 Deduct-only 的限制被实证暴露

当前 runner 每次只评估一个 prediction position，所以 `task_update` slice 不含
同一轨迹的 Task-pre 预测。Deduct-only 因而只能在每条测试轨迹的第一个有效
update 使用 outer-train、task-weighted quantile 冷启动，之后执行：

```text
U_next = U_previous + previous_known_offset
                       - observed_spend
                       - current_known_offset
```

它没有读取测试标签，也没有伪装成拥有 Task-pre seed；missing usage 只使所在区间
未知，后续 missing counter 不再增加时可以恢复增量扣减。

结果显示，这个 within-cell 冷启动版本明显差于 Empirical，并且低估率达到 0.810。
这不能解释为“扣减恒等式无效”：恒等式本身已由标签代数和单元测试验证。它说明
初始状态非常关键——一个不对应当前任务的群体 quantile 一旦作为首个 update 的
状态，后续机械扣减只会传播初始误差。

因此，下一版真正的 B5 必须在统一 Task 生命周期评测中，把 outer-fold Task-pre
模型对同一测试轨迹的预测作为 seed，再进入 Task-update；不能用真实总量，也不能
继续把 cell 内冷启动结果称作完整的 cross-position Deduct-only。

## 结果三：Task-launch 浅层特征仍无可靠增益

| Candidate | Overall MAE | 5-fold MAE mean ± sd | WAPE | Pearson | 低估率 |
|---|---:|---:|---:|---:|---:|
| Empirical Quantile | 650,029 | 650,029 ± 91,080 | 0.489 | 0.085 | 0.514 |
| Task-char linear | 726,758 | 726,758 ± 84,577 | 0.547 | 0.040 | 0.314 |
| LightGBM task shape | 649,821 | 649,821 ± 91,263 | 0.489 | 0.073 | 0.476 |
| LightGBM task shape + repo | 649,821 | 649,821 ± 91,263 | 0.489 | 0.073 | 0.476 |
| LLM self-estimation | 1,221,632 | 1,221,632 ± 115,640 | 0.920 | 0.358 | 1.000 |

LightGBM 相对 Empirical 的主 seed delta 只有约 -208 tokens，置信区间跨 0；
另两个 seed 方向翻转。当前结论不是“Task-launch 不可预测”，而是 problem length、
文本形状和 repo 不足以预测，后续需要语义表示、相似任务统计和 run-level 标签。

## 区间校准

| Cell / candidate | Raw point coverage | Raw task coverage | Calibrated point coverage | Calibrated task coverage | Raw NIW | Calibrated NIW |
|---|---:|---:|---:|---:|---:|---:|
| BAGEN / history-only | 0.864 | 0.651 | 0.978 | 0.937 | 18.78 | 31.12 |
| BAGEN / Deduct-only | 0.833 | 0.833 | 0.945 | 0.937 | 16.20 | 26.53 |
| BAGEN / empirical | 0.892 | 0.675 | 0.981 | 0.937 | 23.00 | 33.46 |
| Spend / task-shape LightGBM | 0.876 | 0.876 | 0.902 | 0.902 | 3.27 | 3.52 |

task-max conformal 达到了接近或高于名义值的 simultaneous task coverage，但
BAGEN 动态轨迹的区间仍很宽，暂时不适合精细预算控制。下一阶段需要在固定 coverage
目标下比较 point-wise、进度分层或相对误差 conformal。

## 结论与下一步

1. 第一阶段的 estimator、状态回放、bundle 和不可变 artifact 链路已真实闭合；
2. 修正 retry 和 G1 语义后，history-only LightGBM 仍有跨 seed 的稳定初步信号；
3. within-cell Deduct-only 明确失败，暴露的是 Task-pre 初始化接口缺口，而不是应该
   被隐藏的负结果；
4. 下一步先实现统一 Task 生命周期的 cross-position seed，再比较 Independent MLP
   和 GRU；
5. 自采 Codex 重复轨迹仍是把公共数据信号升级为项目结论的必要条件。

## 复现

```powershell
python -m pip install ".[estimators,data]"
python scripts/run_lightgbm_preliminary.py `
  --bagen-json workspace/external/bagen/sokoban_openai_5_2_codex_dialogues.json `
  --spend-csv workspace/external/spend_your_money/all_models_averaged_predictions_new.csv `
  --swebench-parquet workspace/external/spend_your_money/swe_bench_verified_test.parquet `
  --bootstrap-iterations 10000
```

数据来源：

- [LongjuBai/agent_token_consumption](https://github.com/LongjuBai/agent_token_consumption)
- [MLL-Lab/BAGEN](https://huggingface.co/datasets/MLL-Lab/BAGEN)
- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

外部原始数据和生成产物只存放在被忽略的本地 `workspace/`，不会随仓库上传。
