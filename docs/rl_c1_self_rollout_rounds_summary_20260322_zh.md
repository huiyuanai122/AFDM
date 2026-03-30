# RL C1 Self-Rollout 轮次总结

日期：2026-03-22

## 1. 这一步做了什么

在第一轮 state ablation 后，当前新的主线 state 变成了：

- 保留 `prev_action_norm`
- 去掉 `prev_reward`
- 去掉 `recent_switch_rate`
- 加入 `physical delta` 特征

对应第一轮最佳模型：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta`

接下来做的事情，是把这条新 state 真正接入 self-rollout 主线，连续做：

1. 用当前最优 checkpoint 重新 rollout 出 on-policy dataset
2. 把 original + 新 rollout 混合训练
3. 继续下一轮 rollout
4. 比较 `r1 / r2 / r3 / r4`

为支持这条线，额外补了一个脚本能力：

- `python/rl_c1/build_dagger_rollout_dataset.py`

它现在支持：

- 只重写源 state 中实际存在的 history 列
- 因而可以直接服务
  `prev_action_only` / `prev_action_only + physical_delta`
  这类不再包含 `prev_reward`、`recent_switch_rate` 的新 state


## 2. 各轮定义

### `r1`

不是自一致 rollout。

它是：

- original variant dataset
- 加固定 probe `dagger_r1 variant dataset`
- 做 mixed training

对应结果目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta`


### `r2`

第一轮真正的 self-rollout：

- 用 `r1` 最优 checkpoint rollout
- 得到：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r2.npz`
- 再做 mixed training

对应结果目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r2_prev_action_only_physical_delta`


### `r3`

第二轮 self-rollout：

- 用 `r2` 最优 checkpoint rollout
- 得到：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r3.npz`
- 再做 mixed training

对应结果目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta`


### `r4`

第三轮 self-rollout：

- 用 `r3` 最优 checkpoint rollout
- 得到：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r4.npz`
- 再做 mixed training

对应结果目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r4_prev_action_only_physical_delta`


## 3. 主结果对比

| 轮次 | best val_ber | original gain | current rollout gain | fixed probe `dagger_r1` gain | original switch | 12dB recovery | action 11 freq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1` | `0.042770` | `1.280%` | `1.273%` | `1.273%` | `0.0024` | `5.59%` | `2.255%` |
| `r2` | `0.042706` | `1.317%` | `1.462%` | `1.326%` | `0.0049` | `8.64%` | `8.485%` |
| `r3` | `0.042776` | `1.374%` | `1.459%` | `1.410%` | `0.0079` | `9.21%` | `13.645%` |
| `r4` | `0.042634` | `1.339%` | `1.429%` | `1.327%` | `0.0018` | `6.00%` | `4.920%` |


## 4. 怎么读这张表

### 4.1 `r2` 是明显有效的

从 `r1` 到 `r2`：

- original gain：`1.280% -> 1.317%`
- current rollout gain：`1.273% -> 1.462%`
- fixed probe `dagger_r1` gain：`1.273% -> 1.326%`
- `12dB recovery`：`5.59% -> 8.64%`

说明：

- 新 state 并不是只在固定 probe 上偶然有效
- 一旦换成它自己的 rollout 分布并重训，收益是继续涨的


### 4.2 `r3` 给出了当前最强的原始集结果

从 `r2` 到 `r3`：

- original gain：`1.317% -> 1.374%`
- fixed probe `dagger_r1` gain：`1.326% -> 1.410%`
- `12dB recovery`：`8.64% -> 9.21%`

这意味着：

- `r3` 不是只在“自己生成的 rollout 分布”上更好
- 在固定 probe 上也继续提升了
- 因此 `r3` 可以视为当前这条主线的最强综合结果


### 4.3 但 `r3` 同时开始出现行为副作用

相对 `r1` / `r2`，`r3` 有两个明显副作用：

1. `switch_rate` 持续升高
   - `0.0024 -> 0.0049 -> 0.0079`

2. action 11 频率重新抬头
   - `2.255% -> 8.485% -> 13.645%`

也就是说：

- `r3` 在 BER 上是最好
- 但它不是完全“更健康”，而是开始用更激进的动作行为换收益


### 4.4 `r4` 证明这条 rollout 深化线已经过了最优点

`r4` 的信号很明确：

- 训练侧 `val_ber` 反而最好
- 但 sequence-level 主指标开始回退
  - original：`1.374% -> 1.339%`
  - current rollout：`1.459% -> 1.429%`
  - fixed probe：`1.410% -> 1.327%`
- `12dB recovery` 也掉回 `6.00%`

这说明：

- 继续堆 rollout 深度已经不是稳定增益
- `r4` 是这条线的转折点


## 5. 当前应该停在哪一轮

当前建议停在：

- `r3`

理由：

1. 它给出了目前最高的 original gain
   - `1.374%`

2. 它在固定 probe `dagger_r1` 上也是当前最高
   - `1.410%`

3. `r4` 已经证明继续加 rollout 轮数不再稳健

所以这条线当前最合理的结论是：

- `r3` 是当前最佳停止点
- 不建议直接继续做 `r5`


## 6. 当前最佳主模型

当前这条主线的推荐 checkpoint 是：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta.pt`

关键 summary：

- original：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_orig_summary.json`
- self-rollout：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_dagger_summary.json`
- fixed probe：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_dagger_probe_r1_summary.json`


## 7. 当前主结论

这条线现在已经可以比较清楚地下结论：

1. `history mismatch` 的主病灶已经被 state redesign 明确击中
2. `prev_action_only + physical delta` 是正确的 state 方向
3. 在这个 state 上，self-rollout 继续做是有收益的，但不是无限涨
4. 这条线目前在 `r3` 达到最好平衡
5. `r4` 已经说明单纯继续 rollout 深化会开始回退


## 8. 下一步建议

现在不建议继续做：

- `r5 / r6`

更合理的下一步是换路线，而不是继续堆 rollout 深度。

优先建议：

1. 基于当前最佳 `r3` state+policy，做 reward / sample weighting 改造
   - 例如按 `oracle gap` 加权

2. 如果 reward 改造后仍然主要卡在“动作头过于单一”
   再考虑：
   - GRU / temporal policy
   - 两阶段动作头

换句话说：

- state 方向已经收敛
- rollout 深度也已经基本摸清
- 下一阶段最该动的是训练目标，而不是继续加 rollout 轮数

