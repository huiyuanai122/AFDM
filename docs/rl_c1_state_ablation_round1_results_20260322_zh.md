# RL C1 第一轮 State Ablation 结果总结

日期：2026-03-22

对应计划：

- `docs/rl_c1_state_ablation_plan_20260322_zh.md`


## 1. 一句话结论

第一轮 state ablation 已经把核心问题验证清楚了：

- `history shortcut` 的判断成立
- `prev_reward` 和 `recent_switch_rate` 是最脆弱、最该优先去掉的 history 分量
- 在保留 `prev_action_norm` 的同时补入 physical delta 特征后，得到了当前这轮最好的结果

当前这一轮的新最佳候选是：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta`

它在统一评测口径下达到：

- original：`ber_gain_vs_base = 1.280%`
- dagger probe：`ber_gain_vs_base = 1.273%`

这两个数都高于此前的 `mix_r1_tinyrl`：

- original：`1.074%`
- dagger probe：`1.110%`

也高于旧的原始 baseline：

- original baseline：`1.121%`


## 2. 本轮实验怎么做的

### 2.1 控制组

控制组直接复用已有结果：

- `full_state = results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl`

它对应当前完整 20 维 state。


### 2.2 三个新变体

1. `prev_action_only`
   - 去掉 `prev_reward`
   - 去掉 `recent_switch_rate`
   - 保留 `prev_action_norm`

2. `no_history`
   - 去掉 `prev_action_norm`
   - 去掉 `prev_reward`
   - 去掉 `recent_switch_rate`

3. `prev_action_only_plus_physical_delta`
   - 在 `prev_action_only` 基础上加入：
     - `alpha_com`
     - `v_norm`
     - `delta_alpha_rms`
     - `abs_alpha_com`
     - `delta_alpha_com`
     - `delta_v_norm`
     - `delta_delta_alpha_rms`


### 2.3 统一训练 / 评测口径

这轮统一复用 `mix_r1_tinyrl` 的训练配方：

- mixed dataset：original + `dagger_r1`
- reward：`reward_relbase_proxy`
- split：`sequence`
- epochs：`20`
- batch size：`256`
- `reinforce_lr = 1e-4`
- `imitation_preserve_coef = 0.5`

说明：

- 这轮把 `dagger_r1` 当作固定 shifted probe 使用
- 目的是先诊断 state shortcut，不是先重定义每个变体自己的 rollout fixed point


## 3. 关键结果表

| 变体 | best val_ber | original gain | dagger gain | orig-dagger gap | original switch | 12dB recovery | top action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full_state` | `0.042566` | `1.074%` | `1.110%` | `0.035%` | `0.0037` | `9.16%` | action `1` (`51.28%`) |
| `prev_action_only` | `0.042812` | `0.796%` | `0.788%` | `0.008%` | `0.0012` | `9.32%` | action `1` (`53.17%`) |
| `no_history` | `0.042548` | `1.011%` | `1.011%` | `0.000%` | `0.0000` | `4.40%` | action `1` (`54.74%`) |
| `prev_action_only_plus_physical_delta` | `0.042770` | `1.280%` | `1.273%` | `0.007%` | `0.0024` | `5.59%` | action `1` (`53.83%`) |


## 4. 结果怎么解释

### 4.1 `history shortcut` 被直接验证了

最强的证据是：

- `no_history` 的 `orig-dagger gap = 0`

也就是说，一旦完全移除 policy history block，
original 和 shifted probe 上的性能差异就直接消失了。

这说明：

- 当前 train-test mismatch 的主要入口，确实就是
  `prev_action_norm / prev_reward / recent_switch_rate`
- 尤其是后两者更可能在制造脆弱 shortcut


### 4.2 `prev_reward` 和 `recent_switch_rate` 比 `prev_action` 更危险

对比 `full_state` 和 `prev_action_only`：

- gain gap 从 `0.035%` 缩到 `0.008%`
- action 11 频率从 `20.445%` 降到 `8.48%`

说明：

- 仅保留 `prev_action_norm` 时，robustness 明显更好
- 真正更脆弱的是 `prev_reward` 和 `recent_switch_rate`

但它也有代价：

- original gain 从 `1.074%` 掉到 `0.796%`

这说明只靠“删 history”还不够，还需要补更有物理意义的状态量。


### 4.3 `no_history` 证明了问题根源，但也证明当前策略会退化成更静态

`no_history` 的特点非常明确：

- `orig` 和 `dagger` 完全一致
- `best val_ber` 甚至略优于 `full_state`
- 但 `switch_rate_policy = 0`
- `12dB recovery` 掉到 `4.40%`

所以它告诉我们两件事：

1. history mismatch 的根源找对了
2. 如果把 policy-history 全去掉，模型会变成更保守、更接近静态的策略

也就是说：

- `no_history` 是一个很好的诊断组
- 但不是最好的最终部署方案


### 4.4 `prev_action_only_plus_physical_delta` 是本轮最值得继续推进的结果

这个变体最关键的点是：

- 它保住了 robustness
  - `orig-dagger gap = 0.007%`
- 又把绝对 BER 拉到了当前最高
  - original `1.280%`
  - dagger `1.273%`

同时它还有两个重要副作用：

- action 11 频率进一步降到 `2.255%`
- low/mid SNR 的 switch gap 也改善到本轮最好

按 `0-10 dB` 平均 `|switch_rl - switch_oracle|` 看：

- `full_state`：`0.357`
- `prev_action_only`：`0.321`
- `no_history`：`0.319`
- `prev_action_only_plus_physical_delta`：`0.283`

这说明：

- 补进 physical delta 后，策略行为更接近 oracle
- 而且这个改善已经转成了 BER 收益


## 5. 还没完全解决的地方

虽然本轮已经明显变好，但还没有“全都解决”。

主要还有两个残留问题：

### 5.1 还是塌向 action 1

四组里 top action 全都是 action `1`：

- `full_state`：`51.28%`
- `prev_action_only`：`53.17%`
- `no_history`：`54.74%`
- `prev_action_only_plus_physical_delta`：`53.83%`

说明：

- action 11 塌缩基本被打掉了
- 但新的单点偏置还在
- 只是这个偏置现在没有以前那么伤 BER


### 5.2 `12 dB recovery fraction` 没有同步最好

`prev_action_only_plus_physical_delta` 虽然 BER 最好，
但 `12 dB recovery` 只有 `5.59%`，
还不如 `full_state / prev_action_only` 的 `9%+`。

这意味着：

- 当前 reward / behavior 指标和最终 BER 之间仍存在一定错位
- 模型还没有把“动态恢复能力”和“最终 BER 收益”同时做到最好


## 6. 当前结论和下一步建议

### 6.1 当前结论

这轮实验已经可以把决策收敛到下面这句：

- 下一阶段不应该优先上 GRU
- 也不应该继续沿着 `full_state` 堆更多 rollout
- 应该先固定新的 state 方向：
  - 去掉 `prev_reward`
  - 去掉 `recent_switch_rate`
  - 保留 `prev_action_norm`
  - 引入 physical delta 特征


### 6.2 下一步最合理的动作

建议按这个顺序继续：

1. 把 `prev_action_only_plus_physical_delta` 作为新的主候选 state
2. 用这个 state 重新做一轮真正属于它自己的 rollout dataset
3. 再做 mixed training，验证它在“自诱导 on-policy 分布”下是否仍然保持优势
4. 如果这个结果仍然成立，再考虑是否需要：
   - GRU / temporal policy
   - `oracle_gap` weighting
   - 两阶段动作头

换句话说：

- 这轮已经把“病灶”找到了
- 也已经拿到了一个比旧 baseline 更好的 state 版本
- 现在最应该做的是把这个 state 版本转成新的主线，而不是立刻做更重的模型升级


## 7. 关键产物路径

### 7.1 新计划文档

- `docs/rl_c1_state_ablation_plan_20260322_zh.md`


### 7.2 新数据

- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_no_history.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1_prev_action_only.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1_no_history.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1_prev_action_only_physical_delta.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1_prev_action_only.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1_no_history.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1_prev_action_only_physical_delta.npz`


### 7.3 新结果目录

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only`
- `results/paper_tsv2seq_vdop_ctrl_mix_r1_no_history`
- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta`


### 7.4 本轮最佳候选

- checkpoint：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta.pt`
- original summary：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta/sequence_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta_orig_summary.json`
- dagger summary：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta/sequence_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta_dagger_summary.json`
- adaptivity：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta/adaptivity_diagnostics_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta.csv`
- action distribution：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta/fig6_action_distribution_freq.csv`

