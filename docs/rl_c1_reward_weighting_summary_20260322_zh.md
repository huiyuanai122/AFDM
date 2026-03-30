# RL C1 Reward Weighting 总结

日期：2026-03-22

## 1. 为什么做这一步

在 state redesign 和 self-rollout 之后，主线已经收敛到：

- state：`prev_action_only + physical_delta`
- rollout 深度：`r3` 最优，`r4` 开始回退

接下来最自然的下一步就是改训练目标，而不是继续堆 rollout 深度。

这轮做的事情是：

- 在 `train_reinforce.py` 里加入最小的 sample weighting 机制
- 用 reward table 的 oracle-gap 统计去强调“更值得调 C1”的帧


## 2. 代码改动

文件：

- `python/rl_c1/train_reinforce.py`

新增能力：

- `--sample_weight_mode`
  - `none`
  - `oracle_gap_base`
  - `oracle_margin`
- `--sample_weight_alpha`
- `--sample_weight_quantile`
- `--sample_weight_cap`

当前实现方式：

1. 先从 reward table 计算每帧分数
2. 再按 train split 的 quantile 做归一化
3. 构造 sample weight
4. 只把权重作用到训练损失
   - imitation warm-start 的 CE
   - reinforce 阶段的 policy loss
   - reinforce 阶段的 imitation preserve CE

权重不会改动：

- 评估口径
- reward table 本身
- sequence-level 评估逻辑


## 3. 两种 weighting 的定义

### 3.1 `oracle_gap_base`

定义：

- `score = reward_top1 - reward_base`

在当前数据里，因为 `reward_relbase_proxy` 的 base action reward 就是 0，
所以它本质上等价于：

- `score = reward_top1`

这更像是在强调“相对 base 的绝对收益大不大”。


### 3.2 `oracle_margin`

定义：

- `score = reward_top1 - reward_top2`

这更像是在强调：

- 当前帧里最优动作是否明显优于次优动作
- 也就是“这帧决策边界清不清楚”


## 4. 第一轮 weighting 结果

训练数据统一使用：

- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz`

对照基线是当前未加权最佳：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta`


### 4.1 `oracle_gap_base` 不成功

实验目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_base`

结果：

- original：`1.154%`
- fixed probe `dagger_r1`：`1.132%`

对比未加权 `r3`：

- original：`1.374%`
- fixed probe：`1.410%`

结论：

- `oracle_gap_base` 这个加权方向不对
- 它把训练过度推向了“相对 base 收益大的帧”
- 但没有转成更好的最终 sequence BER


### 4.2 `oracle_margin` 是正向的

实验目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin`

配置：

- `sample_weight_mode = oracle_margin`
- `sample_weight_alpha = 1.0`
- `sample_weight_quantile = 0.75`
- `sample_weight_cap = 4.0`

关键结果：

- original：`1.599%`
- fixed probe `dagger_r1`：`1.619%`
- current rollout `dagger_r3`：`1.628%`

相对未加权 `r3`：

- original：`1.374% -> 1.599%`
- fixed probe：`1.410% -> 1.619%`
- current rollout：`1.459% -> 1.628%`

这已经是明显的正向结果。


### 4.3 更保守的 `alpha=0.5` 反而不如 `alpha=1.0`

实验目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_a05`

结果：

- original：`1.350%`
- fixed probe：`1.407%`
- current rollout：`1.539%`

结论：

- `oracle_margin` 方向是对的
- 但 `alpha=0.5` 不如 `alpha=1.0`


## 5. weighted self-rollout 验证

在 `oracle_margin alpha=1.0` 这个正向配置上，又继续做了一轮 self-rollout：

### 5.1 生成新 rollout 数据

- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r4_gapw_margin.npz`

### 5.2 mixed + weighted 训练

- `results/paper_tsv2seq_vdop_ctrl_mix_r4_prev_action_only_physical_delta_gapw_margin`

### 5.3 结果

- original：`1.582%`
- fixed probe `dagger_r1`：`1.592%`
- current rollout `dagger_r4_gapw_margin`：`1.747%`

这说明：

- 加权目标和 self-rollout 可以继续形成新的 fixed point
- 但从 balanced 角度看，`r4_gapw_margin` 不如 `r3_gapw_margin`

因为：

- current rollout 上它更高
- 但 original 和 fixed probe 都略低于 `r3_gapw_margin`


## 6. 当前最佳模型是谁

当前全线最新、最强的平衡点是：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.pt`

对应关键 summary：

- original：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_orig_summary.json`
- fixed probe：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_dagger_probe_r1_summary.json`
- current rollout：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_dagger_r3_summary.json`


## 7. 当前最佳模型相对旧最佳提升了多少

相对上一版未加权最佳 `r3`：

- original：`1.374% -> 1.599%`
- fixed probe：`1.410% -> 1.619%`
- current rollout：`1.459% -> 1.628%`

相对更早的 paper baseline：

- old baseline：`1.121%`
- current best：`1.599%`

也就是说：

- 这条线现在不只是“修好了 history mismatch”
- 而是已经把最终 BER 主指标继续往前推了一大截


## 8. 行为侧怎么看

### 8.1 好的地方

和未加权 `r3` 比：

- `12dB recovery`：`9.21% -> 13.25%`
- low/mid SNR 平均 switch gap：
  `0.3315 -> 0.3300`

说明：

- `oracle_margin` weighting 不是靠单纯“更静态”拿到收益
- 在关键中低 SNR 段，它确实提升了恢复能力


### 8.2 不完美的地方

它仍然有一个明显副作用：

- action 11 频率从未加权 `r3` 的 `13.645%`
  升到 `16.11%`

所以当前状态不是“完全更健康”，而是：

- BER 更好
- 关键恢复指标更好
- 但动作分布偏置并没有完全消失


## 9. 一个重要观察

这轮还有一个很值得记下来的现象：

- `val_ber` 并不能稳定预测最终 sequence-level BER

例如：

- `gapw_margin` 训练侧 `best val_ber` 其实比未加权 `r3` 更差
- 但 sequence-level 主指标反而明显更高

这说明：

- 仅靠训练时的单一 val_ber 做 early conclusion 不够
- 后续更应该以
  `original + fixed probe + current rollout`
  这三套 sequence-level summary 联合判断


## 10. 当前建议

当前阶段，建议如下：

1. 把 `r3_gapw_margin` 作为新的主模型
2. 不继续使用 `oracle_gap_base`
3. 不继续在 `alpha=0.5` 这类更保守权重上浪费时间
4. 也不建议立刻继续做 `r5_gapw_margin`

因为现在已经能看出：

- state 方向收敛了
- rollout 深度也基本摸清了
- reward weighting 里 `oracle_margin` 已经给出清晰正向结果

下一步如果还要继续提升，更合理的是：

- 在当前 `r3_gapw_margin` 上做更结构化的 reward 设计
- 或在这个基础上再看 GRU / 两阶段动作头

