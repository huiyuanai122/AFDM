# RL C1 计划与当前最好结果总总结

日期：2026-03-22

## 1. 一句话结论

这条主线已经按原计划完整走完。

当前最好、最值得保留的模型仍然是：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.pt`

它对应的主指标是：

- original：`ber_gain_vs_base = 1.599%`
- fixed probe `dagger_r1`：`1.619%`
- current rollout `dagger_r3`：`1.628%`

后续两条“结构升级”分支都已经做过：

- `stay/adjust + direction/magnitude` 两阶段动作头：负结果
- `GRU / temporal policy`：负结果

所以当前最优停止点仍然是：

- `state redesign + self-rollout to r3 + oracle_margin weighting`


## 2. 原计划与执行状态

原计划可以概括成 6 个阶段。

### 阶段 1：修 train-test mismatch 的基础设施

目标：

- 不再硬编码 state 列号
- 能按 feature name 精确找到 history block
- 能把 offline dataset 里的 history 重写成 policy 自己 rollout 出来的 history

已完成的代码：

- `python/rl_c1/features.py`
- `python/rl_c1/env_c1_bandit.py`
- `python/rl_c1/build_dagger_rollout_dataset.py`

结论：

- 这一步是对的
- `history mismatch` 确实是主问题，不是猜测


### 阶段 2：先做最小 DAgger / mixed training 验证方向

目标：

- 先不改 MATLAB online 推理
- 只重写 history
- 直接复用现有 Python 训练和评估链路

已完成的代码与数据：

- `python/rl_c1/build_mixed_policy_dataset.py`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz`

结论：

- 方向是对的
- pure `dagger_r1` 不够稳
- mixed training 明显比 pure dagger 好


### 阶段 3：做 state ablation，确认真正的 shortcut 在哪

目标：

- 先验证问题是不是 history shortcut
- 不急着上 GRU，先做最有信息量的消融

已完成的结果：

- `prev_reward / recent_switch_rate` 是最脆弱的 history 分量
- `prev_action_only + physical_delta` 是更好的 state 主线

对应文档：

- `docs/rl_c1_state_ablation_plan_20260322_zh.md`
- `docs/rl_c1_state_ablation_round1_results_20260322_zh.md`

关键结论：

- `history shortcut` 被直接验证
- 新 state 不只是更稳，而且在 probe 上已经优于旧主线


### 阶段 4：基于新 state 做 self-rollout

目标：

- 不停留在 probe 验证
- 用新 state 自己的 policy 再 rollout，找新的 fixed point

已完成轮次：

- `r2`
- `r3`
- `r4`

结论：

- `r2` 正向
- `r3` 是 unweighted 最佳停止点
- `r4` 开始回退

对应文档：

- `docs/rl_c1_self_rollout_rounds_summary_20260322_zh.md`


### 阶段 5：做 reward / sample weighting

目标：

- 不再继续盲堆 rollout 深度
- 让训练更关注真正值得调整 C1 的样本

已完成方案：

- `oracle_gap_base`：无效
- `oracle_margin`：有效

当前最优配置：

- `sample_weight_mode = oracle_margin`
- `sample_weight_alpha = 1.0`
- `sample_weight_quantile = 0.75`
- `sample_weight_cap = 4.0`

对应文档：

- `docs/rl_c1_reward_weighting_summary_20260322_zh.md`

这一步之后形成了当前最佳主线：

- `mix_r3_prev_action_only_physical_delta_gapw_margin`


### 阶段 6：做结构升级

原计划最后一步是：

- `GRU / temporal policy`
- 或两阶段动作头 `stay/adjust + direction/magnitude`

这两条都已经做完。

#### 6.1 两阶段动作头

实验目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_factorized_gapw_margin`

结果：

- original：`0.543%`
- fixed probe `dagger_r1`：`0.539%`
- current rollout `dagger_r3`：`0.852%`

结论：

- 明显劣于当前最佳 MLP
- 动作分布回塌到 `action 11`
- 不值得继续沿这条线做 rollout

#### 6.2 GRU / temporal policy

实验目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gru_gapw_margin`

结果：

- original：`0.309%`
- fixed probe `dagger_r1`：`0.317%`
- current rollout `dagger_r3`：`0.363%`

结论：

- 也明显劣于当前最佳 MLP
- 虽然推理逻辑更像“真正的时序模型”，但在当前 state 和目标下并没有转成 BER 收益


## 3. 当前最好结果到底是哪一版

### 3.1 最好的 checkpoint

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.pt`

### 3.2 最好的结果目录

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin`

### 3.3 最关键的 3 个 summary

- original：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_orig_summary.json`
- fixed probe：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_dagger_probe_r1_summary.json`
- current rollout：
  `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/sequence_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin_dagger_r3_summary.json`

### 3.4 关键指标

相对旧 paper baseline：

- old baseline：`1.121%`
- current best：`1.599%`

也就是说，当前最好模型已经把主 BER 指标从 `1.121%` 推到了 `1.599%`。

当前最好模型在三个口径下分别是：

- original：`avg_ber_policy = 0.0341158`，`ber_gain_vs_base = 1.599%`
- fixed probe `dagger_r1`：`avg_ber_policy = 0.0341087`，`ber_gain_vs_base = 1.619%`
- current rollout `dagger_r3`：`avg_ber_policy = 0.0341056`，`ber_gain_vs_base = 1.628%`

这说明它不只是“在原始离线集上分数高”，而且在 on-policy shifted state 分布下也稳定。


## 4. 为什么这版是当前最好停止点

有 3 个原因。

### 4.1 它同时兼顾了 original / probe / rollout

有些模型只在原始集上好，有些模型只在自己的 rollout 集上好。

这版最好模型不是单点最强，而是三个口径一起最平衡：

- original 不掉
- fixed probe 不掉
- current rollout 继续提升

### 4.2 它证明“真正的问题找对了”

这条主线不是靠调超参数碰出来的，而是顺着问题定位一步步收敛出来的：

- 先确认 `history mismatch`
- 再去掉最脆弱的 history shortcut
- 再用 self-rollout 找更稳的 state 分布
- 最后用 `oracle_margin` weighting 拉高关键样本权重

所以这版结果具有解释性，不只是数字更高。

### 4.3 它比后面的结构升级更有效

后面两条看起来更“先进”的结构升级都输了：

- factorized head：`0.543%`
- GRU：`0.309%`

这反过来说明，当前瓶颈首先不是“模型不够复杂”，而是：

- state 设计是否避开 shortcut
- rollout 分布是否对齐
- 训练目标是否更关注真正值得切换的帧


## 5. 当前最好模型的行为特征

这版模型并不是“完美健康”，但它在当前所有版本里最平衡。

### 5.1 好的地方

- `SNR_12dB policy_recovery_fraction = 13.25%`
- `policy_switch_rate@12dB = 0.1629`
- `policy_avg_unique_actions@12dB = 4.19`

这些数说明：

- 它在关键中 SNR 区间比旧主线更能恢复到正确动作
- 它不是完全静态
- 它也没有像失败的结构升级那样把动作空间又压缩回去

### 5.2 还不完美的地方

动作分布仍有偏置，top actions 是：

- `action 1`：`49.895%`
- `action 11`：`16.110%`
- `action 14`：`3.235%`

这说明：

- `action 11` 的大塌缩已经被明显修掉
- 但策略仍然没有真正达到 oracle 那种健康的动态控制


## 6. 最好结果对应的图

最推荐看的两张图是：

### 6.1 最直接的“优化 C1 vs 不优化 C1”

- `figures/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/static_dynamic_baselines_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.png`

### 6.2 论文风格 BER-SNR 主图

- `figures/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/fig3_ber_vs_snr_main.png`


## 7. 现在已经明确被排除的方向

到目前为止，下面这些已经不值得继续盲做：

- 继续做纯 rollout 深化到 `r5/r6`
- 继续沿 factorized head 往下堆
- 继续沿当前这版 GRU 往下堆

原因不是“这些方向永远不行”，而是：

- 在当前 state 和当前目标定义下，它们已经被实验证明不如现有最佳主线


## 8. 当前最合理的结论

如果现在的目标是：

- 做汇报
- 写阶段总结
- 固定一版最优结果

那么应该保留并主推：

- `mix_r3_prev_action_only_physical_delta_gapw_margin`

如果还要继续研究下一步，那么最合理的方向不是继续盲做 rollout 或结构升级，而是：

- 在 state 里加入更强的 detector uncertainty / posterior feature
- 或进一步调整训练目标，而不是仅靠更复杂的 policy 结构


## 9. 最终建议

当前阶段建议把这条线的“最好结果”定格为：

- `state = prev_action_only + physical_delta`
- `self-rollout depth = r3`
- `training = oracle_margin weighting`
- `model = MLP policy`

这是截至目前：

- 数字最好
- 行为最平衡
- 解释最清楚
- 最适合继续对外汇报的一版
