# RL C1 Reward Ablation (smoke20260324_reward)

## 1. 本轮目的

固定当前 best 主线 `prev_action_only + physical_delta + r3 + MLP + W0 margin_baseline + OAMPNet`，不再扩 state 和 weighting，只替换训练 reward，验证 headroom-aware reward 是否能更好学回关键正样本。

主指标仍然只看：
- `closure_8to14`
- `rl_matches_oracle_8to14`
- `positive_recall_8to14`

## 2. 四组 reward 定义

- `R0 reward_current`: 当前 baseline reward，不改定义。
- `R1 reward_static`: `clip((BER_static_best - BER_action) / (BER_static_best + eps), -1, 1)`。
- `R2 reward_headroom`: `r = (1-beta) * r_static + beta * pos * r_headroom`，其中 `r_headroom = clip((BER_static_best - BER_action) / (BER_static_best - BER_oracle + eps), -1, 1)`。
- `R3 reward_headroom_unc`: `r = (1-beta) * r_static + beta * (0.2 + 0.8 * g_unc) * pos * r_headroom`。

- 本轮固定 `beta = 0.8`

## 3. uncertainty gate 定义

- 只在 `R3` 中使用，不进入 state，也不作为额外 sample weighting。
- 使用 `posterior_entropy_p90`、`1 - posterior_margin_p10`、`tau_p90`。
- 三者只用 train split 统计量做 z-score，简单平均得到 `unc_raw`。
- 再做 `clip([-2.0, +2.0])`，最后用 `sigmoid` 映射到 `[0,1]` gate。

## 4. 数据与设置

- 训练集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz`
- 评估集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz` / `dagger_r1` / `dagger_r3`
- sample-level uncertainty CSV：`results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/oampnet_uncertainty_sample_level_20260323.csv`
- seed 数：`1`
- 当前这轮是否单 seed：`是`

## 5. 关键对比

- 最优组：`R3 reward_headroom_unc`
- `closure_8to14 = -0.1159`
- `rl_matches_oracle_8to14 = 43.33%`
- `positive_recall_8to14 = 51.02%`
- `c10 = -0.0438`
- `c12 = -0.0477`
- `action11_freq = 2.56%`

各组 rollout-r3 主指标：
- `R3 reward_headroom_unc`: closure_8to14=-0.1159, rl_match_8to14=43.33%, positive_recall_8to14=51.02%, c10=-0.0438, c12=-0.0477, action11=2.56%

## 6. 输出文件

- 统一对比表：`D:\AIIDE_CODE\project\results\_tmp_reward_ablation_smoke_analysis\reward_ablation_main_table_smoke20260324_reward.csv`
- 分 split 对比：`D:\AIIDE_CODE\project\results\_tmp_reward_ablation_smoke_analysis\reward_ablation_comparison_by_split_smoke20260324_reward.csv`
- per-SNR 聚合：`D:\AIIDE_CODE\project\results\_tmp_reward_ablation_smoke_analysis\reward_ablation_per_snr_agg_smoke20260324_reward.csv`
- 动作分布：`D:\AIIDE_CODE\project\results\_tmp_reward_ablation_smoke_analysis\reward_ablation_action_distribution_rollout_r3_smoke20260324_reward.csv`
- BER vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_reward_ablation_smoke_analysis\reward_ablation_ber_vs_snr_rollout_r3_smoke20260324_reward.png`
- closure vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_reward_ablation_smoke_analysis\reward_ablation_closure_vs_snr_rollout_r3_smoke20260324_reward.png`
- rl_matches_oracle vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_reward_ablation_smoke_analysis\reward_ablation_rl_match_vs_snr_rollout_r3_smoke20260324_reward.png`
- positive recall vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_reward_ablation_smoke_analysis\reward_ablation_positive_recall_vs_snr_rollout_r3_smoke20260324_reward.png`

## 7. 结论

- 运行结果不完整，当前还不能判断哪组 reward 最优。
