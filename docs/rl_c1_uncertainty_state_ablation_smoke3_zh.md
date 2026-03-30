# RL C1 Compact Uncertainty-State Ablation (smoke20260323b)

## 1. 本轮目的

在当前 best 主线 `mix_r3_prev_action_only_physical_delta_gapw_margin` 上，只改 RL state，验证 compact detector-aware uncertainty 特征是否能提升：
- RL dynamic 对 static best 的超越能力
- RL 对 oracle 的逼近程度
- 8–14 dB 关键区间的 headroom closure

## 2. 测试的 4 组 state

- `S0 baseline`: prev_action_only + physical_delta
- `S1 posterior-only`: baseline + posterior_entropy_p90 + posterior_margin_p10
- `S2 tau-only`: baseline + tau_mean + tau_p90
- `S3 posterior+tau`: baseline + posterior_entropy_p90 + posterior_margin_p10 + tau_mean + tau_p90

新增 uncertainty 特征直接接入 dataset，训练时复用 `train_reinforce.py` 现有 train-split z-score 归一化；因此均值/方差只由训练 split 统计，验证和评估复用 checkpoint 中的 `state_mean/state_std`。

## 3. 数据与评估口径

- 训练集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz` 的四个 state variant
- 评估集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz` / `dagger_r1` / `dagger_r3` 对应四个 state variant
- uncertainty 特征来源：`results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/oampnet_uncertainty_sample_level_20260323.csv`
- seed 数：`1`
- reward / rollout / policy 结构保持不变：仍是 `oracle_margin` weighting + `r3` + MLP

## 4. Current Rollout r3 主结果

- 最优 state：`S0 baseline`
- `closure_ratio_8to14 = -0.0360`
- `rl_matches_oracle_8to14 = 47.19%`
- `positive_recall_8to14 = 55.36%`
- `overall avg_ber_policy = 3.410562e-02`

## 5. posterior-only / tau-only / 组合的比较


## 6. 结果文件

- comparison CSV: `D:\AIIDE_CODE\project\results\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_comparison_by_split_smoke20260323b.csv`
- main table CSV: `D:\AIIDE_CODE\project\results\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_main_table_smoke20260323b.csv`
- per-SNR CSV: `D:\AIIDE_CODE\project\results\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_per_snr_agg_smoke20260323b.csv`
- action distribution CSV: `D:\AIIDE_CODE\project\results\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_action_distribution_rollout_r3_smoke20260323b.csv`
- BER vs SNR figure: `D:\AIIDE_CODE\project\figures\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_ber_vs_snr_rollout_r3_smoke20260323b.png`
- closure ratio figure: `D:\AIIDE_CODE\project\figures\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_closure_vs_snr_rollout_r3_smoke20260323b.png`
- rl_matches_oracle figure: `D:\AIIDE_CODE\project\figures\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_rl_match_vs_snr_rollout_r3_smoke20260323b.png`
- positive recall figure: `D:\AIIDE_CODE\project\figures\uncertainty_state_ablation_smoke3\uncertainty_state_ablation_positive_recall_vs_snr_rollout_r3_smoke20260323b.png`

## 7. 改动文件

- `python/rl_c1/features.py`
- `python/rl_c1/build_state_variant_dataset.py`
- `python/rl_c1/train_reinforce.py`
- `python/rl_c1/run_uncertainty_state_ablation.py`
- `python/rl_c1/analyze_uncertainty_state_ablation.py`

## 8. 运行方式

```bash
python python/rl_c1/run_uncertainty_state_ablation.py
```

## 9. 下一步建议

- 本轮若提升有限，则应先在 compact uncertainty-state 方案里继续筛选和稳健化，再决定是否转去 reward。
