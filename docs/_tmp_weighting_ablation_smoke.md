# RL C1 Weighting Ablation (smoke20260324)

## 1. 本轮目的

固定当前 best 主线 `prev_action_only + physical_delta + r3 + MLP + OAMPNet`，不再扩 state，只改 sample weighting，验证关键样本强调是否能比 direct uncertainty-state 注入更有效。

主指标仍然只看：
- `closure_8to14`
- `rl_matches_oracle_8to14`
- `positive_recall_8to14`

## 2. 四组 weighting 定义

- `W0 margin_baseline`: 现有 `oracle_margin` baseline，原始 trainer 内部实现为 `1 + alpha * clip(margin / q75, 0, cap)`，再做 train-mean 归一化。
- `W1 margin_pos`: `W0 * (1 + lambda_pos * I[oracle_beats_static_best=1])`。
- `W2 margin_unc`: `W0 * (1 + lambda_unc * unc_score_norm)`。
- `W3 margin_pos_unc`: `W0 * (1 + lambda_pos * I[oracle_beats_static_best=1]) * (1 + lambda_unc * unc_score_norm)`。

本轮固定参数：
- `alpha = 1.0`
- `lambda_pos = 1.0`
- `lambda_unc = 1.0`

## 3. uncertainty score 定义

- 使用 `sample_weight_csv` 中已经对齐好的样本级 uncertainty 文件。
- 先取三个特征：`posterior_entropy_p90`、`1 - posterior_margin_p10`、`tau_p90`。
- 各自仅用 train split 统计量做 z-score。
- 组合分数：`unc_score_raw = mean(z(entropy_p90), z(1-margin_p10), z(tau_p90))`。
- 再做 `clip([-2.0, +2.0])`，最后用 `relu` 映射到非负强调因子。

## 4. 数据与设置

- 训练集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz`
- 评估集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz` / `dagger_r1` / `dagger_r3`
- sample-level uncertainty CSV：`results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/oampnet_uncertainty_sample_level_20260323.csv`
- seed 数：`1`
- 当前这轮是否单 seed：`是`

## 5. 关键对比

- 最优组：`W3 margin_pos_unc`
- `closure_8to14 = -0.1180`
- `rl_matches_oracle_8to14 = 43.16%`
- `positive_recall_8to14 = 51.41%`
- `c10 = -0.0438`
- `c12 = -0.0477`
- `action11_freq = 0.01%`

各组 rollout-r3 主指标：
- `W3 margin_pos_unc`: closure_8to14=-0.1180, rl_match_8to14=43.16%, positive_recall_8to14=51.41%, c10=-0.0438, c12=-0.0477, action11=0.01%

## 6. 输出文件

- 统一对比表：`D:\AIIDE_CODE\project\results\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_main_table_smoke20260324.csv`
- 分 split 对比：`D:\AIIDE_CODE\project\results\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_comparison_by_split_smoke20260324.csv`
- per-SNR 聚合：`D:\AIIDE_CODE\project\results\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_per_snr_agg_smoke20260324.csv`
- 动作分布：`D:\AIIDE_CODE\project\results\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_action_distribution_rollout_r3_smoke20260324.csv`
- BER vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_ber_vs_snr_rollout_r3_smoke20260324.png`
- closure vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_closure_vs_snr_rollout_r3_smoke20260324.png`
- rl_matches_oracle vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_rl_match_vs_snr_rollout_r3_smoke20260324.png`
- positive recall vs SNR：`D:\AIIDE_CODE\project\figures\_tmp_weighting_ablation_smoke_analysis\weighting_ablation_positive_recall_vs_snr_rollout_r3_smoke20260324.png`

## 7. 结论

- 运行结果不完整，当前还不能判断哪组 weighting 最优。
