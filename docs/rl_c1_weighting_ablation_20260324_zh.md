# RL C1 Weighting Ablation (20260324)

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

- 最优组：`W0 margin_baseline`
- `closure_8to14 = -0.0360`
- `rl_matches_oracle_8to14 = 47.19%`
- `positive_recall_8to14 = 55.36%`
- `c10 = 0.0848`
- `c12 = 0.0349`
- `action11_freq = 16.82%`

各组 rollout-r3 主指标：
- `W0 margin_baseline`: closure_8to14=-0.0360, rl_match_8to14=47.19%, positive_recall_8to14=55.36%, c10=0.0848, c12=0.0349, action11=16.82%
- `W1 margin_pos`: closure_8to14=-0.0528, rl_match_8to14=47.22%, positive_recall_8to14=54.15%, c10=0.1026, c12=0.0105, action11=14.74%
- `W2 margin_unc`: closure_8to14=-0.0706, rl_match_8to14=46.77%, positive_recall_8to14=52.71%, c10=0.0610, c12=-0.0233, action11=2.71%
- `W3 margin_pos_unc`: closure_8to14=-0.0884, rl_match_8to14=46.42%, positive_recall_8to14=52.61%, c10=0.0253, c12=-0.0198, action11=7.46%

## 6. 输出文件

- 统一对比表：`D:\AIIDE_CODE\project\results\weighting_ablation_20260324\weighting_ablation_main_table_20260324.csv`
- 分 split 对比：`D:\AIIDE_CODE\project\results\weighting_ablation_20260324\weighting_ablation_comparison_by_split_20260324.csv`
- per-SNR 聚合：`D:\AIIDE_CODE\project\results\weighting_ablation_20260324\weighting_ablation_per_snr_agg_20260324.csv`
- 动作分布：`D:\AIIDE_CODE\project\results\weighting_ablation_20260324\weighting_ablation_action_distribution_rollout_r3_20260324.csv`
- BER vs SNR：`D:\AIIDE_CODE\project\figures\weighting_ablation_20260324\weighting_ablation_ber_vs_snr_rollout_r3_20260324.png`
- closure vs SNR：`D:\AIIDE_CODE\project\figures\weighting_ablation_20260324\weighting_ablation_closure_vs_snr_rollout_r3_20260324.png`
- rl_matches_oracle vs SNR：`D:\AIIDE_CODE\project\figures\weighting_ablation_20260324\weighting_ablation_rl_match_vs_snr_rollout_r3_20260324.png`
- positive recall vs SNR：`D:\AIIDE_CODE\project\figures\weighting_ablation_20260324\weighting_ablation_positive_recall_vs_snr_rollout_r3_20260324.png`

## 7. 结论

- 本轮没有任何一组在关键三指标上稳定超过 `W0`，说明单纯样本重加权还不足以把 8–14 dB 的 headroom 学回来。
- 当前最优仍是 baseline，说明本轮三种额外强调方式都还没有超过原始 `oracle_margin`。
- 10/12 dB 仍然没有明显学回 headroom，这意味着下一轮更适合进入 reward shaping，而不是继续只调同类 weighting。
- 动作分布没有进一步塌向 action 11。
- 当前结论仍然只基于单 seed，后续若要确认方向是否稳，需要补多 seed 复验。

## 8. 下一步建议

- 建议开始改 reward，而不是继续在同一层 sample weighting 上做大范围搜索。
