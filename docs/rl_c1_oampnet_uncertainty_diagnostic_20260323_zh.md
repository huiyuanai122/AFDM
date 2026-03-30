# RL C1 OAMPNet Uncertainty Diagnostic (20260323)

## 1. 本轮任务目的

验证 OAMPNet detector 的 posterior / uncertainty 特征，能否区分“oracle dynamic 相对 static best single C1 仍值得切换”的样本。
本轮不重训 RL，不改 reward，不改 rollout，只做特征信息量诊断。

## 2. 使用的数据 / 模型 / 结果来源

- 原始 20k policy 样本集：`data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz`
- 当前 best policy checkpoint：`results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.pt`
- OAMPNet uncertainty raw export：`results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_raw_20260323.csv`
- OAMPNet uncertainty raw MAT：`results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_raw_20260323.mat`
- uncertainty 特征由 MATLAB helper 重放 `tsv2seq_vdop_ctrl` paper 配置生成，使用 `base_action = 10 (0-based)` 对应的 base C1 观测。
- `oracle_beats_static_best` 与 `oracle_eq_static_best` 由样本级 `ber_actions` 结合“每个 SNR 下的 best static single C1”补算。
- `rl_matches_oracle` 由当前 best checkpoint 在原始 20k 样本上的预测动作，再与 oracle 样本级 BER 对齐补算。

## 3. 导出的 uncertainty 特征

- posterior entropy: mean / std / p90
- posterior margin: mean / p10 / p25
- max posterior probability: mean / p10 / std
- tau statistics: mean / std / p90 / p95 / max

## 4. 8-14 dB 主要统计结果

- 8-14 dB pooled `oracle_beats_static_best` 占比：`56.05%`
- 8-14 dB pooled `rl_matches_oracle` 占比：`46.94%`
- 8-14 dB pooled 单特征 AUC（取更优方向）Top 5：
  - `posterior_entropy_p90`: AUC=`0.9033`, direction=`higher_in_beats`, mean_diff=`0.087802`
  - `posterior_margin_p10`: AUC=`0.9023`, direction=`lower_in_beats`, mean_diff=`-0.048344`
  - `maxprob_p10`: AUC=`0.9013`, direction=`lower_in_beats`, mean_diff=`-0.024670`
  - `tau_mean`: AUC=`0.8993`, direction=`higher_in_beats`, mean_diff=`0.080901`
  - `tau_p90`: AUC=`0.8986`, direction=`higher_in_beats`, mean_diff=`0.081382`

按 SNR 汇总见：
- `results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_summary_20260323.csv`
- `results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_auc_8to14_20260323.csv`

## 5. 三张图的观察

- `figures\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_posterior_margin_20260323.png`
- `figures\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_posterior_entropy_20260323.png`
- `figures\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_tau_p90_20260323.png`
- 若 `oracle_beats_static_best=1` 与 `oracle_eq_static_best=0` 的箱体 / 中位数在多个 SNR 上稳定分离，则说明 detector uncertainty 对“该不该切 C1”有判别力。

## 6. 最终判断

- 本轮 8-14 dB pooled 最强区分特征是 `posterior_entropy_p90`，AUC=`0.9033`。
- 若后验 margin / maxprob 一类特征排名靠前，说明 posterior confidence 本身能提示哪些样本更值得切换。
- 若 tau 一类特征排名靠前，说明 OAMPNet 内部的 symbol-wise uncertainty / effective noise 也携带切换线索。
- 下一步建议：先把这些 uncertainty / posterior 特征加进 RL state。

## 7. 改动文件

- `python/models/oampnet_v4.py`：新增 `return_diagnostics`，可导出最后一层 logits / prob / tau_vec。
- `matlab/detectors/oampnet_detector.m`：新增可选 `diagnostics` 输出。
- `matlab/detectors/export_oampnet_uncertainty_for_policy_dataset.m`：按 paper 配置重放原始样本并导出 uncertainty raw CSV/MAT。
- `python/rl_c1/analyze_oampnet_uncertainty_for_c1.py`：生成 sample CSV / summary CSV / AUC CSV / plots / markdown。

## 8. 运行方式

```bash
python python/rl_c1/analyze_oampnet_uncertainty_for_c1.py --data data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta.npz --checkpoint results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.pt --results_dir results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin --figures_dir figures/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin
```

## 9. 结果输出位置

- sample CSV: `results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_sample_level_20260323.csv`
- summary CSV: `results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_summary_20260323.csv`
- AUC CSV: `results\paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin\oampnet_uncertainty_auc_8to14_20260323.csv`
- markdown: `D:\AIIDE_CODE\project\docs\rl_c1_oampnet_uncertainty_diagnostic_20260323_zh.md`