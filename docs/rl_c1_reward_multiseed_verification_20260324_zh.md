# RL C1 Reward Multi-Seed Verification (20260324)

## 1. 本轮目的

固定当前 best 主线 `prev_action_only + physical_delta + r3 + MLP + W0 margin_baseline + OAMPNet`，不再发散 reward 定义，只验证 `R1 reward_static` 相对 `R0 reward_current` 的改善是否在多 seed 下稳定存在。

## 2. 实验设置

- reward modes: `R0 reward_current, R1 reward_static, R2 reward_headroom`
- seeds: `41, 42, 43`
- train data: `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz`
- rollout eval data: `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r3.npz`
- weighting fixed at `margin_baseline`
- reward beta fixed at `0.8`

## 3. 跨 Seed 聚合

- `R0 reward_current`: closure_8to14=0.0740±0.0184, rl_match_8to14=47.03%±0.23%, positive_recall_8to14=54.59%±0.56%, c10=0.0720±0.0126, c12=0.0107±0.0213
- `R1 reward_static`: closure_8to14=0.1019±0.0146, rl_match_8to14=48.07%±0.16%, positive_recall_8to14=54.96%±0.15%, c10=0.1039±0.0114, c12=0.0388±0.0061
- `R2 reward_headroom`: closure_8to14=0.0966±0.0141, rl_match_8to14=47.97%±0.18%, positive_recall_8to14=54.59%±0.28%, c10=0.1028±0.0139, c12=0.0176±0.0111

## 4. R1 - R0 Paired Delta

- seed=41: delta_closure_8to14=+0.0235, delta_rl_match_8to14=+1.34%, delta_positive_recall_8to14=+0.67%, delta_c10=+0.0340, delta_c12=+0.0610
- seed=42: delta_closure_8to14=+0.0212, delta_rl_match_8to14=+1.09%, delta_positive_recall_8to14=-0.60%, delta_c10=+0.0314, delta_c12=-0.0047
- seed=43: delta_closure_8to14=+0.0390, delta_rl_match_8to14=+0.71%, delta_positive_recall_8to14=+1.06%, delta_c10=+0.0302, delta_c12=+0.0279

汇总：
- `delta_closure_8to14`: mean=+0.0279, std=0.0079, positive=3/3
- `delta_rl_match_8to14`: mean=+0.0105, std=0.0026, positive=3/3
- `delta_positive_recall_8to14`: mean=+0.0038, std=0.0071, positive=2/3
- `delta_c10`: mean=+0.0319, std=0.0016, positive=3/3
- `delta_c12`: mean=+0.0281, std=0.0268, positive=2/3

## 5. 结论

- 结论判定：`强阳性`。R1 相对 R0 在 closure / rl_match 上稳定为正，且 c10 或 c12 至少有一个跨 seed 保持正向。
- `R1` mean closure_8to14: 0.0740 -> 0.1019
- `R1` mean rl_matches_oracle_8to14: 47.03% -> 48.07%
- `R1` mean positive_recall_8to14: 54.59% -> 54.96%
- `R0` mean c10/c12: 0.0720 / 0.0107
- `R1` mean c10/c12: 0.1039 / 0.0388
- `R2` 保留价值主要在 c10：mean c10=0.1028，但 mean c12=0.0176。

## 6. 输出文件

- per-seed 主表：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_per_seed_20260324.csv`
- aggregated 主表：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_aggregated_20260324.csv`
- aggregated by metric：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_aggregated_by_metric_20260324.csv`
- paired delta：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_paired_delta_R1_minus_R0_20260324.csv`
- paired delta summary：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_paired_delta_summary_R1_minus_R0_20260324.csv`
- per-SNR 明细：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_per_snr_20260324.csv`
- per-SNR 聚合：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_per_snr_aggregated_20260324.csv`
- action distribution：`D:\AIIDE_CODE\project\results\reward_multiseed_verification_20260324\reward_multiseed_action_distribution_rollout_r3_20260324.csv`
- closure 图：`D:\AIIDE_CODE\project\figures\reward_multiseed_verification_20260324\reward_multiseed_closure_8to14_20260324.png`
- rl_match 图：`D:\AIIDE_CODE\project\figures\reward_multiseed_verification_20260324\reward_multiseed_rl_matches_oracle_8to14_20260324.png`
- c10/c12 图：`D:\AIIDE_CODE\project\figures\reward_multiseed_verification_20260324\reward_multiseed_c10_c12_20260324.png`
- delta 图：`D:\AIIDE_CODE\project\figures\reward_multiseed_verification_20260324\reward_multiseed_delta_R1_minus_R0_20260324.png`

## 7. 下一步建议

- 建议进入 `R1 reward_static` 的轻量参数细调，不需要继续发散到新的 reward 族。
