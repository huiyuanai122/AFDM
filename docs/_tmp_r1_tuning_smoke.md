# RL C1 R1 Light Tuning (20260324_smoke)

## Goal

Keep the validated R1 reward_static direction fixed as the new baseline and test only very small, interpretable reward variants.

## Runs

- T1 R1-mix-0.70: mode=reward_static_mix, mix_lambda=0.7, neg_scale=1.25

## Aggregated

- T1 R1-mix-0.70: closure_8to14=-0.0186+/-0.0000, rl_match_8to14=43.13%+/-0.00%, positive_recall_8to14=50.98%+/-0.00%, c10=-0.0438+/-0.0000, c12=-0.0477+/-0.0000

## Paired Delta vs T0

- T1 R1-mix-0.70
  closure_8to14: mean=+nan, std=nan, positive=0/0
  rl_match_8to14: mean=+nan, std=nan, positive=0/0
  positive_recall_8to14: mean=+nan, std=nan, positive=0/0
  c10: mean=+nan, std=nan, positive=0/0
  c12: mean=+nan, std=nan, positive=0/0
- t2_r1_mix_085
  closure_8to14: mean=+nan, std=nan, positive=0/0
  rl_match_8to14: mean=+nan, std=nan, positive=0/0
  positive_recall_8to14: mean=+nan, std=nan, positive=0/0
  c10: mean=+nan, std=nan, positive=0/0
  c12: mean=+nan, std=nan, positive=0/0
- t3_r1_asym_125
  closure_8to14: mean=+nan, std=nan, positive=0/0
  rl_match_8to14: mean=+nan, std=nan, positive=0/0
  positive_recall_8to14: mean=+nan, std=nan, positive=0/0
  c10: mean=+nan, std=nan, positive=0/0
  c12: mean=+nan, std=nan, positive=0/0

## Decision

- Keep T0 R1-base as the baseline. None of the light variants stably beat T0 on the primary criteria.

## Outputs

- per-seed: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_per_seed_20260324_smoke.csv
- aggregated: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_aggregated_20260324_smoke.csv
- aggregated_by_metric: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_aggregated_by_metric_20260324_smoke.csv
- paired_delta: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_paired_delta_vs_t0_20260324_smoke.csv
- paired_delta_summary: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_paired_delta_summary_vs_t0_20260324_smoke.csv
- per_snr: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_per_snr_20260324_smoke.csv
- action_distribution: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_action_distribution_rollout_r3_20260324_smoke.csv
- action_topk_summary: D:\AIIDE_CODE\project\results\_tmp_r1_tuning_smoke\r1_light_tuning_action_topk_summary_20260324_smoke.csv
- closure_fig: D:\AIIDE_CODE\project\figures\_tmp_r1_tuning_smoke\r1_light_tuning_closure_8to14_20260324_smoke.png
- rl_match_fig: D:\AIIDE_CODE\project\figures\_tmp_r1_tuning_smoke\r1_light_tuning_rl_matches_oracle_8to14_20260324_smoke.png
- c10_c12_fig: D:\AIIDE_CODE\project\figures\_tmp_r1_tuning_smoke\r1_light_tuning_c10_c12_20260324_smoke.png
- paired_delta_fig: D:\AIIDE_CODE\project\figures\_tmp_r1_tuning_smoke\r1_light_tuning_paired_delta_20260324_smoke.png
- action_topk_fig: D:\AIIDE_CODE\project\figures\_tmp_r1_tuning_smoke\r1_light_tuning_action_topk_20260324_smoke.png
