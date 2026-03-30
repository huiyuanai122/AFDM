# RL C1 R1 Light Tuning (20260324)

## Goal

Keep the validated R1 reward_static direction fixed as the new baseline and test only very small, interpretable reward variants.

## Runs

- T0 R1-base: mode=reward_static, mix_lambda=0.85, neg_scale=1.25
- T1 R1-mix-0.70: mode=reward_static_mix, mix_lambda=0.7, neg_scale=1.25
- T2 R1-mix-0.85: mode=reward_static_mix, mix_lambda=0.85, neg_scale=1.25
- T3 R1-asym-1.25: mode=reward_static_asym, mix_lambda=0.85, neg_scale=1.25

## Aggregated

- T0 R1-base: closure_8to14=0.1019+/-0.0146, rl_match_8to14=48.07%+/-0.16%, positive_recall_8to14=54.96%+/-0.15%, c10=0.1039+/-0.0114, c12=0.0388+/-0.0061
- T1 R1-mix-0.70: closure_8to14=0.0615+/-0.0580, rl_match_8to14=46.23%+/-2.20%, positive_recall_8to14=53.13%+/-1.54%, c10=0.0514+/-0.0686, c12=-0.0014+/-0.0349
- T2 R1-mix-0.85: closure_8to14=0.0996+/-0.0125, rl_match_8to14=47.92%+/-0.10%, positive_recall_8to14=54.69%+/-0.42%, c10=0.1019+/-0.0134, c12=0.0300+/-0.0099
- T3 R1-asym-1.25: closure_8to14=0.0998+/-0.0167, rl_match_8to14=48.04%+/-0.14%, positive_recall_8to14=55.16%+/-0.24%, c10=0.1006+/-0.0123, c12=0.0403+/-0.0045

## Paired Delta vs T0

- T1 R1-mix-0.70
  closure_8to14: mean=-0.0404, std=0.0454, positive=0/3
  rl_match_8to14: mean=-0.0185, std=0.0216, positive=0/3
  positive_recall_8to14: mean=-0.0183, std=0.0161, positive=0/3
  c10: mean=-0.0525, std=0.0574, positive=0/3
  c12: mean=-0.0401, std=0.0403, positive=1/3
- T2 R1-mix-0.85
  closure_8to14: mean=-0.0022, std=0.0025, positive=1/3
  rl_match_8to14: mean=-0.0016, std=0.0007, positive=0/3
  positive_recall_8to14: mean=-0.0027, std=0.0047, positive=1/3
  c10: mean=-0.0020, std=0.0082, positive=2/3
  c12: mean=-0.0087, std=0.0118, positive=1/3
- T3 R1-asym-1.25
  closure_8to14: mean=-0.0021, std=0.0036, positive=1/3
  rl_match_8to14: mean=-0.0004, std=0.0002, positive=0/3
  positive_recall_8to14: mean=+0.0020, std=0.0013, positive=3/3
  c10: mean=-0.0033, std=0.0113, positive=2/3
  c12: mean=+0.0016, std=0.0038, positive=2/3

## Decision

- Keep T0 R1-base as the baseline. None of the light variants stably beat T0 on the primary criteria.

## Outputs

- per-seed: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_per_seed_20260324.csv
- aggregated: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_aggregated_20260324.csv
- aggregated_by_metric: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_aggregated_by_metric_20260324.csv
- paired_delta: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_paired_delta_vs_t0_20260324.csv
- paired_delta_summary: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_paired_delta_summary_vs_t0_20260324.csv
- per_snr: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_per_snr_20260324.csv
- action_distribution: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_action_distribution_rollout_r3_20260324.csv
- action_topk_summary: D:\AIIDE_CODE\project\results\r1_light_tuning_20260324\r1_light_tuning_action_topk_summary_20260324.csv
- closure_fig: D:\AIIDE_CODE\project\figures\r1_light_tuning_20260324\r1_light_tuning_closure_8to14_20260324.png
- rl_match_fig: D:\AIIDE_CODE\project\figures\r1_light_tuning_20260324\r1_light_tuning_rl_matches_oracle_8to14_20260324.png
- c10_c12_fig: D:\AIIDE_CODE\project\figures\r1_light_tuning_20260324\r1_light_tuning_c10_c12_20260324.png
- paired_delta_fig: D:\AIIDE_CODE\project\figures\r1_light_tuning_20260324\r1_light_tuning_paired_delta_20260324.png
- action_topk_fig: D:\AIIDE_CODE\project\figures\r1_light_tuning_20260324\r1_light_tuning_action_topk_20260324.png
