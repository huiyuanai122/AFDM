# RL C1 Current Best Result Summary

Date: 2026-03-22

## 1. Executive Summary

This document summarizes the current status of the RL-based C1 optimization line under the
`tsv2seq_vdop_ctrl` setting.

There are two different "best" answers depending on the criterion:

1. If the only criterion is the original paper-side offline benchmark
   `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat`,
   the current top result is still the original baseline policy:
   `results/paper_tsv2seq_vdop_ctrl/best_reinforce_policy_tsv2seq_vdop_ctrl_paper.pt`
   with `ber_gain_vs_base = 1.121%`.

2. If the criterion is a better balance between original offline performance and robustness to
   on-policy history mismatch, the current best version is:
   `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`

Its key numbers are:

- On original paper dataset:
  `ber_gain_vs_base = 1.074%`
- On `dagger_r1` rollout dataset:
  `ber_gain_vs_base = 1.110%`
- `SNR_12dB policy_recovery_fraction = 9.16%`
- Action-11 collapse is reduced from `46.88%` in the baseline policy distribution to `20.445%`

Practical recommendation:

- Keep `mix_r1_tinyrl` as the current carry-forward candidate.
- Do not continue pure DAgger rollout depth (`r3`, `r4`, ...) without changing the training recipe.
- `mix_r2_tinyrl` was attempted and did not improve over `mix_r1_tinyrl`.


## 2. Best Artifacts

Current recommended model:

- Checkpoint:
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`
- Training summary:
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_summary_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json`
- Original-dataset evaluation summary:
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig_summary.json`
- On-policy evaluation summary:
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_dagger_summary.json`
- Main policy-side baseline comparison figure:
  `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/static_dynamic_baselines_tsv2seq_vdop_ctrl_mix_r1_tinyrl.png`
- Detector-side BER-vs-SNR figure:
  `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig3_ber_vs_snr_main.png`

Datasets used by the best result:

- Original offline dataset:
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat`
- First rollout dataset:
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz`
- Best mixed dataset:
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz`


## 3. What Was Implemented

The main problem targeted in this round was train-test mismatch in the policy-history block:

- Offline export originally encoded history from the fixed/base action trajectory.
- Online deployment rolls history from the policy's own previous actions.
- Therefore, `prev_action_norm`, `prev_reward`, and `recent_switch_rate` had a distribution shift.

To address this, the following pieces were implemented:

### 3.1 Feature-name indexing utilities

Added in:

- `python/rl_c1/features.py`

Purpose:

- Remove hard-coded state column numbers.
- Resolve history-feature columns by name.

Main helpers:

- `canonicalize_feature_names(...)`
- `feature_name_to_index(...)`
- `feature_index(...)`

### 3.2 Feature-name metadata support in the loader

Added in:

- `python/rl_c1/env_c1_bandit.py`

Purpose:

- Load `feature_names` from both `.npz` and `.mat`.
- Allow later scripts to rewrite state columns safely by name.

### 3.3 DAgger rollout dataset builder

Added in:

- `python/rl_c1/build_dagger_rollout_dataset.py`

Purpose:

- For each sequence, sort by `sequence_id` and `time_index`.
- Roll forward the current checkpoint greedily.
- Rewrite only:
  - `prev_action_norm`
  - `prev_reward`
  - `recent_switch_rate`
- Reuse all original reward tables, labels, BER tables, and metadata.

Initialization used per sequence:

- `prev_action = base_action`
- `prev_reward = 0`
- `switch_hist = 0`

### 3.4 Mixed dataset builder

Added in:

- `python/rl_c1/build_mixed_policy_dataset.py`

Purpose:

- Concatenate the original dataset and the on-policy rollout dataset.
- Preserve schema consistency checks.
- Keep matching `sequence_id` values, so `split_mode=sequence` keeps aligned original/on-policy
  variants in the same split.


## 4. Experiment Path

The experiment path that was actually run is:

1. Start from the current paper baseline:
   `results/paper_tsv2seq_vdop_ctrl`
2. Build `dagger_r1` by rolling out the baseline checkpoint on the original paper dataset.
3. Retrain on pure `dagger_r1`:
   - `dagger_r1_imonly`
   - `dagger_r1_tinyrl`
4. Observe that pure DAgger fixes mismatch symptoms but hurts the main original benchmark.
5. Build `mix_r1 = original + dagger_r1`.
6. Retrain on `mix_r1`:
   - `mix_r1_imonly`
   - `mix_r1_tinyrl`
7. Select `mix_r1_tinyrl` as the best compromise.
8. Build `dagger_r2` using `mix_r1_tinyrl`.
9. Build `mix_r2 = original + dagger_r2`.
10. Retrain `mix_r2_tinyrl`.
11. Observe that `r2` does not improve over `mix_r1_tinyrl`.


## 5. Key Results Table

All BER gains below are `ber_gain_vs_base`.

| Variant | Train Data | best val_ber | Eval on original paper set | Eval on rollout set | Notes |
|---|---|---:|---:|---:|---|
| `paper_tsv2seq_vdop_ctrl` baseline | original only | 0.0424211 | 1.121% | 0.247% on `dagger_r1` eval | Best original benchmark, mismatch-sensitive |
| `dagger_r1_tinyrl` | `dagger_r1` only | 0.0426681 | 0.458% | 0.840% on `dagger_r1` | Fixes mismatch symptoms but hurts original benchmark |
| `mix_r1_tinyrl` | original + `dagger_r1` | 0.0425656 | 1.074% | 1.110% on `dagger_r1` | Current recommended version |
| `mix_r2_tinyrl` | original + `dagger_r2` | 0.0426825 | 0.866% | 0.902% on `dagger_r2` | Regression vs `mix_r1_tinyrl` |

Important interpretation:

- The baseline is still slightly better on the original benchmark.
- `mix_r1_tinyrl` is much more robust under the corrected on-policy history distribution.
- `mix_r2_tinyrl` did not improve either original or on-policy BER.


## 6. Evidence That History Mismatch Is Real

The strongest diagnostic is to evaluate the original baseline checkpoint directly on the
`dagger_r1` rollout dataset, without retraining.

Artifacts:

- `results/paper_tsv2seq_vdop_ctrl_dagger_r1_baselineeval/sequence_tsv2seq_vdop_ctrl_dagger_r1_baselineeval_summary.json`
- `results/paper_tsv2seq_vdop_ctrl_dagger_r1_baselineeval/fig6_action_distribution_freq.csv`

Observed behavior:

- `ber_gain_vs_base` collapses to `0.247%`
- `match_rate` drops to `9.855%`
- `switch_rate_policy` rises to `0.2759`
- Action `11` frequency explodes to `87.955%`

This is the clearest proof that the old training state distribution does not match the
policy-rolled online state distribution.


## 7. Why `mix_r1_tinyrl` Is the Current Best Compromise

Compared with the original baseline:

- Original benchmark BER gain only drops slightly:
  `1.121% -> 1.074%`
- `SNR_12dB policy_recovery_fraction` improves strongly:
  `1.71% -> 9.16%`
- Low/mid-SNR switch-gap to oracle improves:
  mean gap over `0~10 dB` changes from `0.695` to `0.357`
- Action-11 collapse is reduced:
  `46.88% -> 20.445%`

Compared with pure `dagger_r1_tinyrl`:

- Original benchmark BER gain improves strongly:
  `0.458% -> 1.074%`
- On-policy BER gain also improves:
  `0.840% -> 1.110%`

Compared with `mix_r2_tinyrl`:

- Original benchmark BER gain is better:
  `1.074% vs 0.866%`
- On-policy BER gain is better:
  `1.110% vs 0.902%`
- `SNR_12dB policy_recovery_fraction` is better:
  `9.16% vs 6.83%`


## 8. Remaining Problems

`mix_r1_tinyrl` is improved, but not fully solved.

Observed remaining issues:

- It still does not beat the original baseline on the main original benchmark.
- Policy switching is now too conservative:
  `switch_rate_policy = 0.00366` on the original paper evaluation set,
  while oracle is `0.11416`.
- Policy action distribution is still biased:
  top action is now action `1` with frequency `51.28%`.
- At `14 dB`, the policy is still much more static than oracle:
  `policy_switch_rate = 0.00366`, `oracle_switch_rate = 0.11416`

So the current situation is:

- The action-11 collapse was partly fixed.
- The model did not become truly adaptive enough.
- Distribution bias moved from one dominant action to another dominant action.


## 9. How To Reproduce the Best Version

The commands below use the validated interpreter:

- `C:\Users\MYCZ\.conda\envs\pytorch\python.exe`

### 9.1 Build `dagger_r1`

```powershell
& "C:\Users\MYCZ\.conda\envs\pytorch\python.exe" python/rl_c1/build_dagger_rollout_dataset.py `
  --input data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results/paper_tsv2seq_vdop_ctrl/best_reinforce_policy_tsv2seq_vdop_ctrl_paper.pt `
  --output data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --reward_key reward_relbase_proxy `
  --device cuda
```

### 9.2 Build `mix_r1`

```powershell
& "C:\Users\MYCZ\.conda\envs\pytorch\python.exe" python/rl_c1/build_mixed_policy_dataset.py `
  --original data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --onpolicy data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --output data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz `
  --reward_key reward_relbase_proxy
```

### 9.3 Train `mix_r1_tinyrl`

```powershell
& "C:\Users\MYCZ\.conda\envs\pytorch\python.exe" python/rl_c1/train_reinforce.py `
  --data data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz `
  --snr_min 0 `
  --reward_key reward_relbase_proxy `
  --imitation_label auto `
  --reward_teacher_margin 0.005 `
  --split_mode sequence `
  --epochs 20 `
  --batch_size 256 `
  --reinforce_lr 1e-4 `
  --imitation_preserve_coef 0.5 `
  --output_dir results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --checkpoint_name best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --history_name train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --summary_name train_summary_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --device cuda `
  --rl_mode imitation_then_reinforce
```

### 9.4 Evaluate on original paper dataset

```powershell
& "C:\Users\MYCZ\.conda\envs\pytorch\python.exe" python/rl_c1/eval_sequence_policy.py `
  --data data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --reward_key reward_relbase_proxy `
  --device cuda `
  --output_dir results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --prefix sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig `
  --select_snr 14
```

### 9.5 Export standard figures

```powershell
& "C:\Users\MYCZ\.conda\envs\pytorch\python.exe" python/rl_c1/export_standard_figures.py `
  --data data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --reward_key reward_relbase_proxy `
  --train_history results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --oampnet_history data/training_history_v4_tsv2seq_vdop_ctrl_paper.json `
  --results_dir results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --figures_dir figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --paper_tag tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --high_snr_min 12 `
  --device cpu
```


## 10. Evaluation Provenance Note

Two figure families should be interpreted differently:

- `static_dynamic_baselines_*.png`
  This is a direct policy-side comparison from the offline BER table.
  It is the cleanest figure for understanding whether optimized C1 beats fixed C1.

- `fig3_ber_vs_snr_main.png`
  This is detector-side BER-vs-SNR output.
  When no MATLAB online result CSV is provided, the script estimates detector-side RL curves
  by transferring the policy BER gain ratio onto detector baseline curves.
  Therefore, this figure is useful, but it should not be over-claimed as full measured-online
  detector proof unless the MATLAB online CSV is supplied.


## 11. Recommended Next Step

Do not continue to `r3` using the current recipe.

The next more defensible directions are:

1. Tune the original/on-policy mixing ratio instead of using a hard 50/50 mix.
2. Add an anti-collapse regularizer or action-distribution prior.
3. Add an explicit switch-cost or switch-target shaping term only after the action bias is under control.
4. If a single checkpoint must be taken forward today, use:
   `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`
