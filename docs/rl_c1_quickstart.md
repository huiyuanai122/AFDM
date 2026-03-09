# RL C1 Quickstart

## 1) MATLAB evidence scripts

Run in MATLAB:

```matlab
cd matlab/detectors
sanity_check_c1_sweep_refine
sanity_check_c1_sweep_timevary
export_oracle_dataset_for_policy
```

Outputs:

- `sanity_check_c1_sweep_refine_result.mat`
- `sanity_check_c1_sweep_timevary_result.mat`
- `oracle_policy_dataset.mat` (offline bandit data)

`export_oracle_dataset_for_policy.m` now exports both static-compatible fields and
time-vary fields:

- `reward` (default, currently `reward_mix`)
- `reward_ber`, `reward_proxy`, `reward_mix`
- `ber_actions`, `mse_proxy_actions`
- `sequence_id`, `time_index`

`export_oracle_dataset_for_policy.m` also supports detector-aware dataset export:
- `detector_target = 'lmmse' | 'oamp' | 'oampnet'`
- For the online OAMPNet linkage path, set `detector_target = 'oampnet'` before export so `reward_ber/ber_actions` match the final detector objective.

## 2) Train policy in Python (REINFORCE)

```bash
python python/rl_c1/train_reinforce.py ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --reward_key reward ^
  --split_mode sequence ^
  --epochs 80 ^
  --batch_size 256 ^
  --output_dir results/rl_c1
```

Recommended ablation:

```bash
:: imitation only
python python/rl_c1/train_reinforce.py ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --imitation_epochs 20 ^
  --epochs 0 ^
  --output_dir results/rl_c1_imitation

:: imitation + reinforce fine-tuning
python python/rl_c1/train_reinforce.py ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --imitation_epochs 20 ^
  --epochs 80 ^
  --reward_scale 1000 ^
  --output_dir results/rl_c1_warmstart
```

## 3) Evaluate policy

```bash
python python/rl_c1/eval_policy.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1/best_reinforce_policy.pt ^
  --snr_min 14 ^
  --reward_key reward ^
  --save_json results/rl_c1/eval_metrics.json
```

Metrics include:

- policy/base/oracle average reward
- policy/base/oracle average BER proxy (if reward is `-BER`)
- gain vs base
- gap to oracle
- action match rate
- switch rate (if `sequence_id` and `time_index` exist in dataset)

## 4) Progressive one-command runner (recommended)

```bash
python python/rl_c1/run_progressive_stage2.py ^
  --python_exe C:\Users\MYCZ\.conda\envs\pytorch\python.exe ^
  --project_root . ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --imitation_epochs 20 ^
  --reinforce_epochs 80 ^
  --reward_scale 1000 ^
  --device cpu ^
  --run_reward_ber_control ^
  --build_main_package ^
  --detector_csv results/ber_results_matlab_tsv1.csv ^
  --out_dir results/rl_c1_stage2
```

## 5) Sequence-level plotting (fixed/oracle/policy)

```bash
python python/rl_c1/eval_sequence_policy.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt ^
  --reward_key reward_ber ^
  --snr_min 14 ^
  --device cpu ^
  --output_dir results/rl_c1_stage2_full/sequence_eval ^
  --prefix rewardber_best
```

Generated files:
- `*_ber_vs_time.png`
- `*_action_traj_seq*.png`
- `*_switch_hist.png`
- `*_summary.json`

## 6) Unified main package (detector baseline + C1 policy)

```bash
python python/rl_c1/build_main_result_package.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt ^
  --reward_key reward_ber ^
  --snr_min 14 ^
  --device cpu ^
  --detector_csv results/ber_results_matlab_tsv1.csv ^
  --output_dir results/rl_c1_stage2_full/main_package ^
  --title_suffix "AFDM Detector + C1 Policy"
```

Generated files:
- `main_table.csv`
- `main_summary.json`
- `main_figure.png`

## 7) Standardized Fig.1~Fig.11 export

```bash
python python/rl_c1/export_standard_figures.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt ^
  --train_history results/rl_c1_stage2_full/warmstart_rewardber/train_history.json ^
  --reward_key reward_ber ^
  --detector_csv results/ber_results_matlab_tsv1.csv ^
  --refine_mat matlab/detectors/sanity_check_c1_sweep_refine_result.mat ^
  --results_dir results ^
  --figures_dir figures
```

Outputs:
- `results/fig1_ber_vs_c1.{csv,mat}` ... `results/fig11_switch_oracle_gap.{csv,mat}`
- `figures/fig1_ber_vs_c1.png` ... `figures/fig11_switch_oracle_gap.png`
- `results/figure_export_manifest.json`

## 8) Multi-seed reproducibility (recommended before final report)

```bash
python python/rl_c1/run_reproducibility_stage2.py ^
  --python_exe C:\Users\MYCZ\.conda\envs\pytorch\python.exe ^
  --project_root . ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --reward_key reward_ber ^
  --imitation_epochs 20 ^
  --reinforce_epochs 80 ^
  --device cpu ^
  --seeds 7,42,123 ^
  --out_dir results/rl_c1_stage2_repro
```

Outputs:
- `results/rl_c1_stage2_repro/reproducibility_seed_table.csv`
- `results/rl_c1_stage2_repro/reproducibility_summary.json`

## 9) Online measured detector linkage (upgrade Fig3/Fig4)

1) Export RL policy for MATLAB forward inference:

```bash
python python/rl_c1/export_policy_to_matlab.py ^
  --checkpoint results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt ^
  --data data/oracle_policy_dataset.mat ^
  --reward_key reward_ber ^
  --output_mat results/rl_c1_policy_matlab_params.mat
```

2) Run MATLAB online evaluation script:

```matlab
cd matlab/detectors
run_online_policy_oamp_oampnet
```

This produces:
- `results/ber_results_policy_online_oamp_oampnet.csv`
- `results/fig3_ber_vs_snr_main_online.csv`
- `results/fig4_ablation_gain_online.csv`

Notes for the updated script:
- It auto-selects `num_ch = max(num_ch_min, ceil((0.5/target_ber_floor)/(num_noise*bits_per_sample)))`.
- Exported `ber` is floor-adjusted for plotting at high SNR (`0.5/bit_count` when zero errors occur).
- Exported `ber_raw` keeps the raw empirical BER for traceability.

3) Regenerate standardized figures (Fig3/Fig4 will prefer measured online CSV):

```bash
python python/rl_c1/export_standard_figures.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt ^
  --reward_key reward_ber ^
  --online_policy_detector_csv results/ber_results_policy_online_oamp_oampnet.csv ^
  --results_dir results ^
  --figures_dir figures
```
