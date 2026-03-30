# RL C1 Oracle Imitation / BC+RL Summary

- Date: `20260324_cuda_fix1`
- Eval split: `rollout_r3`
- Eval data: `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r3.npz`
- Focus SNR: `8, 10, 12, 14`

## B0/B1/B2 Definitions

- `B0 RL baseline (R1-base)`: Frozen OAMPNet RL baseline: prev_action_only + physical_delta, rollout r3, MLP, W0 margin_baseline, R1 reward_static.
- `B1 pure oracle imitation`: Pure supervised oracle imitation using CE on oracle argmin-BER actions with light BC weighting.
- `B2 BC warm-start + R1 RL fine-tune`: B2 initializes from BC pretrain then runs short RL fine-tune with R1 reward_static and W0 margin_baseline.

## Oracle Label Construction

- `a_oracle = argmin_a BER(action)` from sample-level `ber_actions`.
- `oracle_beats_static_best = 1` when oracle BER is strictly lower than the per-SNR best static single-C1 BER.
- `snr_db` is the rounded sample SNR used for light BC emphasis on `10/12 dB` bins.
- B1 BC sample weight: `1 * (1 + lambda_pos * I[oracle_beats_static_best]) * snr_boost(10/12 dB)` with `lambda_pos=1.0`, `snr_boost=1.25`.
- B2 RL fine-tune keeps `reward=R1 reward_static` and `weighting=W0 margin_baseline`; only the initialization changes.

## Main Comparison

- `B0 RL baseline (R1-base)`: closure_8to14=0.1212, rl_match_8to14=48.28%, positive_recall_8to14=54.76%, c10=0.1162, c12=0.0302, overall_avg_ber=3.402671e-02
- `B1 pure oracle imitation`: closure_8to14=0.0059, rl_match_8to14=44.29%, positive_recall_8to14=51.46%, c10=-0.0329, c12=-0.0744, overall_avg_ber=3.456649e-02
- `B2 BC warm-start + R1 RL fine-tune`: closure_8to14=0.0102, rl_match_8to14=44.44%, positive_recall_8to14=51.50%, c10=-0.0181, c12=-0.0610, overall_avg_ber=3.453942e-02

## B1 vs B0

- closure_8to14: 0.1212 -> 0.0059
- rl_matches_oracle_8to14: 48.28% -> 44.29%
- positive_recall_8to14: 54.76% -> 51.46%
- c10/c12: 0.1162 / 0.0302 -> -0.0329 / -0.0744

## B2 vs B1

- closure_8to14: 0.0059 -> 0.0102
- rl_matches_oracle_8to14: 44.29% -> 44.44%
- positive_recall_8to14: 51.46% -> 51.50%
- c10/c12: -0.0329 / -0.0744 -> -0.0181 / -0.0610

## Artifacts

- Main table: `results\oracle_imitation_ablation_20260324_cuda_fix1\imitation_main_table.csv`
- Per-SNR table: `results\oracle_imitation_ablation_20260324_cuda_fix1\imitation_per_snr.csv`
- Action distribution: `results\oracle_imitation_ablation_20260324_cuda_fix1\imitation_action_distribution.csv`
- B0/B1/B2 comparison: `results\oracle_imitation_ablation_20260324_cuda_fix1\b0_b1_b2_comparison.csv`
- BER vs SNR: `figures\oracle_imitation_ablation_20260324_cuda_fix1\imitation_ber_vs_snr.png`
- Closure vs SNR: `figures\oracle_imitation_ablation_20260324_cuda_fix1\imitation_closure_vs_snr.png`
- RL match vs SNR: `figures\oracle_imitation_ablation_20260324_cuda_fix1\imitation_rl_match_vs_snr.png`
- Positive recall vs SNR: `figures\oracle_imitation_ablation_20260324_cuda_fix1\imitation_positive_recall_vs_snr.png`
