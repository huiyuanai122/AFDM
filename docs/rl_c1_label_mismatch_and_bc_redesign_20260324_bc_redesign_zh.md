# RL C1 Label Mismatch And BC Redesign

- Date tag: `20260324_bc_redesign`
- Eval split: `rollout_r3`
- Eval data: `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r3.npz`
- Focus SNR: `8, 10, 12, 14`

## Definitions

- `B0 RL baseline`: Frozen OAMPNet RL baseline: prev_action_only + physical_delta, rollout r3, MLP, W0 margin_baseline, R1 reward_static.
- `B1 old oracle imitation`: Reference B1: pure argmin-BER oracle imitation with light positive/SNR weighting.
- `C1 Filtered BC`: Train CE only on positive samples whose gap_static >= q50 of positive-train gap distribution.
- `C2 Advantage-weighted BC`: Train soft-action targets from adv_a=max(BER(static_best)-BER(action),0), scaled by train positive-gap median and temperature T=1.0.

## Stage-A diagnosis highlights

- Train positive rate: 52.22%
- B0 rollout closure_8to14: 0.1212
- B1 rollout closure_8to14: 0.0059
- B1 best epoch by val_ber: 13
- B1 best epoch by rollout closure: 20

## Main comparison

- `B0 RL baseline`: closure_8to14=0.1212, rl_match_8to14=48.28%, positive_recall_8to14=54.76%, c10=0.1162, c12=0.0302, overall_avg_ber=3.402671e-02
- `B1 old oracle imitation`: closure_8to14=0.0059, rl_match_8to14=44.29%, positive_recall_8to14=51.46%, c10=-0.0329, c12=-0.0744, overall_avg_ber=3.456649e-02
- `C1 Filtered BC`: closure_8to14=-0.0137, rl_match_8to14=43.75%, positive_recall_8to14=50.64%, c10=-0.0542, c12=-0.0680, overall_avg_ber=3.444235e-02
- `C2 Advantage-weighted BC`: closure_8to14=-0.0159, rl_match_8to14=43.41%, positive_recall_8to14=51.65%, c10=-0.0372, c12=-0.0477, overall_avg_ber=3.460874e-02

## Artifacts

- Main table: `results\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_main_table.csv`
- Per-SNR table: `results\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_per_snr.csv`
- Action distribution: `results\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_action_distribution.csv`
- Comparison: `results\bc_redesign_analysis_20260324_bc_redesign\b0_b1_c1_c2_comparison.csv`
- BER figure: `figures\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_ber_vs_snr.png`
- Closure figure: `figures\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_closure_vs_snr.png`
- RL match figure: `figures\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_rl_match_vs_snr.png`
- Positive recall figure: `figures\bc_redesign_analysis_20260324_bc_redesign\filtered_adv_bc_positive_recall_vs_snr.png`