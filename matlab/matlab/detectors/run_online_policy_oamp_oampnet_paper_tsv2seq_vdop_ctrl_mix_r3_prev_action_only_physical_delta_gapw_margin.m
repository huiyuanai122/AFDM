%% run_online_policy_oamp_oampnet_paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.m
% Formal BER-SNR online rollout for the current best RL-C1 checkpoint.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin';
cfg.version = 'tsv2seq_vdop_ctrl_paper';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.motion_profile = 'maneuver_heave';
cfg.path_projection_mode = 'symmetric_linear';
cfg.beta_min = 0.45;
cfg.beta_max = 1.65;
cfg.target_track_gain = 0.85;
cfg.target_blend = 0.85;
cfg.profile_v_peak = 0.98;
cfg.profile_heave_amp = 0.20;
cfg.profile_secondary_amp = 0.10;
cfg.detector_target = 'oampnet';
cfg.eval_detector_mode = 'both';
cfg.use_oracle_state = false;
% This checkpoint uses a 25-D named feature schema:
% prev_action_only + physical_delta.
cfg.include_physical_doppler_state = false;
cfg.num_frames = 40;
cfg.context_switch_window = 8;

run_online_policy_oamp_oampnet(cfg);
