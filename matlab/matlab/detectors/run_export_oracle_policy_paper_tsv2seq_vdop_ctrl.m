%% run_export_oracle_policy_paper_tsv2seq_vdop_ctrl.m
% Formal offline bandit export for the strong dynamic-control paper profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq_vdop_ctrl';
cfg.mode = 'timevary_sequence';
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
cfg.reward_primary_key = 'reward_relbase_proxy';
cfg.reward_proxy_improve_lambda = 0.20;
cfg.label_eval_repeats_low = 1;
cfg.label_eval_repeats_mid = 3;
cfg.label_eval_repeats_high = 6;
cfg.detector_target = 'oampnet';
cfg.use_oracle_state = false;
% Keep the paper export aligned with the current 20-D policy state
% (prev_action_only + physical_delta, without full physical doppler state).
cfg.include_physical_doppler_state = false;
cfg.num_seq = 500;
cfg.num_frames = 40;
cfg.snr_db_list = 0:2:20;
cfg.num_noise = 1;

export_oracle_dataset_for_policy(cfg);
