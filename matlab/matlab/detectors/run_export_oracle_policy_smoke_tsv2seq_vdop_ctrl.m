%% run_export_oracle_policy_smoke_tsv2seq_vdop_ctrl.m
% Small offline bandit export for strong dynamic-control validation.

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'smoke';
cfg.paper_id = 'tsv2seq_vdop_ctrl_smoke';
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
cfg.detector_target = 'lmmse';
cfg.use_oracle_state = false;
cfg.num_seq = 48;
cfg.num_frames = 40;
cfg.num_noise = 1;
cfg.snr_db_list = [12 14 16];
cfg.output_file = fullfile(project_root, 'data', 'oracle_policy_dataset_tsv2seq_vdop_ctrl_smoke.mat');

export_oracle_dataset_for_policy(cfg);
