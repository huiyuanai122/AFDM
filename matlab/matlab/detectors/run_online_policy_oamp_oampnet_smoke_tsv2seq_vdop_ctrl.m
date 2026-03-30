%% run_online_policy_oamp_oampnet_smoke_tsv2seq_vdop_ctrl.m
% Small-scale online rollout for the strong dynamic-control profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'smoke';
cfg.paper_id = 'tsv2seq_vdop_ctrl';
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
cfg.detector_target = 'lmmse';
cfg.eval_detector_mode = 'target_only';
cfg.use_oracle_state = false;
cfg.snr_db_list = [12 14 16];
cfg.num_seq = 4;
cfg.num_frames = 40;
cfg.num_noise = 1;

run_online_policy_oamp_oampnet(cfg);
