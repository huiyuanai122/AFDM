%% run_generate_oampnet_paper_tsv2seq_vdop_ctrl.m
% Formal detector-dataset generation for the strong dynamic-control profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.mode = 'timevary_sequence';
cfg.run_profile = 'paper';
cfg.num_seq = 500;
cfg.num_frames = 40;
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
cfg.rho_h = 0.98;
cfg.rho_acc = 0.95;
cfg.sigma_acc = 0.03;
cfg.rho_delta = 0.90;
cfg.sigma_delta = 0.05;
cfg.ell_mode = 'static';
cfg.pdp_mode = 'exp_fixed_per_sequence';

generate_dataset_timescaling_n256('train', 20000, 2000, 'tsv2seq_vdop_ctrl_paper', cfg);
cfg_val = cfg;
cfg_val.num_seq = 100;
cfg_val.num_frames = 40;
generate_dataset_timescaling_n256('val', 4000, 2000, 'tsv2seq_vdop_ctrl_paper', cfg_val);
