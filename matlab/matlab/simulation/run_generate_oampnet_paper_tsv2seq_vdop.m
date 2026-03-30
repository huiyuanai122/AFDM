%% run_generate_oampnet_paper_tsv2seq_vdop.m
% Formal detector-dataset generation for the paper_tsv2seq_vdop profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.mode = 'timevary_sequence';
cfg.run_profile = 'paper';
cfg.num_seq = 2000;
cfg.num_frames = 10;
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.rho_h = 0.98;
cfg.rho_acc = 0.95;
cfg.sigma_acc = 0.03;
cfg.rho_delta = 0.90;
cfg.sigma_delta = 0.05;
cfg.ell_mode = 'static';
cfg.pdp_mode = 'exp_fixed_per_sequence';

generate_dataset_timescaling_n256('train', 20000, 2000, 'tsv2seq_vdop_paper', cfg);
cfg_val = cfg;
cfg_val.num_seq = 400;
cfg_val.num_frames = 10;
generate_dataset_timescaling_n256('val', 4000, 2000, 'tsv2seq_vdop_paper', cfg_val);
