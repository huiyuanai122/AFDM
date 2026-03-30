%% run_generate_oampnet_paper_tsv2seq.m
% Formal detector-dataset generation for the paper_tsv2seq profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.mode = 'timevary_sequence';
cfg.run_profile = 'paper';
cfg.num_seq = 2000;
cfg.num_frames = 10;
cfg.rho_alpha = 0.98;
cfg.rho_h = 0.98;
cfg.ell_mode = 'static';
cfg.pdp_mode = 'exp_fixed_per_sequence';

generate_dataset_timescaling_n256('train', 20000, 2000, 'tsv2seq_paper', cfg);
cfg_val = cfg;
cfg_val.num_seq = 400;
cfg_val.num_frames = 10;
generate_dataset_timescaling_n256('val', 4000, 2000, 'tsv2seq_paper', cfg_val);
