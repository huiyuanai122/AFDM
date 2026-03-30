%% run_export_oracle_policy_paper_tsv2seq.m
% Formal offline bandit dataset export for the paper_tsv2seq profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq';
cfg.mode = 'timevary_sequence';
cfg.detector_target = 'oampnet';
cfg.num_seq = 2000;
cfg.num_frames = 10;
cfg.snr_db_list = 0:2:20;
cfg.num_noise = 1;

export_oracle_dataset_for_policy(cfg);
