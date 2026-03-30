%% run_export_oracle_policy_paper_tsv2seq_vdop.m
% Formal offline bandit dataset export for the paper_tsv2seq_vdop profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq_vdop';
cfg.mode = 'timevary_sequence';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.detector_target = 'oampnet';
cfg.use_oracle_state = false;
cfg.num_seq = 2000;
cfg.num_frames = 10;
cfg.snr_db_list = 0:2:20;
cfg.num_noise = 1;

export_oracle_dataset_for_policy(cfg);
