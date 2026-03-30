%% run_export_oracle_policy_smoke_tsv2seq_vdop.m
% Small offline bandit dataset export for common-Doppler smoke validation.

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'smoke';
cfg.paper_id = 'tsv2seq_vdop_smoke';
cfg.mode = 'timevary_sequence';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.detector_target = 'lmmse';
cfg.use_oracle_state = false;
cfg.num_seq = 48;
cfg.num_frames = 5;
cfg.num_noise = 1;
cfg.snr_db_list = [14 18];
cfg.output_file = fullfile(project_root, 'data', 'oracle_policy_dataset_tsv2seq_vdop_smoke.mat');

export_oracle_dataset_for_policy(cfg);
