%% run_online_policy_oamp_oampnet_smoke_tsv2seq_vdop.m
% Small-scale smoke rollout for the common-Doppler tsv2seq_vdop profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'smoke';
cfg.paper_id = 'tsv2seq_vdop';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.detector_target = 'lmmse';
cfg.eval_detector_mode = 'target_only';
cfg.use_oracle_state = false;
cfg.snr_db_list = [14 18];
cfg.num_seq = 4;
cfg.num_frames = 5;
cfg.num_noise = 1;

run_online_policy_oamp_oampnet(cfg);
