%% run_online_policy_oamp_oampnet_paper_tsv2seq_vdop.m
% Formal BER-SNR online rollout for the paper_tsv2seq_vdop profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq_vdop';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.detector_target = 'oampnet';
cfg.eval_detector_mode = 'both';
cfg.use_oracle_state = false;
cfg.num_frames = 10;
cfg.context_switch_window = 5;

run_online_policy_oamp_oampnet(cfg);
