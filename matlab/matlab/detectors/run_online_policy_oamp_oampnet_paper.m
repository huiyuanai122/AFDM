%% run_online_policy_oamp_oampnet_paper.m
% Formal BER-SNR online rollout for the paper_tsv2seq profile.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'paper';
cfg.paper_id = 'tsv2seq';
cfg.detector_target = 'oampnet';
cfg.eval_detector_mode = 'both';
cfg.use_oracle_state = false;
cfg.num_frames = 10;
cfg.context_switch_window = 5;

run_online_policy_oamp_oampnet(cfg);

