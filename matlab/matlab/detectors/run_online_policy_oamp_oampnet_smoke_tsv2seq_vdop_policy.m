%% run_online_policy_oamp_oampnet_smoke_tsv2seq_vdop_policy.m
% Small online rollout using the smoke common-Doppler policy.

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
results_dir = fullfile(project_root, 'results', 'rl_c1_tsv2seq_vdop_smoke');
addpath(this_dir);

cfg = struct();
cfg.run_profile = 'smoke';
cfg.paper_id = 'tsv2seq_vdop_smoke';
cfg.doppler_mode = 'common_with_path_residual';
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.detector_target = 'lmmse';
cfg.eval_detector_mode = 'target_only';
cfg.use_oracle_state = false;
cfg.snr_db_list = [14];
cfg.num_seq = 3;
cfg.num_frames = 4;
cfg.num_noise = 1;
cfg.policy_path = fullfile(results_dir, 'rl_c1_policy_matlab_params_tsv2seq_vdop_smoke.mat');
cfg.output_csv = fullfile(results_dir, 'smoke_vdop_policy_online_detector_frame_rollout.csv');
cfg.output_mat = fullfile(results_dir, 'smoke_vdop_policy_online_detector_eval_result.mat');
cfg.output_csv_wide = fullfile(results_dir, 'smoke_vdop_ber_results_policy_online_oamp_oampnet.csv');
cfg.output_csv_fig3 = fullfile(results_dir, 'smoke_vdop_fig3_ber_vs_snr_main_online.csv');
cfg.output_csv_fig4 = fullfile(results_dir, 'smoke_vdop_fig4_ablation_gain_online.csv');

run_online_policy_oamp_oampnet(cfg);
