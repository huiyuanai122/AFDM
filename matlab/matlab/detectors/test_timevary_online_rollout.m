%% test_timevary_online_rollout.m
% Minimal reproducibility validation for sequence-level online rollout.

clear; clc;

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
results_dir = fullfile(project_root, 'results');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end

cfg = struct();
cfg.snr_db_list = [14];
cfg.num_seq = 1;
cfg.num_frames = 4;
cfg.num_noise = 1;
cfg.use_oracle_state = false;
cfg.seed_base = 20260310;
cfg.detector_target = 'lmmse';
cfg.eval_detector_mode = 'target_only';
cfg.paper_id = 'tsv2seq_vdop';
cfg.doppler_mode = 'common_with_path_residual';
cfg.output_csv = fullfile(results_dir, 'smoke_policy_online_detector_frame_rollout.csv');
cfg.output_mat = fullfile(results_dir, 'smoke_policy_online_detector_eval_result.mat');
cfg.output_csv_wide = fullfile(results_dir, 'smoke_ber_results_policy_online_oamp_oampnet.csv');
cfg.output_csv_fig3 = fullfile(results_dir, 'smoke_fig3_ber_vs_snr_main_online.csv');
cfg.output_csv_fig4 = fullfile(results_dir, 'smoke_fig4_ablation_gain_online.csv');

% Auto-pick latest available OAMPNet params version (avoid hardcoded old default).
data_dir = fullfile(project_root, 'data');
param_list = dir(fullfile(data_dir, 'oampnet_v4_*_params.mat'));
if ~isempty(param_list)
    [~, ix] = max([param_list.datenum]);
    fname = param_list(ix).name;
    token = regexp(fname, '^oampnet_v4_(.+)_params\\.mat$', 'tokens', 'once');
    if ~isempty(token)
        cfg.version = token{1};
    end
end

% Auto-pick latest vdop smoke policy if available.
policy_candidates = [ ...
    dir(fullfile(project_root, 'results', 'rl_c1_tsv2seq_vdop_smoke', 'rl_c1_policy_matlab_params*.mat')); ...
    dir(fullfile(project_root, 'results', '**', 'rl_c1_policy_matlab_params*.mat'))];
if ~isempty(policy_candidates)
    [~, ixp] = max([policy_candidates.datenum]);
    cfg.policy_path = fullfile(policy_candidates(ixp).folder, policy_candidates(ixp).name);
end

result = run_online_policy_oamp_oampnet(cfg);

n_expected = cfg.num_seq * cfg.num_frames * numel(cfg.snr_db_list);
assert(numel(result.sequence_id) == n_expected, 'Unexpected total frame count.');
assert(numel(result.time_index) == n_expected, 'time_index length mismatch.');

seq_ids = double(result.sequence_id(:));
time_idx = double(result.time_index(:));

uniq_seq = unique(seq_ids);
for i = 1:numel(uniq_seq)
    sid = uniq_seq(i);
    idx = find(seq_ids == sid);
    [~, ord] = sort(time_idx(idx));
    t = time_idx(idx(ord));
    if numel(t) > 1
        assert(all(diff(t) > 0), 'time_index is not strictly increasing within sequence.');
    end
end

assert(any(double(result.h_delta_norm(:)) > 0), 'h_t does not evolve across frames.');
assert(any(double(result.alpha_rms(:)) > 0), 'alpha_t statistics are invalid.');
assert(any(abs(diff(double(result.alpha_com(:)))) > 0), 'alpha_com does not evolve across frames.');
assert(any(double(result.delta_alpha_rms(:)) >= 0), 'delta_alpha_rms metadata missing.');

assert(numel(result.chosen_action_rl) == n_expected, 'chosen_action_rl length mismatch.');
assert(numel(result.chosen_action_oracle) == n_expected, 'chosen_action_oracle length mismatch.');
assert(numel(result.chosen_action_fixed) == n_expected, 'chosen_action_fixed length mismatch.');

assert(all(result.chosen_action_rl(:) >= 0), 'chosen_action_rl must be zero-based non-negative.');
assert(all(result.chosen_action_oracle(:) >= 0), 'chosen_action_oracle must be zero-based non-negative.');
assert(all(result.chosen_action_fixed(:) >= 0), 'chosen_action_fixed must be zero-based non-negative.');

disp('==== test_timevary_online_rollout PASSED ====');
fprintf('rows=%d, sequences=%d, frames/seq=%d\n', n_expected, numel(uniq_seq), cfg.num_frames);
fprintf('csv=%s\n', cfg.output_csv);
fprintf('mat=%s\n', cfg.output_mat);
