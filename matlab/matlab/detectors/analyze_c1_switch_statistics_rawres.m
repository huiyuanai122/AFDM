function out = analyze_c1_switch_statistics_rawres(user_cfg)
%ANALYZE_C1_SWITCH_STATISTICS_RAWRES
% Analyze frame-wise oracle C1 switching statistics under residual alpha.
% The script reuses results from run_ber_vs_alphares_baseline_ladder when
% available so that stage-I and stage-II share the same sequence set.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
addpath(this_dir);

cfg = default_cfg(project_root);
cfg = merge_struct(cfg, user_cfg);
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

[results, alpha_idx, source_mat] = load_or_generate_baseline_results(cfg, project_root);
policy_stats = results.seq_level_policy_stats(alpha_idx);
alpha_cfg = results.alpha_res_list(alpha_idx);
alpha_raw_cfg = results.alpha_raw_list(alpha_idx);

alpha_eff = policy_stats.frame_alpha_eff;
best_idx_oracle = policy_stats.dynamic_oracle_best_idx_frame;
best_c1_oracle = policy_stats.dynamic_oracle_best_c1_frame;
ber_oracle_frame = policy_stats.ber_dynamic_oracle_frame;
biterr_oracle_frame = policy_stats.biterr_dynamic_oracle_frame;
ber_static_oracle_best_frame = policy_stats.ber_static_oracle_best_frame;
biterr_static_oracle_best_frame = policy_stats.biterr_static_oracle_best_frame;
ber_dynamic_formula_frame = policy_stats.ber_dynamic_formula_frame;
ber_dynamic_formula_quantized_frame = policy_stats.ber_dynamic_formula_quantized_frame;

num_seq = size(best_idx_oracle, 1);
num_frames = size(best_idx_oracle, 2);
num_actions = size(policy_stats.cfg_action_grid, 2);

bits_per_frame = results.bits_per_frame;
if isempty(cfg.tau_abs)
    tau_abs = 1 / bits_per_frame;
else
    tau_abs = cfg.tau_abs;
end

switch_mask = false(num_seq, num_frames);
if num_frames > 1
    switch_mask(:, 2:end) = diff(best_idx_oracle, 1, 2) ~= 0;
end
switch_count = sum(switch_mask, 2);
switch_rate = switch_count / max(num_frames - 1, 1);

transition_matrix = build_transition_matrix(best_idx_oracle, num_actions);
transition_prob = normalize_rows(transition_matrix);

delta_ber_switch = ber_static_oracle_best_frame - ber_oracle_frame;
delta_bit_errors = biterr_static_oracle_best_frame - biterr_oracle_frame;
worth_switch_label = delta_ber_switch > tau_abs;
best_action_if_switch = zeros(size(best_idx_oracle));
best_action_if_switch(worth_switch_label) = best_idx_oracle(worth_switch_label);

[alpha_edges, alpha_bin_idx, alpha_bin_labels] = build_alpha_bins(alpha_eff, cfg.alpha_eff_num_bins);
best_idx_hist = accumarray(best_idx_oracle(:), 1, [num_actions, 1]);
best_idx_given_alpha_bin = zeros(cfg.alpha_eff_num_bins, num_actions);
delta_gain_by_alpha_bin = zeros(cfg.alpha_eff_num_bins, 1);
delta_gain_switch_only_by_alpha_bin = zeros(cfg.alpha_eff_num_bins, 1);

for kk = 1:cfg.alpha_eff_num_bins
    bin_mask = (alpha_bin_idx == kk);
    if any(bin_mask(:))
        idx_vals = best_idx_oracle(bin_mask);
        best_idx_given_alpha_bin(kk, :) = accumarray(idx_vals, 1, [num_actions, 1]).';
        delta_gain_by_alpha_bin(kk) = mean(delta_ber_switch(bin_mask));
    end
    switch_bin_mask = bin_mask & switch_mask;
    if any(switch_bin_mask(:))
        delta_gain_switch_only_by_alpha_bin(kk) = mean(delta_ber_switch(switch_bin_mask));
    end
end

best_idx_given_alpha_prob = normalize_rows(best_idx_given_alpha_bin);

rep_seq_idx = select_representative_sequences(switch_count, alpha_eff, cfg.num_rep_sequences);
local_transition_fraction = compute_local_transition_fraction(best_idx_oracle);

plot_switch_count_histogram(switch_count, alpha_cfg, cfg);
plot_representative_action_tracks(best_idx_oracle, rep_seq_idx, alpha_cfg, cfg);
plot_action_vs_alpha_heatmap(best_idx_given_alpha_prob, alpha_bin_labels, alpha_cfg, cfg);
plot_transition_matrix_heatmap(transition_prob, alpha_cfg, cfg);
plot_switch_gain_figure(delta_ber_switch, switch_mask, delta_gain_by_alpha_bin, ...
    delta_gain_switch_only_by_alpha_bin, alpha_bin_labels, alpha_cfg, cfg);

stats = struct();
stats.alpha_eff = alpha_eff;
stats.best_idx_oracle = best_idx_oracle;
stats.best_c1_oracle = best_c1_oracle;
stats.switch_count = switch_count;
stats.switch_rate = switch_rate;
stats.transition_matrix = transition_matrix;
stats.transition_prob = transition_prob;
stats.delta_ber_switch = delta_ber_switch;
stats.delta_bit_errors = delta_bit_errors;
stats.ber_oracle_frame = ber_oracle_frame;
stats.ber_static_oracle_best_frame = ber_static_oracle_best_frame;
stats.ber_dynamic_formula_frame = ber_dynamic_formula_frame;
stats.ber_dynamic_formula_quantized_frame = ber_dynamic_formula_quantized_frame;
stats.worth_switch_label = worth_switch_label;
stats.best_action_if_switch = best_action_if_switch;
stats.best_idx_hist = best_idx_hist;
stats.best_idx_given_alpha_bin = best_idx_given_alpha_bin;
stats.best_idx_given_alpha_prob = best_idx_given_alpha_prob;
stats.alpha_eff_bin_edges = alpha_edges;
stats.alpha_eff_bin_labels = alpha_bin_labels;
stats.delta_gain_by_alpha_bin = delta_gain_by_alpha_bin;
stats.delta_gain_switch_only_by_alpha_bin = delta_gain_switch_only_by_alpha_bin;
stats.rep_seq_idx = rep_seq_idx;
stats.local_transition_fraction = local_transition_fraction;
stats.meta_cfg = cfg;
stats.meta_cfg.target_alpha_res = alpha_cfg;
stats.meta_cfg.target_alpha_raw = alpha_raw_cfg;
stats.meta_cfg.source_results_mat = source_mat;
stats.meta_cfg.tau_abs = tau_abs;
stats.meta_cfg.bits_per_frame = bits_per_frame;

save(cfg.output_mat, 'stats', '-v7.3');

fprintf('==== C1 switch statistics summary ====\n');
fprintf('alpha_max_res=%.2e, alpha_max_raw=%.2e, source=%s\n', alpha_cfg, alpha_raw_cfg, source_mat);
fprintf('switch_count: mean=%.3f, median=%.3f, p90=%.3f, p95=%.3f\n', ...
    mean(switch_count), median(switch_count), quantile(switch_count, 0.90), quantile(switch_count, 0.95));
fprintf('worth_switch positive rate: %.4f\n', mean(worth_switch_label(:)));
fprintf('mean DeltaBER(all frames)=%.3e, mean DeltaBER(on switch frames)=%.3e\n', ...
    mean(delta_ber_switch(:)), masked_mean(delta_ber_switch, switch_mask));
fprintf('mean delta_bit_errors(all frames)=%.3f, local_transition_fraction=%.4f\n', ...
    mean(delta_bit_errors(:)), local_transition_fraction);
fprintf('Saved stats: %s\n', cfg.output_mat);

out = stats;
end

function cfg = default_cfg(project_root)
cfg = struct();
cfg.output_dir = fullfile(project_root, 'outputs');
cfg.output_mat = fullfile(cfg.output_dir, 'results_c1_switch_statistics_rawres.mat');
cfg.baseline_results_mat = fullfile(cfg.output_dir, 'results_ber_vs_alphares_baseline_ladder.mat');

cfg.reuse_baseline_results = true;
cfg.force_rerun_baseline = false;
cfg.regenerate_if_needed = true;

cfg.target_alpha_res = 1e-4;
cfg.alpha_max_res = 1e-4;
cfg.alpha_max_raw = 5e-4;
cfg.alpha_raw_scale = 5;

cfg.snr_db = 10;
cfg.num_seq = 200;
cfg.num_frames = 9;
cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.doppler_mode = 'common_with_path_residual';
cfg.motion_profile = 'smooth_ar';
cfg.detector_type = 'oamp';

cfg.rho_h = 0.98;
cfg.rho_acc = 0.95;
cfg.sigma_acc = 0.03;
cfg.rho_delta = 0.90;
cfg.sigma_delta = 0.05;
cfg.path_projection_mode = 'ones';
cfg.beta_min = 1.0;
cfg.beta_max = 1.0;
cfg.ell_mode = 'static';
cfg.pdp_mode = 'exp_fixed_per_sequence';
cfg.oamp_iter = 10;
cfg.oamp_damping = 0.9;
cfg.num_noise = 1;
cfg.seed_base = 20260328;
cfg.c1_ratios = linspace(0.6, 1.4, 21);
cfg.static_oracle_ratios = linspace(0.5, 1.5, 41);
cfg.N = 256;
cfg.Delta_f = 4;
cfg.fc = 12e3;
cfg.ell_max = 16;
cfg.P = 6;
cfg.Q = 0;
cfg.Nv = 2;

cfg.alpha_eff_num_bins = 8;
cfg.num_rep_sequences = 5;
cfg.tau_abs = [];
end

function [results, alpha_idx, source_mat] = load_or_generate_baseline_results(cfg, project_root)
need_regen = cfg.force_rerun_baseline || ~exist(cfg.baseline_results_mat, 'file');
results = [];

if ~need_regen && cfg.reuse_baseline_results
    tmp = load(cfg.baseline_results_mat, 'results');
    results = tmp.results;
    if ~has_compatible_results(results, cfg)
        need_regen = cfg.regenerate_if_needed;
    end
end

if need_regen
    if ~cfg.regenerate_if_needed
        error('Missing or incompatible baseline result: %s', cfg.baseline_results_mat);
    end
    baseline_cfg = baseline_cfg_from_switch_cfg(cfg);
    results = run_ber_vs_alphares_baseline_ladder(baseline_cfg);
    source_mat = baseline_cfg.output_mat;
else
    if isempty(results)
        tmp = load(cfg.baseline_results_mat, 'results');
        results = tmp.results;
    end
    source_mat = cfg.baseline_results_mat;
end

alpha_idx = select_alpha_index(results.alpha_res_list, cfg.target_alpha_res);
end

function tf = has_compatible_results(results, cfg)
tf = isfield(results, 'seq_level_policy_stats') && isfield(results, 'alpha_res_list') && ...
    isfield(results, 'num_seq') && isfield(results, 'num_frames') && ...
    isfield(results, 'detector_type');
if ~tf
    return;
end
tf = tf && (results.num_seq == cfg.num_seq) && (results.num_frames == cfg.num_frames) && ...
    strcmpi(results.detector_type, cfg.detector_type) && ...
    any(abs(results.alpha_res_list(:) - cfg.target_alpha_res) <= 1e-12);
end

function baseline_cfg = baseline_cfg_from_switch_cfg(cfg)
baseline_cfg = struct();
baseline_cfg.output_dir = cfg.output_dir;
baseline_cfg.output_mat = cfg.baseline_results_mat;
baseline_cfg.alpha_res_list = cfg.target_alpha_res;
baseline_cfg.alpha_raw_scale = cfg.alpha_raw_scale;
baseline_cfg.snr_db = cfg.snr_db;
baseline_cfg.num_seq = cfg.num_seq;
baseline_cfg.num_frames = cfg.num_frames;
baseline_cfg.enable_resampling_comp = cfg.enable_resampling_comp;
baseline_cfg.alpha_hat_mode = cfg.alpha_hat_mode;
baseline_cfg.doppler_mode = cfg.doppler_mode;
baseline_cfg.motion_profile = cfg.motion_profile;
baseline_cfg.rho_h = cfg.rho_h;
baseline_cfg.rho_acc = cfg.rho_acc;
baseline_cfg.sigma_acc = cfg.sigma_acc;
baseline_cfg.rho_delta = cfg.rho_delta;
baseline_cfg.sigma_delta = cfg.sigma_delta;
baseline_cfg.path_projection_mode = cfg.path_projection_mode;
baseline_cfg.beta_min = cfg.beta_min;
baseline_cfg.beta_max = cfg.beta_max;
baseline_cfg.ell_mode = cfg.ell_mode;
baseline_cfg.pdp_mode = cfg.pdp_mode;
baseline_cfg.detector_type = cfg.detector_type;
baseline_cfg.oamp_iter = cfg.oamp_iter;
baseline_cfg.oamp_damping = cfg.oamp_damping;
baseline_cfg.num_noise = cfg.num_noise;
baseline_cfg.seed_base = cfg.seed_base;
baseline_cfg.c1_ratios = cfg.c1_ratios;
baseline_cfg.static_oracle_ratios = cfg.static_oracle_ratios;
baseline_cfg.progress_every_seq = max(1, min(25, cfg.num_seq));
baseline_cfg.N = cfg.N;
baseline_cfg.Delta_f = cfg.Delta_f;
baseline_cfg.fc = cfg.fc;
baseline_cfg.ell_max = cfg.ell_max;
baseline_cfg.P = cfg.P;
baseline_cfg.Q = cfg.Q;
baseline_cfg.Nv = cfg.Nv;
end

function alpha_idx = select_alpha_index(alpha_list, target_alpha)
[~, alpha_idx] = min(abs(alpha_list(:) - target_alpha));
end

function transition_matrix = build_transition_matrix(best_idx_oracle, num_actions)
transition_matrix = zeros(num_actions, num_actions);
if size(best_idx_oracle, 2) <= 1
    return;
end

for sid = 1:size(best_idx_oracle, 1)
    idx_from = best_idx_oracle(sid, 1:end - 1);
    idx_to = best_idx_oracle(sid, 2:end);
    for tt = 1:numel(idx_from)
        transition_matrix(idx_from(tt), idx_to(tt)) = transition_matrix(idx_from(tt), idx_to(tt)) + 1;
    end
end
end

function row_norm = normalize_rows(mat_in)
row_sum = sum(mat_in, 2);
row_norm = mat_in;
for rr = 1:size(mat_in, 1)
    if row_sum(rr) > 0
        row_norm(rr, :) = mat_in(rr, :) / row_sum(rr);
    end
end
end

function [edges, bin_idx, labels] = build_alpha_bins(alpha_eff, num_bins)
alpha_min = min(alpha_eff(:));
alpha_max = max(alpha_eff(:));
if alpha_max <= alpha_min
    alpha_min = alpha_min - max(abs(alpha_min), 1e-6);
    alpha_max = alpha_max + max(abs(alpha_max), 1e-6);
end
edges = linspace(alpha_min, alpha_max, num_bins + 1);
bin_idx = discretize(alpha_eff, edges);
bin_idx(isnan(bin_idx)) = num_bins;

labels = strings(num_bins, 1);
for kk = 1:num_bins
    labels(kk) = sprintf('[%.2e, %.2e)', edges(kk), edges(kk + 1));
end
labels(end) = sprintf('[%.2e, %.2e]', edges(end - 1), edges(end));
end

function rep_seq_idx = select_representative_sequences(switch_count, alpha_eff, num_rep_sequences)
seq_alpha_std = std(alpha_eff, 0, 2);
score = switch_count + seq_alpha_std / max(mean(seq_alpha_std), 1e-12);
[~, order] = sort(score, 'descend');
num_pick = min(num_rep_sequences, numel(order));
rep_seq_idx = sort(order(1:num_pick));
end

function frac = compute_local_transition_fraction(best_idx_oracle)
if size(best_idx_oracle, 2) <= 1
    frac = 1.0;
    return;
end
delta_idx = abs(diff(best_idx_oracle, 1, 2));
switch_delta = delta_idx(delta_idx > 0);
if isempty(switch_delta)
    frac = 1.0;
else
    frac = mean(switch_delta <= 1);
end
end

function plot_switch_count_histogram(switch_count, alpha_cfg, cfg)
fig = figure('Color', 'w');
histogram(switch_count, 'BinMethod', 'integers');
grid on;
xlabel('switch_count per sequence');
ylabel('count');
title(sprintf('Switch-count histogram (\\alpha_{max,res}=%.2e)', alpha_cfg));
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_c1_switch_count_hist_rawres'));
end

function plot_representative_action_tracks(best_idx_oracle, rep_seq_idx, alpha_cfg, cfg)
fig = figure('Color', 'w');
hold on;
frame_axis = 1:size(best_idx_oracle, 2);
colors = lines(numel(rep_seq_idx));
for kk = 1:numel(rep_seq_idx)
    sid = rep_seq_idx(kk);
    stairs(frame_axis, best_idx_oracle(sid, :), '-o', ...
        'Color', colors(kk, :), ...
        'LineWidth', 1.4, ...
        'DisplayName', sprintf('seq %d', sid));
end
grid on;
xlabel('frame index');
ylabel('oracle best action index');
title(sprintf('Representative oracle action tracks (\\alpha_{max,res}=%.2e)', alpha_cfg));
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_c1_best_idx_tracks_rawres'));
end

function plot_action_vs_alpha_heatmap(best_idx_given_alpha_prob, alpha_bin_labels, alpha_cfg, cfg)
fig = figure('Color', 'w');
imagesc(best_idx_given_alpha_prob);
axis xy;
colorbar;
xlabel('oracle best action index');
ylabel('\alpha_{eff} bin');
yticks(1:numel(alpha_bin_labels));
yticklabels(cellstr(alpha_bin_labels));
title(sprintf('p(best\\_idx | \\alpha_{eff} bin), \\alpha_{max,res}=%.2e', alpha_cfg));
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_c1_best_idx_vs_alphaeff_rawres'));
end

function plot_transition_matrix_heatmap(transition_prob, alpha_cfg, cfg)
fig = figure('Color', 'w');
imagesc(transition_prob);
axis xy;
colorbar;
xlabel('to action index');
ylabel('from action index');
title(sprintf('Oracle action transition matrix, \\alpha_{max,res}=%.2e', alpha_cfg));
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_c1_transition_matrix_rawres'));
end

function plot_switch_gain_figure(delta_ber_switch, switch_mask, delta_gain_by_alpha_bin, ...
    delta_gain_switch_only_by_alpha_bin, alpha_bin_labels, alpha_cfg, cfg)
fig = figure('Color', 'w');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
histogram(delta_ber_switch(:), 30);
grid on;
xlabel('\Delta BER_{switch}');
ylabel('count');
title('all frames');

nexttile;
histogram(delta_ber_switch(switch_mask), 30);
grid on;
xlabel('\Delta BER_{switch}');
ylabel('count');
title('switch frames only');

nexttile;
bb = bar([delta_gain_by_alpha_bin, delta_gain_switch_only_by_alpha_bin]);
bb(1).DisplayName = 'all frames';
bb(2).DisplayName = 'switch frames';
grid on;
xlabel('\alpha_{eff} bin');
ylabel('mean \Delta BER_{switch}');
xticks(1:numel(alpha_bin_labels));
xticklabels(compose('%d', 1:numel(alpha_bin_labels)));
legend('Location', 'best');
title('gain by \alpha_{eff} bin');

sgtitle(sprintf('Switch gain statistics (\\alpha_{max,res}=%.2e)', alpha_cfg));
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_c1_switch_gain_rawres'));
end

function save_figure_pair(fig, prefix)
saveas(fig, [prefix '.png']);
savefig(fig, [prefix '.fig']);
end

function mu = masked_mean(x, mask)
vals = x(mask);
if isempty(vals)
    mu = 0.0;
else
    mu = mean(vals);
end
end

function dst = merge_struct(dst, src)
if isempty(src)
    return;
end
fn = fieldnames(src);
for i = 1:numel(fn)
    dst.(fn{i}) = src.(fn{i});
end
end
