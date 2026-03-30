function out = plot_alphaeff_vs_frame_ber_rawres(user_cfg)
%PLOT_ALPHAEFF_VS_FRAME_BER_RAWRES
% Frame-level mechanism plots at a fixed alpha_max_res working point.
% The script prioritizes reusing stage-I baseline results and only falls
% back to regeneration when the expected result file is missing or
% incompatible with the requested configuration.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
addpath(this_dir);
addpath(fullfile(project_root, 'matlab', 'common'));

cfg = default_cfg(project_root);
cfg = merge_struct(cfg, user_cfg);
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

[baseline_results, alpha_idx, source_mat] = load_or_generate_baseline_results(cfg);
policy_stats = baseline_results.seq_level_policy_stats(alpha_idx);

alpha_eff_frame = policy_stats.frame_alpha_eff;
ber_frame_dynamic_oracle = policy_stats.ber_dynamic_oracle_frame;
ber_frame_static_oracle_best = policy_stats.ber_static_oracle_best_frame;
ber_frame_dynamic_formula_quantized = policy_stats.ber_dynamic_formula_quantized_frame;
best_idx_oracle = policy_stats.dynamic_oracle_best_idx_frame;
best_c1_oracle = policy_stats.dynamic_oracle_best_c1_frame;

if isfield(policy_stats, 'ber_dynamic_formula_frame')
    ber_frame_dynamic_formula = policy_stats.ber_dynamic_formula_frame;
else
    ber_frame_dynamic_formula = [];
end

num_seq = size(alpha_eff_frame, 1);
num_frames = size(alpha_eff_frame, 2);
sequence_id = repmat((1:num_seq).', 1, num_frames);
frame_index = repmat(1:num_frames, num_seq, 1);

switch_count = sum(diff(best_idx_oracle, 1, 2) ~= 0, 2);
rep_seq_id = select_representative_sequence(alpha_eff_frame, best_idx_oracle, switch_count);

all_alpha_eff = alpha_eff_frame(:);
all_dyn = ber_frame_dynamic_oracle(:);
all_static = ber_frame_static_oracle_best(:);
all_formula_q = ber_frame_dynamic_formula_quantized(:);

stats_dynamic = compute_corr_stats(all_alpha_eff, all_dyn);
stats_static = compute_corr_stats(all_alpha_eff, all_static);
stats_formula_q = compute_corr_stats(all_alpha_eff, all_formula_q);

if ~isempty(ber_frame_dynamic_formula)
    stats_formula = compute_corr_stats(all_alpha_eff, ber_frame_dynamic_formula(:));
else
    stats_formula = struct('pearson', NaN, 'spearman', NaN);
end

[bin_edges, bin_centers, bin_stats] = compute_binned_stats( ...
    all_alpha_eff, all_dyn, all_static, all_formula_q, cfg.num_bins, cfg.bin_mode);

high_low_stats = compute_high_low_group_stats( ...
    all_alpha_eff, all_dyn, all_static, all_formula_q, cfg.low_high_fraction);

plot_dualaxis_rep_sequence(rep_seq_id, alpha_eff_frame, ber_frame_dynamic_oracle, ...
    ber_frame_static_oracle_best, ber_frame_dynamic_formula_quantized, cfg);
plot_bestidx_rep_sequence(rep_seq_id, alpha_eff_frame, best_idx_oracle, cfg);
plot_scatter_triptych(all_alpha_eff, all_dyn, all_static, all_formula_q, ...
    stats_dynamic, stats_static, stats_formula_q, cfg);
plot_binned_mean_curves(bin_centers, bin_stats, cfg);

results = struct();
results.alpha_max_res = baseline_results.alpha_res_list(alpha_idx);
results.alpha_max_raw = baseline_results.alpha_raw_list(alpha_idx);
results.snr_db = baseline_results.snr_db;
results.num_seq = baseline_results.num_seq;
results.num_frames = baseline_results.num_frames;
results.detector_type = baseline_results.detector_type;
results.sequence_id = sequence_id;
results.frame_index = frame_index;
results.alpha_eff_frame = alpha_eff_frame;
results.ber_frame_dynamic_oracle = ber_frame_dynamic_oracle;
results.ber_frame_static_oracle_best = ber_frame_static_oracle_best;
results.ber_frame_dynamic_formula_quantized = ber_frame_dynamic_formula_quantized;
results.best_idx_oracle = best_idx_oracle;
results.best_c1_oracle = best_c1_oracle;
results.rep_seq_id = rep_seq_id;
results.bin_edges = bin_edges;
results.bin_centers = bin_centers;
results.bin_stats = bin_stats;
results.meta_cfg = cfg;
results.meta_cfg.source_results_mat = source_mat;
results.meta_cfg.switch_count = switch_count;
results.corr_dynamic_oracle = stats_dynamic;
results.corr_static_oracle_best = stats_static;
results.corr_dynamic_formula_quantized = stats_formula_q;
results.high_low_stats = high_low_stats;
if ~isempty(ber_frame_dynamic_formula)
    results.ber_frame_dynamic_formula = ber_frame_dynamic_formula;
    results.corr_dynamic_formula = stats_formula;
end

save(cfg.output_mat, 'results', '-v7.3');

fprintf('==== alpha_eff vs frame BER summary ====\n');
fprintf('source=%s\n', source_mat);
fprintf('alpha_max_res=%.2e, alpha_max_raw=%.2e, rep_seq_id=%d\n', ...
    results.alpha_max_res, results.alpha_max_raw, rep_seq_id);
fprintf('Correlation summary\n');
print_corr_line('dynamic_oracle', stats_dynamic);
print_corr_line('static_oracle_best', stats_static);
print_corr_line('dynamic_formula_quantized', stats_formula_q);
if ~isempty(ber_frame_dynamic_formula)
    print_corr_line('dynamic_formula', stats_formula);
end

fprintf('\nHigh/low alpha_eff group comparison (low=bottom %.0f%%, high=top %.0f%%)\n', ...
    cfg.low_high_fraction * 100, cfg.low_high_fraction * 100);
print_group_line('dynamic_oracle', high_low_stats.dynamic_oracle);
print_group_line('static_oracle_best', high_low_stats.static_oracle_best);
print_group_line('dynamic_formula_quantized', high_low_stats.dynamic_formula_quantized);
if isfield(high_low_stats, 'dynamic_formula')
    print_group_line('dynamic_formula', high_low_stats.dynamic_formula);
end

fprintf('\nSaved figure: %s\n', fullfile(cfg.output_dir, 'fig_alphaeff_vs_frame_ber_dualaxis_seq_rep.png'));
fprintf('Saved figure: %s\n', fullfile(cfg.output_dir, 'fig_alphaeff_vs_frame_bestidx_seq_rep.png'));
fprintf('Saved figure: %s\n', fullfile(cfg.output_dir, 'fig_alphaeff_vs_frameber_scatter_triptych.png'));
fprintf('Saved figure: %s\n', fullfile(cfg.output_dir, 'fig_alphaeff_binned_mean_frameber.png'));
fprintf('Saved result: %s\n', cfg.output_mat);

out = results;
end

function cfg = default_cfg(project_root)
cfg = struct();
cfg.output_dir = fullfile(project_root, 'outputs');
cfg.output_mat = fullfile(cfg.output_dir, 'results_alphaeff_vs_frame_ber_rawres.mat');
cfg.baseline_results_mat = fullfile(cfg.output_dir, 'results_ber_vs_alphares_baseline_ladder.mat');

cfg.reuse_baseline_results = true;
cfg.regenerate_if_needed = true;
cfg.force_rerun_baseline = false;

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
cfg.progress_every_seq = 25;

cfg.num_bins = 6;
cfg.bin_mode = 'quantile';
cfg.low_high_fraction = 0.30;

cfg.N = 256;
cfg.Delta_f = 4;
cfg.fc = 12e3;
cfg.ell_max = 16;
cfg.P = 6;
cfg.Q = 0;
cfg.Nv = 2;
end

function [results, alpha_idx, source_mat] = load_or_generate_baseline_results(cfg)
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
    baseline_cfg = baseline_cfg_from_plot_cfg(cfg);
    results = run_ber_vs_alphares_baseline_ladder(baseline_cfg);
    source_mat = baseline_cfg.output_mat;
else
    if isempty(results)
        tmp = load(cfg.baseline_results_mat, 'results');
        results = tmp.results;
    end
    source_mat = cfg.baseline_results_mat;
end

alpha_idx = select_alpha_index(results.alpha_res_list, cfg.alpha_max_res);
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
    any(abs(results.alpha_res_list(:) - cfg.alpha_max_res) <= 1e-12);
end

function baseline_cfg = baseline_cfg_from_plot_cfg(cfg)
baseline_cfg = struct();
baseline_cfg.output_dir = cfg.output_dir;
baseline_cfg.output_mat = cfg.baseline_results_mat;
baseline_cfg.alpha_res_list = cfg.alpha_max_res;
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
baseline_cfg.progress_every_seq = cfg.progress_every_seq;
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

function rep_seq_id = select_representative_sequence(alpha_eff_frame, best_idx_oracle, switch_count)
seq_alpha_std = std(alpha_eff_frame, 0, 2);
seq_alpha_rng = max(alpha_eff_frame, [], 2) - min(alpha_eff_frame, [], 2);
num_unique_actions = zeros(size(best_idx_oracle, 1), 1);
for sid = 1:size(best_idx_oracle, 1)
    num_unique_actions(sid) = numel(unique(best_idx_oracle(sid, :)));
end

valid = num_unique_actions > 1;
if ~any(valid)
    valid = true(size(valid));
end

switch_target = median(switch_count(valid));
std_target = median(seq_alpha_std(valid));
rng_target = median(seq_alpha_rng(valid));

score = abs(switch_count - switch_target) ...
    + abs(seq_alpha_std - std_target) / max(std_target, 1e-12) ...
    + abs(seq_alpha_rng - rng_target) / max(rng_target, 1e-12);
score(~valid) = inf;

[~, rep_seq_id] = min(score);
end

function stats = compute_corr_stats(x, y)
x = x(:);
y = y(:);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);

stats = struct('pearson', NaN, 'spearman', NaN);
if numel(x) < 2 || std(x) < 1e-15 || std(y) < 1e-15
    return;
end

c = corrcoef(x, y);
stats.pearson = c(1, 2);

rx = average_tied_ranks(x);
ry = average_tied_ranks(y);
cs = corrcoef(rx, ry);
stats.spearman = cs(1, 2);
end

function [edges, centers, bin_stats] = compute_binned_stats(alpha_eff, ber_dyn, ber_static, ber_formula_q, num_bins, bin_mode)
alpha_eff = alpha_eff(:);
if strcmpi(bin_mode, 'quantile')
    edges = empirical_quantiles(alpha_eff, linspace(0, 1, num_bins + 1));
else
    edges = linspace(min(alpha_eff), max(alpha_eff), num_bins + 1);
end

edges = enforce_monotone_edges(edges);
centers = 0.5 * (edges(1:end - 1) + edges(2:end));
bin_idx = assign_to_bins(alpha_eff, edges);
num_bins_actual = numel(centers);

bin_stats = struct();
bin_stats.count = zeros(num_bins_actual, 1);
bin_stats.dynamic_oracle_mean = zeros(num_bins_actual, 1);
bin_stats.dynamic_oracle_std = zeros(num_bins_actual, 1);
bin_stats.dynamic_oracle_sem = zeros(num_bins_actual, 1);
bin_stats.static_oracle_best_mean = zeros(num_bins_actual, 1);
bin_stats.static_oracle_best_std = zeros(num_bins_actual, 1);
bin_stats.static_oracle_best_sem = zeros(num_bins_actual, 1);
bin_stats.dynamic_formula_quantized_mean = zeros(num_bins_actual, 1);
bin_stats.dynamic_formula_quantized_std = zeros(num_bins_actual, 1);
bin_stats.dynamic_formula_quantized_sem = zeros(num_bins_actual, 1);

for kk = 1:num_bins_actual
    mask = (bin_idx == kk);
    bin_stats.count(kk) = sum(mask);
    [m, s, se] = mean_std_sem(ber_dyn(mask));
    bin_stats.dynamic_oracle_mean(kk) = m;
    bin_stats.dynamic_oracle_std(kk) = s;
    bin_stats.dynamic_oracle_sem(kk) = se;

    [m, s, se] = mean_std_sem(ber_static(mask));
    bin_stats.static_oracle_best_mean(kk) = m;
    bin_stats.static_oracle_best_std(kk) = s;
    bin_stats.static_oracle_best_sem(kk) = se;

    [m, s, se] = mean_std_sem(ber_formula_q(mask));
    bin_stats.dynamic_formula_quantized_mean(kk) = m;
    bin_stats.dynamic_formula_quantized_std(kk) = s;
    bin_stats.dynamic_formula_quantized_sem(kk) = se;
end
end

function group_stats = compute_high_low_group_stats(alpha_eff, ber_dyn, ber_static, ber_formula_q, frac)
q_low = empirical_quantiles(alpha_eff, frac);
q_high = empirical_quantiles(alpha_eff, 1 - frac);

low_mask = alpha_eff <= q_low;
high_mask = alpha_eff >= q_high;

group_stats = struct();
group_stats.dynamic_oracle = summarize_group_delta(ber_dyn, low_mask, high_mask);
group_stats.static_oracle_best = summarize_group_delta(ber_static, low_mask, high_mask);
group_stats.dynamic_formula_quantized = summarize_group_delta(ber_formula_q, low_mask, high_mask);
end

function plot_dualaxis_rep_sequence(rep_seq_id, alpha_eff_frame, ber_dyn, ber_static, ber_formula_q, cfg)
frame_axis = 1:size(alpha_eff_frame, 2);
fig = figure('Color', 'w');

yyaxis left;
plot(frame_axis, alpha_eff_frame(rep_seq_id, :), '-o', 'LineWidth', 1.8, ...
    'Color', [0.15 0.15 0.15], 'DisplayName', 'alpha_eff');
ylabel('\alpha_{eff}(t)');

yyaxis right;
hold on;
plot(frame_axis, ber_dyn(rep_seq_id, :), '-s', 'LineWidth', 1.6, 'DisplayName', 'dynamic_oracle');
plot(frame_axis, ber_static(rep_seq_id, :), '-d', 'LineWidth', 1.6, 'DisplayName', 'static_oracle_best');
plot(frame_axis, ber_formula_q(rep_seq_id, :), '-^', 'LineWidth', 1.6, 'DisplayName', 'dynamic_formula_quantized');
ylabel('BER_{frame}(t)');

grid on;
xlabel('Frame Index');
title(sprintf('Representative sequence %d: \\alpha_{eff}(t) vs frame BER', rep_seq_id));
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_alphaeff_vs_frame_ber_dualaxis_seq_rep'));
end

function plot_bestidx_rep_sequence(rep_seq_id, alpha_eff_frame, best_idx_oracle, cfg)
frame_axis = 1:size(best_idx_oracle, 2);
fig = figure('Color', 'w');
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(frame_axis, alpha_eff_frame(rep_seq_id, :), '-o', 'LineWidth', 1.6);
grid on;
xlabel('Frame Index');
ylabel('\alpha_{eff}(t)');
title(sprintf('Representative sequence %d: \\alpha_{eff}(t)', rep_seq_id));

nexttile;
stairs(frame_axis, best_idx_oracle(rep_seq_id, :), '-o', 'LineWidth', 1.6);
grid on;
xlabel('Frame Index');
ylabel('best\_idx\_oracle');
title('Oracle best action index');

save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_alphaeff_vs_frame_bestidx_seq_rep'));
end

function plot_scatter_triptych(alpha_eff, ber_dyn, ber_static, ber_formula_q, stats_dyn, stats_static, stats_formula_q, cfg)
fig = figure('Color', 'w');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

scatter_panel(alpha_eff, ber_dyn, stats_dyn, 'dynamic_oracle');
scatter_panel(alpha_eff, ber_static, stats_static, 'static_oracle_best');
scatter_panel(alpha_eff, ber_formula_q, stats_formula_q, 'dynamic_formula_quantized');

save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_alphaeff_vs_frameber_scatter_triptych'));
end

function plot_binned_mean_curves(bin_centers, bin_stats, cfg)
fig = figure('Color', 'w');
hold on;
x_axis = 1:numel(bin_centers);
errorbar(x_axis, bin_stats.dynamic_oracle_mean, bin_stats.dynamic_oracle_sem, ...
    '-o', 'LineWidth', 1.6, 'DisplayName', 'dynamic_oracle');
errorbar(x_axis, bin_stats.static_oracle_best_mean, bin_stats.static_oracle_best_sem, ...
    '-d', 'LineWidth', 1.6, 'DisplayName', 'static_oracle_best');
errorbar(x_axis, bin_stats.dynamic_formula_quantized_mean, bin_stats.dynamic_formula_quantized_sem, ...
    '-^', 'LineWidth', 1.6, 'DisplayName', 'dynamic_formula_quantized');
grid on;
xlabel('\alpha_{eff} quantile bin');
xticks(x_axis);
ylabel('Mean frame BER');
title('Binned mean frame BER vs \alpha_{eff}');
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_alphaeff_binned_mean_frameber'));
end

function scatter_panel(alpha_eff, ber_frame, stats_now, panel_title)
nexttile;
scatter(alpha_eff, ber_frame, 18, 'filled', ...
    'MarkerFaceAlpha', 0.18, ...
    'MarkerEdgeAlpha', 0.18);
grid on;
xlabel('\alpha_{eff}(t)');
ylabel('BER_{frame}(t)');
title({panel_title; ...
    sprintf('Pearson=%.3f, Spearman=%.3f', stats_now.pearson, stats_now.spearman)});
end

function summary = summarize_group_delta(ber_vec, low_mask, high_mask)
summary = struct();
summary.low_mean = mean(ber_vec(low_mask));
summary.high_mean = mean(ber_vec(high_mask));
summary.high_minus_low = summary.high_mean - summary.low_mean;
summary.low_count = sum(low_mask);
summary.high_count = sum(high_mask);
end

function [m, s, se] = mean_std_sem(x)
if isempty(x)
    m = NaN;
    s = NaN;
    se = NaN;
    return;
end
m = mean(x);
if numel(x) == 1
    s = 0.0;
    se = 0.0;
else
    s = std(x);
    se = s / sqrt(numel(x));
end
end

function q = empirical_quantiles(x, probs)
x = sort(x(:));
probs = probs(:);
if isempty(x)
    q = NaN(size(probs));
    return;
end
if numel(x) == 1
    q = repmat(x, size(probs));
    return;
end
pos = 1 + (numel(x) - 1) * probs;
idx_lo = floor(pos);
idx_hi = ceil(pos);
w = pos - idx_lo;
q = (1 - w) .* x(idx_lo) + w .* x(idx_hi);
end

function edges = enforce_monotone_edges(edges)
edges = edges(:).';
for kk = 2:numel(edges)
    if edges(kk) <= edges(kk - 1)
        edges(kk) = edges(kk - 1) + max(abs(edges(kk - 1)), 1e-12) * 1e-6 + 1e-12;
    end
end
end

function bin_idx = assign_to_bins(x, edges)
num_bins = numel(edges) - 1;
bin_idx = zeros(size(x));
for kk = 1:num_bins
    if kk < num_bins
        mask = (x >= edges(kk)) & (x < edges(kk + 1));
    else
        mask = (x >= edges(kk)) & (x <= edges(kk + 1));
    end
    bin_idx(mask) = kk;
end
bin_idx(bin_idx == 0) = num_bins;
end

function r = average_tied_ranks(x)
x = x(:);
[x_sorted, order] = sort(x);
r_sorted = zeros(size(x));
kk = 1;
while kk <= numel(x_sorted)
    jj = kk;
    while jj < numel(x_sorted) && x_sorted(jj + 1) == x_sorted(kk)
        jj = jj + 1;
    end
    rank_val = 0.5 * (kk + jj);
    r_sorted(kk:jj) = rank_val;
    kk = jj + 1;
end
r = zeros(size(x));
r(order) = r_sorted;
end

function print_corr_line(name, stats_now)
fprintf('  %-26s Pearson=% .4f  Spearman=% .4f\n', ...
    name, stats_now.pearson, stats_now.spearman);
end

function print_group_line(name, grp)
fprintf('  %-26s low=%.3e  high=%.3e  high-low=%.3e\n', ...
    name, grp.low_mean, grp.high_mean, grp.high_minus_low);
end

function save_figure_pair(fig, prefix)
saveas(fig, [prefix '.png']);
savefig(fig, [prefix '.fig']);
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
