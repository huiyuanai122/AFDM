function out = sanity_check_timevary_c1_shift_rawres(user_cfg)
%SANITY_CHECK_TIMEVARY_C1_SHIFT_RAWRES
% Smoke-level verification for raw/residual Doppler-scale separation.
%
% Goals:
% 1) Verify alpha_raw and alpha_res are separated and residual is smaller.
% 2) Verify alpha_eff(t) = max_p |alpha_res_p(t)| changes over frames.
% 3) Verify frame-wise BER-vs-C1 minima and oracle C1 indices shift over time.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
common_dir = fullfile(project_root, 'matlab', 'common');
addpath(this_dir);
addpath(common_dir);

cfg = default_cfg(project_root);
cfg = merge_struct(cfg, user_cfg);
if isempty(cfg.target_mid_frame)
    cfg.target_mid_frame = ceil(cfg.num_frames / 2);
end
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

sys = build_system_cfg(cfg);
tv_cfg = build_timevary_cfg(cfg);

fprintf('==== Sanity Check: Time-Varying C1 Shift (raw/residual) ====\n');
fprintf('snr_db=%d, num_seq=%d, num_frames=%d, detector=%s\n', ...
    cfg.snr_db, cfg.num_seq, cfg.num_frames, cfg.detector_type);
fprintf('alpha_max_raw=%.1e, alpha_max_res=%.1e, motion_profile=%s, doppler_mode=%s\n', ...
    cfg.alpha_max_raw, cfg.alpha_max_res, cfg.motion_profile, cfg.doppler_mode);

sequence_data = repmat(struct('frames', [], 'alpha_eff', [], 'rep_score', 0.0), cfg.num_seq, 1);
alpha_eff_matrix = zeros(cfg.num_seq, cfg.num_frames);
alpha_raw_max_matrix = zeros(cfg.num_seq, cfg.num_frames);
alpha_res_max_matrix = zeros(cfg.num_seq, cfg.num_frames);
alpha_hat_matrix = zeros(cfg.num_seq, cfg.num_frames);

for sid = 1:cfg.num_seq
    sequence_data(sid) = generate_sequence_data(sid, cfg, sys, tv_cfg);
    alpha_eff_matrix(sid, :) = sequence_data(sid).alpha_eff(:).';
    alpha_raw_max_matrix(sid, :) = [sequence_data(sid).frames.alpha_raw_max];
    alpha_res_max_matrix(sid, :) = [sequence_data(sid).frames.alpha_res_max];
    alpha_hat_matrix(sid, :) = [sequence_data(sid).frames.alpha_hat];
end

rep_seq_idx = select_representative_sequence(alpha_eff_matrix);
rep_seq = sequence_data(rep_seq_idx);
frame_list = select_frame_triplet(cfg.num_frames, cfg.target_mid_frame);

fprintf('representative_sequence=%d (alpha_eff std=%.3e, range=%.3e)\n', ...
    rep_seq_idx, std(rep_seq.alpha_eff), max(rep_seq.alpha_eff) - min(rep_seq.alpha_eff));

best_idx_track = zeros(cfg.num_frames, 1);
best_c1_track = zeros(cfg.num_frames, 1);
best_ber_track = zeros(cfg.num_frames, 1);
frame_sweep(1:cfg.num_frames) = struct( ...
    'frame_index', 0, ...
    'alpha_eff', 0.0, ...
    'c1_base', 0.0, ...
    'c1_grid', [], ...
    'ber_grid', [], ...
    'bit_err_grid', [], ...
    'bits_per_eval', 0, ...
    'best_idx', 0, ...
    'best_c1', 0.0, ...
    'best_ber', 0.0);

for tt = 1:cfg.num_frames
    sweep = run_frame_c1_sweep(rep_seq.frames(tt), cfg, sys);
    frame_sweep(tt) = sweep;
    best_idx_track(tt) = sweep.best_idx;
    best_c1_track(tt) = sweep.best_c1;
    best_ber_track(tt) = sweep.best_ber;
end

plot_alpha_tracks(rep_seq, rep_seq_idx, cfg);
plot_three_frame_ber_curves(frame_sweep, frame_list, cfg);
plot_best_index_track(best_idx_track, cfg);

fprintf('\nRepresentative sequence frame-wise oracle summary\n');
fprintf(' frame    alpha_eff        best_idx        best_c1         best_ber\n');
for tt = 1:cfg.num_frames
    fprintf(' %5d   %11.3e   %11d   %13.6e   %11.3e\n', ...
        tt, rep_seq.frames(tt).alpha_eff, best_idx_track(tt), best_c1_track(tt), best_ber_track(tt));
end

acceptance = struct();
acceptance.residual_smaller = mean(alpha_res_max_matrix(:)) < mean(alpha_raw_max_matrix(:));
acceptance.alpha_eff_varies = any(abs(diff(rep_seq.alpha_eff(:))) > 1e-12);
acceptance.frame_minima_shift = numel(unique(best_idx_track(frame_list))) > 1;
acceptance.oracle_index_not_constant = numel(unique(best_idx_track)) > 1;

fprintf('\nAcceptance checks\n');
fprintf('  residual_smaller         : %d\n', acceptance.residual_smaller);
fprintf('  alpha_eff_varies         : %d\n', acceptance.alpha_eff_varies);
fprintf('  selected_frame_shift     : %d\n', acceptance.frame_minima_shift);
fprintf('  oracle_index_not_constant: %d\n', acceptance.oracle_index_not_constant);

results = struct();
results.cfg = cfg;
results.sys = sys;
results.tv_cfg = tv_cfg;
results.sequence_data = sequence_data;
results.rep_seq_idx = rep_seq_idx;
results.frame_list = frame_list;
results.frame_sweep = frame_sweep;
results.best_idx_track = best_idx_track;
results.best_c1_track = best_c1_track;
results.best_ber_track = best_ber_track;
results.alpha_eff_matrix = alpha_eff_matrix;
results.alpha_raw_max_matrix = alpha_raw_max_matrix;
results.alpha_res_max_matrix = alpha_res_max_matrix;
results.alpha_hat_matrix = alpha_hat_matrix;
results.acceptance = acceptance;

save(cfg.output_mat, 'results', '-v7.3');
fprintf('\nSaved sanity results to %s\n', cfg.output_mat);

out = results;
end

function cfg = default_cfg(project_root)
cfg = struct();
cfg.output_dir = fullfile(project_root, 'outputs');
cfg.output_mat = fullfile(cfg.output_dir, 'results_sanity_timevary_c1_shift_rawres.mat');

cfg.snr_db = 10;
cfg.num_seq = 10;
cfg.num_frames = 9;
cfg.target_mid_frame = [];

cfg.alpha_max_res = 1e-4;
cfg.alpha_max_raw = 5e-4;

cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.doppler_mode = 'common_with_path_residual';
cfg.motion_profile = 'smooth_ar';

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

cfg.detector_type = 'oamp';
cfg.oamp_iter = 10;
cfg.oamp_damping = 0.9;
cfg.num_noise_sweep = 3;

cfg.seed_base = 20260327;
cfg.c1_ratios = linspace(0.6, 1.4, 21);

cfg.N = 256;
cfg.Delta_f = 4;
cfg.fc = 12e3;
cfg.ell_max = 16;
cfg.P = 6;
cfg.Q = 0;
cfg.Nv = 2;
end

function sys = build_system_cfg(cfg)
sys = struct();
sys.N = cfg.N;
sys.Delta_f = cfg.Delta_f;
sys.T_sym = 1 / cfg.Delta_f;
sys.B = cfg.N * cfg.Delta_f;
sys.dt = 1 / sys.B;
sys.fc = cfg.fc;
sys.ell_max = cfg.ell_max;
sys.P = cfg.P;
sys.Q = cfg.Q;
sys.N_eff = cfg.N - 2 * cfg.Q;
sys.data_idx = (cfg.Q + 1):(cfg.N - cfg.Q);
sys.Nv = cfg.Nv;
sys.alpha_max_res = cfg.alpha_max_res;
sys.alpha_max_raw = cfg.alpha_max_raw;
sys.Lcpp = max(1, ceil(sys.ell_max / (1 - cfg.alpha_max_res)));
sys.Lcps = max(1, ceil(cfg.alpha_max_res * sys.N / (1 + cfg.alpha_max_res)));
sys.L = sys.N + sys.Lcpp + sys.Lcps;
sys.c2 = sqrt(2) / sys.N;
sys.bits_per_frame = 2 * numel(sys.data_idx);
end

function tv_cfg = build_timevary_cfg(cfg)
tv_base = struct( ...
    'rho_alpha', 0.98, ...
    'rho_h', cfg.rho_h, ...
    'alpha_max', cfg.alpha_max_res, ...
    'alpha_max_raw', cfg.alpha_max_raw, ...
    'alpha_max_res', cfg.alpha_max_res, ...
    'num_frames', cfg.num_frames, ...
    'doppler_mode', cfg.doppler_mode, ...
    'rho_acc', cfg.rho_acc, ...
    'sigma_acc', cfg.sigma_acc, ...
    'rho_delta', cfg.rho_delta, ...
    'sigma_delta', cfg.sigma_delta, ...
    'delta_max', [], ...
    'motion_profile', cfg.motion_profile, ...
    'path_projection_mode', cfg.path_projection_mode, ...
    'beta_min', cfg.beta_min, ...
    'beta_max', cfg.beta_max, ...
    'ell_mode', cfg.ell_mode, ...
    'pdp_mode', cfg.pdp_mode, ...
    'enable_resampling_comp', cfg.enable_resampling_comp, ...
    'alpha_hat_mode', cfg.alpha_hat_mode, ...
    'clip_alpha_res', true, ...
    'log_alpha_stats', true);
tv_cfg = get_timevary_defaults(tv_base);
end

function seq_data = generate_sequence_data(seq_idx, cfg, sys, tv_cfg)
seq_seed = safe_rng_seed(cfg.seed_base + 100000 * seq_idx);
rng(seq_seed, 'twister');

[seq_state, ch_t] = init_timevary_channel_state(sys.P, sys.ell_max, cfg.alpha_max_res, tv_cfg);

frames = repmat(struct( ...
    'seq_idx', 0, ...
    'frame_idx', 0, ...
    'ch', [], ...
    'alpha_raw', [], ...
    'alpha_res', [], ...
    'alpha_hat', 0.0, ...
    'h', [], ...
    'alpha_raw_max', 0.0, ...
    'alpha_res_max', 0.0, ...
    'alpha_eff', 0.0, ...
    'snr_db', 0.0, ...
    'noise_var', 0.0, ...
    'c1_base', 0.0, ...
    'c1_grid', [], ...
    'x_seed', 0, ...
    'noise_seed_base', 0, ...
    'x', [], ...
    'Heff_base', [], ...
    'y_base', []), cfg.num_frames, 1);
alpha_eff = zeros(cfg.num_frames, 1);

for tt = 1:cfg.num_frames
    if tt > 1
        [seq_state, ch_t] = step_timevary_channel_state(seq_state);
    end

    frame = struct();
    frame.seq_idx = seq_idx;
    frame.frame_idx = tt;
    frame.ch = ch_t;
    frame.alpha_raw = ch_t.alpha_raw(:);
    frame.alpha_res = ch_t.alpha_res(:);
    frame.alpha_hat = double(ch_t.alpha_hat);
    frame.h = ch_t.h(:);
    frame.alpha_raw_max = max(abs(frame.alpha_raw));
    frame.alpha_res_max = max(abs(frame.alpha_res));
    frame.alpha_eff = frame.alpha_res_max;
    frame.snr_db = cfg.snr_db;
    frame.noise_var = 1 / (10 ^ (cfg.snr_db / 10));
    frame.c1_base = compute_c1_base_from_alpha(frame.alpha_eff, sys);
    frame.c1_grid = frame.c1_base * cfg.c1_ratios(:).';
    frame.x_seed = safe_rng_seed(cfg.seed_base + 100000 * seq_idx + 1000 * tt + 1);
    frame.noise_seed_base = safe_rng_seed(cfg.seed_base + 100000 * seq_idx + 1000 * tt + 100);
    frame.x = generate_qpsk_frame(frame.x_seed, sys);
    [frame.Heff_base, frame.y_base] = simulate_single_observation(frame, frame.c1_base, 1, sys);

    frames(tt) = frame;
    alpha_eff(tt) = frame.alpha_eff;
end

seq_data = struct();
seq_data.frames = frames;
seq_data.alpha_eff = alpha_eff;
seq_data.rep_score = std(alpha_eff);
end

function [Heff, y] = simulate_single_observation(frame, c1, noise_rep, sys)
Heff = build_heff_for_c1(frame.ch, c1, sys);
noise_seed = safe_rng_seed(frame.noise_seed_base + noise_rep);
rng(noise_seed, 'twister');
w = sqrt(frame.noise_var / 2) * (randn(sys.N, 1) + 1j * randn(sys.N, 1));
y = Heff * frame.x + w;
end

function sweep = run_frame_c1_sweep(frame, cfg, sys)
c1_grid = frame.c1_grid(:).';
num_c1 = numel(c1_grid);
bit_err_grid = zeros(1, num_c1);
ber_grid = zeros(1, num_c1);
bits_per_eval = 0;

for m = 1:num_c1
    [bit_err_grid(m), bits_per_eval] = evaluate_frame_for_c1(frame, c1_grid(m), cfg, sys);
    ber_grid(m) = bit_err_grid(m) / max(bits_per_eval, 1);
end

[best_ber, best_idx] = min(ber_grid);

sweep = struct();
sweep.frame_index = frame.frame_idx;
sweep.alpha_eff = frame.alpha_eff;
sweep.c1_base = frame.c1_base;
sweep.c1_grid = c1_grid;
sweep.ber_grid = ber_grid;
sweep.bit_err_grid = bit_err_grid;
sweep.bits_per_eval = bits_per_eval;
sweep.best_idx = best_idx;
sweep.best_c1 = c1_grid(best_idx);
sweep.best_ber = best_ber;
end

function [bit_err_total, bits_total] = evaluate_frame_for_c1(frame, c1, cfg, sys)
Heff = build_heff_for_c1(frame.ch, c1, sys);
bit_err_total = 0;
bits_total = 0;

for ni = 1:cfg.num_noise_sweep
    noise_seed = safe_rng_seed(frame.noise_seed_base + ni);
    rng(noise_seed, 'twister');
    w = sqrt(frame.noise_var / 2) * (randn(sys.N, 1) + 1j * randn(sys.N, 1));
    y = Heff * frame.x + w;

    switch lower(cfg.detector_type)
        case 'oamp'
            x_hat = oamp_detector(y, Heff, frame.noise_var, cfg.oamp_iter, cfg.oamp_damping, sys.Q);
        case 'lmmse'
            x_hat = lmmse_detector(y, Heff, frame.noise_var);
        otherwise
            error('Unsupported detector_type=%s', cfg.detector_type);
    end

    bit_err_total = bit_err_total + count_qpsk_bit_errors(x_hat(sys.data_idx), frame.x(sys.data_idx));
    bits_total = bits_total + sys.bits_per_frame;
end
end

function Heff = build_heff_for_c1(ch, c1, sys)
XTB = precompute_idaf_basis(sys.N, c1, sys.c2);
Xext = add_cpp_cps_matrix(XTB, c1, sys.Lcpp, sys.Lcps);
YT = build_timescaling_G_sparse(sys.N, sys.L, sys.Lcpp, ch, sys.fc, sys.dt) * Xext;

n = (0:sys.N - 1).';
chirp1 = exp(-1j * 2 * pi * c1 * (n .^ 2));
chirp2 = exp(-1j * 2 * pi * sys.c2 * (n .^ 2));
Heff = afdm_demod_matrix(YT, chirp1, chirp2);

Hsub = Heff(sys.data_idx, sys.data_idx);
sub_power = (norm(Hsub, 'fro') ^ 2) / sys.N_eff;
Heff = Heff / sqrt(max(sub_power, 1e-12));
end

function x = generate_qpsk_frame(seed_in, sys)
rng(seed_in, 'twister');
x = zeros(sys.N, 1);
x(sys.data_idx) = qpsk_symbols(numel(sys.data_idx));
x = x * sqrt(sys.N / sys.N_eff);
end

function c1_base = compute_c1_base_from_alpha(alpha_eff, sys)
alpha_eff = max(double(alpha_eff), 0.0);
kmax = ceil((alpha_eff * sys.fc) * sys.T_sym);
den = (1 - 4 * alpha_eff * (sys.N - 1));
if den <= 0
    error('c1 design invalid: den<=0 for alpha_eff=%.3e', alpha_eff);
end
c1_base = (2 * kmax + 2 * alpha_eff * (sys.N - 1) + 2 * sys.Nv + 1) / (2 * sys.N * den);
end

function rep_seq_idx = select_representative_sequence(alpha_eff_matrix)
seq_std = std(alpha_eff_matrix, 0, 2);
seq_rng = max(alpha_eff_matrix, [], 2) - min(alpha_eff_matrix, [], 2);
[~, rep_seq_idx] = max(seq_std + 0.5 * seq_rng);
end

function frame_list = select_frame_triplet(num_frames, mid_frame)
frame_list = unique([2, mid_frame, max(2, num_frames - 1)], 'stable');
fallback = [1, mid_frame, num_frames];
for i = 1:numel(fallback)
    if numel(frame_list) >= 3
        break;
    end
    if ~ismember(fallback(i), frame_list)
        frame_list(end + 1) = fallback(i); %#ok<AGROW>
    end
end
frame_list = sort(frame_list(1:min(3, numel(frame_list))));
end

function plot_alpha_tracks(rep_seq, rep_seq_idx, cfg)
frame_axis = 1:numel(rep_seq.frames);
raw_track = [rep_seq.frames.alpha_raw_max];
res_track = [rep_seq.frames.alpha_res_max];
alpha_hat_track = [rep_seq.frames.alpha_hat];
alpha_eff_track = [rep_seq.frames.alpha_eff];

fig = figure('Color', 'w');
plot(frame_axis, raw_track, '-o', 'LineWidth', 1.4, 'DisplayName', 'max|alpha_{raw}|');
hold on;
plot(frame_axis, res_track, '-s', 'LineWidth', 1.4, 'DisplayName', 'max|alpha_{res}|');
plot(frame_axis, alpha_hat_track, '-d', 'LineWidth', 1.4, 'DisplayName', 'alpha_hat');
plot(frame_axis, alpha_eff_track, '--x', 'LineWidth', 1.6, 'DisplayName', 'alpha_eff(t)');
grid on;
xlabel('Frame Index');
ylabel('Alpha / Max-Abs Alpha');
title(sprintf('Representative Sequence %d: raw/residual alpha tracks', rep_seq_idx));
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_sanity_alpha_tracks_rawres'));
end

function plot_three_frame_ber_curves(frame_sweep, frame_list, cfg)
colors = lines(numel(frame_list));
fig = figure('Color', 'w');
hold on;
for k = 1:numel(frame_list)
    tt = frame_list(k);
    sweep = frame_sweep(tt);
    semilogy(sweep.c1_grid, sweep.ber_grid, '-o', 'Color', colors(k, :), ...
        'LineWidth', 1.4, 'DisplayName', sprintf('frame %d', tt));
    semilogy(sweep.best_c1, sweep.best_ber, 'p', 'Color', colors(k, :), ...
        'MarkerFaceColor', colors(k, :), 'MarkerSize', 10, ...
        'HandleVisibility', 'off');
end
grid on;
xlabel('C1');
ylabel('BER');
title('BER-vs-C1 on early / middle / late frames');
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_sanity_ber_vs_c1_3frames'));
end

function plot_best_index_track(best_idx_track, cfg)
frame_axis = 1:numel(best_idx_track);
fig = figure('Color', 'w');
stairs(frame_axis, best_idx_track, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Frame Index');
ylabel('Best C1 Index');
title('Frame-wise oracle best C1 index');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_sanity_best_c1_index_vs_frame'));
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

function seed_u32 = safe_rng_seed(seed_in)
seed_u32 = mod(double(seed_in), 2^32 - 1);
if seed_u32 < 0
    seed_u32 = seed_u32 + (2^32 - 1);
end
end

function XTB = precompute_idaf_basis(N, c1, c2)
n = (0:N - 1).';
m = 0:N - 1;
phase_n = exp(1j * 2 * pi * c1 * (n .^ 2));
phase_m = exp(1j * 2 * pi * c2 * (m .^ 2));
W = exp(1j * 2 * pi * (n * m) / N) / sqrt(N);
XTB = (phase_n .* W) .* phase_m;
end

function Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps)
N = size(XTB, 1);
if Lcpp == 0 && Lcps == 0
    Xext = XTB;
    return;
end

if Lcpp > 0
    n_pre = (-Lcpp:-1).';
    idx_pre = n_pre + N + 1;
    phase_pre = exp(-1j * 2 * pi * c1 * (N^2 + 2 * N * n_pre));
    Xpre = XTB(idx_pre, :) .* phase_pre;
else
    Xpre = zeros(0, N, 'like', XTB);
end

if Lcps > 0
    n_suf = (N:(N + Lcps - 1)).';
    idx_suf = (n_suf - N + 1);
    phase_suf = exp(+1j * 2 * pi * c1 * (N^2 + 2 * N * n_suf));
    Xsuf = XTB(idx_suf, :) .* phase_suf;
else
    Xsuf = zeros(0, N, 'like', XTB);
end

Xext = [Xpre; XTB; Xsuf];
end

function G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt)
P = ch.P;
ell = ch.ell;
if isfield(ch, 'alpha_res')
    alpha = ch.alpha_res;
else
    alpha = ch.alpha;
end
h = ch.h;

n = (0:N - 1).';
max_nnz = N * P * 2;
I = zeros(max_nnz, 1);
J = zeros(max_nnz, 1);
V = zeros(max_nnz, 1) + 1j * zeros(max_nnz, 1);
ptr = 0;

for i = 1:P
    idx = (1 + alpha(i)) * n - ell(i) + Lcpp;
    idx0 = floor(idx);
    frac = idx - idx0;
    idx1 = idx0 + 1;
    phase = exp(1j * 2 * pi * (alpha(i) * fc) * (n * dt));

    v0 = (idx0 >= 0) & (idx0 <= (L - 1));
    nn0 = sum(v0);
    if nn0 > 0
        rows = find(v0);
        cols = idx0(v0) + 1;
        vals = h(i) * phase(v0) .* (1 - frac(v0));
        I(ptr + 1:ptr + nn0) = rows;
        J(ptr + 1:ptr + nn0) = cols;
        V(ptr + 1:ptr + nn0) = vals;
        ptr = ptr + nn0;
    end

    v1 = (idx1 >= 0) & (idx1 <= (L - 1)) & (frac > 0);
    nn1 = sum(v1);
    if nn1 > 0
        rows = find(v1);
        cols = idx1(v1) + 1;
        vals = h(i) * phase(v1) .* frac(v1);
        I(ptr + 1:ptr + nn1) = rows;
        J(ptr + 1:ptr + nn1) = cols;
        V(ptr + 1:ptr + nn1) = vals;
        ptr = ptr + nn1;
    end
end

I = I(1:ptr);
J = J(1:ptr);
V = V(1:ptr);
G = sparse(I, J, V, N, L);
end

function Heff = afdm_demod_matrix(YT, chirp1, chirp2)
N = size(YT, 1);
tmp = fft(YT .* chirp1, [], 1) / sqrt(N);
Heff = tmp .* chirp2;
end

function s = qpsk_symbols(M)
b1 = randi([0, 1], M, 1);
b2 = randi([0, 1], M, 1);
re = 1 - 2 * b1;
im = 1 - 2 * b2;
s = (re + 1j * im) / sqrt(2);
end

function bit_err = count_qpsk_bit_errors(x_hat, x_true)
b_hat = qpsk_demod_bits_gray(x_hat);
b_true = qpsk_demod_bits_gray(x_true);
bit_err = sum(b_hat ~= b_true);
end

function bits = qpsk_demod_bits_gray(x)
x = x(:);
amp = median(abs(x));
if amp < 1e-12
    amp = 1;
end
x = x / amp;

const = [1 + 1j; -1 + 1j; -1 - 1j; 1 - 1j] / sqrt(2);
labels = [0 0; 0 1; 1 1; 1 0];

d2 = abs(x - const.').^2;
[~, idx] = min(d2, [], 2);
bits2 = labels(idx, :);
bits = reshape(bits2.', [], 1);
end
