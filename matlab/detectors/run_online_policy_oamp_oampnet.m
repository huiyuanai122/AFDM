%% run_online_policy_oamp_oampnet.m
% Online measured evaluation:
% RL-selected C1 + OAMP/OAMPNet detector (plus fixed/oracle baselines)
%
% Outputs:
% - results/ber_results_policy_online_oamp_oampnet.csv
% - results/fig3_ber_vs_snr_main_online.csv
% - results/fig4_ablation_gain_online.csv
% - results/policy_online_detector_eval_result.mat

clear; clc; close all;

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
results_dir = fullfile(project_root, 'results');
data_dir = fullfile(project_root, 'data');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end
addpath(this_dir);

%% ===== Config =====
version = 'tsv1';
snr_db_list = [14 18];
num_ch_min = 80;      % minimum channels per SNR
num_noise = 4;        % noise repeats per (channel, action)
target_ber_floor = 1e-6; % statistical floor target: 0.5/bit_cnt <= target_ber_floor
seed_base = 20260308;

% OAMP config
oamp_iter = 10;
oamp_damping = 0.9;
Q = 0;

%% ===== Load RL policy (MATLAB-exported) =====
policy_path = fullfile(results_dir, 'rl_c1_policy_matlab_params.mat');
if ~exist(policy_path, 'file')
    error('Missing policy mat: %s. Run python/rl_c1/export_policy_to_matlab.py first.', policy_path);
end
policy = load(policy_path);

%% ===== Load OAMPNet params =====
params_path = fullfile(data_dir, ['oampnet_v4_' version '_params.mat']);
if ~exist(params_path, 'file')
    error('Missing OAMPNet params: %s', params_path);
end
oampnet_params = load(params_path);

%% ===== AFDM/System params =====
N = 256;
Delta_f = 4;
T_sym = 1 / Delta_f;
B = N * Delta_f;
dt = 1 / B;
fc = 12e3;
alpha_max = 1e-4;
ell_max = 16;
P = 6;
Nv = 2;
N_eff = N - 2 * Q;
data_idx = (Q + 1):(N - Q);
bits_per_sample = 2 * numel(data_idx);
min_bits_per_snr = ceil(0.5 / target_ber_floor);
num_ch_needed = ceil(min_bits_per_snr / (num_noise * bits_per_sample));
num_ch = max(num_ch_min, num_ch_needed);

Lcpp = max(1, ceil(ell_max / (1 - alpha_max)));
Lcps = max(1, ceil(alpha_max * N / (1 + alpha_max)));

if isfield(policy, 'c1_grid')
    c1_grid = double(policy.c1_grid(:));
else
    kmax = ceil((alpha_max * fc) * T_sym);
    den = (1 - 4 * alpha_max * (N - 1));
    c1_base = (2 * kmax + 2 * alpha_max * (N - 1) + 2 * Nv + 1) / (2 * N * den);
    ratios = linspace(0.6, 1.4, 21);
    c1_grid = c1_base * ratios(:);
end
num_actions = numel(c1_grid);

kmax = ceil((alpha_max * fc) * T_sym);
den = (1 - 4 * alpha_max * (N - 1));
c1_base_formula = (2 * kmax + 2 * alpha_max * (N - 1) + 2 * Nv + 1) / (2 * N * den);
[~, base_idx] = min(abs(c1_grid - c1_base_formula));  % 1-based
c2 = sqrt(2) / N;

fprintf('==== Online Policy + Detector Evaluation ====\n');
fprintf('policy: %s\n', policy_path);
if isfield(policy, 'reward_key')
    reward_key_str = as_text(policy.reward_key);
    fprintf('policy reward_key: %s\n', reward_key_str);
end
fprintf('oampnet params: %s\n', params_path);
fprintf('snr list: %s\n', mat2str(snr_db_list));
fprintf('num_ch=%d (min=%d, needed=%d), num_noise=%d, num_actions=%d, base_idx=%d\n', ...
    num_ch, num_ch_min, num_ch_needed, num_noise, num_actions, base_idx);
fprintf('target_ber_floor=%.1e, min_bits_per_snr=%d, bits_per_sample=%d\n', ...
    target_ber_floor, min_bits_per_snr, bits_per_sample);

%% ===== Containers =====
num_snr = numel(snr_db_list);
ber_fixed_oamp_raw = zeros(num_snr, 1);
ber_fixed_oampnet_raw = zeros(num_snr, 1);
ber_rl_oamp_raw = zeros(num_snr, 1);
ber_rl_oampnet_raw = zeros(num_snr, 1);
ber_oracle_oamp_raw = zeros(num_snr, 1);
ber_oracle_oampnet_raw = zeros(num_snr, 1);

ber_fixed_oamp = zeros(num_snr, 1);      % floor-adjusted for plotting/export
ber_fixed_oampnet = zeros(num_snr, 1);
ber_rl_oamp = zeros(num_snr, 1);
ber_rl_oampnet = zeros(num_snr, 1);
ber_oracle_oamp = zeros(num_snr, 1);
ber_oracle_oampnet = zeros(num_snr, 1);

err_fixed_oamp_all = zeros(num_snr, 1);
err_fixed_oampnet_all = zeros(num_snr, 1);
err_rl_oamp_all = zeros(num_snr, 1);
err_rl_oampnet_all = zeros(num_snr, 1);
err_oracle_oamp_all = zeros(num_snr, 1);
err_oracle_oampnet_all = zeros(num_snr, 1);
bit_cnt_all = zeros(num_snr, 1);
ber_floor = zeros(num_snr, 1);

policy_idx_hist = zeros(num_snr, num_actions);
oracle_idx_oampnet_hist = zeros(num_snr, num_actions);

for si = 1:num_snr
    snr_db = snr_db_list(si);
    noise_var = 1 / (10^(snr_db / 10));

    err_fixed_oamp = 0;
    err_fixed_oampnet = 0;
    err_rl_oamp = 0;
    err_rl_oampnet = 0;
    err_oracle_oamp = 0;
    err_oracle_oampnet = 0;
    bit_cnt_method = 0;

    for ch_k = 1:num_ch
        rng(seed_base + 100000 * si + ch_k);
        ch = gen_channel_paper_aligned(P, ell_max, alpha_max);

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        ber_action_oamp = zeros(num_actions, 1);
        ber_action_oampnet = zeros(num_actions, 1);
        err_action_oamp = zeros(num_actions, 1);
        err_action_oampnet = zeros(num_actions, 1);
        bit_action = zeros(num_actions, 1);
        Heff_base = [];

        for a = 1:num_actions
            c1 = c1_grid(a);
            XTB = precompute_idaf_basis(N, c1, c2);
            Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);
            L = size(Xext, 1);
            G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);
            YT = G * Xext;

            n = (0:N-1).';
            chirp1 = exp(-1j * 2 * pi * c1 * (n .^ 2));
            chirp2 = exp(-1j * 2 * pi * c2 * (n .^ 2));
            Heff = afdm_demod_matrix(YT, chirp1, chirp2);

            Hsub = Heff(data_idx, data_idx);
            sub_power = (norm(Hsub, 'fro')^2) / N_eff;
            Heff = Heff / sqrt(max(sub_power, 1e-12));

            if a == base_idx
                Heff_base = Heff;
            end

            err_o = 0;
            err_n = 0;
            bits = 0;
            for ni = 1:num_noise
                rng(seed_base + 100000 * si + 1000 * ch_k + 10 * a + ni);
                w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                y = Heff * x + w;

                x_o = oamp_detector(y, Heff, noise_var, oamp_iter, oamp_damping, Q);
                x_n = oampnet_detector(y, Heff, noise_var, oampnet_params, Q);

                err_o = err_o + count_qpsk_bit_errors(x_o(data_idx), x(data_idx));
                err_n = err_n + count_qpsk_bit_errors(x_n(data_idx), x(data_idx));
                bits = bits + 2 * numel(data_idx);
            end

            err_action_oamp(a) = err_o;
            err_action_oampnet(a) = err_n;
            bit_action(a) = bits;
            ber_action_oamp(a) = err_o / bits;
            ber_action_oampnet(a) = err_n / bits;
        end

        state_feat = extract_state_features(Heff_base, ch, snr_db, data_idx);
        rl_idx = policy_greedy_from_state(state_feat, policy);
        [~, oracle_idx_o] = min(ber_action_oamp);
        [~, oracle_idx_n] = min(ber_action_oampnet);

        err_fixed_oamp = err_fixed_oamp + err_action_oamp(base_idx);
        err_fixed_oampnet = err_fixed_oampnet + err_action_oampnet(base_idx);
        err_rl_oamp = err_rl_oamp + err_action_oamp(rl_idx);
        err_rl_oampnet = err_rl_oampnet + err_action_oampnet(rl_idx);
        err_oracle_oamp = err_oracle_oamp + err_action_oamp(oracle_idx_o);
        err_oracle_oampnet = err_oracle_oampnet + err_action_oampnet(oracle_idx_n);
        bit_cnt_method = bit_cnt_method + bit_action(base_idx);

        policy_idx_hist(si, rl_idx) = policy_idx_hist(si, rl_idx) + 1;
        oracle_idx_oampnet_hist(si, oracle_idx_n) = oracle_idx_oampnet_hist(si, oracle_idx_n) + 1;
    end

    err_fixed_oamp_all(si) = err_fixed_oamp;
    err_fixed_oampnet_all(si) = err_fixed_oampnet;
    err_rl_oamp_all(si) = err_rl_oamp;
    err_rl_oampnet_all(si) = err_rl_oampnet;
    err_oracle_oamp_all(si) = err_oracle_oamp;
    err_oracle_oampnet_all(si) = err_oracle_oampnet;
    bit_cnt_all(si) = bit_cnt_method;
    ber_floor(si) = 0.5 / max(bit_cnt_method, 1);

    ber_fixed_oamp_raw(si) = err_fixed_oamp / max(bit_cnt_method, 1);
    ber_fixed_oampnet_raw(si) = err_fixed_oampnet / max(bit_cnt_method, 1);
    ber_rl_oamp_raw(si) = err_rl_oamp / max(bit_cnt_method, 1);
    ber_rl_oampnet_raw(si) = err_rl_oampnet / max(bit_cnt_method, 1);
    ber_oracle_oamp_raw(si) = err_oracle_oamp / max(bit_cnt_method, 1);
    ber_oracle_oampnet_raw(si) = err_oracle_oampnet / max(bit_cnt_method, 1);

    ber_fixed_oamp(si) = adjust_ber_for_plot(ber_fixed_oamp_raw(si), err_fixed_oamp, bit_cnt_method);
    ber_fixed_oampnet(si) = adjust_ber_for_plot(ber_fixed_oampnet_raw(si), err_fixed_oampnet, bit_cnt_method);
    ber_rl_oamp(si) = adjust_ber_for_plot(ber_rl_oamp_raw(si), err_rl_oamp, bit_cnt_method);
    ber_rl_oampnet(si) = adjust_ber_for_plot(ber_rl_oampnet_raw(si), err_rl_oampnet, bit_cnt_method);
    ber_oracle_oamp(si) = adjust_ber_for_plot(ber_oracle_oamp_raw(si), err_oracle_oamp, bit_cnt_method);
    ber_oracle_oampnet(si) = adjust_ber_for_plot(ber_oracle_oampnet_raw(si), err_oracle_oampnet, bit_cnt_method);

    fprintf('SNR=%ddB | floor=%.2e, bits=%d | fixed_oamp=%.3e, rl_oamp=%.3e, fixed_oampnet=%.3e, rl_oampnet=%.3e, oracle_oampnet=%.3e\n', ...
        snr_db, ber_floor(si), bit_cnt_method, ber_fixed_oamp(si), ber_rl_oamp(si), ber_fixed_oampnet(si), ber_rl_oampnet(si), ber_oracle_oampnet(si));
end

%% ===== Save wide CSV =====
wide_csv = fullfile(results_dir, 'ber_results_policy_online_oamp_oampnet.csv');
fid = fopen(wide_csv, 'w');
fprintf(fid, ['snr_db,fixed_oamp,fixed_oampnet,rl_oamp,rl_oampnet,oracle_oamp,oracle_oampnet,' ...
    'fixed_oamp_raw,fixed_oampnet_raw,rl_oamp_raw,rl_oampnet_raw,oracle_oamp_raw,oracle_oampnet_raw,' ...
    'err_fixed_oamp,err_fixed_oampnet,err_rl_oamp,err_rl_oampnet,err_oracle_oamp,err_oracle_oampnet,' ...
    'bit_count,ber_floor,num_ch,num_noise,target_ber_floor\n']);
for si = 1:num_snr
    fprintf(fid, ['%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,' ...
        '%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,' ...
        '%d,%d,%d,%d,%d,%d,%d,%.6e,%d,%d,%.6e\n'], ...
        snr_db_list(si), ber_fixed_oamp(si), ber_fixed_oampnet(si), ber_rl_oamp(si), ber_rl_oampnet(si), ...
        ber_oracle_oamp(si), ber_oracle_oampnet(si), ...
        ber_fixed_oamp_raw(si), ber_fixed_oampnet_raw(si), ber_rl_oamp_raw(si), ber_rl_oampnet_raw(si), ...
        ber_oracle_oamp_raw(si), ber_oracle_oampnet_raw(si), ...
        round(err_fixed_oamp_all(si)), round(err_fixed_oampnet_all(si)), round(err_rl_oamp_all(si)), round(err_rl_oampnet_all(si)), ...
        round(err_oracle_oamp_all(si)), round(err_oracle_oampnet_all(si)), round(bit_cnt_all(si)), ...
        ber_floor(si), num_ch, num_noise, target_ber_floor);
end
fclose(fid);

%% ===== Save Fig3 long CSV (online measured) =====
fig3_csv = fullfile(results_dir, 'fig3_ber_vs_snr_main_online.csv');
fid = fopen(fig3_csv, 'w');
fprintf(fid, 'snr_db,method,ber,ber_raw,ber_floor,bit_count\n');
for si = 1:num_snr
    s = snr_db_list(si);
    fprintf(fid, '%d,fixed_oamp,%.6e,%.6e,%.6e,%d\n', s, ber_fixed_oamp(si), ber_fixed_oamp_raw(si), ber_floor(si), round(bit_cnt_all(si)));
    fprintf(fid, '%d,fixed_oampnet,%.6e,%.6e,%.6e,%d\n', s, ber_fixed_oampnet(si), ber_fixed_oampnet_raw(si), ber_floor(si), round(bit_cnt_all(si)));
    fprintf(fid, '%d,rl_oamp,%.6e,%.6e,%.6e,%d\n', s, ber_rl_oamp(si), ber_rl_oamp_raw(si), ber_floor(si), round(bit_cnt_all(si)));
    fprintf(fid, '%d,rl_oampnet,%.6e,%.6e,%.6e,%d\n', s, ber_rl_oampnet(si), ber_rl_oampnet_raw(si), ber_floor(si), round(bit_cnt_all(si)));
    fprintf(fid, '%d,oracle_oampnet,%.6e,%.6e,%.6e,%d\n', s, ber_oracle_oampnet(si), ber_oracle_oampnet_raw(si), ber_floor(si), round(bit_cnt_all(si)));
end
fclose(fid);

%% ===== Save Fig4 ablation CSV (online measured) =====
fig4_csv = fullfile(results_dir, 'fig4_ablation_gain_online.csv');
fid = fopen(fig4_csv, 'w');
fprintf(fid, 'snr_db,method,ber,ber_raw,ber_floor,bit_count,rel_gain_percent,rel_gain_raw_percent\n');
for si = 1:num_snr
    s = snr_db_list(si);
    ref = ber_fixed_oamp(si);
    ref_raw = ber_fixed_oamp_raw(si);
    methods = {'fixed_oamp','fixed_oampnet','rl_oamp','rl_oampnet'};
    vals = [ber_fixed_oamp(si), ber_fixed_oampnet(si), ber_rl_oamp(si), ber_rl_oampnet(si)];
    vals_raw = [ber_fixed_oamp_raw(si), ber_fixed_oampnet_raw(si), ber_rl_oamp_raw(si), ber_rl_oampnet_raw(si)];
    for mi = 1:numel(methods)
        g = (ref - vals(mi)) / max(ref, 1e-12) * 100;
        g_raw = (ref_raw - vals_raw(mi)) / max(ref_raw, 1e-12) * 100;
        fprintf(fid, '%d,%s,%.6e,%.6e,%.6e,%d,%.6f,%.6f\n', ...
            s, methods{mi}, vals(mi), vals_raw(mi), ber_floor(si), round(bit_cnt_all(si)), g, g_raw);
    end
end
fclose(fid);

%% ===== Save MAT =====
result = struct();
result.snr_db_list = snr_db_list;
result.num_ch = num_ch;
result.num_ch_min = num_ch_min;
result.num_noise = num_noise;
result.bits_per_sample = bits_per_sample;
result.min_bits_per_snr = min_bits_per_snr;
result.target_ber_floor = target_ber_floor;
result.c1_grid = c1_grid;
result.base_idx = base_idx;
result.ber_fixed_oamp = ber_fixed_oamp;
result.ber_fixed_oampnet = ber_fixed_oampnet;
result.ber_rl_oamp = ber_rl_oamp;
result.ber_rl_oampnet = ber_rl_oampnet;
result.ber_oracle_oamp = ber_oracle_oamp;
result.ber_oracle_oampnet = ber_oracle_oampnet;
result.ber_fixed_oamp_raw = ber_fixed_oamp_raw;
result.ber_fixed_oampnet_raw = ber_fixed_oampnet_raw;
result.ber_rl_oamp_raw = ber_rl_oamp_raw;
result.ber_rl_oampnet_raw = ber_rl_oampnet_raw;
result.ber_oracle_oamp_raw = ber_oracle_oamp_raw;
result.ber_oracle_oampnet_raw = ber_oracle_oampnet_raw;
result.err_fixed_oamp_all = err_fixed_oamp_all;
result.err_fixed_oampnet_all = err_fixed_oampnet_all;
result.err_rl_oamp_all = err_rl_oamp_all;
result.err_rl_oampnet_all = err_rl_oampnet_all;
result.err_oracle_oamp_all = err_oracle_oamp_all;
result.err_oracle_oampnet_all = err_oracle_oampnet_all;
result.bit_cnt_all = bit_cnt_all;
result.ber_floor = ber_floor;
result.policy_idx_hist = policy_idx_hist;
result.oracle_idx_oampnet_hist = oracle_idx_oampnet_hist;

mat_path = fullfile(results_dir, 'policy_online_detector_eval_result.mat');
save(mat_path, 'result', '-v7.3');

fprintf('\nSaved:\n  %s\n  %s\n  %s\n  %s\n', wide_csv, fig3_csv, fig4_csv, mat_path);

%% ===== Helpers =====
function a_idx = policy_greedy_from_state(state_feat, policy)
    x = (state_feat(:) - policy.state_mean(:)) ./ max(policy.state_std(:), 1e-6);
    L = round(double(policy.num_linear_layers(1)));
    h = x;
    for li = 1:L
        W = policy.(sprintf('W%d', li));
        b = policy.(sprintf('b%d', li));
        h = W * h + b;
        if li < L
            h = max(h, 0); % ReLU
        end
    end
    [~, a_idx] = max(h);
end

function f = extract_state_features(Heff, ch, snr_db, data_idx)
    Hsub = Heff(data_idx, data_idx);
    svals = svd(Hsub);
    svals = svals(:);

    topk = 4;
    sv_feat = zeros(1, topk);
    m = min(topk, numel(svals));
    denom = max(svals(1), 1e-12);
    sv_feat(1:m) = (svals(1:m) / denom).';

    frob_norm = norm(Hsub, 'fro') / sqrt(size(Hsub, 1));
    cond_log10 = log10(cond(Hsub + 1e-6 * eye(size(Hsub))) + 1e-12);
    max_alpha = max(abs(ch.alpha));
    mean_alpha = mean(abs(ch.alpha));
    delay_spread = std(double(ch.ell)) / max(double(max(ch.ell) + 1), 1.0);
    h_l4_over_l2 = norm(ch.h, 4) / max(norm(ch.h, 2), 1e-12);
    snr_norm = snr_db / 30.0;

    f = [frob_norm, cond_log10, sv_feat, max_alpha, mean_alpha, delay_spread, h_l4_over_l2, snr_norm];
end

function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
    ell = sort(randi([0, ell_max], P, 1));
    ell = ell - ell(1);
    alpha = (2 * rand(P, 1) - 1) * alpha_max;

    ell_rms = max(1, ell_max / 3);
    pwr = exp(-ell / ell_rms);
    h = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
    h = h .* sqrt(pwr);
    h = h / max(norm(h), 1e-12);

    ch.P = P;
    ch.ell = ell;
    ch.alpha = alpha;
    ch.h = h;
end

function XTB = precompute_idaf_basis(N, c1, c2)
    n = (0:N-1).';
    m = 0:N-1;
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
    alpha = ch.alpha;
    h = ch.h;

    n = (0:N-1).';
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

function ber_plot = adjust_ber_for_plot(ber_raw, err_cnt, bit_cnt)
    if bit_cnt <= 0
        ber_plot = 1.0;
        return;
    end
    if err_cnt > 0
        ber_plot = ber_raw;
    else
        ber_plot = 0.5 / bit_cnt;
    end
end

function s = as_text(x)
    if iscell(x)
        x = x{1};
    end
    if isstring(x)
        s = char(x);
        return;
    end
    if ischar(x)
        s = x;
        return;
    end
    if isnumeric(x)
        s = char(x(:).');
        return;
    end
    s = 'unknown';
end
