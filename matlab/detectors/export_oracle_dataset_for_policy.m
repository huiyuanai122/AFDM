%% export_oracle_dataset_for_policy.m
% Unified offline dataset export for C1 policy learning.
%
% Exports:
% - state               [S, D]
% - reward              [S, M] (default training reward; currently reward_mix)
% - reward_ber          [S, M] = -BER(a), where BER is measured by detector_target
% - reward_proxy        [S, M] = -proxy_norm(a)
% - reward_mix          [S, M] = -(BER + lambda * proxy_norm)
% - ber_actions         [S, M] true BER table
% - mse_proxy_actions   [S, M] smooth proxy table
% - oracle_action       [S, 1] argmin BER (1-based)
% - oracle_action_reward[S, 1] argmax reward_mix (1-based)
% - snr_db              [S, 1]
% - sequence_id         [S, 1] (for switch-rate evaluation)
% - time_index          [S, 1]
%
% Detector-aware export:
%   detector_target = 'lmmse' | 'oamp' | 'oampnet'
%   (recommended for online OAMPNet linkage: detector_target='oampnet')
%
% Python loader in python/rl_c1 supports:
%   reward_key=reward / reward_ber / reward_proxy / reward_mix

clear; clc; close all;

%% ===== 0) Export mode =====
use_timevary = true;      % true: sequence AR(1) export, false: i.i.d. static export
output_file = 'oracle_policy_dataset.mat';

% Detector target used to build BER/reward table for policy learning.
% Options: 'lmmse' | 'oamp' | 'oampnet'
detector_target = 'oampnet';
oamp_iter = 10;
oamp_damping = 0.9;
oampnet_param_version = 'tsv1';

% Time-vary settings (used when use_timevary=true)
num_seq = 300;
num_frames = 10;          % total samples = num_seq * num_frames
rho_alpha = 0.98;
rho_h = 0.98;

% Static setting (used when use_timevary=false)
num_samples_static = 3000;

%% ===== 1) System/channel parameters =====
N = 256;
Delta_f = 4;
T_sym = 1 / Delta_f;
B = N * Delta_f;
dt = 1 / B;
fc = 12e3;
alpha_max = 1e-4;
ell_max = 16;
P = 6;
Q = 0;
N_eff = N - 2 * Q;
data_idx = (Q + 1):(N - Q);

Lcpp = max(1, ceil(ell_max / (1 - alpha_max)));
Lcps = max(1, ceil(alpha_max * N / (1 + alpha_max)));

Nv = 2;
kmax = ceil((alpha_max * fc) * T_sym);
den = (1 - 4 * alpha_max * (N - 1));
c1_base = (2 * kmax + 2 * alpha_max * (N - 1) + 2 * Nv + 1) / (2 * N * den);
c2 = sqrt(2) / N;

ratios = linspace(0.6, 1.4, 21);
c1_grid = c1_base * ratios;
num_actions = numel(c1_grid);
[~, base_action] = min(abs(c1_grid - c1_base));

%% ===== 2) Dataset settings =====
snr_db_list = [14 18];
num_noise = 8;                 % more averaging to reduce BER quantization
reward_lambda_proxy = 1e-4;    % tie-breaker weight for smooth proxy

if use_timevary
    num_samples = num_seq * num_frames;
else
    num_samples = num_samples_static;
    num_seq = num_samples;
    num_frames = 1;
end

rng(3);
this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
data_dir = fullfile(project_root, 'data');

oampnet_params = [];
if strcmpi(detector_target, 'oampnet')
    oampnet_param_path = fullfile(data_dir, ['oampnet_v4_' oampnet_param_version '_params.mat']);
    if ~exist(oampnet_param_path, 'file')
        error('Missing OAMPNet params: %s', oampnet_param_path);
    end
    oampnet_params = load(oampnet_param_path);
else
    oampnet_param_path = '';
end

% Feature dimension:
% [frob_norm, cond_log10, sv1..sv4, max|alpha|, mean|alpha|,
%  delay_spread_norm, h_l4_over_l2, snr_norm]
feat_dim = 11;

state = zeros(num_samples, feat_dim, 'single');
reward = zeros(num_samples, num_actions, 'single');
reward_ber = zeros(num_samples, num_actions, 'single');
reward_proxy = zeros(num_samples, num_actions, 'single');
reward_mix = zeros(num_samples, num_actions, 'single');
ber_actions = zeros(num_samples, num_actions, 'single');
mse_proxy_actions = zeros(num_samples, num_actions, 'single');

oracle_action = zeros(num_samples, 1, 'int32');
oracle_action_reward = zeros(num_samples, 1, 'int32');
snr_db = zeros(num_samples, 1, 'single');
sequence_id = zeros(num_samples, 1, 'int32');
time_index = zeros(num_samples, 1, 'int32');

sample_idx = 0;
for sid = 1:num_seq
    if use_timevary
        ch0 = gen_channel_paper_aligned(P, ell_max, alpha_max);
        ell = ch0.ell;
        alpha_t = ch0.alpha;
        h_t = ch0.h;

        ell_rms = max(1, ell_max / 3);
        pwr = exp(-ell / ell_rms);

        snr_pick_seq = snr_db_list(randi(numel(snr_db_list)));
    end

    for tt = 1:num_frames
        sample_idx = sample_idx + 1;

        if use_timevary
            if tt > 1
                alpha_t = rho_alpha * alpha_t + sqrt(1 - rho_alpha^2) * alpha_max * randn(P, 1);
                alpha_t = min(max(alpha_t, -alpha_max), alpha_max);

                innov = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
                h_t = rho_h * h_t + sqrt(1 - rho_h^2) * innov;
                h_t = h_t .* sqrt(pwr);
                h_t = h_t / max(norm(h_t), 1e-12);
            end
            ch = struct('P', P, 'ell', ell, 'alpha', alpha_t, 'h', h_t);
            snr_pick = snr_pick_seq;
        else
            ch = gen_channel_paper_aligned(P, ell_max, alpha_max);
            snr_pick = snr_db_list(randi(numel(snr_db_list)));
        end

        noise_var = 1 / (10 ^ (snr_pick / 10));

        XTB0 = precompute_idaf_basis(N, c1_base, c2);
        Xext0 = add_cpp_cps_matrix(XTB0, c1_base, Lcpp, Lcps);
        L = size(Xext0, 1);
        G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        ber_vec = zeros(num_actions, 1);
        proxy_vec = zeros(num_actions, 1);
        Heff_base = [];

        for a = 1:num_actions
            c1 = c1_grid(a);
            XTB = precompute_idaf_basis(N, c1, c2);
            Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);
            YT = G * Xext;

            idx_n = (0:N-1).';
            chirp1 = exp(-1j * 2 * pi * c1 * (idx_n .^ 2));
            chirp2 = exp(-1j * 2 * pi * c2 * (idx_n .^ 2));
            Heff = afdm_demod_matrix(YT, chirp1, chirp2);

            Hsub = Heff(data_idx, data_idx);
            sub_power = (norm(Hsub, 'fro') ^ 2) / N_eff;
            Heff = Heff / sqrt(max(sub_power, 1e-12));

            if a == base_action
                Heff_base = Heff;
            end

            err = 0;
            bits = 0;
            proxy_acc = 0;
            for ni = 1:num_noise
                seed = 1000000 * sample_idx + 100 * a + ni;
                rng(seed);
                w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                y = Heff * x + w;
                x_det = run_target_detector(y, Heff, noise_var, detector_target, oamp_iter, oamp_damping, oampnet_params, Q);

                err = err + count_qpsk_bit_errors(x_det(data_idx), x(data_idx));
                bits = bits + 2 * numel(data_idx);

                x_hard = qpsk_hard_projection(x_det);
                resid = y - Heff * x_hard;
                proxy_acc = proxy_acc + mean(abs(resid) .^ 2) / max(noise_var, 1e-12);
            end

            ber_vec(a) = err / bits;
            proxy_vec(a) = proxy_acc / num_noise;
        end

        proxy_norm = proxy_vec / max(median(proxy_vec), 1e-12);
        reward_ber_vec = -ber_vec;
        reward_proxy_vec = -proxy_norm;
        reward_mix_vec = -(ber_vec + reward_lambda_proxy * proxy_norm);

        [~, a_star_ber] = min(ber_vec);
        [~, a_star_reward] = max(reward_mix_vec);

        state(sample_idx, :) = single(extract_state_features(Heff_base, ch, snr_pick, data_idx));
        reward(sample_idx, :) = single(reward_mix_vec(:).');
        reward_ber(sample_idx, :) = single(reward_ber_vec(:).');
        reward_proxy(sample_idx, :) = single(reward_proxy_vec(:).');
        reward_mix(sample_idx, :) = single(reward_mix_vec(:).');
        ber_actions(sample_idx, :) = single(ber_vec(:).');
        mse_proxy_actions(sample_idx, :) = single(proxy_norm(:).');

        oracle_action(sample_idx) = int32(a_star_ber);            % MATLAB 1-based
        oracle_action_reward(sample_idx) = int32(a_star_reward);  % MATLAB 1-based
        snr_db(sample_idx) = single(snr_pick);
        sequence_id(sample_idx) = int32(sid);
        time_index(sample_idx) = int32(tt);

        if mod(sample_idx, 100) == 0
            fprintf("progress %d/%d\n", sample_idx, num_samples);
        end
    end
end

dataset_mode = int32(use_timevary); % 1: time-vary sequence, 0: static i.i.d.

save(output_file, ...
    'state', 'reward', 'reward_ber', 'reward_proxy', 'reward_mix', ...
    'ber_actions', 'mse_proxy_actions', ...
    'oracle_action', 'oracle_action_reward', ...
    'snr_db', 'sequence_id', 'time_index', ...
    'c1_grid', 'c1_base', 'base_action', ...
    'dataset_mode', 'num_seq', 'num_frames', ...
    'rho_alpha', 'rho_h', 'num_noise', 'reward_lambda_proxy', ...
    'detector_target', 'oamp_iter', 'oamp_damping', ...
    'oampnet_param_version', 'oampnet_param_path', ...
    '-v7.3');

fprintf("\nSaved offline dataset to %s\n", output_file);
fprintf("mode=%d (1=timevary), samples=%d, num_actions=%d\n", dataset_mode, num_samples, num_actions);
fprintf("detector_target=%s\n", detector_target);
if strcmpi(detector_target, 'oampnet')
    fprintf("oampnet_params=%s\n", oampnet_param_path);
end

%% ===== helper functions =====
function x_det = run_target_detector(y, H, noise_var, detector_target, oamp_iter, oamp_damping, oampnet_params, Q)
    if strcmpi(detector_target, 'lmmse')
        x_det = lmmse_detector(y, H, noise_var);
    elseif strcmpi(detector_target, 'oamp')
        x_det = oamp_detector(y, H, noise_var, oamp_iter, oamp_damping, Q);
    elseif strcmpi(detector_target, 'oampnet')
        if isempty(oampnet_params)
            error('oampnet_params is empty while detector_target=oampnet');
        end
        x_det = oampnet_detector(y, H, noise_var, oampnet_params, Q);
    else
        error('Unsupported detector_target=%s', detector_target);
    end
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
    h = h / norm(h);

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

    d2 = abs(x - const.') .^ 2;
    [~, idx] = min(d2, [], 2);
    bits2 = labels(idx, :);
    bits = reshape(bits2.', [], 1);
end

function x_hard = qpsk_hard_projection(x)
    re = sign(real(x));
    im = sign(imag(x));
    re(re == 0) = 1;
    im(im == 0) = 1;
    x_hard = (re + 1j * im) / sqrt(2);
end
