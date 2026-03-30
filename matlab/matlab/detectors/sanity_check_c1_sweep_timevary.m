%% sanity_check_c1_sweep_timevary.m
% Time-varying channel C1 sweep:
% - AR(1) evolution on alpha and h
% - per-frame oracle C1*(t)
% - switch_rate and oracle_gain_timevary

clear; clc; close all;

% ===== 1) System/channel parameters =====
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
num_c1 = numel(c1_grid);
[~, base_idx] = min(abs(c1_grid - c1_base));

% ===== 2) Time-varying experiment settings =====
snr_db = 14;
noise_var = 1 / (10 ^ (snr_db / 10));

num_seq = 60;
num_frames = 50;
num_noise = 2;

rho_alpha = 0.98;
rho_h = 0.98;

rng(2);

% [num_seq, num_frames, num_c1]
ber_seq = zeros(num_seq, num_frames, num_c1);
oracle_idx_seq = zeros(num_seq, num_frames);
policy_lag_idx_seq = zeros(num_seq, num_frames); % baseline: previous oracle action

switch_rate_seq = zeros(num_seq, 1);
oracle_gain_seq = zeros(num_seq, 1);
lag_gap_seq = zeros(num_seq, 1);

for ks = 1:num_seq
    ch0 = gen_channel_paper_aligned(P, ell_max, alpha_max);
    ell = ch0.ell;
    alpha_t = ch0.alpha;
    h_t = ch0.h;

    % Keep a fixed PDP envelope for AR(1) update of h.
    ell_rms = max(1, ell_max / 3);
    pwr = exp(-ell / ell_rms);

    prev_oracle_idx = base_idx;

    for tt = 1:num_frames
        if tt > 1
            alpha_t = rho_alpha * alpha_t + sqrt(1 - rho_alpha^2) * alpha_max * randn(P, 1);
            alpha_t = min(max(alpha_t, -alpha_max), alpha_max);

            innov = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
            h_t = rho_h * h_t + sqrt(1 - rho_h^2) * innov;
            h_t = h_t .* sqrt(pwr);
            h_t = h_t / max(norm(h_t), 1e-12);
        end

        ch_t = struct('P', P, 'ell', ell, 'alpha', alpha_t, 'h', h_t);

        XTB0 = precompute_idaf_basis(N, c1_base, c2);
        Xext0 = add_cpp_cps_matrix(XTB0, c1_base, Lcpp, Lcps);
        L = size(Xext0, 1);
        G = build_timescaling_G_sparse(N, L, Lcpp, ch_t, fc, dt);

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        frame_ber = zeros(num_c1, 1);

        for m = 1:num_c1
            c1 = c1_grid(m);
            XTB = precompute_idaf_basis(N, c1, c2);
            Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);
            YT = G * Xext;

            n = (0:N-1).';
            chirp1 = exp(-1j * 2 * pi * c1 * (n .^ 2));
            chirp2 = exp(-1j * 2 * pi * c2 * (n .^ 2));
            Heff = afdm_demod_matrix(YT, chirp1, chirp2);

            Hsub = Heff(data_idx, data_idx);
            sub_power = (norm(Hsub, 'fro') ^ 2) / N_eff;
            Heff = Heff / sqrt(max(sub_power, 1e-12));

            err = 0;
            bits = 0;
            for ni = 1:num_noise
                rng(1000000 * ks + 10000 * tt + 100 * m + ni);
                w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                y = Heff * x + w;

                x_l = lmmse_detector(y, Heff, noise_var);
                err = err + count_qpsk_bit_errors(x_l(data_idx), x(data_idx));
                bits = bits + 2 * numel(data_idx);
            end
            frame_ber(m) = err / bits;
        end

        ber_seq(ks, tt, :) = frame_ber;
        [~, oracle_idx] = min(frame_ber);
        oracle_idx_seq(ks, tt) = oracle_idx;

        if tt == 1
            policy_lag_idx_seq(ks, tt) = base_idx;
        else
            policy_lag_idx_seq(ks, tt) = prev_oracle_idx;
        end
        prev_oracle_idx = oracle_idx;
    end

    oracle_trace = oracle_idx_seq(ks, :);
    switch_rate_seq(ks) = mean(diff(oracle_trace) ~= 0);

    ber_fixed = mean(squeeze(ber_seq(ks, :, base_idx)));
    ber_oracle = mean(min(squeeze(ber_seq(ks, :, :)), [], 2));
    oracle_gain_seq(ks) = (ber_fixed - ber_oracle) / max(ber_fixed, 1e-12);

    lag_idx = squeeze(policy_lag_idx_seq(ks, :));
    ber_lag = 0;
    for tt = 1:num_frames
        ber_lag = ber_lag + ber_seq(ks, tt, lag_idx(tt));
    end
    ber_lag = ber_lag / num_frames;
    lag_gap_seq(ks) = (ber_lag - ber_oracle) / max(ber_lag, 1e-12);

    if mod(ks, 10) == 0
        fprintf("sequence progress %d/%d\n", ks, num_seq);
    end
end

% ===== 3) Summary =====
switch_mean = mean(switch_rate_seq);
switch_std = std(switch_rate_seq);
gain_mean = mean(oracle_gain_seq);
gain_std = std(oracle_gain_seq);
lag_gap_mean = mean(lag_gap_seq);
lag_gap_std = std(lag_gap_seq);

fprintf("\n===== Time-varying C1 Summary =====\n");
fprintf("SNR = %d dB, num_seq = %d, num_frames = %d\n", snr_db, num_seq, num_frames);
fprintf("switch_rate: mean=%.4f, std=%.4f\n", switch_mean, switch_std);
fprintf("oracle_gain_timevary: mean=%.2f%%, std=%.2f%%\n", 100 * gain_mean, 100 * gain_std);
fprintf("lag_policy_gap: mean=%.2f%%, std=%.2f%%\n", 100 * lag_gap_mean, 100 * lag_gap_std);

% ===== 4) Plot =====
% (a) Oracle C1 index trajectory for first sequence
figure('Name', 'Oracle C1 Trajectory');
stairs(1:num_frames, oracle_idx_seq(1, :), 'LineWidth', 1.3); grid on;
xlabel('frame index');
ylabel('oracle C1 index');
title(sprintf('Oracle C1*(t), sequence #1, switch_rate=%.3f', switch_rate_seq(1)));

% (b) BER fixed vs oracle over time (sequence #1)
figure('Name', 'Fixed vs Oracle BER over Time');
ber_fixed_t = squeeze(ber_seq(1, :, base_idx));
ber_oracle_t = squeeze(min(ber_seq(1, :, :), [], 3));
semilogy(1:num_frames, ber_fixed_t, '-o', 'LineWidth', 1.1, 'DisplayName', 'fixed base');
hold on;
semilogy(1:num_frames, ber_oracle_t, '-s', 'LineWidth', 1.1, 'DisplayName', 'oracle');
grid on;
xlabel('frame index');
ylabel('BER');
title('Fixed C1 vs Oracle C1*(t)');
legend('Location', 'best');

% (c) Distribution of switch_rate across sequences
figure('Name', 'Switch Rate Distribution');
histogram(switch_rate_seq, 15, 'Normalization', 'probability');
grid on;
xlabel('switch_rate');
ylabel('probability');
title('Distribution of switch_rate across sequences');

% ===== 5) Save =====
result = struct();
result.snr_db = snr_db;
result.num_seq = num_seq;
result.num_frames = num_frames;
result.rho_alpha = rho_alpha;
result.rho_h = rho_h;
result.c1_base = c1_base;
result.c1_grid = c1_grid;
result.base_idx = base_idx;
result.ber_seq = ber_seq;
result.oracle_idx_seq = oracle_idx_seq;
result.policy_lag_idx_seq = policy_lag_idx_seq;
result.switch_rate_seq = switch_rate_seq;
result.oracle_gain_seq = oracle_gain_seq;
result.lag_gap_seq = lag_gap_seq;

save('sanity_check_c1_sweep_timevary_result.mat', 'result', '-v7.3');
fprintf("\nSaved result to sanity_check_c1_sweep_timevary_result.mat\n");

%% ===== helper functions =====
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
