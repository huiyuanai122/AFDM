%% sanity_check_c1_sweep_refine.m
% Refined static-channel C1 sweep:
% - denser C1 grid
% - per-channel BER matrix
% - mean/std/SE/95% CI
% - best-index histogram

clear; clc; close all;

% ===== 1) System/channel parameters =====
N = 256;
Delta_f = 4;
T = 1 / Delta_f;
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
kmax = ceil((alpha_max * fc) * T);
den = (1 - 4 * alpha_max * (N - 1));
c1_base = (2 * kmax + 2 * alpha_max * (N - 1) + 2 * Nv + 1) / (2 * N * den);
c2 = sqrt(2) / N;

% Dense grid: 31 points in [0.6, 1.4] * c1_base
ratios = linspace(0.6, 1.4, 31);
c1_grid = c1_base * ratios;
num_c1 = numel(c1_grid);

% ===== 2) Simulation settings =====
snr_db_list = [10 14 18];
num_snr = numel(snr_db_list);
num_ch = 200;
num_noise = 3;
use_oamp = false;
oamp_iter = 10;
oamp_damping = 0.9;

rng(1);

% Per-channel BER tensor:
% [num_ch, num_c1, num_snr]
ber_lmmse_ch = zeros(num_ch, num_c1, num_snr);
if use_oamp
    ber_oamp_ch = zeros(num_ch, num_c1, num_snr);
else
    ber_oamp_ch = [];
end

best_idx_per_ch = zeros(num_ch, num_snr);

for ch_k = 1:num_ch
    ch = gen_channel_paper_aligned(P, ell_max, alpha_max);

    XTB0 = precompute_idaf_basis(N, c1_base, c2);
    Xext0 = add_cpp_cps_matrix(XTB0, c1_base, Lcpp, Lcps);
    L = size(Xext0, 1);
    G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);

    x = zeros(N, 1);
    x(data_idx) = qpsk_symbols(numel(data_idx));
    x = x * sqrt(N / N_eff);

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

        for si = 1:num_snr
            snr_db = snr_db_list(si);
            noise_var = 1 / (10 ^ (snr_db / 10));

            err_l = 0;
            err_o = 0;
            bits = 0;

            for ni = 1:num_noise
                rng(100000 * ch_k + 1000 * m + 10 * si + ni);
                w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                y = Heff * x + w;

                x_l = lmmse_detector(y, Heff, noise_var);
                err_l = err_l + count_qpsk_bit_errors(x_l(data_idx), x(data_idx));

                if use_oamp
                    x_o = oamp_detector(y, Heff, noise_var, oamp_iter, oamp_damping, Q);
                    err_o = err_o + count_qpsk_bit_errors(x_o(data_idx), x(data_idx));
                end

                bits = bits + 2 * numel(data_idx);
            end

            ber_lmmse_ch(ch_k, m, si) = err_l / bits;
            if use_oamp
                ber_oamp_ch(ch_k, m, si) = err_o / bits;
            end
        end
    end

    for si = 1:num_snr
        [~, best_idx_per_ch(ch_k, si)] = min(squeeze(ber_lmmse_ch(ch_k, :, si)));
    end

    if mod(ch_k, 20) == 0
        fprintf("progress %d/%d\n", ch_k, num_ch);
    end
end

% ===== 3) Aggregate statistics =====
ber_mean = squeeze(mean(ber_lmmse_ch, 1));   % [num_c1, num_snr]
ber_std = squeeze(std(ber_lmmse_ch, 0, 1));  % [num_c1, num_snr]
ber_se = ber_std / sqrt(num_ch);
ber_ci95 = 1.96 * ber_se;

[~, base_idx] = min(abs(c1_grid - c1_base));

fprintf("\n===== Summary (LMMSE) =====\n");
for si = 1:num_snr
    ber_base = ber_mean(base_idx, si);
    ber_best = min(ber_mean(:, si));
    rel_gain = (ber_base - ber_best) / max(ber_base, 1e-12);
    p_not_base = mean(best_idx_per_ch(:, si) ~= base_idx);

    fprintf("SNR=%ddB: BER(base)=%.3e, BER(best)=%.3e, rel_gain=%.2f%%, P(best!=base)=%.2f%%\n", ...
        snr_db_list(si), ber_base, ber_best, 100 * rel_gain, 100 * p_not_base);
end

% ===== 4) Plot BER-C1 with CI =====
figure('Name', 'Refined C1 Sweep with 95% CI');
tiledlayout(1, 1);
nexttile; hold on; grid on;

for si = 1:num_snr
    errorbar(c1_grid, ber_mean(:, si), ber_ci95(:, si), '-o', ...
        'LineWidth', 1.2, ...
        'DisplayName', sprintf('LMMSE %ddB', snr_db_list(si)));
end

xline(c1_grid(base_idx), '--k', 'DisplayName', 'base C1');
xlabel('C1');
ylabel('BER');
set(gca, 'YScale', 'log');
title('BER vs C1 (mean \pm 95% CI)');
legend('Location', 'best');

% ===== 5) Plot best-index histogram =====
figure('Name', 'Best C1 Index Distribution');
tiledlayout(num_snr, 1);
for si = 1:num_snr
    nexttile;
    histogram(best_idx_per_ch(:, si), 1:(num_c1 + 1), 'Normalization', 'probability');
    grid on;
    xlabel('best index on C1 grid');
    ylabel('probability');
    title(sprintf('SNR=%ddB, base_idx=%d', snr_db_list(si), base_idx));
end

% ===== 6) Save result =====
result = struct();
result.c1_base = c1_base;
result.c1_grid = c1_grid;
result.base_idx = base_idx;
result.snr_db_list = snr_db_list;
result.num_ch = num_ch;
result.num_noise = num_noise;
result.ber_lmmse_ch = ber_lmmse_ch;
if use_oamp
    result.ber_oamp_ch = ber_oamp_ch;
end
result.ber_mean = ber_mean;
result.ber_std = ber_std;
result.ber_se = ber_se;
result.ber_ci95 = ber_ci95;
result.best_idx_per_ch = best_idx_per_ch;

save('sanity_check_c1_sweep_refine_result.mat', 'result', '-v7.3');
fprintf("\nSaved result to sanity_check_c1_sweep_refine_result.mat\n");

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
