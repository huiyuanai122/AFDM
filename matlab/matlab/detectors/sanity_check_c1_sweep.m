%% sanity_check_c1_sweep.m
clear; clc; close all;

% ===== 1) 系统/信道参数（和你现有工程保持一致）=====
N = 256; Delta_f = 4; T = 1/Delta_f; B = N*Delta_f; dt = 1/B;
fc = 12e3; alpha_max = 1e-4; ell_max = 16; P = 6;
Q = 0;  N_eff = N - 2*Q;  data_idx = (Q+1):(N-Q);

Lcpp = max(1, ceil( ell_max/(1-alpha_max) ));
Lcps = max(1, ceil( alpha_max*N/(1+alpha_max) ));

% 论文式基准 C1（你现有脚本的做法）
Nv = 2;
kmax = ceil((alpha_max*fc) * T);
den = (1 - 4*alpha_max*(N-1));
c1_base = (2*kmax + 2*alpha_max*(N-1) + 2*Nv + 1) / (2*N*den);
c2 = sqrt(2)/N;

% 候选集合（离散扫）
ratios = linspace(0.6, 1.4, 9);
c1_grid = c1_base * ratios;

% ===== 2) 仿真设置 =====
snr_db_list = [10 14 18];          % 先少取几个点就够判断“值不值得学”
num_ch = 200;                      % 信道 realization 数
num_noise = 3;                     % 每个 (ch, c1) 叠加几次噪声平均，减小方差
use_oamp = true;                   % 想快就 false

% 统计量
ber_lmmse = zeros(numel(c1_grid), numel(snr_db_list));
ber_oamp  = zeros(numel(c1_grid), numel(snr_db_list));
best_idx_per_ch = zeros(num_ch, numel(snr_db_list));

rng(1);

for ch_k = 1:num_ch
    % ----- 固定一个“物理信道” -----
    ch = gen_channel_paper_aligned(P, ell_max, alpha_max);

    % G 只由物理信道决定（同一信道下扫 C1：G 不变）
    % 这里 L = N + Lcpp + Lcps（与 CPP/CPS 对齐）
    % 注意：add_cpp_cps_matrix 会决定 Xext 的长度，所以 L 由它返回
    % 我们这里先构造一次 Xext 来确定 L（用 c1_base 只是为了长度）
    XTB0  = precompute_idaf_basis(N, c1_base, c2);
    Xext0 = add_cpp_cps_matrix(XTB0, c1_base, Lcpp, Lcps);
    L = size(Xext0, 1);

    G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);

    % 固定发送符号（同一信道下扫 C1，x 也固定，比较更公平）
    x = zeros(N,1);
    x(data_idx) = qpsk_symbols(numel(data_idx));
    x = x * sqrt(N / N_eff);

    for m = 1:numel(c1_grid)
        c1 = c1_grid(m);

        % ----- 由 (G, c1) 生成等效 Heff -----
        XTB  = precompute_idaf_basis(N, c1, c2);
        Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);

        YT   = G * Xext; % [N x N]

        n = (0:N-1).';
        chirp1 = exp(-1j*2*pi*c1*(n.^2));
        chirp2 = exp(-1j*2*pi*c2*(n.^2));
        Heff = afdm_demod_matrix(YT, chirp1, chirp2);

        % 与数据集生成一致的归一化（避免不同 c1 仅仅是能量缩放差异）
        Hsub = Heff(data_idx, data_idx);
        sub_power = (norm(Hsub, 'fro')^2) / N_eff;
        Heff = Heff / sqrt(max(sub_power, 1e-12));

        for si = 1:numel(snr_db_list)
            snr_db = snr_db_list(si);
            noise_var = 1 / (10^(snr_db/10));

            err_l = 0; err_o = 0; bits = 0;

            for ni = 1:num_noise
                rng(100000*ch_k + 1000*m + 10*si + ni); % 固定噪声种子，保证可复现
                w = sqrt(noise_var/2) * (randn(N,1) + 1j*randn(N,1));
                y = Heff*x + w;

                x_l = lmmse_detector(y, Heff, noise_var);
                err_l = err_l + count_qpsk_bit_errors(x_l(data_idx), x(data_idx));

                if use_oamp
                    x_o = oamp_detector(y, Heff, noise_var, 10, 0.9, Q);
                    err_o = err_o + count_qpsk_bit_errors(x_o(data_idx), x(data_idx));
                end

                bits = bits + 2*numel(data_idx);
            end

            ber_lmmse(m, si) = ber_lmmse(m, si) + err_l / bits;
            if use_oamp
                ber_oamp(m, si)  = ber_oamp(m, si)  + err_o / bits;
            end
        end
    end

    % 每个信道：记录各 SNR 下的最优 c1 索引（用 LMMSE 或 OAMP 都可）
    for si = 1:numel(snr_db_list)
        [~, best_idx_per_ch(ch_k, si)] = min(ber_lmmse(:, si)); % 也可以换成 ber_oamp
    end

    if mod(ch_k, 20)==0
        fprintf("progress %d/%d\n", ch_k, num_ch);
    end
end

% 求平均
ber_lmmse = ber_lmmse / num_ch;
ber_oamp  = ber_oamp  / num_ch;

% ===== 画图：BER vs C1 =====
figure; hold on; grid on;
for si = 1:numel(snr_db_list)
    semilogy(c1_grid, ber_lmmse(:,si), '-o', 'DisplayName', sprintf('LMMSE %ddB', snr_db_list(si)));
end
xlabel('C1'); ylabel('BER'); title('Sanity check: does C1 matter?'); legend('Location','best');

% ===== 输出“值不值得学”的几个关键统计 =====
% baseline 索引（最接近 c1_base 的那个点）
[~, base_idx] = min(abs(c1_grid - c1_base));

for si = 1:numel(snr_db_list)
    ber_base = ber_lmmse(base_idx, si);
    ber_best = min(ber_lmmse(:, si));
    rel_gain = (ber_base - ber_best) / max(ber_base, 1e-12);

    p_not_base = mean(best_idx_per_ch(:, si) ~= base_idx);

    fprintf("\nSNR=%ddB: BER(base)=%.3e, BER(best)=%.3e, rel_gain=%.2f%%, P(best!=base)=%.2f%%\n", ...
        snr_db_list(si), ber_base, ber_best, 100*rel_gain, 100*p_not_base);
end

%% ====== 下面这些函数，直接从你的数据集生成脚本里拷过来即可 ======
% gen_channel_paper_aligned
% precompute_idaf_basis
% add_cpp_cps_matrix
% build_timescaling_G_sparse
% afdm_demod_matrix
% qpsk_symbols
% count_qpsk_bit_errors
function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
    % delay: integer samples, first path referenced to 0
    ell = sort(randi([0, ell_max], P, 1));
    ell = ell - ell(1);

    % time-scaling: continuous uniform
    alpha = (2*rand(P,1)-1) * alpha_max;

    % exponential power delay profile
    ell_rms = max(1, ell_max/3);
    pwr = exp(-ell/ell_rms);

    h = (randn(P,1) + 1j*randn(P,1))/sqrt(2);
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
    phase_n = exp(1j*2*pi*c1*(n.^2));
    phase_m = exp(1j*2*pi*c2*(m.^2));
    W = exp(1j*2*pi*(n*m)/N) / sqrt(N);
    XTB = (phase_n .* W) .* phase_m; % implicit expansion
end

function Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps)
    % XTB: [N x N], each column is xT when x=e_m
    N = size(XTB, 1);
    if Lcpp==0 && Lcps==0
        Xext = XTB;
        return;
    end

    if Lcpp > 0
        n_pre = (-Lcpp:-1).';
        idx_pre = n_pre + N + 1; % MATLAB 1-based
        phase_pre = exp(-1j*2*pi*c1*(N^2 + 2*N*n_pre));
        Xpre = XTB(idx_pre, :) .* phase_pre;
    else
        Xpre = zeros(0, N, 'like', XTB);
    end

    if Lcps > 0
        n_suf = (N:(N+Lcps-1)).';
        idx_suf = (n_suf - N + 1);
        phase_suf = exp(+1j*2*pi*c1*(N^2 + 2*N*n_suf));
        Xsuf = XTB(idx_suf, :) .* phase_suf;
    else
        Xsuf = zeros(0, N, 'like', XTB);
    end

    Xext = [Xpre; XTB; Xsuf];
end

function G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt)
    % Build sparse linear operator: yT[n] = sum_k G(n,k) * x_ext[k]
    % n=0..N-1, k=0..L-1, where k=0 corresponds to -Lcpp.
    P = ch.P;
    ell = ch.ell;
    alpha = ch.alpha;
    h = ch.h;

    n = (0:N-1).';

    % allocate upper bound nnz
    max_nnz = N * P * 2;
    I = zeros(max_nnz, 1);
    J = zeros(max_nnz, 1);
    V = zeros(max_nnz, 1) + 1j*zeros(max_nnz, 1);
    ptr = 0;

    for i = 1:P
        idx = (1 + alpha(i)) * n - ell(i) + Lcpp; % local continuous index
        idx0 = floor(idx);
        frac = idx - idx0;
        idx1 = idx0 + 1;

        phase = exp(1j*2*pi*(alpha(i)*fc) * (n*dt));

        % k0 contribution
        v0 = (idx0 >= 0) & (idx0 <= (L-1));
        nn0 = sum(v0);
        if nn0 > 0
            rows = find(v0);
            cols = idx0(v0) + 1; % to 1-based
            vals = h(i) * phase(v0) .* (1 - frac(v0));
            I(ptr+1:ptr+nn0) = rows;
            J(ptr+1:ptr+nn0) = cols;
            V(ptr+1:ptr+nn0) = vals;
            ptr = ptr + nn0;
        end

        % k1 contribution
        v1 = (idx1 >= 0) & (idx1 <= (L-1)) & (frac > 0);
        nn1 = sum(v1);
        if nn1 > 0
            rows = find(v1);
            cols = idx1(v1) + 1;
            vals = h(i) * phase(v1) .* frac(v1);
            I(ptr+1:ptr+nn1) = rows;
            J(ptr+1:ptr+nn1) = cols;
            V(ptr+1:ptr+nn1) = vals;
            ptr = ptr + nn1;
        end
    end

    I = I(1:ptr);
    J = J(1:ptr);
    V = V(1:ptr);

    G = sparse(I, J, V, N, L);
end

function Heff = afdm_demod_matrix(YT, chirp1, chirp2)
    % Vectorized DAFT demod for matrix input (each column is one waveform)
    N = size(YT, 1);
    tmp = fft(YT .* chirp1, [], 1) / sqrt(N);
    Heff = tmp .* chirp2;
end

function s = qpsk_symbols(M)
    b1 = randi([0,1], M, 1);
    b2 = randi([0,1], M, 1);
    re = 1 - 2*b1;
    im = 1 - 2*b2;
    s = (re + 1j*im)/sqrt(2);
end
function bit_err = count_qpsk_bit_errors(x_hat, x_true)
%COUNT_QPSK_BIT_ERRORS  Count bit errors for QPSK (Gray mapping) using nearest-neighbor decision.
%   x_hat : estimated QPSK symbols (complex vector)
%   x_true: ground-truth QPSK symbols (complex vector)

    b_hat  = qpsk_demod_bits_gray(x_hat);
    b_true = qpsk_demod_bits_gray(x_true);

    bit_err = sum(b_hat ~= b_true);
end

function bits = qpsk_demod_bits_gray(x)
%QPSK_DEMOD_BITS_GRAY  Map complex symbols to bits using nearest constellation point.
% Gray mapping assumed:
%   ( +1 + j )/sqrt(2) -> 00
%   ( -1 + j )/sqrt(2) -> 01
%   ( -1 - j )/sqrt(2) -> 11
%   ( +1 - j )/sqrt(2) -> 10

    x = x(:);

    % --- amplitude normalization (robust to scaling) ---
    amp = median(abs(x));
    if amp < 1e-12
        amp = 1;
    end
    x = x / amp;

    const = [ 1+1j; -1+1j; -1-1j; 1-1j ] / sqrt(2);
    labels = [0 0;
              0 1;
              1 1;
              1 0];   % Gray labels corresponding to const order above

    % nearest-neighbor decision
    d2 = abs(x - const.').^2;       % size: [len(x) x 4]
    [~, idx] = min(d2, [], 2);      % idx in 1..4

    bits2 = labels(idx, :);         % [len x 2]
    bits = reshape(bits2.', [], 1); % stack as [b0;b1;b0;b1;...]
end