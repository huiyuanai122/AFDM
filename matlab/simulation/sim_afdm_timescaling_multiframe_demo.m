%% sim_afdm_timescaling_multiframe_demo.m
% ============================================================
% 连续多帧 AFDM over Wideband Doubly-Dispersive Channel with Time-Scaling
% 对比：正常 CPP/CPS vs 置0（无保护） => 观察跨帧干扰(IBI/ISI)
%
% 发射：DAF x -> IDAF -> 每帧加 CPP/CPS -> 多帧拼接成连续流
% 信道：时间缩放 + 时延 + (可选)载频多普勒相位
% 接收：截取中间帧有效窗 -> DAFT -> 用“单帧等效 Heff”做检测
%
% 关键：Heff 按“单帧隔离模型”构建；无 CPP/CPS 时真实 y 包含邻帧干扰
%       因而出现 BER 明显恶化 / error floor，符合论文中 CPP/CPS 的作用机理。
% ============================================================
clear; clc; close all;

%% ------------------- 物理/系统参数 -------------------
N       = 256;            % 每帧 DAF 域符号数（=一帧包含多个符号）
Delta_f = 4;              % Hz
T       = 1/Delta_f;      % s
B       = N*Delta_f;      % Hz
dt      = 1/B;            % s

fc        = 12e3;         % Hz
alpha_max = 1e-4;         % 时间尺度因子上界（你也可增大到5e-4/1e-3更明显）
ell_max   = 48;           % 最大离散时延（samples）
P         = 6;            % 多径数

% 连续多帧：为了避免边界效应，发 7 帧，检测中间第 4 帧
numBlocks   = 7;
targetBlock = 4; % 1..numBlocks

% BER 统计
SNR_dB    = 0:2:20;
num_mc    = 200;    % Monte Carlo 次数（可增大）
Q_guard   = 0;     % 可选：不统计边缘子载（如设4/8）

% ---- 按论文条件估计 CPP/CPS 长度（离散化） ----
Lcpp_nom = max(1, ceil( (ell_max) / (1 - alpha_max) ));
Lcps_nom = max(1, ceil( alpha_max*N / (1 + alpha_max) ));

% ---- 按论文近似式设计 c1，c2 取无理数/N ----
Nv   = 2;
kmax = ceil((alpha_max*fc) * T); % nu_max/Delta_f, nu_max=alpha_max*fc
den  = (1 - 4*alpha_max*(N-1));
if den <= 0
    error("c1 设计无效：1-4*alpha_max*(N-1)<=0，减小 N 或 alpha_max。");
end
c1 = (2*kmax + 2*alpha_max*(N-1) + 2*Nv + 1) / (2*N*den);
c2 = sqrt(2)/N;

fprintf("N=%d, dt=%.3eus, T=%.3fs, B=%.1fHz\n", N, dt*1e6, T, B);
fprintf("alpha_max=%.1e, ell_max=%d (tau_max=%.3fms)\n", alpha_max, ell_max, ell_max*dt*1e3);
fprintf("Lcpp_nom=%d, Lcps_nom=%d, c1=%.6g, c2=%.6g\n", Lcpp_nom, Lcps_nom, c1, c2);

%% ------------------- 预计算 IDAF 基（加速） -------------------
XTB = precompute_idaf_basis(N, c1, c2); % [N x N]，列m是x=e_m时的xT

%% ------------------- 两种 case：有/无 CPP/CPS -------------------
cases(1).name = "CPP/CPS 按论文取值";
cases(1).Lcpp = Lcpp_nom;
cases(1).Lcps = Lcps_nom;

cases(2).name = "CPP/CPS=0（无保护）";
cases(2).Lcpp = 0;
cases(2).Lcps = 0;

ber = zeros(numel(cases), numel(SNR_dB));
err_cnt = zeros(numel(cases), numel(SNR_dB));
bit_cnt = zeros(numel(cases), numel(SNR_dB));

rng(1);

%% ------------------- Monte Carlo：每次随机信道 + 随机多帧数据 -------------------
for mc = 1:num_mc
    % 生成信道（延迟从0为参考）
    ch = gen_channel_paper_aligned(P, ell_max, alpha_max);
    ch.N  = N; ch.dt = dt; ch.fc = fc; ch.c1 = c1; ch.c2 = c2;

    % 为每个 block 生成 DAF 域符号（QPSK）
    x_blk = cell(numBlocks,1);
    data_idx = (1+Q_guard):(N-Q_guard);
    for b = 1:numBlocks
        x = zeros(N,1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x_blk{b} = x;
    end

    for ci = 1:numel(cases)
        Lcpp = cases(ci).Lcpp;
        Lcps = cases(ci).Lcps;

        % --- 1) 构建单帧等效 Heff（隔离模型） ---
        ch_case = ch;
        ch_case.Lcpp = Lcpp;
        ch_case.Lcps = Lcps;

        frame_len = N + Lcpp + Lcps;                % 每帧在stream里的长度
        s0 = (targetBlock-1) * frame_len;           % 目标帧起点(0-based)
        Heff = build_heff_probe_for_block(ch_case, XTB, s0);   % ✅新函数
        [y_daf_target_noiseless, x_target] = ...
        simulate_stream_get_target_daf(ch_case, XTB, x_blk, numBlocks, targetBlock);




        % --- 3) 加噪 + 检测 + 统计 BER ---
        for s = 1:numel(SNR_dB)
            snr_lin   = 10^(SNR_dB(s)/10);
            noise_var = 1/snr_lin; % Es=1（单位功率QPSK）

            w = sqrt(noise_var/2) * (randn(N,1) + 1j*randn(N,1));
            y = y_daf_target_noiseless + w;

            x_hat = lmmse_detector(y, Heff, noise_var);

            err = count_qpsk_bit_errors(x_hat(data_idx), x_target(data_idx));
            err_cnt(ci,s) = err_cnt(ci,s) + err;
            bit_cnt(ci,s) = bit_cnt(ci,s) + 2*numel(data_idx);
        end
    end
end

for ci = 1:numel(cases)
    ber(ci,:) = err_cnt(ci,:) ./ max(1, bit_cnt(ci,:));
    fprintf("\n[%s]\n", cases(ci).name);
    for s = 1:numel(SNR_dB)
        fprintf("SNR=%2ddB, BER=%.3e\n", SNR_dB(s), ber(ci,s));
    end
end

figure; clf; hold on; box on;

% 先画曲线（用 semilogy）
for ci = 1:numel(cases)
    ber_plot = ber(ci,:);
    % 0 无法画在对数轴上，用统计下界替代
    z = (ber_plot==0);
    ber_plot(z) = 0.5 ./ bit_cnt(ci, z);

    semilogy(SNR_dB, ber_plot, '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', cases(ci).name);
end

% 强制坐标轴为“通信原理”半对数风格
ax = gca;
ax.YScale = 'log';
ax.XLim = [min(SNR_dB) max(SNR_dB)];
ax.YLim = [1e-5 1];

% 强制主刻度为 10^n，并显示成 10^{-k} 形式（像你右图）
exps = -5:0;
ax.YTick = 10.^exps;
ax.YTickLabel = arrayfun(@(e) sprintf('10^{%d}', e), exps, 'UniformOutput', false);
ax.TickLabelInterpreter = 'tex';

grid on;
ax.YMinorGrid = 'on';
ax.XMinorGrid = 'on';

xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR (Semi-log) — Continuous multi-frame AFDM with time-scaling');
legend('Location','southwest');


%% ============================================================
%% Local functions
%% ============================================================

function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
    ell = sort(randi([0, ell_max], P, 1));
    ell = ell - ell(1);                 % 第一条径作为0参考
    alpha = (2*rand(P,1)-1) * alpha_max;

    ell_rms = max(1, ell_max/3);
    pwr = exp(-ell/ell_rms);

    h = (randn(P,1) + 1j*randn(P,1))/sqrt(2);
    h = h .* sqrt(pwr);
    h = h / norm(h);

    ch.P     = P;
    ch.ell   = ell;
    ch.alpha = alpha;
    ch.h     = h;
end

function XTB = precompute_idaf_basis(N, c1, c2)
    n = (0:N-1).';
    m = 0:N-1;
    phase_n = exp(1j*2*pi*c1*(n.^2));   % [N x 1]
    phase_m = exp(1j*2*pi*c2*(m.^2));   % [1 x N]
    W = exp(1j*2*pi*(n*m)/N) / sqrt(N); % [N x N]
    XTB = (phase_n .* W) .* phase_m;    % implicit expansion
end

function xT = afdm_mod_fast(XTB, x)
    xT = XTB * x;
end

function x = afdm_demod(yT, c1, c2)
    N = length(yT);
    n = (0:N-1).'; m = n;
    tmp = (1/sqrt(N)) * fft(yT .* exp(-1j*2*pi*c1*(n.^2)));
    x   = tmp .* exp(-1j*2*pi*c2*(m.^2));
end

function x_ext = add_cpp_cps(xT, c1, Lcpp, Lcps)
    N = length(xT);
    if Lcpp==0 && Lcps==0
        x_ext = xT;
        return;
    end

    if Lcpp > 0
        n_pre = (-Lcpp:-1).';
        x_pre = xT(n_pre + N + 1) .* exp(-1j*2*pi*c1*(N^2 + 2*N*n_pre));
    else
        x_pre = [];
    end

    if Lcps > 0
        n_suf = (N:(N+Lcps-1)).';
        x_suf = xT(n_suf - N + 1) .* exp(+1j*2*pi*c1*(N^2 + 2*N*n_suf));
    else
        x_suf = [];
    end

    x_ext = [x_pre; xT; x_suf];
end

function Heff = build_heff_probe_isolated(ch, XTB)
% 单帧隔离模型：只发一帧（带/不带CPP/CPS），帧外视为0
% 返回：DAF域 Heff，使 y = Heff*x（无噪声）
    N = ch.N; c1 = ch.c1; c2 = ch.c2; %#ok<NASGU>
    Lcpp = ch.Lcpp; Lcps = ch.Lcps;

    Heff = zeros(N,N);
    for m = 1:N
        xT = XTB(:,m);
        x_ext = add_cpp_cps(xT, ch.c1, Lcpp, Lcps);
        yT = timescaling_channel_isolated_window(x_ext, ch); % length N
        y  = afdm_demod(yT, ch.c1, ch.c2);
        Heff(:,m) = y;
    end
end

function yT = timescaling_channel_isolated_window(x_ext, ch)
% 单帧隔离：x_ext 对应采样索引 0..L-1，其中 0 对应 -Lcpp
% 观测窗口取有效部分 n=0..N-1（对应原帧的 xT[0..N-1]）
    N = ch.N;
    P = ch.P; ell = ch.ell; alpha = ch.alpha; h = ch.h;
    fc = ch.fc; dt = ch.dt; Lcpp = ch.Lcpp;

    L = length(x_ext);
    grid = (0:L-1).';
    n = (0:N-1).';

    yT = zeros(N,1);
    for i = 1:P
        idx = (1 + alpha(i)) * n - ell(i) + Lcpp; % +Lcpp把“0点”移到有效窗起点
        xi = interp1(grid, x_ext, idx, 'linear', 0);
        phase = exp(1j*2*pi*(alpha(i)*fc) * (n*dt));
        yT = yT + h(i) * (xi .* phase);
    end
end

function [y_daf_target, x_target] = simulate_stream_get_target_daf(ch, XTB, x_blk, numBlocks, targetBlock)
% 连续流：把每帧（带/不带前后缀）拼成 stream，通过整段信道，然后截取 targetBlock 的有效窗
    N = ch.N;
    Lcpp = ch.Lcpp; Lcps = ch.Lcps;

    % ---- 1) 组装 TX 连续流 ----
    tx_stream = [];
    start_of_block = zeros(numBlocks,1); % 每个block在stream中的起始索引（0-based）
    cur = 0;
    for b = 1:numBlocks
        start_of_block(b) = cur;
        xT = afdm_mod_fast(XTB, x_blk{b});
        x_ext = add_cpp_cps(xT, ch.c1, Lcpp, Lcps);
        tx_stream = [tx_stream; x_ext]; %#ok<AGROW>
        cur = cur + length(x_ext);
    end

    % ---- 2) 整段流过时间尺度信道 ----
    y_stream = timescaling_channel_stream(tx_stream, ch);

    % ---- 3) 截取目标帧有效窗（假设已完成帧同步：以最早径为参考）----
    s0 = start_of_block(targetBlock);     % 0-based
    main0 = s0 + Lcpp;                    % 有效部分起点
    idx = (main0 + (0:N-1)) + 1;          % 转成MATLAB 1-based索引
    yT_target = y_stream(idx);

    % ---- 4) DAFT 得到 DAF 域接收 ----
    y_daf_target = afdm_demod(yT_target, ch.c1, ch.c2);

    x_target = x_blk{targetBlock};
end

function y_stream = timescaling_channel_stream(tx_stream, ch)
% 对整段连续 tx_stream 做时间尺度 + 时延 + 多普勒相位
    P = ch.P; ell = ch.ell; alpha = ch.alpha; h = ch.h;
    fc = ch.fc; dt = ch.dt;

    L = length(tx_stream);
    grid = (0:L-1).';
    n = (0:L-1).';           % 对应接收采样时刻

    y_stream = zeros(L,1);
    for i = 1:P
        idx = (1 + alpha(i)) * n - ell(i);      % 注意：不再 +Lcpp（stream本身已包含前后缀）
        xi = interp1(grid, tx_stream, idx, 'linear', 0);
        phase = exp(1j*2*pi*(alpha(i)*fc) * (n*dt));
        y_stream = y_stream + h(i) * (xi .* phase);
    end
end

function s = qpsk_symbols(M)
    b1 = randi([0,1], M, 1);
    b2 = randi([0,1], M, 1);
    re = 1 - 2*b1;
    im = 1 - 2*b2;
    s = (re + 1j*im)/sqrt(2);
end

function e = count_qpsk_bit_errors(x_hat, x_true)
    xh = x_hat(:);
    xt = x_true(:);

    b1h = real(xh) < 0;
    b2h = imag(xh) < 0;

    b1t = real(xt) < 0;
    b2t = imag(xt) < 0;

    e = sum(b1h ~= b1t) + sum(b2h ~= b2t);
end

function x_hat = lmmse_detector(y, H, noise_var)
    N = size(H,2);
    x_hat = (H' * H + noise_var * eye(N)) \ (H' * y);
end
function Heff = build_heff_probe_for_block(ch, XTB, s0)
% 在连续流的“目标帧绝对位置 s0”处构建该帧的等效 Heff
% 这样 Heff 与 simulate_stream_get_target_daf 中提取的 y 完全同参考系
    N = ch.N;
    Lcpp = ch.Lcpp; Lcps = ch.Lcps; %#ok<NASGU>
    main0 = s0 + Lcpp;  % 目标帧有效窗起点(绝对索引, 0-based)

    Heff = zeros(N,N);
    for m = 1:N
        xT = XTB(:,m);
        x_ext = add_cpp_cps(xT, ch.c1, Lcpp, ch.Lcps); 
        % 注意：x_ext 的“本地索引0”对应绝对索引 s0
        yT = timescaling_channel_window_at_abspos(x_ext, ch, s0, main0);
        y  = afdm_demod(yT, ch.c1, ch.c2);
        Heff(:,m) = y;
    end
end

function yT = timescaling_channel_window_at_abspos(x_ext, ch, s0, main0)
% 计算接收端在绝对窗口 [main0, main0+N-1] 内的时域样本 yT
% x_ext 是目标帧(含CPP/CPS)的波形，放置在绝对起点 s0 处
    N = ch.N;
    P = ch.P; ell = ch.ell; alpha = ch.alpha; h = ch.h;
    fc = ch.fc; dt = ch.dt;

    L = length(x_ext);
    grid = (0:L-1).';                 % x_ext 的本地索引
    n_abs = (main0 + (0:N-1)).';      % 接收采样的绝对索引(0-based)

    yT = zeros(N,1);
    for i = 1:P
        % 输入采样对应的绝对索引：
        idx_abs = (1 + alpha(i)) * n_abs - ell(i);

        % 映射到 x_ext 的本地索引（因为 x_ext 起点放在绝对 s0）
        idx_local = idx_abs - s0;

        xi = interp1(grid, x_ext, idx_local, 'linear', 0);

        % ✅绝对时间的多普勒相位（与 stream 仿真一致）
        phase = exp(1j*2*pi*(alpha(i)*fc) * (n_abs*dt));

        yT = yT + h(i) * (xi .* phase);
    end
end

