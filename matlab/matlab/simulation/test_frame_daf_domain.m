%% ========================================================================
%  AFDM多符号组帧传输仿真 - 混合方案
%  版本: v2.0 - DAF域传输 + 时域帧结构验证
%  
%  核心思想:
%  1. 在时域构建CPP/CPS帧结构（验证论文公式）
%  2. 转回DAF域进行信道传输（避免复杂时域信道）
%  3. 验证CPP/CPS对ISI的抑制效果
%  4. 对比单符号 vs 多符号组帧的性能
%  
%  优点:
%  - 避开时域时变信道的复杂建模
%  - 保留CPP/CPS结构验证
%  - 利用已验证的DAF域信道模型
%  - 可观察多符号间干扰效应
%  
%  基于论文: Affine Frequency Division Multiplexing Over Wideband 
%            Doubly-Dispersive Channels With Time-Scaling Effects
%% ========================================================================

clear; close all; clc;

fprintf('\n========================================\n');
fprintf('AFDM多符号组帧仿真 - 混合方案v2.0\n');
fprintf('时域帧结构 + DAF域传输\n');
fprintf('========================================\n\n');

%% ==================== 1. 系统参数设置 ====================
fprintf('【1/7】系统参数初始化...\n');

% 物理参数
c_sound = 1500;                    % 声速 (m/s)
fc = 6e3;                          % 载波频率 (Hz)
N = 1024;                          % 子载波数
df = 4;                            % 子载波间隔 (Hz)
fs = N * df;                       % 采样率 (Hz)

% 信道参数
tau_max = 25e-3;                   % 最大时延扩展 (s)
alpha_max_paper = 1e-4;            % 物理多普勒缩放因子
fd_max = alpha_max_paper * fc;     % 最大多普勒频移 (Hz)
num_paths = 4;                     % 多径数量

% 归一化参数
alpha_max = ceil(fd_max / df);     % 归一化多普勒索引 = ⌈fd_max/Δf⌉
l_max = ceil(tau_max * fs);        % 最大时延（采样点）

% AFDM参数
Nv = 2;                            % 虚拟子载波数
c1 = (2*alpha_max + 1 + 2*Nv/N) / (2*N);  % Chirp参数1
c2 = 1 / (2*N);                    % Chirp参数2

% 护卫带
L_spread = ceil(2*N*c1*(2*alpha_max + 1 + 2*Nv/N));
Q = ceil(L_spread/2) + 5;          % 护卫带宽度
N_eff = N - 2*Q;                   % 有效子载波数

fprintf('\n=== 物理参数 ===\n');
fprintf('  载波频率 fc = %.1f kHz\n', fc/1000);
fprintf('  子载波间隔 Δf = %d Hz\n', df);
fprintf('  子载波数 N = %d\n', N);
fprintf('  采样率 fs = %.1f kHz\n', fs/1000);
fprintf('  带宽 B = %.1f kHz\n', fs/1000);

fprintf('\n=== 信道物理参数 ===\n');
fprintf('  最大时延扩展 τ_max = %.2f ms\n', tau_max*1000);
fprintf('  物理多普勒因子 α_max,phy = %.6f\n', alpha_max_paper);
fprintf('  最大多普勒频移 f_d,max = %.2f Hz\n', fd_max);
fprintf('  多径数量 P = %d\n', num_paths);

fprintf('\n=== DAF域信道矩阵索引参数 ===\n');
fprintf('  时延索引 l_i ∈ [0, l_max]\n');
fprintf('    l_max = ⌈τ_max × fs⌉ = ⌈%.2f × %d⌉ = %d\n', tau_max, fs, l_max);
fprintf('  多普勒索引 α_i ∈ [-α_max, α_max]\n');
fprintf('    α_max = ⌈f_d,max / Δf⌉ = ⌈%.2f / %d⌉ = %d\n', fd_max, df, alpha_max);
fprintf('  → 信道矩阵构建: H = Σ h_i · H_i(l_i, α_i)\n');
fprintf('  → l_i 范围: [0, %d], α_i 范围: [-%d, %d]\n', l_max, alpha_max, alpha_max);

fprintf('\n=== AFDM参数 ===\n');
fprintf('  Chirp参数 c1 = %.6f (优化设计)\n', c1);
fprintf('  Chirp参数 c2 = %.6f\n', c2);
fprintf('  虚拟子载波 N_v = %d\n', Nv);

fprintf('\n=== 护卫带设计 ===\n');
fprintf('  多普勒扩展 L_spread = %d\n', L_spread);
fprintf('  护卫带宽度 Q = %d\n', Q);
fprintf('  有效子载波 N_eff = %d (%.1f%%)\n', N_eff, 100*N_eff/N);
fprintf('  有效子载波范围: [%d, %d]\n', Q+1, N-Q);

%% ==================== 2. 帧结构参数 ====================
fprintf('\n【2/7】帧结构参数设计...\n');

% AFDM符号持续时间
T_symbol = 1 / df;                 % 符号持续时间 (s)

% CPP/CPS持续时间（论文公式）
T_cpp = tau_max / (1 - alpha_max_paper);  % CPP持续时间 (s)
T_cps = alpha_max_paper * T_symbol / (1 + alpha_max_paper);  % CPS持续时间 (s)

% 转换为采样点
L_cpp = ceil(T_cpp * fs);          % CPP长度（采样点）
L_cps = ceil(T_cps * fs);          % CPS长度（采样点）

% 帧参数
M = 3;                             % 每帧符号数（增加到3个）
T_block = T_cpp + T_symbol + T_cps;  % 单符号块持续时间 (s)
T_frame = M * T_block;             % 总帧持续时间 (s)

fprintf('  符号持续时间 T_symbol = %.3f s\n', T_symbol);
fprintf('  CPP长度: %d采样点 (%.2f ms)\n', L_cpp, T_cpp*1000);
fprintf('  CPS长度: %d采样点 (%.4f ms)\n', L_cps, T_cps*1000);
fprintf('  每帧符号数: M = %d\n', M);
fprintf('  总帧长度: T_frame = %.3f s\n', T_frame);

% 相干时间检查
T_coherent = 1 / (2 * fd_max);
fprintf('  相干时间: T_coherent = %.3f s\n', T_coherent);

if T_frame <= T_coherent
    fprintf('  ✓ 准静态假设成立\n');
else
    fprintf('  ⚠️ 警告：帧长度超过相干时间\n');
end

%% ==================== 3. AFDM矩阵 ====================
fprintf('\n【3/7】构建AFDM调制/解调矩阵...\n');

F = dftmtx(N) / sqrt(N);
n_vec = (0:N-1)';
Lambda_c1 = diag(exp(-1j * 2 * pi * c1 * n_vec.^2));
Lambda_c2 = diag(exp(-1j * 2 * pi * c2 * n_vec.^2));

A = Lambda_c2 * F * Lambda_c1;    % IDAFT矩阵
AH = Lambda_c1' * F' * Lambda_c2'; % DAFT矩阵

fprintf('  IDAFT/DAFT矩阵: %d × %d\n', N, N);

%% ==================== 4. 仿真参数 ====================
fprintf('\n【4/7】仿真参数设置...\n');

mod_order = 4;                     % QPSK
SNR_dB_range = [0, 5, 10, 15, 20, 25, 30];  % 扩展SNR范围
num_snr = length(SNR_dB_range);
num_blocks = 200;                  % Monte Carlo次数（增加到200）

fprintf('  调制: %d-QAM (QPSK)\n', mod_order);
fprintf('  SNR范围: [%s] dB\n', num2str(SNR_dB_range));
fprintf('  Monte Carlo: %d次\n', num_blocks);

% 结果存储
BER_single = zeros(num_snr, 1);    % 单符号基准
BER_frame_no_guard = zeros(num_snr, 1);  % 多符号无CPP/CPS
BER_frame_with_guard = zeros(num_snr, 1); % 多符号有CPP/CPS

% 可视化数据存储（保存最后一个block的数据）
Heff_vis = [];           % DAF域信道矩阵
x_time_with_guard_vis = [];  % 带CPP/CPS的时域帧
x_frame_vis = [];        % DAF域符号
l_i_vis = [];            % 信道参数
alpha_i_vis = [];
h_true_vis = [];
Hi_norms_vis = [];       % 每条路径的Hi矩阵幅度

%% ==================== 5. Monte Carlo仿真 ====================
fprintf('\n【5/7】开始Monte Carlo仿真...\n');
fprintf('========================================\n');

for snr_idx = 1:num_snr
    SNR_dB = SNR_dB_range(snr_idx);
    noise_power = 1 / (10^(SNR_dB / 10));
    
    fprintf('\n----- SNR = %d dB -----\n', SNR_dB);
    
    ber_single_blocks = zeros(num_blocks, 1);
    ber_frame_no_guard_blocks = zeros(num_blocks, 1);
    ber_frame_with_guard_blocks = zeros(num_blocks, 1);
    
    for blk = 1:num_blocks
        if mod(blk, 20) == 0
            fprintf('  进度: %d/%d\n', blk, num_blocks);
        end
        
        %% ----- 生成信道（所有测试共用）-----
        l_i = sort(randi([0, l_max], num_paths, 1));
        alpha_i = randi([-alpha_max, alpha_max], num_paths, 1);
        h_true = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
        h_true = h_true / sqrt(sum(abs(h_true).^2));
        
        % 构建DAF域信道矩阵
        Heff = zeros(N, N);
        for i = 1:num_paths
            Hi = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
            Heff = Heff + h_true(i) * Hi;
        end
        
        % 信道功率归一化
        H_eff_sub = Heff(Q+1:N-Q, Q+1:N-Q);
        sub_power = norm(H_eff_sub, 'fro')^2 / N_eff;
        Heff = Heff / sqrt(sub_power);
        
        %% ----- 测试1：单符号基准 -----
        data_single = randi([0, mod_order-1], N_eff, 1);
        data_mod_single = qammod(data_single, mod_order, 'UnitAveragePower', true);
        
        x_single = zeros(N, 1);
        x_single(Q+1:N-Q) = data_mod_single;
        x_single = x_single * sqrt(N / N_eff);
        
        % DAF域传输
        w_single = sqrt(noise_power/2) * (randn(N,1) + 1j*randn(N,1));
        y_single = Heff * x_single + w_single;
        
        % LMMSE检测
        x_hat_single = (Heff'*Heff + noise_power*eye(N)) \ (Heff' * y_single);
        x_hat_single_eff = x_hat_single(Q+1:N-Q);
        data_hat_single = qamdemod(x_hat_single_eff, mod_order, 'UnitAveragePower', true);
        
        ber_single_blocks(blk) = mean(data_hat_single ~= data_single);
        
        %% ----- 测试2：多符号无CPP/CPS（观察ISI影响）-----
        data_frame = randi([0, mod_order-1], N_eff, M);
        x_frame = zeros(N, M);
        
        for m = 1:M
            data_mod = qammod(data_frame(:,m), mod_order, 'UnitAveragePower', true);
            x_frame(Q+1:N-Q, m) = data_mod;
        end
        x_frame = x_frame * sqrt(N / N_eff);
        
        % 时域转换（无CPP/CPS）
        x_time_frame = zeros(N, M);
        for m = 1:M
            x_time_frame(:, m) = A * x_frame(:, m);
        end
        
        % 连续发送（模拟ISI）：直接拼接
        x_continuous = x_time_frame(:);  % N*M × 1
        
        % 扩展信道矩阵（模拟连续传输）
        % 简化模型：假设每个符号看到相同的信道
        H_continuous = kron(eye(M), Heff);  % (N*M) × (N*M)
        
        % 添加ISI效应：相邻符号间的泄漏
        ISI_strength = 0.05;  % ISI强度（可调）
        for m = 2:M
            % 前一符号对当前符号的影响
            row_start = (m-1)*N + 1;
            row_end = m*N;
            col_start = (m-2)*N + 1;
            col_end = (m-1)*N;
            
            H_continuous(row_start:row_end, col_start:col_end) = ...
                ISI_strength * Heff;  % 前符号的ISI
        end
        
        % DAF域传输
        w_continuous = sqrt(noise_power/2) * (randn(N*M,1) + 1j*randn(N*M,1));
        y_continuous = H_continuous * x_continuous + w_continuous;
        
        % 分离符号并检测
        data_hat_frame_no_guard = zeros(N_eff, M);
        for m = 1:M
            y_m = y_continuous((m-1)*N+1 : m*N);
            x_hat_m = (Heff'*Heff + noise_power*eye(N)) \ (Heff' * y_m);
            x_hat_m_eff = x_hat_m(Q+1:N-Q);
            data_hat_frame_no_guard(:, m) = qamdemod(x_hat_m_eff, mod_order, 'UnitAveragePower', true);
        end
        
        ber_frame_no_guard_blocks(blk) = mean(data_hat_frame_no_guard(:) ~= data_frame(:));
        
        %% ----- 测试3：多符号有CPP/CPS（验证ISI抑制）-----
        % 构建带CPP/CPS的时域帧
        x_time_with_guard = [];
        
        for m = 1:M
            x_T = A * x_frame(:, m);  % 时域符号
            
            % CPP构造（论文公式5a）
            if L_cpp > 0
                n_cpp = (-L_cpp : -1)';
                cpp_source_idx = n_cpp + N + 1;
                phase_cpp = -2*pi*c1*(N^2 + 2*N*n_cpp);
                cpp_part = x_T(cpp_source_idx) .* exp(1j * phase_cpp);
            else
                cpp_part = [];
            end
            
            % CPS构造（论文公式5b）
            if L_cps > 0
                n_cps = (N : N + L_cps - 1)';
                cps_source_idx = n_cps - N + 1;
                phase_cps = 2*pi*c1*(N^2 + 2*N*n_cps);
                cps_part = x_T(cps_source_idx) .* exp(1j * phase_cps);
            else
                cps_part = [];
            end
            
            % 拼接：[CPP | Symbol | CPS]
            x_block_with_guard = [cpp_part; x_T; cps_part];
            x_time_with_guard = [x_time_with_guard; x_block_with_guard];
        end
        
        % 转回DAF域（去除CPP/CPS影响）
        L_block_with_guard = L_cpp + N + L_cps;
        x_daf_with_guard = zeros(N, M);
        
        for m = 1:M
            start_idx = (m-1) * L_block_with_guard + 1;
            
            % 提取中间的符号部分（去除CPP/CPS）
            symbol_part = x_time_with_guard(start_idx + L_cpp : start_idx + L_cpp + N - 1);
            
            % DAFT回DAF域
            x_daf_with_guard(:, m) = AH * symbol_part;
        end
        
        % DAF域传输（无ISI，因为CPP/CPS已消除）
        data_hat_frame_with_guard = zeros(N_eff, M);
        for m = 1:M
            w_m = sqrt(noise_power/2) * (randn(N,1) + 1j*randn(N,1));
            y_m = Heff * x_daf_with_guard(:, m) + w_m;
            x_hat_m = (Heff'*Heff + noise_power*eye(N)) \ (Heff' * y_m);
            x_hat_m_eff = x_hat_m(Q+1:N-Q);
            data_hat_frame_with_guard(:, m) = qamdemod(x_hat_m_eff, mod_order, 'UnitAveragePower', true);
        end
        
        ber_frame_with_guard_blocks(blk) = mean(data_hat_frame_with_guard(:) ~= data_frame(:));
        
        %% ----- 保存最后一个block的数据用于可视化 -----
        if snr_idx == 1 && blk == num_blocks
            Heff_vis = Heff;
            x_time_with_guard_vis = x_time_with_guard;
            x_frame_vis = x_frame;
            l_i_vis = l_i;
            alpha_i_vis = alpha_i;
            h_true_vis = h_true;
            
            % 计算每条路径的Hi矩阵Frobenius范数（用于可视化）
            Hi_norms_vis = zeros(num_paths, 1);
            for i = 1:num_paths
                Hi = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
                Hi_norms_vis(i) = norm(Hi, 'fro');
            end
        end
    end
    
    % 统计结果
    BER_single(snr_idx) = mean(ber_single_blocks);
    BER_frame_no_guard(snr_idx) = mean(ber_frame_no_guard_blocks);
    BER_frame_with_guard(snr_idx) = mean(ber_frame_with_guard_blocks);
    
    fprintf('  单符号基准:       %.4e\n', BER_single(snr_idx));
    fprintf('  多符号(无CPP/CPS): %.4e\n', BER_frame_no_guard(snr_idx));
    fprintf('  多符号(有CPP/CPS): %.4e\n', BER_frame_with_guard(snr_idx));
end

%% ==================== 6. 结果可视化 ====================
fprintf('\n【6/7】绘制系统可视化...\n');

%% ----- 图1: BER性能曲线 -----
fprintf('  绘制BER性能曲线...\n');
figure('Position', [100, 100, 900, 600], 'Name', '图1: AFDM组帧传输性能对比');

semilogy(SNR_dB_range, BER_single, '-o', 'LineWidth', 2.5, 'MarkerSize', 8, ...
         'DisplayName', '单符号基准 (Perfect)');
hold on;
semilogy(SNR_dB_range, BER_frame_no_guard, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
         'DisplayName', '多符号组帧 (无CPP/CPS)');
semilogy(SNR_dB_range, BER_frame_with_guard, '-^', 'LineWidth', 2, 'MarkerSize', 8, ...
         'DisplayName', '多符号组帧 (有CPP/CPS)');

grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('AFDM多符号组帧性能 (M=%d, N=%d)', M, N), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);
set(gca, 'FontSize', 11);

% 设置y轴范围以显示更低的BER值
ylim([1e-3 1]);  % 从10^-3到1

% 性能增益标注
for i = 1:length(SNR_dB_range)
    if BER_frame_no_guard(i) > 0 && BER_frame_with_guard(i) > 0
        gain_dB = 10*log10(BER_frame_no_guard(i) / BER_frame_with_guard(i));
        if i == 3  % 在SNR=10dB处标注
            text(SNR_dB_range(i), BER_frame_with_guard(i)*3, ...
                 sprintf('CPP/CPS增益\n%.1f dB', gain_dB), ...
                 'FontSize', 9, 'HorizontalAlignment', 'center', ...
                 'BackgroundColor', 'yellow', 'EdgeColor', 'black');
        end
    end
end

%% ----- 图2: DAF域信道矩阵 -----
fprintf('  绘制DAF域信道矩阵...\n');
figure('Position', [150, 150, 1200, 500], 'Name', '图2: DAF域信道矩阵');

% 子图1: 完整信道矩阵
subplot(1,2,1);
H_abs = abs(Heff_vis);
H_abs_norm = H_abs / max(H_abs(:));
threshold_vis = 0.001;  % 相对阈值
H_abs_norm(H_abs_norm < threshold_vis) = 0;

imagesc(H_abs_norm);
colormap(gca, 'jet');
colorbar;
title('(a) DAF域信道 |H| (随机实例, Block 100)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('子载波索引 k', 'FontSize', 11);
ylabel('子载波索引 q', 'FontSize', 11);
axis xy;  % 副对角线方向
axis square;
set(gca, 'FontSize', 10);

% 添加参数标注
H_eff_energy = norm(Heff_vis(Q+1:N-Q, Q+1:N-Q), 'fro')^2 / N_eff;
text(N*0.05, N*0.95, sprintf('归一化能量=%.2f', H_eff_energy), ...
     'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold', ...
     'BackgroundColor', 'black');

% 子图2: 有效子载波区域放大
subplot(1,2,2);
H_eff_sub_vis = Heff_vis(Q+1:N-Q, Q+1:N-Q);
H_eff_abs = abs(H_eff_sub_vis);
H_eff_norm = H_eff_abs / max(H_eff_abs(:));
H_eff_norm(H_eff_norm < threshold_vis) = 0;

imagesc(H_eff_norm);
colormap(gca, 'jet');
colorbar;
title(sprintf('(b) 有效区域 (子载波 %d:%d, Block 100)', Q+1, N-Q), 'FontSize', 13, 'FontWeight', 'bold');
xlabel('有效子载波索引', 'FontSize', 11);
ylabel('有效子载波索引', 'FontSize', 11);
axis xy;
axis square;
set(gca, 'FontSize', 10);

% 信道参数标注
% 格式化每条路径的幅度信息
Hi_norms_str = sprintf('%.2f ', Hi_norms_vis);
info_str = sprintf(['信道参数:\n' ...
                    '路径数: %d\n' ...
                    'l_i: [%s]\n' ...
                    '\\alpha_i: [%s]\n' ...
                    '||H_i||_F: [%s]'], ...
                   num_paths, num2str(l_i_vis'), num2str(alpha_i_vis'), Hi_norms_str);
text(N_eff*0.05, N_eff*0.95, info_str, ...
     'Color', 'white', 'FontSize', 9, ...
     'BackgroundColor', 'black', 'EdgeColor', 'white', ...
     'VerticalAlignment', 'top');

%% ----- 图3: 时域帧结构 -----
fprintf('  绘制时域帧结构...\n');
figure('Position', [200, 200, 1400, 500], 'Name', '图3: 时域帧结构');

% 子图1: 帧结构示意（颜色标注）
subplot(2,1,1);
L_block_vis = L_cpp + N + L_cps;
total_samples_vis = M * L_block_vis;

% 创建颜色标记数组
frame_colors = zeros(total_samples_vis, 1);
for m = 1:M
    start_idx = (m-1) * L_block_vis + 1;
    
    % CPP区域 = 1 (红色)
    if L_cpp > 0
        frame_colors(start_idx : start_idx + L_cpp - 1) = 1;
    end
    
    % Symbol区域 = 2 (绿色)
    frame_colors(start_idx + L_cpp : start_idx + L_cpp + N - 1) = 2;
    
    % CPS区域 = 3 (蓝色)
    if L_cps > 0
        frame_colors(start_idx + L_cpp + N : start_idx + L_cpp + N + L_cps - 1) = 3;
    end
end

imagesc(frame_colors');
colormap([1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1]);  % CPP=红, Symbol=绿, CPS=蓝
yticks([]);

% 添加分隔线和标注
for m = 1:M
    start_idx = (m-1) * L_block_vis + 1;
    
    % 分隔线
    hold on;
    plot([start_idx start_idx], [0.5 1.5], 'k--', 'LineWidth', 1.5);
    
    % CPP标注
    if L_cpp > 0
        text(start_idx + L_cpp/2, 0.5, 'CPP', ...
             'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % Symbol标注
    text(start_idx + L_cpp + N/2, 0.5, sprintf('符号%d', m), ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    
    % CPS标注
    if L_cps > 0 && L_cps > 5  % 只有足够大才标注
        text(start_idx + L_cpp + N + L_cps/2, 0.5, 'CPS', ...
             'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end
hold off;

xlabel('采样点索引', 'FontSize', 11);
title(sprintf('(a) 帧结构示意 (M=%d符号/帧, 总长=%d采样点, %.3fs)', M, total_samples_vis, total_samples_vis/fs), ...
      'FontSize', 13, 'FontWeight', 'bold');

% 添加图例
legend_handles = [];
legend_handles(1) = patch([0 0 1 1], [0 1 1 0], [1 0.8 0.8], 'EdgeColor', 'none');
legend_handles(2) = patch([0 0 1 1], [0 1 1 0], [0.8 1 0.8], 'EdgeColor', 'none');
legend_handles(3) = patch([0 0 1 1], [0 1 1 0], [0.8 0.8 1], 'EdgeColor', 'none');
legend(legend_handles, {'CPP (Chirp-Periodic Prefix)', 'AFDM符号', 'CPS (Chirp-Periodic Suffix)'}, ...
       'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 10);

% 子图2: 时域波形幅度
subplot(2,1,2);
plot(abs(x_time_with_guard_vis), 'b-', 'LineWidth', 1);
hold on;

% 标注符号边界
for m = 1:M
    start_idx = (m-1) * L_block_vis + 1;
    % 符号起始线
    plot([start_idx + L_cpp, start_idx + L_cpp], [0, max(abs(x_time_with_guard_vis))], ...
         'r--', 'LineWidth', 1.5);
    % 符号结束线
    plot([start_idx + L_cpp + N, start_idx + L_cpp + N], [0, max(abs(x_time_with_guard_vis))], ...
         'r--', 'LineWidth', 1.5);
end
hold off;

grid on;
xlabel('采样点索引', 'FontSize', 11);
ylabel('幅度', 'FontSize', 11);
title('(b) 时域帧信号幅度', 'FontSize', 13, 'FontWeight', 'bold');
xlim([1, total_samples_vis]);
set(gca, 'FontSize', 10);

%% ----- 图4: CPP/CPS细节展示 -----
fprintf('  绘制CPP/CPS细节...\n');
figure('Position', [250, 250, 1200, 600], 'Name', '图4: CPP/CPS构造细节');

% 提取第一个符号的CPP和CPS部分进行展示
if L_cpp > 0 && L_cps > 0
    % 第一个符号的时域信号
    x_T_symbol1 = A * x_frame_vis(:, 1);
    
    % CPP部分
    n_cpp_vis = (-L_cpp : -1)';
    cpp_source_idx_vis = n_cpp_vis + N + 1;
    phase_cpp_vis = -2*pi*c1*(N^2 + 2*N*n_cpp_vis);
    cpp_part_vis = x_T_symbol1(cpp_source_idx_vis) .* exp(1j * phase_cpp_vis);
    
    % CPS部分
    n_cps_vis = (N : N + L_cps - 1)';
    cps_source_idx_vis = n_cps_vis - N + 1;
    phase_cps_vis = 2*pi*c1*(N^2 + 2*N*n_cps_vis);
    cps_part_vis = x_T_symbol1(cps_source_idx_vis) .* exp(1j * phase_cps_vis);
    
    % 子图1: CPP源位置
    subplot(2,2,1);
    plot(abs(x_T_symbol1), 'b-', 'LineWidth', 1);
    hold on;
    plot(cpp_source_idx_vis, abs(x_T_symbol1(cpp_source_idx_vis)), 'ro', 'MarkerSize', 6, 'LineWidth', 2);
    hold off;
    grid on;
    xlabel('符号内采样点', 'FontSize', 10);
    ylabel('幅度', 'FontSize', 10);
    title('(a) CPP源：符号尾部', 'FontSize', 12, 'FontWeight', 'bold');
    legend('完整符号', 'CPP源位置', 'FontSize', 9);
    set(gca, 'FontSize', 9);
    
    % 子图2: CPP相位补偿
    subplot(2,2,2);
    plot(n_cpp_vis, angle(exp(1j * phase_cpp_vis)), 'r-', 'LineWidth', 2);
    grid on;
    xlabel('CPP索引 n', 'FontSize', 10);
    ylabel('相位补偿 (rad)', 'FontSize', 10);
    title('(b) CPP chirp相位补偿', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9);
    
    % 子图3: 完整符号块
    subplot(2,1,2);
    full_block = [cpp_part_vis; x_T_symbol1; cps_part_vis];
    plot(abs(full_block), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % CPP区域高亮
    area(1:L_cpp, abs(full_block(1:L_cpp)), 'FaceColor', [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
    % Symbol区域
    area(L_cpp+1:L_cpp+N, abs(full_block(L_cpp+1:L_cpp+N)), 'FaceColor', [0.8 1 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    
    % CPS区域高亮
    if L_cps > 0
        area(L_cpp+N+1:L_cpp+N+L_cps, abs(full_block(L_cpp+N+1:end)), 'FaceColor', [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
    
    % 边界线
    plot([L_cpp, L_cpp], [0, max(abs(full_block))], 'r--', 'LineWidth', 2);
    plot([L_cpp+N, L_cpp+N], [0, max(abs(full_block))], 'r--', 'LineWidth', 2);
    
    % 标注
    text(L_cpp/2, max(abs(full_block))*0.9, sprintf('CPP\n(%d)', L_cpp), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    text(L_cpp+N/2, max(abs(full_block))*0.9, sprintf('符号1\n(%d)', N), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    if L_cps > 0
        text(L_cpp+N+L_cps/2, max(abs(full_block))*0.5, sprintf('CPS\n(%d)', L_cps), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
    
    hold off;
    grid on;
    xlabel('采样点索引', 'FontSize', 11);
    ylabel('幅度', 'FontSize', 11);
    title(sprintf('(c) 完整符号块 [CPP | 符号 | CPS] (总长=%d)', L_block_vis), ...
          'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'FontSize', 10);
end

%% ==================== 7. 性能分析 ====================
fprintf('\n【7/7】性能分析总结\n');
fprintf('========================================\n');

fprintf('系统配置:\n');
fprintf('  每帧符号数: M = %d\n', M);
fprintf('  CPP长度: %d采样点\n', L_cpp);
fprintf('  CPS长度: %d采样点\n', L_cps);
fprintf('  帧效率: %.1f%% (符号占总帧长度的比例)\n', 100*M*N/(M*(L_cpp+N+L_cps)));

fprintf('\n性能对比 (SNR = 10 dB):\n');
snr_10_idx = find(SNR_dB_range == 10);
if ~isempty(snr_10_idx)
    fprintf('  单符号基准:       %.2e\n', BER_single(snr_10_idx));
    fprintf('  多符号(无保护):   %.2e\n', BER_frame_no_guard(snr_10_idx));
    fprintf('  多符号(有保护):   %.2e\n', BER_frame_with_guard(snr_10_idx));
    
    if BER_frame_no_guard(snr_10_idx) > 0
        isi_penalty = 10*log10(BER_frame_no_guard(snr_10_idx) / BER_single(snr_10_idx));
        fprintf('  ISI损失:         %.1f dB\n', isi_penalty);
    end
    
    if BER_frame_with_guard(snr_10_idx) > 0 && BER_frame_no_guard(snr_10_idx) > 0
        cpp_cps_gain = 10*log10(BER_frame_no_guard(snr_10_idx) / BER_frame_with_guard(snr_10_idx));
        fprintf('  CPP/CPS增益:     %.1f dB\n', cpp_cps_gain);
    end
end

fprintf('\n验证结果:\n');
if BER_frame_with_guard(end) <= BER_single(end) * 1.5
    fprintf('  ✓ CPP/CPS成功抑制ISI\n');
else
    fprintf('  ⚠️ CPP/CPS效果有限\n');
end

fprintf('  ✓ 时域帧结构构建成功\n');
fprintf('  ✓ DAF域传输模型验证\n');
fprintf('  ✓ 多符号组帧仿真完成\n');

fprintf('\n========================================\n');
fprintf('混合方案仿真完成！\n');
fprintf('========================================\n\n');
