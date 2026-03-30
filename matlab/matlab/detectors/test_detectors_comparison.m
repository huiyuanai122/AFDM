clear;clc;close all;

%% ============================================================
%% 宽带AFDM系统 - 基于论文Section II信道建模的仿真验证
%% 参考论文："Affine Frequency Division Multiplexing Over Wideband 
%%           Doubly-Dispersive Channels With Time-Scaling Effects"
%% 
%% ========== Section II 信道建模要点 ==========
%% 1. 宽带双色散信道模型（Wideband Doubly-Dispersive Channel）
%%    - 考虑时间尺度效应（Time-Scaling Effects）
%%    - 信道输入输出关系（论文公式(3)）：
%%      y(t) = Σ_i h_i · x[(1+αᵢ)t - τᵢ] + w(t)
%%    - h_i: 第i条路径的复增益
%%    - τᵢ: 第i条路径的时延
%%    - αᵢ: 第i条路径的多普勒缩放因子（时间尺度效应）
%%
%% 2. DAF域等效信道（论文公式(27)-(29)）
%%    - 单径信道矩阵Hi的(p,q)元素：
%%      Hi[p,q] = (1/N)Σ_{n=0}^{N-1} exp{j2π/N·Φ(p,q,n)}
%%    - 相位函数Φ包含时间尺度效应项αᵢ·n²
%%    - POSP近似：稀疏对角带状结构
%%
%% 3. 时域等效信道（论文公式及分析）
%%    - 时域信道：H_T = A^H * H * A
%%    - Figure 5展示：时域信道比DAF域更稀疏
%%
%% ========== 仿真目标 ==========
%% - 实现论文Figure 5的信道矩阵可视化
%% - 对比DAF域和时域信道的稀疏性
%% - 验证检测算法性能
%% ============================================================

fprintf('\n========================================\n');
fprintf('宽带AFDM: 信道矩阵可视化与检测算法对比\n');
fprintf('========================================\n\n');

%% ==================== 物理参数设置（符合论文Section VI）====================
c_sound = 1500;                % 水中声速 (m/s)
fc = 6e3;                      % 载波频率 6kHz (论文Section VI设置)
% N值选择：
%   N = 256   → 快速验证（2-3分钟）
%   N = 512   → 标准仿真（5-10分钟）
%   N = 1024  → 完整仿真（15-30分钟）
N = 1024;                      
df = 4;                        % 子载波间隔 4Hz (论文设置)
B = N * df;                    % 信号带宽 = N·Δf

alpha_max_paper = 1e-4;        % 最大多普勒缩放因子αmax (论文Section II.A)
tau_max = 25e-3;               % 最大时延扩展τmax = 25ms
num_paths = 4;                 % 路径数P (论文Section VI: P=4)

SNR_dB_range = 0:5:20;         % SNR范围 (dB) - 完整仿真用0:5:20

%% ==================== 系统参数 ====================
fprintf('===== 系统参数 =====\n');
fprintf('子载波数 N: %d\n', N);
fprintf('载波频率 fc: %.0f kHz\n', fc/1e3);
fprintf('子载波间隔 Δf: %.0f Hz\n', df);
fprintf('信号带宽 B: %.2f kHz\n', B/1e3);
fprintf('带宽比 B/fc: %.3f (宽带)\n', B/fc);
fprintf('路径数 P: %d\n', num_paths);
fprintf('\n');

%% ==================== 归一化参数（论文Section II.B）====================
% 论文公式(6)：离散时间归一化
fd_max = alpha_max_paper * fc;     % 最大多普勒频移 fd = α·fc
alpha_max = ceil(fd_max / df);     % 归一化多普勒索引（离散化）
l_max = ceil(tau_max * df * N);    % 归一化时延索引 l = τ·Δf·N

fprintf('\n=== 信道物理参数（Section II）===\n');
fprintf('  最大时延扩展 τ_max = %.2f ms\n', tau_max*1000);
fprintf('  物理多普勒因子 α_max,phy = %.6f (论文公式3)\n', alpha_max_paper);
fprintf('  最大多普勒频移 f_d,max = %.2f Hz\n', fd_max);
fprintf('  带宽比 B/fc = %.3f (%.1f%% → 宽带)\n', B/fc, B/fc*100);

fprintf('\n=== DAF域信道矩阵索引参数 ===\n');
fprintf('  时延索引 l_i ∈ [0, l_max]\n');
fprintf('    l_max = ⌈τ_max × fs⌉ = ⌈%.2f × %d⌉ = %d\n', tau_max, fs, l_max);
fprintf('  多普勒索引 α_i ∈ [-α_max, α_max]\n');
fprintf('    α_max = ⌈f_d,max / Δf⌉ = ⌈%.2f / %d⌉ = %d\n', fd_max, df, alpha_max);
fprintf('  → 信道矩阵构建: H = Σ h_i · H_i(l_i, α_i)\n');
fprintf('  → l_i 范围: [0, %d], α_i 范围: [-%d, %d]\n', l_max, alpha_max, alpha_max);

fprintf('\n=== 宽带信道特性 ===\n');
fprintf('  → 论文Section II.A: 宽带信道，必须考虑时间尺度效应\n');
fprintf('  → 信道模型包含 α_i·n² 项（论文公式27-29）\n');
fprintf('\n');

%% ==================== AFDM参数（论文Section II.C）====================
% 论文公式(10)-(11): Chirp参数设计
Nv = 2;  % 虚拟子载波数（边界保护）
c1 = (2*alpha_max + 1 + 2*Nv/N) / (2*N);  % 论文公式(11)
c2 = 1 / (2*N);                            % 论文公式(10)

% 论文公式(34): 单径最大扩散宽度
L_spread = 2*N*c1*(2*alpha_max + 1 + 2*Nv/N);

% 护卫带设计（保守策略）
% Q应该大于L_spread/2，并留有裕量防止边界效应和数值误差
Q = ceil(L_spread / 2) + 5;  % 增加裕量（论文隐含要求）
N_eff = N - 2*Q;  % 有效子载波数（排除护卫带）

% 安全性检查
if N_eff < N/2
    warning('护卫带过大！有效子载波仅占%.1f%%，考虑增大N或减小alpha_max', 100*N_eff/N);
end

fprintf('===== AFDM参数 =====\n');
fprintf('Chirp参数 c1: %.8f\n', c1);
fprintf('Chirp参数 c2: %.8f\n', c2);
fprintf('预期信道扩散宽度: %.1f 子载波\n', L_spread);
fprintf('护卫带 Q: %d\n', Q);
fprintf('有效子载波数 N_eff = N-2Q: %d\n', N_eff);
fprintf('→ 数据传输在子载波 %d:%d，边缘置零作为护卫带\n', Q+1, N-Q);
fprintf('\n');

%% ==================== 仿真参数 ====================
mod_order = 4;              % QPSK调制 (论文Section VI)

% Monte Carlo块数选择（根据N值和精度要求）：
%   num_blocks = 50   → 快速验证（15分钟），低精度
%   num_blocks = 100  → 标准仿真（30分钟），中等精度（推荐）
%   num_blocks = 200  → 完整仿真（60分钟），高精度
%   num_blocks = 500+ → 论文级别（2-3小时），很高精度
% 
% 原则：确保至少100-200个误码用于统计
%       总比特数 = num_blocks × N × log2(mod_order)
num_blocks = 100;           % 推荐100（N=1024时）

fprintf('===== 检测算法列表 =====\n');
fprintf('1. ZF (Zero Forcing)\n');
fprintf('2. LMMSE (Linear MMSE)\n');
fprintf('\n');

fprintf('===== 仿真设置 =====\n');
fprintf('调制方式: QPSK\n');
fprintf('Monte Carlo块数: %d\n', num_blocks);
fprintf('假设: Perfect CSI\n');
fprintf('SNR范围: %d:%d:%d dB\n', SNR_dB_range(1), SNR_dB_range(2)-SNR_dB_range(1), SNR_dB_range(end));
fprintf('========================================\n\n');

%% ==================== 构建AFDM调制矩阵 ====================
F = dftmtx(N)/sqrt(N);
Lambda_c1 = diag(exp(-1j * 2 * pi * c1 * (0:N-1).^2));
Lambda_c2 = diag(exp(-1j * 2 * pi * c2 * (0:N-1).^2));
A  = Lambda_c2 * F * Lambda_c1;
AH = Lambda_c1' * F' * Lambda_c2';

%% ==================== 结果存储 ====================
num_detectors = 2;
detector_names = {'ZF', 'LMMSE'};

BER_results = zeros(num_detectors, length(SNR_dB_range));

%% ==================== Monte Carlo 仿真 ====================
fprintf('开始仿真...\n\n');

% ===== 新增：保存真实仿真信道用于可视化 =====
num_snapshots = 3;  % 保存前3个block的信道矩阵
H_snapshots = cell(num_snapshots, 1);  % 保存信道矩阵
params_snapshots = cell(num_snapshots, 1);  % 保存信道参数
snapshot_saved = false(num_snapshots, 1);  % 标记是否已保存

% ===== 新增：计算平均信道矩阵（时变信道的统计特性）=====
H_sum = zeros(N, N);  % 累加所有block的信道矩阵
H_count = 0;  % 计数

for snr_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(snr_idx);
    SNR = 10^(SNR_dB / 10);
    noise_power = 1 / SNR;
    
    ber_blk = zeros(num_detectors, num_blocks);
    
    for blk = 1:num_blocks
        %% 发射端：只在有效子载波上传输数据，边缘护卫带置零
        data = randi([0, mod_order-1], N_eff, 1);  % 只生成N_eff个符号
        data_mod = qammod(data, mod_order, 'UnitAveragePower', true);
        
        % 构造完整的N维发射向量：中间有效，边缘置零
        x = zeros(N, 1);
        x(Q+1:N-Q) = data_mod;  % 只在中间的N_eff个子载波上传输数据
        
        % 发射功率归一化（关键！）
        % 目标：E[|x|²] = sum(|x|²)/N = 1（全局平均功率）
        % 当前：只有N_eff个位置非零，每个功率=1（qammod已归一化）
        % 需要：放大因子 sqrt(N/N_eff)，使得 (N_eff*1)/N * (N/N_eff) = 1
        x = x * sqrt(N / N_eff);
        
        % 验证：实际功率应该接近1
        % actual_power = mean(abs(x).^2);
        % assert(abs(actual_power - 1) < 0.01, '发射功率归一化错误！');
        
        %% 信道生成（论文Section II.A 宽带双色散信道模型）
        % 公式(3): y(t) = Σᵢ hᵢ·x[(1+αᵢ)t - τᵢ] + w(t)
        
        % 时延τᵢ：均匀分布在[0, τmax]
        l_i = sort(randi([0, l_max], num_paths, 1));  % 修正：确保l_i ∈ [0, l_max]
        
        % 多普勒缩放因子αᵢ ∈ [-αmax, αmax]（论文Section II.A）
        % 归一化索引空间均匀采样
        alpha_i = randi([-alpha_max, alpha_max], num_paths, 1);
        
        % 路径增益hᵢ：Rayleigh衰落（论文Section VI）
        h_true = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
        h_true = h_true / sqrt(sum(abs(h_true).^2));  % 总功率归一化
        
        % 构建DAF域等效信道矩阵（论文公式(27)-(29)）
        % H = Σᵢ hᵢ·Hᵢ，其中Hᵢ是单径信道矩阵
        Heff = zeros(N, N);
        for i = 1:num_paths
            % 单径信道矩阵Hᵢ（论文公式(27)）
            % Hᵢ[p,q] = (1/N)Σₙ exp{j2π/N·Φ(p,q,n)}
            % Φ包含时间尺度效应项αᵢ·n²
            Hi = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
            Heff = Heff + h_true(i) * Hi;
        end
        
        % ===== 新增：保存归一化前的信道矩阵（用于可视化）=====
        % 只在第一个SNR点的前几个block保存
        if snr_idx == 1 && blk <= num_snapshots && ~snapshot_saved(blk)
            H_snapshots{blk} = Heff;  % 保存归一化前的H矩阵（幅值大）
            params_snapshots{blk} = struct('l_i', l_i, 'alpha_i', alpha_i, ...
                                           'h_true', h_true, 'blk', blk, 'SNR_dB', SNR_dB);
            snapshot_saved(blk) = true;
            fprintf('  [保存] Block %d的信道快照已保存（归一化前）\n', blk);
        end
        
        % ===== 新增：累加归一化前的信道矩阵用于计算平均 =====
        % 只在第一个SNR点累加（避免重复）
        if snr_idx == 1
            H_sum = H_sum + Heff;  % 累加归一化前的值
            H_count = H_count + 1;
        end
        
        % 信道功率归一化（修正版）
        % ⚠️ 注意：归一化在保存快照之后，确保可视化时幅值足够大
        % 策略：确保接收功率 E[|Heff*x|²] ≈ E[|x|²] = 1
        % 
        % 方法1（当前）：基于有效子空间归一化（推荐！）
        % 只考虑承载数据的子载波对应的信道子矩阵
        H_eff_sub = Heff(Q+1:N-Q, Q+1:N-Q);  % N_eff × N_eff 子矩阵
        
        % 计算子矩阵的能量范数（Frobenius范数）
        % 理想情况：||H_sub||_F² / N_eff ≈ N_eff（单位增益）
        sub_power = norm(H_eff_sub, 'fro')^2 / N_eff;
        if sub_power > 1e-10
            Heff = Heff / sqrt(sub_power);
        end
        
        % 方法2（备选）：对角线归一化（原来的方法）
        % diag_power = mean(abs(diag(Heff)).^2);
        % if diag_power > 0
        %     Heff = Heff / sqrt(diag_power);
        % end
        
        %% 信道传输（DAF域等效模型）
        % 根据论文，Heff已经是DAF域等效信道矩阵（公式27）
        % 等效模型：y = Heff * x + w （在DAF域直接计算）
        % 这等价于：y = A^H * H_time * A * x + w
        w = sqrt(noise_power/2) * (randn(N,1) + 1j*randn(N,1));
        y = Heff * x + w;
        
        %% ========== 检测算法对比（Perfect CSI）==========
        % 使用完整N×N信道矩阵进行检测，不提取子矩阵！
        % 因为x的边缘已经是零，检测后自动得到边缘为零的结果
        
        % 调试：检查第一个块的信道特性
        if blk == 1 && snr_idx == 1
            fprintf('  [调试] Heff条件数: %.2e\n', cond(Heff));
            fprintf('  [调试] Heff秩: %d / %d\n', rank(Heff, 1e-6), N);
            fprintf('  [调试] 接收信号功率: %.4f\n', mean(abs(y).^2));
            fprintf('  [调试] 发射信号功率: %.4f\n', mean(abs(x).^2));
        end
        
        % 1. ZF（在完整N维空间检测）
        x_hat_zf = detector_ZF(y, Heff);
        % 只对中间有效子载波解调
        x_hat_zf_eff = x_hat_zf(Q+1:N-Q);
        demod_zf = qamdemod(x_hat_zf_eff, mod_order, 'UnitAveragePower', true);
        ber_blk(1, blk) = mean(demod_zf ~= data);
        
        % 2. LMMSE（在完整N维空间检测）
        x_hat_lmmse = detector_LMMSE(y, Heff, noise_power);
        % 只对中间有效子载波解调
        x_hat_lmmse_eff = x_hat_lmmse(Q+1:N-Q);
        demod_lmmse = qamdemod(x_hat_lmmse_eff, mod_order, 'UnitAveragePower', true);
        ber_blk(2, blk) = mean(demod_lmmse ~= data);
    end
    
    % 统计
    total_bits = N_eff * num_blocks * log2(mod_order);  % 修正：只统计有效子载波的比特数
    ber_floor = 0.5 / total_bits;
    
    for d = 1:num_detectors
        BER_results(d, snr_idx) = max(mean(ber_blk(d, :)), ber_floor);
    end
    
    fprintf('[SNR=%2d dB] ZF=%.4e LMMSE=%.4e\n', SNR_dB, BER_results(1,snr_idx), BER_results(2,snr_idx));
end

fprintf('\n仿真完成！\n\n');

%% ==================== 计算平均信道矩阵 ====================
fprintf('\n========================================\n');
fprintf('计算时变信道的统计平均\n');
fprintf('========================================\n\n');

H_avg = H_sum / H_count;  % 平均信道矩阵
fprintf('累加了 %d 个block的信道矩阵\n', H_count);
fprintf('平均信道矩阵 E[H]:\n');
fprintf('  维度: %d × %d\n', N, N);
fprintf('  非零元素: %d (%.2f%%)\n', nnz(H_avg), 100*nnz(H_avg)/N^2);
fprintf('  最大幅值: %.4f\n', max(abs(H_avg(:))));
fprintf('\n注意：由于随机相位抵消，|E[H]|通常远小于单次信道的|H|\n\n');

%% ==================== 真实仿真信道可视化（新增）====================
fprintf('\n========================================\n');
fprintf('可视化真实BER仿真中使用的信道矩阵\n');
fprintf('========================================\n\n');

% 选择要展示的信道快照（默认第一个）
snapshot_idx = 1;  % 可以改为1/2/3查看不同的block
Heff_real = H_snapshots{snapshot_idx};
params_real = params_snapshots{snapshot_idx};

fprintf('===== 真实仿真信道 (Block %d) =====\n', params_real.blk);
fprintf('路径参数:\n');
for i = 1:num_paths
    fprintf('  路径%d: l_i=%3d, alpha_i=%2d, |h_i|=%.3f, ∠h_i=%.2f°\n', ...
            i, params_real.l_i(i), params_real.alpha_i(i), ...
            abs(params_real.h_true(i)), angle(params_real.h_true(i))*180/pi);
end
fprintf('SNR: %d dB\n', params_real.SNR_dB);

% 统计信息
nnz_real = nnz(Heff_real);
sparsity_real = 100 * nnz_real / N^2;
fprintf('\nDAF域信道矩阵 H:\n');
fprintf('  维度: %d × %d\n', N, N);
fprintf('  非零元素: %d (%.2f%%)\n', nnz_real, sparsity_real);
fprintf('  条件数: %.2e\n', cond(Heff_real));

% 计算时域信道矩阵
HT_real = AH * Heff_real * A;
threshold_time_real = 1e-3 * max(abs(HT_real(:)));
HT_real_sparse = HT_real;
HT_real_sparse(abs(HT_real_sparse) < threshold_time_real) = 0;

nnz_time_real = nnz(HT_real_sparse);
sparsity_time_real = 100 * nnz_time_real / N^2;
fprintf('\n时域信道矩阵 H_T:\n');
fprintf('  非零元素（阈值后）: %d (%.2f%%)\n', nnz_time_real, sparsity_time_real);
fprintf('\n');

%% 可视化：真实仿真信道的DAF域和时域对比
figure('Position', [50, 50, 1400, 600], 'Name', sprintf('真实仿真信道 (Block %d)', snapshot_idx));

% 子图1: DAF域信道矩阵
subplot(1,2,1);
H_real_abs = abs(Heff_real);
H_real_abs_norm = H_real_abs / max(H_real_abs(:));
% 使用自适应阈值：保留最大值的0.1%以上的元素
threshold_vis_adaptive = 0.001;  % 相对阈值（最大值的0.1%）
H_real_abs_norm(H_real_abs_norm < threshold_vis_adaptive) = 0;

imagesc(H_real_abs_norm);
colormap(gca, 'jet');
colorbar;
title(sprintf('(a) 真实DAF域信道 |H| (Block %d)', snapshot_idx), 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Symbol index', 'FontSize', 12);
ylabel('Symbol index', 'FontSize', 12);
axis xy;  % 副对角线方向
axis square;
set(gca, 'FontSize', 11);
% 添加说明
text(N*0.05, N*0.95, sprintf('max|H|=%.3f', max(H_real_abs(:))), ...
     'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

% 子图2: 时域信道矩阵
subplot(1,2,2);
H_time_abs = abs(HT_real_sparse);
H_time_abs_norm = H_time_abs / max(abs(HT_real(:)));
H_time_abs_norm(H_time_abs_norm < threshold_vis_adaptive) = 0;

imagesc(H_time_abs_norm);
colormap(gca, 'jet');
colorbar;
title('(b) 时域信道 |H_T|', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Symbol index', 'FontSize', 12);
ylabel('Symbol index', 'FontSize', 12);
axis xy;
axis square;
set(gca, 'FontSize', 11);
text(N*0.05, N*0.95, sprintf('max|H_T|=%.3f', max(abs(HT_real(:)))), ...
     'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

sgtitle(sprintf('真实BER仿真中的信道 - Block %d (SNR=%d dB)', ...
        params_real.blk, params_real.SNR_dB), 'FontSize', 15, 'FontWeight', 'bold');

%% ==================== 多个信道快照对比（时变特性）====================
figure('Position', [100, 100, 1400, 800], 'Name', '信道时变特性展示');

for idx = 1:num_snapshots
    H_snap = H_snapshots{idx};
    params_snap = params_snapshots{idx};
    
    subplot(2, num_snapshots, idx);
    H_snap_norm = abs(H_snap) / max(abs(H_snap(:)));
    H_snap_norm(H_snap_norm < threshold_vis_adaptive) = 0;  % 使用相对阈值
    
    imagesc(H_snap_norm);
    colormap(gca, 'jet');
    colorbar;
    title(sprintf('Block %d (max=%.3f)', idx, max(abs(H_snap(:)))), ...
          'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Symbol index');
    ylabel('Symbol index');
    axis xy;
    axis square;
    
    % 下方显示参数
    subplot(2, num_snapshots, num_snapshots + idx);
    axis off;
    param_text = sprintf('Block %d 信道参数:\n\n', idx);
    for i = 1:num_paths
        param_text = [param_text, sprintf('路径%d:\n  l=%d, α=%d\n  |h|=%.2f\n\n', ...
            i, params_snap.l_i(i), params_snap.alpha_i(i), abs(params_snap.h_true(i)))];
    end
    text(0.1, 0.9, param_text, 'FontSize', 9, 'VerticalAlignment', 'top', ...
         'FontName', 'Courier New', 'Interpreter', 'none');
end

sgtitle('信道时变特性：连续3个符号遇到的不同信道矩阵', 'FontSize', 14, 'FontWeight', 'bold');

%% ==================== 时变信道的不同表示方式对比 ====================
figure('Position', [150, 150, 1400, 500], 'Name', '时变信道的三种表示');

% 子图1: 单次快照（Block 1）
subplot(1,3,1);
H1_norm = abs(H_snapshots{1}) / max(abs(H_snapshots{1}(:)));
H1_norm(H1_norm < threshold_vis_adaptive) = 0;
imagesc(H1_norm);
colormap(gca, 'jet');
colorbar;
title(sprintf('(a) 单次快照 H^{(1)} (max=%.3f)', max(abs(H_snapshots{1}(:)))), ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('Symbol index');
ylabel('Symbol index');
axis xy;
axis square;

% 子图2: 平均信道 E[H]
subplot(1,3,2);
H_avg_norm = abs(H_avg) / max(abs(H_avg(:)));
H_avg_norm(H_avg_norm < threshold_vis_adaptive) = 0;
imagesc(H_avg_norm);
colormap(gca, 'jet');
colorbar;
title(sprintf('(b) 平均信道 E[H] (%d blocks, max=%.3f)', H_count, max(abs(H_avg(:)))), ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('Symbol index');
ylabel('Symbol index');
axis xy;
axis square;

% 子图3: 幅度差异对比
subplot(1,3,3);
max_vals = [max(abs(H_snapshots{1}(:))), max(abs(H_snapshots{2}(:))), ...
            max(abs(H_snapshots{3}(:))), max(abs(H_avg(:)))];
bar(max_vals);
set(gca, 'XTickLabel', {'Block 1', 'Block 2', 'Block 3', 'E[H]'});
ylabel('最大幅值');
title('(c) 信道矩阵最大幅值对比', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
for i = 1:4
    text(i, max_vals(i), sprintf('%.3f', max_vals(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
end

sgtitle('时变信道的表示方式：单次快照 vs 统计平均', 'FontSize', 15, 'FontWeight', 'bold');

%% ==================== 【可选】论文风格可视化（精心设计的参数）====================
fprintf('\n========================================\n');
fprintf('【可选】生成论文风格的信道可视化（精心设计的参数）\n');
fprintf('========================================\n\n');

% 生成一个典型多径信道实例（符合论文设置）
% 不重叠条件：相邻路径中心间隔 > 两者带宽之和

% 先设置alpha_i（决定带宽）
% ⚠️ 必须符合物理范围：[-alpha_max, alpha_max]
% 当前 alpha_max = 1，所以只能用 {-1, 0, 1}
if l_max >= 30
    alpha_i_vis = [-1, -1, 1, 1];  % 使用相同带宽，通过l_i区分
else
    alpha_i_vis = [-1, 0, 0, 1];   % 使用不同带宽
end
fprintf('⚠️ 注意：alpha_max=%d，可视化alpha_i必须在[%d,%d]范围内\n', ...
        alpha_max, -alpha_max, alpha_max);

% 计算每条路径的半带宽
half_widths = zeros(num_paths, 1);
for i = 1:num_paths
    half_widths(i) = ceil(N * c1 * (2*abs(alpha_i_vis(i)) + 1));
end

% 手动设置路径时延，确保明显分离（对应论文Figure 2）
% 论文中峰值间距约500个索引，这里设置较大间隔
% 对于N=1024，l_max=102，我们设置间隔使得q空间峰间距约250-300
l_i_vis = [0, round(l_max*0.25), round(l_max*0.5), round(l_max*0.75)];
l_i_vis = min(l_i_vis, l_max);  % 确保不超过l_max

fprintf('\n===== 可视化路径参数（手动设置以确保明显分离） =====\n');
fprintf('路径时延 l_i: [%d, %d, %d, %d]\n', l_i_vis(1), l_i_vis(2), l_i_vis(3), l_i_vis(4));

% 估算q空间的峰值间距
for i = 1:num_paths-1
    delta_l = l_i_vis(i+1) - l_i_vis(i);
    delta_q = delta_l * abs(1 - 2*N*c1);
    fprintf('  路径%d到路径%d: Δl=%d, 预期q间距≈%.0f\n', i, i+1, delta_l, delta_q);
end
fprintf('\n');

h_vis = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
h_vis = h_vis / sqrt(sum(abs(h_vis).^2));  % 总功率归一化（与仿真主循环一致）

% 单径信道矩阵（第一条路径）
Hi_single = build_Hi_correct(N, c1, c2, l_i_vis(1), alpha_i_vis(1));

% 多径合成信道矩阵（保存每条路径）
Heff_vis = zeros(N, N);
Hi_paths = cell(num_paths, 1);  % 保存每条路径的Hi矩阵
for i = 1:num_paths
    Hi_temp = build_Hi_correct(N, c1, c2, l_i_vis(i), alpha_i_vis(i));
    Hi_paths{i} = h_vis(i) * Hi_temp;  % 保存加权后的单径矩阵
    Heff_vis = Heff_vis + Hi_paths{i};
end

% 统计信息
nnz_single = nnz(Hi_single);
nnz_multi = nnz(Heff_vis);
sparsity_single = 100 * nnz_single / N^2;
sparsity_multi = 100 * nnz_multi / N^2;

% ==== 计算时域信道矩阵 H_T ====
% 根据论文，时域信道: H_T = A^H * H * A
% 其中 A 是 IDAF 变换矩阵
HT_vis = AH * Heff_vis * A;  % 时域信道矩阵

% 应用阈值处理（去除数值误差）
threshold_time_vis = 1e-3 * max(abs(HT_vis(:)));
HT_vis_sparse = HT_vis;
HT_vis_sparse(abs(HT_vis_sparse) < threshold_time_vis) = 0;

fprintf('===== DAF域信道矩阵 H =====\n');
fprintf('维度: %d x %d\n', N, N);
fprintf('非零元素: %d (%.2f%%)\n', nnz_multi, sparsity_multi);
fprintf('特点: 对角带状结构（符合POSP近似）\n\n');

nnz_time = nnz(HT_vis_sparse);
sparsity_time = 100 * nnz_time / N^2;
fprintf('===== 时域信道矩阵 H_T =====\n');
fprintf('维度: %d x %d\n', N, N);
fprintf('非零元素（阈值处理后）: %d (%.2f%%)\n', nnz_time, sparsity_time);
fprintf('特点: 时域信道，阈值后显示有效稀疏性\n\n');

fprintf('路径参数:\n');
for i = 1:num_paths
    fprintf('  路径%d: l_i=%d, alpha_i=%d, |h_i|=%.3f\n', ...
            i, l_i_vis(i), alpha_i_vis(i), abs(h_vis(i)));
end
fprintf('\n');

%% ==================== 论文Figure 5: DAF域和时域信道矩阵对比 ====================
figure('Position', [50, 50, 1400, 600]);

% (a) DAF域信道矩阵 |H|
subplot(1,2,1);
H_daf_abs = abs(Heff_vis);
% 处理零值，使用对数尽度更好地显示稀疏结构
H_daf_abs_norm = H_daf_abs / max(H_daf_abs(:));
threshold_vis = 1e-3;  % 显示阈值
H_daf_abs_norm(H_daf_abs_norm < threshold_vis) = 0;

imagesc(H_daf_abs_norm);
colormap(gca, 'jet');
colorbar;
title('(a) DAF domain |H|', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Symbol index', 'FontSize', 12);
ylabel('Symbol index', 'FontSize', 12);
axis xy;  % 使用数学坐标系：y轴从下到上，使得对角线变为副对角线方向
axis square;
set(gca, 'FontSize', 11);

% (b) 时域信道矩阵 |H_T|（使用阈值处理后的）
subplot(1,2,2);
H_time_abs = abs(HT_vis_sparse);
% 处理零值，使用对数尽度
H_time_abs_norm = H_time_abs / max(abs(HT_vis(:)));  % 归一化使用原始最大值
H_time_abs_norm(H_time_abs_norm < threshold_vis) = 0;

imagesc(H_time_abs_norm);
colormap(gca, 'jet');
colorbar;
title('(b) Time domain |H_T|', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Symbol index', 'FontSize', 12);
ylabel('Symbol index', 'FontSize', 12);
axis xy;  % 使用数学坐标系
axis square;
set(gca, 'FontSize', 11);

sgtitle('Fig. 5. Equivalent channel structure for DAF domain and time domain.', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ==================== 详细分析图: DAF域多径信道横截面（对应论文Figure 2） ====================
figure('Position', [50, 100, 900, 600]);

% 固定p值取中间位置
p_select = round(N/2);
path_colors = {'c', 'y', 'g', 'r'};
path_names = {'Path 1', 'Path 2', 'Path 3', 'Path 4'};

% 构建单路径矩阵（用于分离展示）
Hi_array = cell(1, num_paths);
for i = 1:num_paths
    Hi_array{i} = build_Hi_correct(N, c1, c2, l_i_vis(i), alpha_i_vis(i));
end

hold on;

% 绘制总信道（类似论文中的蓝色实线）
profile_total = abs(Heff_vis(p_select, :));
plot(1:N, profile_total, 'b-', 'LineWidth', 3, 'DisplayName', 'Total channel');

% 绘制每条路径的贡献（归一化显示）
for i = 1:num_paths
    profile_i = abs(Hi_array{i}(p_select, :)) * abs(h_vis(i));
    
    % 为了清晰显示，使用虚线
    plot(1:N, profile_i, '--', 'LineWidth', 2, 'Color', path_colors{i}, ...
         'DisplayName', path_names{i});
    
    % 标记峰值位置
    [peak_val, peak_idx] = max(profile_i);
    plot(peak_idx, peak_val, 'o', 'Color', path_colors{i}, ...
         'MarkerSize', 12, 'LineWidth', 2.5, 'MarkerFaceColor', path_colors{i});
    
    % 添加位置标签
    text(peak_idx, peak_val*1.15, sprintf('q=%d', peak_idx), ...
         'Color', path_colors{i}, 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
end

grid on;
legend('Location', 'northeast', 'FontSize', 11);

title(sprintf('DAF Domain Equivalent Channel (p=%d)\nFour-path separation', p_select), ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('Symbol index (q)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
xlim([1, N]);
set(gca, 'FontSize', 11);

%% ==================== 详细分析图2: 时域信道矩阵结构 ====================
figure('Position', [100, 150, 1400, 600]);

% 子图1: 时域信道全图（阈值处理后）
subplot(2,2,1);
imagesc(abs(HT_vis_sparse));
colorbar;
colormap(gca, 'jet');
title('时域信道矩阵 |H_T| 全图（阈值处理后）', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Symbol index');
ylabel('Symbol index');
axis xy;
axis square;

% 子图2: 时域信道局部放大
subplot(2,2,2);
zoom_range = 1:min(512, N);
imagesc(abs(HT_vis_sparse(zoom_range, zoom_range)));
colorbar;
colormap(gca, 'jet');
title('时域信道 局部放大', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Symbol index');
ylabel('Symbol index');
axis xy;
axis square;

% 子图3: DAF域信道全图（对比）
subplot(2,2,3);
imagesc(abs(Heff_vis));
colorbar;
colormap(gca, 'jet');
title('DAF域信道矩阵 |H| 全图', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Symbol index');
ylabel('Symbol index');
axis xy;
axis square;

% 子图4: 稀疏性对比
subplot(2,2,4);
sparsity_data = [sparsity_multi, sparsity_time];
bar(sparsity_data);
set(gca, 'XTickLabel', {'DAF domain', 'Time domain'});
ylabel('Sparsity (%)');
title('信道矩阵稀疏性对比', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
for i = 1:2
    text(i, sparsity_data(i), sprintf('%.2f%%', sparsity_data(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 11);
end

sgtitle(sprintf('时域 vs DAF域信道矩阵对比 (N=%d, P=%d路径)', N, num_paths), ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ==================== BER性能对比图 ====================
figure('Position', [100, 100, 900, 600]);

markers = {'-o', '-s', '-^', '-d', '-v'};
colors = lines(num_detectors);

for d = 1:num_detectors
    semilogy(SNR_dB_range, BER_results(d,:), markers{d}, ...
             'LineWidth', 2.5, 'MarkerSize', 9, 'Color', colors(d,:));
    hold on;
end

grid on;
xlabel('SNR (dB)', 'FontSize', 13);
ylabel('BER', 'FontSize', 13);
legend(detector_names, 'Location', 'southwest', 'FontSize', 12);
title('宽带AFDM检测器性能对比 (Perfect CSI) - Figure 5系统', 'FontSize', 14, 'FontWeight', 'bold');
ylim([1e-5, 1]);

text_str = sprintf('N=%d, fc=%.0fkHz, Δf=%.0fHz\nα_{max}=10^{-4}, P=%d, QPSK', ...
                   N, fc/1e3, df, num_paths);
text(0.02, 0.98, text_str, 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'FontSize', 11, 'BackgroundColor', 'white');

%% ==================== 性能总结 ====================
fprintf('===== 性能总结 =====\n');
fprintf('%-15s', '检测器');
for snr = SNR_dB_range
    fprintf(' | %2ddB', snr);
end
fprintf('\n');
fprintf(repmat('-', 1, 80));
fprintf('\n');

for d = 1:num_detectors
    fprintf('%-15s', detector_names{d});
    for snr_idx = 1:length(SNR_dB_range)
        fprintf(' | %.2e', BER_results(d, snr_idx));
    end
    fprintf('\n');
end
fprintf('\n');

% 性能排序(SNR=15dB)
snr_15_idx = find(SNR_dB_range == 15);
if ~isempty(snr_15_idx)
    [sorted_ber, rank] = sort(BER_results(:, snr_15_idx));
    fprintf('性能排序 @ SNR=15dB (从优到差):\n');
    for i = 1:num_detectors
        fprintf('%d. %s: %.4e\n', i, detector_names{rank(i)}, sorted_ber(i));
    end
end

%% ==================== 辅助函数 ====================

function Hi = build_Hi_correct(N, c1, c2, l_i, alpha_i)
    % 论文公式(19-20): DAF域等效信道矩阵
    % Hi[p,q] = (1/N) * sum_n exp{j*2*pi/N * Phi_i(p,q,n)}
    % 
    % 关键：当 alpha_i = 0 时，POSP近似失效，需要特殊处理！
    
    Hi = zeros(N, N);
    Nv = 2;
    
    if alpha_i == 0
        % ========== alpha_i = 0 的特殊情况 ==========
        % 此时相位函数退化为线性，不是二次相位
        % 可以解析求解，不需要POSP近似
        
        for p = 1:N
            % 对于 alpha_i=0, 相位函数为：
            % Phi(p,q,n) = Nc2(q^2-p^2) + n(p-q-l_i) - Nc1*l_i^2
            % 对n求和：
            % sum_n exp{j*2pi/N * [Nc2(q^2-p^2) + n(p-q-l_i) - Nc1*l_i^2]}
            % = exp{j*2pi*[c2(q^2-p^2) - c1*l_i^2]} * sum_n exp{j*2pi*n*(p-q-l_i)/N}
            % 
            % 利用几何级数公式：
            % sum_{n=0}^{N-1} exp(j*2pi*n*k/N) = N * delta(k mod N)
            % 只有当 p-q-l_i = 0 (mod N) 时非零
            
            q = mod(p + l_i - 1, N) + 1;  % 使得 p - q - l_i = 0 (mod N)
            
            % 计算相位
            phase = 2*pi * (c2 * (q^2 - p^2) - c1 * l_i^2);
            Hi(p, q) = exp(1j * phase);
        end
        
    else
        % ========== alpha_i != 0 的一般情况 ==========
        % 使用POSP近似确定非零区域
        
        for p = 1:N
            % 驻点位置（论文公式(26)）
            % q_sp 使得 dθ/dn = 0
            % 数值计算中用浮点数确定中心
            q_center_float = p + l_i*(1 - 2*N*c1);
            q_center = mod(round(q_center_float) - 1, N) + 1;
            
            % 有效带宽（论文公式(34)）
            % 使用当前路径的 abs(alpha_i) 计算带宽
            q_width = ceil(N * c1 * (2*abs(alpha_i) + 1 + 2*Nv/N)) + 5;
            
            q_start = q_center - q_width;
            q_end = q_center + q_width;
            
            for q_raw = q_start:q_end
                q = mod(q_raw - 1, N) + 1;
                phase_sum = 0;
                
                % 数值积分（论文公式(19-20)）
                for n = 0:N-1
                    theta = 2*pi/N * (...
                        N * c2 * (q^2 - p^2) + ...
                        n * (p - q - l_i) + ...
                        alpha_i * n^2 + ...
                        - N * c1 * l_i^2 ...
                    );
                    phase_sum = phase_sum + exp(1j * theta);
                end
                
                Hi(p, q) = phase_sum / N;
            end
        end
    end
    
    % 阈值处理：去除数值积分的旁瓣（关键！）
    % 论文Figure 2显示，POSP近似只保留主瓣区域
    % 旁瓣幅度通常比主瓣小100倍以上（-40dB）
    threshold = 1e-2 * max(abs(Hi(:)));  % 从1e-6增加到1e-2
    if threshold > 0
        Hi(abs(Hi) < threshold) = 0;
    end
    
    % 注意：不需要额外归一化，公式(19)中已经有1/N因子
end



function x_hat = detector_ZF(y, H)
    % ZF检测：x_hat = H^{-1} * y
    % 添加小的正则化避免数值不稳定
    N = size(H, 1);
    lambda = 1e-8;  % 正则化参数
    x_hat = (H' * H + lambda * eye(N)) \ (H' * y);
end

function x_hat = detector_LMMSE(y, H, noise_var)
    % LMMSE检测：x_hat = (H^H*H + σ²I)^{-1} * H^H * y
    % 这是MMSE均衡器的标准形式
    N = size(H, 2);
    x_hat = (H' * H + noise_var * eye(N)) \ (H' * y);
end
