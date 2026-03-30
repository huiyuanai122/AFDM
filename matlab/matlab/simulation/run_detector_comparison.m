% =========================================================================
% 检测器性能对比仿真（time-scaling 数据集 / 论文风格统计与作图）
% =========================================================================
% 对比 LMMSE / OAMP / OAMP-Net 的 BER 性能。
%
% 你现在看到的高 SNR 区域 BER=0，通常不是“真的 0”，而是统计比特数不够，
% 导致“没观察到错误”。本脚本做了两件事让结果更“论文级”：
%   1) 用统计下界 floor=0.5/bit_cnt 替换 0 错误点再作图
%   2) 用 target_ber_floor 自动决定每个 SNR 需要的最小比特数 min_bits_per_snr
%
% 运行前置条件（建议按这个顺序）：
%   1) MATLAB 生成数据: generate_dataset_timescaling_n256('test', ... , ..., 'tsv1');
%   2) Python 训练: train_oampnet_v4.py --version tsv1
%   3) Python 导出参数: python python/training/export_oampnet_v4_to_matlab.py --version tsv1
%   4) MATLAB 运行本脚本 -> 得到 BER 对比图/结果文件
%
% NOTE:
% - 测试集通常是 *_partK.mat 分片保存（-v7.3/hdf5）。本脚本采用 hyperslab 流式读取，
%   不会一次性把整个 20GB 数据集读进内存。
% =========================================================================

clear; clc; close all;

% 让脚本与“当前工作目录”无关：所有相对路径都相对于本文件所在目录
this_dir = fileparts(mfilename('fullpath'));

% 添加检测器目录到路径
addpath(fullfile(this_dir, '..', 'detectors'));

%% ======================== 配置区（你主要改这里） ========================
version = 'tsv1';            % 数据集/模型版本后缀（与生成数据/训练时一致）；若无后缀填 ''
snr_list = 0:2:20;           % 评估的 SNR 点

% 统计精度目标：希望能分辨到的 BER 量级（越小越慢）
%   1e-6：一般够写论文； 1e-7：更严谨但更耗时。
target_ber_floor = 1e-7;

% 噪声重复次数：对同一个 (H,x) 生成多次独立噪声，实现 Monte-Carlo 加速压低 BER 统计下界
% 例如 noise_reps=10 且 min_bits_per_snr=5e6 时，高 SNR 不容易再出现 "0(<floor)" 饱和
noise_reps = 10;

% 计算每个 SNR 点需要的最小比特数：0 错误时 floor = 0.5/bit_cnt <= target_ber_floor
min_bits_per_snr = ceil(0.5 / target_ber_floor);

% 安全上限：每个 SNR 最多抽多少个样本（防止无限跑太久）
% 每个样本比特数大约是 2*(N-2Q)。N=256,Q=0 时每样本 512 bit。
max_samples_per_snr = 5000;  % 建议：2000(≈1e-6)、5000(更稳)

% OAMP 设置
num_iterations_oamp = 10;    % OAMP 迭代次数（一般与 OAMPNet 层数一致）
damping_oamp = 0.9;          % OAMP 阻尼

% 噪声随机种子（保证可复现）
base_seed = 12345;

% 数据/结果目录（相对本文件目录 matlab/simulation/）
data_dir = fullfile(this_dir, '..','..','data');
results_dir = fullfile(this_dir, '..','..','results');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end

%% ======================== 载入 OAMPNet 参数 =============================
version_str = '';
if ~isempty(version)
    version_str = ['_' version];
end

params_path = fullfile(data_dir, ['oampnet_v4' version_str '_params.mat']);
if ~exist(params_path, 'file')
    error('未找到 OAMPNet 参数文件: %s\n请先运行 export_oampnet_v4_to_matlab.py --version %s', params_path, version);
end

oampnet_params = load(params_path);

fprintf('========================================\n');
fprintf('加载 OAMPNet 参数\n');
fprintf('========================================\n');
fprintf('已加载: %s\n', params_path);
if isfield(oampnet_params, 'num_layers')
    fprintf('  num_layers: %d\n', round(double(oampnet_params.num_layers(1))));
    % 可选：让 OAMP 迭代次数至少等于层数
    num_iterations_oamp = max(num_iterations_oamp, round(double(oampnet_params.num_layers(1))));
end

%% ======================== 找到测试集文件列表 ============================
base = ['afdm_n256_test' version_str];
single_file = fullfile(data_dir, [base '.mat']);

files = {};
if exist(single_file, 'file')
    files = {single_file};
else
    d = dir(fullfile(data_dir, [base '_part*.mat']));
    if isempty(d)
        error('找不到测试集文件: %s 或 %s', single_file, fullfile(data_dir,[base '_part*.mat']));
    end
    [~, idx] = sort({d.name});
    d = d(idx);
    for k = 1:numel(d)
        files{end+1} = fullfile(d(k).folder, d(k).name); %#ok<SAGROW>
    end
end

fprintf('\n========================================\n');
fprintf('测试集文件\n');
fprintf('========================================\n');
fprintf('共 %d 个文件:\n', numel(files));
for k = 1:numel(files)
    fprintf('  [%d/%d] %s\n', k, numel(files), get_filename_only(files{k}));
end

%% ======================== 读取 system_params ============================
first_file = files{1};
N = read_scalar_h5(first_file, '/system_params/N');
Q = read_scalar_h5(first_file, '/system_params/Q');
N_eff = read_scalar_h5(first_file, '/system_params/N_eff');

if isempty(N) || N <= 0
    error('无法从 system_params 读取 N');
end
if isempty(Q)
    Q = 0;
end

fprintf('\n========================================\n');
fprintf('system_params\n');
fprintf('========================================\n');
fprintf('N=%d, Q=%d, N_eff=%d\n', N, Q, N_eff);

% 有效数据索引（只统计有效区）
if Q > 0
    data_idx = (Q+1):(N-Q);
else
    data_idx = 1:N;
end
bits_per_sample = 2 * numel(data_idx); % QPSK: 每符号 2bit

fprintf('\n========================================\n');
fprintf('统计配置\n');
fprintf('========================================\n');
fprintf('target_ber_floor=%.1e -> min_bits_per_snr=%d\n', target_ber_floor, min_bits_per_snr);
fprintf('max_samples_per_snr=%d, bits_per_sample=%d, noise_reps=%d (bits/unique=%.0f)\n', max_samples_per_snr, bits_per_sample, noise_reps, bits_per_sample*noise_reps);

%% ======================== 统计 BER（流式读取） ==========================
num_snr = numel(snr_list);
err_lmmse   = zeros(1, num_snr);
err_oamp    = zeros(1, num_snr);
err_oampnet = zeros(1, num_snr);
bit_cnt     = zeros(1, num_snr);
used_cnt    = zeros(1, num_snr);

fprintf('\n========================================\n');
fprintf('开始 BER 仿真（流式读取）\n');
fprintf('========================================\n');

for fidx = 1:numel(files)
    fpath = files{fidx};
    snr_vec = h5read(fpath, '/snr_dataset');
    snr_vec = double(snr_vec(:));

    S = numel(snr_vec);
    fprintf('\n[%d/%d] %s, samples=%d\n', fidx, numel(files), get_filename_only(fpath), S);

    % 读取数据集维度信息（用于 hyperslab 读取）
    H_info = h5info(fpath, '/H_dataset');
    H_size = H_info.Dataspace.Size;
    x_info = h5info(fpath, '/x_dataset');
    x_size = x_info.Dataspace.Size;

    for sidx = 1:S
        snr_db = snr_vec(sidx);
        snr_pos = find(snr_list == snr_db, 1);
        if isempty(snr_pos)
            continue;
        end

        % 这个 SNR 已达到目标比特数 或 已达到样本上限 -> 跳过
        if bit_cnt(snr_pos) >= min_bits_per_snr
            continue;
        end
        if ~isinf(max_samples_per_snr) && used_cnt(snr_pos) >= max_samples_per_snr
            continue;
        end

        % --- 读取第 sidx 个样本（hyperslab） ---
        H = read_complex_slice(fpath, '/H_dataset', H_size, sidx);
        x_true = read_complex_slice(fpath, '/x_dataset', x_size, sidx);
        x_true = x_true(:);

        noise_var = 1.0 / (10^(snr_db/10));

        % --- Monte-Carlo：对同一个 (H,x) 重复生成多次独立噪声 ---
        reps_done = 0;
        for rep = 1:noise_reps
            if bit_cnt(snr_pos) >= min_bits_per_snr
                break;
            end

            rng(base_seed + 100000*fidx + 1000*sidx + rep);
            sigma = sqrt(noise_var/2);
            w = sigma * (randn(N,1) + 1j*randn(N,1));
            y = H * x_true + w;

            % --- 检测 ---
            x_lmmse   = lmmse_detector(y, H, noise_var);
            x_oamp    = oamp_detector(y, H, noise_var, num_iterations_oamp, damping_oamp, Q);
            x_oampnet = oampnet_detector(y, H, noise_var, oampnet_params, Q);

            % --- 统计误比特（只统计有效区） ---
            err_lmmse(snr_pos)   = err_lmmse(snr_pos)   + count_bit_errors(x_lmmse(data_idx),   x_true(data_idx));
            err_oamp(snr_pos)    = err_oamp(snr_pos)    + count_bit_errors(x_oamp(data_idx),    x_true(data_idx));
            err_oampnet(snr_pos) = err_oampnet(snr_pos) + count_bit_errors(x_oampnet(data_idx), x_true(data_idx));
            bit_cnt(snr_pos)     = bit_cnt(snr_pos) + bits_per_sample;

            reps_done = reps_done + 1;
        end

        % used_cnt 统计“使用了多少个唯一(H,x)样本”，即使 rep>1 也只加 1
        if reps_done > 0
            used_cnt(snr_pos) = used_cnt(snr_pos) + 1;
        end

        % 所有 SNR 都达到目标比特数 -> 提前结束读取
        if all(bit_cnt >= min_bits_per_snr)
            break;
        end
    end

    if all(bit_cnt >= min_bits_per_snr)
        fprintf('所有 SNR 已达到 min_bits_per_snr=%d（target_ber_floor=%.1e），提前结束读取。\n', min_bits_per_snr, target_ber_floor);
        break;
    end
end

%% ======================== 汇总结果 =====================================
results = struct();
results.version = version;
results.snr = snr_list;
results.used_cnt = used_cnt;
results.bit_cnt = bit_cnt;
results.err_lmmse = err_lmmse;
results.err_oamp = err_oamp;
results.err_oampnet = err_oampnet;
results.lmmse = err_lmmse ./ max(bit_cnt, 1);
results.oamp  = err_oamp  ./ max(bit_cnt, 1);
results.oampnet = err_oampnet ./ max(bit_cnt, 1);
results.ber_floor = 0.5 ./ max(bit_cnt, 1);  % 0 错误 -> BER < 0.5/bit_cnt
results.target_ber_floor = target_ber_floor;
results.min_bits_per_snr = min_bits_per_snr;
results.max_samples_per_snr = max_samples_per_snr;
results.bits_per_sample = bits_per_sample;
results.noise_reps = noise_reps;
results.bits_per_unique_sample = bits_per_sample * noise_reps;

fprintf('\n========================================\n');
fprintf('结果汇总（每点使用的样本数/比特数/统计下界）\n');
fprintf('========================================\n');
for i = 1:num_snr
    floor_i = results.ber_floor(i);
    if results.err_lmmse(i) == 0
        lmmse_str = sprintf('0 ( < %.2e )', floor_i);
    else
        lmmse_str = sprintf('%.3e', results.lmmse(i));
    end
    if results.err_oamp(i) == 0
        oamp_str = sprintf('0 ( < %.2e )', floor_i);
    else
        oamp_str = sprintf('%.3e', results.oamp(i));
    end
    if results.err_oampnet(i) == 0
        oampnet_str = sprintf('0 ( < %.2e )', floor_i);
    else
        oampnet_str = sprintf('%.3e', results.oampnet(i));
    end

    fprintf('SNR=%2ddB: used=%d, bits=%d, floor=%.2e | LMMSE=%s, OAMP=%s, OAMPNet=%s\n', ...
        snr_list(i), used_cnt(i), bit_cnt(i), floor_i, lmmse_str, oamp_str, oampnet_str);
end

%% ======================== 画图（通信原理半对数） =========================
figure('Position', [100, 100, 920, 680]);

ber_lmmse = results.lmmse;
ber_oamp  = results.oamp;
ber_oampnet = results.oampnet;

% 用统计下界替换 0（而不是强行夹到常数）
floor_vec = results.ber_floor;
ber_lmmse(ber_lmmse==0) = floor_vec(ber_lmmse==0);
ber_oamp(ber_oamp==0)   = floor_vec(ber_oamp==0);
ber_oampnet(ber_oampnet==0) = floor_vec(ber_oampnet==0);

semilogy(snr_list, ber_lmmse, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogy(snr_list, ber_oamp,  'g-s', 'LineWidth', 2, 'MarkerSize', 8);
semilogy(snr_list, ber_oampnet,'r-^', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('SNR (dB)', 'FontSize', 14);
ylabel('BER', 'FontSize', 14);
title(sprintf('AFDM Detector Comparison (MATLAB, %s)', version), 'FontSize', 16);
legend('LMMSE', 'OAMP', 'OAMP-Net', 'Location', 'southwest', 'FontSize', 12);

grid on;
set(gca, 'YScale', 'log');
set(gca, 'YMinorGrid', 'on');
set(gca, 'XMinorGrid', 'on');
xlim([min(snr_list), max(snr_list)]);

% y 轴下界按统计下界自动设置（略放宽，避免贴边）
ymin = min(floor_vec(floor_vec>0));
if isempty(ymin) || ~isfinite(ymin)
    ymin = target_ber_floor;
else
    ymin = ymin/2;
end
ylim([max(ymin, 1e-12), 1]);

% 根据 ymin 自动生成 10^k 刻度
kmin = floor(log10(max(ymin, 1e-12)));
yticks(10.^(kmin:0));

% 标注统计信息（便于你写论文解释“0错”）
annotation('textbox', [0.13 0.80 0.40 0.12], 'String', ...
    {sprintf('target floor=%.1e', target_ber_floor), ...
     sprintf('min bits/SNR=%d', min_bits_per_snr), ...
     sprintf('noise reps=%d', noise_reps), ...
     sprintf('min observed floor=%.2e', min(floor_vec))}, ...
    'FitBoxToText','on', 'BackgroundColor','w');

hold off;

png_path = fullfile(results_dir, sprintf('ber_comparison_matlab_%s.png', version));
fig_path = fullfile(results_dir, sprintf('ber_comparison_matlab_%s.fig', version));
saveas(gcf, png_path);
saveas(gcf, fig_path);
fprintf('\n图表已保存: %s\n', png_path);

mat_path = fullfile(results_dir, sprintf('matlab_ber_results_%s.mat', version));
save(mat_path, 'results');
fprintf('结果已保存: %s\n', mat_path);

csv_path = fullfile(results_dir, sprintf('ber_results_matlab_%s.csv', version));
fid = fopen(csv_path, 'w');
fprintf(fid, 'SNR,used_cnt,bit_cnt,ber_floor,LMMSE,OAMP,OAMPNet\n');
for i = 1:numel(snr_list)
    fprintf(fid, '%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n', ...
        snr_list(i), used_cnt(i), bit_cnt(i), results.ber_floor(i), results.lmmse(i), results.oamp(i), results.oampnet(i));
end
fclose(fid);
fprintf('CSV已保存: %s\n', csv_path);

fprintf('\n仿真完成!\n');

%% ======================== 辅助函数 ======================================

function v = read_scalar_h5(file, path)
    try
        a = h5read(file, path);
        a = double(a);
        v = a(1);
    catch
        v = [];
    end
end

function A = read_complex_slice(file, dataset_path, dsize, sample_idx)
    % 读取 v7.3/hdf5 复数结构体（real/imag）的一条样本
    if numel(dsize) == 3
        start = [1, 1, sample_idx];
        count = [dsize(1), dsize(2), 1];
    elseif numel(dsize) == 2
        start = [1, sample_idx];
        count = [dsize(1), 1];
    else
        error('不支持的数据维度: [%s]', num2str(dsize));
    end

    tmp = h5read(file, dataset_path, start, count);

    if isstruct(tmp)
        A = double(tmp.real) + 1j * double(tmp.imag);
    else
        A = double(tmp);
    end

    A = squeeze(A);
end

function name = get_filename_only(path)
    [~, name, ext] = fileparts(path);
    name = [name ext];
end

function errors = count_bit_errors(x_hat, x_true)
    % 适用于 QPSK: x \in {(\pm1 \pm j)/sqrt(2)}
    xr = sign(real(x_hat)); xi = sign(imag(x_hat));
    xr(xr == 0) = 1; xi(xi == 0) = 1;
    x_dec = (xr + 1j*xi) / sqrt(2);

    bit_err_real = sum(sign(real(x_dec)) ~= sign(real(x_true)));
    bit_err_imag = sum(sign(imag(x_dec)) ~= sign(imag(x_true)));
    errors = bit_err_real + bit_err_imag;
end
