% =========================================================================
% 检测器验证脚本 - 验证各检测器实现的正确性
% =========================================================================
clear; clc; close all;

addpath('../detectors');

%% ==================== 测试1: 无噪声情况 ====================
fprintf('========================================\n');
fprintf('测试1: 无噪声情况 (应该完美恢复)\n');
fprintf('========================================\n');

N = 64;
Q = 5;

% 生成简单的对角占优信道
H = eye(N) + 0.1 * (randn(N) + 1j*randn(N)) / sqrt(N);

% 生成QPSK符号
x = zeros(N, 1);
for i = 1:(N-2*Q)
    bit1 = randi([0,1]);
    bit2 = randi([0,1]);
    x(Q+i) = ((1-2*bit1) + 1j*(1-2*bit2)) / sqrt(2);
end

% 无噪声接收
y = H * x;
noise_var = 1e-10;

% 测试各检测器
x_lmmse = lmmse_detector(y, H, noise_var);
x_oamp = oamp_detector(y, H, noise_var, 10, true, Q);  % 硬判决，传入Q

% 计算MSE（只计算有效子载波）
mse_lmmse = mean(abs(x_lmmse(Q+1:N-Q) - x(Q+1:N-Q)).^2);
mse_oamp = mean(abs(x_oamp(Q+1:N-Q) - x(Q+1:N-Q)).^2);

fprintf('LMMSE MSE (有效子载波): %.2e (应该接近0)\n', mse_lmmse);
fprintf('OAMP MSE (有效子载波):  %.2e (应该接近0)\n', mse_oamp);

% 调试：检查LMMSE输出是否已经正确
x_lmmse_hard = qpsk_hard(x_lmmse);
x_true_hard = qpsk_hard(x);
errors_lmmse = sum(x_lmmse_hard(Q+1:N-Q) ~= x_true_hard(Q+1:N-Q));
fprintf('LMMSE硬判决符号错误数: %d / %d\n', errors_lmmse, N-2*Q);

% 检查OAMP输出
errors_oamp = sum(x_oamp(Q+1:N-Q) ~= x(Q+1:N-Q));
fprintf('OAMP符号错误数: %d / %d\n', errors_oamp, N-2*Q);

%% ==================== 测试2: 不同SNR下的性能 ====================
fprintf('\n========================================\n');
fprintf('测试2: 不同SNR下的性能对比\n');
fprintf('========================================\n');

SNR_test = [0, 10, 20, 30];
num_trials = 100;

for snr_db = SNR_test
    noise_var = 1 / (10^(snr_db/10));
    
    errors_lmmse = 0;
    errors_oamp = 0;
    total_symbols = 0;
    
    for trial = 1:num_trials
        H = eye(N) + 0.3 * (randn(N) + 1j*randn(N)) / sqrt(N);
        
        x = zeros(N, 1);
        for i = 1:(N-2*Q)
            x(Q+i) = ((1-2*randi([0,1])) + 1j*(1-2*randi([0,1]))) / sqrt(2);
        end
        
        sigma = sqrt(noise_var / 2);
        w = sigma * (randn(N, 1) + 1j * randn(N, 1));
        y = H * x + w;
        
        x_lmmse = lmmse_detector(y, H, noise_var);
        x_oamp = oamp_detector(y, H, noise_var, 10, true, Q);  % 硬判决，传入Q
        
        % 硬判决比较
        x_lmmse_hard = qpsk_hard(x_lmmse);
        x_oamp_hard = qpsk_hard(x_oamp);  % OAMP已经是硬判决输出
        x_true_hard = qpsk_hard(x);
        
        errors_lmmse = errors_lmmse + sum(x_lmmse_hard(Q+1:N-Q) ~= x_true_hard(Q+1:N-Q));
        errors_oamp = errors_oamp + sum(x_oamp(Q+1:N-Q) ~= x_true_hard(Q+1:N-Q));
        total_symbols = total_symbols + (N - 2*Q);
    end
    
    ser_lmmse = errors_lmmse / total_symbols;
    ser_oamp = errors_oamp / total_symbols;
    
    fprintf('SNR=%2d dB | LMMSE SER: %.4e | OAMP SER: %.4e | 差异: %.2fx\n', ...
        snr_db, ser_lmmse, ser_oamp, ser_oamp/max(ser_lmmse, 1e-10));
end

%% ==================== 测试3: OAMP迭代收敛性 ====================
fprintf('\n========================================\n');
fprintf('测试3: OAMP迭代收敛性\n');
fprintf('========================================\n');

snr_db = 15;
noise_var = 1 / (10^(snr_db/10));

H = eye(N) + 0.3 * (randn(N) + 1j*randn(N)) / sqrt(N);
x = zeros(N, 1);
for i = 1:(N-2*Q)
    x(Q+i) = ((1-2*randi([0,1])) + 1j*(1-2*randi([0,1]))) / sqrt(2);
end

sigma = sqrt(noise_var / 2);
w = sigma * (randn(N, 1) + 1j * randn(N, 1));
y = H * x + w;

fprintf('迭代次数 | MSE (有效子载波)\n');
fprintf('---------|------------------\n');
for num_iter = [1, 2, 5, 10, 20]
    x_oamp = oamp_detector(y, H, noise_var, num_iter, true, Q);
    mse = mean(abs(x_oamp(Q+1:N-Q) - x(Q+1:N-Q)).^2);
    fprintf('%8d | %.4e\n', num_iter, mse);
end

fprintf('\n验证完成！\n');

%% ==================== 辅助函数 ====================
function x_hard = qpsk_hard(x)
    x_hard = sign(real(x)) + 1j * sign(imag(x));
    x_hard = x_hard / sqrt(2);
end
