function x_hat = oamp_detector(y, H, noise_var, num_iterations, damping, Q)
% =========================================================================
% OAMP检测器 - 正交近似消息传递算法
% =========================================================================
% 与Python版本(oamp_final.py)同步的MATLAB实现
%
% 输入:
%   y                 - 接收信号 [N × 1]
%   H                 - 信道矩阵 [N × N]
%   noise_var         - 噪声方差 (标量)
%   num_iterations    - 迭代次数 (默认: 10)
%   damping           - 阻尼系数 (默认: 0.9)
%   Q                 - 保护子载波数 (默认: 0)
%
% 输出:
%   x_hat             - 估计符号 [N × 1]
%
% 算法特点（与Python版本一致）:
%   1. 使用固定noise_var作为正则化项
%   2. 正确的方差传递公式: tau = (1 - trace(WH)/N) * v + noise_var
%   3. 不使用Onsager校正（对我们的信道效果更好）
%   4. 使用阻尼更新稳定收敛
%   5. 早停机制选择最佳结果
% =========================================================================

    if nargin < 3 || isempty(noise_var)
        noise_var = 0.1;
    end
    if nargin < 4 || isempty(num_iterations)
        num_iterations = 10;
    end
    if nargin < 5 || isempty(damping)
        damping = 0.9;
    end
    if nargin < 6 || isempty(Q)
        Q = 0;
    end
    
    N = size(H, 1);
    H_H = H';
    
    % ===== 预计算 =====
    % 计算 W = (H^H·H + σ²·I)^{-1} · H^H (LMMSE滤波器)
    HH = H_H * H;
    W = (HH + noise_var * eye(N)) \ H_H;
    
    % 计算 trace(WH)/N
    WH = W * H;
    trace_WH = real(trace(WH)) / N;
    
    % ===== 初始化 =====
    x = zeros(N, 1);  % 从零开始（与Python一致）
    v = 1.0;          % 信号方差
    
    % 记录最佳结果
    best_x = x;
    best_mse = inf;
    
    % ===== OAMP迭代 =====
    for t = 1:num_iterations
        x_prev = x;
        v_prev = v;
        
        % ----- 1. 线性估计器 -----
        residual = y - H * x;
        r = x + W * residual;
        
        % ----- 2. 方差传递 -----
        tau = (1 - trace_WH) * v_prev + noise_var;
        tau = max(tau, 1e-10);
        
        % ----- 3. 非线性估计器 (MMSE) -----
        [x_nle, v_nle] = qpsk_mmse_estimator(r, tau);
        
        % ----- 4. 阻尼更新（不使用Onsager校正）-----
        x = damping * x_nle + (1 - damping) * x_prev;
        v = damping * v_nle + (1 - damping) * v_prev;
        
        % ----- 5. 保护子载波置零 -----
        if Q > 0
            x(1:Q) = 0;
            x(N-Q+1:N) = 0;
        end
        
        % ----- 6. 早停：记录最佳结果 -----
        x_hard = qpsk_hard_projection(x);
        mse = mean(abs(y - H * x_hard).^2);
        if mse < best_mse
            best_mse = mse;
            best_x = x;
        end
    end
    
    x_hat = best_x;
end


function [x_mmse, v_mmse] = qpsk_mmse_estimator(r, tau)
% QPSK MMSE估计器（与Python版本一致）
% 输入:
%   r   - 线性估计器输出 [N × 1]
%   tau - 等效噪声方差 (标量)
%
% 输出:
%   x_mmse - MMSE估计 [N × 1]
%   v_mmse - 后验方差 (标量)

    % QPSK星座点
    qpsk_points = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2);
    
    N = length(r);
    x_mmse = zeros(N, 1);
    
    % 防止除零
    tau = max(tau, 1e-10);
    
    for i = 1:N
        % 计算到各星座点的距离平方
        dist_sq = abs(r(i) - qpsk_points).^2;
        
        % 后验概率 p(s|r) ∝ exp(-|r-s|²/tau)
        log_prob = -dist_sq / tau;
        log_prob = log_prob - max(log_prob);  % 数值稳定
        prob = exp(log_prob);
        prob = prob / sum(prob);
        
        % MMSE估计 = E[s|r]
        x_mmse(i) = sum(prob .* qpsk_points);
    end
    
    % 后验方差
    v_mmse = 1 - mean(abs(x_mmse).^2);
    v_mmse = max(min(v_mmse, 1.0), 1e-10);
end


function x_proj = qpsk_hard_projection(r)
% QPSK硬判决投影

    real_part = sign(real(r));
    imag_part = sign(imag(r));
    
    % 处理零值情况
    real_part(real_part == 0) = 1;
    imag_part(imag_part == 0) = 1;
    
    x_proj = (real_part + 1j * imag_part) / sqrt(2);
end
