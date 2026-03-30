function x_hat = correct_oamp_detector(y, H, noise_var, num_iterations, Q, use_soft_decision, temperature, damping)
% =========================================================================
% 修复后的OAMP检测器 - 正交近似消息传递算法
% =========================================================================
% 关键改进（相比原实现）：
% 1. 添加Onsager校正项 - OAMP区别于普通迭代的关键
% 2. 实现方差传递机制 - NLE自适应调整
% 3. 使用MMSE软判决 - 利用概率信息
%
% 输入:
%   y                 - 接收信号 [N × 1]
%   H                 - 信道矩阵 [N × N]
%   noise_var         - 噪声方差 (标量)
%   num_iterations    - 迭代次数 (默认: 10)
%   Q                 - 保护子载波数 (默认: 0)
%   use_soft_decision - 是否使用软判决 (默认: true)
%   temperature       - 软判决温度参数 (默认: 0.5)
%   damping           - 阻尼系数 (默认: 0.8)
%
% 输出:
%   x_hat             - 估计符号 [N × 1]
%
% 参考论文: "Model-Driven Deep Learning for MIMO Detection"
% =========================================================================

    % 默认参数
    if nargin < 4 || isempty(num_iterations)
        num_iterations = 10;
    end
    if nargin < 5 || isempty(Q)
        Q = 0;
    end
    if nargin < 6 || isempty(use_soft_decision)
        use_soft_decision = true;
    end
    if nargin < 7 || isempty(temperature)
        temperature = 0.5;
    end
    if nargin < 8 || isempty(damping)
        damping = 0.8;
    end
    
    N = size(H, 1);
    H_H = H';
    
    % 计算 H^H · H
    HH = H_H * H;
    
    % 计算LMMSE滤波器 W = (H^H·H + σ²I)^{-1}·H^H
    W = (HH + noise_var * eye(N)) \ H_H;
    
    % 计算Onsager校正所需的trace项
    WH = W * H;
    trace_WH = real(trace(WH)) / N;
    
    % 初始化
    x = zeros(N, 1);  % 零初始化
    v = 1;  % 符号方差（QPSK为1）
    
    % 有效数据索引
    if Q > 0
        data_idx = (Q+1):(N-Q);
    else
        data_idx = 1:N;
    end
    
    % OAMP迭代
    for t = 1:num_iterations
        x_prev = x;
        v_prev = v;
        
        % ===== 1. 线性估计器 (LE) =====
        residual = y - H * x;
        r = x + W * residual;
        
        % ===== 2. 方差传递 =====
        tau = (1 - trace_WH) * v_prev + noise_var;
        tau = max(tau, 1e-10);  % 防止除零
        
        % ===== 3. 非线性估计器 (NLE) =====
        if use_soft_decision
            % MMSE软判决
            [x_nle, v_nle] = mmse_qpsk_estimator(r(data_idx), tau, temperature);
            x_full = x;
            x_full(data_idx) = x_nle;
            x_nle = x_full;
        else
            % 硬判决
            x_nle = x;
            x_nle(data_idx) = qpsk_hard_projection(r(data_idx));
            v_nle = 0;
        end
        
        % ===== 4. Onsager校正 =====
        if t > 1
            onsager_coef = v_nle / (tau + 1e-10);
            onsager_correction = onsager_coef * (r - x_prev);
            x_corrected = x_nle - 0.1 * onsager_correction;
        else
            x_corrected = x_nle;
        end
        
        % ===== 5. 阻尼混合 =====
        x = damping * x_corrected + (1 - damping) * x_prev;
        v = damping * v_nle + (1 - damping) * v_prev;
        
        % 保护子载波置零
        if Q > 0
            x(1:Q) = 0;
            x(N-Q+1:N) = 0;
        end
    end
    
    x_hat = x;
end


function [x_mmse, v_mmse] = mmse_qpsk_estimator(r, tau, temperature)
% QPSK MMSE估计器
% 计算 E[x | r, τ] 和 Var[x | r, τ]

    % QPSK星座点
    qpsk_points = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2);
    
    N = length(r);
    x_mmse = zeros(N, 1);
    
    effective_tau = tau * temperature;
    
    for i = 1:N
        % 计算到各星座点的距离平方
        dist_sq = abs(r(i) - qpsk_points).^2;
        
        % Log概率（数值稳定）
        log_prob = -dist_sq / (effective_tau + 1e-10);
        log_prob = log_prob - max(log_prob);
        
        % Softmax得到后验概率
        prob = exp(log_prob);
        prob = prob / sum(prob);
        
        % MMSE估计
        x_mmse(i) = sum(prob .* qpsk_points);
    end
    
    % 后验方差
    v_mmse = 1 - mean(abs(x_mmse).^2);
    v_mmse = max(min(v_mmse, 1), 1e-10);
end


function x_proj = qpsk_hard_projection(r)
% QPSK硬判决投影

    real_part = sign(real(r));
    imag_part = sign(imag(r));
    
    % 处理零值
    real_part(real_part == 0) = 1;
    imag_part(imag_part == 0) = 1;
    
    x_proj = (real_part + 1j * imag_part) / sqrt(2);
end
