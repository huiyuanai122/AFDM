function x_hat = oampnet_detector(y, H, noise_var, params, Q)
% =========================================================================
% OAMPNet检测器 - 基于深度展开的OAMP网络
% =========================================================================
% 与Python版本(oampnet_v4.py)同步的MATLAB实现
% 使用从Python训练导出的参数
%
% 输入:
%   y         - 接收信号 [N × 1]
%   H         - 信道矩阵 [N × N]
%   noise_var - 噪声方差 (标量)
%   params    - 从Python导出的参数结构体，包含:
%               .gamma       - 步长参数 [num_layers × 1]
%               .damping     - 阻尼系数 [num_layers × 1]
%               .temperature - 温度参数 [num_layers × 1]
%               .num_layers  - 层数
%   Q         - 保护子载波数 (默认: 0)
%
% 输出:
%   x_hat     - 估计符号 [N × 1]
%
% 算法特点（与Python OAMPNetV4一致）:
%   1. 可学习的步长参数gamma
%   2. 可学习的阻尼系数
%   3. 可学习的NLE温度参数
%   4. 使用固定noise_var作为正则化项
%   5. 正确的方差传递公式
% =========================================================================

    if nargin < 5 || isempty(Q)
        Q = 0;
    end
    
    N = size(H, 1);
    num_layers = params.num_layers;
    
    H_H = H';
    
    % ===== 预计算 =====
    HH = H_H * H;
    W = (HH + noise_var * eye(N)) \ H_H;
    
    % 预计算 WH = W*H（用于向量tau的误差传播估计）
    WH = W * H;

    % alpha = trace(WH)/N，用于线性模块的“去相关/去偏置”归一化
    % 说明：DAFT/DAF 域下 H 通常不满足 i.i.d. 假设，直接用 W 更新会有缩放偏差。
    alpha = real(trace(WH)) / N;
    alpha = max(alpha, 1e-6);
    
    % ===== 初始化 =====
    x = zeros(N, 1);
    v = 1.0;
    
    % ===== OAMPNet迭代 =====
    for t = 1:num_layers
        x_prev = x;
        v_prev = v;
        
        % ----- 1. 线性估计器 (带可学习步长) -----
        residual = y - H * x;
        gamma_t = params.gamma(t);
        step = gamma_t / alpha;
        r = x + step * (W * residual);
        
        % ----- 2. 方差传递（向量 tau，更贴 DAFT/DAF 域的非均匀耦合结构）-----
        % 近似假设:
        %   e = x - x_true ~ CN(0, v I),  w ~ CN(0, noise_var I)
        %   r - x_true = (I - gamma W H) e + (gamma W) w
        % 则逐符号等效方差:
        %   tau_i = v * ||row_i(I - gamma W H)||^2 + noise_var * ||row_i(gamma W)||^2
        A = eye(N) - step * WH;             % [N,N]
        Bmat = step * W;                    % [N,N]
        row_energy_A = sum(abs(A).^2, 2);   % [N,1]
        row_energy_B = sum(abs(Bmat).^2, 2);
        tau_vec = v_prev * row_energy_A + noise_var * row_energy_B;
        tau_vec = max(tau_vec, 1e-10);
        
        % ----- 3. 非线性估计器 (带可学习温度) -----
        temp_t = params.temperature(t);
        x_nle = qpsk_mmse_estimator(r, tau_vec * temp_t);

        % 后验方差（逐符号）: QPSK下E|x|^2=1
        v_sym = 1 - abs(x_nle).^2;
        v_sym = max(min(real(v_sym), 1.0), 1e-10);
        if Q > 0
            v_nle = mean(v_sym(Q+1:N-Q));
        else
            v_nle = mean(v_sym);
        end
        
        % ----- 4. 阻尼更新 (可学习阻尼系数) -----
        damping_t = params.damping(t);
        x = damping_t * x_nle + (1 - damping_t) * x_prev;
        v = damping_t * v_nle + (1 - damping_t) * v_prev;
        
        % ----- 5. 保护子载波置零 -----
        if Q > 0
            x(1:Q) = 0;
            x(N-Q+1:N) = 0;
        end
    end
    
    x_hat = x;
end


function x_mmse = qpsk_mmse_estimator(r, tau)
% QPSK MMSE估计器
% 输入:
%   r   - 线性估计器输出 [N × 1]
%   tau - 有效噪声方差 (可以是标量或 [N×1] 向量，已乘以温度)
%
% 输出:
%   x_mmse - MMSE估计 [N × 1]
%
% 说明:
%   为了与Python端一致并提高速度，这里采用向量化实现。

    % QPSK星座点
    qpsk_points = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2);
    
    N = length(r);

    % 防止除零 + 保证向量形状
    if isscalar(tau)
        tau_vec = max(tau, 1e-10) * ones(N, 1);
    else
        tau_vec = max(tau(:), 1e-10);
    end

    % 计算距离平方: [N,4]
    dist_sq = abs(r - qpsk_points).^2;  % MATLAB 自动broadcast: r(N,1) - points(1,4)

    % logits = -dist_sq ./ tau
    logits = -dist_sq ./ tau_vec;
    logits = logits - max(logits, [], 2);
    prob = exp(logits);
    prob = prob ./ sum(prob, 2);

    x_mmse = prob * qpsk_points.';
end
