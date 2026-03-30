function x_hat = cnn_detector(y, H, params)
% =========================================================================
% CNN检测器 - 使用Python训练的参数
% =========================================================================
% 输入:
%   y         - 接收信号 [N × 1]
%   H         - 信道矩阵 [N × N]
%   params    - 从Python导出的参数结构体
%               包含CNN各层的权重和偏置
%
% 输出:
%   x_hat     - 估计符号 [N × 1]
%
% 算法:
%   1. 匹配滤波初始化: x_mf = H^H · y
%   2. 残差CNN修正
%   3. QPSK幅度缩放
% =========================================================================

    N = size(H, 1);
    
    % 1. 匹配滤波初始化
    H_H = H';
    x_mf = H_H * y;
    
    % 归一化
    x_mf_abs_max = max(abs(x_mf)) + 1e-8;
    x_mf_norm = x_mf / x_mf_abs_max;
    
    % 2. 转换为实数格式 [2, N] (实部和虚部)
    x_real = [real(x_mf_norm).'; imag(x_mf_norm).'];  % [2, N]
    
    % 3. 残差CNN修正
    % 注意: 这里需要实现1D卷积，MATLAB中需要手动实现
    residual = apply_refine_net(x_real, params);
    
    % 4. 残差连接
    x_refined = x_real + residual;
    
    % 5. 转回复数
    x_hat_raw = x_refined(1, :).' + 1j * x_refined(2, :).';
    
    % 6. 归一化到单位幅度，然后缩放到QPSK幅度
    x_hat_normalized = x_hat_raw ./ (abs(x_hat_raw) + 1e-8);
    scale = abs(params.scale) * 0.7071;
    x_hat = x_hat_normalized * scale;
end


function out = apply_refine_net(x, params)
% 应用残差修正网络
% 输入: x [2, N]
% 输出: out [2, N]

    % 简化实现：如果没有完整的CNN参数，使用恒等映射
    if ~isfield(params, 'refine_net_0_weight')
        out = zeros(size(x));
        return;
    end
    
    % Conv1d层实现
    % Layer 0: Conv1d(2, hidden, kernel=5, padding=2)
    h = conv1d_layer(x, params.refine_net_0_weight, params.refine_net_0_bias, 2);
    h = leaky_relu(h, 0.1);
    
    % Layer 2: Conv1d(hidden, hidden, kernel=5, padding=2)
    h = conv1d_layer(h, params.refine_net_2_weight, params.refine_net_2_bias, 2);
    h = leaky_relu(h, 0.1);
    
    % Layer 4: Conv1d(hidden, hidden, kernel=3, padding=1)
    h = conv1d_layer(h, params.refine_net_4_weight, params.refine_net_4_bias, 1);
    h = leaky_relu(h, 0.1);
    
    % Layer 6: Conv1d(hidden, 2, kernel=3, padding=1)
    out = conv1d_layer(h, params.refine_net_6_weight, params.refine_net_6_bias, 1);
end


function out = conv1d_layer(x, weight, bias, padding)
% 1D卷积层
% 输入: x [C_in, N]
% 权重: weight [C_out, C_in, kernel_size]
% 输出: out [C_out, N]

    [C_in, N] = size(x);
    [C_out, ~, K] = size(weight);
    
    % 填充
    x_padded = [zeros(C_in, padding), x, zeros(C_in, padding)];
    
    % 卷积
    out = zeros(C_out, N);
    for c_out = 1:C_out
        for c_in = 1:C_in
            kernel = squeeze(weight(c_out, c_in, :));
            out(c_out, :) = out(c_out, :) + conv(x_padded(c_in, :), flip(kernel), 'valid');
        end
        out(c_out, :) = out(c_out, :) + bias(c_out);
    end
end


function out = leaky_relu(x, alpha)
% Leaky ReLU激活函数
    out = max(x, alpha * x);
end
