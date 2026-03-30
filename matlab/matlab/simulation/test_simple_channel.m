%% 最简单的信道测试 - 单径零多普勒
% 这是最理想的情况，应该完全没有ICI
clear; clc; close all;

fprintf('========================================\n');
fprintf('最简单信道测试 (l=0, alpha=0)\n');
fprintf('========================================\n\n');

%% 参数设置
N = 1024;
df = 4;
fc = 6e3;

alpha_max_paper = 1e-4;
tau_max = 25e-3;

fd_max = alpha_max_paper * fc;
alpha_max = ceil(fd_max / df);
l_max = ceil(tau_max * df * N);

Nv = 2;
c1 = (2*alpha_max + 1 + 2*Nv/N) / (2*N);
c2 = 1 / (2*N);

L_spread = 2*N*c1*(2*alpha_max + 1 + 2*Nv/N);
Q = ceil(L_spread / 2) + 5;
N_eff = N - 2*Q;

fprintf('===== 系统参数 =====\n');
fprintf('N = %d, N_eff = %d, Q = %d\n', N, N_eff, Q);
fprintf('c1 = %.8f, c2 = %.8f\n', c1, c2);
fprintf('alpha_max = %d, l_max = %d\n\n', alpha_max, l_max);

%% 测试1：理想单径信道 (l=0, alpha=0)
fprintf('===== 测试1：理想单径 (l=0, alpha=0) =====\n');

% 生成信道
Hi = build_Hi_correct(N, c1, c2, 0, 0);

fprintf('信道矩阵属性:\n');
fprintf('  非零元素: %d / %d (%.2f%%)\n', nnz(Hi), N^2, 100*nnz(Hi)/N^2);
fprintf('  Frobenius范数: %.6f\n', norm(Hi, 'fro'));
fprintf('  是否是单位矩阵: %d\n', norm(Hi - eye(N), 'fro') < 1e-10);

% 发射信号
mod_order = 4;
data = randi([0, mod_order-1], N_eff, 1);
data_mod = qammod(data, mod_order, 'UnitAveragePower', true);

x = zeros(N, 1);
x(Q+1:N-Q) = data_mod;
x = x * sqrt(N / N_eff);

fprintf('\n发射信号:\n');
fprintf('  全局功率: %.6f\n', mean(abs(x).^2));

% 无噪声传输
y = Hi * x;

fprintf('  接收功率: %.6f\n', mean(abs(y).^2));
fprintf('  功率偏差: %.4f%%\n', 100*abs(mean(abs(y).^2) - 1));

% 完美检测
x_hat = y;  % 单位矩阵，直接解调
x_hat_eff = x_hat(Q+1:N-Q);
demod = qamdemod(x_hat_eff, mod_order, 'UnitAveragePower', true);
ber = mean(demod ~= data);

fprintf('  无噪声BER: %.4e (应该是0)\n', ber);

if ber == 0
    fprintf('  ✓ 理想信道测试通过！\n');
else
    fprintf('  ✗ 理想信道测试失败！\n');
end
fprintf('\n');

%% 测试2：单径非零延迟 (l=10, alpha=0)
fprintf('===== 测试2：单径非零延迟 (l=10, alpha=0) =====\n');

l_test = 10;
Hi2 = build_Hi_correct(N, c1, c2, l_test, 0);

fprintf('信道矩阵属性:\n');
fprintf('  非零元素: %d / %d\n', nnz(Hi2), N^2);
fprintf('  Frobenius范数: %.6f\n', norm(Hi2, 'fro'));

% 检查是否是循环移位矩阵
is_shift = sum(sum(abs(Hi2), 2) > 0) == N && sum(sum(abs(Hi2), 1) > 0) == N;
fprintf('  是否是循环移位: %d\n', is_shift);

% 传输测试
data2 = randi([0, mod_order-1], N_eff, 1);
data_mod2 = qammod(data2, mod_order, 'UnitAveragePower', true);
x2 = zeros(N, 1);
x2(Q+1:N-Q) = data_mod2;
x2 = x2 * sqrt(N / N_eff);

y2 = Hi2 * x2;

fprintf('\n传输测试:\n');
fprintf('  发射功率: %.6f\n', mean(abs(x2).^2));
fprintf('  接收功率: %.6f\n', mean(abs(y2).^2));

% LMMSE检测
x_hat2 = Hi2' * y2;  % 匹配滤波（无噪声时等价于最优检测）
x_hat2_eff = x_hat2(Q+1:N-Q);
demod2 = qamdemod(x_hat2_eff, mod_order, 'UnitAveragePower', true);
ber2 = mean(demod2 ~= data2);

fprintf('  无噪声BER: %.4e\n', ber2);

if ber2 < 1e-4
    fprintf('  ✓ 单径延迟测试通过！\n');
else
    fprintf('  ✗ 单径延迟测试失败！\n');
end
fprintf('\n');

%% 测试3：单径非零多普勒 (l=0, alpha=1)
fprintf('===== 测试3：单径非零多普勒 (l=0, alpha=1) =====\n');

alpha_test = 1;
Hi3 = build_Hi_correct(N, c1, c2, 0, alpha_test);

fprintf('信道矩阵属性:\n');
fprintf('  非零元素: %d / %d\n', nnz(Hi3), N^2);
fprintf('  Frobenius范数: %.6f\n', norm(Hi3, 'fro'));

% 检查每行的支撑宽度
support_widths = zeros(N, 1);
for k = 1:N
    support_widths(k) = sum(abs(Hi3(k,:)) > 0);
end
fprintf('  平均支撑宽度: %.2f 个子载波\n', mean(support_widths));
fprintf('  最大支撑宽度: %d 个子载波\n', max(support_widths));
fprintf('  理论带宽: %.2f 个子载波\n', 2*N*c1*(2*abs(alpha_test) + 1 + 2*Nv/N));

% 传输测试
data3 = randi([0, mod_order-1], N_eff, 1);
data_mod3 = qammod(data3, mod_order, 'UnitAveragePower', true);
x3 = zeros(N, 1);
x3(Q+1:N-Q) = data_mod3;
x3 = x3 * sqrt(N / N_eff);

y3 = Hi3 * x3;

fprintf('\n传输测试:\n');
fprintf('  发射功率: %.6f\n', mean(abs(x3).^2));
fprintf('  接收功率: %.6f\n', mean(abs(y3).^2));

% LMMSE检测
x_hat3 = (Hi3' * Hi3) \ (Hi3' * y3);  % 零强迫检测
x_hat3_eff = x_hat3(Q+1:N-Q);
demod3 = qamdemod(x_hat3_eff, mod_order, 'UnitAveragePower', true);
ber3 = mean(demod3 ~= data3);

fprintf('  无噪声BER: %.4e\n', ber3);

if ber3 < 1e-4
    fprintf('  ✓ 单径多普勒测试通过！\n');
else
    fprintf('  ⚠ 单径多普勒测试有少量误差\n');
end
fprintf('\n');

%% 测试4：多径信道
fprintf('===== 测试4：多径信道 (4路径) =====\n');

num_paths = 4;
l_i = [0, 10, 20, 30];
alpha_i = [0, -1, 1, 0];
h_true = ones(num_paths, 1) / sqrt(num_paths);

Heff = zeros(N, N);
for i = 1:num_paths
    Hi_temp = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
    fprintf('路径%d (l=%d, alpha=%d): 非零=%d, 范数=%.4f\n', ...
        i, l_i(i), alpha_i(i), nnz(Hi_temp), norm(Hi_temp, 'fro'));
    Heff = Heff + h_true(i) * Hi_temp;
end

fprintf('\n合成信道:\n');
fprintf('  非零元素: %d / %d (%.2f%%)\n', nnz(Heff), N^2, 100*nnz(Heff)/N^2);
fprintf('  Frobenius范数: %.6f\n', norm(Heff, 'fro'));
fprintf('  条件数: %.2e\n', cond(Heff));

% 归一化测试：Frobenius方法
frob_power = norm(Heff, 'fro')^2 / N;
Heff_norm = Heff / sqrt(frob_power);

% 传输测试
data4 = randi([0, mod_order-1], N_eff, 1);
data_mod4 = qammod(data4, mod_order, 'UnitAveragePower', true);
x4 = zeros(N, 1);
x4(Q+1:N-Q) = data_mod4;
x4 = x4 * sqrt(N / N_eff);

y4 = Heff_norm * x4;

fprintf('\n传输测试:\n');
fprintf('  发射功率: %.6f\n', mean(abs(x4).^2));
fprintf('  接收功率: %.6f\n', mean(abs(y4).^2));
fprintf('  功率偏差: %.4f%%\n', 100*abs(mean(abs(y4).^2) - 1));

% 护卫带检查
y_guard = [y4(1:Q); y4(N-Q+1:N)];
y_data = y4(Q+1:N-Q);
leak_ratio = mean(abs(y_guard).^2) / mean(abs(y_data).^2);
fprintf('  护卫带泄漏比: %.2e (%.2f dB)\n', leak_ratio, 10*log10(leak_ratio));

% LMMSE检测
x_hat4 = (Heff_norm' * Heff_norm) \ (Heff_norm' * y4);
x_hat4_eff = x_hat4(Q+1:N-Q);
demod4 = qamdemod(x_hat4_eff, mod_order, 'UnitAveragePower', true);
ber4 = mean(demod4 ~= data4);

fprintf('  无噪声BER: %.4e\n', ber4);

if ber4 < 1e-4
    fprintf('  ✓ 多径测试通过！\n');
else
    fprintf('  ⚠ 多径测试有误差\n');
end
fprintf('\n');

fprintf('========================================\n');
fprintf('测试完成！\n');
fprintf('========================================\n');

%% 辅助函数
function Hi = build_Hi_correct(N, c1, c2, l_i, alpha_i)
    Hi = zeros(N, N);
    Nv = 2;
    
    if alpha_i == 0
        for p = 1:N
            q = mod(p + l_i - 1, N) + 1;
            phase = 2*pi * (c2 * (q^2 - p^2) - c1 * l_i^2);
            Hi(p, q) = exp(1j * phase);
        end
    else
        for p = 1:N
            q_center_float = p + l_i*(1 - 2*N*c1);
            q_center = mod(round(q_center_float) - 1, N) + 1;
            q_width = ceil(N * c1 * (2*abs(alpha_i) + 1 + 2*Nv/N)) + 5;
            
            q_start = q_center - q_width;
            q_end = q_center + q_width;
            
            for q_raw = q_start:q_end
                q = mod(q_raw - 1, N) + 1;
                phase_sum = 0;
                
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
    
    threshold = 1e-2 * max(abs(Hi(:)));
    if threshold > 0
        Hi(abs(Hi) < threshold) = 0;
    end
end
