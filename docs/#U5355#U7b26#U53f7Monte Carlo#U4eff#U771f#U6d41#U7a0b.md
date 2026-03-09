# AFDM系统仿真代码流程详解

## 📋 目录
1. [仿真概述](#仿真概述)
2. [为什么是单符号传输](#为什么是单符号传输)
3. [完整代码流程](#完整代码流程)
4. [关键代码段解析](#关键代码段解析)
5. [与实际系统的区别](#与实际系统的区别)

---

## 仿真概述

### 仿真目标
验证宽带AFDM系统在双色散信道（时延+多普勒）下的检测性能（BER曲线）

### 仿真类型
**单符号Monte Carlo仿真**
- 每次Monte Carlo迭代传输一个AFDM符号
- 在不同SNR下重复多次
- 统计平均误码率（BER）

### 核心假设
1. **Perfect CSI**: 接收端完全知道信道H
2. **单符号传输**: 每次只传一个符号，不考虑符号间干扰
3. **理想同步**: 时间和频率同步完美

---

## 为什么是单符号传输

### 1. 仿真目的决定

```
目标：评估AFDM在双色散信道下的检测性能
      ↓
核心问题：给定信道H和噪声，检测器能否正确恢复数据？
      ↓
关注点：信道效应 + 检测算法，而非帧同步、信道估计等工程问题
      ↓
方案：单符号传输 + Perfect CSI
```

### 2. 学术界标准做法

**论文仿真通常采用单符号传输**，因为：

- ✓ **隔离核心问题**: 只关注信道建模和检测算法性能
- ✓ **简化分析**: 避免ISI、同步误差等次要因素干扰
- ✓ **公平对比**: 不同调制方案在相同条件下比较
- ✓ **计算效率**: 每次只需处理N个符号，而非整个帧

### 3. 代码结构体现

```matlab
for snr_idx = 1:length(SNR_dB_range)
    for blk = 1:num_blocks
        % ========== 每次迭代是独立的 ==========
        
        % 1. 生成新的随机数据（N_eff个符号）
        data = randi([0, mod_order-1], N_eff, 1);
        
        % 2. 生成新的随机信道（多径参数随机）
        l_i = sort(randperm(l_max+1, num_paths) - 1);
        alpha_i = randi([-alpha_max, alpha_max], num_paths, 1);
        h_true = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
        
        % 3. 传输一个符号
        y = Heff * x + w;
        
        % 4. 检测
        x_hat = detector_LMMSE(y, Heff, noise_power);
        
        % 5. 统计BER
        ber_blk(blk) = mean(demod ~= data);
        
        % ========== 下一次迭代与本次无关 ==========
    end
end
```

**关键特征**：
- ✗ 没有连续的符号序列 `[Symbol1, Symbol2, Symbol3, ...]`
- ✗ 没有前后符号的依赖关系
- ✓ 每个`blk`是独立的信道实现 + 独立的数据
- ✓ 等价于在不同时刻、不同信道条件下传输单个符号

### 4. 类比说明

#### 单符号传输（本代码）
```
时间轴: |--符号1--| 间隔 |--符号2--| 间隔 |--符号3--|
信道:    H1               H2               H3
说明:    每个符号传输完后，等待足够长时间，让信道完全衰减
        下一个符号到来时，信道状态已完全改变（新的多径参数）
```

#### 连续多符号传输（实际系统）
```
时间轴: [CPP|符号1|CPS][CPP|符号2|CPS][CPP|符号3|CPS]
信道:    -------- 同一个信道 H --------
说明:    连续传输，前一个符号的尾部可能干扰下一个符号
        需要CPP/CPS消除ISI
```

---

## 完整代码流程

### 阶段1：参数初始化（第37-106行）

```matlab
%% 物理参数
fc = 6e3;                      % 载波频率
N = 1024;                      % 子载波数
df = 4;                        % 子载波间隔
alpha_max_paper = 1e-4;        % 物理多普勒缩放因子
tau_max = 25e-3;               % 最大时延扩展
num_paths = 4;                 % 路径数

%% 归一化参数
alpha_max = ceil(fd_max / df); % 离散化多普勒索引
l_max = ceil(tau_max * df * N);% 离散化时延索引

%% AFDM参数（论文公式11）
c1 = (2*alpha_max + 1 + 2*Nv/N) / (2*N);  % 获得最大分集增益
c2 = 1 / (2*N);

%% 护卫带设计
Q = ceil(L_spread / 2) + 5;    % 保守策略
N_eff = N - 2*Q;               % 有效子载波数
```

**为什么这些参数？**
- `c1`: 论文公式(45)，确保不同路径非零元素不重叠 → 最大分集增益
- `Q`: 护卫带，防止信道扩散导致的ICI
- `N_eff`: 实际承载数据的子载波数

---

### 阶段2：AFDM调制矩阵构建（第133-138行）

```matlab
%% IDAF变换矩阵
F = dftmtx(N)/sqrt(N);                                  % DFT矩阵
Lambda_c1 = diag(exp(-1j * 2 * pi * c1 * (0:N-1).^2)); % Chirp调制1
Lambda_c2 = diag(exp(-1j * 2 * pi * c2 * (0:N-1).^2)); % Chirp调制2
A  = Lambda_c2 * F * Lambda_c1;                         % IDAF矩阵
AH = Lambda_c1' * F' * Lambda_c2';                      % DAF矩阵
```

**作用**：
- 定义AFDM的调制变换
- `A`: DAF域 → 时域（发射端）
- `AH`: 时域 → DAF域（接收端）

---

### 阶段3：Monte Carlo循环（第148-262行）

#### 外循环：遍历SNR

```matlab
for snr_idx = 1:length(SNR_dB_range)  % SNR = [0, 5, 10, 15, 20] dB
    SNR_dB = SNR_dB_range(snr_idx);
    SNR = 10^(SNR_dB / 10);
    noise_power = 1 / SNR;
```

#### 内循环：Monte Carlo迭代

```matlab
    for blk = 1:num_blocks  % 100次迭代
```

##### 步骤1：生成发射信号（第157-168行）

```matlab
% 生成随机数据（只在有效子载波上）
data = randi([0, mod_order-1], N_eff, 1);  % N_eff个QPSK符号
data_mod = qammod(data, mod_order, 'UnitAveragePower', true);

% 构造N维DAF域符号（边缘置零）
x = zeros(N, 1);
x(Q+1:N-Q) = data_mod;  % 只在中间的N_eff个位置放数据

% 功率归一化：E[|x|²] = 1
x = x * sqrt(N / N_eff);
```

**为什么这样设计？**
```
原始信号: x[1]=0, ..., x[Q]=0, x[Q+1]=d1, ..., x[N-Q]=dN_eff, x[N-Q+1]=0, ..., x[N]=0
                 |<-- 护卫带 -->|<---- 数据区 ---->|<-- 护卫带 -->|

作用：
1. 护卫带为零 → 避免边缘ICI
2. 功率归一化 → 确保发射功率为1
```

##### 步骤2：生成随机信道（第174-197行）

```matlab
% ========== 关键：每次迭代的信道都是独立随机的 ==========

% 随机时延（模拟多径）
l_i = sort(randperm(l_max+1, num_paths) - 1);  % 4个随机时延

% 随机多普勒缩放因子
alpha_i = randi([-alpha_max, alpha_max], num_paths, 1);  % 4个随机多普勒

% 随机路径增益（Rayleigh衰落）
h_true = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
h_true = h_true / sqrt(sum(abs(h_true).^2));

% 构建DAF域信道矩阵
Heff = zeros(N, N);
for i = 1:num_paths
    Hi = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
    Heff = Heff + h_true(i) * Hi;  % 多径叠加
end

% 信道功率归一化
H_eff_sub = Heff(Q+1:N-Q, Q+1:N-Q);
sub_power = norm(H_eff_sub, 'fro')^2 / N_eff;
Heff = Heff / sqrt(sub_power);
```

**这就是"单符号传输"的体现！**
- 每次`blk`循环：新的随机信道参数 `{l_i, alpha_i, h_true}`
- 等价于：不同时刻在不同信道条件下传输单个符号
- 而非：在同一信道下连续传输多个符号

##### 步骤3：信道传输（第219-224行）

```matlab
% DAF域等效模型
w = sqrt(noise_power/2) * (randn(N,1) + 1j*randn(N,1));
y = Heff * x + w;
```

**模型解释**：
```
完整模型: y_time = H_time * x_time + w_time  (时域，连续时间)
         ↓ 采样 + AFDM处理
等效模型: y = Heff * x + w  (DAF域，离散)

其中：
- x: DAF域发射符号（N维）
- Heff: DAF域等效信道（N×N矩阵，稀疏对角带状）
- w: 高斯白噪声
- y: DAF域接收信号
```

##### 步骤4：检测（第238-250行）

```matlab
% ZF检测
x_hat_zf = detector_ZF(y, Heff);  % x_hat = inv(H) * y

% LMMSE检测
x_hat_lmmse = detector_LMMSE(y, Heff, noise_power);  
% x_hat = inv(H'*H + σ²*I) * H' * y

% 只对有效子载波解调
x_hat_lmmse_eff = x_hat_lmmse(Q+1:N-Q);
demod_lmmse = qamdemod(x_hat_lmmse_eff, mod_order, 'UnitAveragePower', true);

% 计算BER
ber_blk(2, blk) = mean(demod_lmmse ~= data);
```

##### 步骤5：统计（第253-261行）

```matlab
    end  % 结束blk循环
    
    % 平均BER（100次Monte Carlo）
    BER_results(d, snr_idx) = mean(ber_blk(d, :));
end  % 结束snr_idx循环
```

---

### 阶段4：结果可视化（第266行之后）

```matlab
%% 绘制BER曲线
figure;
semilogy(SNR_dB_range, BER_results(1,:), '-o', 'DisplayName', 'ZF');
semilogy(SNR_dB_range, BER_results(2,:), '-s', 'DisplayName', 'LMMSE');
```

---

## 关键代码段解析

### 1. 为什么每次生成新信道？

```matlab
for blk = 1:num_blocks
    % 每次迭代都重新生成
    l_i = sort(randperm(l_max+1, num_paths) - 1);
    alpha_i = randi([-alpha_max, alpha_max], num_paths, 1);
    h_true = (randn(num_paths,1) + 1j*randn(num_paths,1)) / sqrt(2);
    ...
end
```

**原因**：模拟不同时刻、不同信道条件下的传输
- 实际系统：信道随时间变化（移动、散射体改变）
- Monte Carlo：通过空间集合平均（不同信道实现）近似时间平均

**类比**：
```
实际系统: 在时刻t1, t2, ..., t100分别传输一个符号，每次信道不同
仿真:     生成100个独立的信道实现H1, H2, ..., H100，每个传输一次
```

### 2. 信道矩阵build_Hi_correct

```matlab
Hi = build_Hi_correct(N, c1, c2, l_i(i), alpha_i(i));
```

**作用**：根据单径参数`{l_i, alpha_i}`构建N×N的DAF域信道矩阵

**理论依据**：论文公式(27-29)
```
Hi[p,q] = (1/N) * Σ_{n=0}^{N-1} exp{j·2π/N·Φ(p,q,n)}

其中相位函数Φ包含：
- 延时项: n*(p-q-l_i)
- 多普勒项: alpha_i * n²  ← 时间尺度效应！
- Chirp项: N*c2*(q²-p²) - N*c1*l_i²
```

**稀疏性**：POSP近似 → Hi是稀疏对角带状矩阵

### 3. 功率归一化的两次操作

#### 发射端归一化（第168行）
```matlab
x = x * sqrt(N / N_eff);
```
确保：`E[|x|²] = 1`（发射功率为1）

#### 信道归一化（第204-211行）
```matlab
H_eff_sub = Heff(Q+1:N-Q, Q+1:N-Q);
sub_power = norm(H_eff_sub, 'fro')^2 / N_eff;
Heff = Heff / sqrt(sub_power);
```
确保：`E[|Heff*x|²] ≈ E[|x|²] = 1`（信道不改变平均功率）

**这样SNR定义才准确**：
```
SNR = 信号功率 / 噪声功率 = 1 / noise_power
```

---

## 与实际系统的区别

### 本仿真（单符号）
```
功能: 评估信道+检测器性能
假设: Perfect CSI, 理想同步
传输: 每次一个符号，间隔足够长
优点: 简单、快速、隔离核心问题
缺点: 不涉及ISI、信道估计、同步
```

### 实际系统（连续多符号）
```
功能: 完整通信链路
需要: 信道估计、时间同步、帧同步
传输: [CPP|符号1|CPS][CPP|符号2|CPS]...
优点: 完整的系统实现
缺点: 复杂度高，难以分析性能瓶颈
```

### 对比表

| 特性 | 单符号仿真 | 实际系统 |
|------|-----------|---------|
| **符号间关系** | 独立 | 连续 |
| **CPP/CPS** | 不需要 | 必须 |
| **信道估计** | Perfect CSI | 需要实现 |
| **时间同步** | 理想 | 需要算法 |
| **ISI问题** | 无 | 需要CPP/CPS消除 |
| **计算复杂度** | 低 | 高 |
| **适用场景** | 性能评估 | 实际部署 |

---

## Monte Carlo方法说明

### 为什么需要多次迭代？

**单次传输**：
```
一个信道实现H1 + 一组随机数据 + 一组随机噪声 → BER1
```
结果不可靠，因为：
- 运气成分（噪声可能刚好小）
- 特殊信道（可能刚好条件数好）

**100次Monte Carlo**：
```
BER = (BER1 + BER2 + ... + BER100) / 100
```
结果可靠，因为：
- 平均掉了随机性
- 涵盖了各种信道条件

### 统计原理

**中心极限定理**：
```
BER_measured = E[BER] ± σ/√num_blocks
```

当`num_blocks = 100`时：
- 标准误差 `σ/10`，相对准确
- 论文级别通常用`num_blocks = 500-1000`

---

## 总结

### 核心流程（一图流）

```
┌─────────────────────────────────────────────────────┐
│                   初始化参数                         │
│  fc, N, df, alpha_max, tau_max, c1, c2, Q, N_eff   │
└──────────────────┬──────────────────────────────────┘
                   ↓
         ┌─────────────────────┐
         │  for snr_idx = 1:5  │ ← 外循环：遍历SNR
         └──────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ for blk = 1:100    │ ← 内循环：Monte Carlo
          └─────────┬──────────┘
                    ↓
         ┌──────────────────────────┐
         │ 1. 生成随机数据（N_eff） │
         │    data = randi(...)      │
         └──────────┬───────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 2. 生成随机信道（4径）    │ ← 每次都不同！
         │    l_i, alpha_i, h_true  │
         │    Heff = Σ h_i * Hi     │
         └──────────┬───────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 3. 传输单个符号          │
         │    y = Heff * x + w      │
         └──────────┬───────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 4. 检测                  │
         │    x_hat = LMMSE(y, H)   │
         └──────────┬───────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 5. 计算BER               │
         │    ber_blk(blk) = ...    │
         └──────────┬───────────────┘
                    ↓
          └────────────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 6. 统计平均BER           │
         │    BER(snr) = mean(...)  │
         └──────────┬───────────────┘
                    ↓
         └─────────────────────┘
                    ↓
         ┌──────────────────────────┐
         │ 7. 绘制BER曲线           │
         └──────────────────────────┘
```

### 关键点

1. **单符号传输** = 每次Monte Carlo迭代传输一个独立符号
2. **独立信道** = 每次生成新的随机信道参数
3. **不需要CPP/CPS** = 符号间无关联，无ISI
4. **Perfect CSI** = 检测时已知准确的信道H
5. **Monte Carlo平均** = 统计大量独立传输的平均性能

### 适用范围

✓ **适合**：
- 评估信道建模正确性
- 对比不同检测算法
- 验证理论分析（BER界）
- 快速原型验证

✗ **不适合**：
- 评估信道估计算法
- 测试同步算法
- 硬件实现验证
- 端到端系统性能

---

## 参考

- 论文：*Affine Frequency Division Multiplexing Over Wideband Doubly-Dispersive Channels With Time-Scaling Effects*
- 代码：`test_detectors_comparison.m`
- 信道建模：`build_Hi_correct` 函数（第551-630行）
