# AFDM宽带双色散信道仿真项目

基于论文：*Affine Frequency Division Multiplexing Over Wideband Doubly-Dispersive Channels With Time-Scaling Effects*

---

## 📁 项目结构

### 核心代码

#### 1. **test_detectors_comparison.m** (主仿真程序)
- **用途**: 完整的AFDM系统BER性能仿真
- **功能**:
  - 宽带双色散信道建模（多径时延 + 多普勒效应）
  - AFDM调制/解调
  - ZF和LMMSE检测器
  - Monte Carlo BER曲线生成
- **运行时间**: 15-30分钟 (N=1024, 100 blocks)
- **输出**: BER vs SNR曲线图

**运行方式**:
```matlab
>> test_detectors_comparison
```

---

#### 2. **build_Hi_correct.m** (信道矩阵构建)
- **用途**: 根据单径参数构建DAF域等效信道矩阵
- **输入**: `(N, c1, c2, l_i, alpha_i)`
- **输出**: `Hi` (N×N稀疏矩阵)
- **特性**:
  - α_i = 0: 解析解（单对角线）
  - α_i ≠ 0: POSP近似 + 数值积分（对角带状）
  - 带宽计算：使用abs(alpha_i)而非alpha_max，更准确
  
**函数签名**:
```matlab
function Hi = build_Hi_correct(N, c1, c2, l_i, alpha_i)
```

---

### 验证工具

#### 3. **test_simple_channel.m** (快速验证)
- **用途**: 验证信道矩阵构建正确性
- **测试场景**:
  - 理想信道 (BER应为0)
  - 单径延迟 (BER应为0)
  - 单径多普勒 (BER应为0)
  - 多径信道 (BER应为0)
- **运行时间**: ~5秒
- **判断标准**: 所有场景BER = 0 → 信道建模正确

**运行方式**:
```matlab
>> test_simple_channel
```

---

#### 4. **validate_system_params.m** (参数验证)
- **用途**: 验证AFDM系统参数是否满足论文约束
- **检查内容**:
  - 宽带条件 (B/fc)
  - C1参数有效性
  - N值可行范围 (论文公式51-52)
  - 信道扩散宽度
  - 护卫带合理性
  - 多普勒条件
- **运行时间**: 瞬时

**运行方式**:
```matlab
>> validate_system_params(1024, 1, 103, 2, 6e3, 4)
% 参数: (N, alpha_max, l_max, Nv, fc, df)
```

---

### 文档

#### 5. **AFDM_SIMULATION_FLOW.md**
- **内容**: 仿真代码完整流程说明
- **章节**:
  - 仿真概述
  - 为什么是单符号传输
  - 完整代码流程（5个阶段）
  - 关键代码段解析
  - 与实际系统的区别
  - Monte Carlo方法说明

---

#### 6. **CHANNEL_MATRIX_PART1.md**
- **内容**: 信道矩阵计算详解（第1部分）
- **章节**:
  - 物理信道模型
  - 离散化与归一化
  - DAF域等效信道推导
  - 单径信道矩阵Hi的计算

---

#### 7. **CHANNEL_MATRIX_PART2.md**
- **内容**: 信道矩阵计算详解（第2部分）
- **章节**:
  - 多径信道矩阵的叠加
  - 完整代码解析
  - 数值示例
  - 参数到矩阵的映射
  - 关键技术细节
  - 常见错误与调试

---

## 🚀 快速开始

### 步骤1: 验证系统参数
```matlab
>> validate_system_params(1024, 1, 103, 2, 6e3, 4)
```
确保所有检查通过 ✓

### 步骤2: 快速验证信道
```matlab
>> test_simple_channel
```
确保所有场景 BER = 0 ✓

### 步骤3: 运行完整仿真
```matlab
>> test_detectors_comparison
```
等待15-30分钟，获得BER曲线

---

## 📊 系统参数

### 默认配置 (符合论文Section VI)
```
N = 1024              % 子载波数
fc = 6 kHz            % 载波频率
df = 4 Hz             % 子载波间隔
B = 4.096 kHz         % 信号带宽
B/fc = 0.683          % 带宽比（宽带）

alpha_max = 1e-4      % 物理多普勒缩放因子
tau_max = 25 ms       % 最大时延扩展
num_paths = 4         % 路径数

c1 = 0.00146          % Chirp参数（获得最大分集增益）
c2 = 0.00049          % Chirp参数
Q = 14                % 护卫带
N_eff = 996           % 有效子载波数

mod_order = 4         % QPSK调制
num_blocks = 100      % Monte Carlo块数
SNR_range = 0:5:20 dB % 信噪比范围
```

---

## 📈 预期结果

### BER性能基准 (N=1024, QPSK)

| SNR (dB) | 理论AWGN | DAF-LMMSE (预期) |
|----------|----------|------------------|
| 10       | ~1e-2    | 2-5e-2          |
| 15       | ~1e-4    | 2-5e-4          |
| 20       | ~1e-6    | 5e-6            |

**注意**: 实际结果会因信道随机性略有波动

---

## 🔧 修改参数

### 修改N值 (子载波数)
```matlab
% test_detectors_comparison.m 第44行
N = 512;   % 快速验证（5-10分钟）
N = 1024;  % 标准仿真（15-30分钟）
N = 2048;  % 高精度（60-90分钟）
```

### 修改Monte Carlo块数
```matlab
% test_detectors_comparison.m 第119行
num_blocks = 50;   % 快速（低精度）
num_blocks = 100;  % 标准（中精度）
num_blocks = 500;  % 论文级别（高精度）
```

### 修改SNR范围
```matlab
% test_detectors_comparison.m 第52行
SNR_dB_range = 0:2:20;   % 更密集的SNR点
SNR_dB_range = [10, 15]; % 只测试特定SNR
```

---

## 🛠️ 故障排除

### 问题1: BER始终为0
**原因**: 可能是SNR设置过高  
**解决**: 检查第151行 `noise_power = 1 / SNR`

### 问题2: BER异常高（>0.5）
**原因**: 信道矩阵或功率归一化错误  
**解决**: 运行 `test_simple_channel` 验证

### 问题3: 运行时间过长
**原因**: N值或num_blocks设置过大  
**解决**: 先用N=512, num_blocks=50快速验证

### 问题4: 内存不足
**原因**: N过大导致矩阵过大  
**解决**: 降低N值或使用稀疏矩阵

---

## 📚 参考资料

### 论文
- **标题**: Affine Frequency Division Multiplexing Over Wideband Doubly-Dispersive Channels With Time-Scaling Effects
- **链接**: https://arxiv.org/html/2507.03537

### 关键公式
- **信道模型**: 论文公式(3)
- **DAF域等效信道**: 论文公式(19-20, 27-29)
- **C1优化**: 论文公式(45)
- **POSP近似**: 论文公式(26, 34)

---

## ✅ 验证清单

运行完整仿真前，确保：

- [ ] `validate_system_params` 所有检查通过
- [ ] `test_simple_channel` 所有场景 BER = 0
- [ ] N值在可行范围内 (xL,N < N < xH,N)
- [ ] 护卫带设置合理 (N_eff > 70% N)
- [ ] SNR范围设置正确
- [ ] num_blocks设置合理（至少50）

---

## 📝 更新日志

### 2024-11-24
- ✅ 完成信道建模验证
- ✅ 修复build_Hi_correct的alpha_i=0特殊情况
- ✅ 修复q_width计算错误（使用abs(alpha_i)）
- ✅ 调整阈值为1e-2以去除旁瓣
- ✅ 创建完整文档（流程说明+信道矩阵详解）
- ✅ 清理冗余测试代码

---

## 📧 说明

本项目实现了论文Section II的信道建模和Section V的基本检测算法（ZF, LMMSE）。

**已实现**:
- ✅ 宽带双色散信道（时延 + 多普勒 + 时间尺度效应）
- ✅ C1参数优化（最大分集增益）
- ✅ DAF域检测器（ZF, LMMSE）

**未实现**:
- ⚠️ 时域检测器（Time-domain LMMSE）
- ⚠️ CD-D-OAMP算法（论文Algorithm 1）
- ⚠️ 信道估计
- ⚠️ CPP/CPS帧结构（仅适用于连续多符号传输）

**核心特性**: 单符号Monte Carlo仿真 + Perfect CSI假设
