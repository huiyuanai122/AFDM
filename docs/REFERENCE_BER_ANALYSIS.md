# 参考BER曲线分析

## 参考图信息

来源: "Affine Frequency Division Multiplexing Over Wideband Doubly-Dispersive Channels With Time-Scaling Effects" 论文 Fig. 13

**重要**: 这是针对AFDM系统的BER性能分析，与我们的项目直接相关！

## 检测器性能排序（从差到好）

| 检测器 | 类型 | 说明 | 相对LMMSE增益 (BER=10⁻³) |
|--------|------|------|-------------------------|
| LMMSE [23] | 线性 | 基准检测器 | 0 dB |
| AMP [30] | 迭代 | 近似消息传递 | ~0.5 dB |
| GMP Damping=0.7 [25] | 迭代 | 高斯消息传递 | ~1 dB |
| D-OAMP C=1 (OAMP [32]) | 迭代 | 标准OAMP，C=1表示基本配置 | ~2-3 dB |
| D-OAMP C=32 | 迭代 | 增强OAMP | ~3 dB |
| D-OAMP C=64 | 迭代 | 高复杂度OAMP | ~3.5 dB |
| CD-D-OAMP C=1 | 深度展开 | 基于协方差驱动的深度OAMP | ~3 dB |
| CD-D-OAMP C=16 | 深度展开 | | ~3.5 dB |
| CD-D-OAMP C=32 | 深度展开 | | ~4 dB |
| CD-D-OAMP C=64 | 深度展开 | 最佳性能 | ~4.5 dB |

## 术语解释（基于AFDM论文）

- **D-OAMP**: Damped OAMP，带阻尼的OAMP
- **CD-D-OAMP**: Covariance-Driven Damped OAMP，协方差驱动的阻尼OAMP（深度学习版本）
- **C**: 可能是迭代次数或协方差矩阵的近似阶数

## 关键观察

### 1. OAMP vs LMMSE 应有显著差距
- 标准D-OAMP (C=1) 应该比LMMSE好 **2-3 dB**
- 我们当前实现只有 ~1% 改进，说明实现有问题

### 2. 参数C的含义
- C 可能代表迭代次数或某种复杂度参数
- C越大，性能越好，但计算复杂度也越高

### 3. CD-D-OAMP vs D-OAMP
- CD 可能代表 "Covariance-Driven" 或 "Conjugate Direction"
- CD-D-OAMP 是深度展开版本，通过学习参数获得额外增益
- 在相同C下，CD-D-OAMP 比 D-OAMP 好约 0.5-1 dB

## 我们的问题分析

### 当前结果
```
LMMSE:      BER = 0.0749
原OAMP:     BER = 0.0771 (比LMMSE差!)
修复后OAMP: BER = 0.0738 (仅1.4%改进)
```

### 预期结果 (根据参考图)
- 在SNR=15dB时，LMMSE BER ≈ 10⁻²
- D-OAMP BER ≈ 10⁻³ (好10倍!)
- CD-D-OAMP BER ≈ 10⁻⁴ (好100倍!)

### 可能的问题

1. **信道矩阵条件数问题**
   - AFDM信道矩阵可能与MIMO信道不同
   - 需要检查信道矩阵的特性

2. **方差传递公式不准确**
   - 当前使用简化版方差传递
   - 可能需要更精确的公式

3. **Onsager校正系数**
   - 当前使用固定小系数 (0.1)
   - 可能需要根据方差动态计算

4. **初始化问题**
   - 当前使用零初始化
   - 可能需要使用匹配滤波初始化

## 改进方向

1. **实现更精确的方差传递**
   ```
   τ^t = (1/N) · tr(I - W·H) · v^{t-1} + σ² · (1/N) · ||W||_F²
   ```

2. **使用正确的Onsager校正**
   ```
   α = v / τ  (动态计算，而非固定值)
   ```

3. **检查信道矩阵归一化**
   - 确保 E[|H_{ij}|²] = 1/N

4. **增加迭代次数**
   - 参考图中C=64表现最好
   - 尝试增加到20-30次迭代

---
*更新日期: 2024-12-21*
