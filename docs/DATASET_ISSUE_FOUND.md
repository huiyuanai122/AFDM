# 数据集问题发现与修复

## 问题描述

在验证信道矩阵正确性时，发现 MATLAB 数据集中的 `y_dataset` 与 `H_dataset` 不匹配。

## 根本原因

**H 矩阵需要转置！**

MATLAB 使用列优先 (column-major) 存储，Python/NumPy 使用行优先 (row-major) 存储。当 MATLAB 保存 `H_dataset(:, :, idx) = Heff` 到 HDF5 文件时，维度顺序会发生变化，导致 H 矩阵实际上被转置了。

## 验证结果

### 修复前 (SNR=30dB)

| 指标 | 使用 H | 使用 H.T |
|------|--------|----------|
| 与 H*x 相关系数 | 0.037 | 0.9995 |
| 残差功率 \|y - H*x\|² | 2.30 | 0.001 |
| LMMSE MSE | 10.33 | 0.012 |
| LMMSE BER | 0.49 | 0.00 |

### 修复后

在 `python/utils.py` 的 `AFDMDataset` 类中添加了 H 矩阵转置：

```python
def _reorder_H(arr, N):
    s = arr.shape
    if len(s) == 3 and s[1] == N and s[2] == N:
        # [S, N, N] -> 需要对每个样本的 H 矩阵转置
        return np.transpose(arr, (0, 2, 1))  # 转置每个 H 矩阵
    ...
```

### 修复后验证结果

| SNR (dB) | 相关系数 | 残差功率 | LMMSE BER |
|----------|----------|----------|-----------|
| 10 | 0.95+ | ~0.1 | ~0.05 |
| 20 | 0.996 | ~0.01 | ~0.003 |
| 25 | 0.999 | ~0.003 | 0.00 |
| 30 | 0.9996 | ~0.001 | 0.00 |

## 影响

### 对 Python 训练的影响

**已修复**。`AFDMDataset` 类现在正确转置 H 矩阵，训练和测试数据都是正确的。

### 对 MATLAB 仿真的影响

**无影响**。MATLAB 仿真代码 (`ber_comparison.m`) 正确地重新生成接收信号：
```matlab
y = H * x_true + w;
```
而不是使用 `y_dataset`。

## 验证脚本

```bash
# 验证修复
python diagnostic/verify_fix.py

# 检查 H 转置问题
python diagnostic/check_h_transpose.py
```

---
*发现日期: 2024-12-21*
*修复日期: 2024-12-21*
