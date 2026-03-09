"""
============================================================================
端到端CNN检测器模块
============================================================================
提供基于卷积神经网络的端到端MIMO检测器实现

主要组件:
  - CNNDetector: 端到端CNN检测器（改进版，使用匹配滤波初始化）
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class CNNDetector(nn.Module):
    """
    改进版端到端CNN检测器
    
    改进点:
    1. 使用匹配滤波 (H^H * y) 作为初始估计输入
    2. 网络学习残差修正
    3. 移除过度的BatchNorm避免输出被压缩
    4. 添加输出缩放确保幅度正确
    
    Attributes:
        N: 符号维度
        hidden_channels: 隐藏层通道数
    """
    
    def __init__(self, N: int, hidden_channels: int = 64):
        """
        初始化CNN检测器
        
        Args:
            N: 符号维度 (信道矩阵大小 N×N)
            hidden_channels: 隐藏层通道数 (default: 64)
        """
        super().__init__()
        self.N = N
        self.hidden_channels = hidden_channels
        
        # 残差修正网络
        # 输入: 匹配滤波输出 [B, 2, N] (实部和虚部)
        self.refine_net = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        
        # 信道特征提取 (用于调制残差)
        self.channel_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 32, 1, 1]
        )
        
        # 缩放因子（可学习）
        self.scale = nn.Parameter(torch.ones(1))
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CNNDetector: N={N}, hidden_channels={hidden_channels}, 参数量≈{total_params/1e3:.1f}K")
    
    def forward(self, y: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        端到端CNN检测
        
        Args:
            y: 接收信号 [B, N, 1], complex64
            H: 信道矩阵 [B, N, N], complex64
        
        Returns:
            x_hat: 估计符号 [B, N, 1], complex64
        """
        B, N, _ = y.shape
        device = y.device
        
        # 1. 匹配滤波初始化: x_mf = H^H * y
        H_H = H.conj().transpose(1, 2)  # [B, N, N]
        x_mf = torch.matmul(H_H, y)  # [B, N, 1]
        
        # 归一化匹配滤波输出（避免数值问题）
        x_mf_abs_max = torch.abs(x_mf).reshape(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1) + 1e-8
        x_mf_norm = x_mf / x_mf_abs_max
        
        # 2. 转换为实数格式 [B, 2, N]
        x_mf_squeezed = x_mf_norm.squeeze(-1)  # [B, N]
        x_real = torch.stack([x_mf_squeezed.real, x_mf_squeezed.imag], dim=1).float()  # [B, 2, N]
        
        # 3. 残差修正
        residual = self.refine_net(x_real)  # [B, 2, N]
        
        # 4. 残差连接
        x_refined = x_real + residual  # [B, 2, N]
        
        # 5. 转回复数并缩放到QPSK幅度
        x_hat = x_refined[:, 0, :] + 1j * x_refined[:, 1, :]  # [B, N]
        
        # 归一化到单位幅度，然后缩放到QPSK幅度 (1/sqrt(2))
        x_hat_normalized = x_hat / (torch.abs(x_hat) + 1e-8)
        x_hat_scaled = x_hat_normalized * (self.scale.abs() * 0.7071)
        
        x_hat_scaled = x_hat_scaled.unsqueeze(-1)  # [B, N, 1]
        
        return x_hat_scaled.to(torch.complex64)


if __name__ == "__main__":
    # 简单测试
    print("测试 CNNDetector:")
    
    N = 256
    B = 4
    
    model = CNNDetector(N=N, hidden_channels=64)
    
    # 生成测试数据
    y = torch.randn(B, N, 1, dtype=torch.complex64)
    H = torch.randn(B, N, N, dtype=torch.complex64)
    
    x_hat = model(y, H)
    print(f"输入 y 形状: {y.shape}")
    print(f"输入 H 形状: {H.shape}")
    print(f"输出 x_hat 形状: {x_hat.shape}")
    print(f"输出 dtype: {x_hat.dtype}")
    print(f"输出幅度均值: {torch.abs(x_hat).mean():.4f}")
    
    # 测试 CUDA 兼容性
    if torch.cuda.is_available():
        print("\n测试 CUDA 兼容性:")
        model_cuda = model.cuda()
        y_cuda = y.cuda()
        H_cuda = H.cuda()
        x_hat_cuda = model_cuda(y_cuda, H_cuda)
        print(f"CUDA 输出形状: {x_hat_cuda.shape}")
        print(f"CUDA 输出设备: {x_hat_cuda.device}")
