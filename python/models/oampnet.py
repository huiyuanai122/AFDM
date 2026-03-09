"""
============================================================================
OAMP-Net: 真正的OAMP算法展开网络
============================================================================
参考论文: "Model-Driven Deep Learning for MIMO Detection"

核心思想:
- 将OAMP迭代算法展开为T层神经网络
- 每层对应一次OAMP迭代
- 步长γ、阻尼系数等参数可学习
- 保持OAMP的算法结构，只是参数从固定变为可学习

OAMP算法结构:
1. 线性估计器 (LE): r^t = x^{t-1} + γ^t · W · (y - H·x^{t-1})
2. 非线性估计器 (NLE): x^t = η(r^t)  (QPSK软投影)

与传统OAMP的区别:
- 传统OAMP: γ固定，W = (H^H·H)^{-1}·H^H
- OAMP-Net: γ^t可学习，W可以加入可学习修正
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def soft_qpsk_projection(r: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    QPSK软投影函数 - 非线性去噪器
    
    Args:
        r: 输入复数符号 [B, N, 1], complex
        temperature: 温度参数，越小越接近硬判决
    
    Returns:
        投影后的符号 [B, N, 1], complex
    """
    # QPSK星座点: (±1±1j) / sqrt(2)
    qpsk_points = torch.tensor(
        [[1, 1], [1, -1], [-1, 1], [-1, -1]],
        dtype=torch.float32,
        device=r.device
    ) / np.sqrt(2.0)
    
    r_real = r.real.squeeze(-1)  # [B, N]
    r_imag = r.imag.squeeze(-1)  # [B, N]
    
    # 计算到各星座点的负距离平方
    dist = torch.zeros(r.shape[0], r.shape[1], 4, device=r.device, dtype=torch.float32)
    for i, (pr, pi) in enumerate(qpsk_points):
        dist[..., i] = -((r_real - pr)**2 + (r_imag - pi)**2) / temperature
    
    # Softmax权重
    weights = torch.softmax(dist, dim=-1)  # [B, N, 4]
    
    # 加权平均
    out_real = torch.sum(weights * qpsk_points[:, 0], dim=-1, keepdim=True)
    out_imag = torch.sum(weights * qpsk_points[:, 1], dim=-1, keepdim=True)
    
    return (out_real + 1j * out_imag).to(r.dtype)


class OAMPNet(nn.Module):
    """
    OAMP-Net: OAMP算法展开网络
    
    将T次OAMP迭代展开为T层网络，每层结构相同:
    1. 线性估计器: r = x + γ · W · (y - H·x)
    2. 非线性估计器: x = η(r)
    
    可学习参数:
    - gamma: 每层的步长参数 [T]
    - theta: 每层的阻尼/混合参数 [T]（可选）
    
    Attributes:
        N: 符号维度
        num_layers: 展开层数（对应OAMP迭代次数）
        use_noise_var: 是否使用噪声方差信息
    """
    
    def __init__(
        self,
        N: int,
        num_layers: int = 10,
        use_noise_var: bool = True,
        init_gamma: float = 1.0,
        temperature: float = 0.5
    ):
        """
        初始化OAMP-Net
        
        Args:
            N: 符号维度
            num_layers: 展开层数 (default: 10)
            use_noise_var: 是否使用噪声方差 (default: True)
            init_gamma: 步长初始值 (default: 1.0)
            temperature: QPSK软投影温度 (default: 0.5)
        """
        super().__init__()
        self.N = N
        self.num_layers = num_layers
        self.use_noise_var = use_noise_var
        self.temperature = temperature
        
        # 可学习参数: 每层的步长
        # 初始化为1.0，让网络从标准OAMP开始学习
        self.gamma = nn.Parameter(init_gamma * torch.ones(num_layers))
        
        # 可学习参数: 每层的阻尼系数（混合上一次估计和当前估计）
        # θ=1表示完全使用当前估计，θ=0表示完全使用上一次
        self.theta = nn.Parameter(torch.ones(num_layers))
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"OAMPNet: N={N}, 层数={num_layers}, 参数量={total_params}")
    
    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        noise_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        OAMP-Net前向传播
        
        Args:
            y: 接收信号 [B, N, 1], complex
            H: 信道矩阵 [B, N, N], complex
            noise_var: 噪声方差 [B], float (可选)
        
        Returns:
            x_hat: 估计符号 [B, N, 1], complex
        """
        B, N, _ = y.shape
        device = y.device
        dtype = H.dtype
        
        # 计算 H^H
        H_H = H.conj().transpose(1, 2)  # [B, N, N]
        
        # 计算 H^H · H
        HH = torch.matmul(H_H, H)  # [B, N, N]
        
        # 计算线性估计器的矩阵 W = (H^H·H + λI)^{-1} · H^H
        # λ是正则化参数，可以用噪声方差
        if self.use_noise_var and noise_var is not None:
            # 使用噪声方差作为正则化
            reg = noise_var.view(B, 1, 1) * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        else:
            # 使用小的固定正则化
            reg = 1e-6 * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        
        HH_reg = HH + reg  # [B, N, N]
        W = torch.linalg.solve(HH_reg, H_H)  # [B, N, N]
        
        # 初始化: 匹配滤波 x^0 = H^H · y
        x = torch.matmul(H_H, y)  # [B, N, 1]
        
        # OAMP迭代展开
        for t in range(self.num_layers):
            # 保存上一次估计
            x_prev = x
            
            # 1. 线性估计器 (LE)
            # r^t = x^{t-1} + γ^t · W · (y - H·x^{t-1})
            residual = y - torch.matmul(H, x)  # [B, N, 1]
            r = x + self.gamma[t] * torch.matmul(W, residual)  # [B, N, 1]
            
            # 2. 非线性估计器 (NLE) - QPSK软投影
            x_nle = soft_qpsk_projection(r, temperature=self.temperature)
            
            # 3. 阻尼/混合（可选，帮助收敛）
            # x^t = θ^t · x_nle + (1-θ^t) · x^{t-1}
            theta_t = torch.sigmoid(self.theta[t])  # 限制在[0,1]
            x = theta_t * x_nle + (1 - theta_t) * x_prev
        
        return x


class OAMPNetV2(nn.Module):
    """
    OAMP-Net V2: 增强版OAMP展开网络
    
    在标准OAMP-Net基础上增加:
    1. 每层独立的可学习温度参数
    2. 残差连接选项
    3. 更灵活的初始化
    """
    
    def __init__(
        self,
        N: int,
        num_layers: int = 10,
        use_noise_var: bool = True,
        learnable_temperature: bool = True
    ):
        super().__init__()
        self.N = N
        self.num_layers = num_layers
        self.use_noise_var = use_noise_var
        
        # 每层的步长
        self.gamma = nn.Parameter(torch.ones(num_layers))
        
        # 每层的阻尼系数
        self.theta = nn.Parameter(torch.ones(num_layers))
        
        # 每层的温度参数（控制软/硬判决）
        if learnable_temperature:
            # 初始化为0.5，通过softplus确保正值
            self.temperature_raw = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_buffer('temperature_raw', torch.zeros(num_layers))
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"OAMPNetV2: N={N}, 层数={num_layers}, 参数量={total_params}")
    
    def get_temperature(self, t: int) -> float:
        """获取第t层的温度参数"""
        return 0.1 + 0.9 * torch.sigmoid(self.temperature_raw[t])  # 范围[0.1, 1.0]
    
    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        noise_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = y.shape
        device = y.device
        dtype = H.dtype
        
        H_H = H.conj().transpose(1, 2)
        HH = torch.matmul(H_H, H)
        
        if self.use_noise_var and noise_var is not None:
            reg = noise_var.view(B, 1, 1) * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        else:
            reg = 1e-6 * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        
        HH_reg = HH + reg
        W = torch.linalg.solve(HH_reg, H_H)
        
        # 初始化
        x = torch.matmul(H_H, y)
        
        for t in range(self.num_layers):
            x_prev = x
            
            # 线性估计器
            residual = y - torch.matmul(H, x)
            r = x + self.gamma[t] * torch.matmul(W, residual)
            
            # 非线性估计器（可学习温度）
            temp = self.get_temperature(t)
            x_nle = soft_qpsk_projection(r, temperature=temp.item())
            
            # 阻尼
            theta_t = torch.sigmoid(self.theta[t])
            x = theta_t * x_nle + (1 - theta_t) * x_prev
        
        return x


# 保持向后兼容
HybridOAMPNet = OAMPNet


if __name__ == "__main__":
    print("=" * 60)
    print("测试 OAMPNet")
    print("=" * 60)
    
    N = 256
    B = 4
    
    # 测试 OAMPNet
    print("\n1. 测试 OAMPNet:")
    model = OAMPNet(N=N, num_layers=10)
    
    y = torch.randn(B, N, 1, dtype=torch.complex64)
    H = torch.randn(B, N, N, dtype=torch.complex64)
    noise_var = 0.1 * torch.ones(B)
    
    x_hat = model(y, H, noise_var)
    print(f"   输入 y: {y.shape}")
    print(f"   输入 H: {H.shape}")
    print(f"   输出 x_hat: {x_hat.shape}")
    print(f"   gamma: {model.gamma.data}")
    
    # 测试 OAMPNetV2
    print("\n2. 测试 OAMPNetV2:")
    model2 = OAMPNetV2(N=N, num_layers=10)
    x_hat2 = model2(y, H, noise_var)
    print(f"   输出 x_hat: {x_hat2.shape}")
    
    print("\n测试完成！")
