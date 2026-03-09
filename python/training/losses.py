"""
============================================================================
损失函数模块
============================================================================
提供用于训练OAMP-Net的各种损失函数

主要组件:
  - MSELoss: 标准MSE损失
  - BERAwareLoss: BER感知损失（结合MSE和符号错误率）
  - SymbolErrorLoss: 软符号错误损失
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    标准MSE损失
    
    L = E[|x_hat - x_true|²]
    """
    
    def __init__(self, Q: int = 0):
        """
        Args:
            Q: 保护子载波数（排除边缘）
        """
        super().__init__()
        self.Q = Q
    
    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        计算MSE损失
        
        Args:
            x_hat: 估计符号 [B, N, 1]
            x_true: 真实符号 [B, N, 1]
            
        Returns:
            loss: 标量损失
        """
        if self.Q > 0:
            x_hat_eff = x_hat[:, self.Q:-self.Q, :]
            x_true_eff = x_true[:, self.Q:-self.Q, :]
        else:
            x_hat_eff = x_hat
            x_true_eff = x_true
        
        # 复数MSE = |real差|² + |imag差|²
        diff_real = x_hat_eff.real - x_true_eff.real
        diff_imag = x_hat_eff.imag - x_true_eff.imag
        
        loss = diff_real.pow(2).mean() + diff_imag.pow(2).mean()
        return loss


class SymbolErrorLoss(nn.Module):
    """
    软符号错误损失
    
    使用软近似计算符号错误率，使其可微
    
    L = 1 - Σ softmax(-d²/τ)
    其中 d = 到最近星座点的距离
    """
    
    def __init__(self, Q: int = 0, temperature: float = 0.1):
        """
        Args:
            Q: 保护子载波数
            temperature: 温度参数（越小越接近硬判决）
        """
        super().__init__()
        self.Q = Q
        self.temperature = temperature
        
        # QPSK星座点
        self.register_buffer(
            'constellation',
            torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64) / np.sqrt(2)
        )
    
    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        计算软符号错误损失
        
        Args:
            x_hat: 估计符号 [B, N, 1]
            x_true: 真实符号 [B, N, 1]
            
        Returns:
            loss: 标量损失
        """
        if self.Q > 0:
            x_hat_eff = x_hat[:, self.Q:-self.Q, :]
            x_true_eff = x_true[:, self.Q:-self.Q, :]
        else:
            x_hat_eff = x_hat
            x_true_eff = x_true
        
        B, N_eff, _ = x_hat_eff.shape
        
        # 找到真实符号对应的星座点索引
        x_true_flat = x_true_eff.reshape(-1)  # [B*N_eff]
        # 确保 constellation 在正确的设备上
        constellation = self.constellation.to(x_true_flat.device)
        dist_true = torch.abs(x_true_flat.unsqueeze(-1) - constellation)  # [B*N_eff, 4]
        true_idx = torch.argmin(dist_true, dim=-1)  # [B*N_eff]
        
        # 计算估计符号到各星座点的距离
        x_hat_flat = x_hat_eff.reshape(-1)  # [B*N_eff]
        dist_hat = torch.abs(x_hat_flat.unsqueeze(-1) - constellation)  # [B*N_eff, 4]
        
        # 软判决概率
        log_prob = -dist_hat.pow(2) / self.temperature
        prob = F.softmax(log_prob, dim=-1)  # [B*N_eff, 4]
        
        # 正确符号的概率
        correct_prob = prob.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)  # [B*N_eff]
        
        # 损失 = 1 - 正确概率
        loss = 1 - correct_prob.mean()
        
        return loss


class BERAwareLoss(nn.Module):
    """
    BER感知损失函数
    
    结合MSE和符号错误率：
    L = α · MSE + β · SymbolErrorRate
    
    这样可以同时优化：
    1. 估计精度（MSE）
    2. 检测正确率（SER）
    """
    
    def __init__(
        self,
        Q: int = 0,
        alpha: float = 1.0,
        beta: float = 0.5,
        temperature: float = 0.1
    ):
        """
        Args:
            Q: 保护子载波数
            alpha: MSE损失权重
            beta: 符号错误损失权重
            temperature: 软判决温度
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.mse_loss = MSELoss(Q)
        self.ser_loss = SymbolErrorLoss(Q, temperature)
    
    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        计算BER感知损失
        
        Args:
            x_hat: 估计符号 [B, N, 1]
            x_true: 真实符号 [B, N, 1]
            
        Returns:
            loss: 标量损失
        """
        mse = self.mse_loss(x_hat, x_true)
        ser = self.ser_loss(x_hat, x_true)
        
        loss = self.alpha * mse + self.beta * ser
        return loss


class AdaptiveBERAwareLoss(nn.Module):
    """
    自适应BER感知损失
    
    根据训练进度自动调整MSE和SER的权重：
    - 训练初期：主要优化MSE（稳定收敛）
    - 训练后期：增加SER权重（提高检测准确率）
    """
    
    def __init__(
        self,
        Q: int = 0,
        initial_alpha: float = 1.0,
        final_alpha: float = 0.5,
        initial_beta: float = 0.1,
        final_beta: float = 1.0,
        warmup_epochs: int = 20,
        temperature: float = 0.1
    ):
        """
        Args:
            Q: 保护子载波数
            initial_alpha: 初始MSE权重
            final_alpha: 最终MSE权重
            initial_beta: 初始SER权重
            final_beta: 最终SER权重
            warmup_epochs: 预热轮数
            temperature: 软判决温度
        """
        super().__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warmup_epochs = warmup_epochs
        
        self.mse_loss = MSELoss(Q)
        self.ser_loss = SymbolErrorLoss(Q, temperature)
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def get_weights(self) -> tuple:
        """获取当前权重"""
        if self.current_epoch < self.warmup_epochs:
            # 线性插值
            ratio = self.current_epoch / self.warmup_epochs
            alpha = self.initial_alpha + ratio * (self.final_alpha - self.initial_alpha)
            beta = self.initial_beta + ratio * (self.final_beta - self.initial_beta)
        else:
            alpha = self.final_alpha
            beta = self.final_beta
        
        return alpha, beta
    
    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """计算自适应损失"""
        alpha, beta = self.get_weights()
        
        mse = self.mse_loss(x_hat, x_true)
        ser = self.ser_loss(x_hat, x_true)
        
        loss = alpha * mse + beta * ser
        return loss


class QPSKSymbolCrossEntropyLoss(nn.Module):
    """QPSK 符号交叉熵损失（基于网络输出的 logits）

    该损失假设网络在最后一层 MMSE 中计算了四个星座点的 logits：
      logits[b, n, k] ∝ -|r - s_k|^2 / (tau_eff)

    这样做比“仅用距离做软SER”更贴近 MAP/ML 检测目标，并且可以直接优化 BER。

    输入:
      - logits: [B, N, 4]
      - x_true: [B, N, 1]  (QPSK 复符号)
    """

    def __init__(self, Q: int = 0):
        super().__init__()
        self.Q = Q
        self.register_buffer(
            'constellation',
            torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64) / np.sqrt(2)
        )

    def forward(self, logits: torch.Tensor, x_true: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if logits is None:
            raise ValueError("logits is None. Make sure model(..., return_posterior=True) and returns logits.")

        # slice有效子载波
        if self.Q > 0:
            logits_eff = logits[:, self.Q:-self.Q, :]
            x_true_eff = x_true[:, self.Q:-self.Q, :]
        else:
            logits_eff = logits
            x_true_eff = x_true

        # true symbol -> label index
        x_true_flat = x_true_eff.reshape(-1)  # [B*N_eff]
        # 确保 constellation 在正确的设备上
        constellation = self.constellation.to(x_true_flat.device)
        dist = torch.abs(x_true_flat.unsqueeze(-1) - constellation)  # [B*N_eff, 4]
        labels = torch.argmin(dist, dim=-1)  # [B*N_eff]

        # cross entropy
        logits_flat = logits_eff.reshape(-1, 4)
        if reduction == "none":
            loss = F.cross_entropy(logits_flat, labels, reduction="none")
            batch_size = logits_eff.shape[0]
            num_sym = logits_eff.shape[1]
            return loss.view(batch_size, num_sym).mean(dim=1)
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return F.cross_entropy(logits_flat, labels)


class MSEPlusCrossEntropyLoss(nn.Module):
    """??MSE + QPSK????????

    loss = alpha * MSE(x_hat, x_true) + beta * CE(logits, x_true)
    """

    def __init__(
        self,
        Q: int = 0,
        alpha: float = 1.0,
        beta: float = 0.1,
        layer_weights: str = "linear"
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.layer_weights = layer_weights
        self.mse = MSELoss(Q)
        self.ce = QPSKSymbolCrossEntropyLoss(Q)

    def _get_layer_weights(self, num_layers: int, device: torch.device) -> torch.Tensor:
        if num_layers <= 0:
            return torch.tensor([], device=device)
        if self.layer_weights is None:
            weights = torch.ones(num_layers, device=device)
        elif self.layer_weights == "linear":
            weights = torch.linspace(1.0, float(num_layers), num_layers, device=device)
        elif self.layer_weights == "exp":
            weights = torch.exp(torch.linspace(0.0, 1.0, num_layers, device=device))
        else:
            raise ValueError(f"Unsupported layer_weights: {self.layer_weights}")
        return weights / weights.sum()

    def _mse_per_sample(self, x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        if self.mse.Q > 0:
            x_hat_eff = x_hat[:, self.mse.Q:-self.mse.Q, :]
            x_true_eff = x_true[:, self.mse.Q:-self.mse.Q, :]
        else:
            x_hat_eff = x_hat
            x_true_eff = x_true

        diff_real = x_hat_eff.real - x_true_eff.real
        diff_imag = x_hat_eff.imag - x_true_eff.imag
        return diff_real.pow(2).mean(dim=(1, 2)) + diff_imag.pow(2).mean(dim=(1, 2))

    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor, logits, sample_weights: torch.Tensor = None) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            if len(logits) == 0:
                raise ValueError("logits list is empty.")
            layer_w = self._get_layer_weights(len(logits), logits[0].device)
            ce_per = 0.0
            for idx, logit in enumerate(logits):
                ce_per = ce_per + layer_w[idx] * self.ce(logit, x_true, reduction="none")
        else:
            ce_per = self.ce(logits, x_true, reduction="none")

        if sample_weights is None:
            mse = self.mse(x_hat, x_true)
            ce = ce_per.mean()
            return self.alpha * mse + self.beta * ce

        weights = sample_weights.to(ce_per.device)
        weights = weights / (weights.mean() + 1e-8)

        mse_per = self._mse_per_sample(x_hat, x_true)
        mse = (mse_per * weights).mean()
        ce = (ce_per * weights).mean()
        return self.alpha * mse + self.beta * ce


if __name__ == "__main__":
    print("测试损失函数...")
    
    B, N = 4, 256
    Q = 10
    
    # 生成测试数据
    x_true = torch.randn(B, N, 1, dtype=torch.complex64)
    x_hat = x_true + 0.1 * torch.randn(B, N, 1, dtype=torch.complex64)
    
    # 测试MSE损失
    mse_loss = MSELoss(Q)
    loss_mse = mse_loss(x_hat, x_true)
    print(f"MSE Loss: {loss_mse.item():.6f}")
    
    # 测试符号错误损失
    ser_loss = SymbolErrorLoss(Q)
    loss_ser = ser_loss(x_hat, x_true)
    print(f"SER Loss: {loss_ser.item():.6f}")
    
    # 测试BER感知损失
    ber_loss = BERAwareLoss(Q)
    loss_ber = ber_loss(x_hat, x_true)
    print(f"BER-Aware Loss: {loss_ber.item():.6f}")
    
    # 测试自适应损失
    adaptive_loss = AdaptiveBERAwareLoss(Q)
    for epoch in [0, 10, 20, 30]:
        adaptive_loss.set_epoch(epoch)
        loss = adaptive_loss(x_hat, x_true)
        alpha, beta = adaptive_loss.get_weights()
        print(f"Epoch {epoch}: Loss={loss.item():.6f}, alpha={alpha:.2f}, beta={beta:.2f}")
    
    print("\n测试完成！")
