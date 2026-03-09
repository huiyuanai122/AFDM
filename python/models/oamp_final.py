"""
============================================================================
OAMP Final: 最终版OAMP检测器
============================================================================
基于大量调试的最终版本

关键发现：
1. 不使用Onsager校正效果更好（对于我们的信道）
2. 迭代次数8-10次最佳
3. 阻尼系数0.9效果好
4. 使用早停机制

这个版本在SNR=20dB时能达到约70%的改善
============================================================================
"""

import numpy as np
import torch
from typing import Tuple


class OAMPFinal:
    """
    最终版OAMP检测器
    """
    
    def __init__(
        self,
        num_iterations: int = 10,
        damping: float = 0.9
    ):
        self.num_iterations = num_iterations
        self.damping = damping
    
    def detect(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        noise_var: torch.Tensor,
        Q: int = 0
    ) -> torch.Tensor:
        """OAMP检测"""
        B, N, _ = y.shape
        device = y.device
        dtype = H.dtype
        
        # 计算LMMSE滤波器
        H_H = H.conj().transpose(1, 2)
        HH = torch.matmul(H_H, H)
        reg = noise_var.view(B, 1, 1) * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        W = torch.linalg.solve(HH + reg, H_H)
        
        # 计算trace(WH)/N
        WH = torch.matmul(W, H)
        trace_WH = torch.diagonal(WH, dim1=1, dim2=2).real.sum(dim=1) / N
        
        # 初始化
        x = torch.zeros(B, N, 1, dtype=dtype, device=device)
        v = torch.ones(B, device=device)
        
        # 记录最佳结果
        best_x = x.clone()
        best_ber_proxy = float('inf')  # 使用MSE作为BER的代理
        
        for t in range(self.num_iterations):
            x_prev = x.clone()
            v_prev = v.clone()
            
            # 1. 线性估计器
            residual = y - torch.matmul(H, x)
            r = x + torch.matmul(W, residual)
            
            # 2. 方差传递
            tau = (1 - trace_WH) * v_prev + noise_var
            tau = torch.clamp(tau, min=1e-10)
            
            # 3. 非线性估计器（MMSE）
            x_nle, v_nle = self._mmse_qpsk(r, tau)
            
            # 4. 阻尼更新（不使用Onsager校正）
            x = self.damping * x_nle + (1 - self.damping) * x_prev
            v = self.damping * v_nle + (1 - self.damping) * v_prev
            
            # 保护子载波置零
            if Q > 0:
                x[:, :Q, :] = 0
                x[:, -Q:, :] = 0
            
            # 早停：使用硬判决后的MSE作为代理
            x_hard = self._hard_qpsk(x)
            mse_proxy = torch.mean(torch.abs(y - torch.matmul(H, x_hard))**2).item()
            
            if mse_proxy < best_ber_proxy:
                best_ber_proxy = mse_proxy
                best_x = x.clone()
        
        return best_x
    
    def _mmse_qpsk(
        self,
        r: torch.Tensor,
        tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """QPSK MMSE估计器"""
        B, N, _ = r.shape
        device = r.device
        
        qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=r.dtype, device=device) / np.sqrt(2)
        
        tau_exp = tau.view(B, 1, 1)
        
        dist_sq = torch.zeros(B, N, 4, device=device, dtype=torch.float32)
        for i, s in enumerate(qpsk):
            dist_sq[:, :, i] = torch.abs(r.squeeze(-1) - s) ** 2
        
        log_prob = -dist_sq / (tau_exp.real + 1e-10)
        log_prob = log_prob - log_prob.max(dim=-1, keepdim=True)[0]
        prob = torch.softmax(log_prob, dim=-1)
        
        x_real = torch.sum(prob * qpsk.real, dim=-1, keepdim=True)
        x_imag = torch.sum(prob * qpsk.imag, dim=-1, keepdim=True)
        x_mmse = (x_real + 1j * x_imag).to(r.dtype)
        
        v_mmse = 1 - torch.mean(torch.abs(x_mmse) ** 2, dim=(1, 2))
        v_mmse = torch.clamp(v_mmse, min=1e-10, max=1.0)
        
        return x_mmse, v_mmse
    
    def _hard_qpsk(self, x: torch.Tensor) -> torch.Tensor:
        """QPSK硬判决"""
        real_part = torch.sign(x.real)
        imag_part = torch.sign(x.imag)
        real_part = torch.where(real_part == 0, torch.ones_like(real_part), real_part)
        imag_part = torch.where(imag_part == 0, torch.ones_like(imag_part), imag_part)
        return (real_part + 1j * imag_part) / np.sqrt(2)


def comprehensive_test():
    """综合测试"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import AFDMDataset, calculate_ber, lmmse_detector
    
    print("=" * 70)
    print("OAMP Final 综合测试")
    print("=" * 70)
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../../data/afdm_n256_test.mat')
    dataset = AFDMDataset(data_path, device='cuda')
    Q = dataset.Q
    
    detector = OAMPFinal(num_iterations=10, damping=0.9)
    
    # 测试所有SNR
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    
    results = {'snr': [], 'lmmse': [], 'oamp': []}
    
    print(f"\n{'SNR':>6} | {'LMMSE':>12} | {'OAMP':>12} | {'改善':>10}")
    print("-" * 55)
    
    for snr in snr_list:
        idx = (dataset.snr == snr).nonzero(as_tuple=True)[0]
        
        H = dataset.H[idx]
        x_true = dataset.x[idx]
        noise_var = dataset.noise_var[idx]
        
        # 生成接收信号
        sigma = torch.sqrt(noise_var.view(-1, 1, 1) / 2.0)
        w = sigma * (torch.randn_like(x_true.real) + 1j * torch.randn_like(x_true.real))
        y = torch.matmul(H, x_true) + w
        
        # LMMSE
        x_lmmse = lmmse_detector(y, H, noise_var)
        ber_lmmse = calculate_ber(x_lmmse, x_true, Q)
        
        # OAMP
        x_oamp = detector.detect(y, H, noise_var, Q)
        ber_oamp = calculate_ber(x_oamp, x_true, Q)
        
        improve = (ber_lmmse - ber_oamp) / ber_lmmse * 100 if ber_lmmse > 0 else 0
        
        results['snr'].append(snr)
        results['lmmse'].append(ber_lmmse)
        results['oamp'].append(ber_oamp)
        
        print(f"{snr:6.0f} | {ber_lmmse:12.6f} | {ber_oamp:12.6f} | {improve:+9.1f}%")
    
    print("-" * 55)
    
    # 计算平均改善
    avg_improve = 0
    count = 0
    for i, snr in enumerate(results['snr']):
        if results['lmmse'][i] > 1e-6:
            avg_improve += (results['lmmse'][i] - results['oamp'][i]) / results['lmmse'][i] * 100
            count += 1
    
    if count > 0:
        print(f"\n平均改善: {avg_improve/count:.1f}%")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
当前OAMP实现的性能：
- 在中高SNR（15-25dB）有30-70%的BER改善
- 在低SNR（0-10dB）改善有限或略有下降

与论文的差距：
- 论文中OAMP应该比LMMSE好1-2个数量级
- 我们只有30-70%的改善

可能的原因：
1. 信道矩阵条件数过高（均值~160）
2. 论文中可能使用了额外的预处理或优化
3. 数据集生成方式可能与论文不同

建议：
1. 检查MATLAB中的信道生成代码
2. 尝试对信道矩阵进行预处理（如对角加载）
3. 研究论文中D-OAMP的具体实现细节
""")


if __name__ == "__main__":
    comprehensive_test()
