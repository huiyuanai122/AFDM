"""
============================================================================
OAMPNetV4: 基于正确OAMP实现的深度展开网络
============================================================================
基于对OAMP算法的详细分析和调试，实现正确的深度展开版本。

关键改进（相比V3）：
1. 使用固定的noise_var作为正则化项（更稳定）
2. 使用正确的方差传递公式
3. 移除Onsager校正（分析表明不使用效果更好）
4. 简化参数结构，只保留有效的可学习参数

可学习参数：
1. gamma: 步长参数，控制线性估计器的更新幅度
2. theta: 阻尼系数，稳定收敛
3. temperature: NLE温度，控制软/硬判决程度

参考: oamp_final.py 中验证过的正确OAMP实现
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class OAMPNetV4(nn.Module):
    """
    基于正确OAMP实现的深度展开网络
    
    与oamp_final.py保持一致的算法结构，只将固定参数替换为可学习参数
    """
    
    def __init__(
        self,
        N: int,
        num_layers: int = 10,
        init_gamma: float = 1.0,
        init_damping: float = 0.9,
        init_temperature: float = 1.0
    ):
        """
        初始化OAMPNetV4
        
        Args:
            N: 符号维度
            num_layers: 展开层数
            init_gamma: 步长初始值
            init_damping: 阻尼系数初始值
            init_temperature: 温度初始值
        """
        super().__init__()
        self.N = N
        self.num_layers = num_layers
        
        # ===== 可学习参数 =====
        
        # 1. 步长参数 γ^t（线性估计器）
        #    控制 r = x + γ * W * (y - H*x) 中的步长
        self.gamma = nn.Parameter(init_gamma * torch.ones(num_layers))
        
        # 2. 阻尼系数 θ^t
        #    控制 x = θ * x_new + (1-θ) * x_prev
        #    使用sigmoid确保在(0,1)范围内
        #    初始化使得sigmoid(theta_raw) ≈ init_damping
        init_theta_raw = torch.log(torch.tensor(init_damping / (1 - init_damping + 1e-6)))
        self.theta_raw = nn.Parameter(init_theta_raw * torch.ones(num_layers))
        
        # 3. NLE温度参数 T^t
        #    控制MMSE估计器的软/硬程度
        #    使用softplus确保正值
        self.temperature_raw = nn.Parameter(torch.zeros(num_layers))
        self._init_temperature = init_temperature
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"OAMPNetV4: N={N}, 层数={num_layers}, 参数量={total_params}")
    
    def get_damping(self, t: int) -> torch.Tensor:
        """获取第t层的阻尼系数"""
        return torch.sigmoid(self.theta_raw[t])
    
    def get_temperature(self, t: int) -> torch.Tensor:
        """获取第t层的温度参数"""
        # softplus确保正值，加上初始温度偏移
        return self._init_temperature * (0.5 + torch.nn.functional.softplus(self.temperature_raw[t]))
    
    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        noise_var: Optional[torch.Tensor] = None,
        Q: int = 0,
        return_posterior: bool = False,
        return_all_logits: bool = False,
    ):
        """
        深度展开OAMP前向传播
        
        与oamp_final.py保持一致的算法结构
        
        Args:
            y: 接收信号 [B, N, 1], complex
            H: 信道矩阵 [B, N, N], complex
            noise_var: 噪声方差 [B], float
            Q: 保护子载波数
            
        Returns:
            x_hat: 估计符号 [B, N, 1], complex

            若 return_posterior=True，则返回 (x_hat, logits)
            - logits: [B, N, 4]，对应 QPSK 四个星座点的未归一化对数似然（可直接做交叉熵）
        """
        B, N, _ = y.shape
        device = y.device
        dtype = H.dtype
        
        # 默认噪声方差
        if noise_var is None:
            noise_var = 0.1 * torch.ones(B, device=device)
        
        # ===== 预计算（与oamp_final.py一致）=====
        H_H = H.conj().transpose(1, 2)
        HH = torch.matmul(H_H, H)
        
        # 使用固定的noise_var作为正则化（关键！）
        reg = noise_var.view(B, 1, 1) * torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        W = torch.linalg.solve(HH + reg, H_H)
        
        # 计算 WH = W H（后续用于等效误差传播）
        WH = torch.matmul(W, H)

        # alpha = trace(WH)/N，用于线性模块的“去相关/去偏置”归一化
        # 说明：在 DAFT/DAF 域等效信道下，H 往往不满足 i.i.d. 假设，
        # 直接 r = x + gamma * W * (y - Hx) 会引入系统性缩放偏差。
        # 用 alpha 归一化相当于把线性模块更贴近 OAMP/VAMP 的正交化假设。
        alpha = torch.diagonal(WH, dim1=1, dim2=2).real.sum(dim=1) / N
        alpha = torch.clamp(alpha, min=1e-6)
        
        # ===== 初始化 =====
        x = torch.zeros(B, N, 1, dtype=dtype, device=device)
        v = torch.ones(B, device=device)  # 信号方差
        
        logits_last = None
        logits_per_layer = [] if (return_posterior and return_all_logits) else None

        # ===== OAMP迭代展开 =====
        for t in range(self.num_layers):
            x_prev = x.clone()
            v_prev = v.clone()
            
            # ----- 1. 线性估计器 -----
            # r = x + γ * W * (y - H*x)
            residual = y - torch.matmul(H, x)
            gamma_t = self.gamma[t]
            step = (gamma_t / alpha).view(B, 1, 1)
            r = x + step * torch.matmul(W, residual)
            
            # ----- 2. 方差传递（向量 tau，更贴 DAFT/DAF 域的非均匀耦合结构）-----
            # 推导假设：
            #   e = x - x_true ~ CN(0, v I)
            #   w ~ CN(0, noise_var I)
            #   r - x_true = (I - γ W H) e + (γ W) w
            # 则逐符号等效方差：
            #   tau_i = v * ||row_i(I - γ W H)||^2 + noise_var * ||row_i(γ W)||^2
            I = torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
            # 注意：与线性更新保持一致，必须使用同一个 step
            A = I - step * WH                            # [B,N,N]
            Bmat = step * W                               # [B,N,N]
            row_energy_A = torch.sum(torch.abs(A) ** 2, dim=2).real   # [B,N]
            row_energy_B = torch.sum(torch.abs(Bmat) ** 2, dim=2).real # [B,N]

            tau_vec = v_prev.view(B, 1) * row_energy_A + noise_var.view(B, 1) * row_energy_B
            tau_vec = torch.clamp(tau_vec, min=1e-10)
            
            # ----- 3. 非线性估计器 (MMSE) -----
            temp = self.get_temperature(t)
            # 仅在最后一层（训练时）输出 logits，用于 CE/NLL 类损失
            want_logits = return_posterior and (return_all_logits or t == self.num_layers - 1)
            x_nle, v_sym, logits = self._mmse_qpsk(r, tau_vec, temp, return_logits=want_logits)

            if Q > 0:
                v_nle = v_sym[:, Q:-Q, :].mean(dim=(1, 2))
            else:
                v_nle = v_sym.mean(dim=(1, 2))

            if want_logits:
                if return_all_logits:
                    logits_per_layer.append(logits)
                else:
                    logits_last = logits
            
            # ----- 4. 阻尼更新 -----
            damping = self.get_damping(t)
            x = damping * x_nle + (1 - damping) * x_prev
            v = damping * v_nle + (1 - damping) * v_prev
            
            # ----- 5. 保护子载波置零 -----
            if Q > 0:
                x[:, :Q, :] = 0
                x[:, -Q:, :] = 0

        if return_posterior:
            if return_all_logits:
                return x, logits_per_layer
            return x, logits_last
        return x
    
    def _mmse_qpsk(
        self,
        r: torch.Tensor,
        tau: torch.Tensor,
        temperature: torch.Tensor,
        return_logits: bool = False,
    ):
        """
        QPSK MMSE估计器（与oamp_final.py一致）
        
        Args:
            r: 线性估计器输出 [B, N, 1]
        tau: 等效噪声方差
             - 支持 [B]（标量）或 [B,N]（向量）
            temperature: 温度参数 (标量)

        return_logits:
            若为 True，则额外返回 logits: [B,N,4]
            
        Returns:
            x_mmse: MMSE估计 [B, N, 1]
            v_mmse: 后验方差 [B]
        """
        B, N, _ = r.shape
        device = r.device
        
        # QPSK星座点
        qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=r.dtype, device=device) / np.sqrt(2)
        
        # 有效方差 = tau * temperature
        if tau.dim() == 1:
            tau_eff = tau.view(B, 1, 1) * temperature
        elif tau.dim() == 2:
            tau_eff = tau.view(B, N, 1) * temperature
        else:
            # [B,N,1]
            tau_eff = tau * temperature
        
        # 计算到各星座点的距离平方
        dist_sq = torch.zeros(B, N, 4, device=device, dtype=torch.float32)
        for i, s in enumerate(qpsk):
            dist_sq[:, :, i] = torch.abs(r.squeeze(-1) - s) ** 2
        
        # logits（未归一化对数似然），用于 softmax / CE
        logits = -dist_sq / (tau_eff.real + 1e-10)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        prob = torch.softmax(logits, dim=-1)
        
        # MMSE估计
        x_real = torch.sum(prob * qpsk.real, dim=-1, keepdim=True)
        x_imag = torch.sum(prob * qpsk.imag, dim=-1, keepdim=True)
        x_mmse = (x_real + 1j * x_imag).to(r.dtype)
        
        # 后验方差（逐符号）: Var(x|r)=E|x|^2-|E[x]|^2，QPSK下E|x|^2=1
        v_sym = 1.0 - torch.abs(x_mmse) ** 2
        v_sym = torch.clamp(v_sym.real, min=1e-10, max=1.0)

        if return_logits:
            return x_mmse, v_sym, logits
        return x_mmse, v_sym, None
    
    def get_learned_params(self) -> dict:
        """获取学习到的参数"""
        return {
            'gamma': self.gamma.detach().cpu().numpy(),
            'damping': torch.stack([self.get_damping(t) for t in range(self.num_layers)]).detach().cpu().numpy(),
            'temperature': torch.stack([self.get_temperature(t) for t in range(self.num_layers)]).detach().cpu().numpy(),
            'num_layers': self.num_layers,
            'N': self.N
        }


def test_oampnet_v4():
    """测试OAMPNetV4"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import AFDMDataset, calculate_ber, lmmse_detector
    from models.oamp_final import OAMPFinal
    
    print("=" * 70)
    print("OAMPNetV4 测试")
    print("=" * 70)
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../../data/afdm_n256_test.mat')
    dataset = AFDMDataset(data_path, device='cuda')
    Q = dataset.Q
    N = dataset.N
    
    # 创建模型
    model = OAMPNetV4(N=N, num_layers=10).cuda()
    oamp = OAMPFinal(num_iterations=10, damping=0.9)
    
    # 测试
    snr_list = [10, 15, 20, 25]
    
    print(f"\n{'SNR':>6} | {'LMMSE':>10} | {'OAMP':>10} | {'OAMPNetV4':>10}")
    print("-" * 50)
    
    for snr in snr_list:
        idx = (dataset.snr == snr).nonzero(as_tuple=True)[0]
        
        H = dataset.H[idx]
        x_true = dataset.x[idx]
        noise_var = dataset.noise_var[idx]
        
        sigma = torch.sqrt(noise_var.view(-1, 1, 1) / 2.0)
        torch.manual_seed(42)
        w = sigma * (torch.randn_like(x_true.real) + 1j * torch.randn_like(x_true.real))
        y = torch.matmul(H, x_true) + w
        
        # LMMSE
        x_lmmse = lmmse_detector(y, H, noise_var)
        ber_lmmse = calculate_ber(x_lmmse, x_true, Q)
        
        # OAMP
        x_oamp = oamp.detect(y, H, noise_var, Q)
        ber_oamp = calculate_ber(x_oamp, x_true, Q)
        
        # OAMPNetV4 (未训练)
        with torch.no_grad():
            x_net = model(y, H, noise_var, Q)
        ber_net = calculate_ber(x_net, x_true, Q)
        
        print(f"{snr:6.0f} | {ber_lmmse:10.6f} | {ber_oamp:10.6f} | {ber_net:10.6f}")
    
    print("-" * 50)
    print("\n注意: OAMPNetV4未训练时应该与OAMP性能相近")
    
    # 打印参数
    print("\n初始参数:")
    params = model.get_learned_params()
    print(f"  gamma: {params['gamma']}")
    print(f"  damping: {params['damping']}")
    print(f"  temperature: {params['temperature']}")
    
    # 测试梯度
    print("\n测试梯度计算...")
    x_net = model(y[:10], H[:10], noise_var[:10], Q)
    loss = torch.mean(torch.abs(x_net - x_true[:10])**2)
    loss.backward()
    print(f"  gamma.grad: {model.gamma.grad}")
    print(f"  theta_raw.grad: {model.theta_raw.grad}")
    print(f"  temperature_raw.grad: {model.temperature_raw.grad}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_oampnet_v4()
