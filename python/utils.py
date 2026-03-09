"""
============================================================================
AFDM 工具函数模块
============================================================================
提供AFDM系统的通用工具函数

主要组件:
  - AFDMDataset: 从 MATLAB v7.3 .mat 文件中读取 AFDM 数据（支持分片加载）
  - lmmse_detector: 传统 LMMSE 检测器（作为基准）
  - calculate_ber: BER 计算函数
  - qam_demod: QPSK 解调函数
============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List
import h5py
import os
import glob


class AFDMDataset(Dataset):
    """
    AFDM 数据集封装

    只从 .mat 中读取无噪声 H, x, snr，噪声在 Python 端按 SNR 重新生成。
    支持加载分片数据集（自动检测 _part1, _part2, ... 文件）
    """

    def __init__(self, mat_file: str, device: Union[torch.device, str] = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # 检测是否有分片文件
        mat_files = self._find_mat_files(mat_file)
        
        if len(mat_files) == 1:
            # 单文件模式
            self._load_single_file(mat_files[0])
        else:
            # 多文件分片模式
            self._load_multiple_files(mat_files)

        print(f"  样本数: {self.num_samples}")
        print(f"  矩阵维度: {self.N}×{self.N}")
        print(f"  有效子载波: {self.N_eff}")
        print(f"  SNR范围: {self.snr.min().item():.0f} ~ {self.snr.max().item():.0f} dB")

    def _find_mat_files(self, mat_file: str) -> List[str]:
        """
        查找所有相关的 .mat 文件
        如果指定的文件存在，直接返回
        否则查找 _part1, _part2, ... 分片文件
        """
        if os.path.exists(mat_file):
            # 检查是否有对应的分片文件
            base_name = mat_file.replace('.mat', '')
            part_pattern = f"{base_name}_part*.mat"
            part_files = sorted(glob.glob(part_pattern))
            
            if part_files:
                print(f"发现 {len(part_files)} 个分片文件，将合并加载")
                return part_files
            else:
                return [mat_file]
        else:
            # 尝试查找分片文件
            base_name = mat_file.replace('.mat', '')
            part_pattern = f"{base_name}_part*.mat"
            part_files = sorted(glob.glob(part_pattern))
            
            if part_files:
                print(f"发现 {len(part_files)} 个分片文件，将合并加载")
                return part_files
            else:
                raise FileNotFoundError(f"找不到数据文件: {mat_file} 或其分片文件")

    def _load_single_file(self, mat_file: str):
        """加载单个 .mat 文件"""
        print(f"加载数据: {mat_file}")
        
        with h5py.File(mat_file, "r") as f:
            H_np, x_np, snr_np = self._read_data_from_file(f)

        self._finalize_data(H_np, x_np, snr_np)

    def _load_multiple_files(self, mat_files: List[str]):
        """加载多个分片 .mat 文件并合并"""
        print(f"加载 {len(mat_files)} 个分片文件:")
        
        H_list = []
        x_list = []
        snr_list = []
        
        for i, mat_file in enumerate(mat_files):
            print(f"  [{i+1}/{len(mat_files)}] {os.path.basename(mat_file)}")
            
            with h5py.File(mat_file, "r") as f:
                H_np, x_np, snr_np = self._read_data_from_file(f)
                H_list.append(H_np)
                x_list.append(x_np)
                snr_list.append(snr_np)
        
        # 合并所有分片
        H_np = np.concatenate(H_list, axis=0)
        x_np = np.concatenate(x_list, axis=0)
        snr_np = np.concatenate(snr_list, axis=0)
        
        print(f"  合并完成，总样本数: {H_np.shape[0]}")
        
        self._finalize_data(H_np, x_np, snr_np)

    def _read_data_from_file(self, f: h5py.File):
        """从单个 HDF5 文件中读取数据"""
        def _read_complex(name: str):
            dset = f[name]
            arr = dset[()]
            if hasattr(arr, "dtype") and arr.dtype.names is not None:
                real = arr["real"]
                imag = arr["imag"]
                arr = real + 1j * imag
            return np.array(arr)

        sys_grp = f["system_params"]
        self.N = int(np.array(sys_grp["N"][()])[0, 0])
        self.N_eff = int(np.array(sys_grp["N_eff"][()])[0, 0])
        self.Q = int(np.array(sys_grp["Q"][()])[0, 0])

        H_np = _read_complex("H_dataset")
        x_np = _read_complex("x_dataset")
        snr_np = np.array(f["snr_dataset"][()]).reshape(-1)

        # 重排到 [S, N, N] / [S, N, 1]
        def _reorder_H(arr, N):
            s = arr.shape
            if len(s) == 3 and s[1] == N and s[2] == N:
                return np.transpose(arr, (0, 2, 1))
            if len(s) == 3 and s[0] == N and s[1] == N:
                arr = np.transpose(arr, (2, 0, 1))
                return np.transpose(arr, (0, 2, 1))
            raise ValueError(f"Unexpected H_dataset shape {s} for N={N}")

        def _reorder_vec(arr, N):
            s = arr.shape
            if len(s) == 3:
                if s[1] == 1 and s[2] == N:
                    return np.transpose(arr, (0, 2, 1))
                if s[1] == N and s[2] == 1:
                    return arr
                if s[0] == N and s[1] == 1:
                    return np.transpose(arr, (2, 0, 1))
                if s[0] == N and s[2] == 1:
                    return np.transpose(arr, (1, 0, 2))
            raise ValueError(f"Unexpected vec shape {s} for N={N}")

        H_np = _reorder_H(H_np, self.N)
        x_np = _reorder_vec(x_np, self.N)

        return H_np, x_np, snr_np

    def _finalize_data(self, H_np, x_np, snr_np):
        """将 numpy 数据转换为 torch tensor"""
        H_np = H_np.astype(np.complex64)
        x_np = x_np.astype(np.complex64)
        snr_np = snr_np.astype(np.float32)

        self.H = torch.from_numpy(H_np).to(self.device)
        self.x = torch.from_numpy(x_np).to(self.device)
        self.snr = torch.from_numpy(snr_np).to(self.device)
        self.noise_var = 1.0 / (10.0 ** (self.snr / 10.0))

        self.num_samples = self.H.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        H = self.H[idx]            # [N,N]
        x = self.x[idx]            # [N,1]
        snr = self.snr[idx]        # scalar
        noise_var = self.noise_var[idx]

        sigma = torch.sqrt(noise_var / 2.0)
        noise_real = torch.randn_like(x.real)
        noise_imag = torch.randn_like(x.real)
        w = sigma * (noise_real + 1j * noise_imag)
        y = torch.matmul(H, x) + w

        return {
            "H": H,
            "y": y,
            "x": x,
            "snr": snr,
            "noise_var": noise_var,
        }


def lmmse_detector(y: torch.Tensor, H: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """
    LMMSE 检测器
    y: [B,N,1], H: [B,N,N], noise_var: [B] or scalar
    返回: x_hat [B,N,1]
    """
    B, N, _ = H.shape
    if isinstance(noise_var, float):
        noise_var = torch.tensor(noise_var, device=H.device, dtype=torch.float32)

    if noise_var.dim() == 0:
        noise_var = noise_var.expand(B)

    x_list = []
    for i in range(B):
        H_i = H[i]
        y_i = y[i]
        sigma2 = noise_var[i].item()
        HH = torch.matmul(H_i.conj().t(), H_i)
        HH_reg = HH + sigma2 * torch.eye(N, dtype=H.dtype, device=H.device)
        Hy = torch.matmul(H_i.conj().t(), y_i)
        x_i = torch.linalg.solve(HH_reg, Hy)
        x_list.append(x_i.unsqueeze(0))

    return torch.cat(x_list, dim=0)


def calculate_ber(x_hat: torch.Tensor, x_true: torch.Tensor, Q: int, mod_order: int = 4) -> float:
    """
    计算 BER（仅有效子载波）
    """
    if Q > 0:
        x_hat_eff = x_hat[:, Q:-Q, :]
        x_true_eff = x_true[:, Q:-Q, :]
    else:
        x_hat_eff = x_hat
        x_true_eff = x_true

    data_hat = qam_demod(x_hat_eff, mod_order)
    data_true = qam_demod(x_true_eff, mod_order)

    errors = (data_hat != data_true).sum().item()
    total_bits = data_hat.numel() * int(np.log2(mod_order))
    return errors / total_bits


def qam_demod(x: torch.Tensor, mod_order: int) -> torch.Tensor:
    """
    QAM 硬判决解调
    返回整数标签 [B, N_eff, 1]
    """
    if mod_order != 4:
        raise NotImplementedError("Only QPSK (mod_order=4) is supported currently.")

    qpsk_const = torch.tensor(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=x.dtype,
        device=x.device,
    ) / np.sqrt(2.0)

    x_flat = x.reshape(-1, 1)           # [B*N_eff,1]
    const = qpsk_const.view(1, -1)      # [1,4]

    dist = torch.abs(x_flat - const)    # [B*N_eff,4]
    idx = torch.argmin(dist, dim=1)     # [B*N_eff]
    data = idx.view(x.shape[0], -1, 1)
    return data
