"""
============================================================================
模型快速验证脚本
============================================================================
用于验证训练好的深度学习模型在测试集上的性能

用法:
    python validate_model.py --model oampnet --device cuda
    python validate_model.py --model cnn --device cuda
============================================================================
"""

import argparse
import numpy as np
import torch

from utils import AFDMDataset, calculate_ber, lmmse_detector
from models.oampnet import OAMPNet
from models.cnn_detector import CNNDetector


def validate_model(model, dataset, device, Q, model_name="Model"):
    """
    验证模型在不同SNR下的BER性能
    
    Args:
        model: 深度学习模型
        dataset: 测试数据集
        device: 计算设备
        Q: 保护子载波数
        model_name: 模型名称（用于打印）
    """
    model.eval()
    
    snr_unique = torch.unique(dataset.snr).cpu().numpy()
    snr_unique = np.sort(snr_unique)
    
    print("\n" + "=" * 60)
    print(f"{model_name} 性能验证")
    print("=" * 60)
    print(f"{'SNR (dB)':>10} | {model_name:>12} | {'LMMSE':>12}")
    print("-" * 40)
    
    with torch.no_grad():
        for snr_db in snr_unique:
            idx = (dataset.snr == snr_db).nonzero(as_tuple=True)[0]
            
            H = dataset.H[idx].to(device)
            x = dataset.x[idx].to(device)
            noise_var = dataset.noise_var[idx].to(device)
            
            # 生成接收信号
            sigma = torch.sqrt(noise_var.view(-1, 1, 1) / 2.0)
            w = sigma * (torch.randn_like(x.real) + 1j * torch.randn_like(x.real))
            y = torch.matmul(H, x) + w
            
            # 模型检测
            if model_name == "CNN":
                x_hat = model(y, H)
            else:
                x_hat = model(y, H, noise_var)
            ber_model = calculate_ber(x_hat, x, Q)
            
            # LMMSE基准
            x_hat_lmmse = lmmse_detector(y, H, noise_var)
            ber_lmmse = calculate_ber(x_hat_lmmse, x, Q)
            
            print(f"{snr_db:10.0f} | {ber_model:12.4e} | {ber_lmmse:12.4e}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="模型快速验证")
    parser.add_argument("--model", type=str, default="oampnet",
                        choices=["oampnet", "cnn"], help="模型类型")
    parser.add_argument("--data", type=str, default="../data/afdm_n256_test.mat",
                        help="测试数据文件")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型文件路径")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载测试数据...")
    dataset = AFDMDataset(args.data, device=device)
    N = dataset.N
    Q = dataset.Q
    print(f"测试样本数: {len(dataset)}, N={N}, Q={Q}")
    
    # 加载模型
    if args.model == "oampnet":
        model_path = args.model_path or "../data/best_oampnet_model.pth"
        model = OAMPNet(N=N, num_layers=10).to(device)
        model_name = "OAMP-Net"
    else:
        model_path = args.model_path or "../data/best_cnn_model.pth"
        model = CNNDetector(N=N, hidden_channels=64).to(device)
        model_name = "CNN"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"模型已加载: {model_path}")
    except FileNotFoundError:
        print(f"错误: 模型文件不存在 ({model_path})")
        return
    
    # 验证
    validate_model(model, dataset, device, Q, model_name)
    print("\n验证完成！")


if __name__ == "__main__":
    main()
