"""
============================================================================
OAMPNetV4 训练脚本
============================================================================
基于正确OAMP实现的深度展开网络训练

用法:
    python python/training/train_oampnet_v4.py --version tsv1 --epochs 50 --batch_size 64 --lr 1e-2 --num_layers 10 --device cuda

============================================================================
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import AFDMDataset, calculate_ber, lmmse_detector
from models.oampnet_v4 import OAMPNetV4
from models.oamp_final import OAMPFinal
from training.losses import MSEPlusCrossEntropyLoss


def compute_snr_weights(snr_db, mode, ref_db, scale, max_w):
    if mode == "none":
        return None
    if mode == "linear":
        weights = 1.0 + scale * (snr_db - ref_db)
    elif mode == "exp":
        weights = torch.exp(scale * (snr_db - ref_db))
    else:
        raise ValueError(f"Unsupported snr_weighting: {mode}")
    weights = torch.clamp(weights, min=1.0, max=max_w)
    return weights



def train_one_epoch(model, dataloader, optimizer, criterion, device, Q, snr_weight_cfg=None, deep_supervision=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ber = 0.0
    num_batches = 0

    for batch in dataloader:
        H = batch["H"].to(device)
        y = batch["y"].to(device)
        x_true = batch["x"].to(device)
        noise_var = batch["noise_var"].to(device)
        snr_db = batch["snr"].to(device)

        x_hat, logits = model(
            y, H, noise_var, Q, return_posterior=True, return_all_logits=deep_supervision
        )
        snr_weights = None
        if snr_weight_cfg is not None:
            snr_weights = compute_snr_weights(
                snr_db,
                snr_weight_cfg["mode"],
                snr_weight_cfg["ref_db"],
                snr_weight_cfg["scale"],
                snr_weight_cfg["max_w"],
            )
        loss = criterion(x_hat, x_true, logits, sample_weights=snr_weights)
        mse_loss = criterion.mse(x_hat, x_true)

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        with torch.no_grad():
            ber = calculate_ber(x_hat, x_true, Q)
            total_ber += ber
        num_batches += 1

    return total_loss / num_batches, total_mse / num_batches, total_ber / num_batches


def validate(model, dataloader, criterion, device, Q, oamp_detector=None, deep_supervision=False):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ber = 0.0
    total_ber_lmmse = 0.0
    total_ber_oamp = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            H = batch["H"].to(device)
            y = batch["y"].to(device)
            x_true = batch["x"].to(device)
            noise_var = batch["noise_var"].to(device)

            # OAMPNetV4
            x_hat, logits = model(
                y, H, noise_var, Q, return_posterior=True, return_all_logits=deep_supervision
            )
            loss = criterion(x_hat, x_true, logits)
            mse_loss = criterion.mse(x_hat, x_true)
            total_loss += loss.item()
            total_mse += mse_loss.item()
            ber = calculate_ber(x_hat, x_true, Q)
            total_ber += ber

            # LMMSE基准
            x_lmmse = lmmse_detector(y, H, noise_var)
            ber_lmmse = calculate_ber(x_lmmse, x_true, Q)
            total_ber_lmmse += ber_lmmse

            # OAMP基准
            if oamp_detector is not None:
                x_oamp = oamp_detector.detect(y, H, noise_var, Q)
                ber_oamp = calculate_ber(x_oamp, x_true, Q)
                total_ber_oamp += ber_oamp

            num_batches += 1

    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_ber / num_batches,
        total_ber_lmmse / num_batches,
        total_ber_oamp / num_batches if oamp_detector else 0.0
    )


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history["train_loss"], label="Train MSE", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val MSE", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Training and Validation MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # BER曲线
    axes[1].semilogy(history["train_ber"], label="Train BER (OAMPNetV4)", linewidth=2)
    axes[1].semilogy(history["val_ber"], label="Val BER (OAMPNetV4)", linewidth=2)
    axes[1].semilogy(history["val_ber_oamp"], "--", label="Val BER (OAMP)", linewidth=2)
    axes[1].semilogy(history["val_ber_lmmse"], ":", label="Val BER (LMMSE)", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BER")
    axes[1].set_title("BER Performance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n训练曲线已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="OAMPNetV4 训练")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--version", type=str, default="", help="数据集版本后缀，如 'v2'")
    parser.add_argument("--ce_alpha", type=float, default=1.0)
    parser.add_argument("--ce_beta", type=float, default=0.05)
    parser.add_argument("--ce_warmup_epochs", type=int, default=15)
    parser.add_argument("--deep_supervision", action="store_true")
    parser.add_argument("--layer_weights", type=str, default="none", choices=["linear", "exp", "none"])
    parser.add_argument("--snr_weighting", type=str, default="none", choices=["none", "linear", "exp"])
    parser.add_argument("--snr_weight_ref", type=float, default=12.0)
    parser.add_argument("--snr_weight_scale", type=float, default=0.12)
    parser.add_argument("--snr_weight_max", type=float, default=4.0)
    args = parser.parse_args()

    # 设置数据目录
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, "..", "..", "data")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("\n========================================")
    print("加载数据集")
    print("========================================")
    
    # 构建数据集文件名（支持版本后缀）
    version_str = f"_{args.version}" if args.version else ""
    train_path = os.path.join(args.data_dir, f"afdm_n256_train{version_str}.mat")
    val_path = os.path.join(args.data_dir, f"afdm_n256_val{version_str}.mat")
    
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path}")
    
    train_dataset = AFDMDataset(train_path, device=device)
    val_dataset = AFDMDataset(val_path, device=device)

    N = train_dataset.N
    Q = train_dataset.Q

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n========================================")
    print("创建模型")
    print("========================================")
    
    model = OAMPNetV4(N=N, num_layers=args.num_layers).to(device)
    oamp_detector = OAMPFinal(num_iterations=args.num_layers, damping=0.9)
    
    # 组合损失：MSE（稳定收敛） + 符号交叉熵（更贴近 BER 目标）
    # beta 建议从 0.05~0.2 之间试起；若高 SNR 提升明显但低 SNR 不稳，可减小 beta。
    layer_weights = None if args.layer_weights == "none" else args.layer_weights
    criterion = MSEPlusCrossEntropyLoss(
        Q=Q,
        alpha=args.ce_alpha,
        beta=args.ce_beta,
        layer_weights=layer_weights,
    )
    snr_weight_cfg = None
    if args.snr_weighting != "none":
        snr_weight_cfg = {
            "mode": args.snr_weighting,
            "ref_db": args.snr_weight_ref,
            "scale": args.snr_weight_scale,
            "max_w": args.snr_weight_max,
        }
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {
        "train_loss": [],
        "train_loss_total": [],
        "train_ber": [],
        "val_loss": [],
        "val_loss_total": [],
        "val_ber": [],
        "val_ber_lmmse": [],
        "val_ber_oamp": [],
    }

    best_val_ber = float("inf")
    best_epoch = 0

    print("\n========================================")
    print("开始训练")
    print("========================================\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        if args.ce_warmup_epochs > 0:
            warm_ratio = min(1.0, epoch / args.ce_warmup_epochs)
            criterion.beta = args.ce_beta * warm_ratio
        else:
            criterion.beta = args.ce_beta

        train_loss_total, train_loss_mse, train_ber = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            Q,
            snr_weight_cfg,
            args.deep_supervision,
        )
        val_loss_total, val_loss_mse, val_ber, val_ber_lmmse, val_ber_oamp = validate(
            model,
            val_loader,
            criterion,
            device,
            Q,
            oamp_detector,
            args.deep_supervision,
        )

        scheduler.step(val_loss_mse)

        history["train_loss"].append(train_loss_mse)
        history["train_loss_total"].append(train_loss_total)
        history["train_ber"].append(train_ber)
        history["val_loss"].append(val_loss_mse)
        history["val_loss_total"].append(val_loss_total)
        history["val_ber"].append(val_ber)
        history["val_ber_lmmse"].append(val_ber_lmmse)
        history["val_ber_oamp"].append(val_ber_oamp)

        if val_ber < best_val_ber:
            best_val_ber = val_ber
            best_epoch = epoch
            model_name = f"best_oampnet_v4{version_str}_model.pth"
            save_path = os.path.join(args.data_dir, model_name)
            torch.save(model.state_dict(), save_path)

        ep_time = time.time() - ep_start
        
        # 计算相对于LMMSE和OAMP的改进
        improve_lmmse = (val_ber_lmmse - val_ber) / val_ber_lmmse * 100 if val_ber_lmmse > 0 else 0
        improve_oamp = (val_ber_oamp - val_ber) / val_ber_oamp * 100 if val_ber_oamp > 0 else 0
        
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Time: {ep_time:.1f}s | "
            f"Train BER: {train_ber:.4e} | "
            f"Val BER: {val_ber:.4e} | "
            f"OAMP: {val_ber_oamp:.4e} | "
            f"vs LMMSE: {improve_lmmse:+.1f}% | "
            f"vs OAMP: {improve_oamp:+.1f}%"
        )

        # 早停
        if epoch - best_epoch > 15:
            print(f"\n早停: 验证集BER已经 {epoch - best_epoch} 轮没有改善")
            break

    total_time = time.time() - start_time
    print("\n========================================")
    print("训练完成")
    print("========================================")
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳验证BER: {best_val_ber:.4e}")
    print(f"OAMP基准: {history['val_ber_oamp'][best_epoch-1]:.4e}")
    print(f"LMMSE基准: {history['val_ber_lmmse'][best_epoch-1]:.4e}")
    
    improve_lmmse = (history['val_ber_lmmse'][best_epoch-1] - best_val_ber) / history['val_ber_lmmse'][best_epoch-1] * 100
    improve_oamp = (history['val_ber_oamp'][best_epoch-1] - best_val_ber) / history['val_ber_oamp'][best_epoch-1] * 100
    print(f"相对LMMSE改善: {improve_lmmse:.1f}%")
    print(f"相对OAMP改善: {improve_oamp:.1f}%")

    # 保存训练历史
    history_name = f"training_history_v4{version_str}.npz"
    history_path = os.path.join(args.data_dir, history_name)
    np.savez(history_path, **history)
    
    # 绘制训练曲线
    plot_name = f"training_curves_v4{version_str}.png"
    plot_path = os.path.join(args.data_dir, "..", "results", plot_name)
    plot_training_curves(history, plot_path)
    
    # 打印学习到的参数
    print("\n学习到的参数:")
    params = model.get_learned_params()
    print(f"  gamma: {params['gamma']}")
    print(f"  damping: {params['damping']}")
    print(f"  temperature: {params['temperature']}")


if __name__ == "__main__":
    main()
