"""
============================================================================
CNN检测器训练脚本
============================================================================
用法示例:
    python train_cnn.py --epochs 100 --batch_size 64 --lr 1e-3 --device cuda
============================================================================
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '..')
from utils import AFDMDataset, calculate_ber, lmmse_detector
from models.cnn_detector import CNNDetector


def train_one_epoch(model, dataloader, optimizer, device, Q):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_ber = 0.0
    num_batches = 0

    for batch in dataloader:
        H = batch["H"].to(device)
        y = batch["y"].to(device)
        x_true = batch["x"].to(device)

        # CNN检测器不需要噪声方差
        x_hat = model(y, H)

        # 只在有效子载波上计算复数 MSE
        if Q > 0:
            x_hat_eff = x_hat[:, Q:-Q, :]
            x_true_eff = x_true[:, Q:-Q, :]
        else:
            x_hat_eff = x_hat
            x_true_eff = x_true

        diff_r = x_hat_eff.real - x_true_eff.real
        diff_i = x_hat_eff.imag - x_true_eff.imag
        loss = diff_r.pow(2).mean() + diff_i.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            ber = calculate_ber(x_hat, x_true, Q)
            total_ber += ber
        num_batches += 1

    return total_loss / num_batches, total_ber / num_batches


def validate(model, dataloader, device, Q):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_ber = 0.0
    total_ber_lmmse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            H = batch["H"].to(device)
            y = batch["y"].to(device)
            x_true = batch["x"].to(device)
            noise_var = batch["noise_var"].to(device)

            # CNN检测器不需要噪声方差
            x_hat = model(y, H)

            if Q > 0:
                x_hat_eff = x_hat[:, Q:-Q, :]
                x_true_eff = x_true[:, Q:-Q, :]
            else:
                x_hat_eff = x_hat
                x_true_eff = x_true

            diff_r = x_hat_eff.real - x_true_eff.real
            diff_i = x_hat_eff.imag - x_true_eff.imag
            loss = diff_r.pow(2).mean() + diff_i.pow(2).mean()

            total_loss += loss.item()

            ber = calculate_ber(x_hat, x_true, Q)
            total_ber += ber

            # LMMSE作为基准
            x_hat_lmmse = lmmse_detector(y, H, noise_var)
            ber_lmmse = calculate_ber(x_hat_lmmse, x_true, Q)
            total_ber_lmmse += ber_lmmse

            num_batches += 1

    return (
        total_loss / num_batches,
        total_ber / num_batches,
        total_ber_lmmse / num_batches,
    )


def plot_training_curves(history, save_path="cnn_training_curves.png"):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN Detector: Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(history["train_ber"], label="Train BER (CNN)", linewidth=2)
    ax2.semilogy(history["val_ber"], label="Val BER (CNN)", linewidth=2)
    ax2.semilogy(
        history["val_ber_lmmse"], "--", label="Val BER (LMMSE)", linewidth=2
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BER")
    ax2.set_title("CNN Detector: BER Performance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n训练曲线已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CNN检测器训练")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_channels", type=int, default=64, help="隐藏层通道数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--model_path", type=str, default="../../data/best_cnn_model.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    print("\n========================================", flush=True)
    print("加载数据集", flush=True)
    print("========================================", flush=True)
    train_dataset = AFDMDataset(f"{args.data_dir}/afdm_n256_train.mat", device=device)
    val_dataset = AFDMDataset(f"{args.data_dir}/afdm_n256_val.mat", device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    N = train_dataset.N
    Q = train_dataset.Q

    print("\n========================================", flush=True)
    print("创建CNN检测器模型", flush=True)
    print("========================================", flush=True)
    model = CNNDetector(N=N, hidden_channels=args.hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [],
        "train_ber": [],
        "val_loss": [],
        "val_ber": [],
        "val_ber_lmmse": [],
    }

    best_val_ber = float("inf")
    best_epoch = 0

    print("\n========================================", flush=True)
    print("开始训练", flush=True)
    print("========================================\n", flush=True)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        train_loss, train_ber = train_one_epoch(
            model, train_loader, optimizer, device, Q
        )
        val_loss, val_ber, val_ber_lmmse = validate(
            model, val_loader, device, Q
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_ber"].append(train_ber)
        history["val_loss"].append(val_loss)
        history["val_ber"].append(val_ber)
        history["val_ber_lmmse"].append(val_ber_lmmse)

        if val_ber < best_val_ber:
            best_val_ber = val_ber
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)

        ep_time = time.time() - ep_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Time: {ep_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train BER: {train_ber:.4e} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val BER: {val_ber:.4e} | "
            f"LMMSE BER: {val_ber_lmmse:.4e}",
            flush=True
        )

        # 早停
        if epoch - best_epoch > 20:
            print(f"\n早停: 验证集BER已经 {epoch - best_epoch} 轮没有改善", flush=True)
            break

    total_time = time.time() - start_time
    print("\n========================================")
    print("训练完成")
    print("========================================")
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳验证BER: {best_val_ber:.4e}")
    print(f"LMMSE基准: {history['val_ber_lmmse'][best_epoch-1]:.4e}")
    print(f"模型已保存: {args.model_path}")

    np.savez("../../data/cnn_training_history.npz", **history)
    plot_training_curves(history, save_path="../../results/cnn_training_curves.png")


if __name__ == "__main__":
    main()
