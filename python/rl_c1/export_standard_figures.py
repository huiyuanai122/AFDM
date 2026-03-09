"""
Export standardized Fig.1~Fig.11 artifacts for the AFDM C1 project.

Outputs:
- results/fig*.csv
- results/fig*.mat
- figures/fig*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import load_offline_bandit_data
from rl_c1.eval_policy import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export standardized figures (Fig.1~Fig.11).")
    p.add_argument("--data", type=str, default="data/oracle_policy_dataset.mat")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt",
    )
    p.add_argument(
        "--train_history",
        type=str,
        default="results/rl_c1_stage2_full/warmstart_rewardber/train_history.json",
    )
    p.add_argument("--reward_key", type=str, default="reward_ber")
    p.add_argument("--snr_min", type=float, default=None)
    p.add_argument("--snr_max", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--detector_csv", type=str, default="results/ber_results_matlab_tsv1.csv")
    p.add_argument(
        "--online_policy_detector_csv",
        type=str,
        default="results/ber_results_policy_online_oamp_oampnet.csv",
        help="Measured online detector CSV produced by MATLAB policy+detector linkage.",
    )
    p.add_argument(
        "--refine_mat",
        type=str,
        default="matlab/detectors/sanity_check_c1_sweep_refine_result.mat",
    )
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--figures_dir", type=str, default="figures")
    p.add_argument("--example_sequence", type=int, default=-1)
    p.add_argument("--runtime_repeats", type=int, default=40)
    p.add_argument(
        "--channel_metric_idx",
        type=int,
        default=6,
        help="State feature index used in Fig.10 (default=6 means max|alpha| in current exporter).",
    )
    return p.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_mat(path: Path, payload: Dict) -> None:
    out = {}
    for k, v in payload.items():
        if isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = v
    savemat(str(path), out)


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def _read_detector_csv(path: Path) -> Dict[str, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows in detector csv: {path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    return {
        "snr_db": col("SNR"),
        "lmmse": col("LMMSE"),
        "oamp": col("OAMP"),
        "oampnet": col("OAMPNet"),
    }


def _read_online_policy_detector_csv(path: Path) -> Dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    required = ["snr_db", "fixed_oamp", "fixed_oampnet", "rl_oamp", "rl_oampnet", "oracle_oampnet"]
    for k in required:
        if k not in rows[0]:
            raise ValueError(f"Missing column '{k}' in online detector csv: {path}")
    out = {k: col(k) for k in required}
    if "oracle_oamp" in rows[0]:
        out["oracle_oamp"] = col("oracle_oamp")
    return out


def _load_refine_result(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if "result" not in f:
            return {}
        g = f["result"]
        for k in g.keys():
            if isinstance(g[k], h5py.Dataset):
                out[k] = np.array(g[k][()])
    return out


def _align_snr_c1(table: np.ndarray, snr_count: int, c1_count: int) -> np.ndarray:
    t = np.asarray(table)
    if t.shape == (c1_count, snr_count):
        return t
    if t.shape == (snr_count, c1_count):
        return t.T
    raise ValueError(f"Unexpected table shape {t.shape}, expected ({c1_count},{snr_count}) or ({snr_count},{c1_count})")


def _compute_policy_outputs(
    data,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    pack = load_policy(str(checkpoint_path), device)
    model = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6
    x = (data.states.astype(np.float32) - state_mean) / state_std
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(xt).cpu().numpy().astype(np.float64)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    actions = np.argmax(logits, axis=1).astype(np.int64)
    return actions, probs


def _metrics_by_snr(
    ber_actions: np.ndarray,
    base_action: int,
    oracle_actions: np.ndarray,
    policy_actions: np.ndarray,
    snr_db: np.ndarray | None,
) -> List[Dict]:
    if snr_db is None:
        idx_groups = [("ALL", np.arange(ber_actions.shape[0], dtype=np.int64))]
    else:
        idx_groups = []
        for s in np.sort(np.unique(snr_db.astype(np.int64))):
            idx = np.where(snr_db.astype(np.int64) == s)[0]
            if idx.size > 0:
                idx_groups.append((int(s), idx))

    rows = []
    for snr_tag, idx in idx_groups:
        b = ber_actions[idx]
        bp = float(np.mean(b[np.arange(idx.size), policy_actions[idx]]))
        bo = float(np.mean(b[np.arange(idx.size), oracle_actions[idx]]))
        bb = float(np.mean(b[:, base_action]))
        rows.append(
            {
                "snr_db": snr_tag,
                "ber_base": bb,
                "ber_policy": bp,
                "ber_oracle": bo,
                "ratio_policy_to_base": bp / max(bb, 1e-12),
                "ratio_oracle_to_base": bo / max(bb, 1e-12),
            }
        )
    return rows


def _plot_save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_fig1_fig2(
    results_dir: Path,
    figures_dir: Path,
    refine: Dict[str, np.ndarray],
    fallback_data,
) -> Dict[str, str]:
    # Prefer MATLAB refine output; fallback to dataset-derived sweep.
    if refine:
        c1_grid = np.asarray(refine["c1_grid"]).reshape(-1)
        snr_list = np.asarray(refine["snr_db_list"]).reshape(-1)
        c1_n = c1_grid.size
        snr_n = snr_list.size
        ber_mean = _align_snr_c1(refine["ber_mean"], snr_n, c1_n)
        ber_std = _align_snr_c1(refine["ber_std"], snr_n, c1_n)
        ber_se = _align_snr_c1(refine["ber_se"], snr_n, c1_n)
        base_idx = int(np.asarray(refine["base_idx"]).reshape(-1)[0])  # likely 1-based
        best_idx_each_snr = np.argmin(ber_mean, axis=0) + 1

        best_idx_per_ch = np.asarray(refine["best_idx_per_ch"])
        if best_idx_per_ch.shape[0] != snr_n:
            best_idx_per_ch = best_idx_per_ch.T
        if best_idx_per_ch.min() < 1:
            best_idx_per_ch = best_idx_per_ch + 1
    else:
        if fallback_data.ber_actions is None or fallback_data.snr_db is None or fallback_data.c1_grid is None:
            raise ValueError("Cannot build Fig.1/2: missing refine mat and dataset fields.")
        c1_grid = fallback_data.c1_grid.reshape(-1)
        snr_list = np.sort(np.unique(fallback_data.snr_db.astype(np.int64)))
        c1_n = c1_grid.size
        snr_n = snr_list.size
        ber_mean = np.zeros((c1_n, snr_n), dtype=np.float64)
        ber_std = np.zeros((c1_n, snr_n), dtype=np.float64)
        ber_se = np.zeros((c1_n, snr_n), dtype=np.float64)
        best_idx_per_ch_rows = []
        for j, s in enumerate(snr_list):
            idx = np.where(fallback_data.snr_db.astype(np.int64) == s)[0]
            b = fallback_data.ber_actions[idx].astype(np.float64)
            ber_mean[:, j] = np.mean(b, axis=0)
            ber_std[:, j] = np.std(b, axis=0, ddof=1)
            ber_se[:, j] = ber_std[:, j] / math.sqrt(max(1, idx.size))
            best_idx_per_ch_rows.append(np.argmin(b, axis=1) + 1)
        best_idx_each_snr = np.argmin(ber_mean, axis=0) + 1
        best_idx_per_ch = np.array(best_idx_per_ch_rows, dtype=np.int64)
        base_idx = int(fallback_data.base_action + 1)

    # Fig.1 CSV + MAT
    rows1 = []
    for j, snr in enumerate(snr_list):
        for i, c1 in enumerate(c1_grid):
            rows1.append(
                {
                    "c1": float(c1),
                    "snr_db": float(snr),
                    "ber_mean": float(ber_mean[i, j]),
                    "ber_std": float(ber_std[i, j]),
                    "ber_se": float(ber_se[i, j]),
                    "is_base": int((i + 1) == base_idx),
                    "is_best": int((i + 1) == int(best_idx_each_snr[j])),
                }
            )
    p_csv1 = results_dir / "fig1_ber_vs_c1.csv"
    _write_csv(
        p_csv1,
        ["c1", "snr_db", "ber_mean", "ber_std", "ber_se", "is_base", "is_best"],
        rows1,
    )
    p_mat1 = results_dir / "fig1_ber_vs_c1.mat"
    _save_mat(
        p_mat1,
        {
            "c1_grid": c1_grid,
            "snr_db_list": snr_list,
            "ber_mean": ber_mean,
            "ber_std": ber_std,
            "ber_se": ber_se,
            "base_idx": base_idx,
            "best_idx_each_snr": best_idx_each_snr,
        },
    )

    fig = plt.figure(figsize=(8.2, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    markers = ["o", "s", "^", "d", "x"]
    for j, snr in enumerate(snr_list):
        ax.semilogy(
            c1_grid,
            ber_mean[:, j],
            marker=markers[j % len(markers)],
            linewidth=1.7,
            label=f"SNR={int(snr)} dB",
        )
        lo = np.maximum(ber_mean[:, j] - ber_se[:, j], 1e-12)
        hi = ber_mean[:, j] + ber_se[:, j]
        ax.fill_between(c1_grid, lo, hi, alpha=0.15)
    ax.axvline(c1_grid[base_idx - 1], color="k", linestyle="--", linewidth=1.2, label="base C1")
    ax.set_xlabel("C1")
    ax.set_ylabel("BER")
    ax.set_title("Fig.1 BER vs Discrete C1")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig1_ber_vs_c1.png")

    # Fig.2 CSV + MAT
    rows2 = []
    action_hist = np.zeros((snr_list.size, c1_grid.size), dtype=np.int64)
    action_prob = np.zeros_like(action_hist, dtype=np.float64)
    for j, snr in enumerate(snr_list):
        idx_arr = best_idx_per_ch[j].reshape(-1).astype(np.int64)
        for a in range(1, c1_grid.size + 1):
            cnt = int(np.sum(idx_arr == a))
            prob = float(cnt / max(1, idx_arr.size))
            action_hist[j, a - 1] = cnt
            action_prob[j, a - 1] = prob
            rows2.append(
                {
                    "snr_db": float(snr),
                    "action_idx": a,
                    "count": cnt,
                    "probability": prob,
                }
            )
    p_csv2 = results_dir / "fig2_best_action_hist.csv"
    _write_csv(p_csv2, ["snr_db", "action_idx", "count", "probability"], rows2)
    p_mat2 = results_dir / "fig2_best_action_hist.mat"
    _save_mat(
        p_mat2,
        {
            "best_action_per_channel": best_idx_per_ch,
            "action_hist": action_hist,
            "action_prob": action_prob,
            "snr_db_list": snr_list,
        },
    )

    fig = plt.figure(figsize=(8.8, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(c1_grid.size)
    bw = 0.8 / max(1, snr_list.size)
    for j, snr in enumerate(snr_list):
        ax.bar(x + (j - (snr_list.size - 1) / 2) * bw, action_prob[j], width=bw, label=f"{int(snr)} dB")
    ax.set_xlabel("Action Index (1-based)")
    ax.set_ylabel("Probability")
    ax.set_title("Fig.2 Oracle-Optimal Action Distribution")
    ax.set_xticks(x[:: max(1, c1_grid.size // 12)])
    ax.set_xticklabels([str(int(v)) for v in (x[:: max(1, c1_grid.size // 12)] + 1)])
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig2_best_action_hist.png")

    return {
        "fig1_csv": str(p_csv1),
        "fig1_mat": str(p_mat1),
        "fig2_csv": str(p_csv2),
        "fig2_mat": str(p_mat2),
    }


def build_fig3_fig4(
    results_dir: Path,
    figures_dir: Path,
    detector: Dict[str, np.ndarray] | None,
    per_snr_metrics: List[Dict],
    online_measured: Dict[str, np.ndarray] | None,
) -> Dict[str, str]:
    if detector is None and online_measured is None:
        raise ValueError("build_fig3_fig4 requires at least detector baseline csv or online measured csv.")

    if online_measured is not None:
        snr = online_measured["snr_db"].astype(np.int64)
        fixed_oamp = online_measured["fixed_oamp"].astype(np.float64)
        fixed_oampnet = online_measured["fixed_oampnet"].astype(np.float64)
        rl_oamp = online_measured["rl_oamp"].astype(np.float64)
        rl_oampnet = online_measured["rl_oampnet"].astype(np.float64)
        oracle_oampnet = online_measured["oracle_oampnet"].astype(np.float64)
        source_note = "rl/oracle detector curves are MATLAB-online measured results."

        fixed_lmmse = np.full_like(fixed_oamp, np.nan, dtype=np.float64)
        if detector is not None:
            det_snr = detector["snr_db"].astype(np.int64)
            det_lmmse = detector["lmmse"].astype(np.float64)
            m = {int(det_snr[i]): float(det_lmmse[i]) for i in range(det_snr.size)}
            for i, s in enumerate(snr):
                if int(s) in m:
                    fixed_lmmse[i] = m[int(s)]
    else:
        snr = detector["snr_db"].astype(np.int64)
        fixed_lmmse = detector["lmmse"].astype(np.float64)
        fixed_oamp = detector["oamp"].astype(np.float64)
        fixed_oampnet = detector["oampnet"].astype(np.float64)

        ratio_policy = {}
        ratio_oracle = {}
        for r in per_snr_metrics:
            if r["snr_db"] == "ALL":
                continue
            s = int(r["snr_db"])
            ratio_policy[s] = float(r["ratio_policy_to_base"])
            ratio_oracle[s] = float(r["ratio_oracle_to_base"])

        rl_oamp = np.full_like(fixed_oamp, np.nan)
        rl_oampnet = np.full_like(fixed_oampnet, np.nan)
        oracle_oampnet = np.full_like(fixed_oampnet, np.nan)
        for i, s in enumerate(snr):
            if int(s) in ratio_policy:
                rl_oamp[i] = fixed_oamp[i] * ratio_policy[int(s)]
                rl_oampnet[i] = fixed_oampnet[i] * ratio_policy[int(s)]
                oracle_oampnet[i] = fixed_oampnet[i] * ratio_oracle[int(s)]
        source_note = "rl/oracle detector curves are estimated by transferring C1 gain ratio from policy dataset at matched SNR."

    methods = {
        "fixed_lmmse": fixed_lmmse,
        "fixed_oamp": fixed_oamp,
        "fixed_oampnet": fixed_oampnet,
        "rl_oamp": rl_oamp,
        "rl_oampnet": rl_oampnet,
        "oracle_oampnet": oracle_oampnet,
    }

    rows3 = []
    for i, s in enumerate(snr):
        for m, y in methods.items():
            rows3.append({"snr_db": int(s), "method": m, "ber": float(y[i])})
    p_csv3 = results_dir / "fig3_ber_vs_snr_main.csv"
    _write_csv(p_csv3, ["snr_db", "method", "ber"], rows3)
    p_mat3 = results_dir / "fig3_ber_vs_snr_main.mat"
    _save_mat(
        p_mat3,
        {
            "snr_db_list": snr,
            "ber_fixed_lmmse": fixed_lmmse,
            "ber_fixed_oamp": fixed_oamp,
            "ber_fixed_oampnet": fixed_oampnet,
            "ber_rl_oamp": rl_oamp,
            "ber_rl_oampnet": rl_oampnet,
            "ber_oracle_oampnet": oracle_oampnet,
            "note": source_note,
        },
    )

    fig = plt.figure(figsize=(9.3, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    if np.any(np.isfinite(fixed_lmmse)):
        ax.semilogy(snr, fixed_lmmse, "-o", label="fixed_lmmse", linewidth=1.7)
    ax.semilogy(snr, fixed_oamp, "-s", label="fixed_oamp", linewidth=1.7)
    ax.semilogy(snr, fixed_oampnet, "-^", label="fixed_oampnet", linewidth=1.7)
    tag = "measured" if online_measured is not None else "est."
    ax.semilogy(snr, rl_oamp, "--s", label=f"rl_oamp ({tag})", linewidth=1.7)
    ax.semilogy(snr, rl_oampnet, "--^", label=f"rl_oampnet ({tag})", linewidth=1.7)
    ax.semilogy(snr, oracle_oampnet, ":^", label=f"oracle_oampnet ({tag})", linewidth=1.9)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Fig.3 BER vs SNR (Main Comparison)")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig3_ber_vs_snr_main.png")

    # Fig.4 ablation: fixed/rl across OAMP and OAMPNet.
    if online_measured is not None:
        snr_targets = [int(s) for s in snr.tolist()]
    else:
        ratio_policy = {int(r["snr_db"]): float(r["ratio_policy_to_base"]) for r in per_snr_metrics if r["snr_db"] != "ALL"}
        snr_targets = [int(s) for s in sorted(ratio_policy.keys()) if int(s) in set(snr.tolist())]
    rows4 = []
    ablation_methods = ["fixed_oamp", "fixed_oampnet", "rl_oamp", "rl_oampnet"]
    for s in snr_targets:
        i = int(np.where(snr == s)[0][0])
        vals = {
            "fixed_oamp": float(fixed_oamp[i]),
            "fixed_oampnet": float(fixed_oampnet[i]),
            "rl_oamp": float(rl_oamp[i]),
            "rl_oampnet": float(rl_oampnet[i]),
        }
        ref = vals["fixed_oamp"]
        for m in ablation_methods:
            ber = vals[m]
            gain = float((ref - ber) / max(ref, 1e-12) * 100.0)
            rows4.append({"snr_db": s, "method": m, "ber": ber, "rel_gain_percent": gain})

    p_csv4 = results_dir / "fig4_ablation_gain.csv"
    _write_csv(p_csv4, ["snr_db", "method", "ber", "rel_gain_percent"], rows4)
    p_mat4 = results_dir / "fig4_ablation_gain.mat"
    ber_table = np.full((len(snr_targets), len(ablation_methods)), np.nan, dtype=np.float64)
    gain_table = np.full_like(ber_table, np.nan)
    for r in rows4:
        si = snr_targets.index(int(r["snr_db"]))
        mi = ablation_methods.index(r["method"])
        ber_table[si, mi] = r["ber"]
        gain_table[si, mi] = r["rel_gain_percent"]
    _save_mat(
        p_mat4,
        {
            "snr_target": np.array(snr_targets, dtype=np.float64),
            "method_names": np.array(ablation_methods, dtype=object),
            "ber_ablation": ber_table,
            "relative_gain_percent": gain_table,
        },
    )

    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(snr_targets))
    bw = 0.18
    for j, m in enumerate(ablation_methods):
        y = [gain_table[i, j] for i in range(len(snr_targets))]
        ax.bar(x + (j - 1.5) * bw, y, width=bw, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in snr_targets])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Relative Gain vs fixed_oamp (%)")
    ax.set_title("Fig.4 Ablation Gain")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig4_ablation_gain.png")

    return {
        "fig3_csv": str(p_csv3),
        "fig3_mat": str(p_mat3),
        "fig4_csv": str(p_csv4),
        "fig4_mat": str(p_mat4),
    }


def build_fig5(
    train_history_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> Dict[str, str]:
    with train_history_path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    rows = []
    for item in hist:
        rows.append(
            {
                "episode": int(item.get("epoch", 0)),
                "train_reward": item.get("train_avg_reward"),
                "val_reward": item.get("val_avg_reward"),
                "val_ber": item.get("val_avg_ber"),
                "policy_entropy": item.get("entropy"),
            }
        )
    p_csv = results_dir / "fig5_rl_convergence.csv"
    _write_csv(p_csv, ["episode", "train_reward", "val_reward", "val_ber", "policy_entropy"], rows)

    ep = np.array([r["episode"] for r in rows], dtype=np.float64)
    train_reward = np.array([np.nan if r["train_reward"] is None else float(r["train_reward"]) for r in rows])
    val_reward = np.array([np.nan if r["val_reward"] is None else float(r["val_reward"]) for r in rows])
    val_ber = np.array([np.nan if r["val_ber"] is None else float(r["val_ber"]) for r in rows])
    entropy = np.array([np.nan if r["policy_entropy"] is None else float(r["policy_entropy"]) for r in rows])

    p_mat = results_dir / "fig5_rl_convergence.mat"
    _save_mat(
        p_mat,
        {
            "episode_idx": ep,
            "train_reward_mean": train_reward,
            "val_reward_mean": val_reward,
            "val_ber": val_ber,
            "policy_entropy": entropy,
        },
    )

    fig = plt.figure(figsize=(9.0, 5.2))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(ep, train_reward, "-o", label="train_reward", linewidth=1.6, markersize=3)
    ax1.plot(ep, val_reward, "-s", label="val_reward", linewidth=1.6, markersize=3)
    ax1.set_xlabel("Episode / Epoch")
    ax1.set_ylabel("Reward")
    ax1.set_title("Fig.5 RL Training Convergence")
    ax2 = ax1.twinx()
    ax2.semilogy(ep, val_ber, "--^", color="tab:red", label="val_ber", linewidth=1.4, markersize=3)
    ax2.set_ylabel("BER (val)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig5_rl_convergence.png")

    return {"fig5_csv": str(p_csv), "fig5_mat": str(p_mat)}


def build_fig6(
    results_dir: Path,
    figures_dir: Path,
    policy_probs: np.ndarray,
    policy_actions: np.ndarray,
    oracle_actions: np.ndarray,
) -> Dict[str, str]:
    n, m = policy_probs.shape
    rows = []
    for i in range(n):
        for a in range(m):
            rows.append(
                {
                    "sample_id": i + 1,
                    "action_idx": a + 1,
                    "policy_prob": float(policy_probs[i, a]),
                    "is_selected": int(policy_actions[i] == a),
                    "is_oracle": int(oracle_actions[i] == a),
                }
            )
    p_csv = results_dir / "fig6_action_distribution.csv"
    _write_csv(p_csv, ["sample_id", "action_idx", "policy_prob", "is_selected", "is_oracle"], rows)

    rl_freq = np.bincount(policy_actions, minlength=m).astype(np.float64) / max(1, n)
    oracle_freq = np.bincount(oracle_actions, minlength=m).astype(np.float64) / max(1, n)
    rows_freq = [
        {"action_idx": a + 1, "rl_freq": float(rl_freq[a]), "oracle_freq": float(oracle_freq[a])}
        for a in range(m)
    ]
    p_csv_freq = results_dir / "fig6_action_distribution_freq.csv"
    _write_csv(p_csv_freq, ["action_idx", "rl_freq", "oracle_freq"], rows_freq)

    p_mat = results_dir / "fig6_action_distribution.mat"
    _save_mat(
        p_mat,
        {
            "policy_probs": policy_probs,
            "selected_action": policy_actions + 1,
            "oracle_action": oracle_actions + 1,
            "action_freq_rl": rl_freq,
            "action_freq_oracle": oracle_freq,
        },
    )

    fig = plt.figure(figsize=(9.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(m)
    bw = 0.4
    ax.bar(x - bw / 2, rl_freq, width=bw, label="rl_freq")
    ax.bar(x + bw / 2, oracle_freq, width=bw, label="oracle_freq")
    ax.set_xlabel("Action Index (1-based)")
    ax.set_ylabel("Frequency")
    ax.set_title("Fig.6 Action Distribution")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig6_action_distribution.png")

    return {"fig6_csv": str(p_csv), "fig6_csv_freq": str(p_csv_freq), "fig6_mat": str(p_mat)}


def build_fig7(
    results_dir: Path,
    figures_dir: Path,
    data,
    checkpoint_path: Path,
    device: torch.device,
    repeats: int,
) -> Dict[str, str]:
    pack = load_policy(str(checkpoint_path), device)
    model = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6

    x = (data.states.astype(np.float32) - state_mean) / state_std
    xt = torch.from_numpy(x).to(device)
    b = data.ber_actions.astype(np.float64)
    n = b.shape[0]
    m = b.shape[1]
    base_action = int(data.base_action)

    def bench(fn):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0 / max(1, n))  # ms/sample
        return float(np.mean(ts)), float(np.std(ts, ddof=1) if len(ts) > 1 else 0.0)

    def run_fixed():
        _ = np.mean(b[:, base_action])

    def run_rl():
        with torch.no_grad():
            a = model.greedy_action(xt).cpu().numpy().astype(np.int64)
        _ = np.mean(b[np.arange(n), a])

    def run_oracle():
        a = np.argmin(b, axis=1)
        _ = np.mean(b[np.arange(n), a])

    fixed_mean, fixed_std = bench(run_fixed)
    rl_mean, rl_std = bench(run_rl)
    oracle_mean, oracle_std = bench(run_oracle)

    num_params_policy = int(sum(int(p.numel()) for p in model.parameters()))
    methods = [
        ("fixed", fixed_mean, fixed_std, 0, "Decision with fixed C1 baseline."),
        ("rl", rl_mean, rl_std, num_params_policy, "Policy forward + lookup."),
        ("oracle", oracle_mean, oracle_std, 0, "Exhaustive action argmin on table."),
    ]
    rows = [
        {
            "method": method_name,
            "avg_runtime_ms": float(mu),
            "std_runtime_ms": float(sd),
            "num_params": int(param_cnt),
            "notes": note,
        }
        for (method_name, mu, sd, param_cnt, note) in methods
    ]
    p_csv = results_dir / "fig7_runtime_complexity.csv"
    _write_csv(p_csv, ["method", "avg_runtime_ms", "std_runtime_ms", "num_params", "notes"], rows)
    p_mat = results_dir / "fig7_runtime_complexity.mat"
    _save_mat(
        p_mat,
        {
            "method": np.array([r["method"] for r in rows], dtype=object),
            "avg_runtime_ms": np.array([r["avg_runtime_ms"] for r in rows], dtype=np.float64),
            "std_runtime_ms": np.array([r["std_runtime_ms"] for r in rows], dtype=np.float64),
            "num_params": np.array([r["num_params"] for r in rows], dtype=np.float64),
            "num_actions": float(m),
            "num_params_policy": float(num_params_policy),
        },
    )

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    xloc = np.arange(len(rows))
    y = [r["avg_runtime_ms"] for r in rows]
    yerr = [r["std_runtime_ms"] for r in rows]
    ax.bar(xloc, y, yerr=yerr, capsize=4)
    ax.set_xticks(xloc)
    ax.set_xticklabels([r["method"] for r in rows])
    ax.set_ylabel("Runtime (ms/sample)")
    ax.set_title("Fig.7 Runtime / Complexity")
    _plot_save(fig, figures_dir / "fig7_runtime_complexity.png")

    return {"fig7_csv": str(p_csv), "fig7_mat": str(p_mat)}


def _sequence_arrays(data, policy_actions: np.ndarray):
    if data.sequence_id is None or data.time_index is None:
        raise ValueError("Dataset has no sequence_id/time_index; Fig.8~Fig.11 require time-varying export.")
    seq = data.sequence_id.astype(np.int64)
    t = data.time_index.astype(np.int64)
    o = data.oracle_actions.astype(np.int64)
    b = data.ber_actions.astype(np.float64)
    return seq, t, o, b, policy_actions.astype(np.int64)


def build_fig8(
    results_dir: Path,
    figures_dir: Path,
    data,
    policy_actions: np.ndarray,
) -> Dict[str, str]:
    _seq, t, o, b, p = _sequence_arrays(data, policy_actions)
    base = int(data.base_action)
    tu = np.sort(np.unique(t))
    ber_fixed = []
    ber_rl = []
    ber_oracle = []
    for tt in tu:
        idx = np.where(t == tt)[0]
        bb = b[idx]
        ber_fixed.append(float(np.mean(bb[:, base])))
        ber_rl.append(float(np.mean(bb[np.arange(idx.size), p[idx]])))
        ber_oracle.append(float(np.mean(bb[np.arange(idx.size), o[idx]])))
    rows = [
        {
            "frame_idx": int(tu[i]),
            "ber_fixed_t": ber_fixed[i],
            "ber_rl_t": ber_rl[i],
            "ber_oracle_t": ber_oracle[i],
        }
        for i in range(len(tu))
    ]
    p_csv = results_dir / "fig8_ber_over_time.csv"
    _write_csv(p_csv, ["frame_idx", "ber_fixed_t", "ber_rl_t", "ber_oracle_t"], rows)
    p_mat = results_dir / "fig8_ber_over_time.mat"
    _save_mat(
        p_mat,
        {
            "frame_idx": tu,
            "ber_fixed_t": np.array(ber_fixed),
            "ber_rl_t": np.array(ber_rl),
            "ber_oracle_t": np.array(ber_oracle),
        },
    )

    fig = plt.figure(figsize=(8.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(tu, ber_fixed, "-o", label="fixed", linewidth=1.7)
    ax.semilogy(tu, ber_rl, "-s", label="rl", linewidth=1.7)
    ax.semilogy(tu, ber_oracle, "-^", label="oracle", linewidth=1.7)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("BER")
    ax.set_title("Fig.8 BER over Time")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig8_ber_over_time.png")

    return {"fig8_csv": str(p_csv), "fig8_mat": str(p_mat)}


def _pick_sequence(seq: np.ndarray, example_sequence: int) -> int:
    seq_u = np.sort(np.unique(seq))
    if example_sequence >= 0 and example_sequence in set(seq_u.tolist()):
        return int(example_sequence)
    return int(seq_u[len(seq_u) // 2])


def build_fig9_fig10(
    results_dir: Path,
    figures_dir: Path,
    data,
    policy_actions: np.ndarray,
    example_sequence: int,
    channel_metric_idx: int,
) -> Dict[str, str]:
    seq, t, o, _b, p = _sequence_arrays(data, policy_actions)
    sid = _pick_sequence(seq, example_sequence)
    idx = np.where(seq == sid)[0]
    order = np.argsort(t[idx])
    idx = idx[order]
    tt = t[idx]
    ao = o[idx]
    ap = p[idx]
    c1_grid = data.c1_grid.reshape(-1) if data.c1_grid is not None else np.arange(int(np.max(np.r_[ao, ap])) + 1)
    c1_oracle = c1_grid[ao]
    c1_rl = c1_grid[ap]

    # Fig.9
    rows9 = [
        {
            "frame_idx": int(tt[i]),
            "oracle_action_t": int(ao[i] + 1),
            "rl_action_t": int(ap[i] + 1),
            "oracle_c1_t": float(c1_oracle[i]),
            "rl_c1_t": float(c1_rl[i]),
        }
        for i in range(tt.size)
    ]
    p_csv9 = results_dir / "fig9_action_over_time.csv"
    _write_csv(
        p_csv9,
        ["frame_idx", "oracle_action_t", "rl_action_t", "oracle_c1_t", "rl_c1_t"],
        rows9,
    )
    p_mat9 = results_dir / "fig9_action_over_time.mat"
    _save_mat(
        p_mat9,
        {
            "frame_idx": tt,
            "oracle_action_t": ao + 1,
            "rl_action_t": ap + 1,
            "oracle_c1_t": c1_oracle,
            "rl_c1_t": c1_rl,
            "sequence_id": float(sid),
        },
    )
    fig = plt.figure(figsize=(8.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.step(tt, ao + 1, where="mid", label="oracle_action", linewidth=1.8)
    ax.step(tt, ap + 1, where="mid", label="rl_action", linewidth=1.8)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Action Index (1-based)")
    ax.set_title(f"Fig.9 Actions over Time (sequence={sid})")
    ax.legend(loc="best", fontsize=8)
    _plot_save(fig, figures_dir / "fig9_action_over_time.png")

    # Fig.10
    feat_idx = int(np.clip(channel_metric_idx, 0, data.states.shape[1] - 1))
    metric = data.states[idx, feat_idx].astype(np.float64)
    rows10 = [
        {
            "frame_idx": int(tt[i]),
            "channel_metric_t": float(metric[i]),
            "oracle_action_t": int(ao[i] + 1),
            "rl_action_t": int(ap[i] + 1),
        }
        for i in range(tt.size)
    ]
    p_csv10 = results_dir / "fig10_channel_action_correlation.csv"
    _write_csv(p_csv10, ["frame_idx", "channel_metric_t", "oracle_action_t", "rl_action_t"], rows10)
    p_mat10 = results_dir / "fig10_channel_action_correlation.mat"
    _save_mat(
        p_mat10,
        {
            "frame_idx": tt,
            "channel_metric_t": metric,
            "oracle_action_t": ao + 1,
            "rl_action_t": ap + 1,
            "sequence_id": float(sid),
            "metric_feature_index": float(feat_idx),
        },
    )
    fig = plt.figure(figsize=(8.8, 5.0))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(tt, metric, "-o", color="tab:blue", label="channel_metric", linewidth=1.5, markersize=4)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Channel Metric", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.step(tt, ao + 1, where="mid", color="tab:green", label="oracle_action", linewidth=1.5)
    ax2.step(tt, ap + 1, where="mid", color="tab:red", label="rl_action", linewidth=1.5)
    ax2.set_ylabel("Action Index (1-based)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.set_title(f"Fig.10 Channel Metric vs Action Switching (sequence={sid})")
    _plot_save(fig, figures_dir / "fig10_channel_action_correlation.png")

    return {
        "fig9_csv": str(p_csv9),
        "fig9_mat": str(p_mat9),
        "fig10_csv": str(p_csv10),
        "fig10_mat": str(p_mat10),
        "fig9_sequence_id": str(sid),
    }


def build_fig11(
    results_dir: Path,
    figures_dir: Path,
    data,
    policy_actions: np.ndarray,
) -> Dict[str, str]:
    seq, t, o, b, p = _sequence_arrays(data, policy_actions)
    base = int(data.base_action)

    scenarios: List[Tuple[str, np.ndarray]] = []
    if data.snr_db is not None:
        for s in np.sort(np.unique(data.snr_db.astype(np.int64))):
            scenarios.append((f"SNR_{int(s)}dB", np.where(data.snr_db.astype(np.int64) == s)[0]))
    scenarios.append(("ALL", np.arange(seq.size, dtype=np.int64)))

    rows = []
    for name, idx_s in scenarios:
        seq_u = np.unique(seq[idx_s])
        sw_rl = []
        sw_or = []
        gap_fix = []
        gap_rl = []
        for sid in seq_u:
            idx = idx_s[seq[idx_s] == sid]
            if idx.size <= 1:
                continue
            ord_ = np.argsort(t[idx])
            idx = idx[ord_]
            ap = p[idx]
            ao = o[idx]
            bb = b[idx]
            sw_rl.append(float(np.mean(ap[1:] != ap[:-1])))
            sw_or.append(float(np.mean(ao[1:] != ao[:-1])))
            ber_fix = float(np.mean(bb[:, base]))
            ber_rl = float(np.mean(bb[np.arange(idx.size), ap]))
            ber_or = float(np.mean(bb[np.arange(idx.size), ao]))
            gap_fix.append(float((ber_fix - ber_or) / max(ber_fix, 1e-12)))
            gap_rl.append(float((ber_rl - ber_or) / max(ber_rl, 1e-12)))
        rows.append(
            {
                "scenario": name,
                "switch_rate_rl": float(np.mean(sw_rl) if sw_rl else np.nan),
                "switch_rate_oracle": float(np.mean(sw_or) if sw_or else np.nan),
                "oracle_gap_fixed": float(np.mean(gap_fix) if gap_fix else np.nan),
                "oracle_gap_rl": float(np.mean(gap_rl) if gap_rl else np.nan),
            }
        )

    p_csv = results_dir / "fig11_switch_oracle_gap.csv"
    _write_csv(
        p_csv,
        ["scenario", "switch_rate_rl", "switch_rate_oracle", "oracle_gap_fixed", "oracle_gap_rl"],
        rows,
    )
    p_mat = results_dir / "fig11_switch_oracle_gap.mat"
    _save_mat(
        p_mat,
        {
            "scenario": np.array([r["scenario"] for r in rows], dtype=object),
            "switch_rate_rl": np.array([r["switch_rate_rl"] for r in rows], dtype=np.float64),
            "switch_rate_oracle": np.array([r["switch_rate_oracle"] for r in rows], dtype=np.float64),
            "oracle_gap_fixed": np.array([r["oracle_gap_fixed"] for r in rows], dtype=np.float64),
            "oracle_gap_rl": np.array([r["oracle_gap_rl"] for r in rows], dtype=np.float64),
        },
    )

    labels = [r["scenario"] for r in rows]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(9.0, 5.0))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x - 0.18, [r["switch_rate_rl"] for r in rows], width=0.36, label="switch_rl")
    ax1.bar(x + 0.18, [r["switch_rate_oracle"] for r in rows], width=0.36, label="switch_oracle")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_title("Switch Rate")
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x - 0.18, [100.0 * r["oracle_gap_fixed"] for r in rows], width=0.36, label="gap_fixed")
    ax2.bar(x + 0.18, [100.0 * r["oracle_gap_rl"] for r in rows], width=0.36, label="gap_rl")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_title("Oracle Gap (%)")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Fig.11 Switch Rate and Oracle Gap")
    _plot_save(fig, figures_dir / "fig11_switch_oracle_gap.png")

    return {"fig11_csv": str(p_csv), "fig11_mat": str(p_mat)}


def main() -> None:
    args = parse_args()
    _set_plot_style()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    _ensure_dir(results_dir)
    _ensure_dir(figures_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = load_offline_bandit_data(
        args.data,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        reward_key=args.reward_key,
    )
    if data.ber_actions is None:
        raise ValueError("Dataset must include ber_actions for figure export.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    policy_actions, policy_probs = _compute_policy_outputs(data, checkpoint_path, device)

    manifest: Dict[str, str] = {}
    refine = _load_refine_result(Path(args.refine_mat))
    manifest.update(build_fig1_fig2(results_dir, figures_dir, refine, data))

    detector_path = Path(args.detector_csv)
    detector = _read_detector_csv(detector_path) if detector_path.exists() else None
    online_measured = _read_online_policy_detector_csv(Path(args.online_policy_detector_csv))
    metrics_snr = _metrics_by_snr(
        data.ber_actions.astype(np.float64),
        int(data.base_action),
        data.oracle_actions.astype(np.int64),
        policy_actions.astype(np.int64),
        data.snr_db,
    )
    manifest.update(build_fig3_fig4(results_dir, figures_dir, detector, metrics_snr, online_measured))

    train_history = Path(args.train_history)
    if train_history.exists():
        manifest.update(build_fig5(train_history, results_dir, figures_dir))
    else:
        manifest["fig5_warning"] = f"train_history not found: {train_history}"

    manifest.update(
        build_fig6(
            results_dir=results_dir,
            figures_dir=figures_dir,
            policy_probs=policy_probs,
            policy_actions=policy_actions,
            oracle_actions=data.oracle_actions.astype(np.int64),
        )
    )

    manifest.update(
        build_fig7(
            results_dir=results_dir,
            figures_dir=figures_dir,
            data=data,
            checkpoint_path=checkpoint_path,
            device=device,
            repeats=max(3, int(args.runtime_repeats)),
        )
    )

    manifest.update(build_fig8(results_dir, figures_dir, data, policy_actions))
    manifest.update(
        build_fig9_fig10(
            results_dir=results_dir,
            figures_dir=figures_dir,
            data=data,
            policy_actions=policy_actions,
            example_sequence=int(args.example_sequence),
            channel_metric_idx=int(args.channel_metric_idx),
        )
    )
    manifest.update(build_fig11(results_dir, figures_dir, data, policy_actions))

    manifest_path = results_dir / "figure_export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("==== Standard Figure Export Complete ====")
    print(f"Saved manifest: {manifest_path}")
    for k, v in manifest.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
