"""
Build a unified main-result package for the C1-policy integration stage.

Combines:
1) Policy-side C1 metrics (fixed/oracle/policy) from offline dataset + checkpoint
2) Detector benchmark BER curve (LMMSE/OAMP/OAMPNet) from MATLAB CSV (optional)

Outputs:
- main_summary.json
- main_table.csv
- main_figure.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import load_offline_bandit_data
from rl_c1.eval_policy import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified C1-policy main result package.")
    p.add_argument("--data", type=str, required=True, help="Offline dataset path (.mat/.npz)")
    p.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    p.add_argument("--reward_key", type=str, default="reward_ber")
    p.add_argument("--snr_min", type=float, default=None)
    p.add_argument("--snr_max", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--detector_csv",
        type=str,
        default="results/ber_results_matlab_tsv1.csv",
        help="Optional detector BER benchmark CSV",
    )
    p.add_argument("--output_dir", type=str, default="results/rl_c1_stage2_full/main_package")
    p.add_argument("--title_suffix", type=str, default="C1 Policy Integration")
    return p.parse_args()


def _load_detector_csv(path: Path) -> Optional[Dict[str, np.ndarray]]:
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

    keys = set(rows[0].keys())

    # Legacy detector benchmark format:
    # SNR,LMMSE,OAMP,OAMPNet
    if "SNR" in keys:
        out = {"snr": col("SNR"), "source": "legacy_detector_benchmark"}
        for k in ["LMMSE", "OAMP", "OAMPNet"]:
            if k in rows[0]:
                out[k.lower()] = col(k)
        return out

    # Online measured policy+detector format:
    # snr_db,fixed_oamp,fixed_oampnet,rl_oamp,rl_oampnet,oracle_oampnet,...
    if "snr_db" in keys:
        out = {"snr": col("snr_db"), "source": "online_policy_detector_measured"}
        if "fixed_oamp" in rows[0]:
            out["oamp"] = col("fixed_oamp")
        if "fixed_oampnet" in rows[0]:
            out["oampnet"] = col("fixed_oampnet")
        if "rl_oamp" in rows[0]:
            out["rl_oamp"] = col("rl_oamp")
        if "rl_oampnet" in rows[0]:
            out["rl_oampnet"] = col("rl_oampnet")
        if "oracle_oampnet" in rows[0]:
            out["oracle_oampnet"] = col("oracle_oampnet")
        return out

    raise ValueError(f"Unsupported detector csv schema: {path}")


def _switch_rate_for_subset(seq: np.ndarray, t: np.ndarray, a: np.ndarray) -> float:
    uniq = np.unique(seq)
    rates = []
    for sid in uniq:
        idx = np.where(seq == sid)[0]
        if idx.size <= 1:
            continue
        order = np.argsort(t[idx])
        aa = a[idx][order]
        rates.append(float(np.mean(aa[1:] != aa[:-1])))
    if not rates:
        return 0.0
    return float(np.mean(rates))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_offline_bandit_data(
        args.data,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        reward_key=args.reward_key,
    )
    if data.ber_actions is None:
        raise ValueError("Dataset must include ber_actions for unified reporting.")

    pack = load_policy(args.checkpoint, device)
    model = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6

    x = (data.states.astype(np.float32) - state_mean) / state_std
    with torch.no_grad():
        actions = model.greedy_action(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.int64)

    ber = data.ber_actions.astype(np.float64)
    oracle = data.oracle_actions.astype(np.int64)
    base_action = int(data.base_action)

    # fixed-best (global)
    avg_ber_per_action = ber.mean(axis=0)
    best_fixed_action = int(np.argmin(avg_ber_per_action))

    snr_vals = np.unique(data.snr_db.astype(np.int64)) if data.snr_db is not None else np.array([-1], dtype=np.int64)
    snr_vals = np.sort(snr_vals)

    table_rows: List[Dict] = []
    policy_line = []
    base_line = []
    oracle_line = []
    fixed_best_line = []

    for snr in snr_vals:
        if snr < 0:
            idx = np.arange(data.num_samples)
        else:
            idx = np.where(data.snr_db.astype(np.int64) == snr)[0]
        if idx.size == 0:
            continue

        b = ber[idx]
        a_pol = actions[idx]
        a_orc = oracle[idx]

        ber_base = float(np.mean(b[:, base_action]))
        ber_policy = float(np.mean(b[np.arange(idx.size), a_pol]))
        ber_oracle = float(np.mean(b[np.arange(idx.size), a_orc]))
        ber_fixed_best = float(np.mean(b[:, best_fixed_action]))

        gain_pol = float((ber_base - ber_policy) / max(ber_base, 1e-12))
        gain_orc = float((ber_base - ber_oracle) / max(ber_base, 1e-12))
        gain_best = float((ber_base - ber_fixed_best) / max(ber_base, 1e-12))

        match = float(np.mean(a_pol == a_orc))
        if data.sequence_id is not None and data.time_index is not None:
            sw_pol = _switch_rate_for_subset(
                data.sequence_id[idx].astype(np.int64),
                data.time_index[idx].astype(np.int64),
                a_pol,
            )
            sw_orc = _switch_rate_for_subset(
                data.sequence_id[idx].astype(np.int64),
                data.time_index[idx].astype(np.int64),
                a_orc,
            )
        else:
            sw_pol = 0.0
            sw_orc = 0.0

        table_rows.append(
            {
                "snr_db": int(snr),
                "avg_ber_base": ber_base,
                "avg_ber_policy": ber_policy,
                "avg_ber_oracle": ber_oracle,
                "avg_ber_fixed_best": ber_fixed_best,
                "gain_policy_vs_base": gain_pol,
                "gain_oracle_vs_base": gain_orc,
                "gain_fixed_best_vs_base": gain_best,
                "match_rate": match,
                "switch_rate_policy": sw_pol,
                "switch_rate_oracle": sw_orc,
            }
        )

        policy_line.append(ber_policy)
        base_line.append(ber_base)
        oracle_line.append(ber_oracle)
        fixed_best_line.append(ber_fixed_best)

    # overall row
    idx_all = np.arange(data.num_samples)
    b_all = ber[idx_all]
    ber_base_all = float(np.mean(b_all[:, base_action]))
    ber_pol_all = float(np.mean(b_all[np.arange(idx_all.size), actions]))
    ber_orc_all = float(np.mean(b_all[np.arange(idx_all.size), oracle]))
    ber_best_all = float(np.mean(b_all[:, best_fixed_action]))
    row_all = {
        "snr_db": "ALL",
        "avg_ber_base": ber_base_all,
        "avg_ber_policy": ber_pol_all,
        "avg_ber_oracle": ber_orc_all,
        "avg_ber_fixed_best": ber_best_all,
        "gain_policy_vs_base": float((ber_base_all - ber_pol_all) / max(ber_base_all, 1e-12)),
        "gain_oracle_vs_base": float((ber_base_all - ber_orc_all) / max(ber_base_all, 1e-12)),
        "gain_fixed_best_vs_base": float((ber_base_all - ber_best_all) / max(ber_base_all, 1e-12)),
        "match_rate": float(np.mean(actions == oracle)),
        "switch_rate_policy": _switch_rate_for_subset(
            data.sequence_id.astype(np.int64),
            data.time_index.astype(np.int64),
            actions,
        )
        if (data.sequence_id is not None and data.time_index is not None)
        else 0.0,
        "switch_rate_oracle": _switch_rate_for_subset(
            data.sequence_id.astype(np.int64),
            data.time_index.astype(np.int64),
            oracle,
        )
        if (data.sequence_id is not None and data.time_index is not None)
        else 0.0,
    }

    detector = _load_detector_csv(Path(args.detector_csv))

    # Save CSV table
    table_csv = out_dir / "main_table.csv"
    fields = [
        "snr_db",
        "avg_ber_base",
        "avg_ber_policy",
        "avg_ber_oracle",
        "avg_ber_fixed_best",
        "gain_policy_vs_base",
        "gain_oracle_vs_base",
        "gain_fixed_best_vs_base",
        "match_rate",
        "switch_rate_policy",
        "switch_rate_oracle",
    ]
    with table_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)
        w.writerow(row_all)

    # Save summary JSON
    summary = {
        "base_action": base_action,
        "best_fixed_action": best_fixed_action,
        "per_snr": table_rows,
        "overall": row_all,
        "detector_csv_used": str(Path(args.detector_csv).resolve()) if detector is not None else None,
    }
    summary_json = out_dir / "main_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot main figure
    fig = plt.figure(figsize=(12.5, 5.0))

    ax1 = fig.add_subplot(1, 2, 1)
    if detector is not None:
        if "lmmse" in detector:
            ax1.semilogy(detector["snr"], detector["lmmse"], "-o", label="LMMSE (detector bench)", linewidth=1.8)
        if "oamp" in detector:
            lbl_oamp = "OAMP (detector bench)"
            if detector.get("source", "") == "online_policy_detector_measured":
                lbl_oamp = "OAMP fixed-C1 (online)"
            ax1.semilogy(detector["snr"], detector["oamp"], "-s", label=lbl_oamp, linewidth=1.8)
        if "oampnet" in detector:
            y = detector["oampnet"].copy()
            y[y <= 0] = np.nan
            lbl_oampnet = "OAMPNet (detector bench)"
            if detector.get("source", "") == "online_policy_detector_measured":
                lbl_oampnet = "OAMPNet fixed-C1 (online)"
            ax1.semilogy(detector["snr"], y, "-^", label=lbl_oampnet, linewidth=1.8)

    ax1.semilogy(snr_vals, base_line, "k--o", label="C1 fixed-base", linewidth=2.0)
    ax1.semilogy(snr_vals, fixed_best_line, "c--d", label="C1 fixed-best", linewidth=2.0)
    ax1.semilogy(snr_vals, policy_line, "r-o", label="C1 policy", linewidth=2.2)
    ax1.semilogy(snr_vals, oracle_line, "g-^", label="C1 oracle", linewidth=2.2)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title("BER-SNR Unified View")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    x_pos = np.arange(len(snr_vals))
    wbar = 0.25
    gain_best = [r["gain_fixed_best_vs_base"] * 100 for r in table_rows]
    gain_pol = [r["gain_policy_vs_base"] * 100 for r in table_rows]
    gain_orc = [r["gain_oracle_vs_base"] * 100 for r in table_rows]
    ax2.bar(x_pos - wbar, gain_best, width=wbar, label="fixed-best gain")
    ax2.bar(x_pos, gain_pol, width=wbar, label="policy gain")
    ax2.bar(x_pos + wbar, gain_orc, width=wbar, label="oracle gain")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(int(v)) for v in snr_vals])
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Gain vs Fixed-Base (%)")
    ax2.set_title("C1 Adaptation Gain")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(args.title_suffix, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig_png = out_dir / "main_figure.png"
    fig.savefig(fig_png, dpi=160)
    plt.close(fig)

    print("==== Main Package Built ====")
    print(f"Saved table:   {table_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved figure:  {fig_png}")
    print(f"Overall BER gain(policy vs base): {row_all['gain_policy_vs_base']*100:.2f}%")
    print(f"Overall BER gain(fixed-best vs base): {row_all['gain_fixed_best_vs_base']*100:.2f}%")


if __name__ == "__main__":
    main()
