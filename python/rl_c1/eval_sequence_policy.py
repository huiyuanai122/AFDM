"""
Sequence-level policy evaluation with paper-ready plots.

Generates:
1) BER-vs-time curve: fixed base vs oracle vs policy
2) Action trajectory on one sample sequence: oracle vs policy vs base
3) Switch-rate histogram across sequences: oracle vs policy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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
    p = argparse.ArgumentParser(description="Sequence-level C1 policy evaluation and plotting.")
    p.add_argument("--data", type=str, required=True, help="Dataset path (.mat/.npz)")
    p.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    p.add_argument("--reward_key", type=str, default="reward")
    p.add_argument("--snr_min", type=float, default=None)
    p.add_argument("--snr_max", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_dir", type=str, default="results/rl_c1_sequence_eval")
    p.add_argument(
        "--example_sequence",
        type=int,
        default=-1,
        help="Sequence id to plot. -1 means choose one automatically.",
    )
    p.add_argument("--prefix", type=str, default="policy_seq_eval")
    return p.parse_args()


def _compute_switch_rate(actions: np.ndarray) -> float:
    if actions.size <= 1:
        return 0.0
    return float(np.mean(actions[1:] != actions[:-1]))


def _group_sequence_stats(
    seq_id: np.ndarray,
    time_idx: np.ndarray,
    base_action: int,
    policy_actions: np.ndarray,
    oracle_actions: np.ndarray,
    ber_actions: np.ndarray,
) -> Dict:
    seq_unique = np.unique(seq_id)
    time_unique = np.unique(time_idx)
    time_unique = np.sort(time_unique)

    ber_base_t = []
    ber_policy_t = []
    ber_oracle_t = []

    seq_switch_policy = []
    seq_switch_oracle = []
    seq_ber_policy = []
    seq_ber_oracle = []
    seq_ber_base = []

    for sid in seq_unique:
        idx = np.where(seq_id == sid)[0]
        order = np.argsort(time_idx[idx])
        idx = idx[order]

        a_pol = policy_actions[idx]
        a_orc = oracle_actions[idx]
        b = ber_actions[idx]

        ber_pol = b[np.arange(idx.size), a_pol]
        ber_orc = b[np.arange(idx.size), a_orc]
        ber_bas = b[:, base_action]

        seq_switch_policy.append(_compute_switch_rate(a_pol))
        seq_switch_oracle.append(_compute_switch_rate(a_orc))
        seq_ber_policy.append(float(np.mean(ber_pol)))
        seq_ber_oracle.append(float(np.mean(ber_orc)))
        seq_ber_base.append(float(np.mean(ber_bas)))

    for tt in time_unique:
        m = time_idx == tt
        b = ber_actions[m]
        p = policy_actions[m]
        o = oracle_actions[m]
        ber_base_t.append(float(np.mean(b[:, base_action])))
        ber_policy_t.append(float(np.mean(b[np.arange(b.shape[0]), p])))
        ber_oracle_t.append(float(np.mean(b[np.arange(b.shape[0]), o])))

    return {
        "time_unique": time_unique.astype(int),
        "ber_base_t": np.array(ber_base_t),
        "ber_policy_t": np.array(ber_policy_t),
        "ber_oracle_t": np.array(ber_oracle_t),
        "seq_switch_policy": np.array(seq_switch_policy),
        "seq_switch_oracle": np.array(seq_switch_oracle),
        "seq_ber_policy": np.array(seq_ber_policy),
        "seq_ber_oracle": np.array(seq_ber_oracle),
        "seq_ber_base": np.array(seq_ber_base),
    }


def _plot_ber_vs_time(stats: Dict, out_png: Path) -> None:
    t = stats["time_unique"]
    plt.figure(figsize=(8.2, 4.8))
    plt.semilogy(t, stats["ber_base_t"], "-o", label="fixed base", linewidth=1.8, markersize=4)
    plt.semilogy(t, stats["ber_policy_t"], "-s", label="policy", linewidth=1.8, markersize=4)
    plt.semilogy(t, stats["ber_oracle_t"], "-^", label="oracle", linewidth=1.8, markersize=4)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Time Index")
    plt.ylabel("BER")
    plt.title("BER vs Time")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_action_trajectory(
    seq_id: np.ndarray,
    time_idx: np.ndarray,
    policy_actions: np.ndarray,
    oracle_actions: np.ndarray,
    base_action: int,
    sequence_to_plot: int,
    out_png: Path,
) -> None:
    idx = np.where(seq_id == sequence_to_plot)[0]
    if idx.size == 0:
        return
    order = np.argsort(time_idx[idx])
    idx = idx[order]

    t = time_idx[idx]
    ap = policy_actions[idx]
    ao = oracle_actions[idx]

    plt.figure(figsize=(8.2, 4.8))
    plt.step(t, ao, where="mid", label="oracle action", linewidth=1.8)
    plt.step(t, ap, where="mid", label="policy action", linewidth=1.8)
    plt.axhline(base_action, linestyle="--", color="k", linewidth=1.2, label="base action")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time Index")
    plt.ylabel("Action Index (0-based)")
    plt.title(f"Action Trajectory on Sequence {sequence_to_plot}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_switch_hist(stats: Dict, out_png: Path) -> None:
    plt.figure(figsize=(8.2, 4.8))
    bins = np.linspace(0.0, 1.0, 21)
    plt.hist(stats["seq_switch_oracle"], bins=bins, alpha=0.6, label="oracle", density=True)
    plt.hist(stats["seq_switch_policy"], bins=bins, alpha=0.6, label="policy", density=True)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Switch Rate")
    plt.ylabel("Density")
    plt.title("Switch Rate Distribution Across Sequences")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


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
        raise ValueError("Dataset has no ber_actions field. Please re-export with the new MATLAB script.")
    if data.sequence_id is None or data.time_index is None:
        raise ValueError("Dataset has no sequence_id/time_index. Please re-export with time-vary mode.")

    pack = load_policy(args.checkpoint, device)
    model = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6

    states = (data.states.astype(np.float32) - state_mean) / state_std
    st = torch.from_numpy(states).to(device)
    with torch.no_grad():
        policy_actions = model.greedy_action(st).cpu().numpy().astype(np.int64)

    seq_id = data.sequence_id.astype(np.int64)
    time_idx = data.time_index.astype(np.int64)
    oracle_actions = data.oracle_actions.astype(np.int64)
    base_action = int(data.base_action)
    ber_actions = data.ber_actions.astype(np.float64)

    stats = _group_sequence_stats(
        seq_id=seq_id,
        time_idx=time_idx,
        base_action=base_action,
        policy_actions=policy_actions,
        oracle_actions=oracle_actions,
        ber_actions=ber_actions,
    )

    # Global BER metrics
    s = np.arange(ber_actions.shape[0], dtype=np.int64)
    ber_policy = float(np.mean(ber_actions[s, policy_actions]))
    ber_oracle = float(np.mean(ber_actions[s, oracle_actions]))
    ber_base = float(np.mean(ber_actions[:, base_action]))

    result = {
        "num_samples": int(data.num_samples),
        "num_sequences": int(np.unique(seq_id).size),
        "num_actions": int(data.num_actions),
        "base_action": base_action,
        "avg_ber_base": ber_base,
        "avg_ber_policy": ber_policy,
        "avg_ber_oracle": ber_oracle,
        "ber_gain_vs_base": float((ber_base - ber_policy) / max(ber_base, 1e-12)),
        "ber_gap_to_oracle": float((ber_policy - ber_oracle) / max(ber_policy, 1e-12)),
        "match_rate": float(np.mean(policy_actions == oracle_actions)),
        "switch_rate_policy": float(np.mean(stats["seq_switch_policy"])),
        "switch_rate_oracle": float(np.mean(stats["seq_switch_oracle"])),
        "time_index": stats["time_unique"].tolist(),
        "ber_base_t": stats["ber_base_t"].tolist(),
        "ber_policy_t": stats["ber_policy_t"].tolist(),
        "ber_oracle_t": stats["ber_oracle_t"].tolist(),
    }

    # Choose example sequence id.
    seq_unique = np.unique(seq_id)
    if args.example_sequence >= 0:
        seq_plot = int(args.example_sequence)
    else:
        seq_plot = int(seq_unique[len(seq_unique) // 2])

    p1 = out_dir / f"{args.prefix}_ber_vs_time.png"
    p2 = out_dir / f"{args.prefix}_action_traj_seq{seq_plot}.png"
    p3 = out_dir / f"{args.prefix}_switch_hist.png"
    pj = out_dir / f"{args.prefix}_summary.json"

    _plot_ber_vs_time(stats, p1)
    _plot_action_trajectory(
        seq_id=seq_id,
        time_idx=time_idx,
        policy_actions=policy_actions,
        oracle_actions=oracle_actions,
        base_action=base_action,
        sequence_to_plot=seq_plot,
        out_png=p2,
    )
    _plot_switch_hist(stats, p3)

    with pj.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("==== Sequence Evaluation ====")
    print(f"samples={result['num_samples']}, sequences={result['num_sequences']}")
    print(f"avg_ber(base)   = {result['avg_ber_base']:.6e}")
    print(f"avg_ber(policy) = {result['avg_ber_policy']:.6e}")
    print(f"avg_ber(oracle) = {result['avg_ber_oracle']:.6e}")
    print(f"ber_gain_vs_base= {100*result['ber_gain_vs_base']:.2f}%")
    print(f"ber_gap_oracle  = {100*result['ber_gap_to_oracle']:.2f}%")
    print(f"match_rate      = {100*result['match_rate']:.2f}%")
    print(f"switch(policy)  = {result['switch_rate_policy']:.4f}")
    print(f"switch(oracle)  = {result['switch_rate_oracle']:.4f}")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {pj}")


if __name__ == "__main__":
    main()
