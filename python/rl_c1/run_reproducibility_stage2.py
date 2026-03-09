"""
Run multi-seed reproducibility experiments for stage-2 RL C1 policy.

For each seed:
1) train policy (imitation + REINFORCE)
2) evaluate policy and save eval_metrics.json
3) compute overall and per-SNR BER gain summary

Outputs:
- reproducibility_seed_table.csv
- reproducibility_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import load_offline_bandit_data
from rl_c1.eval_policy import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-seed reproducibility for RL-C1 stage-2.")
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--data", type=str, default="data/oracle_policy_dataset.mat")
    p.add_argument("--snr_min", type=float, default=14.0)
    p.add_argument("--snr_max", type=float, default=None)
    p.add_argument("--reward_key", type=str, default="reward_ber")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--imitation_epochs", type=int, default=20)
    p.add_argument("--reinforce_epochs", type=int, default=80)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--seeds", type=str, default="7,42,123")
    p.add_argument("--out_dir", type=str, default="results/rl_c1_stage2_repro")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_seed_list(seed_text: str) -> List[int]:
    vals = []
    for s in seed_text.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("No valid seed found in --seeds.")
    return vals


def compute_per_snr_metrics(
    data_path: Path,
    checkpoint_path: Path,
    reward_key: str,
    snr_min: float | None,
    snr_max: float | None,
    device: str,
) -> Dict:
    d = load_offline_bandit_data(
        str(data_path),
        snr_min=snr_min,
        snr_max=snr_max,
        reward_key=reward_key,
    )
    if d.ber_actions is None:
        raise ValueError("Dataset must include ber_actions.")

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    pack = load_policy(str(checkpoint_path), torch_device)
    model = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6

    x = (d.states.astype(np.float32) - state_mean) / state_std
    with torch.no_grad():
        act = model.greedy_action(torch.from_numpy(x).to(torch_device)).cpu().numpy().astype(np.int64)

    b = d.ber_actions.astype(np.float64)
    o = d.oracle_actions.astype(np.int64)
    base = int(d.base_action)

    out_rows = []
    snr_vals = np.sort(np.unique(d.snr_db.astype(np.int64))) if d.snr_db is not None else np.array([-1], dtype=np.int64)
    for s in snr_vals:
        if s < 0:
            idx = np.arange(d.num_samples)
        else:
            idx = np.where(d.snr_db.astype(np.int64) == s)[0]
        if idx.size == 0:
            continue
        bb = float(np.mean(b[idx, base]))
        bp = float(np.mean(b[idx, act[idx]]))
        bo = float(np.mean(b[idx, o[idx]]))
        out_rows.append(
            {
                "snr_db": int(s),
                "ber_base": bb,
                "ber_policy": bp,
                "ber_oracle": bo,
                "ber_gain_vs_base": float((bb - bp) / max(bb, 1e-12)),
                "ber_gap_to_oracle": float((bp - bo) / max(bp, 1e-12)),
            }
        )

    bb_all = float(np.mean(b[:, base]))
    bp_all = float(np.mean(b[np.arange(d.num_samples), act]))
    bo_all = float(np.mean(b[np.arange(d.num_samples), o]))
    overall = {
        "ber_base": bb_all,
        "ber_policy": bp_all,
        "ber_oracle": bo_all,
        "ber_gain_vs_base": float((bb_all - bp_all) / max(bb_all, 1e-12)),
        "ber_gap_to_oracle": float((bp_all - bo_all) / max(bp_all, 1e-12)),
        "action_match_rate": float(np.mean(act == o)),
    }

    return {"overall": overall, "per_snr": out_rows}


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    root = Path(args.project_root).resolve()
    out_root = (root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    py = args.python_exe
    train_py = str((root / "python" / "rl_c1" / "train_reinforce.py").resolve())
    eval_py = str((root / "python" / "rl_c1" / "eval_policy.py").resolve())
    data_path = (root / args.data).resolve()

    rows: List[Dict] = []
    per_seed_details = {}

    for seed in seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            py,
            train_py,
            "--data",
            str(data_path),
            "--snr_min",
            str(args.snr_min),
            "--reward_key",
            args.reward_key,
            "--split_mode",
            "sequence",
            "--batch_size",
            str(args.batch_size),
            "--imitation_epochs",
            str(args.imitation_epochs),
            "--epochs",
            str(args.reinforce_epochs),
            "--reward_scale",
            str(args.reward_scale),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--output_dir",
            str(seed_dir),
        ]
        if args.snr_max is not None:
            train_cmd += ["--snr_max", str(args.snr_max)]
        run_cmd(train_cmd, cwd=root)

        eval_json = seed_dir / "eval_metrics.json"
        eval_cmd = [
            py,
            eval_py,
            "--data",
            str(data_path),
            "--checkpoint",
            str(seed_dir / "best_reinforce_policy.pt"),
            "--snr_min",
            str(args.snr_min),
            "--reward_key",
            args.reward_key,
            "--device",
            args.device,
            "--save_json",
            str(eval_json),
        ]
        if args.snr_max is not None:
            eval_cmd += ["--snr_max", str(args.snr_max)]
        run_cmd(eval_cmd, cwd=root)

        with eval_json.open("r", encoding="utf-8") as f:
            em = json.load(f)

        extra = compute_per_snr_metrics(
            data_path=data_path,
            checkpoint_path=seed_dir / "best_reinforce_policy.pt",
            reward_key=args.reward_key,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            device=args.device,
        )
        per_seed_details[str(seed)] = extra

        row = {
            "seed": seed,
            "avg_ber_base": float(em.get("avg_ber_base", np.nan)),
            "avg_ber_policy": float(em.get("avg_ber_policy", np.nan)),
            "avg_ber_oracle": float(em.get("avg_ber_oracle", np.nan)),
            "ber_gain_vs_base": float(em.get("ber_gain_vs_base", np.nan)),
            "ber_gap_to_oracle": float(em.get("ber_gap_to_oracle", np.nan)),
            "action_match_rate": float(em.get("action_match_rate", np.nan)),
            "policy_switch_rate": float(em.get("policy_switch_rate", np.nan))
            if em.get("policy_switch_rate") is not None
            else np.nan,
            "oracle_switch_rate": float(em.get("oracle_switch_rate", np.nan))
            if em.get("oracle_switch_rate") is not None
            else np.nan,
        }
        rows.append(row)

    table_csv = out_root / "reproducibility_seed_table.csv"
    fields = [
        "seed",
        "avg_ber_base",
        "avg_ber_policy",
        "avg_ber_oracle",
        "ber_gain_vs_base",
        "ber_gap_to_oracle",
        "action_match_rate",
        "policy_switch_rate",
        "oracle_switch_rate",
    ]
    with table_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def stats(key: str) -> Dict:
        x = np.array([float(r[key]) for r in rows], dtype=np.float64)
        return {
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x, ddof=1) if x.size > 1 else 0.0),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
        }

    summary = {
        "config": {
            "data": str(data_path),
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
            "reward_key": args.reward_key,
            "batch_size": args.batch_size,
            "imitation_epochs": args.imitation_epochs,
            "reinforce_epochs": args.reinforce_epochs,
            "reward_scale": args.reward_scale,
            "device": args.device,
            "seeds": seeds,
        },
        "aggregate": {
            "ber_gain_vs_base": stats("ber_gain_vs_base"),
            "ber_gap_to_oracle": stats("ber_gap_to_oracle"),
            "action_match_rate": stats("action_match_rate"),
            "policy_switch_rate": stats("policy_switch_rate"),
        },
        "seed_rows": rows,
        "per_seed_per_snr": per_seed_details,
    }
    summary_json = out_root / "reproducibility_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Reproducibility Done ====")
    print(f"Saved table:   {table_csv}")
    print(f"Saved summary: {summary_json}")
    print(
        "BER gain vs base (mean+/-std): "
        f"{100*summary['aggregate']['ber_gain_vs_base']['mean']:.2f}% +/- "
        f"{100*summary['aggregate']['ber_gain_vs_base']['std']:.2f}%"
    )


if __name__ == "__main__":
    main()
