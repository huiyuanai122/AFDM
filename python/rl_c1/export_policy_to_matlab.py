"""
Export trained RL C1 policy checkpoint to MATLAB-friendly .mat parameters.

Saved fields:
- state_mean, state_std
- num_linear_layers
- W1,b1,W2,b2,... (column-vector compatible)
- action_dim, state_dim
- c1_grid (if available)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import savemat
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import load_offline_bandit_data
from rl_c1.eval_policy import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RL C1 policy to MATLAB .mat")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/rl_c1_stage2_full/warmstart_rewardber/best_reinforce_policy.pt",
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/oracle_policy_dataset.mat",
        help="Optional dataset to fetch c1_grid if checkpoint doesn't store it.",
    )
    p.add_argument("--reward_key", type=str, default="reward_ber")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--output_mat",
        type=str,
        default="results/rl_c1_policy_matlab_params.mat",
    )
    return p.parse_args()


def extract_linear_layers(model: nn.Module) -> List[nn.Linear]:
    layers: List[nn.Linear] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layers.append(m)
    if not layers:
        raise ValueError("No nn.Linear layer found in policy model.")
    return layers


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_mat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pack = load_policy(args.checkpoint, device)
    model = pack["model"]
    state_mean = np.asarray(pack["state_mean"], dtype=np.float64).reshape(-1, 1)
    state_std = np.asarray(pack["state_std"], dtype=np.float64).reshape(-1, 1)

    layers = extract_linear_layers(model)
    payload: Dict[str, np.ndarray] = {
        "state_mean": state_mean,
        "state_std": state_std,
        "num_linear_layers": np.array([[len(layers)]], dtype=np.float64),
        "state_dim": np.array([[state_mean.size]], dtype=np.float64),
        "action_dim": np.array([[layers[-1].out_features]], dtype=np.float64),
    }

    for i, layer in enumerate(layers, start=1):
        w = layer.weight.detach().cpu().numpy().astype(np.float64)  # [out, in]
        b = layer.bias.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)  # [out,1]
        payload[f"W{i}"] = w
        payload[f"b{i}"] = b

    # Try to keep c1_grid and reward_key for MATLAB-side tracing/debugging.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    reward_key = ckpt.get("reward_key", "unknown")
    payload["reward_key"] = np.array([[str(reward_key)]], dtype=object)

    c1_grid = ckpt.get("c1_grid", None)
    if c1_grid is None:
        try:
            d = load_offline_bandit_data(args.data, reward_key=args.reward_key)
            c1_grid = d.c1_grid
        except Exception:
            c1_grid = None
    if c1_grid is not None:
        payload["c1_grid"] = np.asarray(c1_grid, dtype=np.float64).reshape(-1, 1)

    savemat(str(out_path), payload)

    print("==== Export RL Policy To MATLAB ====")
    print(f"checkpoint: {args.checkpoint}")
    print(f"saved_mat:  {out_path}")
    print(f"layers:     {len(layers)}")
    print(f"state_dim:  {state_mean.size}")
    print(f"action_dim: {layers[-1].out_features}")
    print(f"reward_key: {reward_key}")


if __name__ == "__main__":
    main()
