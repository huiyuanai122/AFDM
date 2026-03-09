"""
Evaluate learned C1 policy against fixed-base and oracle baselines.

Example:
python python/rl_c1/eval_policy.py ^
  --data data/oracle_policy_dataset.mat ^
  --checkpoint results/rl_c1/best_reinforce_policy.pt ^
  --snr_min 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import (
    compute_switch_rate,
    evaluate_actions,
    load_offline_bandit_data,
)
from rl_c1.policy_net import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C1 contextual-bandit policy.")
    parser.add_argument("--data", type=str, required=True, help="Offline dataset (.mat/.npz).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr_min", type=float, default=None)
    parser.add_argument("--snr_max", type=float, default=None)
    parser.add_argument("--reward_key", type=str, default="reward", help="Reward field name in dataset.")
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def load_policy(checkpoint_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden_dims = ckpt.get("hidden_dims", "128,64")
    if isinstance(hidden_dims, (list, tuple)):
        hidden_dims = ",".join(str(x) for x in hidden_dims)

    model = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32)
    base_action = int(ckpt.get("base_action", 0))

    return {
        "model": model,
        "state_mean": state_mean,
        "state_std": state_std,
        "base_action_ckpt": base_action,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = load_offline_bandit_data(
        args.data, snr_min=args.snr_min, snr_max=args.snr_max, reward_key=args.reward_key
    )
    pack = load_policy(args.checkpoint, device)

    model: MLPPolicy = pack["model"]
    state_mean = pack["state_mean"]
    state_std = pack["state_std"] + 1e-6
    base_action = data.base_action

    states_norm = (data.states.astype(np.float32) - state_mean) / state_std
    states_t = torch.from_numpy(states_norm).to(device)
    with torch.no_grad():
        actions = model.greedy_action(states_t).cpu().numpy().astype(np.int64)

    metric = evaluate_actions(
        rewards=data.rewards,
        actions=actions,
        oracle_actions=data.oracle_actions,
        base_action=base_action,
        ber_actions=data.ber_actions,
    )

    oracle_switch = compute_switch_rate(data.oracle_actions, data.sequence_id, data.time_index)
    policy_switch = compute_switch_rate(actions, data.sequence_id, data.time_index)

    result = {
        "num_samples": int(data.num_samples),
        "state_dim": int(data.state_dim),
        "num_actions": int(data.num_actions),
        "base_action": int(base_action),
        **metric,
        "oracle_switch_rate": oracle_switch,
        "policy_switch_rate": policy_switch,
    }

    print("==== Policy Evaluation ====")
    print(f"samples={result['num_samples']}, actions={result['num_actions']}")
    print(f"avg_reward(base)   = {result['avg_reward_base']:.6e}")
    print(f"avg_reward(policy) = {result['avg_reward_policy']:.6e}")
    print(f"avg_reward(oracle) = {result['avg_reward_oracle']:.6e}")
    print(f"gain_vs_base       = {100*result['rel_gain_vs_base']:.2f}%")
    print(f"gap_to_oracle      = {100*result['rel_gap_to_oracle']:.2f}%")
    print(f"match_rate         = {100*result['action_match_rate']:.2f}%")
    if "avg_ber_policy" in result:
        print(f"avg_ber(base)      = {result['avg_ber_base']:.6e}")
        print(f"avg_ber(policy)    = {result['avg_ber_policy']:.6e}")
        print(f"avg_ber(oracle)    = {result['avg_ber_oracle']:.6e}")
        print(f"ber_gain_vs_base   = {100*result['ber_gain_vs_base']:.2f}%")
        print(f"ber_gap_to_oracle  = {100*result['ber_gap_to_oracle']:.2f}%")
    if result["policy_switch_rate"] is not None:
        print(f"switch_rate(policy)= {result['policy_switch_rate']:.4f}")
    if result["oracle_switch_rate"] is not None:
        print(f"switch_rate(oracle)= {result['oracle_switch_rate']:.4f}")

    save_json = args.save_json
    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
