"""
Train a contextual-bandit policy for C1 selection with REINFORCE.

Example:
python python/rl_c1/train_reinforce.py ^
  --data data/oracle_policy_dataset.mat ^
  --snr_min 14 ^
  --epochs 80 ^
  --batch_size 256 ^
  --output_dir results/rl_c1
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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(CURRENT_DIR)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from rl_c1.env_c1_bandit import evaluate_actions, load_offline_bandit_data, split_indices
from rl_c1.env_c1_bandit import split_indices_by_group
from rl_c1.policy_net import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C1 contextual-bandit policy (REINFORCE).")
    parser.add_argument("--data", type=str, required=True, help="Offline bandit table (.mat v7.3 or .npz).")
    parser.add_argument("--output_dir", type=str, default="../results/rl_c1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--snr_min", type=float, default=14.0)
    parser.add_argument("--snr_max", type=float, default=None)
    parser.add_argument("--reward_key", type=str, default="reward", help="Reward field name in dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="sequence",
        choices=["sample", "sequence"],
        help="Train/val split strategy. 'sequence' uses sequence_id if available.",
    )

    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--baseline_momentum", type=float, default=0.95)
    parser.add_argument("--reward_scale", type=float, default=1000.0)
    parser.add_argument("--imitation_epochs", type=int, default=20)
    parser.add_argument("--imitation_lr", type=float, default=1e-3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_subset(
    policy: MLPPolicy,
    states: torch.Tensor,
    rewards: np.ndarray,
    oracle_actions: np.ndarray,
    base_action: int,
    indices: np.ndarray,
    device: torch.device,
    ber_actions: np.ndarray | None = None,
) -> Dict[str, float]:
    policy.eval()
    with torch.no_grad():
        st = states[indices].to(device)
        greedy = policy.greedy_action(st).cpu().numpy().astype(np.int64)
    return evaluate_actions(
        rewards=rewards[indices],
        actions=greedy,
        oracle_actions=oracle_actions[indices],
        base_action=base_action,
        ber_actions=ber_actions[indices] if ber_actions is not None else None,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_offline_bandit_data(
        args.data, snr_min=args.snr_min, snr_max=args.snr_max, reward_key=args.reward_key
    )
    if args.split_mode == "sequence" and data.sequence_id is not None:
        train_idx, val_idx = split_indices_by_group(data.sequence_id, args.val_ratio, args.seed)
        split_used = "sequence"
    else:
        train_idx, val_idx = split_indices(data.num_samples, args.val_ratio, args.seed)
        split_used = "sample"

    states_np = data.states.astype(np.float32)
    rewards_np = data.rewards.astype(np.float32)
    oracle_np = data.oracle_actions.astype(np.int64)
    ber_np = data.ber_actions.astype(np.float32) if data.ber_actions is not None else None

    state_mean = states_np[train_idx].mean(axis=0, keepdims=True)
    state_std = states_np[train_idx].std(axis=0, keepdims=True) + 1e-6
    states_norm = (states_np - state_mean) / state_std

    states = torch.from_numpy(states_norm)
    rewards = torch.from_numpy(rewards_np)

    policy = MLPPolicy(
        state_dim=data.state_dim,
        action_dim=data.num_actions,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    running_baseline = 0.0

    history = []
    best_val_reward = -float("inf")
    best_ckpt = out_dir / "best_reinforce_policy.pt"

    print("==== Offline REINFORCE Training ====")
    print(f"samples={data.num_samples}, state_dim={data.state_dim}, actions={data.num_actions}")
    print(f"train={train_idx.size}, val={val_idx.size}, base_action={data.base_action}")
    print(f"reward_key={args.reward_key}, has_ber_actions={data.ber_actions is not None}")
    print(f"split_mode={split_used}")

    # ===== 1) Supervised warm-start with oracle labels =====
    if args.imitation_epochs > 0:
        print("\n==== Oracle Imitation Warm-start ====")
        imitation_opt = optim.Adam(policy.parameters(), lr=args.imitation_lr, weight_decay=args.weight_decay)

        for ie in range(1, args.imitation_epochs + 1):
            policy.train()
            perm = np.random.permutation(train_idx)
            ce_sum = 0.0
            step_count = 0

            for start in range(0, perm.size, args.batch_size):
                batch_idx = perm[start : start + args.batch_size]
                st = states[batch_idx].to(device)
                target = torch.from_numpy(oracle_np[batch_idx]).to(device)

                logits = policy(st)
                loss = F.cross_entropy(logits, target)

                imitation_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
                imitation_opt.step()

                ce_sum += float(loss.item())
                step_count += 1

            train_metric = evaluate_subset(
                policy, states, rewards_np, oracle_np, data.base_action, train_idx, device, ber_np
            )
            val_metric = evaluate_subset(
                policy, states, rewards_np, oracle_np, data.base_action, val_idx, device, ber_np
            )

            print(
                f"imit={ie:03d} "
                f"ce={ce_sum / max(step_count, 1):.4f} "
                f"val_gain_vs_base={100*val_metric['rel_gain_vs_base']:.2f}% "
                f"val_match={100*val_metric['action_match_rate']:.2f}%"
            )

        # Save warm-start checkpoint before RL fine-tuning.
        warm_train_metric = evaluate_subset(
            policy, states, rewards_np, oracle_np, data.base_action, train_idx, device, ber_np
        )
        warm_val_metric = evaluate_subset(
            policy, states, rewards_np, oracle_np, data.base_action, val_idx, device, ber_np
        )
        if "avg_ber_policy" in warm_val_metric:
            best_metric_value = float(warm_val_metric["avg_ber_policy"])
            best_metric_name = "val_ber"
        else:
            best_metric_value = float(warm_val_metric["avg_reward_policy"])
            best_metric_name = "val_reward"
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "state_mean": state_mean.astype(np.float32),
                "state_std": state_std.astype(np.float32),
                "state_dim": data.state_dim,
                "action_dim": data.num_actions,
                "hidden_dims": args.hidden_dims,
                "base_action": int(data.base_action),
                "c1_grid": data.c1_grid,
                "reward_key": args.reward_key,
            },
            best_ckpt,
        )
        history.append(
            {
                "epoch": 0,
                "stage": "imitation_end",
                "loss": None,
                "sampled_reward": None,
                "entropy": None,
                "train_avg_reward": warm_train_metric["avg_reward_policy"],
                "train_gain_vs_base": warm_train_metric["rel_gain_vs_base"],
                "val_avg_reward": warm_val_metric["avg_reward_policy"],
                "val_gain_vs_base": warm_val_metric["rel_gain_vs_base"],
                "val_gap_to_oracle": warm_val_metric["rel_gap_to_oracle"],
                "val_match_rate": warm_val_metric["action_match_rate"],
                "val_avg_ber": warm_val_metric.get("avg_ber_policy"),
            }
        )
        print(
            f"imitation_end val_reward={warm_val_metric['avg_reward_policy']:.5e} "
            f"val_gain_vs_base={100*warm_val_metric['rel_gain_vs_base']:.2f}% "
            f"val_match={100*warm_val_metric['action_match_rate']:.2f}%"
        )
    else:
        if ber_np is not None:
            best_metric_name = "val_ber"
            best_metric_value = float("inf")
        else:
            best_metric_name = "val_reward"
            best_metric_value = -float("inf")

    # ===== 2) Reinforce fine-tuning =====
    for epoch in range(1, args.epochs + 1):
        policy.train()

        perm = np.random.permutation(train_idx)
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_entropy = 0.0
        steps = 0

        for start in range(0, perm.size, args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            st = states[batch_idx].to(device)
            rw = rewards[batch_idx].to(device)

            logits = policy(st)
            dist = Categorical(logits=logits)
            action = dist.sample()

            chosen = rw.gather(1, action.unsqueeze(1)).squeeze(1)
            chosen_scaled = chosen * args.reward_scale
            mean_reward = chosen_scaled.mean().item()
            running_baseline = (
                args.baseline_momentum * running_baseline
                + (1 - args.baseline_momentum) * mean_reward
            )

            advantage = chosen_scaled - running_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            loss = -(log_prob * advantage.detach()).mean() - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_reward += float(mean_reward)
            epoch_entropy += float(entropy.item())
            steps += 1

        train_metric = evaluate_subset(
            policy, states, rewards_np, oracle_np, data.base_action, train_idx, device, ber_np
        )
        val_metric = evaluate_subset(
            policy, states, rewards_np, oracle_np, data.base_action, val_idx, device, ber_np
        )

        row = {
            "epoch": epoch,
            "loss": epoch_loss / max(steps, 1),
            "sampled_reward": epoch_reward / max(steps, 1),
            "entropy": epoch_entropy / max(steps, 1),
            "train_avg_reward": train_metric["avg_reward_policy"],
            "train_gain_vs_base": train_metric["rel_gain_vs_base"],
            "val_avg_reward": val_metric["avg_reward_policy"],
            "val_gain_vs_base": val_metric["rel_gain_vs_base"],
            "val_gap_to_oracle": val_metric["rel_gap_to_oracle"],
            "val_match_rate": val_metric["action_match_rate"],
            "val_avg_ber": val_metric.get("avg_ber_policy"),
        }
        history.append(row)

        if "avg_ber_policy" in val_metric:
            candidate_name = "val_ber"
            candidate_value = float(val_metric["avg_ber_policy"])
            is_better = candidate_value < best_metric_value
        else:
            candidate_name = "val_reward"
            candidate_value = float(val_metric["avg_reward_policy"])
            is_better = candidate_value > best_metric_value

        if is_better:
            best_metric_name = candidate_name
            best_metric_value = candidate_value
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "state_mean": state_mean.astype(np.float32),
                    "state_std": state_std.astype(np.float32),
                    "state_dim": data.state_dim,
                    "action_dim": data.num_actions,
                    "hidden_dims": args.hidden_dims,
                    "base_action": int(data.base_action),
                    "c1_grid": data.c1_grid,
                    "reward_key": args.reward_key,
                },
                best_ckpt,
            )

        val_ber_str = f"{row['val_avg_ber']:.5e}" if row["val_avg_ber"] is not None else "NA"
        print(
            f"epoch={epoch:03d} "
            f"loss={row['loss']:.5f} "
            f"val_reward={row['val_avg_reward']:.5e} "
            f"val_ber={val_ber_str} "
            f"val_gain_vs_base={100*row['val_gain_vs_base']:.2f}% "
            f"val_match={100*row['val_match_rate']:.2f}%"
        )

    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining done.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"History: {out_dir / 'train_history.json'}")
    print(f"Best metric: {best_metric_name}={best_metric_value:.6e}")


if __name__ == "__main__":
    main()
