"""
Offline contextual-bandit dataset loader and evaluation helpers for C1 selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


@dataclass
class OfflineBanditData:
    states: np.ndarray  # [S, D], float32
    rewards: np.ndarray  # [S, M], float32
    oracle_actions: np.ndarray  # [S], int64, zero-based
    base_action: int  # zero-based
    snr_db: Optional[np.ndarray] = None  # [S]
    sequence_id: Optional[np.ndarray] = None  # [S]
    time_index: Optional[np.ndarray] = None  # [S]
    c1_grid: Optional[np.ndarray] = None  # [M]
    ber_actions: Optional[np.ndarray] = None  # [S, M], optional true BER table

    @property
    def num_samples(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def num_actions(self) -> int:
        return int(self.rewards.shape[1])


class OfflineC1BanditEnv:
    """
    One-step contextual-bandit environment wrapper over offline table.
    """

    def __init__(
        self,
        data: OfflineBanditData,
        indices: Optional[np.ndarray] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.indices = (
            np.arange(data.num_samples, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self._cursor = 0
        self._current_local = None
        self._last_state = None

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def reset(self) -> np.ndarray:
        if self._cursor >= self.indices.size:
            self._cursor = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)

        self._current_local = int(self._cursor)
        self._cursor += 1

        global_idx = int(self.indices[self._current_local])
        state = self.data.states[global_idx].astype(np.float32)
        self._last_state = state
        return state

    def step(self, action: int):
        if self._current_local is None:
            raise RuntimeError("Call reset() before step().")
        global_idx = int(self.indices[self._current_local])
        action = int(action)
        if action < 0 or action >= self.data.num_actions:
            raise ValueError(f"Action out of range: {action}")

        reward = float(self.data.rewards[global_idx, action])
        info = {
            "index": global_idx,
            "oracle_action": int(self.data.oracle_actions[global_idx]),
            "base_action": int(self.data.base_action),
            "oracle_reward": float(self.data.rewards[global_idx, self.data.oracle_actions[global_idx]]),
            "base_reward": float(self.data.rewards[global_idx, self.data.base_action]),
        }
        if self.data.ber_actions is not None:
            info["oracle_ber"] = float(self.data.ber_actions[global_idx, self.data.oracle_actions[global_idx]])
            info["base_ber"] = float(self.data.ber_actions[global_idx, self.data.base_action])
            info["chosen_ber"] = float(self.data.ber_actions[global_idx, action])
        done = True
        next_state = self._last_state
        self._current_local = None
        return next_state, reward, done, info


def _align_table_by_samples(array: Optional[np.ndarray], num_samples: int, name: str) -> Optional[np.ndarray]:
    if array is None:
        return None
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {array.shape}")
    if array.shape[0] == num_samples:
        return array
    if array.shape[1] == num_samples:
        return array.T
    raise ValueError(f"Cannot align {name} with num_samples={num_samples}, shape={array.shape}")


def _read_h5_optional(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.array(f[key][()])


def _to_1d(a: Optional[np.ndarray], expected_len: int) -> Optional[np.ndarray]:
    if a is None:
        return None
    a = np.asarray(a).reshape(-1)
    if a.size != expected_len:
        raise ValueError(f"Unexpected length for optional field: {a.size}, expected {expected_len}")
    return a


def _load_from_npz(path: Path, reward_key: str = "reward") -> OfflineBanditData:
    data = np.load(path, allow_pickle=False)
    states = np.asarray(data["state"], dtype=np.float32)
    if reward_key in data:
        rewards = np.asarray(data[reward_key], dtype=np.float32)
    elif "reward" in data:
        rewards = np.asarray(data["reward"], dtype=np.float32)
    else:
        raise KeyError(f"Neither '{reward_key}' nor 'reward' found in npz file.")

    num_samples = states.shape[0]

    oracle = data["oracle_action"] if "oracle_action" in data else np.argmax(rewards, axis=1)
    oracle = np.asarray(oracle).reshape(-1).astype(np.int64)
    if oracle.min() == 1:
        oracle = oracle - 1

    base_action = int(np.asarray(data["base_action"]).reshape(-1)[0]) if "base_action" in data else 0
    if base_action >= 1 and base_action <= rewards.shape[1]:
        base_action -= 1

    snr_db = _to_1d(data["snr_db"] if "snr_db" in data else None, num_samples)
    seq_id = _to_1d(data["sequence_id"] if "sequence_id" in data else None, num_samples)
    time_idx = _to_1d(data["time_index"] if "time_index" in data else None, num_samples)
    c1_grid = np.asarray(data["c1_grid"]).reshape(-1) if "c1_grid" in data else None
    ber_actions = np.asarray(data["ber_actions"], dtype=np.float32) if "ber_actions" in data else None
    ber_actions = _align_table_by_samples(ber_actions, num_samples, "ber_actions")

    return OfflineBanditData(
        states=states,
        rewards=rewards,
        oracle_actions=oracle,
        base_action=base_action,
        snr_db=snr_db,
        sequence_id=seq_id,
        time_index=time_idx,
        c1_grid=c1_grid,
        ber_actions=ber_actions,
    )


def _load_from_mat_h5(path: Path, reward_key: str = "reward") -> OfflineBanditData:
    with h5py.File(path, "r") as f:
        raw_state = np.array(f["state"][()])
        if reward_key in f:
            raw_reward = np.array(f[reward_key][()])
        elif "reward" in f:
            raw_reward = np.array(f["reward"][()])
        else:
            raise KeyError(f"Neither '{reward_key}' nor 'reward' found in mat file.")

        # MATLAB v7.3 stores [D,S] for a [S,D] matrix.
        if raw_state.ndim != 2 or raw_reward.ndim != 2:
            raise ValueError("state/reward must be 2-D arrays in .mat file")

        if raw_state.shape[1] == raw_reward.shape[1]:
            # likely [D,S] / [M,S]
            states = raw_state.T.astype(np.float32)
            rewards = raw_reward.T.astype(np.float32)
        elif raw_state.shape[0] == raw_reward.shape[0]:
            states = raw_state.astype(np.float32)
            rewards = raw_reward.astype(np.float32)
        else:
            raise ValueError(f"Cannot align state {raw_state.shape} and reward {raw_reward.shape}")

        num_samples = states.shape[0]

        oracle_raw = _read_h5_optional(f, "oracle_action")
        if oracle_raw is None:
            oracle = np.argmax(rewards, axis=1).astype(np.int64)
        else:
            oracle = np.asarray(oracle_raw).reshape(-1).astype(np.int64)
            if oracle.size != num_samples:
                oracle = oracle.T.reshape(-1).astype(np.int64)
            if oracle.min() >= 1:
                oracle = oracle - 1

        base_raw = _read_h5_optional(f, "base_action")
        if base_raw is None:
            base_action = 0
        else:
            base_action = int(np.asarray(base_raw).reshape(-1)[0])
            if base_action >= 1 and base_action <= rewards.shape[1]:
                base_action -= 1

        snr_db = _read_h5_optional(f, "snr_db")
        sequence_id = _read_h5_optional(f, "sequence_id")
        time_index = _read_h5_optional(f, "time_index")
        c1_grid = _read_h5_optional(f, "c1_grid")
        ber_actions_raw = _read_h5_optional(f, "ber_actions")

    snr_db = _to_1d(snr_db, num_samples)
    sequence_id = _to_1d(sequence_id, num_samples)
    time_index = _to_1d(time_index, num_samples)
    c1_grid = np.asarray(c1_grid).reshape(-1) if c1_grid is not None else None
    ber_actions = _align_table_by_samples(ber_actions_raw, num_samples, "ber_actions")
    if ber_actions is not None:
        ber_actions = ber_actions.astype(np.float32)

    return OfflineBanditData(
        states=states,
        rewards=rewards,
        oracle_actions=oracle,
        base_action=base_action,
        snr_db=snr_db,
        sequence_id=sequence_id,
        time_index=time_index,
        c1_grid=c1_grid,
        ber_actions=ber_actions,
    )


def load_offline_bandit_data(
    dataset_path: str,
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None,
    reward_key: str = "reward",
) -> OfflineBanditData:
    """
    Load offline bandit table from .mat(v7.3) or .npz.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    if path.suffix.lower() == ".npz":
        data = _load_from_npz(path, reward_key=reward_key)
    elif path.suffix.lower() == ".mat":
        data = _load_from_mat_h5(path, reward_key=reward_key)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    if snr_min is None and snr_max is None:
        return data
    if data.snr_db is None:
        raise ValueError("SNR filtering requested but dataset has no snr_db field.")

    mask = np.ones(data.num_samples, dtype=bool)
    if snr_min is not None:
        mask &= data.snr_db >= snr_min
    if snr_max is not None:
        mask &= data.snr_db <= snr_max

    if not np.any(mask):
        raise ValueError("No sample remains after SNR filtering.")

    idx = np.where(mask)[0]
    return OfflineBanditData(
        states=data.states[idx],
        rewards=data.rewards[idx],
        oracle_actions=data.oracle_actions[idx],
        base_action=data.base_action,
        snr_db=data.snr_db[idx] if data.snr_db is not None else None,
        sequence_id=data.sequence_id[idx] if data.sequence_id is not None else None,
        time_index=data.time_index[idx] if data.time_index is not None else None,
        c1_grid=data.c1_grid,
        ber_actions=data.ber_actions[idx] if data.ber_actions is not None else None,
    )


def split_indices(num_samples: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    val_size = int(round(num_samples * val_ratio))
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])
    return train_idx, val_idx


def split_indices_by_group(
    group_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group-aware split. Samples sharing the same group id are kept in the same split.
    Useful for sequence data to avoid frame-level leakage.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    gid = np.asarray(group_ids).reshape(-1)
    uniq = np.unique(gid)
    if uniq.size < 2:
        # fallback to sample-level split
        return split_indices(gid.size, val_ratio, seed)

    rng = np.random.default_rng(seed)
    perm_groups = rng.permutation(uniq)
    val_group_count = max(1, int(round(uniq.size * val_ratio)))
    val_groups = set(perm_groups[:val_group_count].tolist())

    val_mask = np.array([g in val_groups for g in gid], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    if train_idx.size == 0 or val_idx.size == 0:
        return split_indices(gid.size, val_ratio, seed)
    return np.sort(train_idx), np.sort(val_idx)


def evaluate_actions(
    rewards: np.ndarray,
    actions: np.ndarray,
    oracle_actions: np.ndarray,
    base_action: int,
    ber_actions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate a vector of chosen actions against base and oracle.
    """
    s = np.arange(rewards.shape[0], dtype=np.int64)
    chosen_reward = rewards[s, actions]
    oracle_reward = rewards[s, oracle_actions]
    base_reward = rewards[s, base_action]

    avg_policy = float(np.mean(chosen_reward))
    avg_oracle = float(np.mean(oracle_reward))
    avg_base = float(np.mean(base_reward))

    action_acc = float(np.mean(actions == oracle_actions))
    regret = float(np.mean(oracle_reward - chosen_reward))
    rel_gain_vs_base = float((avg_policy - avg_base) / max(abs(avg_base), 1e-8))
    rel_gap_to_oracle = float((avg_oracle - avg_policy) / max(abs(avg_oracle), 1e-8))

    out = {
        "avg_reward_policy": avg_policy,
        "avg_reward_oracle": avg_oracle,
        "avg_reward_base": avg_base,
        "oracle_regret": regret,
        "action_match_rate": action_acc,
        "rel_gain_vs_base": rel_gain_vs_base,
        "rel_gap_to_oracle": rel_gap_to_oracle,
    }

    # If true BER table is provided, use it for BER metrics.
    if ber_actions is not None:
        ber_policy = float(np.mean(ber_actions[s, actions]))
        ber_oracle = float(np.mean(ber_actions[s, oracle_actions]))
        ber_base = float(np.mean(ber_actions[s, base_action]))
        out["avg_ber_policy"] = ber_policy
        out["avg_ber_oracle"] = ber_oracle
        out["avg_ber_base"] = ber_base
        out["ber_gain_vs_base"] = float((ber_base - ber_policy) / max(ber_base, 1e-12))
        out["ber_gap_to_oracle"] = float((ber_policy - ber_oracle) / max(ber_policy, 1e-12))
    elif np.max(rewards) <= 1e-12:
        # Otherwise fall back to interpreting reward ~= -BER.
        ber_policy = float(-avg_policy)
        ber_oracle = float(-avg_oracle)
        ber_base = float(-avg_base)
        out["avg_ber_policy"] = ber_policy
        out["avg_ber_oracle"] = ber_oracle
        out["avg_ber_base"] = ber_base
        out["ber_gain_vs_base"] = float((ber_base - ber_policy) / max(ber_base, 1e-12))
        out["ber_gap_to_oracle"] = float((ber_policy - ber_oracle) / max(ber_policy, 1e-12))

    return out


def compute_switch_rate(
    actions: np.ndarray,
    sequence_id: Optional[np.ndarray],
    time_index: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compute average switch_rate across sequences.
    """
    if sequence_id is None:
        return None
    seq = sequence_id.astype(np.int64)
    uniq = np.unique(seq)
    rates = []

    for sid in uniq:
        idx = np.where(seq == sid)[0]
        if idx.size <= 1:
            continue

        if time_index is not None:
            order = np.argsort(time_index[idx])
            idx = idx[order]
        seq_actions = actions[idx]
        rate = np.mean(seq_actions[1:] != seq_actions[:-1])
        rates.append(rate)

    if not rates:
        return None
    return float(np.mean(rates))
