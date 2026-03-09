"""
Feature extraction helpers for C1 contextual-bandit state construction.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def feature_from_heff(
    heff: np.ndarray,
    q: int = 0,
    top_k_sv: int = 4,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build a low-dimensional state feature from one effective channel matrix.

    Args:
        heff: complex matrix [N, N]
        q: number of guard subcarriers trimmed on both sides
        top_k_sv: number of leading singular values to keep
    """
    if heff.ndim != 2 or heff.shape[0] != heff.shape[1]:
        raise ValueError(f"Expected square [N,N] matrix, got {heff.shape}")

    n = heff.shape[0]
    if q > 0:
        heff = heff[q : n - q, q : n - q]

    svals = np.linalg.svd(heff, compute_uv=False)
    sv_ref = max(float(svals[0]), eps)
    sv_feat = np.zeros(top_k_sv, dtype=np.float32)
    take = min(top_k_sv, svals.size)
    sv_feat[:take] = (svals[:take] / sv_ref).astype(np.float32)

    frob = np.linalg.norm(heff, ord="fro") / np.sqrt(max(1, heff.shape[0]))
    cond = np.linalg.cond(heff + eps * np.eye(heff.shape[0], dtype=heff.dtype))
    cond_log10 = np.log10(cond + eps)

    mag = np.abs(heff)
    mag_mean = float(mag.mean())
    mag_std = float(mag.std())
    diag_ratio = float(np.trace(mag) / (np.sum(mag) + eps))

    feat = np.concatenate(
        [
            np.array([frob, cond_log10, mag_mean, mag_std, diag_ratio], dtype=np.float32),
            sv_feat,
        ],
        axis=0,
    )
    return feat


def feature_from_channel_params(
    alpha: np.ndarray,
    ell: np.ndarray,
    h: np.ndarray,
    snr_db: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build state feature from physical channel parameters.
    """
    alpha = np.asarray(alpha).reshape(-1)
    ell = np.asarray(ell).reshape(-1)
    h = np.asarray(h).reshape(-1)

    max_alpha = float(np.max(np.abs(alpha))) if alpha.size else 0.0
    mean_alpha = float(np.mean(np.abs(alpha))) if alpha.size else 0.0
    delay_spread = float(np.std(ell) / max(np.max(ell) + 1.0, 1.0)) if ell.size else 0.0
    h_l4_over_l2 = float(np.linalg.norm(h, ord=4) / max(np.linalg.norm(h, ord=2), eps)) if h.size else 0.0
    snr_norm = float(snr_db / 30.0)

    return np.array([max_alpha, mean_alpha, delay_spread, h_l4_over_l2, snr_norm], dtype=np.float32)


def feature_from_detector_stats(
    residual_norm: float,
    logits_entropy: float,
    mse_proxy: float,
) -> np.ndarray:
    """
    Build state feature from detector internal statistics.
    """
    return np.array([residual_norm, logits_entropy, mse_proxy], dtype=np.float32)


def merge_feature_parts(parts: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    """
    Merge multiple feature parts into a single 1-D vector.
    """
    vectors = []
    for _, value in parts.items():
        if value is None:
            continue
        v = np.asarray(value, dtype=np.float32).reshape(-1)
        vectors.append(v)
    if not vectors:
        raise ValueError("No non-empty feature parts provided.")
    return np.concatenate(vectors, axis=0)
