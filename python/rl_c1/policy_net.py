"""
Simple MLP policy network for discrete C1 action selection.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


def _parse_hidden_dims(hidden_dims: Iterable[int] | str) -> List[int]:
    if isinstance(hidden_dims, str):
        parts = [p.strip() for p in hidden_dims.split(",") if p.strip()]
        if not parts:
            raise ValueError("hidden_dims string is empty.")
        return [int(p) for p in parts]
    dims = [int(d) for d in hidden_dims]
    if not dims:
        raise ValueError("hidden_dims must be non-empty.")
    return dims


class MLPPolicy(nn.Module):
    """
    MLP policy that outputs logits over discrete C1 grid actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] | str = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = _parse_hidden_dims(hidden_dims)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)
