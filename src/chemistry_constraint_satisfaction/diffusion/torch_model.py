"""
PyTorch GNN for the molecular diffusion model (same architecture as NumPy version).

Used for gradient-based training; weights can be copied into MolecularDiffusionModel
for inference with the existing supervisor.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ATOM_FEAT_DIM, NUM_ELEM


class TorchGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = (adj @ x) / deg
        return F.relu(self.w_self(x) + self.w_neigh(agg))


class MolDiffusionNet(nn.Module):
    """Predicts clean atom features and bond classes from noisy (x_t, adj_t)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gc1 = TorchGraphConv(ATOM_FEAT_DIM, hidden_dim)
        self.gc2 = TorchGraphConv(hidden_dim, hidden_dim)
        self.atom_head = nn.Linear(hidden_dim, ATOM_FEAT_DIM)
        self.bond_head = nn.Linear(hidden_dim * 2, 4)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, F), adj: (N, N)
        Returns atom_logits (N, ATOM_FEAT_DIM), bond_logits (N, N, 4)
        """
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        atom_logits = self.atom_head(h2)
        n = x.shape[0]
        bond_logits = torch.zeros(n, n, 4, device=x.device, dtype=x.dtype)
        for i in range(n):
            for j in range(i + 1, n):
                logits = self.bond_head(torch.cat([h2[i], h2[j]], dim=-1))
                bond_logits[i, j] = logits
                bond_logits[j, i] = logits
        return atom_logits, bond_logits


def alpha_bar_beta(t: int, T: int) -> Tuple[float, float]:
    """Cumulative alpha_bar at t and beta_t (same schedule as NumPy model)."""
    beta_t = 1e-4 + (t / T) * (0.1 - 1e-4)
    ab = 1.0
    for s in range(1, t + 1):
        b = 1e-4 + (s / T) * (0.1 - 1e-4)
        ab *= 1.0 - b
    return ab, beta_t


def add_noise_to_x(
    x0: torch.Tensor, t: int, T: int, rng: torch.Generator | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian noise on continuous atom features."""
    ab, _ = alpha_bar_beta(t, T)
    sqrt_ab = math.sqrt(ab)
    sqrt_1m = math.sqrt(1.0 - ab)
    eps = torch.randn(x0.shape, generator=rng, device=x0.device, dtype=x0.dtype)
    x_t = sqrt_ab * x0 + sqrt_1m * eps
    return x_t, eps


def add_noise_to_adj(
    adj0: torch.Tensor, t: int, T: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Discrete bond corruption (matches NumPy forward_noisy idea)."""
    ab, _ = alpha_bar_beta(t, T)
    device, dtype = adj0.device, adj0.dtype
    n = adj0.shape[0]
    u = torch.rand((n, n), generator=generator, device=device, dtype=torch.float32)
    corrupt = u < (1.0 - ab)
    rnd = torch.randint(0, 4, (n, n), generator=generator, device=device)
    adj_noisy = adj0.clone()
    adj_noisy = torch.where(corrupt, rnd.float(), adj_noisy)
    triu = torch.triu(adj_noisy, 1)
    adj_sym = triu + triu.T
    return adj_sym.to(dtype=dtype)
