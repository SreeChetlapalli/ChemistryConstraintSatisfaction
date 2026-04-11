"""
PyTorch training loop for the molecular diffusion model.

The main pieces here:
- TorchMolecularDenoiser  (nn.Module version of the GNN)
- constraint-aware loss    (penalises valency / conservation violations)
- curriculum schedule      (ramp supervisor strictness over epochs)
- train()                  (puts it all together)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constraints.chemical_axioms import (
    Atom,
    MolecularState,
    ConstraintResult,
    MAX_VALENCY,
    CHARGE_VALENCY_DELTA,
    check_intermediate,
)
from .model import (
    ELEMENTS,
    ELEM_TO_IDX,
    NUM_ELEM,
    ATOM_FEAT_DIM,
    atom_to_feat,
    encode_molecule,
)


# ---------------------------------------------------------------------------
# Trainable PyTorch GNN
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Self + mean-neighbour transform with ReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.bmm(adj, x) / deg if x.dim() == 3 else (adj @ x) / deg
        return F.relu(self.W_self(x) + self.W_neigh(agg))


class TorchMolecularDenoiser(nn.Module):
    """
    PyTorch version of the GNN denoiser (same arch as the NumPy one,
    but trainable).

    forward(x_t, adj_t, t_frac) -> (x0_pred, bond_logits)
      x_t      : (B, N, F) or (N, F)
      adj_t    : (B, N, N) or (N, N)
      t_frac   : (B,) or scalar, in [0, 1]
      x0_pred  : same shape as x_t
      bond_logits : (B, N, N, 4)  — logits for bond orders 0..3
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(ATOM_FEAT_DIM, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.atom_head = nn.Linear(hidden_dim, ATOM_FEAT_DIM)
        self.bond_head = nn.Linear(hidden_dim * 2, 4)

    def forward(
        self,
        x_t: torch.Tensor,
        adj_t: torch.Tensor,
        t_frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batched = x_t.dim() == 3
        if not batched:
            x_t = x_t.unsqueeze(0)
            adj_t = adj_t.unsqueeze(0)
            t_frac = t_frac.unsqueeze(0)

        B, N, _ = x_t.shape

        t_emb = self.time_embed(t_frac.unsqueeze(-1))          # (B, H)
        h = self.input_proj(x_t) + t_emb.unsqueeze(1)

        for layer in self.layers:
            h = layer(h, adj_t)

        x0_pred = self.atom_head(h)

        # pairwise concat for bond prediction
        hi = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)
        hj = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        pair = torch.cat([hi, hj], dim=-1)
        bond_logits = self.bond_head(pair)

        if not batched:
            x0_pred = x0_pred.squeeze(0)
            bond_logits = bond_logits.squeeze(0)

        return x0_pred, bond_logits


# ---------------------------------------------------------------------------
# Noise schedule (differentiable, supports cosine + linear)
# ---------------------------------------------------------------------------

def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021). ``t`` in [0, 1]."""
    f_t = torch.cos((t + s) / (1 + s) * (math.pi / 2)) ** 2
    f_0 = math.cos(s / (1 + s) * (math.pi / 2)) ** 2
    return (f_t / f_0).clamp(min=1e-5, max=1.0)


def linear_alpha_bar(t: torch.Tensor, beta_min: float = 1e-4, beta_max: float = 0.1) -> torch.Tensor:
    """Linear schedule matching the NumPy model."""
    beta = beta_min + t * (beta_max - beta_min)
    log_ab = -0.5 * t * (beta_min + beta)
    return torch.exp(log_ab).clamp(min=1e-5, max=1.0)


# ---------------------------------------------------------------------------
# Constraint-aware loss
# ---------------------------------------------------------------------------

def _valency_penalty(
    x0_pred: torch.Tensor,
    bond_logits: torch.Tensor,
) -> torch.Tensor:
    """Soft penalty: mean(relu(predicted_bonds - allowed_valency)).
    Uses soft argmax over element probs so it stays differentiable."""
    elem_probs = F.softmax(x0_pred[..., :NUM_ELEM], dim=-1)   # (*, N, E)
    max_vals = torch.tensor(
        [MAX_VALENCY.get(e, 4) for e in ELEMENTS], dtype=x0_pred.dtype, device=x0_pred.device
    )
    allowed = (elem_probs * max_vals).sum(dim=-1)              # (*, N)

    bond_probs = F.softmax(bond_logits, dim=-1)                # (*, N, N, 4)
    orders = torch.arange(4, dtype=bond_logits.dtype, device=bond_logits.device)
    expected_bonds = (bond_probs * orders).sum(dim=-1)         # (*, N, N)
    total_bonds = expected_bonds.sum(dim=-1)                   # (*, N)

    excess = F.relu(total_bonds - allowed)
    return excess.mean()


def _element_conservation_penalty(
    x0_pred: torch.Tensor,
    x_clean: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and target element-count distributions
    (soft, so gradients still flow)."""
    # Softmax over element dim (-1), not atom dim (-2)
    pred_probs = F.softmax(x0_pred[..., :NUM_ELEM], dim=-1)
    pred_elem_counts = pred_probs.sum(dim=-2)         # (*, E)

    # Use hard target counts for the clean state
    # x_clean has one-hot elems in first NUM_ELEM slots
    target_elem_counts = x_clean[..., :NUM_ELEM].sum(dim=-2)   # (*, E)

    return F.mse_loss(pred_elem_counts, target_elem_counts)


@dataclass
class LossWeights:
    denoising: float = 1.0
    bond_ce: float = 1.0
    valency_penalty: float = 0.5
    element_conservation: float = 1.0


def compute_loss(
    x0_pred: torch.Tensor,
    bond_logits: torch.Tensor,
    x_clean: torch.Tensor,
    adj_clean: torch.Tensor,
    weights: Optional[LossWeights] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Returns (total_loss, breakdown_dict)."""
    w = weights or LossWeights()
    breakdown: Dict[str, float] = {}

    l_denoise = F.mse_loss(x0_pred, x_clean)
    breakdown["denoising"] = l_denoise.item()

    adj_target = adj_clean.long().clamp(0, 3)
    l_bond = F.cross_entropy(
        bond_logits.reshape(-1, 4), adj_target.reshape(-1)
    )
    breakdown["bond_ce"] = l_bond.item()

    l_val = _valency_penalty(x0_pred, bond_logits)
    breakdown["valency_penalty"] = l_val.item()

    l_elem = _element_conservation_penalty(x0_pred, x_clean)
    breakdown["element_conservation"] = l_elem.item()

    total = (
        w.denoising * l_denoise
        + w.bond_ce * l_bond
        + w.valency_penalty * l_val
        + w.element_conservation * l_elem
    )
    breakdown["total"] = total.item()
    return total, breakdown


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    """Ramp penalty weights from 0 (during warmup) up to max_penalty_weight."""
    warmup_fraction: float = 0.25
    max_penalty_weight: float = 1.0

    def penalty_scale(self, epoch: int, total_epochs: int) -> float:
        frac = epoch / max(total_epochs, 1)
        if frac < self.warmup_fraction:
            return 0.0
        progress = (frac - self.warmup_fraction) / (1.0 - self.warmup_fraction)
        return self.max_penalty_weight * min(progress, 1.0)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def molecules_to_tensors(
    molecules: List[MolecularState],
    device: torch.device | str = "cpu",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert MolecularStates to (x, adj) tensor pairs.
    Returns a list (not a batch) since atom counts differ per molecule."""
    pairs = []
    for mol in molecules:
        x_np, adj_np = encode_molecule(mol)
        x = torch.from_numpy(x_np).to(device)
        adj = torch.from_numpy(adj_np).to(device)
        pairs.append((x, adj))
    return pairs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    schedule: str = "cosine"
    hidden_dim: int = 64
    num_layers: int = 2
    loss_weights: LossWeights = field(default_factory=LossWeights)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    log_every: int = 10
    device: str = "cpu"


@dataclass
class TrainResult:
    epoch_losses: List[Dict[str, float]]
    final_model: TorchMolecularDenoiser

    @property
    def best_loss(self) -> float:
        return min(d["total"] for d in self.epoch_losses) if self.epoch_losses else float("inf")


def train(
    molecules: List[MolecularState],
    config: Optional[TrainConfig] = None,
    verbose: bool = True,
) -> TrainResult:
    """Train the denoiser on known-valid molecules.

    Each epoch: sample random timesteps, add noise, predict clean state,
    backprop through denoising + constraint penalties (ramped by curriculum).
    """
    cfg = config or TrainConfig()
    device = torch.device(cfg.device)

    model = TorchMolecularDenoiser(
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    data = molecules_to_tensors(molecules, device=device)
    if not data:
        raise ValueError("Need at least one molecule to train on.")

    alpha_bar_fn = cosine_alpha_bar if cfg.schedule == "cosine" else linear_alpha_bar

    epoch_losses: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        batch_breakdowns: List[Dict[str, float]] = []

        for x_clean, adj_clean in data:
            N = x_clean.shape[0]
            t_frac = torch.rand(1, device=device)
            ab = alpha_bar_fn(t_frac).item()
            sqrt_ab = math.sqrt(ab)
            sqrt_1mab = math.sqrt(1.0 - ab)

            eps = torch.randn_like(x_clean)
            x_noisy = sqrt_ab * x_clean + sqrt_1mab * eps

            flip = (torch.rand(N, N, device=device) < (1.0 - ab)).float()
            adj_noisy = adj_clean.clone()
            adj_noisy = adj_noisy * (1 - flip) + (3 - adj_noisy).clamp(0, 3) * flip
            adj_noisy = torch.triu(adj_noisy, diagonal=1)
            adj_noisy = adj_noisy + adj_noisy.T

            x0_pred, bond_logits = model(x_noisy, adj_noisy, t_frac.squeeze())

            penalty_scale = cfg.curriculum.penalty_scale(epoch, cfg.epochs)
            cur_weights = LossWeights(
                denoising=cfg.loss_weights.denoising,
                bond_ce=cfg.loss_weights.bond_ce,
                valency_penalty=cfg.loss_weights.valency_penalty * penalty_scale,
                element_conservation=cfg.loss_weights.element_conservation * penalty_scale,
            )

            loss, breakdown = compute_loss(
                x0_pred, bond_logits, x_clean, adj_clean, cur_weights
            )

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_breakdowns.append(breakdown)

        avg = {
            k: sum(d[k] for d in batch_breakdowns) / len(batch_breakdowns)
            for k in batch_breakdowns[0]
        }
        epoch_losses.append(avg)

        if verbose and (epoch + 1) % cfg.log_every == 0:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "total")
            print(f"  [epoch {epoch+1:4d}/{cfg.epochs}]  loss={avg['total']:.4f}  ({parts})")

    return TrainResult(epoch_losses=epoch_losses, final_model=model)


# ---------------------------------------------------------------------------
# Convert trained PyTorch model -> NumPy model (for inference w/ Supervisor)
# ---------------------------------------------------------------------------

def export_to_numpy(
    torch_model: TorchMolecularDenoiser,
    schedule: str = "cosine",
) -> "MolecularDiffusionModel":
    """Copy weights into a NumPy MolecularDiffusionModel for use with the Supervisor."""
    from .model import MolecularDiffusionModel, NumpyLinear, NumpyGraphConv

    np_model = MolecularDiffusionModel(
        hidden_dim=torch_model.hidden_dim,
        seed=0,
        use_input_proj=True,
        schedule=schedule,
    )

    def _copy_linear(src: nn.Linear, dst: NumpyLinear) -> None:
        dst.W = src.weight.detach().cpu().numpy().astype(np.float32)
        dst.b = src.bias.detach().cpu().numpy().astype(np.float32)

    _copy_linear(torch_model.input_proj, np_model.input_proj)
    _copy_linear(torch_model.layers[0].W_self, np_model.gc1.W_self)
    _copy_linear(torch_model.layers[0].W_neigh, np_model.gc1.W_neigh)
    if len(torch_model.layers) > 1:
        _copy_linear(torch_model.layers[1].W_self, np_model.gc2.W_self)
        _copy_linear(torch_model.layers[1].W_neigh, np_model.gc2.W_neigh)

    _copy_linear(torch_model.atom_head, np_model.atom_head)
    _copy_linear(torch_model.bond_head, np_model.bond_head)

    return np_model
