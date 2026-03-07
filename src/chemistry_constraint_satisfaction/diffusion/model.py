"""
model.py
~~~~~~~~
Lightweight graph-based molecular diffusion model.

Architecture
------------
We represent a molecule as a fixed-size atom feature matrix X ∈ ℝ^{N×F}
and an adjacency (bond) matrix A ∈ {0,1,2,3}^{N×N} (bond orders).

The forward process adds Gaussian noise to X and Bernoulli noise to A.
The reverse (denoising) process is a small Graph Neural Network that
predicts the clean state from the noisy one.

For correctness-by-design, the model exposes ``step()`` so the supervisor
can intercept and verify each denoising step before it is committed.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import — CPU-only fallback using numpy when unavailable
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    TORCH_AVAILABLE = False

from ..constraints.chemical_axioms import (
    Atom, MolecularState, ATOMIC_MASS, MAX_VALENCY,
)


# ---------------------------------------------------------------------------
# Element encoding
# ---------------------------------------------------------------------------

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
NUM_ELEM = len(ELEMENTS)

# Feature vector per atom: one-hot element (10) + bond_count (1) + charge (1)
ATOM_FEAT_DIM = NUM_ELEM + 2


def atom_to_feat(atom: Atom) -> np.ndarray:
    """Encode an Atom into a fixed-length feature vector."""
    feat = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    idx = ELEM_TO_IDX.get(atom.element, 1)  # default to C
    feat[idx] = 1.0
    feat[NUM_ELEM]     = atom.bonds / 4.0           # normalised bond count
    feat[NUM_ELEM + 1] = atom.formal_charge / 2.0   # normalised charge
    return feat


def feat_to_atom(feat: np.ndarray, bond_row: np.ndarray) -> Atom:
    """Decode a feature vector back to an Atom (argmax decoding)."""
    elem_idx = int(np.argmax(feat[:NUM_ELEM]))
    element  = ELEMENTS[elem_idx]
    bonds    = int(round(np.sum(bond_row)))   # sum of bond orders to neighbours
    charge   = int(round(feat[NUM_ELEM + 1] * 2.0))
    implicit_h = max(0, MAX_VALENCY.get(element, 4) - bonds - charge)
    return Atom(element=element, bonds=bonds, formal_charge=charge,
                implicit_h=implicit_h)


# ---------------------------------------------------------------------------
# Numpy-only GNN (no PyTorch required for basic inference)
# ---------------------------------------------------------------------------

class NumpyLinear:
    """Single linear layer: y = x @ W.T + b (numpy)."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = math.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class NumpyGraphConv:
    """
    One round of mean-aggregation graph convolution:
        h_i = ReLU(W_self · x_i + W_neigh · mean_{j∈N(i)} x_j + b)
    """

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        self.W_self  = NumpyLinear(in_dim, out_dim, rng)
        self.W_neigh = NumpyLinear(in_dim, out_dim, rng)

    def __call__(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        # adj: (N, N) float; x: (N, F)
        deg = adj.sum(axis=1, keepdims=True).clip(min=1)
        agg = (adj @ x) / deg          # mean neighbour features
        return relu(self.W_self(x) + self.W_neigh(agg))


class MolecularDiffusionModel:
    """
    Lightweight denoising model for molecular graphs.

    Uses two graph-conv layers followed by linear heads for:
      - atom feature reconstruction  (atom_head)
      - bond order reconstruction    (bond_head, symmetric)

    Works entirely in NumPy so it runs anywhere (Colab CPU, local, etc.).
    Can be replaced with a proper PyTorch GNN for production use.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.gc1       = NumpyGraphConv(ATOM_FEAT_DIM, hidden_dim, rng)
        self.gc2       = NumpyGraphConv(hidden_dim,    hidden_dim, rng)
        self.atom_head = NumpyLinear(hidden_dim, ATOM_FEAT_DIM, rng)
        self.bond_head = NumpyLinear(hidden_dim * 2,  4, rng)   # 4 bond orders: 0,1,2,3
        self.hidden_dim = hidden_dim
        self._rng = rng

    # ------------------------------------------------------------------
    # Noise schedule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _beta(t: int, T: int) -> float:
        """Cosine-like linear beta schedule: β_t grows from 1e-4 to 0.1 over T steps."""
        return 1e-4 + (t / T) * (0.1 - 1e-4)

    @staticmethod
    def _alpha_bar(t: int, T: int) -> float:
        """Cumulative product of (1 - β_s) for s in 1..t."""
        result = 1.0
        for s in range(1, t + 1):
            result *= 1.0 - MolecularDiffusionModel._beta(s, T)
        return result

    # ------------------------------------------------------------------
    # Forward process (add noise)
    # ------------------------------------------------------------------

    def forward_noisy(
        self,
        x: np.ndarray,
        adj: np.ndarray,
        t: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise at timestep t using the closed-form q(x_t | x_0).
        Returns (x_noisy, adj_noisy).
        """
        alpha_bar = self._alpha_bar(t, T)
        sqrt_ab   = math.sqrt(alpha_bar)
        sqrt_1mab = math.sqrt(1.0 - alpha_bar)

        eps_x   = self._rng.standard_normal(x.shape).astype(np.float32)
        x_noisy = sqrt_ab * x + sqrt_1mab * eps_x

        # Bernoulli noise on adjacency: flip each bond with prob (1 - alpha_bar)
        flip_mask = self._rng.random(adj.shape) < (1.0 - alpha_bar)
        adj_noisy = adj.copy().astype(np.float32)
        adj_noisy[flip_mask] = 1.0 - adj_noisy[flip_mask]
        # Keep symmetric
        adj_noisy = np.triu(adj_noisy, 1)
        adj_noisy = adj_noisy + adj_noisy.T

        return x_noisy, adj_noisy

    # ------------------------------------------------------------------
    # Reverse step (one denoising step)
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t: np.ndarray,
        adj_t: np.ndarray,
        t: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict x_{t-1} and adj_{t-1} from x_t and adj_t.

        The GNN predicts the clean state x_0; we then interpolate back to
        obtain x_{t-1} via the posterior mean formula.
        """
        # ---- GNN forward pass ----------------------------------------
        h1 = self.gc1(x_t, adj_t)             # (N, hidden)
        h2 = self.gc2(h1, adj_t)              # (N, hidden)

        # Atom feature prediction
        x0_pred = self.atom_head(h2)           # (N, FEAT)
        # Soft-max over element logits; keep bond/charge continuous
        x0_pred[:, :NUM_ELEM] = _softmax(x0_pred[:, :NUM_ELEM])

        # Bond prediction: concatenate pair embeddings
        N = x_t.shape[0]
        adj_pred = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                pair = np.concatenate([h2[i], h2[j]])
                logits = self.bond_head(pair)   # (4,) → bond orders 0..3
                bond_order = int(np.argmax(logits))
                adj_pred[i, j] = bond_order
                adj_pred[j, i] = bond_order

        # ---- Posterior mean x_{t-1} ----------------------------------
        ab_t   = self._alpha_bar(t, T)
        ab_tm1 = self._alpha_bar(t - 1, T) if t > 1 else 1.0
        beta_t = self._beta(t, T)

        coef1  = math.sqrt(ab_tm1) * beta_t / (1.0 - ab_t)
        coef2  = math.sqrt(1.0 - beta_t) * (1.0 - ab_tm1) / (1.0 - ab_t)
        x_tm1  = coef1 * x0_pred + coef2 * x_t

        # Add small noise if t > 1
        if t > 1:
            sigma = math.sqrt(beta_t * (1.0 - ab_tm1) / (1.0 - ab_t))
            x_tm1 += sigma * self._rng.standard_normal(x_tm1.shape).astype(np.float32)

        return x_tm1, adj_pred

    # ------------------------------------------------------------------
    # Decode to MolecularState
    # ------------------------------------------------------------------

    def decode(self, x: np.ndarray, adj: np.ndarray, name: str = "product") -> MolecularState:
        """Convert continuous feature matrix + adjacency to a MolecularState."""
        atoms = []
        for i in range(x.shape[0]):
            atom = feat_to_atom(x[i], adj[i])
            atoms.append(atom)
        return MolecularState(name=name, atoms=atoms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Factory: encode a known MolecularState into (x, adj) tensors
# ---------------------------------------------------------------------------

def encode_molecule(mol: MolecularState) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a MolecularState into feature matrix X and adjacency A."""
    N = len(mol.atoms)
    x = np.stack([atom_to_feat(a) for a in mol.atoms])   # (N, F)

    # Build adjacency from .bonds field (heuristic: distribute evenly)
    adj = np.zeros((N, N), dtype=np.float32)
    for i, atom in enumerate(mol.atoms):
        remaining = atom.bonds
        for j in range(N):
            if j != i and remaining > 0:
                bond_order = min(remaining, MAX_VALENCY.get(mol.atoms[j].element, 4))
                adj[i, j] = bond_order
                remaining -= bond_order

    return x, adj
