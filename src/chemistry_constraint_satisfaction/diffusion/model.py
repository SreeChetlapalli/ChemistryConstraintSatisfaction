"""
Numpy-only diffusion model for molecule-like graphs.

This module is intentionally small: it provides helpers to encode a
`MolecularState` into fixed-size arrays and to run one reverse diffusion step
without PyTorch.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ..constraints.chemical_axioms import (
    Atom, MolecularState, MAX_VALENCY, CHARGE_VALENCY_DELTA,
)


# ---------------------------------------------------------------------------
# Element encoding
# ---------------------------------------------------------------------------

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
NUM_ELEM = len(ELEMENTS)

# Feature vector per atom:
# - one-hot element
# - bond_count (normalized)
# - formal_charge (normalized)
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
    """Convert a feature vector back into an Atom."""
    elem_idx = int(np.argmax(feat[:NUM_ELEM]))
    element  = ELEMENTS[elem_idx]
    bonds    = int(round(np.sum(bond_row)))
    charge   = int(round(feat[NUM_ELEM + 1] * 2.0))
    base_valency = MAX_VALENCY.get(element, 4)
    delta_map = CHARGE_VALENCY_DELTA.get(element, {})
    eff_valency = base_valency + delta_map.get(charge, 0)
    implicit_h = max(0, eff_valency - bonds)
    return Atom(element=element, bonds=bonds, formal_charge=charge,
                implicit_h=implicit_h)


# ---------------------------------------------------------------------------
# Numpy-only GNN (no PyTorch required for basic inference)
# ---------------------------------------------------------------------------

class NumpyLinear:
    """A single affine layer: y = x @ W.T + b."""

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
    Graph conv using mean aggregation + a ReLU.
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
    Small denoising model used by the supervisor loop.
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
        self._alpha_bar_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Noise schedule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _beta(t: int, T: int) -> float:
        """Linear schedule for beta: grows from 1e-4 to 0.1 over `T` steps."""
        return 1e-4 + (t / T) * (0.1 - 1e-4)

    @staticmethod
    def _alpha_bar(t: int, T: int) -> float:
        """Cumulative product of (1 - beta_s) for s in 1..t."""
        result = 1.0
        for s in range(1, t + 1):
            result *= 1.0 - MolecularDiffusionModel._beta(s, T)
        return result

    def _alpha_bar_cached(self, t: int, T: int) -> float:
        """Read alpha_bar(t, T) from a per-T cache."""
        if T not in self._alpha_bar_cache:
            alpha_vals = np.empty(T + 1, dtype=np.float64)
            alpha_vals[0] = 1.0
            running = 1.0
            for step in range(1, T + 1):
                running *= 1.0 - self._beta(step, T)
                alpha_vals[step] = running
            self._alpha_bar_cache[T] = alpha_vals
        t_clamped = min(max(t, 0), T)
        return float(self._alpha_bar_cache[T][t_clamped])

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
        Add noise to `x` and `adj` at timestep `t`.

        Returns (x_noisy, adj_noisy).
        """
        alpha_bar = self._alpha_bar_cached(t, T)
        sqrt_ab   = math.sqrt(alpha_bar)
        sqrt_1mab = math.sqrt(1.0 - alpha_bar)

        eps_x   = self._rng.standard_normal(x.shape).astype(np.float32)
        x_noisy = sqrt_ab * x + sqrt_1mab * eps_x

        # With probability (1 - alpha_bar), resample bond orders from {0,1,2,3}.
        corrupt_mask = self._rng.random(adj.shape) < (1.0 - alpha_bar)
        adj_noisy = adj.copy().astype(np.float32)
        random_orders = self._rng.integers(0, 4, size=adj.shape).astype(np.float32)
        adj_noisy[corrupt_mask] = random_orders[corrupt_mask]
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
        Predict x_{t-1} and adj_{t-1} from the current x_t/adj_t.
        """
        # GNN forward pass
        h1 = self.gc1(x_t, adj_t)             # (N, hidden)
        h2 = self.gc2(h1, adj_t)              # (N, hidden)

        # Atom feature prediction
        x0_pred = self.atom_head(h2)           # (N, FEAT)
        # Convert element logits to probabilities
        x0_pred[:, :NUM_ELEM] = _softmax(x0_pred[:, :NUM_ELEM])

        # Bond prediction from concatenated pair embeddings
        N = x_t.shape[0]
        adj_pred = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                pair = np.concatenate([h2[i], h2[j]])
                logits = self.bond_head(pair)   # (4,) → bond orders 0..3
                bond_order = int(np.argmax(logits))
                adj_pred[i, j] = bond_order
                adj_pred[j, i] = bond_order

        # Posterior mean (interpolation back toward x_{t-1})
        ab_t   = self._alpha_bar_cached(t, T)
        ab_tm1 = self._alpha_bar_cached(t - 1, T) if t > 1 else 1.0
        beta_t = self._beta(t, T)

        coef1  = math.sqrt(ab_tm1) * beta_t / (1.0 - ab_t)
        coef2  = math.sqrt(1.0 - beta_t) * (1.0 - ab_tm1) / (1.0 - ab_t)
        x_tm1  = coef1 * x0_pred + coef2 * x_t

        # Add extra noise for t > 1
        if t > 1:
            sigma = math.sqrt(beta_t * (1.0 - ab_tm1) / (1.0 - ab_t))
            x_tm1 += sigma * self._rng.standard_normal(x_tm1.shape).astype(np.float32)

        return x_tm1, adj_pred

    # ------------------------------------------------------------------
    # Decode to MolecularState
    # ------------------------------------------------------------------

    def decode(self, x: np.ndarray, adj: np.ndarray, name: str = "product") -> MolecularState:
        """Build a MolecularState from model outputs."""
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
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-8, None)


# ---------------------------------------------------------------------------
# Factory: encode a known MolecularState into (x, adj) tensors
# ---------------------------------------------------------------------------

def encode_molecule(mol: MolecularState) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a MolecularState into (x, adj)."""
    N = len(mol.atoms)
    x = np.stack([atom_to_feat(a) for a in mol.atoms])   # (N, F)

    # Build symmetric adjacency from each atom's bond budget.
    adj = np.zeros((N, N), dtype=np.float32)
    remaining = [a.bonds for a in mol.atoms]
    for i in range(N):
        for j in range(i + 1, N):
            if remaining[i] <= 0 or remaining[j] <= 0:
                continue
            bond_order = min(remaining[i], remaining[j])
            adj[i, j] = bond_order
            adj[j, i] = bond_order
            remaining[i] -= bond_order
            remaining[j] -= bond_order
    return x, adj
