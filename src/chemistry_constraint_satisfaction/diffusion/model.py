"""
Numpy-only diffusion model for molecule-like graphs.

Handles encoding MolecularStates into fixed-size arrays, running one
reverse-diffusion step, and saving/loading trained weights.  No PyTorch
needed here -- see training.py for the trainable version.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, Tuple, Union

import numpy as np

from ..constraints.chemical_axioms import (
    Atom, MolecularState, MAX_VALENCY, CHARGE_VALENCY_DELTA,
)


# ---------------------------------------------------------------------------
# Element encoding
# ---------------------------------------------------------------------------

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Na"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
NUM_ELEM = len(ELEMENTS)

# Feature vector per atom:
# - one-hot element
# - bond_count (normalized)
# - formal_charge (normalized)
ATOM_FEAT_DIM = NUM_ELEM + 2


def atom_to_feat(atom: Atom) -> np.ndarray:
    feat = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    idx = ELEM_TO_IDX.get(atom.element, 1)  # default to C
    feat[idx] = 1.0
    feat[NUM_ELEM]     = atom.bonds / 4.0           # normalised bond count
    feat[NUM_ELEM + 1] = atom.formal_charge / 2.0   # normalised charge
    return feat


def feat_to_atom(feat: np.ndarray, bond_row: np.ndarray) -> Atom:
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
    """y = x @ W.T + b"""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = math.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class NumpyGraphConv:
    """Graph conv: mean-aggregate neighbours, combine with self, ReLU."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        self.W_self  = NumpyLinear(in_dim, out_dim, rng)
        self.W_neigh = NumpyLinear(in_dim, out_dim, rng)

    def __call__(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        # adj: (N, N) float; x: (N, F)
        deg = adj.sum(axis=1, keepdims=True).clip(min=1)
        agg = (adj @ x) / deg          # mean neighbour features
        return relu(self.W_self(x) + self.W_neigh(agg))


class MolecularDiffusionModel:
    """Small GNN denoiser that the supervisor loop calls at each step."""

    def __init__(
        self,
        hidden_dim: int = 64,
        seed: int = 42,
        schedule: str = "linear",
        use_input_proj: bool = False,
    ):
        rng = np.random.default_rng(seed)
        self.use_input_proj = use_input_proj
        if use_input_proj:
            self.input_proj = NumpyLinear(ATOM_FEAT_DIM, hidden_dim, rng)
            self.gc1 = NumpyGraphConv(hidden_dim, hidden_dim, rng)
        else:
            self.input_proj = None
            self.gc1 = NumpyGraphConv(ATOM_FEAT_DIM, hidden_dim, rng)
        self.gc2       = NumpyGraphConv(hidden_dim,    hidden_dim, rng)
        self.atom_head = NumpyLinear(hidden_dim, ATOM_FEAT_DIM, rng)
        self.bond_head = NumpyLinear(hidden_dim * 2,  4, rng)
        self.hidden_dim = hidden_dim
        self.schedule   = schedule
        self._rng = rng
        self._alpha_bar_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Noise schedule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _beta_linear(t: int, T: int) -> float:
        return 1e-4 + (t / T) * (0.1 - 1e-4)

    _beta = _beta_linear  # backward-compatible alias

    @staticmethod
    def _alpha_bar_cosine(t: int, T: int, s: float = 0.008) -> float:
        """Cosine schedule (Nichol & Dhariwal 2021)."""
        f = lambda u: math.cos((u / T + s) / (1 + s) * (math.pi / 2)) ** 2
        return max(f(t) / f(0), 1e-5)

    @staticmethod
    def _alpha_bar_linear(t: int, T: int) -> float:
        result = 1.0
        for step in range(1, t + 1):
            result *= 1.0 - MolecularDiffusionModel._beta_linear(step, T)
        return result

    @staticmethod
    def _alpha_bar(t: int, T: int) -> float:
        return MolecularDiffusionModel._alpha_bar_linear(t, T)

    def _alpha_bar_cached(self, t: int, T: int) -> float:
        key = (T, self.schedule)
        if key not in self._alpha_bar_cache:
            alpha_vals = np.empty(T + 1, dtype=np.float64)
            alpha_vals[0] = 1.0
            if self.schedule == "cosine":
                for step in range(T + 1):
                    alpha_vals[step] = self._alpha_bar_cosine(step, T)
            else:
                running = 1.0
                for step in range(1, T + 1):
                    running *= 1.0 - self._beta_linear(step, T)
                    alpha_vals[step] = running
            self._alpha_bar_cache[key] = alpha_vals
        t_clamped = min(max(t, 0), T)
        return float(self._alpha_bar_cache[key][t_clamped])

    # ------------------------------------------------------------------
    # Forward (noise)
    # ------------------------------------------------------------------

    def forward_noisy(
        self,
        x: np.ndarray,
        adj: np.ndarray,
        t: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise at timestep t -> (x_noisy, adj_noisy)."""
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
    # Reverse step
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t: np.ndarray,
        adj_t: np.ndarray,
        t: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One denoising step: predict (x_{t-1}, adj_{t-1})."""
        inp = self.input_proj(x_t) if self.input_proj is not None else x_t
        h1 = self.gc1(inp, adj_t)
        h2 = self.gc2(h1, adj_t)

        x0_pred = self.atom_head(h2)
        x0_pred[:, :NUM_ELEM] = _softmax(x0_pred[:, :NUM_ELEM])

        # bond prediction from concatenated pair embeddings
        N = x_t.shape[0]
        adj_pred = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                pair = np.concatenate([h2[i], h2[j]])
                logits = self.bond_head(pair)
                bond_order = int(np.argmax(logits))
                adj_pred[i, j] = bond_order
                adj_pred[j, i] = bond_order

        ab_t   = self._alpha_bar_cached(t, T)
        ab_tm1 = self._alpha_bar_cached(t - 1, T) if t > 1 else 1.0
        beta_t = self._beta(t, T)

        coef1  = math.sqrt(ab_tm1) * beta_t / (1.0 - ab_t)
        coef2  = math.sqrt(1.0 - beta_t) * (1.0 - ab_tm1) / (1.0 - ab_t)
        x_tm1  = coef1 * x0_pred + coef2 * x_t

        if t > 1:  # stochastic noise for non-final steps
            sigma = math.sqrt(beta_t * (1.0 - ab_tm1) / (1.0 - ab_t))
            x_tm1 += sigma * self._rng.standard_normal(x_tm1.shape).astype(np.float32)

        return x_tm1, adj_pred

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, x: np.ndarray, adj: np.ndarray, name: str = "product") -> MolecularState:
        atoms = []
        for i in range(x.shape[0]):
            atom = feat_to_atom(x[i], adj[i])
            atoms.append(atom)
        return MolecularState(name=name, atoms=atoms)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        arrays: Dict[str, np.ndarray] = {}
        if self.input_proj is not None:
            arrays["input_proj.W"] = self.input_proj.W
            arrays["input_proj.b"] = self.input_proj.b
        for prefix, layer in [("gc1", self.gc1), ("gc2", self.gc2)]:
            arrays[f"{prefix}.W_self.W"] = layer.W_self.W
            arrays[f"{prefix}.W_self.b"] = layer.W_self.b
            arrays[f"{prefix}.W_neigh.W"] = layer.W_neigh.W
            arrays[f"{prefix}.W_neigh.b"] = layer.W_neigh.b
        arrays["atom_head.W"] = self.atom_head.W
        arrays["atom_head.b"] = self.atom_head.b
        arrays["bond_head.W"] = self.bond_head.W
        arrays["bond_head.b"] = self.bond_head.b
        np.savez(path, **arrays)

        meta = {
            "hidden_dim": self.hidden_dim,
            "schedule": self.schedule,
            "use_input_proj": self.use_input_proj,
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "MolecularDiffusionModel":
        path = pathlib.Path(path)
        meta_path = path.with_suffix(".json")
        meta: Dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        model = cls(
            hidden_dim=meta.get("hidden_dim", 64),
            seed=0,
            schedule=meta.get("schedule", "linear"),
            use_input_proj=meta.get("use_input_proj", False),
        )
        data = np.load(path)
        if model.input_proj is not None and "input_proj.W" in data:
            model.input_proj.W = data["input_proj.W"]
            model.input_proj.b = data["input_proj.b"]
        for prefix, layer in [("gc1", model.gc1), ("gc2", model.gc2)]:
            layer.W_self.W = data[f"{prefix}.W_self.W"]
            layer.W_self.b = data[f"{prefix}.W_self.b"]
            layer.W_neigh.W = data[f"{prefix}.W_neigh.W"]
            layer.W_neigh.b = data[f"{prefix}.W_neigh.b"]
        model.atom_head.W = data["atom_head.W"]
        model.atom_head.b = data["atom_head.b"]
        model.bond_head.W = data["bond_head.W"]
        model.bond_head.b = data["bond_head.b"]
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-8, None)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_molecule(mol: MolecularState) -> Tuple[np.ndarray, np.ndarray]:
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
