"""
tests/test_diffusion_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the molecular diffusion model (MolecularDiffusionModel).
All tests run on CPU with numpy — no GPU required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import numpy as np
import pytest

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState,
)
from chemistry_constraint_satisfaction.diffusion.model import (
    MolecularDiffusionModel,
    encode_molecule,
    atom_to_feat,
    feat_to_atom,
    ATOM_FEAT_DIM,
    NUM_ELEM,
    ELEMENTS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_mol():
    """CH₄ — methane."""
    return MolecularState("CH4", [
        Atom("C", bonds=4),
        Atom("H", bonds=1),
        Atom("H", bonds=1),
        Atom("H", bonds=1),
        Atom("H", bonds=1),
    ])


@pytest.fixture
def model():
    return MolecularDiffusionModel(hidden_dim=32, seed=0)


# ===========================================================================
# Tests: atom encoding / decoding
# ===========================================================================

class TestEncoding:
    def test_feat_shape(self):
        a = Atom("C", bonds=4)
        f = atom_to_feat(a)
        assert f.shape == (ATOM_FEAT_DIM,)

    def test_element_one_hot(self):
        a = Atom("C", bonds=4)
        f = atom_to_feat(a)
        c_idx = ELEMENTS.index("C")
        assert f[c_idx] == 1.0
        assert f[:NUM_ELEM].sum() == pytest.approx(1.0)

    def test_bond_normalised(self):
        a = Atom("C", bonds=4)
        f = atom_to_feat(a)
        assert f[NUM_ELEM] == pytest.approx(4.0 / 4.0)

    def test_charge_normalised(self):
        a = Atom("N", bonds=3, formal_charge=1)
        f = atom_to_feat(a)
        assert f[NUM_ELEM + 1] == pytest.approx(1.0 / 2.0)

    def test_encode_molecule_shapes(self, simple_mol):
        x, adj = encode_molecule(simple_mol)
        N = len(simple_mol.atoms)
        assert x.shape   == (N, ATOM_FEAT_DIM)
        assert adj.shape == (N, N)

    def test_adjacency_symmetric(self, simple_mol):
        _, adj = encode_molecule(simple_mol)
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_feat_to_atom_roundtrip(self):
        """Encoding then decoding should recover the element."""
        for elem in ["C", "N", "O", "H", "Br"]:
            a    = Atom(elem, bonds=1)
            feat = atom_to_feat(a)
            bond_row = np.zeros(5)
            bond_row[0] = 1
            a2 = feat_to_atom(feat, bond_row)
            assert a2.element == elem


# ===========================================================================
# Tests: noise schedule
# ===========================================================================

class TestNoiseSchedule:
    def test_beta_increases_with_t(self, model):
        betas = [model._beta(t, 100) for t in range(1, 101)]
        assert all(betas[i] <= betas[i+1] for i in range(len(betas)-1))

    def test_alpha_bar_decreases_with_t(self, model):
        abs_vals = [model._alpha_bar(t, 100) for t in range(1, 101)]
        assert all(abs_vals[i] >= abs_vals[i+1] for i in range(len(abs_vals)-1))

    def test_alpha_bar_bounds(self, model):
        # ᾱ should be close to 1 at t=1 and close to 0 at t=T
        assert model._alpha_bar(1, 100) > 0.99
        assert model._alpha_bar(100, 100) < 0.10

    def test_forward_noisy_shape(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=5, T=50)
        assert x_n.shape   == x.shape
        assert adj_n.shape == adj.shape

    def test_forward_noisy_changes_state(self, model, simple_mol):
        x, adj  = encode_molecule(simple_mol)
        x_n, _  = model.forward_noisy(x, adj, t=10, T=50)
        assert not np.allclose(x, x_n)

    def test_forward_noisy_t0_near_clean(self, model, simple_mol):
        """At t=0, no noise is added (ᾱ=1)."""
        x, adj  = encode_molecule(simple_mol)
        # t=0 is not used in practice, test t=1 which is very small noise
        x_n, _  = model.forward_noisy(x, adj, t=1, T=1000)
        # Should be very close to original
        assert np.mean(np.abs(x - x_n)) < 0.05


# ===========================================================================
# Tests: reverse step
# ===========================================================================

class TestReverseStep:
    def test_reverse_step_shapes(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=5, T=50)
        x_prev, adj_prev = model.reverse_step(x_n, adj_n, t=5, T=50)
        assert x_prev.shape   == x.shape
        assert adj_prev.shape == adj.shape

    def test_reverse_step_adjacency_symmetric(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=5, T=50)
        _, adj_prev = model.reverse_step(x_n, adj_n, t=5, T=50)
        np.testing.assert_array_almost_equal(adj_prev, adj_prev.T)

    def test_reverse_step_produces_valid_bond_orders(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=5, T=50)
        _, adj_prev = model.reverse_step(x_n, adj_n, t=5, T=50)
        # Bond orders should be 0, 1, 2, or 3
        assert np.all(adj_prev >= 0)
        assert np.all(adj_prev <= 3)

    def test_reverse_step_t1_no_added_noise(self, model, simple_mol):
        """At t=1, no stochastic noise is added (deterministic step)."""
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=1, T=50)
        x1, _ = model.reverse_step(x_n, adj_n, t=1, T=50)
        x2, _ = model.reverse_step(x_n, adj_n, t=1, T=50)
        # Same input → same output (deterministic at t=1)
        np.testing.assert_array_almost_equal(x1, x2)


# ===========================================================================
# Tests: decode
# ===========================================================================

class TestDecode:
    def test_decode_returns_molecular_state(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        mol = model.decode(x, adj, name="test")
        assert isinstance(mol, MolecularState)
        assert mol.name == "test"
        assert len(mol.atoms) == len(simple_mol.atoms)

    def test_decode_atoms_have_known_elements(self, model, simple_mol):
        x, adj  = encode_molecule(simple_mol)
        decoded = model.decode(x, adj)
        for atom in decoded.atoms:
            assert atom.element in ELEMENTS

    def test_full_forward_then_decode(self, model, simple_mol):
        x, adj = encode_molecule(simple_mol)
        x_n, adj_n = model.forward_noisy(x, adj, t=10, T=50)
        for t in range(10, 0, -1):
            x_n, adj_n = model.reverse_step(x_n, adj_n, t=t, T=50)
        mol = model.decode(x_n, adj_n)
        assert len(mol.atoms) == len(simple_mol.atoms)


# ===========================================================================
# Tests: GNN submodules (light smoke tests)
# ===========================================================================

class TestGNN:
    def test_graph_conv_output_shape(self, model):
        N, F = 5, ATOM_FEAT_DIM
        x   = np.random.randn(N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        h   = model.gc1(x, adj)
        assert h.shape == (N, model.hidden_dim)

    def test_graph_conv_non_negative(self, model):
        N, F = 4, ATOM_FEAT_DIM
        x   = np.random.randn(N, F).astype(np.float32)
        adj = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
        h   = model.gc1(x, adj)
        assert np.all(h >= 0)   # ReLU activation

    def test_atom_head_output_shape(self, model):
        h = np.random.randn(3, model.hidden_dim).astype(np.float32)
        out = model.atom_head(h)
        assert out.shape == (3, ATOM_FEAT_DIM)
