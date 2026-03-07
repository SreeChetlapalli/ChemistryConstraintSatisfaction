"""
run_tests.py
~~~~~~~~~~~~
Self-contained test runner — uses only stdlib (no pytest required).
Run with:  python run_tests.py
Or with pytest when available:  pytest tests/
"""

import sys
import os
import math
import unittest
import numpy as np

# Make the src package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState, ConstraintResult,
    check_reaction, check_intermediate,
    _check_pure_python,
    ATOMIC_MASS, MAX_VALENCY,
)
from chemistry_constraint_satisfaction.diffusion.model import (
    MolecularDiffusionModel,
    encode_molecule,
    atom_to_feat, feat_to_atom,
    ATOM_FEAT_DIM, NUM_ELEM, ELEMENTS,
)
from chemistry_constraint_satisfaction.diffusion.supervisor import (
    Supervisor, GenerationResult, StepRecord,
    _fix_valency, _fix_mass,
)

try:
    import z3 as _z3_mod
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_ch3br():
    return MolecularState("CH3Br", [
        Atom("C",  bonds=4), Atom("Br", bonds=1),
        Atom("H",  bonds=1), Atom("H", bonds=1), Atom("H", bonds=1),
    ])

def make_oh_minus():
    return MolecularState("OH-", [
        Atom("O", bonds=1, formal_charge=-1),
        Atom("H", bonds=1),
    ])

def make_ch3oh():
    return MolecularState("CH3OH", [
        Atom("C", bonds=4), Atom("O", bonds=2),
        Atom("H", bonds=1), Atom("H", bonds=1),
        Atom("H", bonds=1), Atom("H", bonds=1),
    ])

def make_br_minus():
    return MolecularState("Br-", [Atom("Br", bonds=0, formal_charge=-1)])


# ===========================================================================
# 1. Chemical axioms
# ===========================================================================

class TestAtom(unittest.TestCase):
    def test_effective_valency_carbon(self):
        self.assertEqual(Atom("C", bonds=4).effective_valency, 4)

    def test_effective_valency_nitrogen_positive(self):
        self.assertEqual(Atom("N", bonds=4, formal_charge=+1).effective_valency, 4)

    def test_effective_valency_oxygen_negative(self):
        self.assertEqual(Atom("O", bonds=1, formal_charge=-1).effective_valency, 1)

    def test_total_bonds_includes_implicit_h(self):
        self.assertEqual(Atom("C", bonds=2, implicit_h=2).total_bonds, 4)


class TestMolecularState(unittest.TestCase):
    def test_total_mass_ch3br(self):
        mol      = make_ch3br()
        expected = ATOMIC_MASS["C"] + ATOMIC_MASS["Br"] + 3 * ATOMIC_MASS["H"]
        self.assertAlmostEqual(mol.total_mass(), expected, places=2)

    def test_total_charge_neutral(self):
        self.assertEqual(make_ch3br().total_charge(), 0)

    def test_total_charge_negative(self):
        self.assertEqual(make_oh_minus().total_charge(), -1)


class TestCheckIntermediate(unittest.TestCase):
    def test_valid_carbon(self):
        self.assertTrue(check_intermediate(MolecularState("ok", [Atom("C", 4)])).sat)

    def test_overvalenced_carbon(self):
        cr = check_intermediate(MolecularState("bad", [Atom("C", 5)]))
        self.assertFalse(cr.sat)
        self.assertTrue(any("C" in v for v in cr.violations))

    def test_valid_water(self):
        water = MolecularState("H2O", [Atom("O",2), Atom("H",1), Atom("H",1)])
        self.assertTrue(check_intermediate(water).sat)

    def test_overvalenced_hydrogen(self):
        self.assertFalse(check_intermediate(MolecularState("bad", [Atom("H", 2)])).sat)

    def test_nitrogen_ammonium(self):
        nh4 = MolecularState("NH4+", [Atom("N", bonds=4, formal_charge=+1)])
        self.assertTrue(check_intermediate(nh4).sat)

    def test_implicit_h_counts(self):
        bad = MolecularState("bad", [Atom("C", bonds=4, implicit_h=1)])
        self.assertFalse(check_intermediate(bad).sat)

    def test_sulphur_6_bonds(self):
        self.assertTrue(check_intermediate(MolecularState("S6", [Atom("S", 6)])).sat)

    def test_phosphorus_5_bonds(self):
        self.assertTrue(check_intermediate(MolecularState("P5", [Atom("P", 5)])).sat)

    def test_bromine_max_1(self):
        self.assertTrue(check_intermediate(MolecularState("Br1", [Atom("Br", 1)])).sat)
        self.assertFalse(check_intermediate(MolecularState("Br2", [Atom("Br", 2)])).sat)


class TestCheckReaction(unittest.TestCase):
    def _check(self, r, p, **kw):
        return _check_pure_python(r, p, **kw)

    def test_sn2_valid(self):
        cr = self._check([make_ch3br(), make_oh_minus()], [make_ch3oh(), make_br_minus()])
        self.assertTrue(cr.sat, cr.reason)

    def test_mass_not_conserved(self):
        cr = self._check([make_ch3br(), make_oh_minus()], [make_ch3oh()])
        self.assertFalse(cr.sat)
        self.assertTrue(any("Mass" in v or "mass" in v for v in cr.violations))

    def test_charge_not_conserved(self):
        oh_neutral = MolecularState("OH", [Atom("O",1), Atom("H",1)])
        cr = self._check([make_ch3br(), oh_neutral], [make_ch3oh(), make_br_minus()])
        self.assertFalse(cr.sat)
        self.assertTrue(any("Charge" in v or "charge" in v for v in cr.violations))

    def test_valency_violation_product(self):
        bad = MolecularState("bad", [
            Atom("C", bonds=5), Atom("O", bonds=2), Atom("Br", bonds=0, formal_charge=-1),
            Atom("H",1), Atom("H",1), Atom("H",1), Atom("H",1),
        ])
        cr = self._check([make_ch3br(), make_oh_minus()], [bad])
        self.assertFalse(cr.sat)

    def test_empty_products(self):
        cr = self._check([make_ch3br()], [])
        self.assertFalse(cr.sat)

    def test_result_bool(self):
        self.assertTrue(bool(ConstraintResult(sat=True)))
        self.assertFalse(bool(ConstraintResult(sat=False, violations=["x"])))

    def test_result_reason_sat(self):
        self.assertIn("satisfied", ConstraintResult(sat=True).reason.lower())

    def test_result_reason_unsat(self):
        cr = ConstraintResult(sat=False, violations=["Mass not conserved"])
        self.assertIn("Mass", cr.reason)


# ===========================================================================
# 2. Diffusion model
# ===========================================================================

def make_simple_mol():
    return MolecularState("CH4", [
        Atom("C",4), Atom("H",1), Atom("H",1), Atom("H",1), Atom("H",1),
    ])


class TestEncoding(unittest.TestCase):
    def test_feat_shape(self):
        self.assertEqual(atom_to_feat(Atom("C", 4)).shape, (ATOM_FEAT_DIM,))

    def test_element_one_hot(self):
        f = atom_to_feat(Atom("C", 4))
        c_idx = ELEMENTS.index("C")
        self.assertAlmostEqual(f[c_idx], 1.0)
        self.assertAlmostEqual(float(f[:NUM_ELEM].sum()), 1.0)

    def test_bond_normalised(self):
        f = atom_to_feat(Atom("C", 4))
        self.assertAlmostEqual(float(f[NUM_ELEM]), 1.0)

    def test_encode_molecule_shapes(self):
        mol = make_simple_mol()
        x, adj = encode_molecule(mol)
        N = len(mol.atoms)
        self.assertEqual(x.shape,   (N, ATOM_FEAT_DIM))
        self.assertEqual(adj.shape, (N, N))

    def test_adjacency_symmetric(self):
        _, adj = encode_molecule(make_simple_mol())
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_feat_to_atom_roundtrip(self):
        for elem in ["C", "N", "O", "H", "Br"]:
            feat = atom_to_feat(Atom(elem, bonds=1))
            bond_row = np.zeros(5); bond_row[0] = 1
            a2 = feat_to_atom(feat, bond_row)
            self.assertEqual(a2.element, elem)


class TestNoiseSchedule(unittest.TestCase):
    def setUp(self):
        self.model = MolecularDiffusionModel(hidden_dim=16, seed=0)

    def test_beta_increases_with_t(self):
        betas = [self.model._beta(t, 100) for t in range(1, 101)]
        self.assertTrue(all(betas[i] <= betas[i+1] for i in range(len(betas)-1)))

    def test_alpha_bar_decreases(self):
        abs_vals = [self.model._alpha_bar(t, 100) for t in range(1, 101)]
        self.assertTrue(all(abs_vals[i] >= abs_vals[i+1] for i in range(len(abs_vals)-1)))

    def test_alpha_bar_bounds(self):
        self.assertGreater(self.model._alpha_bar(1,   100), 0.99)
        self.assertLess   (self.model._alpha_bar(100, 100), 0.10)

    def test_forward_noisy_shapes(self):
        mol = make_simple_mol()
        x, adj = encode_molecule(mol)
        x_n, adj_n = self.model.forward_noisy(x, adj, t=5, T=50)
        self.assertEqual(x_n.shape,   x.shape)
        self.assertEqual(adj_n.shape, adj.shape)

    def test_forward_noisy_changes_state(self):
        x, adj = encode_molecule(make_simple_mol())
        x_n, _ = self.model.forward_noisy(x, adj, t=10, T=50)
        self.assertFalse(np.allclose(x, x_n))


class TestReverseStep(unittest.TestCase):
    def setUp(self):
        self.model = MolecularDiffusionModel(hidden_dim=16, seed=0)
        mol = make_simple_mol()
        self.x, self.adj = encode_molecule(mol)
        self.x_n, self.adj_n = self.model.forward_noisy(self.x, self.adj, t=5, T=50)

    def test_reverse_step_shapes(self):
        x_p, adj_p = self.model.reverse_step(self.x_n, self.adj_n, t=5, T=50)
        self.assertEqual(x_p.shape,   self.x.shape)
        self.assertEqual(adj_p.shape, self.adj.shape)

    def test_reverse_step_adj_symmetric(self):
        _, adj_p = self.model.reverse_step(self.x_n, self.adj_n, t=5, T=50)
        np.testing.assert_array_almost_equal(adj_p, adj_p.T)

    def test_bond_orders_in_range(self):
        _, adj_p = self.model.reverse_step(self.x_n, self.adj_n, t=5, T=50)
        self.assertTrue(np.all(adj_p >= 0))
        self.assertTrue(np.all(adj_p <= 3))

    def test_t1_deterministic(self):
        """At t=1 no noise is added → same x_n gives same output."""
        x_n, adj_n = self.model.forward_noisy(self.x, self.adj, t=1, T=50)
        x1, _ = self.model.reverse_step(x_n, adj_n, t=1, T=50)
        x2, _ = self.model.reverse_step(x_n, adj_n, t=1, T=50)
        np.testing.assert_array_almost_equal(x1, x2)

    def test_decode_returns_correct_atom_count(self):
        x_p, adj_p = self.model.reverse_step(self.x_n, self.adj_n, t=5, T=50)
        mol = self.model.decode(x_p, adj_p)
        self.assertEqual(len(mol.atoms), self.x.shape[0])

    def test_gnn_output_nonnegative(self):
        """Graph conv uses ReLU → hidden features >= 0."""
        x, adj = encode_molecule(make_simple_mol())
        h = self.model.gc1(x, adj)
        self.assertTrue(np.all(h >= 0))


# ===========================================================================
# 3. Supervisor
# ===========================================================================

class TestCorrectionHelpers(unittest.TestCase):
    def test_fix_valency_reduces_excess(self):
        bad   = MolecularState("bad", [Atom("C", bonds=5)])
        fixed = _fix_valency(bad)
        self.assertLessEqual(fixed.atoms[0].bonds, 4)

    def test_fix_valency_preserves_valid(self):
        ok    = MolecularState("ok", [Atom("C", bonds=4), Atom("O", bonds=2)])
        fixed = _fix_valency(ok)
        self.assertEqual(fixed.atoms[0].bonds, 4)
        self.assertEqual(fixed.atoms[1].bonds, 2)

    def test_fix_mass_adds_hydrogen(self):
        light  = MolecularState("light", [Atom("C", bonds=0, implicit_h=0)])
        target = light.total_mass() + 4 * 1.008
        fixed  = _fix_mass(light, target, tolerance=0.02)
        self.assertGreater(fixed.total_mass(), light.total_mass())

    def test_fix_mass_no_change_within_tolerance(self):
        mol    = MolecularState("ok", [Atom("C", bonds=4)])
        target = mol.total_mass() + 0.01
        fixed  = _fix_mass(mol, target, tolerance=0.02)
        self.assertAlmostEqual(fixed.total_mass(), mol.total_mass(), places=3)


class TestSupervisor(unittest.TestCase):
    def _reactants(self):
        return [make_ch3br(), make_oh_minus()]

    def _model(self):
        return MolecularDiffusionModel(hidden_dim=16, seed=123)

    def test_run_returns_result(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        self.assertIsInstance(result, GenerationResult)

    def test_product_atom_count(self):
        reactants = self._reactants()
        total = sum(len(m.atoms) for m in reactants)
        result = Supervisor(self._model(), reactants, T=5).run()
        self.assertEqual(len(result.product.atoms), total)

    def test_step_log_non_empty(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        self.assertGreater(len(result.step_log), 0)

    def test_step_log_valid_actions(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        valid  = {"commit", "corrected", "backtrack", "skip"}
        for s in result.step_log:
            self.assertIn(s.action, valid)

    def test_wall_time_positive(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        self.assertGreater(result.wall_time_s, 0)

    def test_max_backtracks_respected(self):
        result = Supervisor(self._model(), self._reactants(), T=10, max_backtracks=2).run()
        self.assertLessEqual(result.total_backtracks, 2)

    def test_summary_contains_product(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        self.assertIn("product", result.summary().lower())

    def test_short_trajectory(self):
        result = Supervisor(self._model(), self._reactants(), T=2).run()
        self.assertIsInstance(result, GenerationResult)

    def test_single_reactant(self):
        water = MolecularState("H2O", [Atom("O",2), Atom("H",1), Atom("H",1)])
        result = Supervisor(self._model(), [water], T=3).run()
        self.assertIsInstance(result, GenerationResult)

    def test_corrections_non_negative(self):
        result = Supervisor(self._model(), self._reactants(), T=5).run()
        self.assertGreaterEqual(result.total_corrections, 0)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestAtom, TestMolecularState,
        TestCheckIntermediate, TestCheckReaction,
        TestEncoding, TestNoiseSchedule, TestReverseStep,
        TestCorrectionHelpers, TestSupervisor,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
