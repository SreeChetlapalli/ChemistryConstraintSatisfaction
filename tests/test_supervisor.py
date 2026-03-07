"""
tests/test_supervisor.py
~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the Digital Supervisor.
Tests the full supervised generation loop, correction strategies,
backtracking logic, and the GenerationResult data class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState,
)
from chemistry_constraint_satisfaction.diffusion.model import MolecularDiffusionModel
from chemistry_constraint_satisfaction.diffusion.supervisor import (
    Supervisor, GenerationResult, StepRecord,
    _fix_valency, _fix_mass,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sn2_reactants():
    """CH₃Br + OH⁻ — canonical SN2 reaction."""
    ch3br = MolecularState("CH3Br", [
        Atom("C", bonds=4), Atom("Br", bonds=1),
        Atom("H", bonds=1), Atom("H", bonds=1), Atom("H", bonds=1),
    ])
    oh_minus = MolecularState("OH-", [
        Atom("O", bonds=1, formal_charge=-1),
        Atom("H", bonds=1),
    ])
    return [ch3br, oh_minus]


@pytest.fixture
def small_model():
    return MolecularDiffusionModel(hidden_dim=16, seed=123)


# ===========================================================================
# Tests: correction helpers
# ===========================================================================

class TestCorrectionHelpers:
    def test_fix_valency_removes_excess(self):
        """Carbon with 5 bonds should be corrected to 4."""
        bad = MolecularState("bad", [Atom("C", bonds=5)])
        fixed = _fix_valency(bad)
        assert fixed.atoms[0].bonds <= 4

    def test_fix_valency_leaves_valid_atoms(self):
        ok = MolecularState("ok", [Atom("C", bonds=4), Atom("O", bonds=2)])
        fixed = _fix_valency(ok)
        assert fixed.atoms[0].bonds == 4
        assert fixed.atoms[1].bonds == 2

    def test_fix_valency_multiple_violations(self):
        bad = MolecularState("bad", [
            Atom("C", bonds=5), Atom("H", bonds=3)
        ])
        fixed = _fix_valency(bad)
        for atom in fixed.atoms:
            assert atom.total_bonds <= atom.effective_valency

    def test_fix_mass_adds_hydrogen(self):
        """If product is too light, add implicit H to bring mass up."""
        light = MolecularState("light", [Atom("C", bonds=0, implicit_h=0)])
        target = light.total_mass() + 4 * 1.008   # need 4 H
        fixed = _fix_mass(light, target_mass=target, tolerance=0.02)
        assert fixed.total_mass() > light.total_mass()

    def test_fix_mass_removes_hydrogen(self):
        """If product is too heavy, reduce implicit H."""
        heavy = MolecularState("heavy", [Atom("C", bonds=0, implicit_h=4)])
        target = heavy.total_mass() - 2 * 1.008
        fixed  = _fix_mass(heavy, target_mass=target, tolerance=0.02)
        assert fixed.total_mass() < heavy.total_mass()

    def test_fix_mass_no_change_within_tolerance(self):
        mol    = MolecularState("ok", [Atom("C", bonds=4)])
        target = mol.total_mass() + 0.01   # within 0.02 tolerance
        fixed  = _fix_mass(mol, target, tolerance=0.02)
        assert fixed.total_mass() == pytest.approx(mol.total_mass())


# ===========================================================================
# Tests: GenerationResult
# ===========================================================================

class TestGenerationResult:
    def _make_result(self, sat=True, backtracks=0, corrections=0):
        from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
            ConstraintResult,
        )
        product = MolecularState("product", [Atom("C", bonds=4)])
        reactants = [MolecularState("R", [Atom("C", bonds=4)])]
        cr = ConstraintResult(sat=sat, violations=[] if sat else ["test violation"])
        return GenerationResult(
            product=product,
            reactants=reactants,
            final_check=cr,
            step_log=[],
            total_backtracks=backtracks,
            total_corrections=corrections,
            wall_time_s=0.01,
        )

    def test_success_when_sat(self):
        r = self._make_result(sat=True)
        assert r.success is True

    def test_failure_when_unsat(self):
        r = self._make_result(sat=False)
        assert r.success is False

    def test_summary_contains_product_name(self):
        r = self._make_result(sat=True)
        assert "product" in r.summary().lower()

    def test_summary_shows_valid(self):
        r = self._make_result(sat=True)
        assert "YES" in r.summary() or "✓" in r.summary()

    def test_summary_shows_invalid(self):
        r = self._make_result(sat=False)
        assert "NO" in r.summary() or "✗" in r.summary()

    def test_summary_shows_backtracks(self):
        r = self._make_result(backtracks=3)
        assert "3" in r.summary()


# ===========================================================================
# Tests: Supervisor integration
# ===========================================================================

class TestSupervisor:
    def test_run_returns_generation_result(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert isinstance(result, GenerationResult)

    def test_product_has_correct_atom_count(self, small_model, sn2_reactants):
        """Product should have same number of atoms as combined reactants."""
        total_atoms = sum(len(m.atoms) for m in sn2_reactants)
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert len(result.product.atoms) == total_atoms

    def test_step_log_non_empty(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert len(result.step_log) > 0

    def test_step_log_actions_are_valid(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        valid_actions = {"commit", "corrected", "backtrack", "skip"}
        for s in result.step_log:
            assert s.action in valid_actions

    def test_wall_time_positive(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert result.wall_time_s > 0

    def test_backtracks_non_negative(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert result.total_backtracks >= 0

    def test_corrections_non_negative(self, small_model, sn2_reactants):
        sup = Supervisor(small_model, sn2_reactants, T=5, verbose=False)
        result = sup.run()
        assert result.total_corrections >= 0

    def test_max_backtracks_respected(self, small_model, sn2_reactants):
        """Supervisor should not exceed max_backtracks."""
        max_bt = 2
        sup = Supervisor(
            small_model, sn2_reactants, T=10,
            max_backtracks=max_bt, verbose=False
        )
        result = sup.run()
        assert result.total_backtracks <= max_bt

    def test_short_trajectory(self, small_model, sn2_reactants):
        """T=2 should complete without error."""
        sup = Supervisor(small_model, sn2_reactants, T=2, verbose=False)
        result = sup.run()
        assert isinstance(result, GenerationResult)

    def test_longer_trajectory(self, sn2_reactants):
        """T=20 with hidden_dim=32 — a more realistic run."""
        model = MolecularDiffusionModel(hidden_dim=32, seed=7)
        sup   = Supervisor(model, sn2_reactants, T=20, verbose=False)
        result = sup.run()
        assert isinstance(result, GenerationResult)

    def test_verbose_runs_without_error(self, small_model, sn2_reactants, capsys):
        sup = Supervisor(small_model, sn2_reactants, T=3, verbose=True)
        result = sup.run()
        captured = capsys.readouterr()
        assert "Summary" in captured.out or "supervisor" in captured.out.lower() or len(captured.out) > 0

    def test_single_reactant(self, small_model):
        """Generation should work with a single reactant molecule."""
        water = MolecularState("H2O", [
            Atom("O", bonds=2),
            Atom("H", bonds=1),
            Atom("H", bonds=1),
        ])
        sup = Supervisor(small_model, [water], T=3, verbose=False)
        result = sup.run()
        assert isinstance(result, GenerationResult)


# ===========================================================================
# Tests: StepRecord
# ===========================================================================

class TestStepRecord:
    def test_fields(self):
        from chemistry_constraint_satisfaction.constraints.chemical_axioms import ConstraintResult
        cr = ConstraintResult(sat=True)
        sr = StepRecord(t=5, attempt=1, constraint_result=cr, action="commit", elapsed_ms=1.2)
        assert sr.t == 5
        assert sr.attempt == 1
        assert sr.action == "commit"
        assert sr.elapsed_ms == pytest.approx(1.2)
