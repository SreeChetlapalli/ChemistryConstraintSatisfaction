"""
tests/test_chemical_axioms.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the Z3-backed chemical axiom checker.
Covers mass conservation, charge conservation, and bond valency.
Tests pass with or without Z3 installed (pure-Python fallback is also tested).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState, ConstraintResult,
    check_reaction, check_intermediate,
    _check_pure_python,
    ATOMIC_MASS, MAX_VALENCY,
)


# ===========================================================================
# Fixtures — canonical reaction: CH₃Br + OH⁻ → CH₃OH + Br⁻
# ===========================================================================

def make_ch3br() -> MolecularState:
    """Bromomethane: C bonded to 3 H and 1 Br (4 bonds total)."""
    return MolecularState(name="CH3Br", atoms=[
        Atom(element="C",  bonds=4, formal_charge=0, implicit_h=0),
        Atom(element="Br", bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H",  bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H",  bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H",  bonds=1, formal_charge=0, implicit_h=0),
    ])


def make_oh_minus() -> MolecularState:
    """Hydroxide ion: O with 1 bond, charge −1."""
    return MolecularState(name="OH-", atoms=[
        Atom(element="O", bonds=1, formal_charge=-1, implicit_h=0),
        Atom(element="H", bonds=1, formal_charge=0,  implicit_h=0),
    ])


def make_ch3oh() -> MolecularState:
    """Methanol: C bonded to 3 H + 1 O; O bonded to 1 C + 1 H."""
    return MolecularState(name="CH3OH", atoms=[
        Atom(element="C", bonds=4, formal_charge=0, implicit_h=0),
        Atom(element="O", bonds=2, formal_charge=0, implicit_h=0),
        Atom(element="H", bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H", bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H", bonds=1, formal_charge=0, implicit_h=0),
        Atom(element="H", bonds=1, formal_charge=0, implicit_h=0),
    ])


def make_br_minus() -> MolecularState:
    """Bromide ion: Br with no bonds, charge −1."""
    return MolecularState(name="Br-", atoms=[
        Atom(element="Br", bonds=0, formal_charge=-1, implicit_h=0),
    ])


# ===========================================================================
# Tests: Atom helpers
# ===========================================================================

class TestAtom:
    def test_effective_valency_default(self):
        assert Atom("C", bonds=4).effective_valency == 4

    def test_effective_valency_nitrogen_neutral(self):
        assert Atom("N", bonds=3).effective_valency == 3

    def test_effective_valency_nitrogen_positive(self):
        assert Atom("N", bonds=4, formal_charge=+1).effective_valency == 4

    def test_effective_valency_oxygen_negative(self):
        assert Atom("O", bonds=1, formal_charge=-1).effective_valency == 1

    def test_total_bonds_includes_implicit_h(self):
        a = Atom("C", bonds=2, implicit_h=2)
        assert a.total_bonds == 4

    def test_total_bonds_no_implicit(self):
        assert Atom("Br", bonds=1).total_bonds == 1


class TestMolecularState:
    def test_total_mass_ch3br(self):
        mol = make_ch3br()
        expected = ATOMIC_MASS["C"] + ATOMIC_MASS["Br"] + 3 * ATOMIC_MASS["H"]
        assert abs(mol.total_mass() - expected) < 1e-3

    def test_total_charge_neutral(self):
        assert make_ch3br().total_charge() == 0

    def test_total_charge_negative(self):
        assert make_oh_minus().total_charge() == -1

    def test_total_charge_sum(self):
        # CH3Br (0) + OH- (-1) = -1
        reactant_charge = make_ch3br().total_charge() + make_oh_minus().total_charge()
        assert reactant_charge == -1


# ===========================================================================
# Tests: check_intermediate (valency only)
# ===========================================================================

class TestCheckIntermediate:
    def test_valid_carbon(self):
        mol = MolecularState("ok", [Atom("C", bonds=4)])
        assert check_intermediate(mol).sat

    def test_overvalenced_carbon(self):
        mol = MolecularState("bad", [Atom("C", bonds=5)])
        cr = check_intermediate(mol)
        assert not cr.sat
        assert any("C" in v for v in cr.violations)

    def test_valid_water(self):
        water = MolecularState("H2O", [
            Atom("O", bonds=2),
            Atom("H", bonds=1),
            Atom("H", bonds=1),
        ])
        assert check_intermediate(water).sat

    def test_overvalenced_hydrogen(self):
        bad_h = MolecularState("bad_H", [Atom("H", bonds=2)])
        assert not check_intermediate(bad_h).sat

    def test_nitrogen_with_charge(self):
        # NH4+ has 4 bonds — allowed because charge is +1
        nh4 = MolecularState("NH4+", [Atom("N", bonds=4, formal_charge=+1)])
        assert check_intermediate(nh4).sat

    def test_implicit_h_counts_toward_valency(self):
        # C with 4 bonds + 1 implicit H = 5 total → overvalenced
        bad = MolecularState("bad_C", [Atom("C", bonds=4, implicit_h=1)])
        assert not check_intermediate(bad).sat


# ===========================================================================
# Tests: check_reaction — pure-Python backend
# ===========================================================================

class TestCheckReactionPurePython:
    """These tests always use the pure-Python backend."""

    def _check(self, reactants, products, **kw):
        return _check_pure_python(reactants, products, **kw)

    def test_sn2_valid(self):
        """CH₃Br + OH⁻ → CH₃OH + Br⁻  should be fully valid."""
        cr = self._check(
            [make_ch3br(), make_oh_minus()],
            [make_ch3oh(), make_br_minus()],
        )
        assert cr.sat, cr.reason

    def test_mass_not_conserved(self):
        """Remove Br⁻ from products — mass drops by ~80 u."""
        cr = self._check(
            [make_ch3br(), make_oh_minus()],
            [make_ch3oh()],          # missing Br-
        )
        assert not cr.sat
        assert any("Mass" in v or "mass" in v for v in cr.violations)

    def test_charge_not_conserved(self):
        """Replace OH⁻ with neutral OH — charge is no longer conserved."""
        oh_neutral = MolecularState("OH", atoms=[
            Atom("O", bonds=1, formal_charge=0),
            Atom("H", bonds=1),
        ])
        cr = self._check(
            [make_ch3br(), oh_neutral],
            [make_ch3oh(), make_br_minus()],
        )
        assert not cr.sat
        assert any("Charge" in v or "charge" in v for v in cr.violations)

    def test_valency_violation_in_product(self):
        """Give carbon 5 bonds in the product."""
        bad_product = MolecularState("bad", atoms=[
            Atom("C", bonds=5),  # illegal
            Atom("O", bonds=2),
            Atom("Br", bonds=0, formal_charge=-1),
            Atom("H", bonds=1), Atom("H", bonds=1),
            Atom("H", bonds=1), Atom("H", bonds=1),
        ])
        cr = self._check(
            [make_ch3br(), make_oh_minus()],
            [bad_product],
        )
        assert not cr.sat
        assert any("bonds" in v for v in cr.violations)

    def test_tolerance_respected(self):
        """A tiny mass difference (< tolerance) should still pass."""
        cr = self._check(
            [make_ch3br(), make_oh_minus()],
            [make_ch3oh(), make_br_minus()],
            tolerance=10.0,   # very generous
        )
        assert cr.sat

    def test_result_bool(self):
        cr_valid = ConstraintResult(sat=True)
        cr_bad   = ConstraintResult(sat=False, violations=["oops"])
        assert bool(cr_valid) is True
        assert bool(cr_bad)   is False

    def test_result_reason_on_sat(self):
        cr = ConstraintResult(sat=True)
        assert "satisfied" in cr.reason.lower()

    def test_result_reason_on_unsat(self):
        cr = ConstraintResult(sat=False, violations=["Mass not conserved"])
        assert "Mass" in cr.reason


# ===========================================================================
# Tests: check_reaction — Z3 backend (skipped if not installed)
# ===========================================================================

try:
    import z3 as _z3_mod
    _z3_imported = True
except ImportError:
    _z3_imported = False

@pytest.mark.skipif(not _z3_imported, reason="z3-solver not installed")
class TestCheckReactionZ3:
    """Run the same logical checks but through the Z3 solver."""

    def test_sn2_valid_z3(self):
        cr = check_reaction(
            [make_ch3br(), make_oh_minus()],
            [make_ch3oh(), make_br_minus()],
            prefer_z3=True,
        )
        assert cr.sat, cr.reason

    def test_mass_violation_z3(self):
        cr = check_reaction(
            [make_ch3br(), make_oh_minus()],
            [make_ch3oh()],
            prefer_z3=True,
        )
        assert not cr.sat

    def test_valency_violation_z3(self):
        bad = MolecularState("bad", atoms=[Atom("C", bonds=5)])
        cr  = check_reaction(
            [make_ch3br(), make_oh_minus()],
            [bad],
            prefer_z3=True,
        )
        assert not cr.sat


# ===========================================================================
# Tests: edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_products(self):
        """No products — mass clearly not conserved."""
        cr = _check_pure_python([make_ch3br()], [])
        assert not cr.sat

    def test_single_atom_self_consistent(self):
        """Single neutral hydrogen atom — valid in isolation."""
        h = MolecularState("H", [Atom("H", bonds=0)])
        assert check_intermediate(h).sat

    def test_bromine_max_valency(self):
        """Br can have at most 1 bond (typical)."""
        br_ok  = MolecularState("Br", [Atom("Br", bonds=1)])
        br_bad = MolecularState("Br", [Atom("Br", bonds=2)])
        assert check_intermediate(br_ok).sat
        assert not check_intermediate(br_bad).sat

    def test_sulphur_high_valency(self):
        """S can have up to 6 bonds."""
        s6 = MolecularState("S", [Atom("S", bonds=6)])
        assert check_intermediate(s6).sat

    def test_phosphorus_five_bonds(self):
        """P can have 5 bonds (e.g., PCl5)."""
        p5 = MolecularState("P", [Atom("P", bonds=5)])
        assert check_intermediate(p5).sat
