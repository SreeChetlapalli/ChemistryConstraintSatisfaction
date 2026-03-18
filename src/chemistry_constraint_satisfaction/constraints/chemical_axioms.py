"""
Chemical constraint checks.

The supervisor uses these helpers to verify candidates against a few
simple rules (mass, charge, and bond valency). When `z3` is available we
can run the same checks with a solver; otherwise we fall back to a pure
Python implementation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional Z3 import — graceful fallback for environments without the solver
# ---------------------------------------------------------------------------
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    z3 = None  # type: ignore
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

#: Standard atomic masses (u) for common elements.
ATOMIC_MASS: Dict[str, float] = {
    "H":  1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

#: Maximum covalent valency for each element.
MAX_VALENCY: Dict[str, int] = {
    "H":  1,
    "C":  4,
    "N":  3,   # can be 4 when positively charged (ammonium)
    "O":  2,
    "F":  1,
    "P":  5,
    "S":  6,
    "Cl": 1,
    "Br": 1,
    "I":  1,
}

#: Extra valency allowed when an atom has a formal charge.
#  Example: N with +1 charge can hold 4 bonds; O with -1 keeps valency at 1.
CHARGE_VALENCY_DELTA: Dict[str, Dict[int, int]] = {
    "N": {+1: +1, -1: -1},
    "O": {-1: -1},
    "S": {+1: +1, +2: +2},
    "P": {+1: +1},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Atom:
    """Lightweight atom representation used by the constraint engine."""
    element: str
    bonds: int            # total bond order to other atoms
    formal_charge: int = 0
    implicit_h: int = 0   # implicit hydrogen count

    @property
    def effective_valency(self) -> int:
        base = MAX_VALENCY.get(self.element, 4)
        delta_map = CHARGE_VALENCY_DELTA.get(self.element, {})
        return base + delta_map.get(self.formal_charge, 0)

    @property
    def total_bonds(self) -> int:
        """Bonds to heavy atoms + implicit hydrogens."""
        return self.bonds + self.implicit_h


@dataclasses.dataclass
class MolecularState:
    """Snapshot of a (partial) molecule during diffusion."""
    name: str
    atoms: List[Atom]

    def total_mass(self) -> float:
        mass = 0.0
        for atom in self.atoms:
            mass += ATOMIC_MASS.get(atom.element, 0.0)
            mass += atom.implicit_h * ATOMIC_MASS["H"]
        return mass

    def total_charge(self) -> int:
        return sum(a.formal_charge for a in self.atoms)


@dataclasses.dataclass
class ConstraintResult:
    sat: bool
    violations: List[str] = dataclasses.field(default_factory=list)
    z3_model: Optional[object] = None  # z3.ModelRef when available

    @property
    def reason(self) -> str:
        if self.sat:
            return "All constraints satisfied."
        return "; ".join(self.violations)

    def __bool__(self) -> bool:
        return self.sat


# ---------------------------------------------------------------------------
# Pure-Python fallback checker (no Z3 required)
# ---------------------------------------------------------------------------

def _check_pure_python(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
) -> ConstraintResult:
    """Check constraints without Z3."""
    violations: List[str] = []

    # 1. Mass conservation
    r_mass = sum(m.total_mass() for m in reactants)
    p_mass = sum(m.total_mass() for m in products)
    if abs(r_mass - p_mass) > tolerance:
        violations.append(
            f"Mass not conserved: reactants={r_mass:.3f} u, "
            f"products={p_mass:.3f} u, Δ={abs(r_mass-p_mass):.3f} u"
        )

    # 2. Charge conservation
    r_charge = sum(m.total_charge() for m in reactants)
    p_charge = sum(m.total_charge() for m in products)
    if r_charge != p_charge:
        violations.append(
            f"Charge not conserved: reactants={r_charge:+d}, "
            f"products={p_charge:+d}"
        )

    # 3. Bond valency for each product atom
    for mol in products:
        for atom in mol.atoms:
            if atom.total_bonds > atom.effective_valency:
                violations.append(
                    f"{mol.name}: {atom.element} has {atom.total_bonds} bonds "
                    f"(max {atom.effective_valency})"
                )

    return ConstraintResult(sat=len(violations) == 0, violations=violations)


# ---------------------------------------------------------------------------
# Z3-backed checker
# ---------------------------------------------------------------------------

def _check_z3(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
) -> ConstraintResult:
    """
    Check constraints using Z3.

    This version uses real arithmetic for mass and integer arithmetic for
    charge and valency.
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("z3 is not installed; use _check_pure_python instead.")

    solver = z3.Solver()
    violations: List[str] = []

    # Set up Z3 constraints. We treat the provided values as fixed facts and
    # ask whether they violate the valency bounds.

    # Mass conservation (Real arithmetic)
    r_mass_expr = z3.RealVal(0)
    for mol in reactants:
        for atom in mol.atoms:
            r_mass_expr = r_mass_expr + z3.RealVal(ATOMIC_MASS.get(atom.element, 0.0))
            r_mass_expr = r_mass_expr + z3.RealVal(atom.implicit_h) * z3.RealVal(ATOMIC_MASS["H"])

    p_mass_expr = z3.RealVal(0)
    for mol in products:
        for atom in mol.atoms:
            p_mass_expr = p_mass_expr + z3.RealVal(ATOMIC_MASS.get(atom.element, 0.0))
            p_mass_expr = p_mass_expr + z3.RealVal(atom.implicit_h) * z3.RealVal(ATOMIC_MASS["H"])

    mass_diff = z3.simplify(r_mass_expr - p_mass_expr)
    tol = z3.RealVal(tolerance)
    solver.add(z3.Or(mass_diff > tol, mass_diff < -tol))  # UNSAT means conserved

    result_mass = solver.check()
    solver.reset()
    if result_mass == z3.sat:
        # If the "violation" query is satisfiable, the masses differ too much.
        from fractions import Fraction
        r_val = float(Fraction(str(z3.simplify(r_mass_expr))))
        p_val = float(Fraction(str(z3.simplify(p_mass_expr))))
        violations.append(
            f"Mass not conserved: reactants≈{r_val:.3f} u, products≈{p_val:.3f} u"
        )

    # Charge conservation (Int arithmetic)
    r_chg = sum(sum(a.formal_charge for a in m.atoms) for m in reactants)
    p_chg = sum(sum(a.formal_charge for a in m.atoms) for m in products)
    if r_chg != p_chg:
        violations.append(
            f"Charge not conserved: reactants={r_chg:+d}, products={p_chg:+d}"
        )

    # Valency per atom (Int arithmetic)
    for mol in products:
        for i, atom in enumerate(mol.atoms):
            bonds_var = z3.Int(f"{mol.name}_{atom.element}_{i}_bonds")
            max_val   = z3.IntVal(atom.effective_valency)
            solver.add(bonds_var == z3.IntVal(atom.total_bonds))
            solver.add(bonds_var > max_val)          # ask if valency is exceeded
            if solver.check() == z3.sat:
                violations.append(
                    f"{mol.name}: {atom.element}[{i}] has {atom.total_bonds} bonds "
                    f"(max {atom.effective_valency} for charge {atom.formal_charge:+d})"
                )
            solver.reset()

    return ConstraintResult(sat=len(violations) == 0, violations=violations)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_reaction(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
    prefer_z3: bool = True,
) -> ConstraintResult:
    """
    Verify that a proposed reaction satisfies all chemical axioms.

    Parameters
    ----------
    reactants   : molecules before the reaction
    products    : molecules after the reaction (may be partial / intermediate)
    tolerance   : mass tolerance in atomic mass units (default 0.02 u)
    prefer_z3   : use Z3 when available; fall back to pure-Python otherwise

    Returns
    -------
    ConstraintResult with .sat, .violations, and .reason
    """
    if prefer_z3 and Z3_AVAILABLE:
        return _check_z3(reactants, products, tolerance)
    return _check_pure_python(reactants, products, tolerance)


def check_intermediate(
    molecule: MolecularState,
    tolerance: float = 0.02,
) -> ConstraintResult:
    """
    Check a single intermediate molecule's internal consistency
    (valency only — mass conservation requires reactant context).
    """
    violations: List[str] = []
    for i, atom in enumerate(molecule.atoms):
        if atom.total_bonds > atom.effective_valency:
            violations.append(
                f"{molecule.name}: {atom.element}[{i}] has "
                f"{atom.total_bonds} bonds (max {atom.effective_valency})"
            )
    return ConstraintResult(sat=len(violations) == 0, violations=violations)
