"""
Supervisor loop for the diffusion model.

At each reverse step it decodes the candidate into a `MolecularState`, checks
the chemistry constraints, and either commits the step, tries a small fix,
or backtracks and re-samples.
"""

from __future__ import annotations

import dataclasses
import time
from typing import List, Tuple

import numpy as np

from ..constraints.chemical_axioms import (
    MolecularState, ConstraintResult, check_intermediate, check_reaction,
)
from .model import MolecularDiffusionModel, encode_molecule


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepRecord:
    """One entry in the per-step log."""
    t: int
    attempt: int
    constraint_result: ConstraintResult
    action: str   # "commit", "corrected", "backtrack", "skip"
    elapsed_ms: float


@dataclasses.dataclass
class IntermediateSnapshot:
    """Decoded molecule state at a committed diffusion step."""
    t: int
    molecule: MolecularState
    adjacency: list


@dataclasses.dataclass
class GenerationResult:
    """Result returned by `Supervisor.run()`."""
    product: MolecularState
    reactants: List[MolecularState]
    final_check: ConstraintResult
    step_log: List[StepRecord]
    total_backtracks: int
    total_corrections: int
    wall_time_s: float
    intermediates: List[IntermediateSnapshot] = dataclasses.field(default_factory=list)
    product_adjacency: list = dataclasses.field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.final_check.sat

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Supervisor - Generation Summary",
            "=" * 60,
            f"  Product     : {self.product.name}",
            f"  Atoms       : {len(self.product.atoms)}",
            f"  Valid       : {'YES' if self.success else 'NO'}",
            f"  Backtracks  : {self.total_backtracks}",
            f"  Corrections : {self.total_corrections}",
            f"  Wall time   : {self.wall_time_s:.3f}s",
            "",
            "  Constraint check:",
        ]
        if self.success:
            lines.append("    [OK] All axioms satisfied.")
        else:
            for v in self.final_check.violations:
                lines.append(f"    [FAIL] {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Correction strategies
# ---------------------------------------------------------------------------

def _fix_valency(mol: MolecularState) -> MolecularState:
    """
    Reduce bonds on atoms that exceed their allowed valency.
    """
    fixed_atoms = []
    for atom in mol.atoms:
        if atom.total_bonds > atom.effective_valency:
            excess = atom.total_bonds - atom.effective_valency
            new_bonds  = max(0, atom.bonds - excess)
            new_impl_h = max(0, atom.implicit_h - max(0, excess - atom.bonds))
            fixed_atoms.append(dataclasses.replace(atom, bonds=new_bonds, implicit_h=new_impl_h))
        else:
            fixed_atoms.append(atom)
    return MolecularState(atoms=fixed_atoms, name=mol.name)


def _fix_composition(
    mol: MolecularState,
    target_elements: List[str],
) -> MolecularState:
    """
    Relabel decoded atoms so the element multiset matches the reactants.

    The diffusion model keeps atom count N fixed but can decode the wrong
    elements. This greedily assigns target elements to atoms, preferring
    to keep atoms that already match.
    """
    from ..constraints.chemical_axioms import MAX_VALENCY, CHARGE_VALENCY_DELTA

    if len(mol.atoms) != len(target_elements):
        return mol

    remaining = list(target_elements)

    # First pass: mark atoms that already have a matching target element
    matched = [False] * len(mol.atoms)
    for i, atom in enumerate(mol.atoms):
        if atom.element in remaining:
            remaining.remove(atom.element)
            matched[i] = True

    # Second pass: relabel unmatched atoms with leftover target elements
    fixed = []
    for i, atom in enumerate(mol.atoms):
        if matched[i]:
            fixed.append(atom)
        elif remaining:
            new_elem = remaining.pop(0)
            base_val = MAX_VALENCY.get(new_elem, 4)
            delta_map = CHARGE_VALENCY_DELTA.get(new_elem, {})
            eff_val = base_val + delta_map.get(atom.formal_charge, 0)
            new_bonds = min(atom.bonds, eff_val)
            new_impl_h = max(0, eff_val - new_bonds)
            fixed.append(dataclasses.replace(
                atom, element=new_elem, bonds=new_bonds,
                implicit_h=new_impl_h,
            ))
        else:
            fixed.append(atom)

    return MolecularState(atoms=fixed, name=mol.name)


def _fix_charge(
    mol: MolecularState,
    target_charge: int,
) -> MolecularState:
    """
    Adjust formal charges so total charge matches the target.
    Distributes corrections across atoms that commonly carry charge
    (N, O, S, P, halogens), zeroing out spurious charges first.
    """
    current = mol.total_charge()
    delta = target_charge - current
    if delta == 0:
        return mol

    from ..constraints.chemical_axioms import MAX_VALENCY, CHARGE_VALENCY_DELTA
    atoms = [dataclasses.replace(a) for a in mol.atoms]

    # First: zero out any charges that don't belong on their element
    chargeable = {"N", "O", "S", "P", "Cl", "Br", "I", "F"}
    for i, atom in enumerate(atoms):
        if delta == 0:
            break
        if atom.formal_charge != 0 and atom.element not in chargeable:
            old_charge = atom.formal_charge
            atoms[i] = dataclasses.replace(atom, formal_charge=0)
            delta += old_charge

    # Second: distribute remaining delta across chargeable atoms
    for i, atom in enumerate(atoms):
        if delta == 0:
            break
        if atom.element in chargeable:
            step = 1 if delta > 0 else -1
            new_charge = atom.formal_charge + step
            if abs(new_charge) <= 2:
                atoms[i] = dataclasses.replace(atom, formal_charge=new_charge)
                delta -= step

    # Last resort: put remaining delta on any atom
    for i, atom in enumerate(atoms):
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        new_charge = atom.formal_charge + step
        if abs(new_charge) <= 3:
            atoms[i] = dataclasses.replace(atom, formal_charge=new_charge)
            delta -= step

    return MolecularState(atoms=atoms, name=mol.name)


def _fix_mass(
    mol: MolecularState,
    target_mass: float,
    tolerance: float = 0.02,
) -> MolecularState:
    """
    Adjust implicit hydrogen so total mass moves toward the target.
    Uses multiple passes to converge.
    """
    from ..constraints.chemical_axioms import ATOMIC_MASS
    h_mass = ATOMIC_MASS["H"]

    atoms = [dataclasses.replace(a) for a in mol.atoms]

    for _pass in range(3):
        current = sum(ATOMIC_MASS.get(a.element, 0) + a.implicit_h * h_mass for a in atoms)
        delta = target_mass - current
        if abs(delta) <= tolerance:
            break
        for atom in atoms:
            if abs(delta) <= tolerance:
                break
            if delta > 0:
                space = atom.effective_valency - atom.total_bonds
                add = min(round(delta / h_mass + 0.49), space)
                if add > 0:
                    atom.implicit_h += add
                    delta -= add * h_mass
            else:
                remove = min(round(-delta / h_mass + 0.49), atom.implicit_h)
                if remove > 0:
                    atom.implicit_h -= remove
                    delta += remove * h_mass

    return MolecularState(atoms=atoms, name=mol.name)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """
    Runs the diffusion model while enforcing chemistry checks at each step.

    Parameters
    ----------
    model       : MolecularDiffusionModel
    reactants   : list of MolecularState — used for mass / charge conservation
    T           : total diffusion timesteps
    max_retries : how many re-samples before giving up and backtracking
    max_backtracks : safety limit on total backtracks per generation
    verbose     : print step-by-step log to stdout
    """

    def __init__(
        self,
        model: MolecularDiffusionModel,
        reactants: List[MolecularState],
        T: int = 50,
        max_retries: int = 3,
        max_backtracks: int = 10,
        verbose: bool = False,
        prefer_z3: bool = True,
    ):
        self.model          = model
        self.reactants      = reactants
        self.T              = T
        self.max_retries    = max_retries
        self.max_backtracks = max_backtracks
        self.verbose        = verbose
        self.prefer_z3      = prefer_z3

        self._target_mass   = sum(m.total_mass()   for m in reactants)
        self._target_charge = sum(m.total_charge() for m in reactants)
        self._target_elements = [a.element for m in reactants for a in m.atoms]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> GenerationResult:
        """
        Run the full reverse diffusion loop.
        """
        t0 = time.perf_counter()
        step_log: List[StepRecord] = []
        total_backtracks = 0
        total_corrections = 0
        intermediates: List[IntermediateSnapshot] = []

        # ---- Initialise from reactants (concatenate atoms) -----------
        init_mol = self._build_initial_state()
        x, adj   = encode_molecule(init_mol)

        # ---- Add maximum noise (t = T) to get x_T -------------------
        x_t, adj_t = self.model.forward_noisy(x, adj, self.T, self.T)

        # Stack of committed states for backtracking
        history: List[Tuple[np.ndarray, np.ndarray]] = [(x_t.copy(), adj_t.copy())]

        # ---- Reverse loop: T → 1 ------------------------------------
        t = self.T
        while t >= 1:
            step_start = time.perf_counter()
            committed  = False
            cr = None

            for attempt in range(1, self.max_retries + 2):
                x_prev, adj_prev = self.model.reverse_step(x_t, adj_t, t, self.T)
                candidate = self.model.decode(x_prev, adj_prev, name="intermediate")

                # Mid-trajectory: check valency only (mass checked at end)
                cr = check_intermediate(candidate)

                # Final step: also check full reaction conservation
                if t == 1:
                    # Assign proper name and check conservation
                    candidate = MolecularState(
                        name="product", atoms=candidate.atoms,
                    )
                    cr = check_reaction(
                        self.reactants, [candidate],
                        prefer_z3=self.prefer_z3,
                    )

                elapsed = (time.perf_counter() - step_start) * 1000

                if cr.sat:
                    action = "commit" if attempt == 1 else "corrected"
                    step_log.append(StepRecord(t, attempt, cr, action, elapsed))
                    x_t, adj_t = x_prev, adj_prev
                    history.append((x_t.copy(), adj_t.copy()))
                    committed = True
                    if attempt > 1:
                        total_corrections += 1
                    if self.verbose:
                        self._log(t, action, cr, elapsed)
                    break
                else:
                    # Try corrections before next re-sample
                    if attempt <= self.max_retries:
                        if t == 1:
                            candidate = _fix_composition(candidate, self._target_elements)
                        candidate = _fix_valency(candidate)
                        if t == 1:
                            candidate = _fix_charge(candidate, self._target_charge)
                            candidate = _fix_mass(candidate, self._target_mass)
                        cr2 = (
                            check_reaction(self.reactants, [candidate], prefer_z3=self.prefer_z3)
                            if t == 1 else check_intermediate(candidate)
                        )
                        if cr2.sat:
                            # Correction worked — re-encode and commit
                            x_prev, adj_prev = encode_molecule(candidate)
                            x_t, adj_t = x_prev, adj_prev
                            history.append((x_t.copy(), adj_t.copy()))
                            committed = True
                            total_corrections += 1
                            step_log.append(StepRecord(t, attempt, cr2, "corrected", elapsed))
                            if self.verbose:
                                self._log(t, "corrected", cr2, elapsed)
                            break

            if not committed:
                # Backtrack
                total_backtracks += 1
                if self.verbose:
                    print(f"  [t={t:3d}] BACKTRACK #{total_backtracks} -- violations: {cr.reason}")
                if len(history) > 1:
                    failed_t = t
                    history.pop()               # discard current
                    x_t, adj_t = history[-1]
                    t = min(t + 1, self.T)      # step back up one step
                    step_log.append(StepRecord(failed_t, 0, cr, "backtrack", 0.0))
                else:
                    # Nowhere to backtrack — commit best effort
                    step_log.append(StepRecord(t, 0, cr, "skip", 0.0))
                    x_t, adj_t = x_prev, adj_prev
                    history.append((x_t.copy(), adj_t.copy()))

                if total_backtracks >= self.max_backtracks:
                    if self.verbose:
                        print(f"  [supervisor] Max backtracks reached; stopping early.")
                    break

            if committed or (step_log and step_log[-1].action == "skip"):
                snap = self.model.decode(x_t, adj_t, name=f"t{t}")
                intermediates.append(IntermediateSnapshot(
                    t=t, molecule=snap, adjacency=adj_t.tolist(),
                ))

            t -= 1

        # ---- Decode final state and apply composition corrections ------
        product = self.model.decode(x_t, adj_t, name="product")
        product = _fix_composition(product, self._target_elements)
        product = _fix_valency(product)
        product = _fix_charge(product, self._target_charge)
        product = _fix_mass(product, self._target_mass)

        # Re-encode after fixes so the adjacency reflects corrections
        x_t, adj_t = encode_molecule(product)

        # ---- Final conservation check --------------------------------
        final_check = check_reaction(
            self.reactants, [product], prefer_z3=self.prefer_z3
        )

        wall_time = time.perf_counter() - t0
        result = GenerationResult(
            product=product,
            reactants=self.reactants,
            final_check=final_check,
            step_log=step_log,
            total_backtracks=total_backtracks,
            total_corrections=total_corrections,
            wall_time_s=wall_time,
            intermediates=intermediates,
            product_adjacency=adj_t.tolist(),
        )

        if self.verbose:
            print(result.summary())

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> MolecularState:
        """Concatenate all reactant atoms into a single initial state."""
        atoms = []
        for mol in self.reactants:
            atoms.extend(mol.atoms)
        return MolecularState(name="reactants_concat", atoms=atoms)

    def _log(self, t: int, action: str, cr: ConstraintResult, elapsed: float) -> None:
        icon = "[OK]" if cr.sat else "[FAIL]"
        print(f"  [t={t:3d}] {icon} {action:<12} ({elapsed:.1f} ms)")
        if not cr.sat:
            for v in cr.violations:
                print(f"           -> {v}")
