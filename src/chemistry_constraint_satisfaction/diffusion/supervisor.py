"""
supervisor.py
~~~~~~~~~~~~~
Digital Supervisor — the heart of Correct-by-Design molecular generation.

The supervisor wraps the diffusion model and intercepts every reverse step.
At each step it:

  1. Calls ``model.reverse_step()`` to get a candidate x_{t-1}, adj_{t-1}.
  2. Decodes the candidate into a ``MolecularState``.
  3. Runs ``check_intermediate()`` (and optionally ``check_reaction()``) via Z3.
  4a. If VALID   → commits the step and moves on.
  4b. If INVALID → attempts a correction; if that also fails, backtracks
      to the previous valid state and re-samples (up to ``max_retries``).

After the full trajectory completes, the supervisor always runs a final
``check_reaction()`` to verify mass and charge conservation end-to-end.

Usage example
-------------
>>> from chemistry_constraint_satisfaction.constraints import Atom, MolecularState
>>> from chemistry_constraint_satisfaction.diffusion.model import (
...     MolecularDiffusionModel, encode_molecule)
>>> from chemistry_constraint_satisfaction.diffusion.supervisor import Supervisor
>>>
>>> ch3br = MolecularState(name="CH3Br", atoms=[
...     Atom("C", bonds=4), Atom("Br", bonds=1),
...     Atom("H", bonds=1), Atom("H", bonds=1), Atom("H", bonds=1),
... ])
>>> oh_minus = MolecularState(name="OH-", atoms=[
...     Atom("O", bonds=1, formal_charge=-1), Atom("H", bonds=1),
... ])
>>> model = MolecularDiffusionModel()
>>> sup   = Supervisor(model, reactants=[ch3br, oh_minus], T=10, verbose=True)
>>> result = sup.run()
>>> print(result.final_check)
"""

from __future__ import annotations

import copy
import dataclasses
import time
from typing import List, Optional, Tuple

import numpy as np

from ..constraints.chemical_axioms import (
    Atom, MolecularState, ConstraintResult,
    check_reaction, check_intermediate,
)
from .model import MolecularDiffusionModel, encode_molecule


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepRecord:
    """Log entry for a single denoising step."""
    t: int
    attempt: int
    constraint_result: ConstraintResult
    action: str   # "commit", "corrected", "backtrack", "skip"
    elapsed_ms: float


@dataclasses.dataclass
class GenerationResult:
    """Full result of a supervised generation run."""
    product: MolecularState
    reactants: List[MolecularState]
    final_check: ConstraintResult
    step_log: List[StepRecord]
    total_backtracks: int
    total_corrections: int
    wall_time_s: float

    @property
    def success(self) -> bool:
        return self.final_check.sat

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Digital Supervisor — Generation Summary",
            "=" * 60,
            f"  Product     : {self.product.name}",
            f"  Atoms       : {len(self.product.atoms)}",
            f"  Valid       : {'✓ YES' if self.success else '✗ NO'}",
            f"  Backtracks  : {self.total_backtracks}",
            f"  Corrections : {self.total_corrections}",
            f"  Wall time   : {self.wall_time_s:.3f}s",
            "",
            "  Constraint check:",
        ]
        if self.success:
            lines.append("    ✓ All axioms satisfied.")
        else:
            for v in self.final_check.violations:
                lines.append(f"    ✗ {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Correction strategies
# ---------------------------------------------------------------------------

def _fix_valency(mol: MolecularState) -> MolecularState:
    """
    Minimal correction: reduce bond count on over-valenced atoms.
    Returns a new MolecularState (does not mutate in place).
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


def _fix_mass(
    mol: MolecularState,
    target_mass: float,
    tolerance: float = 0.02,
) -> MolecularState:
    """
    Adjust implicit-H counts to bring mass closer to target.
    Only modifies hydrogen budget (safest change).
    """
    from ..constraints.chemical_axioms import ATOMIC_MASS
    current = mol.total_mass()
    delta   = target_mass - current
    h_mass  = ATOMIC_MASS["H"]

    if abs(delta) <= tolerance:
        return mol

    # Distribute hydrogen adjustments across atoms
    atoms = [dataclasses.replace(a) for a in mol.atoms]
    for atom in atoms:
        if abs(delta) <= tolerance:
            break
        if delta > 0:  # need more mass → add H
            add = min(int(delta / h_mass), atom.effective_valency - atom.total_bonds)
            atom.implicit_h += add
            delta -= add * h_mass
        else:           # need less mass → remove H
            remove = min(int(-delta / h_mass), atom.implicit_h)
            atom.implicit_h -= remove
            delta += remove * h_mass

    return MolecularState(atoms=atoms, name=mol.name)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """
    Digital Supervisor that wraps the diffusion model and enforces chemical
    axioms at every denoising step.

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

        # Compute target mass and charge from reactants
        self._target_mass   = sum(m.total_mass()   for m in reactants)
        self._target_charge = sum(m.total_charge() for m in reactants)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> GenerationResult:
        """
        Run the full supervised reverse diffusion trajectory.
        Returns a GenerationResult regardless of success.
        """
        t0 = time.perf_counter()
        step_log: List[StepRecord] = []
        total_backtracks = 0
        total_corrections = 0

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

            for attempt in range(1, self.max_retries + 2):  # +1 for correction attempt
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
                    # Try a lightweight correction before next re-sample
                    if attempt <= self.max_retries:
                        candidate = _fix_valency(candidate)
                        if t == 1:
                            candidate = _fix_mass(candidate, self._target_mass)
                        cr2 = (
                            check_reaction(self.reactants, [candidate], prefer_z3=self.prefer_z3)
                            if t == 1 else check_intermediate(candidate)
                        )
                        if cr2.sat:
                            # Correction worked — re-encode and commit
                            from .model import encode_molecule as _enc
                            x_prev, adj_prev = _enc(candidate)
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
                    print(f"  [t={t:3d}] BACKTRACK #{total_backtracks} — violations: {cr.reason}")
                if len(history) > 1:
                    history.pop()               # discard current
                    x_t, adj_t = history[-1]
                    t = min(t + 1, self.T)      # step back up one step
                    step_log.append(StepRecord(t, 0, cr, "backtrack", 0.0))
                else:
                    # Nowhere to backtrack — commit best effort
                    step_log.append(StepRecord(t, 0, cr, "skip", 0.0))
                    x_t, adj_t = x_prev, adj_prev
                    history.append((x_t.copy(), adj_t.copy()))

                if total_backtracks >= self.max_backtracks:
                    if self.verbose:
                        print(f"  [supervisor] Max backtracks reached; stopping early.")
                    break

            t -= 1

        # ---- Decode final state -------------------------------------
        product = self.model.decode(x_t, adj_t, name="product")

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
        icon = "✓" if cr.sat else "✗"
        print(f"  [t={t:3d}] {icon} {action:<12} ({elapsed:.1f} ms)")
        if not cr.sat:
            for v in cr.violations:
                print(f"           ↳ {v}")
