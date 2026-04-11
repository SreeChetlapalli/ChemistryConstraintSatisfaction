#!/usr/bin/env python3
"""
Quick demo: constraint checks, one supervised run, and a benchmark.

    python scripts/demo.py
"""

import sys, os, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chemistry_constraint_satisfaction.constraints import (
    Atom, MolecularState, check_reaction, check_intermediate,
    Z3_AVAILABLE,
)
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel, Supervisor,
)

SEP = "-" * 60


# ---------------------------------------------------------------------------
# Reaction library
# ---------------------------------------------------------------------------

def sn2_reactants():
    """CH3Br + OH-"""
    return [
        MolecularState("CH3Br", [
            Atom("C", 4), Atom("Br", 1),
            Atom("H", 1), Atom("H", 1), Atom("H", 1),
        ]),
        MolecularState("OH-", [
            Atom("O", 1, formal_charge=-1), Atom("H", 1),
        ]),
    ]

def sn2_products():
    """CH3OH + Br-"""
    return [
        MolecularState("CH3OH", [
            Atom("C", 4), Atom("O", 2),
            Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
        ]),
        MolecularState("Br-", [Atom("Br", 0, formal_charge=-1)]),
    ]

def combustion_reactants():
    """CH4 + 2 O2  (simplified: one O2 pair per MolecularState)"""
    return [
        MolecularState("CH4", [
            Atom("C", 4),
            Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
        ]),
        MolecularState("O2_a", [Atom("O", 2), Atom("O", 2)]),
        MolecularState("O2_b", [Atom("O", 2), Atom("O", 2)]),
    ]

def combustion_products():
    """CO2 + 2 H2O"""
    return [
        MolecularState("CO2", [
            Atom("C", 4), Atom("O", 2), Atom("O", 2),
        ]),
        MolecularState("H2O_a", [Atom("O", 2), Atom("H", 1), Atom("H", 1)]),
        MolecularState("H2O_b", [Atom("O", 2), Atom("H", 1), Atom("H", 1)]),
    ]

def acid_base_reactants():
    """HCl + NaOH (Na+ / OH- so atom counts balance with products)."""
    return [
        MolecularState("HCl", [
            Atom("H", 1), Atom("Cl", 1),
        ]),
        MolecularState("NaOH", [
            Atom("Na", 0, formal_charge=+1),
            Atom("O", 1, formal_charge=-1),
            Atom("H", 1),
        ]),
    ]

def acid_base_products():
    """H2O + NaCl"""
    return [
        MolecularState("H2O", [
            Atom("O", 2), Atom("H", 1), Atom("H", 1),
        ]),
        MolecularState("NaCl", [
            Atom("Na", 0, formal_charge=+1),
            Atom("Cl", 0, formal_charge=-1),
        ]),
    ]

REACTION_LIBRARY = [
    ("SN2: CH3Br + OH-", sn2_reactants),
    ("Combustion: CH4 + 2O2", combustion_reactants),
    ("Acid-base: HCl + NaOH", acid_base_reactants),
]


# ---------------------------------------------------------------------------
# Part 1: Constraint checks
# ---------------------------------------------------------------------------

def demo_constraints():
    print(SEP)
    print("  PART 1 - Chemical Axiom Checks")
    print(f"  (Z3 solver: {'available' if Z3_AVAILABLE else 'not installed - using pure-Python fallback'})")
    print(SEP)

    def _show(label, cr):
        icon = "[OK]" if cr.sat else "[FAIL]"
        print(f"\n  {icon} {label}")
        print(f"    {cr.reason}")

    _show("CH3Br + OH- -> CH3OH + Br-  (valid SN2)",
          check_reaction(sn2_reactants(), sn2_products()))

    _show("CH4 + 2 O2 -> CO2 + 2 H2O  (valid combustion)",
          check_reaction(combustion_reactants(), combustion_products()))

    _show("Carbon with 5 bonds (overvalenced)",
          check_intermediate(MolecularState("bad", [Atom("C", 5)])))

    _show("CH3Br + OH- -> CH3OH  (missing Br-)",
          check_reaction(sn2_reactants(), [sn2_products()[0]]))

    _show("CH3Br + OH- -> CH3OH + Br (neutral Br - charge mismatch)",
          check_reaction(sn2_reactants(),
                         [sn2_products()[0], MolecularState("Br", [Atom("Br", 0)])]))

    _show("NH4+ - nitrogen with 4 bonds (formal charge +1)",
          check_intermediate(MolecularState("NH4+", [Atom("N", 4, formal_charge=+1)])))


# ---------------------------------------------------------------------------
# Part 2: Supervised generation
# ---------------------------------------------------------------------------

def demo_generation():
    print(f"\n{SEP}")
    print("  PART 2 - Supervisor loop demo")
    print("  Reaction: CH3Br + OH-  ->  (supervised diffusion)  ->  ?")
    print(SEP)

    model = MolecularDiffusionModel(hidden_dim=64, seed=42)
    sup   = Supervisor(
        model,
        reactants=sn2_reactants(),
        T=20,
        max_retries=3,
        max_backtracks=5,
        verbose=True,
        prefer_z3=Z3_AVAILABLE,
    )
    result = sup.run()

    print("\n  Product atoms:")
    for i, atom in enumerate(result.product.atoms):
        print(f"    [{i}] {atom.element:3s}  bonds={atom.bonds}  "
              f"implicit_H={atom.implicit_h}  charge={atom.formal_charge:+d}")

    print(f"\n  Metrics: {result.metrics}")
    return result


# ---------------------------------------------------------------------------
# Part 3: Multi-reaction benchmark
# ---------------------------------------------------------------------------

def demo_benchmark(n: int = 50):
    print(f"\n{SEP}")
    print(f"  PART 3 - Benchmark: {n} generations x {len(REACTION_LIBRARY)} reactions")
    print(f"  Comparing supervised vs unsupervised (random-weight model)")
    print(SEP)

    from chemistry_constraint_satisfaction.diffusion.model import encode_molecule

    results_by_rxn = {}

    for rxn_name, rxn_fn in REACTION_LIBRARY:
        reactants = rxn_fn()
        sup_valency_ok = 0
        raw_valency_ok = 0
        sup_full_ok = 0
        raw_full_ok = 0
        sup_times = []
        raw_times = []

        for i in range(n):
            m = MolecularDiffusionModel(hidden_dim=32, seed=i)

            # ---- SUPERVISED ----
            t0 = time.perf_counter()
            sup = Supervisor(m, reactants, T=10, max_retries=2, max_backtracks=3,
                             verbose=False, prefer_z3=False)
            r = sup.run()
            sup_times.append(time.perf_counter() - t0)
            cr_val = check_intermediate(r.product)
            if cr_val.sat:
                sup_valency_ok += 1
            if r.success:
                sup_full_ok += 1

            # ---- UNSUPERVISED (raw) ----
            m2 = MolecularDiffusionModel(hidden_dim=32, seed=i)
            init_atoms = []
            for mol in reactants:
                init_atoms.extend(mol.atoms)
            init_mol = MolecularState("init", init_atoms)
            x, adj = encode_molecule(init_mol)
            t0 = time.perf_counter()
            x_n, adj_n = m2.forward_noisy(x, adj, t=10, T=10)
            for t in range(10, 0, -1):
                x_n, adj_n = m2.reverse_step(x_n, adj_n, t=t, T=10)
            raw_times.append(time.perf_counter() - t0)
            raw_product = m2.decode(x_n, adj_n, name="raw_product")
            cr_raw_val = check_intermediate(raw_product)
            if cr_raw_val.sat:
                raw_valency_ok += 1
            cr_raw_full = check_reaction(reactants, [raw_product], prefer_z3=False)
            if cr_raw_full.sat:
                raw_full_ok += 1

        results_by_rxn[rxn_name] = {
            "sup_valency": sup_valency_ok / n,
            "raw_valency": raw_valency_ok / n,
            "sup_full": sup_full_ok / n,
            "raw_full": raw_full_ok / n,
            "sup_avg_ms": sum(sup_times) / n * 1000,
            "raw_avg_ms": sum(raw_times) / n * 1000,
        }

    print(f"\n  Results over {n} generations per reaction  (random GNN weights):\n")
    header = f"    {'Reaction':<30} {'Metric':<28} {'Supervised':>10}  {'Unsupervised':>12}"
    print(header)
    print(f"    {'-'*82}")

    for rxn_name, stats in results_by_rxn.items():
        print(f"    {rxn_name:<30} {'Valency validity':<28} "
              f"{stats['sup_valency']*100:>9.1f}%  {stats['raw_valency']*100:>11.1f}%")
        print(f"    {'':<30} {'Full conservation':<28} "
              f"{stats['sup_full']*100:>9.1f}%  {stats['raw_full']*100:>11.1f}%")
        print(f"    {'':<30} {'Avg time (ms)':<28} "
              f"{stats['sup_avg_ms']:>9.1f}   {stats['raw_avg_ms']:>11.1f}")
        print()

    # aggregate
    n_rxn = len(results_by_rxn)
    avg_sup_val = sum(s["sup_valency"] for s in results_by_rxn.values()) / n_rxn * 100
    avg_raw_val = sum(s["raw_valency"] for s in results_by_rxn.values()) / n_rxn * 100
    print(f"    {'AVERAGE':<30} {'Valency validity':<28} "
          f"{avg_sup_val:>9.1f}%  {avg_raw_val:>11.1f}%")

    print()
    print("  Note: the supervisor focuses on per-step valency consistency.")
    print("  Mass/charge conservation depends on trained model weights.")
    print("  Use the training module to improve full conservation validity.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_constraints()
    result = demo_generation()
    demo_benchmark(n=50)

    print(f"\n{SEP}")
    print("  Done. For training, see: from chemistry_constraint_satisfaction.diffusion.training import train")
    print("  For GPU-accelerated training, open notebooks/demo.ipynb in Colab.")
    print(SEP)
