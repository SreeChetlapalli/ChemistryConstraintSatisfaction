"""Flask API for Chemistry Constraint Satisfaction."""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, jsonify, request
from flask_cors import CORS

from chemistry_constraint_satisfaction.constraints import (
    Atom, MolecularState, check_reaction, check_intermediate,
    Z3_AVAILABLE, ATOMIC_MASS, MAX_VALENCY, ATOMIC_NUMBER,
)
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel, Supervisor,
)
from chemistry_constraint_satisfaction.diffusion.model import encode_molecule

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_molecule(data):
    atoms = [
        Atom(
            element=a["element"],
            bonds=a["bonds"],
            formal_charge=a.get("formal_charge", 0),
            implicit_h=a.get("implicit_h", 0),
        )
        for a in data["atoms"]
    ]
    return MolecularState(name=data.get("name", "molecule"), atoms=atoms)


def serialize_cr(cr):
    return {"sat": cr.sat, "violations": cr.violations, "reason": cr.reason}


def serialize_atom(a):
    return {
        "element": a.element,
        "bonds": a.bonds,
        "formal_charge": a.formal_charge,
        "implicit_h": a.implicit_h,
        "effective_valency": a.effective_valency,
        "total_bonds": a.total_bonds,
    }


def lipinski_properties(mol):
    mw = mol.total_mass()
    hbd = sum(1 for a in mol.atoms if a.element in ("N", "O") and a.implicit_h > 0)
    hba = sum(1 for a in mol.atoms if a.element in ("N", "O", "F"))
    heavy = sum(1 for a in mol.atoms if a.element != "H")
    passes = mw < 500 and hbd <= 5 and hba <= 10
    return {
        "mw": round(mw, 3),
        "hbd": hbd,
        "hba": hba,
        "heavy_atoms": heavy,
        "passes_ro5": passes,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/info")
def info():
    return jsonify({
        "z3_available": Z3_AVAILABLE,
        "elements": sorted(ATOMIC_MASS.keys(), key=lambda s: ATOMIC_NUMBER.get(s, 999)),
        "atomic_masses": ATOMIC_MASS,
        "max_valencies": MAX_VALENCY,
        "atomic_numbers": ATOMIC_NUMBER,
        "version": "0.3.0",
    })


@app.route("/api/presets")
def presets():
    ch3br = {"name": "CH3Br", "atoms": [
        {"element": "C", "bonds": 4}, {"element": "Br", "bonds": 1},
        {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
    ]}
    oh_minus = {"name": "OH-", "atoms": [
        {"element": "O", "bonds": 1, "formal_charge": -1}, {"element": "H", "bonds": 1},
    ]}
    ch3oh = {"name": "CH3OH", "atoms": [
        {"element": "C", "bonds": 4}, {"element": "O", "bonds": 2},
        {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
        {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
    ]}
    br_minus = {"name": "Br-", "atoms": [{"element": "Br", "bonds": 0, "formal_charge": -1}]}
    br_neutral = {"name": "Br", "atoms": [{"element": "Br", "bonds": 0}]}

    return jsonify({
        "reactions": [
            {"name": "SN2: CH3Br + OH- -> CH3OH + Br-",
             "reactants": [ch3br, oh_minus], "products": [ch3oh, br_minus]},
            {"name": "Missing product: CH3Br + OH- -> CH3OH",
             "reactants": [ch3br, oh_minus], "products": [ch3oh]},
            {"name": "Charge mismatch: CH3Br + OH- -> CH3OH + Br",
             "reactants": [ch3br, oh_minus], "products": [ch3oh, br_neutral]},
        ],
        "molecules": [
            {"name": "Water (H2O)", "atoms": [
                {"element": "O", "bonds": 2}, {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
            ]},
            {"name": "Methane (CH4)", "atoms": [
                {"element": "C", "bonds": 4},
                {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
                {"element": "H", "bonds": 1}, {"element": "H", "bonds": 1},
            ]},
            {"name": "Ammonium (NH4+)", "atoms": [
                {"element": "N", "bonds": 4, "formal_charge": 1},
            ]},
            {"name": "Invalid: Carbon 5 bonds", "atoms": [
                {"element": "C", "bonds": 5},
            ]},
        ],
    })


@app.route("/api/check-reaction", methods=["POST"])
def api_check_reaction():
    data = request.json
    try:
        reactants = [parse_molecule(m) for m in data["reactants"]]
        products = [parse_molecule(m) for m in data["products"]]
        t0 = time.perf_counter()
        cr = check_reaction(reactants, products, prefer_z3=data.get("prefer_z3", True))
        elapsed = (time.perf_counter() - t0) * 1000
        return jsonify({"result": serialize_cr(cr), "elapsed_ms": round(elapsed, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/check-intermediate", methods=["POST"])
def api_check_intermediate():
    data = request.json
    try:
        mol = parse_molecule(data["molecule"])
        t0 = time.perf_counter()
        cr = check_intermediate(mol)
        elapsed = (time.perf_counter() - t0) * 1000
        return jsonify({
            "result": serialize_cr(cr),
            "elapsed_ms": round(elapsed, 2),
            "total_mass": mol.total_mass(),
            "total_charge": mol.total_charge(),
            "lipinski": lipinski_properties(mol),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/run-supervisor", methods=["POST"])
def api_run_supervisor():
    data = request.json
    try:
        reactants = [parse_molecule(m) for m in data["reactants"]]
        model = MolecularDiffusionModel(
            hidden_dim=data.get("hidden_dim", 64), seed=data.get("seed", 42),
        )
        sup = Supervisor(
            model, reactants=reactants,
            T=data.get("T", 20),
            max_retries=data.get("max_retries", 3),
            max_backtracks=data.get("max_backtracks", 5),
            verbose=False,
            prefer_z3=data.get("prefer_z3", Z3_AVAILABLE),
        )
        result = sup.run()
        step_log = [
            {
                "t": s.t, "attempt": s.attempt, "action": s.action,
                "elapsed_ms": round(s.elapsed_ms, 2),
                "constraint_sat": s.constraint_result.sat,
                "constraint_reason": s.constraint_result.reason,
                "violations": s.constraint_result.violations,
            }
            for s in result.step_log
        ]
        intermediates = [
            {
                "t": snap.t,
                "atoms": [serialize_atom(a) for a in snap.molecule.atoms],
                "adjacency": snap.adjacency,
                "total_mass": snap.molecule.total_mass(),
                "total_charge": snap.molecule.total_charge(),
            }
            for snap in result.intermediates
        ]
        return jsonify({
            "success": result.success,
            "product": {
                "name": result.product.name,
                "atoms": [serialize_atom(a) for a in result.product.atoms],
                "total_mass": result.product.total_mass(),
                "total_charge": result.product.total_charge(),
                "adjacency": result.product_adjacency,
                "lipinski": lipinski_properties(result.product),
            },
            "final_check": serialize_cr(result.final_check),
            "step_log": step_log,
            "intermediates": intermediates,
            "total_backtracks": result.total_backtracks,
            "total_corrections": result.total_corrections,
            "wall_time_s": round(result.wall_time_s, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/noise-schedule")
def api_noise_schedule():
    T = request.args.get("T", 50, type=int)
    T = max(1, min(T, 200))
    model = MolecularDiffusionModel(hidden_dim=8, seed=0)
    schedule = []
    for t in range(1, T + 1):
        beta = model._beta(t, T)
        ab = model._alpha_bar_cached(t, T)
        schedule.append({"t": t, "beta": round(beta, 6), "alpha_bar": round(ab, 6)})
    return jsonify({"T": T, "schedule": schedule})


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.json
    try:
        reactants = [parse_molecule(m) for m in data["reactants"]]
        n = min(data.get("n", 20), 100)
        sup_val = sup_full = raw_val = raw_full = 0
        runs = []

        for i in range(n):
            m = MolecularDiffusionModel(hidden_dim=32, seed=i)
            r = Supervisor(
                m, reactants, T=10, max_retries=2, max_backtracks=3,
                verbose=False, prefer_z3=False,
            ).run()
            s_val = check_intermediate(r.product).sat
            s_full = r.success
            if s_val:
                sup_val += 1
            if s_full:
                sup_full += 1

            m2 = MolecularDiffusionModel(hidden_dim=32, seed=i)
            init_atoms = [a for mol in reactants for a in mol.atoms]
            x, adj = encode_molecule(MolecularState("init", init_atoms))
            x_n, adj_n = m2.forward_noisy(x, adj, t=10, T=10)
            for t in range(10, 0, -1):
                x_n, adj_n = m2.reverse_step(x_n, adj_n, t=t, T=10)
            raw_prod = m2.decode(x_n, adj_n, name="raw")
            u_val = check_intermediate(raw_prod).sat
            u_full = check_reaction(reactants, [raw_prod], prefer_z3=False).sat
            if u_val:
                raw_val += 1
            if u_full:
                raw_full += 1

            runs.append({
                "seed": i,
                "supervised_valency": s_val,
                "supervised_conservation": s_full,
                "unsupervised_valency": u_val,
                "unsupervised_conservation": u_full,
            })

        return jsonify({
            "n": n,
            "runs": runs,
            "summary": {
                "supervised_valency_pct": round(sup_val / n * 100, 1),
                "unsupervised_valency_pct": round(raw_val / n * 100, 1),
                "supervised_full_pct": round(sup_full / n * 100, 1),
                "unsupervised_full_pct": round(raw_full / n * 100, 1),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def api_train():
    """Evolutionary model selection: evaluate many seeds, return fitness curve."""
    data = request.json
    try:
        reactants = [parse_molecule(m) for m in data["reactants"]]
        pop_size = min(data.get("population", 16), 32)
        generations = min(data.get("generations", 10), 20)
        hidden_dim = data.get("hidden_dim", 32)
        T = data.get("T", 10)

        history = []
        seeds = list(range(pop_size))

        for gen in range(generations):
            scores = []
            for seed in seeds:
                m = MolecularDiffusionModel(hidden_dim=hidden_dim, seed=seed)
                r = Supervisor(
                    m, reactants, T=T, max_retries=2, max_backtracks=3,
                    verbose=False, prefer_z3=False,
                ).run()
                val_ok = 1 if check_intermediate(r.product).sat else 0
                full_ok = 1 if r.success else 0
                bt_penalty = r.total_backtracks * 0.05
                score = val_ok * 0.6 + full_ok * 0.4 - bt_penalty
                scores.append((seed, score, val_ok, full_ok, r.total_backtracks))

            scores.sort(key=lambda x: -x[1])
            best = scores[0]
            avg_score = sum(s[1] for s in scores) / len(scores)
            val_rate = sum(s[2] for s in scores) / len(scores) * 100
            full_rate = sum(s[3] for s in scores) / len(scores) * 100

            history.append({
                "generation": gen,
                "best_seed": best[0],
                "best_score": round(best[1], 4),
                "avg_score": round(avg_score, 4),
                "valency_rate": round(val_rate, 1),
                "conservation_rate": round(full_rate, 1),
                "population_size": len(seeds),
            })

            top_k = max(2, len(seeds) // 4)
            top_seeds = [s[0] for s in scores[:top_k]]
            rng = random.Random(gen)
            new_seeds = list(top_seeds)
            while len(new_seeds) < pop_size:
                parent = rng.choice(top_seeds)
                mutated = parent + rng.randint(-50, 50)
                if mutated < 0:
                    mutated = rng.randint(0, 9999)
                new_seeds.append(mutated)
            seeds = new_seeds

        final_best = max(history, key=lambda h: h["best_score"])
        return jsonify({
            "history": history,
            "best_seed": final_best["best_seed"],
            "best_score": final_best["best_score"],
            "config": {
                "hidden_dim": hidden_dim,
                "T": T,
                "population": pop_size,
                "generations": generations,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/monte-carlo", methods=["POST"])
def api_monte_carlo():
    data = request.json
    try:
        reactants = [parse_molecule(m) for m in data["reactants"]]
        n = min(data.get("n_samples", 100), 500)
        hidden_dim = data.get("hidden_dim", 32)
        T = data.get("T", 10)

        runs = []
        violation_counts = {}
        valid_count = 0
        masses = []
        lip_pass = 0

        for seed in range(n):
            m = MolecularDiffusionModel(hidden_dim=hidden_dim, seed=seed)
            r = Supervisor(
                m, reactants, T=T, max_retries=2, max_backtracks=3,
                verbose=False, prefer_z3=False,
            ).run()
            val_ok = check_intermediate(r.product).sat
            full_ok = r.success
            prod_mass = r.product.total_mass()
            lip = lipinski_properties(r.product)
            if lip["passes_ro5"]:
                lip_pass += 1

            viols = r.final_check.violations if not full_ok else []
            for v in viols:
                vtype = v.split(":")[0].strip() if ":" in v else v.split(" ")[0]
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

            if full_ok:
                valid_count += 1
            masses.append(prod_mass)

            runs.append({
                "seed": seed,
                "valid": full_ok,
                "valency_ok": val_ok,
                "atom_count": len(r.product.atoms),
                "mass": round(prod_mass, 2),
                "backtracks": r.total_backtracks,
                "corrections": r.total_corrections,
                "wall_time_ms": round(r.wall_time_s * 1000, 1),
                "lipinski_pass": lip["passes_ro5"],
                "violations": viols,
            })

        avg_mass = sum(masses) / len(masses) if masses else 0
        return jsonify({
            "n": n,
            "runs": runs,
            "summary": {
                "validity_rate": round(valid_count / n * 100, 1),
                "avg_backtracks": round(sum(r["backtracks"] for r in runs) / n, 2),
                "avg_corrections": round(sum(r["corrections"] for r in runs) / n, 2),
                "avg_mass": round(avg_mass, 2),
                "lipinski_pass_rate": round(lip_pass / n * 100, 1),
                "violation_breakdown": violation_counts,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/pathway", methods=["POST"])
def api_pathway():
    data = request.json
    try:
        steps_input = data["steps"]
        hidden_dim = data.get("hidden_dim", 64)
        seed = data.get("seed", 42)
        T = data.get("T", 10)

        step_results = []
        prev_products = None

        for i, step in enumerate(steps_input):
            if step.get("reactants"):
                reactants = [parse_molecule(m) for m in step["reactants"]]
            elif prev_products:
                reactants = prev_products
            else:
                return jsonify({"error": f"Step {i} has no reactants and no previous products"}), 400

            m = MolecularDiffusionModel(hidden_dim=hidden_dim, seed=seed + i)
            sup = Supervisor(
                m, reactants, T=T, max_retries=2, max_backtracks=3,
                verbose=False, prefer_z3=False,
            )
            result = sup.run()

            product_mol = result.product
            prev_products = [product_mol]

            step_results.append({
                "step": i,
                "success": result.success,
                "product": {
                    "name": product_mol.name,
                    "atoms": [serialize_atom(a) for a in product_mol.atoms],
                    "total_mass": product_mol.total_mass(),
                    "total_charge": product_mol.total_charge(),
                    "adjacency": result.product_adjacency,
                    "lipinski": lipinski_properties(product_mol),
                },
                "final_check": serialize_cr(result.final_check),
                "backtracks": result.total_backtracks,
                "corrections": result.total_corrections,
                "wall_time_s": round(result.wall_time_s, 4),
            })

        first_reactants = [parse_molecule(m) for m in steps_input[0]["reactants"]]
        last_product = prev_products[0] if prev_products else first_reactants[0]
        chain_check = check_reaction(first_reactants, [last_product], prefer_z3=False)

        return jsonify({
            "steps": step_results,
            "chain_valid": chain_check.sat,
            "chain_check": serialize_cr(chain_check),
            "total_steps": len(step_results),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
