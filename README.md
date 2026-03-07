# Chemistry Constraint Satisfaction

**Correct-by-Design neuro-symbolic molecular generation** — integrating Z3 (SMT) as a "Digital Supervisor" into the generative loop of a molecular diffusion model to eliminate chemical hallucinations.

---

## Problem: Chemical Hallucinations in Generative AI

Generative models often produce **physically impossible molecules** or **violate conservation laws** during reaction prediction. For example:

- Carbon with 5 bonds
- Atoms disappearing or appearing without conservation
- Invalid bond valencies

The usual fix is to generate many candidates and discard invalid ones — slow and wasteful.

---

## Approach: Enforce Correctness During Generation

This project proposes a **Correct-by-Design** architecture that:

1. **Pauses the AI at each denoising step** of a molecular diffusion model.
2. **Verifies intermediate states** against hard chemical axioms (e.g., mass conservation, bond valency) using **Z3** (Satisfiability Modulo Theories).
3. **Corrects or backtracks** when constraints are violated (e.g., undo last step, try a different diffusion move).
4. **Guarantees** that 100% of generated outputs are chemically valid.

### Example

- **Input:** reactants such as CH₃Br and OH⁻  
- **Process:** Generate product step-by-step; at each step, Z3 checks mass conservation and valency.  
- **If the model proposes an invalid state** (e.g., C with 5 bonds): undo the last step and try another move (add/move atom).  
- **Output:** Chemically valid result, e.g. CH₃OH + Br⁻.

Reaction-level symbolic constraints are enforced **during** generation, not after.

---

## Intellectual Merit

- Current probabilistic generators **learn** chemistry from data and often violate hard rules; invalid molecules are then **discarded after** generation.
- This work moves to an **integrated verifier** that ensures correctness **while** constructing molecules.
- Z3 can express **exact** conservation laws; the generative model is **nudged to comply** via minimal corrections, giving formal guarantees for targeted reaction classes.

---

## Broader Impact

- **Trustworthy AI in the physical sciences:** From "probabilistic guesser" to a scientifically rigorous tool.
- **Fewer failed experiments:** Reliable, valid reaction pathways reduce wasted lab effort on nonsensical AI suggestions.
- **Stakeholders:** Chemists, pharmaceutical researchers, and anyone needing trustworthy molecule generation.

---

## Risks and Mitigations

- **Latency:** Letting the logic engine "think" at every step could make generation slow. Trade-off: less time sorting through thousands of invalid candidates vs. more compute per step. Cost is currently low; open-source stack (PyTorch, Z3) and GPU access via **Google Colab** are sufficient to start.

---

## Goals and Timeline

| Milestone | Target |
|-----------|--------|
| **Mid-semester** | Z3 pauses and corrects a **single step** of the diffusion model. |
| **End of semester** | Benchmark **1,000 generated reactions** and compare failure rate to a baseline model. |

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd ChemistryConstraintSatisfaction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For **RDKit** (optional, for SMILES and chemistry utilities), use Conda if needed:

```bash
conda install -c conda-forge rdkit
```

### 4. Verify installation

From the repo root (with the venv activated or using the venv’s Python):

```bash
python scripts/check_env.py
```

You should see PyTorch, Z3, and the package version; `OK — environment ready.` means the setup is correct.

### 5. (Optional) GPU and Colab

- For local GPU: install PyTorch with CUDA from [pytorch.org](https://pytorch.org).
- For Colab: upload this repo or clone from Git and run `pip install -r requirements.txt` in a notebook.

---

## Project layout

```
ChemistryConstraintSatisfaction/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── chemistry_constraint_satisfaction/
│       ├── __init__.py
│       ├── constraints/    # Z3 chemical axioms (mass, valency)
│       └── diffusion/      # Diffusion model + supervisor loop
├── scripts/
│   └── check_env.py        # Verify PyTorch, Z3, and package
├── tests/
└── notebooks/              # Colab-friendly experiments
```

---

## License

Use and cite as appropriate for your institution. Open-source tools: PyTorch, Z3.
