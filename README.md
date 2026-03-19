# Chemistry Constraint Satisfaction

**Constraint-checked molecular generation** — uses Z3 to verify mass/charge conservation and bond valency during a diffusion-style generation loop.

---

## Problem: Invalid molecules from generative models

Generative models often produce **physically impossible molecules** or **violate conservation laws** during reaction prediction. For example:

- Carbon with 5 bonds
- Atoms disappearing or appearing without conservation
- Invalid bond valencies

The usual workaround is to generate many candidates and discard the invalid ones. That can be slow and wasteful.

---

## Approach: Check constraints during generation

The codebase implements a constraint-checking wrapper around the diffusion model:

1. After each reverse/denoising step, decode the candidate into a `MolecularState`.
2. Check intermediate states against chemical axioms (mass/charge conservation, bond valency). If `z3` is installed, the Z3-backed checker is used.
3. If a step fails, try a small correction and/or backtrack (bounded by `max_retries` and `max_backtracks`).
4. At the end, validate the full reaction with `check_reaction`.

### Example

- **Input:** reactants such as CH₃Br and OH⁻  
- **Process:** Generate product step-by-step; after each step, the constraint checker validates valency (and the final step validates mass/charge conservation).  
- **If a decoded state is invalid** (e.g., carbon with 5 bonds): backtrack and try again (within the retry/backtrack limits).  
- **Output:** Chemically valid result, e.g. CH₃OH + Br⁻.

Constraint checks happen during generation, rather than only filtering after the fact.

---

## Design Notes

- Instead of filtering invalid candidates after generation, this wrapper only commits steps that pass the selected checks.
- When Z3 is installed, checks are run with the solver; otherwise there is a pure-Python fallback.
- The final output is still verified with `check_reaction` to confirm mass and charge conservation.

---

## Why this helps

- Fewer invalid candidates make it easier to evaluate reaction pathways.
- Moving checks earlier in the pipeline can reduce wasted downstream work.

---

## Trade-offs

- Runtime overhead: solver checks add cost per step. The retry/backtrack limits keep the worst case bounded; you can also switch to the pure-Python checker for faster runs.

---

## Goals and Timeline

| Milestone | Target |
|-----------|--------|
| **Mid-semester** | Add step-level correction based on Z3-backed checks. |
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
