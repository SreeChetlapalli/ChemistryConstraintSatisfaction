# Chemistry Constraint Satisfaction

**Constraint-checked molecular generation** — Z3 (or a pure-Python fallback) verifies mass/charge conservation and bond valency while a small diffusion-style model runs inside a **supervisor** loop.

---

## Navigate this repo

| I want to… | Go here |
|------------|---------|
| **Run something in 2 minutes** | [Quick start](#quick-start) |
| **Install and import the package** | [Setup](#setup) · [Usage examples](#usage-examples) |
| **Run tests** | [Tests](#tests) · [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Try a full demo (CLI)** | `python scripts/demo.py` |
| **Try Jupyter / Colab** | [Notebooks](#notebooks) |
| **See every folder** | [Repository map](#repository-map) |
| **Change the code** | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Table of contents

- [Navigate this repo](#navigate-this-repo)
- [Quick start](#quick-start)
- [What this project does](#what-this-project-does)
- [Repository map](#repository-map)
- [Setup](#setup)
- [Usage examples](#usage-examples)
- [Module map](#module-map)
- [Z3 vs pure Python](#z3-vs-pure-python)
- [Notebooks](#notebooks)
- [Tests](#tests)
- [Scripts](#scripts)
- [Design notes & trade-offs](#design-notes--trade-offs)
- [Goals (example timeline)](#goals-example-timeline)
- [Contributing & license](#contributing--license)

---

## Quick start

1. **Clone** the repository and `cd` into the folder (name may be `ChemistryConstraintSatisfaction` or `ChemistryConstraintSatisfaction-1` depending on how you cloned).

2. **Create a venv** (Python **3.10+** recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** (pick one):

   ```bash
   # Recommended: editable install so `import chemistry_constraint_satisfaction` works everywhere
   pip install -e ".[dev]"
   ```

   ```bash
   # Alternative: requirements file only (you may need to set PYTHONPATH=src for some tools)
   pip install -r requirements.txt
   ```

4. **Verify**:

   ```bash
   python scripts/check_env.py
   ```

   You want to see PyTorch, Z3, and the package version, ending with `OK — environment ready.`

5. **Run tests**:

   ```bash
   python run_tests.py
   ```

6. **Run the demo**:

   ```bash
   python scripts/demo.py
   ```

---

## What this project does

Generative models can propose **invalid** structures (e.g. wrong valency, broken conservation). This repo wraps a small **NumPy** graph denoising model with a **supervisor** that:

1. Decodes each step to a `MolecularState`.
2. Runs **constraint checks** (valency during the trajectory; full reaction check at the end).
3. **Corrects or backtracks** within configured limits (`max_retries`, `max_backtracks`).

If `z3-solver` is installed, checks can use Z3; otherwise a **pure-Python** checker is used.

---

## Repository map

```
ChemistryConstraintSatisfaction/
├── README.md                 ← You are here (overview + navigation)
├── CONTRIBUTING.md           ← Tests, layout, how to contribute
├── pyproject.toml            ← Package metadata + editable install + pytest config
├── requirements.txt          ← Same runtime deps as pyproject (pip -r friendly)
├── run_tests.py              ← Run all tests with stdlib unittest only
│
├── src/chemistry_constraint_satisfaction/   ← Installable Python package
│   ├── __init__.py           ← Package version
│   ├── constraints/          ← Atoms, molecules, check_reaction / check_intermediate
│   └── diffusion/            ← MolecularDiffusionModel, Supervisor, encode/decode
│
├── tests/                    ← pytest suites (also mirrored in run_tests.py)
├── scripts/
│   ├── check_env.py          ← Quick environment sanity check
│   └── demo.py               ← End-to-end CLI demo + small benchmark
└── notebooks/
    └── demo.ipynb            ← Interactive walkthrough + optional PyTorch training
```

---

## Setup

### Clone

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### Dependencies

- **Required:** `numpy`, `torch`, `z3-solver`, `tqdm` (see `requirements.txt` or `pyproject.toml`).
- **Optional:** RDKit (often via Conda) for SMILES / extra chemistry tooling — not required for the core demos.

```bash
conda install -c conda-forge rdkit
```

### GPU (optional)

- Local: install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org).
- Colab: use `notebooks/demo.ipynb` and set the runtime to GPU.

---

## Usage examples

### Install the package (recommended)

From the repository root:

```bash
pip install -e .
```

With test tools:

```bash
pip install -e ".[dev]"
```

This registers `chemistry_constraint_satisfaction` on your Python path so imports work from any working directory.

### Check a reaction (mass, charge, valency)

```python
from chemistry_constraint_satisfaction.constraints import (
    Atom,
    MolecularState,
    check_reaction,
)

reactants = [
    MolecularState("CH3Br", [
        Atom("C", 4), Atom("Br", 1),
        Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("OH-", [
        Atom("O", 1, formal_charge=-1),
        Atom("H", 1),
    ]),
]
products = [
    MolecularState("CH3OH", [
        Atom("C", 4), Atom("O", 2),
        Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("Br-", [Atom("Br", 0, formal_charge=-1)]),
]

result = check_reaction(reactants, products)
print(result.sat, result.reason)
```

### Run supervised generation (diffusion + supervisor)

```python
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel,
    Supervisor,
)

model = MolecularDiffusionModel(hidden_dim=64, seed=42)
sup = Supervisor(
    model,
    reactants=reactants,
    T=20,
    verbose=True,
)
out = sup.run()
print(out.success, len(out.product.atoms))
```

---

## Module map

| Area | Module path | Role |
|------|-------------|------|
| Atoms / molecules / checks | `chemistry_constraint_satisfaction.constraints` | `Atom`, `MolecularState`, `check_reaction`, `check_intermediate` |
| Diffusion model | `chemistry_constraint_satisfaction.diffusion.model` | `MolecularDiffusionModel`, `encode_molecule` |
| Supervisor loop | `chemistry_constraint_satisfaction.diffusion.supervisor` | `Supervisor`, `GenerationResult` |

---

## Z3 vs pure Python

- If `z3-solver` is installed, `check_reaction(..., prefer_z3=True)` can use the solver.
- Otherwise the same API falls back to a pure-Python checker (`Z3_AVAILABLE` in `constraints`).

---

## Notebooks

### `demo.ipynb`

End-to-end walkthrough:

1. Install dependencies (Z3, etc.) — Colab-friendly cells at the top.
2. Constraint checking on example molecules and reactions.
3. One supervised diffusion run (`Supervisor`).
4. Optional PyTorch training loop for the denoising GNN and simple benchmarks.

### Open locally

```bash
# from repo root, with venv activated
pip install -e ".[dev]"   # or pip install -r requirements.txt + jupyter
jupyter notebook notebooks/demo.ipynb
```

### Open in Google Colab

Upload the repo or clone it in a Colab cell, then open `notebooks/demo.ipynb`. Edit the clone URL in the setup cell to match your fork. Enable **Runtime → Change runtime type → GPU** if you want faster training.

---

## Tests

| Command | When to use |
|---------|-------------|
| `python run_tests.py` | No pytest installed; uses **unittest** only |
| `pytest tests/ -v` | After `pip install -e ".[dev]"` |

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for details.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `python scripts/check_env.py` | Confirms PyTorch, Z3, and package import |
| `python scripts/demo.py` | Constraint demos + supervised generation + short benchmark |

---

## Design notes & trade-offs

- Steps are only committed when they pass the configured checks (or after correction).
- Z3 adds **runtime cost**; use `prefer_z3=False` or the pure-Python path when you need speed over solver-backed checks.
- Final validity still depends on the **model**; the supervisor enforces **checked** constraints, not "magic chemistry."

---

## Goals (example timeline)

| Milestone | Target |
|-----------|--------|
| Mid-term | Step-level correction with Z3-backed checks |
| End-term | Larger benchmark vs a baseline generator |

---

## Contributing & license

- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **License:** Use and cite as appropriate for your institution. This project builds on open-source tools (e.g. PyTorch, Z3).

---

## Publishing note

If you fork this repo, update **`pyproject.toml`** `name` / `version` as needed before publishing to PyPI. The `[project.urls]` block is intentionally omitted so you can add your real repository URL when you publish.
