# Chemistry Constraint Satisfaction

**Constraint-checked molecular generation** — a GNN diffusion model generates molecular structures while a supervisor loop enforces chemical rules (mass conservation, charge conservation, bond valency) at every denoising step using Z3 or a pure-Python fallback.

The project ships as a full-stack web application with an interactive React frontend, a Flask API, and a core Python engine.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Servers](#running-the-servers)
- [What This Project Does](#what-this-project-does)
- [Architecture](#architecture)
- [Frontend — UI Pages](#frontend--ui-pages)
- [Backend — API Endpoints](#backend--api-endpoints)
- [Core Python Package](#core-python-package)
- [CLI Demo & Notebooks](#cli-demo--notebooks)
- [Tests](#tests)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Design Notes & Trade-offs](#design-notes--trade-offs)
- [Contributing & License](#contributing--license)

---

## Quick Start

> Get the full web app running in ~5 minutes.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd ChemistryConstraintSatisfaction-2

# 2. Set up Python (3.10+)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install Python dependencies
pip install -e ".[dev]"
pip install flask flask-cors

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. Start both servers (two terminals)

# Terminal 1 — Backend (Flask API on port 5000):
python backend/app.py

# Terminal 2 — Frontend (Vite dev server on port 3000):
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser. The frontend proxies all `/api` requests to the Flask backend on port 5000.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Core engine, Flask API, PyTorch training |
| **Node.js** | 18+ | Frontend dev server |
| **npm** | 9+ | Frontend package management |
| **Git** | Any | Cloning the repo |

Optional:
- **CUDA-capable GPU** — speeds up PyTorch model training (CPU works fine for everything else)
- **RDKit** — for SMILES parsing (not required for core functionality)

---

## Installation

### Python Environment

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the core package (editable) and backend dependencies:

```bash
pip install -e ".[dev]"
pip install flask flask-cors
```

This installs: `numpy`, `torch`, `z3-solver`, `tqdm`, `pytest`, `flask`, and `flask-cors`.

Verify the environment:

```bash
python scripts/check_env.py
```

You should see PyTorch, Z3, and the package version, ending with `OK — environment ready.`

### Frontend

```bash
cd frontend
npm install
```

This installs React, Vite, Three.js, Tailwind CSS, and other frontend dependencies.

---

## Running the Servers

The web application requires **two servers running simultaneously**: the Flask backend API and the Vite frontend dev server.

### Terminal 1 — Backend API

From the repository root:

```bash
python backend/app.py
```

This starts the Flask API server on **http://localhost:5000**. The backend handles all computation: constraint checking, diffusion model inference, training, benchmarking, and Monte Carlo simulations.

### Terminal 2 — Frontend Dev Server

From the `frontend/` directory:

```bash
cd frontend
npm run dev
```

This starts the Vite dev server on **http://localhost:3000**. The frontend proxies all `/api/*` requests to the backend at `localhost:5000` (configured in `vite.config.js`), so you only need to open **http://localhost:3000** in your browser.

### Stopping the Servers

Press `Ctrl+C` in each terminal to stop the servers. On Windows, you can also kill Node and Python processes:

```powershell
Get-Process -Name "node", "python" -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## What This Project Does

Generative models can propose **chemically invalid** molecular structures (wrong valency, broken mass conservation, charge mismatches). This project solves that by wrapping a small **GNN diffusion model** with a **supervisor loop** that enforces correctness at every step.

### The Pipeline

1. **Input** — Define reactant molecules (e.g., CH₃Br + OH⁻)
2. **Forward diffusion** — Add noise to destroy molecular structure
3. **Reverse diffusion** — The GNN denoises step by step, predicting atom features and bond orders
4. **Supervisor check** — At each step, the supervisor decodes the intermediate and verifies:
   - **Valency** — No atom exceeds its allowed bond count
   - **Mass conservation** — Total mass of products equals total mass of reactants (final step)
   - **Charge conservation** — Total charge is preserved (final step)
5. **Correction or backtrack** — If a step violates constraints, the supervisor attempts targeted fixes (relabel atoms, reduce bonds, adjust charges/hydrogens). If corrections fail, it backtracks to a previous valid state and re-samples.
6. **Output** — A chemically valid product molecule

### Constraint Verification

Constraints can be checked using two backends:
- **Z3 SMT solver** — Uses formal verification with `z3-solver`. More rigorous; produces solver-backed proofs.
- **Pure Python fallback** — Arithmetic checks without Z3. Faster; used automatically when Z3 is not installed.

Both produce identical results for the constraints checked (mass, charge, valency).

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                    Browser                         │
│         React + Three.js (port 3000)               │
│   8 interactive pages, 3D molecule viewer          │
└──────────────────────┬─────────────────────────────┘
                       │  /api/* (Vite proxy)
┌──────────────────────▼─────────────────────────────┐
│               Flask API (port 5000)                │
│   10 REST endpoints, JSON request/response         │
│   backend/app.py                                   │
└──────────────────────┬─────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────┐
│         Core Python Package                        │
│   src/chemistry_constraint_satisfaction/            │
│                                                    │
│   constraints/    Atom, MolecularState,             │
│                   check_reaction, check_intermediate│
│                   Z3 + pure-Python checkers         │
│                                                    │
│   diffusion/      MolecularDiffusionModel (NumPy), │
│                   Supervisor, Trainer,              │
│                   MolDiffusionNet (PyTorch)         │
└────────────────────────────────────────────────────┘
```

### Training Pipeline

The model uses a split architecture:
- **Training** uses PyTorch (`MolDiffusionNet` in `torch_model.py`) to train via gradient descent on a denoising objective
- **Inference** uses NumPy (`MolecularDiffusionModel` in `model.py`) — the trained PyTorch weights are exported into the NumPy model
- Trained weights are saved to `checkpoints/diffusion_weights.pt` and automatically loaded on backend startup

---

## Frontend — UI Pages

The frontend has **8 pages**, all accessible from the top navigation bar:

| Page | Tab Label | Description |
|------|-----------|-------------|
| **Overview** | Overview | Landing page with architecture summary, supported elements (Z=1–86), and navigation to all tools |
| **Molecule Lab** | Lab | Interactive 2D atom editor with live valency checking, Lipinski rule-of-five evaluation, and a 3D molecular structure preview (Three.js) |
| **Constraint Checker** | Constraints | Verify reactions against mass, charge, and valency constraints. Pick from preset reactions or define custom ones |
| **Supervisor** | Supervisor | Run step-by-step constrained diffusion. See the full timeline of commits, corrections, and backtracks. Scrub through intermediate molecular states |
| **Training** | Training | Train the GNN model via gradient descent (PyTorch Adam) or evolutionary seed search. View loss curves in real time |
| **Benchmark** | Benchmark | Side-by-side comparison of supervised vs. unsupervised molecular generation. Shows valency and conservation validity rates |
| **Simulation** | Simulation | Monte Carlo batch generation across many seeds with statistical analysis: validity rates, violation breakdowns, mass distributions, Lipinski pass rates |
| **Pathways** | Pathways | Multi-step reaction synthesis — define a chain of reactions where each step's product becomes the next step's reactant |

### 3D Molecule Viewer

The Lab and Supervisor pages include an interactive 3D molecular structure viewer built with **Three.js** (via `@react-three/fiber` and `@react-three/drei`). Atoms are rendered as color-coded spheres and bonds as cylinders.

---

## Backend — API Endpoints

The Flask backend (`backend/app.py`) exposes the following REST API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/info` | Server status: version, Z3 availability, supported elements, checkpoint metadata |
| GET | `/api/presets` | Preset molecules and reactions for quick testing |
| GET | `/api/noise-schedule?T=50` | Diffusion noise schedule (beta and alpha_bar values) for up to `T` steps |
| POST | `/api/check-reaction` | Check mass, charge, and valency constraints for a reaction |
| POST | `/api/check-intermediate` | Check valency constraints for a single molecule, plus Lipinski properties |
| POST | `/api/run-supervisor` | Run a full supervised diffusion generation with step log and intermediates |
| POST | `/api/benchmark` | Batch benchmark: supervised vs. unsupervised over `n` seeds |
| POST | `/api/train` | Evolutionary model selection: evaluate seeds over generations |
| POST | `/api/train-weights` | PyTorch gradient training: train the GNN and save a checkpoint |
| POST | `/api/monte-carlo` | Monte Carlo sampling: generate `n` molecules and compute statistics |
| POST | `/api/pathway` | Multi-step reaction pathway: chain reactions end-to-end |

All POST endpoints accept and return JSON. The proxy timeout is 5 minutes to accommodate training runs.

---

## Core Python Package

The installable package lives in `src/chemistry_constraint_satisfaction/` and can be used independently of the web app.

### Module Map

| Module | Key Exports | Role |
|--------|-------------|------|
| `constraints` | `Atom`, `MolecularState`, `check_reaction`, `check_intermediate`, `Z3_AVAILABLE` | Chemical data structures + constraint verification |
| `constraints.chemical_axioms` | `ATOMIC_MASS`, `MAX_VALENCY`, `ATOMIC_NUMBER`, `CHARGE_VALENCY_DELTA` | Element lookup tables (86 elements, H through Rn) |
| `diffusion.model` | `MolecularDiffusionModel`, `encode_molecule` | NumPy GNN inference model: forward noise, reverse denoising, encode/decode |
| `diffusion.supervisor` | `Supervisor`, `GenerationResult`, `StepRecord` | Supervisor loop: per-step checking, correction strategies, backtracking |
| `diffusion.trainer` | `train_diffusion_weights`, `train_and_export`, `save_checkpoint`, `load_checkpoint_into_numpy` | PyTorch training + checkpoint management |
| `diffusion.torch_model` | `MolDiffusionNet` | PyTorch GNN definition (used only for training) |

### Programmatic Usage

```python
from chemistry_constraint_satisfaction.constraints import (
    Atom, MolecularState, check_reaction,
)
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel, Supervisor,
)

# Define reactants
reactants = [
    MolecularState("CH3Br", [
        Atom("C", 4), Atom("Br", 1),
        Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("OH-", [
        Atom("O", 1, formal_charge=-1), Atom("H", 1),
    ]),
]

# Run supervised generation
model = MolecularDiffusionModel(hidden_dim=64, seed=42)
sup = Supervisor(model, reactants=reactants, T=20, verbose=True)
result = sup.run()
print(result.success, len(result.product.atoms))
```

---

## CLI Demo & Notebooks

### CLI Demo

Run the full demo (constraint checks + supervised generation + benchmark + train-then-generate):

```bash
python scripts/demo.py
```

The demo has 4 parts:
1. **Chemical axiom checks** — Valid/invalid reaction examples
2. **Supervisor loop** — One generation with step-by-step output
3. **Benchmark** — 50 runs comparing supervised vs. unsupervised
4. **Train-then-generate** — Train the GNN via gradient descent, then generate with learned weights and benchmark

### Jupyter Notebook

```bash
pip install jupyter
jupyter notebook notebooks/demo.ipynb
```

The notebook walks through the same pipeline interactively. For GPU-accelerated training, upload to **Google Colab** and enable `Runtime → GPU`.

---

## Tests

| Command | When to use |
|---------|-------------|
| `python run_tests.py` | No pytest installed; uses **unittest** only |
| `pytest tests/ -v` | After `pip install -e ".[dev]"` |

Test suites:
- `test_chemical_axioms.py` — Constraint checking (mass, charge, valency, Z3 vs. Python)
- `test_diffusion_model.py` — Encoding, decoding, forward/reverse diffusion
- `test_supervisor.py` — Supervisor loop, corrections, backtracking
- `test_train_and_generate.py` — Training pipeline, checkpoint save/load

---

## Repository Structure

```
ChemistryConstraintSatisfaction-2/
├── README.md                              ← This file
├── pyproject.toml                         ← Package metadata, editable install config
├── requirements.txt                       ← Python runtime dependencies
├── run_tests.py                           ← Run all tests (unittest, no pytest needed)
│
├── backend/
│   ├── app.py                             ← Flask API server (10 endpoints, port 5000)
│   └── requirements.txt                   ← Backend-specific deps (flask, flask-cors)
│
├── frontend/
│   ├── package.json                       ← Node dependencies and scripts
│   ├── vite.config.js                     ← Vite config (port 3000, API proxy to 5000)
│   ├── tailwind.config.js                 ← Tailwind CSS configuration
│   ├── index.html                         ← HTML entry point
│   └── src/
│       ├── App.jsx                        ← Root component with tab navigation
│       ├── api.js                         ← API client (fetch wrapper for all endpoints)
│       ├── main.jsx                       ← React entry point
│       ├── index.css                      ← Global styles
│       ├── pages/                         ← 8 page components (Overview, Lab, etc.)
│       ├── components/                    ← Shared components (3D viewer, badges, etc.)
│       ├── data/                          ← Static element data for the periodic table
│       └── utils/                         ← Frontend utility functions
│
├── src/chemistry_constraint_satisfaction/ ← Core Python package
│   ├── __init__.py                        ← Package version
│   ├── constraints/
│   │   ├── __init__.py                    ← Public API: Atom, MolecularState, check_*
│   │   └── chemical_axioms.py             ← Element tables, data classes, Z3/Python checkers
│   └── diffusion/
│       ├── __init__.py                    ← Public API: MolecularDiffusionModel, Supervisor
│       ├── model.py                       ← NumPy GNN (inference only, no PyTorch needed)
│       ├── supervisor.py                  ← Supervisor loop + correction strategies
│       ├── torch_model.py                 ← PyTorch GNN definition (training only)
│       └── trainer.py                     ← Training loop, checkpoint I/O, weight export
│
├── checkpoints/
│   └── diffusion_weights.pt              ← Trained model weights (auto-loaded by backend)
│
├── tests/                                 ← pytest / unittest suites
│   ├── test_chemical_axioms.py
│   ├── test_diffusion_model.py
│   ├── test_supervisor.py
│   └── test_train_and_generate.py
│
├── scripts/
│   ├── check_env.py                       ← Quick environment sanity check
│   └── demo.py                            ← End-to-end CLI demo + benchmark
│
└── notebooks/
    └── demo.ipynb                         ← Interactive Jupyter walkthrough
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Constraint solver** | Z3 (`z3-solver`) | Formal verification of chemical axioms |
| **Constraint fallback** | Pure Python | Arithmetic checks when Z3 is unavailable |
| **Diffusion model (inference)** | NumPy | Fast, dependency-light GNN inference |
| **Diffusion model (training)** | PyTorch | Gradient-based training with Adam optimizer |
| **Backend API** | Flask + flask-cors | REST API for all computation |
| **Frontend framework** | React 18 | Component-based UI |
| **Build tool** | Vite 5 | Frontend dev server + bundler |
| **3D rendering** | Three.js (`@react-three/fiber`, `@react-three/drei`) | Interactive molecule visualization |
| **Styling** | Tailwind CSS 3 | Utility-first CSS framework |
| **Icons** | Lucide React | UI icons |

---

## Design Notes & Trade-offs

- **NumPy for inference, PyTorch for training** — The model is trained with PyTorch but exported to NumPy for inference. This means the supervisor loop has zero PyTorch dependency at runtime, making it lightweight and fast.
- **Z3 is optional** — If `z3-solver` is installed, constraint checks can use the SMT solver for formal verification. Otherwise, the same checks run in pure Python. Both produce identical results for the axioms checked.
- **Supervisor corrections are conservative** — The supervisor fixes valency violations by capping bonds, relabels atoms to match the reactant element composition, and adjusts charge/mass through implicit hydrogen. Only targeted corrections are applied, not arbitrary graph surgery.
- **Backtracking has limits** — `max_backtracks` prevents infinite loops. If corrections and re-sampling both fail, the supervisor commits the best-effort result and flags the violation.
- **All pages stay mounted** — The frontend keeps all 8 pages mounted (via `display: none/block`) so that results, charts, and form state persist when switching tabs.
- **Checkpoint auto-load** — On backend startup, the Flask API automatically loads `checkpoints/diffusion_weights.pt` (if it exists) into the NumPy inference model. Training via the web UI saves new checkpoints to the same location.
- **5-minute proxy timeout** — The Vite proxy timeout is set to 300 seconds because gradient training and large Monte Carlo simulations can take minutes to complete.

---

## Contributing & License

- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **License:** Use and cite as appropriate for your institution. This project builds on open-source tools (PyTorch, Z3, React, Three.js).
