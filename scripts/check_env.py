#!/usr/bin/env python3
"""Quick check that the environment has PyTorch, Z3, and the project package."""
import sys
from pathlib import Path

# Allow importing the package when run from repo root
repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

def main():
    errors = []
    print("Checking Chemistry Constraint Satisfaction environment...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")

    try:
        import z3
        print(f"  Z3:     {z3.get_version_string()}")
    except ImportError as e:
        errors.append(f"Z3: {e}")

    try:
        import chemistry_constraint_satisfaction as ccs
        print(f"  Package: {ccs.__version__}")
    except ImportError as e:
        errors.append(f"Package: {e}")

    if errors:
        print("FAILED:", "; ".join(errors))
        sys.exit(1)
    print("OK — environment ready.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
