"""
Molecular diffusion model and supervisor loop.
Each step is verified and optionally corrected by the constraint layer.
"""

from .model import (
    MolecularDiffusionModel,
    encode_molecule,
    atom_to_feat,
    feat_to_atom,
    ATOM_FEAT_DIM,
    NUM_ELEM,
    ELEMENTS,
)
from .supervisor import Supervisor, GenerationResult, StepRecord

__all__ = [
    "MolecularDiffusionModel",
    "encode_molecule",
    "atom_to_feat",
    "feat_to_atom",
    "ATOM_FEAT_DIM",
    "NUM_ELEM",
    "ELEMENTS",
    "Supervisor",
    "GenerationResult",
    "StepRecord",
]
