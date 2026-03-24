"""
Gradient-based training for MolDiffusionNet and export to NumPy MolecularDiffusionModel.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..constraints.chemical_axioms import Atom, MolecularState
from .model import (
    ATOM_FEAT_DIM,
    MolecularDiffusionModel,
    NUM_ELEM,
    encode_molecule,
)
from .torch_model import MolDiffusionNet, add_noise_to_adj, add_noise_to_x


def _torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_preset_molecule(data: dict) -> MolecularState:
    atoms = [
        Atom(
            element=a["element"],
            bonds=a["bonds"],
            formal_charge=a.get("formal_charge", 0),
            implicit_h=a.get("implicit_h", 0),
        )
        for a in data["atoms"]
    ]
    return MolecularState(name=data.get("name", "mol"), atoms=atoms)


def build_training_molecules_from_presets(presets: dict) -> List[MolecularState]:
    """Collect unique training graphs from preset molecules and reaction products."""
    seen: set[str] = set()
    out: List[MolecularState] = []

    def add(mol: MolecularState):
        key = mol.name + str([(a.element, a.bonds, a.formal_charge) for a in mol.atoms])
        if key not in seen:
            seen.add(key)
            out.append(mol)

    for m in presets.get("molecules", []):
        add(_parse_preset_molecule(m))
    for rxn in presets.get("reactions", []):
        for p in rxn.get("products", []):
            add(_parse_preset_molecule(p))
    return out


def export_torch_to_numpy(
    net: MolDiffusionNet, npm: MolecularDiffusionModel
) -> None:
    """Copy weights from trained PyTorch module into NumPy MolecularDiffusionModel."""
    sd = net.state_dict()

    def lin_to_numpy(prefix: str, np_lin):
        w = sd[f"{prefix}.weight"].detach().cpu().numpy()
        b = sd[f"{prefix}.bias"].detach().cpu().numpy()
        np_lin.W = w.astype(np.float32)
        np_lin.b = b.astype(np.float32)

    lin_to_numpy("gc1.w_self", npm.gc1.W_self)
    lin_to_numpy("gc1.w_neigh", npm.gc1.W_neigh)
    lin_to_numpy("gc2.w_self", npm.gc2.W_self)
    lin_to_numpy("gc2.w_neigh", npm.gc2.W_neigh)
    lin_to_numpy("atom_head", npm.atom_head)
    lin_to_numpy("bond_head", npm.bond_head)


def load_checkpoint_into_numpy(
    npm: MolecularDiffusionModel, path: str, map_location: str | None = None
) -> bool:
    """Load a saved checkpoint into an existing NumPy model. Returns False if missing."""
    if not os.path.isfile(path):
        return False
    loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _torch_load(path, map_location=loc)
    hidden = ckpt.get("hidden_dim", npm.hidden_dim)
    if hidden != npm.hidden_dim:
        return False
    net = MolDiffusionNet(hidden_dim=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    export_torch_to_numpy(net, npm)
    return True


def checkpoint_meta(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        ckpt = _torch_load(path, map_location="cpu")
        return {
            "hidden_dim": ckpt.get("hidden_dim"),
            "epochs_trained": ckpt.get("epochs"),
            "n_molecules": ckpt.get("n_molecules"),
        }
    except Exception:
        return None


def train_diffusion_weights(
    molecules: List[MolecularState],
    hidden_dim: int = 64,
    T: int = 20,
    epochs: int = 40,
    lr: float = 1e-3,
    steps_per_epoch: int = 80,
    device: torch.device | None = None,
    seed: int = 42,
) -> Tuple[List[float], MolDiffusionNet]:
    """
    Denoising objective: predict x0 (atom features + bond orders) from noisy inputs.
    """
    if not molecules:
        raise ValueError("No molecules for training")
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    net = MolDiffusionNet(hidden_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for mol in molecules:
        x0, adj0 = encode_molecule(mol)
        graphs.append(
            (
                torch.tensor(x0, dtype=torch.float32, device=dev),
                torch.tensor(adj0, dtype=torch.float32, device=dev),
            )
        )

    loss_history: List[float] = []
    net.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            x0, adj0 = random.choice(graphs)
            n = x0.shape[0]
            t = random.randint(1, T)
            x_t, _ = add_noise_to_x(x0, t, T)
            adj_t = add_noise_to_adj(adj0, t, T)

            atom_logits, bond_logits = net(x_t, adj_t)

            elem_target = x0[:, :NUM_ELEM].argmax(dim=-1).clamp(0, NUM_ELEM - 1)
            loss_elem = F.cross_entropy(atom_logits[:, :NUM_ELEM], elem_target)
            loss_cont = F.mse_loss(atom_logits[:, NUM_ELEM:], x0[:, NUM_ELEM:])

            bond_targets = adj0.long().clamp(0, 3)
            edges = 0
            loss_bond = torch.tensor(0.0, device=dev)
            for i in range(n):
                for j in range(i + 1, n):
                    loss_bond = loss_bond + F.cross_entropy(bond_logits[i, j], bond_targets[i, j])
                    edges += 1
            if edges > 0:
                loss_bond = loss_bond / edges

            loss = loss_elem + loss_cont + 0.5 * loss_bond

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        loss_history.append(ep_loss / max(1, steps_per_epoch))

    net.eval()
    return loss_history, net


def save_checkpoint(
    net: MolDiffusionNet, path: str, extra: Optional[Dict[str, Any]] = None
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "hidden_dim": net.hidden_dim,
        "atom_feat_dim": ATOM_FEAT_DIM,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def default_checkpoint_path() -> str:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints")
    )
    return os.path.join(root, "diffusion_weights.pt")
