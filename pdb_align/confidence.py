"""Ingestion of predicted-model confidence files (PAE / pLDDT / ipTM) and
PAE-based interface scoring (pDockQ2).

Supported confidence formats (auto-detected by keys):

- **AF2 / ColabFold** scores JSON: ``pae`` or ``predicted_aligned_error``,
  ``plddt``, ``ptm``, ``iptm``, ``max_pae``.
- **AlphaFold 3** ``*_summary_confidences.json`` (``iptm``, ``ptm``,
  ``chain_pair_iptm``, ``chain_iptm``, ``ranking_score``) and
  ``*_confidences.json`` (``pae``, ``token_chain_ids``).
- **Boltz** ``confidence_*.json`` (``iptm``, ``ptm``, ``pair_chains_iptm``,
  ``complex_plddt``) and ``pae_*.npz``.
- Bare ``.npy`` / ``.npz`` PAE matrices (key ``pae`` or
  ``predicted_aligned_error``, or a single array).

pDockQ2 (Zhu, Shenoy, Kundrotas & Elofsson, Bioinformatics 2023, 39:btad424;
constants cross-checked against the reference implementation in
gitlab.com/ElofssonLab/afm-benchmark src/pdockq2.py):

    X = < 1 / (1 + (PAE_contact / d0)^2) > * <pLDDT>_interface,  d0 = 10 A
    pDockQ2 = L / (1 + exp(-k (X - x0))) + b
    L = 1.31034849, x0 = 84.7326239, k = 0.0747157696, b = 0.00501886443

Contacts are CB-CB pairs (CA for glycine) within 8 A between the two chain
groups (the pDockQ convention); PAE is taken at the contact residue pairs and
<pLDDT> over the combined unique interface residues. pDockQ2 is direction-
dependent (PAE is asymmetric); both directions are reported along with their
mean.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# pDockQ2 sigmoid constants (Zhu et al. 2023, reference implementation)
PDOCKQ2_L = 1.31034849
PDOCKQ2_X0 = 84.7326239
PDOCKQ2_K = 0.0747157696
PDOCKQ2_B = 0.00501886443
PDOCKQ2_D0 = 10.0          # A, PAE scaling
PDOCKQ2_CUTOFF = 8.0       # A, CB-CB (CA for Gly) contact cutoff


@dataclass
class ModelConfidence:
    """Parsed confidence data for one predicted model."""
    sources: List[str] = field(default_factory=list)
    format: str = "unknown"           # af2/colabfold | af3 | boltz | pae-array | unknown
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    ranking_score: Optional[float] = None
    pae: Optional[np.ndarray] = None              # (N, N)
    plddt: Optional[np.ndarray] = None            # per-residue, if present
    chain_pair_iptm: Optional[np.ndarray] = None  # (C, C) when available
    chain_ids: Optional[List[str]] = None         # order of chains for the above
    token_chain_ids: Optional[List[str]] = None   # AF3 per-token chain ids

    def to_dict(self) -> dict:
        return {
            "sources": [os.path.basename(s) for s in self.sources],
            "format": self.format,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "ranking_score": self.ranking_score,
            "has_pae": self.pae is not None,
            "pae_shape": list(self.pae.shape) if self.pae is not None else None,
        }


def _merge(dst: ModelConfidence, src: ModelConfidence) -> ModelConfidence:
    for name in ("iptm", "ptm", "ranking_score", "pae", "plddt",
                 "chain_pair_iptm", "chain_ids", "token_chain_ids"):
        if getattr(dst, name) is None and getattr(src, name) is not None:
            setattr(dst, name, getattr(src, name))
    dst.sources.extend(src.sources)
    if dst.format == "unknown":
        dst.format = src.format
    return dst


def _pae_from_json(d: dict) -> Optional[np.ndarray]:
    for key in ("pae", "predicted_aligned_error"):
        if key in d and d[key] is not None:
            arr = np.asarray(d[key], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr
    return None


def _parse_json(path: str) -> ModelConfidence:
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict):
        d = d[0]  # AF2 pae json is sometimes a 1-element list
    if not isinstance(d, dict):
        raise ValueError(f"Unrecognized confidence JSON structure in {path}")

    c = ModelConfidence(sources=[path])
    c.pae = _pae_from_json(d)
    if "plddt" in d and isinstance(d["plddt"], (list, tuple)):
        c.plddt = np.asarray(d["plddt"], dtype=float)

    def _num(key):
        v = d.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    c.iptm = _num("iptm")
    c.ptm = _num("ptm")
    c.ranking_score = _num("ranking_score")

    if "token_chain_ids" in d:
        c.format = "af3"
        c.token_chain_ids = [str(x) for x in d["token_chain_ids"]]
    if "chain_pair_iptm" in d and d["chain_pair_iptm"] is not None:
        c.format = "af3"
        c.chain_pair_iptm = np.asarray(d["chain_pair_iptm"], dtype=float)
    if "pair_chains_iptm" in d and d["pair_chains_iptm"] is not None:
        # Boltz: {"0": {"0": v, "1": v}, ...}
        c.format = "boltz"
        pc = d["pair_chains_iptm"]
        keys = sorted(pc.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        mat = np.array([[float(pc[a][b]) for b in keys] for a in keys])
        c.chain_pair_iptm = mat
        c.chain_ids = [str(k) for k in keys]
    if c.format == "unknown":
        if "max_pae" in d or "plddt" in d or c.pae is not None:
            c.format = "af2/colabfold"
    return c


def _parse_array(path: str) -> ModelConfidence:
    c = ModelConfidence(sources=[path], format="pae-array")
    if path.endswith(".npz"):
        with np.load(path) as z:
            for key in ("pae", "predicted_aligned_error"):
                if key in z:
                    c.pae = np.asarray(z[key], dtype=float)
                    break
            else:
                names = list(z.keys())
                if len(names) == 1:
                    c.pae = np.asarray(z[names[0]], dtype=float)
    else:
        c.pae = np.asarray(np.load(path), dtype=float)
    if c.pae is not None and (c.pae.ndim != 2 or c.pae.shape[0] != c.pae.shape[1]):
        raise ValueError(f"{path}: expected a square PAE matrix, got shape {c.pae.shape}")
    return c


def load_confidence(paths: Union[str, Sequence[str]]) -> ModelConfidence:
    """Parse one or several confidence files into a single ModelConfidence.

    Multiple files are merged (e.g. AF3's summary JSON for ipTM plus the full
    confidences JSON for PAE); the first file providing a field wins.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = ModelConfidence()
    for p in paths:
        p = str(p)
        if p.endswith((".npz", ".npy")):
            out = _merge(out, _parse_array(p))
        else:
            out = _merge(out, _parse_json(p))
    return out


def find_confidence_files(model_path: str) -> List[str]:
    """Best-effort discovery of confidence files next to a model file.

    Checks the naming conventions of AF3 (``*_summary_confidences.json`` +
    ``*_confidences.json``), Boltz (``confidence_<stem>.json`` +
    ``pae_<stem>.npz``), ColabFold (``..._scores_rank...json`` sibling of an
    ``..._unrelaxed_rank...`` model), and plain ``<stem>.json`` /
    ``<stem>_scores.json`` / ``<stem>_pae.json``.
    """
    model_path = str(model_path)
    d = os.path.dirname(model_path) or "."
    stem = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        f"{stem}_summary_confidences.json",
        f"{stem}_confidences.json",
        f"confidence_{stem}.json",
        f"pae_{stem}.npz",
        f"{stem}_scores.json",
        f"{stem}_pae.json",
        f"{stem}.json",
    ]
    if stem.endswith("_model"):
        base = stem[: -len("_model")]
        candidates += [f"{base}_summary_confidences.json", f"{base}_confidences.json"]
    for tag in ("_unrelaxed_", "_relaxed_"):
        if tag in stem:
            candidates.append(stem.replace(tag, "_scores_") + ".json")
    seen, found = set(), []
    for c in candidates:
        p = os.path.join(d, c)
        if p not in seen and os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(model_path):
            seen.add(p)
            found.append(p)
    return found


# ---------------------------------------------------------------------------
# pDockQ2
# ---------------------------------------------------------------------------

@dataclass
class PDockQ2Result:
    pdockq2: float                  # mean of the two directions
    pdockq2_ab: float               # PAE rows = group A, cols = group B
    pdockq2_ba: float
    n_contacts: int
    mean_interface_plddt: float
    mean_interface_pae: float

    def to_dict(self) -> dict:
        return {
            "pdockq2": round(self.pdockq2, 4),
            "pdockq2_ab": round(self.pdockq2_ab, 4),
            "pdockq2_ba": round(self.pdockq2_ba, 4),
            "n_contacts": self.n_contacts,
            "mean_interface_plddt": round(self.mean_interface_plddt, 2),
            "mean_interface_pae": round(self.mean_interface_pae, 2),
        }


def _pdockq2_sigmoid(x: float) -> float:
    return PDOCKQ2_L / (1.0 + math.exp(-PDOCKQ2_K * (x - PDOCKQ2_X0))) + PDOCKQ2_B


def _global_residue_indices(struct, chain_names):
    """Map requested chains to global residue indices in file order.

    The PAE matrix of AF2/AF3/Boltz is indexed by residues/tokens in the
    order they appear in the model file, so the index of a residue is its
    position in the concatenation of all chains' protein residues.
    Returns ({chain: [global_idx, ...]}, total_residue_count).
    """
    from .interface import _chain_residues
    model = struct[0]
    all_chains = [ch.name for ch in model]
    res_by_chain = _chain_residues(struct, all_chains)
    idx_map: Dict[str, List[int]] = {}
    counter = 0
    for cname in all_chains:
        n = len(res_by_chain[cname])
        if cname in chain_names:
            idx_map[cname] = list(range(counter, counter + n))
        counter += n
    return idx_map, counter, res_by_chain


def compute_pdockq2(
    structure,
    chains_a: Sequence[str],
    chains_b: Sequence[str],
    pae: Union[np.ndarray, ModelConfidence],
    cutoff: float = PDOCKQ2_CUTOFF,
    d0: float = PDOCKQ2_D0,
) -> PDockQ2Result:
    """
    pDockQ2 interface confidence (Zhu et al. 2023) for a predicted complex.

    Requires the model's PAE matrix (pass the array or a ModelConfidence) and
    pLDDT in the structure's B-factor column. The PAE matrix must be indexed
    by residues in file order; when it is not the same size as the number of
    protein residues, a ValueError is raised rather than guessing.
    """
    from .interface import _as_structure
    if isinstance(pae, ModelConfidence):
        if pae.pae is None:
            raise ValueError("ModelConfidence carries no PAE matrix.")
        pae = pae.pae
    pae = np.asarray(pae, dtype=float)

    struct = _as_structure(structure)
    idx_map, n_total, res_by_chain = _global_residue_indices(
        struct, list(chains_a) + list(chains_b))
    if pae.shape[0] != n_total:
        raise ValueError(
            f"PAE matrix is {pae.shape[0]}x{pae.shape[1]} but the model has "
            f"{n_total} protein residues; cannot map PAE to residues. "
            "(Non-protein tokens, e.g. ligands/nucleic acids, are not supported.)")

    res_a = [(r, g) for c in chains_a for r, g in zip(res_by_chain[c], idx_map[c])]
    res_b = [(r, g) for c in chains_b for r, g in zip(res_by_chain[c], idx_map[c])]

    A = np.array([r.cb_or_ca() for r, _ in res_a])
    B = np.array([r.cb_or_ca() for r, _ in res_b])
    from scipy.spatial import cKDTree
    neighbors = cKDTree(A).query_ball_tree(cKDTree(B), cutoff)

    pae_ab, pae_ba = [], []
    ia, ib = set(), set()
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            ga, gb = res_a[i][1], res_b[j][1]
            pae_ab.append(pae[ga, gb])
            pae_ba.append(pae[gb, ga])
            ia.add(i)
            ib.add(j)
    n_contacts = len(pae_ab)
    if n_contacts == 0:
        return PDockQ2Result(0.0, 0.0, 0.0, 0, 0.0, 0.0)

    plddts = ([res_a[i][0].plddt for i in ia if res_a[i][0].plddt is not None]
              + [res_b[j][0].plddt for j in ib if res_b[j][0].plddt is not None])
    if not plddts:
        raise ValueError("No pLDDT (B-factor) values available for the interface.")
    mean_plddt = float(np.mean(plddts))
    if mean_plddt <= 1.0 or mean_plddt > 100.0:
        warnings.warn(
            f"Mean interface B-factor {mean_plddt:.2f} does not look like pLDDT; "
            "pDockQ2 is only meaningful for predicted models.",
            UserWarning, stacklevel=2)

    def _x(vals):
        return float(np.mean(1.0 / (1.0 + (np.asarray(vals) / d0) ** 2))) * mean_plddt

    p_ab = _pdockq2_sigmoid(_x(pae_ab))
    p_ba = _pdockq2_sigmoid(_x(pae_ba))
    return PDockQ2Result(
        pdockq2=float((p_ab + p_ba) / 2.0), pdockq2_ab=float(p_ab),
        pdockq2_ba=float(p_ba), n_contacts=n_contacts,
        mean_interface_plddt=mean_plddt,
        mean_interface_pae=float(np.mean(pae_ab + pae_ba)))
