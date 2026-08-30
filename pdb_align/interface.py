"""Interface quality metrics for protein complexes.

Implements the DockQ family of metrics against their published definitions:

- **DockQ** (Basu & Wallner, PLoS ONE 2016; definitions cross-checked against
  the reference implementation, github.com/bjornwallner/DockQ):
  fnat contacts are residue pairs across the interface with any heavy-atom
  pair < 5 A; interface residues for iRMSD are defined by any heavy-atom
  pair < 10 A *in the native*; iRMSD and LRMSD are computed over backbone
  atoms (N, CA, C, O); LRMSD after superposing on the receptor backbone;
  DockQ = (fnat + 1/(1+(iRMSD/1.5)^2) + 1/(1+(LRMSD/8.5)^2)) / 3, with CAPRI
  classes incorrect < 0.23 <= acceptable < 0.49 <= medium < 0.80 <= high.

- **Epitope/paratope site comparison** for immune complexes: the native
  epitope is the set of antigen residues with any heavy atom within 4.5 A of
  the antibody in the reference; precision/recall/F1 of the model's epitope
  against it separate "right epitope, imperfect pose" from "wrong face of
  the antigen", which a single DockQ number conflates.

- **pDockQ** (Bryant, Pozzati & Elofsson, Nat Commun 2022; constants from the
  reference implementation, FoldDock src/pdockq.py): reference-free interface
  confidence for predicted complexes. Contacts are CB-CB pairs (CA for Gly)
  <= 8 A between the two chain groups; x = <interface pLDDT> * log10(n_contacts);
  pDockQ = 0.724 / (1 + exp(-0.052*(x - 152.611))) + 0.018.

Residue correspondence between native and model is established per chain pair
by BLOSUM62 sequence alignment, so mismatched author numbering, expression
tags, or unmodeled loops do not silently mis-pair residues.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import gemmi

from .core import (
    AA_DICT, _kabsch, perform_sequence_alignment,
)

# --- published constants (see module docstring for sources) -----------------
FNAT_CONTACT_CUTOFF = 5.0      # A, any heavy-atom pair (DockQ)
INTERFACE_CUTOFF = 10.0        # A, native heavy-atom cutoff defining iRMSD set
DOCKQ_D1_IRMSD = 1.5           # A, iRMSD scaling in the DockQ formula
DOCKQ_D2_LRMSD = 8.5           # A, LRMSD scaling in the DockQ formula
EPITOPE_CUTOFF = 4.5           # A, heavy-atom cutoff for epitope/paratope sets
BACKBONE_ATOMS = ("N", "CA", "C", "O")
PDOCKQ_CUTOFF = 8.0            # A, CB-CB (CA for Gly) contact cutoff
_PDOCKQ_L, _PDOCKQ_X0, _PDOCKQ_K, _PDOCKQ_B = 0.724, 152.611, 0.052, 0.018
_MAX_MAPPING_PERMUTATIONS = 720  # cap on the homomultimer fnat search


def capri_class(dockq: float) -> str:
    """CAPRI quality class for a DockQ value (Basu & Wallner 2016)."""
    if dockq >= 0.80:
        return "high"
    if dockq >= 0.49:
        return "medium"
    if dockq >= 0.23:
        return "acceptable"
    return "incorrect"


# --------------------------------------------------------------------------
# structure access helpers
# --------------------------------------------------------------------------

@dataclass
class _Res:
    chain: str
    seqid: int
    icode: str
    name: str
    letter: str
    heavy: Dict[str, np.ndarray]
    plddt: Optional[float]

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain, self.seqid, self.icode)

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.seqid}{self.icode}"

    def cb_or_ca(self) -> Optional[np.ndarray]:
        if self.name == "GLY":
            return self.heavy.get("CA")
        return self.heavy.get("CB", self.heavy.get("CA"))


def _as_structure(x: Union[str, gemmi.Structure]) -> gemmi.Structure:
    if isinstance(x, gemmi.Structure):
        return x
    return gemmi.read_structure(str(x))


def _chain_residues(struct: gemmi.Structure, chain_names: Sequence[str]) -> Dict[str, List[_Res]]:
    """Protein residues (CA-bearing, heavy atoms only) per requested chain."""
    model = struct[0]
    available = {ch.name for ch in model}
    missing = [c for c in chain_names if c not in available]
    if missing:
        raise ValueError(f"Chain(s) {missing} not found; available: {sorted(available)}")
    out: Dict[str, List[_Res]] = {c: [] for c in chain_names}
    for chain in model:
        if chain.name not in out:
            continue
        for res in chain:
            if res.name not in AA_DICT:
                continue
            heavy = {}
            plddt = None
            for atom in res:
                if atom.element.name == "H":
                    continue
                if atom.name not in heavy:  # first altloc wins
                    heavy[atom.name] = np.array(atom.pos.tolist(), dtype=float)
                if atom.name == "CA" and plddt is None:
                    plddt = float(atom.b_iso)
            if "CA" not in heavy:
                continue
            icode = res.seqid.icode if res.seqid.icode and res.seqid.icode != " " else ""
            out[chain.name].append(_Res(
                chain=chain.name, seqid=int(res.seqid.num), icode=icode.strip(),
                name=res.name, letter=AA_DICT[res.name], heavy=heavy, plddt=plddt))
    return out


def _pair_residues(ref_res: List[_Res], mob_res: List[_Res],
                   _cache: Optional[dict] = None) -> List[Tuple[int, int]]:
    """Index pairs (ref_i, mob_j) matched by BLOSUM62 sequence alignment.

    ``_cache`` (keyed by the two chain ids) avoids re-aligning the same chain
    pair during the homomultimer permutation search."""
    if not ref_res or not mob_res:
        return []
    if _cache is not None:
        ck = (ref_res[0].chain, mob_res[0].chain)
        if ck in _cache:
            return _cache[ck]
    seq_a = "".join(r.letter for r in ref_res)
    seq_b = "".join(r.letter for r in mob_res)
    aln = perform_sequence_alignment(seq_a, seq_b, -10.0, -0.5)
    if aln is None:
        return []
    pairs = []
    i = j = 0
    for a, b in zip(aln.seqA, aln.seqB):
        if a != "-" and b != "-":
            pairs.append((i, j))
        if a != "-":
            i += 1
        if b != "-":
            j += 1
    if _cache is not None:
        _cache[(ref_res[0].chain, mob_res[0].chain)] = pairs
    return pairs


def _match_chain_groups(ref_struct, model_struct, ref_chains, model_chains):
    """1:1 ref->model chain mapping within a group via Hungarian on % identity."""
    from .core import extract_sequences_and_lengths
    from .chains import match_chains
    ref_seqs, _ = extract_sequences_and_lengths(ref_struct, "ref")
    mob_seqs, _ = extract_sequences_and_lengths(model_struct, "model")
    mapping = match_chains(ref_seqs, mob_seqs, ref_struct, model_struct,
                           list(ref_chains), list(model_chains))
    return [(a, b) for a, b, *_ in mapping.pairs]


def _residue_contacts(res_a: List[_Res], res_b: List[_Res], cutoff: float):
    """Set of (key_a, key_b) residue pairs with any heavy-atom pair < cutoff."""
    from scipy.spatial import cKDTree
    if not res_a or not res_b:
        return set()
    coords_a, idx_a = [], []
    for i, r in enumerate(res_a):
        for c in r.heavy.values():
            coords_a.append(c)
            idx_a.append(i)
    coords_b, idx_b = [], []
    for j, r in enumerate(res_b):
        for c in r.heavy.values():
            coords_b.append(c)
            idx_b.append(j)
    tree_a = cKDTree(np.asarray(coords_a))
    tree_b = cKDTree(np.asarray(coords_b))
    contacts = set()
    for ia, neighbors in enumerate(tree_a.query_ball_tree(tree_b, cutoff)):
        if neighbors:
            ra = res_a[idx_a[ia]]
            for ib in neighbors:
                contacts.add((ra.key, res_b[idx_b[ib]].key))
    return contacts


# --------------------------------------------------------------------------
# DockQ
# --------------------------------------------------------------------------

@dataclass
class DockQResult:
    dockq: float
    capri: str
    fnat: float
    fnonnat: float
    irmsd: float
    lrmsd: float
    n_native_contacts: int
    n_model_contacts: int
    n_interface_residues: int
    receptor_mapping: List[Tuple[str, str]]
    ligand_mapping: List[Tuple[str, str]]
    # ligand-side residues of the 10 A native interface (NOT the 4.5 A
    # epitope — use epitope_metrics for that)
    interface_ligand_residues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dockq": round(self.dockq, 4),
            "capri": self.capri,
            "fnat": round(self.fnat, 4),
            "fnonnat": round(self.fnonnat, 4),
            "irmsd": round(self.irmsd, 3),
            "lrmsd": round(self.lrmsd, 3),
            "n_native_contacts": self.n_native_contacts,
            "n_model_contacts": self.n_model_contacts,
            "n_interface_residues": self.n_interface_residues,
            "receptor_mapping": [f"{a}->{b}" for a, b in self.receptor_mapping],
            "ligand_mapping": [f"{a}->{b}" for a, b in self.ligand_mapping],
        }

    def report(self) -> str:
        lines = [
            f"DockQ  : {self.dockq:.3f}  ({self.capri})",
            f"fnat   : {self.fnat:.3f}   (fnonnat {self.fnonnat:.3f}; "
            f"{self.n_native_contacts} native / {self.n_model_contacts} model contacts)",
            f"iRMSD  : {self.irmsd:.3f} A  ({self.n_interface_residues} interface residues)",
            f"LRMSD  : {self.lrmsd:.3f} A",
            "Receptor " + ", ".join(f"{a}->{b}" for a, b in self.receptor_mapping)
            + " | Ligand " + ", ".join(f"{a}->{b}" for a, b in self.ligand_mapping),
        ]
        return "\n".join(lines)


def dockq_formula(fnat: float, irmsd: float, lrmsd: float) -> float:
    """DockQ = (fnat + scaled iRMSD + scaled LRMSD)/3 (Basu & Wallner 2016)."""
    return (fnat
            + 1.0 / (1.0 + (irmsd / DOCKQ_D1_IRMSD) ** 2)
            + 1.0 / (1.0 + (lrmsd / DOCKQ_D2_LRMSD) ** 2)) / 3.0


class _GroupCorrespondence:
    """Residue correspondence for one chain group (native <-> model)."""

    def __init__(self, ref_res_by_chain, model_res_by_chain, chain_pairs,
                 pair_cache: Optional[dict] = None):
        self.chain_pairs = chain_pairs
        self.ref_res: List[_Res] = []
        self.model_for_ref: Dict[tuple, _Res] = {}   # ref key -> model _Res
        self.ref_for_model: Dict[tuple, tuple] = {}  # model key -> ref key
        self.model_res_all: List[_Res] = []
        for rc, mc in chain_pairs:
            r_list = ref_res_by_chain[rc]
            m_list = model_res_by_chain[mc]
            self.model_res_all.extend(m_list)
            for i, j in _pair_residues(r_list, m_list, _cache=pair_cache):
                self.model_for_ref[r_list[i].key] = m_list[j]
                self.ref_for_model[m_list[j].key] = r_list[i].key
            self.ref_res.extend(r_list)


def _collect_backbone(pairs: List[Tuple[_Res, _Res]]):
    """Matched backbone coords (native, model) for (ref, model) residue pairs."""
    P, Q = [], []
    for r_ref, r_mod in pairs:
        for name in BACKBONE_ATOMS:
            if name in r_ref.heavy and name in r_mod.heavy:
                P.append(r_ref.heavy[name])
                Q.append(r_mod.heavy[name])
    if not P:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(P), np.vstack(Q)


def compute_dockq(
    reference: Union[str, gemmi.Structure],
    model: Union[str, gemmi.Structure],
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    model_receptor_chains: Optional[Sequence[str]] = None,
    model_ligand_chains: Optional[Sequence[str]] = None,
) -> DockQResult:
    """
    DockQ of a model complex against a reference (native) complex.

    The receptor/ligand split follows the DockQ convention: the two sides of
    the interface being scored. Multi-chain sides (e.g. antibody H+L as the
    receptor) are treated as one merged unit, which is the standard way to
    score antibody-antigen complexes. Chains are given as *reference* chain
    names; the model's chains are matched by sequence (Hungarian assignment)
    unless given explicitly. For groups containing sequence-identical chains
    the assignment is refined by maximizing fnat over their permutations
    (up to a bounded number), which handles symmetric homomultimers.

    Raises ValueError when the receptor and ligand do not form an interface
    in the reference structure.
    """
    ref_struct = _as_structure(reference)
    model_struct = _as_structure(model)

    ref_rec = _chain_residues(ref_struct, receptor_chains)
    ref_lig = _chain_residues(ref_struct, ligand_chains)

    if model_receptor_chains is None or model_ligand_chains is None:
        model_chain_names = [ch.name for ch in model_struct[0]]
        rec_pairs = _match_chain_groups(ref_struct, model_struct,
                                        receptor_chains, model_chain_names)
        used = {b for _, b in rec_pairs}
        lig_candidates = [c for c in model_chain_names if c not in used]
        lig_pairs = _match_chain_groups(ref_struct, model_struct,
                                        ligand_chains, lig_candidates)
    else:
        if len(model_receptor_chains) != len(receptor_chains) or \
           len(model_ligand_chains) != len(ligand_chains):
            raise ValueError("model chain lists must match reference chain lists in length")
        rec_pairs = list(zip(receptor_chains, model_receptor_chains))
        lig_pairs = list(zip(ligand_chains, model_ligand_chains))

    if not rec_pairs or not lig_pairs:
        raise ValueError("Could not establish a chain correspondence between "
                         "reference and model for the requested groups.")

    mod_rec = _chain_residues(model_struct, [b for _, b in rec_pairs])
    mod_lig = _chain_residues(model_struct, [b for _, b in lig_pairs])

    # Native contacts (fixed) in reference-key space.
    rec_res_flat = [r for c in receptor_chains for r in ref_rec[c]]
    lig_res_flat = [r for c in ligand_chains for r in ref_lig[c]]
    native_contacts = _residue_contacts(rec_res_flat, lig_res_flat, FNAT_CONTACT_CUTOFF)
    if not native_contacts:
        raise ValueError("The receptor and ligand groups share no contacts "
                         f"within {FNAT_CONTACT_CUTOFF} A in the reference — "
                         "there is no native interface to score.")

    # Homomultimer handling: enumerate permutations of model chains within
    # sequence-identical equivalence classes, keep the mapping with max fnat.
    rec_pair_options = _mapping_permutations(rec_pairs, ref_rec, mod_rec)
    lig_pair_options = _mapping_permutations(lig_pairs, ref_lig, mod_lig)
    if len(rec_pair_options) * len(lig_pair_options) > _MAX_MAPPING_PERMUTATIONS:
        rec_pair_options, lig_pair_options = [rec_pairs], [lig_pairs]

    best = None
    pair_cache: dict = {}
    for rp in rec_pair_options:
        for lp in lig_pair_options:
            rec_corr = _GroupCorrespondence(ref_rec, mod_rec, rp, pair_cache)
            lig_corr = _GroupCorrespondence(ref_lig, mod_lig, lp, pair_cache)
            model_contacts_model_keys = _residue_contacts(
                rec_corr.model_res_all, lig_corr.model_res_all, FNAT_CONTACT_CUTOFF)
            # translate model contacts into reference-key space
            model_contacts = set()
            for (ka, kb) in model_contacts_model_keys:
                ra = rec_corr.ref_for_model.get(ka, ("?",) + ka)
                rb = lig_corr.ref_for_model.get(kb, ("?",) + kb)
                model_contacts.add((ra, rb))
            shared = len(native_contacts & model_contacts)
            fnat = shared / len(native_contacts)
            fnonnat = (1.0 - shared / len(model_contacts)) if model_contacts else 0.0
            cand = (fnat, rec_corr, lig_corr, model_contacts, fnonnat)
            if best is None or fnat > best[0]:
                best = cand
    fnat, rec_corr, lig_corr, model_contacts, fnonnat = best

    # Interface residues (native 10 A heavy-atom definition).
    interface_contacts = _residue_contacts(rec_res_flat, lig_res_flat, INTERFACE_CUTOFF)
    iface_rec_keys = {a for a, _ in interface_contacts}
    iface_lig_keys = {b for _, b in interface_contacts}

    iface_pairs = []
    for r in rec_res_flat:
        if r.key in iface_rec_keys and r.key in rec_corr.model_for_ref:
            iface_pairs.append((r, rec_corr.model_for_ref[r.key]))
    for r in lig_res_flat:
        if r.key in iface_lig_keys and r.key in lig_corr.model_for_ref:
            iface_pairs.append((r, lig_corr.model_for_ref[r.key]))

    P_iface, Q_iface = _collect_backbone(iface_pairs)
    if len(P_iface) >= 3:
        _, _, irmsd = _kabsch(P_iface, Q_iface)
    else:
        irmsd = float("inf")

    # LRMSD: superpose on the receptor backbone, measure the ligand backbone.
    rec_all_pairs = [(r, rec_corr.model_for_ref[r.key])
                     for r in rec_res_flat if r.key in rec_corr.model_for_ref]
    lig_all_pairs = [(r, lig_corr.model_for_ref[r.key])
                     for r in lig_res_flat if r.key in lig_corr.model_for_ref]
    P_rec, Q_rec = _collect_backbone(rec_all_pairs)
    P_lig, Q_lig = _collect_backbone(lig_all_pairs)
    if len(P_rec) >= 3 and len(P_lig) >= 1:
        R, t, _ = _kabsch(P_rec, Q_rec)
        Q_lig_sup = (R @ Q_lig.T).T + t
        lrmsd = float(np.sqrt(np.mean(np.sum((P_lig - Q_lig_sup) ** 2, axis=1))))
    else:
        lrmsd = float("inf")

    if not math.isfinite(irmsd) or not math.isfinite(lrmsd):
        raise ValueError("Too few corresponding backbone atoms to compute "
                         "iRMSD/LRMSD — check the chain selections and mapping.")

    dockq = dockq_formula(fnat, irmsd, lrmsd)
    return DockQResult(
        dockq=float(dockq), capri=capri_class(dockq),
        fnat=float(fnat), fnonnat=float(fnonnat),
        irmsd=float(irmsd), lrmsd=float(lrmsd),
        n_native_contacts=len(native_contacts),
        n_model_contacts=len(model_contacts),
        n_interface_residues=len(iface_rec_keys) + len(iface_lig_keys),
        receptor_mapping=list(rec_corr.chain_pairs),
        ligand_mapping=list(lig_corr.chain_pairs),
        interface_ligand_residues=sorted(f"{k[0]}:{k[1]}{k[2]}" for k in iface_lig_keys),
    )


def _mapping_permutations(chain_pairs, ref_res_by_chain, mod_res_by_chain):
    """Alternative (ref, model) chain pairings permuting sequence-identical
    model chains. Returns at least the input pairing."""
    seqs = {}
    for _, mc in chain_pairs:
        seqs[mc] = "".join(r.letter for r in mod_res_by_chain[mc])
    classes: Dict[str, List[str]] = {}
    for mc, s in seqs.items():
        classes.setdefault(s, []).append(mc)
    swappable = [v for v in classes.values() if len(v) > 1]
    if not swappable:
        return [list(chain_pairs)]
    n_perms = 1
    for group in swappable:
        n_perms *= math.factorial(len(group))
    if n_perms > _MAX_MAPPING_PERMUTATIONS:
        return [list(chain_pairs)]
    options = []
    perm_sets = [list(itertools.permutations(g)) for g in swappable]
    for combo in itertools.product(*perm_sets):
        remap = {}
        for group, perm in zip(swappable, combo):
            remap.update(dict(zip(group, perm)))
        options.append([(rc, remap.get(mc, mc)) for rc, mc in chain_pairs])
    return options


# --------------------------------------------------------------------------
# epitope / paratope site comparison
# --------------------------------------------------------------------------

@dataclass
class SiteComparison:
    """Native-vs-model comparison of one binding site (residue sets)."""
    precision: float
    recall: float
    f1: float
    jaccard: float
    native_residues: List[str]
    model_residues: List[str]

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "jaccard": round(self.jaccard, 4),
            "n_native": len(self.native_residues),
            "n_model": len(self.model_residues),
            "native_residues": self.native_residues,
            "model_residues": self.model_residues,
        }


def _compare_sites(native_keys: set, model_keys_in_ref_space: set) -> SiteComparison:
    inter = native_keys & model_keys_in_ref_space
    union = native_keys | model_keys_in_ref_space
    precision = len(inter) / len(model_keys_in_ref_space) if model_keys_in_ref_space else 0.0
    recall = len(inter) / len(native_keys) if native_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jac = len(inter) / len(union) if union else 0.0
    fmt = lambda ks: sorted(f"{k[0]}:{k[1]}{k[2]}" for k in ks)
    return SiteComparison(precision=precision, recall=recall, f1=f1, jaccard=jac,
                          native_residues=fmt(native_keys),
                          model_residues=fmt(model_keys_in_ref_space))


def epitope_metrics(
    reference: Union[str, gemmi.Structure],
    model: Union[str, gemmi.Structure],
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    model_receptor_chains: Optional[Sequence[str]] = None,
    model_ligand_chains: Optional[Sequence[str]] = None,
    cutoff: float = EPITOPE_CUTOFF,
) -> Dict[str, SiteComparison]:
    """
    Epitope (ligand-side) and paratope (receptor-side) residue-set comparison.

    A site residue has any heavy atom within ``cutoff`` (default 4.5 A) of the
    other group. Model residues are mapped into reference numbering through
    the per-chain sequence alignment before comparison, so the metrics are
    robust to numbering offsets. Returns ``{"epitope": ..., "paratope": ...}``.
    """
    ref_struct = _as_structure(reference)
    model_struct = _as_structure(model)
    ref_rec = _chain_residues(ref_struct, receptor_chains)
    ref_lig = _chain_residues(ref_struct, ligand_chains)

    if model_receptor_chains is None or model_ligand_chains is None:
        model_chain_names = [ch.name for ch in model_struct[0]]
        rec_pairs = _match_chain_groups(ref_struct, model_struct,
                                        receptor_chains, model_chain_names)
        used = {b for _, b in rec_pairs}
        lig_pairs = _match_chain_groups(ref_struct, model_struct, ligand_chains,
                                        [c for c in model_chain_names if c not in used])
    else:
        rec_pairs = list(zip(receptor_chains, model_receptor_chains))
        lig_pairs = list(zip(ligand_chains, model_ligand_chains))

    mod_rec = _chain_residues(model_struct, [b for _, b in rec_pairs])
    mod_lig = _chain_residues(model_struct, [b for _, b in lig_pairs])
    rec_corr = _GroupCorrespondence(ref_rec, mod_rec, rec_pairs)
    lig_corr = _GroupCorrespondence(ref_lig, mod_lig, lig_pairs)

    rec_flat = [r for c in receptor_chains for r in ref_rec[c]]
    lig_flat = [r for c in ligand_chains for r in ref_lig[c]]
    native = _residue_contacts(rec_flat, lig_flat, cutoff)
    model_c = _residue_contacts(rec_corr.model_res_all, lig_corr.model_res_all, cutoff)

    native_paratope = {a for a, _ in native}
    native_epitope = {b for _, b in native}
    model_paratope = {rec_corr.ref_for_model.get(a, ("?",) + a) for a, _ in model_c}
    model_epitope = {lig_corr.ref_for_model.get(b, ("?",) + b) for _, b in model_c}

    return {
        "epitope": _compare_sites(native_epitope, model_epitope),
        "paratope": _compare_sites(native_paratope, model_paratope),
    }


# --------------------------------------------------------------------------
# immune-complex convenience wrapper
# --------------------------------------------------------------------------

@dataclass
class ImmuneComplexResult:
    dockq: DockQResult
    epitope: SiteComparison
    paratope: SiteComparison

    def to_dict(self) -> dict:
        return {"dockq": self.dockq.to_dict(),
                "epitope": self.epitope.to_dict(),
                "paratope": self.paratope.to_dict()}

    def report(self) -> str:
        e, p = self.epitope, self.paratope
        lines = [self.dockq.report(),
                 f"Epitope : recall {e.recall:.2f}, precision {e.precision:.2f}, "
                 f"F1 {e.f1:.2f} ({len(e.native_residues)} native residues)",
                 f"Paratope: recall {p.recall:.2f}, precision {p.precision:.2f}, "
                 f"F1 {p.f1:.2f}"]
        if self.dockq.dockq < 0.23 and e.f1 >= 0.5:
            lines.append("Note: pose is incorrect by DockQ but the model targets "
                         "the correct epitope (high epitope F1) — likely a "
                         "mis-oriented binding mode, not a wrong site.")
        elif e.f1 < 0.25 and self.dockq.dockq < 0.23:
            lines.append("Note: the model binds a different antigen surface "
                         "(wrong epitope).")
        return "\n".join(lines)


def evaluate_antibody_complex(
    reference: Union[str, gemmi.Structure],
    model: Union[str, gemmi.Structure],
    antibody_chains: Sequence[str],
    antigen_chains: Sequence[str],
    model_antibody_chains: Optional[Sequence[str]] = None,
    model_antigen_chains: Optional[Sequence[str]] = None,
) -> ImmuneComplexResult:
    """
    Score a modelled antibody-antigen (or nanobody/TCR) complex against a
    reference: DockQ with the antibody chains (e.g. H+L) merged as the
    receptor — the standard convention for immune complexes — plus epitope
    and paratope precision/recall/F1, which distinguish "right epitope,
    imperfect pose" from "wrong face of the antigen".
    """
    dq = compute_dockq(reference, model, antibody_chains, antigen_chains,
                       model_antibody_chains, model_antigen_chains)
    sites = epitope_metrics(reference, model, antibody_chains, antigen_chains,
                            model_antibody_chains, model_antigen_chains)
    return ImmuneComplexResult(dockq=dq, epitope=sites["epitope"],
                               paratope=sites["paratope"])


# --------------------------------------------------------------------------
# pDockQ (reference-free)
# --------------------------------------------------------------------------

@dataclass
class PDockQResult:
    pdockq: float
    n_contacts: int
    mean_interface_plddt: float

    def to_dict(self) -> dict:
        return {"pdockq": round(self.pdockq, 4),
                "n_contacts": self.n_contacts,
                "mean_interface_plddt": round(self.mean_interface_plddt, 2)}


def compute_pdockq(
    structure: Union[str, gemmi.Structure],
    chains_a: Sequence[str],
    chains_b: Sequence[str],
    cutoff: float = PDOCKQ_CUTOFF,
) -> PDockQResult:
    """
    pDockQ interface confidence of a *predicted* complex (no reference needed).

    Bryant, Pozzati & Elofsson, Nat Commun 2022: contacts are CB-CB pairs
    (CA for glycine) within 8 A between the two groups; x = mean pLDDT over
    the unique interface residues of both sides times log10(n_contacts);
    pDockQ = 0.724/(1+exp(-0.052*(x-152.611))) + 0.018.

    The B-factor column must hold pLDDT values (as AlphaFold/Boltz write
    them); a warning is emitted when it does not look like pLDDT.
    """
    struct = _as_structure(structure)
    res_by_chain = _chain_residues(struct, list(chains_a) + list(chains_b))
    res_a = [r for c in chains_a for r in res_by_chain[c]]
    res_b = [r for c in chains_b for r in res_by_chain[c]]
    if not res_a or not res_b:
        raise ValueError("Empty chain group for pDockQ.")

    A = np.array([r.cb_or_ca() for r in res_a])
    B = np.array([r.cb_or_ca() for r in res_b])
    from scipy.spatial import cKDTree
    pairs = cKDTree(A).query_ball_tree(cKDTree(B), cutoff)
    ia, ib = set(), set()
    n_contacts = 0
    for i, neigh in enumerate(pairs):
        for j in neigh:
            n_contacts += 1
            ia.add(i)
            ib.add(j)
    if n_contacts == 0:
        return PDockQResult(pdockq=0.0, n_contacts=0, mean_interface_plddt=0.0)

    plddts = ([res_a[i].plddt for i in ia if res_a[i].plddt is not None]
              + [res_b[j].plddt for j in ib if res_b[j].plddt is not None])
    if not plddts:
        raise ValueError("No pLDDT (B-factor) values available for the interface.")
    avg_plddt = float(np.mean(plddts))
    if avg_plddt <= 1.0 or avg_plddt > 100.0:
        warnings.warn(
            f"Mean interface B-factor {avg_plddt:.2f} does not look like pLDDT "
            "(expected 0-100 from a predicted model); pDockQ is only meaningful "
            "for predicted structures with pLDDT in the B-factor column.",
            UserWarning, stacklevel=2)

    x = avg_plddt * math.log10(n_contacts)
    pdockq = _PDOCKQ_L / (1.0 + math.exp(-_PDOCKQ_K * (x - _PDOCKQ_X0))) + _PDOCKQ_B
    return PDockQResult(pdockq=float(pdockq), n_contacts=int(n_contacts),
                        mean_interface_plddt=avg_plddt)
