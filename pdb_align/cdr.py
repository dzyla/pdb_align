"""CDR annotation and per-CDR RMSD for antibody models.

IMGT CDR definitions (Lefranc et al., Dev Comp Immunol 2003): CDR1 = IMGT
positions 27-38, CDR2 = 56-65, CDR3 = 105-117; everything else in the
numbered V-domain is framework.

Numbering is delegated to **ANARCI** (Dunbar & Deane, Bioinformatics 2016)
when installed (``pip install anarci`` plus an HMMER 3.3.x ``hmmscan`` on
PATH — note that HMMER >= 3.4 text output is not parsed correctly by
Biopython/ANARCI at the time of writing). Any callable with the same
signature can be injected instead (``numberer(seq) -> (numbering, start,
chain_type)``), which also keeps the geometry fully unit-testable without
ANARCI.

The headline metric is **CDR-H3 RMSD after framework superposition**: the
model's framework backbone (all antibody chains combined) is superposed onto
the reference framework, and each CDR's backbone RMSD is measured in that
frame — the standard convention in antibody modelling assessment (e.g. AMA-II,
Almagro et al. 2014).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMGT_CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}

# numberer contract: seq -> (numbering, start, chain_type) where numbering is
# a list of ((imgt_number, insertion_code), aa_or_dash) covering the V domain,
# whose non-dash residues map sequentially onto seq[start:], and chain_type is
# "H", "K", or "L". Return None if the sequence is not an antibody V domain.
Numberer = Callable[[str], Optional[Tuple[list, int, str]]]


def anarci_numberer(seq: str) -> Optional[Tuple[list, int, str]]:
    """Default numberer: ANARCI with the IMGT scheme."""
    try:
        from anarci import anarci as _anarci_fn
    except ImportError as e:
        raise RuntimeError(
            "CDR annotation needs ANARCI (`pip install anarci`) and an HMMER "
            "3.3.x `hmmscan` on PATH (e.g. `conda install -c bioconda "
            "'hmmer=3.3*'`; HMMER >= 3.4 output is not parsed correctly by "
            "ANARCI's Biopython parser). Alternatively pass your own "
            "`numberer` callable.") from e
    try:
        numbered, details, _ = _anarci_fn([("q", seq)], scheme="imgt")
    except Exception as e:
        raise RuntimeError(
            f"ANARCI failed ({e}). If hmmscan is HMMER >= 3.4, install 3.3.x "
            "(`conda install -c bioconda 'hmmer=3.3*'`).") from e
    if not numbered or numbered[0] is None:
        return None  # not an antibody variable domain
    numbering, start, _end = numbered[0][0]
    chain_type = details[0][0]["chain_type"]
    return numbering, int(start), str(chain_type)


@dataclass
class CDRAnnotation:
    chain_type: str                       # "H", "K", or "L"
    regions: Dict[str, List[int]]         # CDR1/2/3 -> 0-based sequence indices
    framework: List[int]                  # numbered non-CDR sequence indices
    sequences: Dict[str, str]             # CDR label -> amino-acid string


def annotate_cdrs(seq: str, numberer: Optional[Numberer] = None) -> Optional[CDRAnnotation]:
    """IMGT CDR/framework annotation of one chain sequence.

    Returns None when the sequence is not recognized as an antibody variable
    domain. Sequence indices are 0-based positions in ``seq``.
    """
    numberer = numberer or anarci_numberer
    out = numberer(seq)
    if out is None:
        return None
    numbering, start, chain_type = out

    regions: Dict[str, List[int]] = {k: [] for k in IMGT_CDR_RANGES}
    framework: List[int] = []
    pos = start
    for (num, _icode), aa in numbering:
        if aa == "-":
            continue
        if pos >= len(seq):
            break
        for name, (lo, hi) in IMGT_CDR_RANGES.items():
            if lo <= num <= hi:
                regions[name].append(pos)
                break
        else:
            framework.append(pos)
        pos += 1
    sequences = {name: "".join(seq[i] for i in idxs)
                 for name, idxs in regions.items()}
    return CDRAnnotation(chain_type=chain_type, regions=regions,
                         framework=framework, sequences=sequences)


@dataclass
class CDRResult:
    """Per-CDR backbone RMSD after framework superposition."""
    cdr_rmsd: Dict[str, float]            # e.g. {"H1": ..., "H3": ..., "L2": ...}
    cdr_n_residues: Dict[str, int]
    cdr_sequences: Dict[str, str]
    framework_rmsd: float
    n_framework_atoms: int
    chain_types: Dict[str, str]           # ref chain id -> H/K/L

    @property
    def h3(self) -> Optional[float]:
        return self.cdr_rmsd.get("H3")

    def to_dict(self) -> dict:
        return {
            "cdr_rmsd": {k: round(v, 3) for k, v in self.cdr_rmsd.items()},
            "cdr_n_residues": dict(self.cdr_n_residues),
            "cdr_sequences": dict(self.cdr_sequences),
            "framework_rmsd": round(self.framework_rmsd, 3),
            "chain_types": dict(self.chain_types),
        }

    def report(self) -> str:
        parts = [f"Framework RMSD: {self.framework_rmsd:.2f} A"]
        for name in sorted(self.cdr_rmsd):
            parts.append(f"{name}: {self.cdr_rmsd[name]:.2f} A "
                         f"({self.cdr_n_residues[name]} res)")
        return "\n".join(parts)


def cdr_rmsd(
    reference,
    model,
    antibody_chains: Sequence[str],
    model_antibody_chains: Optional[Sequence[str]] = None,
    numberer: Optional[Numberer] = None,
) -> CDRResult:
    """
    Per-CDR backbone RMSD of a modelled antibody against a reference.

    The reference antibody chains are IMGT-numbered (via ``numberer``,
    default ANARCI); residue correspondence to the model comes from the same
    per-chain sequence alignment machinery as DockQ, so numbering offsets are
    irrelevant. All chains' framework backbones are superposed with one Kabsch
    fit, and each CDR's backbone RMSD is evaluated in that frame (CDR-H3 RMSD
    being the standard antibody-modelling headline number).

    CDR labels are ``H1..H3`` / ``L1..L3`` (kappa reported as L); if two
    chains of the same type are present, later ones get a ``@chain`` suffix.
    """
    from .core import _kabsch
    from .interface import (
        _as_structure,
        _chain_residues,
        _collect_backbone,
        _match_chain_groups,
        _pair_residues,
    )

    ref_struct = _as_structure(reference)
    model_struct = _as_structure(model)
    ref_res = _chain_residues(ref_struct, antibody_chains)

    if model_antibody_chains is None:
        model_chain_names = [ch.name for ch in model_struct[0]]
        chain_pairs = _match_chain_groups(ref_struct, model_struct,
                                          antibody_chains, model_chain_names)
    else:
        chain_pairs = list(zip(antibody_chains, model_antibody_chains))
    if not chain_pairs:
        raise ValueError("No antibody chain correspondence between reference and model.")
    mod_res = _chain_residues(model_struct, [b for _, b in chain_pairs])

    framework_pairs = []      # (ref_res, model_res) across all chains
    region_pairs: Dict[str, list] = {}
    region_seqs: Dict[str, str] = {}
    chain_types: Dict[str, str] = {}
    seen_labels: Dict[str, int] = {}

    for rc, mc in chain_pairs:
        r_list, m_list = ref_res[rc], mod_res[mc]
        seq = "".join(r.letter for r in r_list)
        ann = annotate_cdrs(seq, numberer=numberer)
        if ann is None:
            warnings.warn(f"Chain {rc} was not recognized as an antibody "
                          "variable domain; skipped for CDR metrics.",
                          UserWarning, stacklevel=2)
            continue
        chain_types[rc] = ann.chain_type
        prefix = "H" if ann.chain_type == "H" else "L"
        corr = dict(_pair_residues(r_list, m_list))

        for i in ann.framework:
            if i in corr:
                framework_pairs.append((r_list[i], m_list[corr[i]]))
        for name, idxs in ann.regions.items():
            label = prefix + name[-1]          # CDR3 -> H3 / L3
            seen_labels[label] = seen_labels.get(label, 0) + 1
            if seen_labels[label] > 1:
                label = f"{label}@{rc}"
            pairs = [(r_list[i], m_list[corr[i]]) for i in idxs if i in corr]
            region_pairs[label] = pairs
            region_seqs[label] = ann.sequences[name]

    if not chain_types:
        raise ValueError("None of the given chains was recognized as an "
                         "antibody variable domain.")

    P_fw, Q_fw = _collect_backbone(framework_pairs)
    if len(P_fw) < 3:
        raise ValueError("Too few corresponding framework backbone atoms for "
                         "superposition.")
    R, t, fw_rmsd = _kabsch(P_fw, Q_fw)

    cdr_out: Dict[str, float] = {}
    for label, pairs in region_pairs.items():
        P, Q = _collect_backbone(pairs)
        if len(P) == 0:
            continue
        Qs = (R @ Q.T).T + t
        cdr_out[label] = float(np.sqrt(np.mean(np.sum((P - Qs) ** 2, axis=1))))

    return CDRResult(cdr_rmsd=cdr_out,
                     cdr_n_residues={k: len(region_pairs[k]) for k in cdr_out},
                     cdr_sequences={k: region_seqs[k] for k in cdr_out},
                     framework_rmsd=float(fw_rmsd),
                     n_framework_atoms=len(P_fw),
                     chain_types=chain_types)
