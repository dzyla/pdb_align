"""Low-level structure handling, residue selection, pairing and geometry.

Design rule that the rest of the package depends on
---------------------------------------------------
**One selection, one sequence.** :func:`select_residues` is the only place
that decides which residues take part in a comparison. It returns a
:class:`Selection` whose ``sequence`` and ``residues`` are the same residues in
the same order, so ``len(sequence) == len(residues)`` always holds. Sequence
alignment therefore maps onto coordinates by *index*, never by re-scanning for
a matching one-letter code.

This is not a stylistic preference. When the sequence was extracted
independently of the coordinate list (as it was before), any filter applied to
one and not the other — a residue-range selector, a B-factor cutoff, a
pLDDT cutoff — silently shifted the pairing. Two identical copies of 1UBQ with
two residues filtered out of one side produced 24 pairs, four of them joining
different residues, and an RMSD of 5.62 A between a structure and itself.

Residue naming follows gemmi's CCD-backed tables, so modified residues
(selenomethionine, phosphoserine, methylated lysines, ...) resolve to their
parent one-letter code instead of vanishing from the sequence.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gemmi
import numpy as np

logger = logging.getLogger(__name__)

# numba is optional and imported lazily.
#
# It costs ~90 ms to import and is the dependency most likely to block an
# install on a new Python or NumPy release, while only two code paths need it
# (the sequence-free distance-matrix kernels and the hinge-detection window).
# `lazy_jit` therefore compiles on first call: a run that never touches those
# paths never imports numba, and an environment without numba runs the same
# code in pure Python.
#
# `prange` starts as the builtin `range` so the un-JITted fallback works, and
# is rebound to numba's `prange` before compilation so the parallel loops are
# still parallel (numba resolves globals at compile time).
prange = range
_numba_state: Dict[str, Any] = {"checked": False, "available": False}


def numba_available() -> bool:
    """True if numba is importable (checked once, on first use)."""
    if not _numba_state["checked"]:
        _numba_state["checked"] = True
        try:
            import numba  # noqa: F401
            _numba_state["available"] = True
        except ImportError:  # pragma: no cover - depends on the environment
            _numba_state["available"] = False
    return bool(_numba_state["available"])


def lazy_jit(**jit_kwargs):
    """Compile *func* with numba on first call; fall back to Python without it."""
    def decorator(func):
        compiled: Dict[str, Any] = {}

        @wraps(func)
        def wrapper(*args):
            fn = compiled.get("fn")
            if fn is None:
                if numba_available():
                    import numba
                    globals()["prange"] = numba.prange
                    try:
                        fn = numba.njit(**jit_kwargs)(func)
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "numba could not compile %s (%s); using the pure-"
                            "Python implementation.", func.__name__, exc)
                        fn = func
                else:
                    fn = func
                compiled["fn"] = fn
            return fn(*args)

        wrapper.__wrapped_py__ = func
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# residue identity
# ---------------------------------------------------------------------------

# Amino acids whose standard one-letter code is absent from BLOSUM62; mapped to
# the chemically closest coded parent so sequence alignment stays well defined.
_UNCODED_PARENT = {"U": "C", "O": "K"}

# Largest selection (residues x residues) the sequence-free path will attempt
# before refusing; the distance-matrix algorithms are inherently O(N^2) in
# memory and failing with an explanation beats being killed by the OOM reaper.
MAX_SEQFREE_RESIDUES = 20000


@lru_cache(maxsize=4096)
def residue_letter(resname: str) -> Optional[str]:
    """One-letter code for a protein residue, or ``None`` if it is not one.

    Uses gemmi's chemical-component tables, which return the *parent* code in
    lower case for modified residues (``MSE`` -> ``m`` -> ``M``). Non-amino
    acids (waters, ligands, nucleotides) return ``None``. ``UNK`` maps to
    ``X``, which BLOSUM62 scores, so unknown residues keep their place in the
    chain instead of opening a phantom gap.
    """
    info = gemmi.find_tabulated_residue(resname)
    if info is None or not info.is_amino_acid():
        return None
    code = (info.one_letter_code or "").strip()
    if not code:
        return None
    upper = code.upper()
    return _UNCODED_PARENT.get(upper, upper)


def is_protein_residue(resname: str) -> bool:
    return residue_letter(resname) is not None


@lru_cache(maxsize=1)
def blosum62():
    """BLOSUM62 substitution matrix (loaded once; the load is not cheap)."""
    from Bio.Align import substitution_matrices
    return substitution_matrices.load("BLOSUM62")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

@dataclass
class ResidueSel:
    """One selected residue: identity, CA, and its heavy atoms by name."""
    chain_id: str
    seqid: int
    icode: str
    name: str
    letter: str
    ca: np.ndarray
    b_iso: float
    atoms: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.seqid, self.icode)

    @property
    def label(self) -> str:
        return f"{self.chain_id}:{self.seqid}{self.icode}"


@dataclass
class Selection:
    """A set of residues plus the sequence that *exactly* describes them."""
    residues: List[ResidueSel]
    sequence: str
    chain_order: List[str]
    lens: Dict[str, int]
    source: str = ""

    def __post_init__(self):
        if len(self.sequence) != len(self.residues):
            raise AssertionError(
                "Selection invariant violated: sequence length "
                f"{len(self.sequence)} != {len(self.residues)} residues")

    def __len__(self) -> int:
        return len(self.residues)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def ca_coords(self) -> np.ndarray:
        if not self.residues:
            return np.empty((0, 3))
        return np.vstack([r.ca for r in self.residues])

    @property
    def chain_ids(self) -> List[str]:
        return [r.chain_id for r in self.residues]

    def chain_start_indices(self) -> List[int]:
        """Indices at which a new chain begins (always includes 0 if non-empty).

        Used to stop per-residue analyses (hinge detection, contiguous-region
        flagging) from running across a chain boundary as if it were sequence.
        """
        starts: List[int] = []
        prev = None
        for i, r in enumerate(self.residues):
            if r.chain_id != prev:
                starts.append(i)
                prev = r.chain_id
        return starts

    def sub(self, indices: Sequence[int]) -> "Selection":
        residues = [self.residues[i] for i in indices]
        lens: Dict[str, int] = {}
        for r in residues:
            lens[r.chain_id] = lens.get(r.chain_id, 0) + 1
        order = [c for c in self.chain_order if c in lens]
        return Selection(residues=residues,
                         sequence="".join(r.letter for r in residues),
                         chain_order=order, lens=lens, source=self.source)


def _parse_chain_selector(selector: str) -> Tuple[str, Optional[int], Optional[int]]:
    """``"A"`` -> (A, None, None); ``"A:10-150"`` -> (A, 10, 150); ``"A:42"`` -> (A, 42, 42)."""
    text = str(selector)
    if ":" not in text:
        return text, None, None
    chain_id, _, range_str = text.partition(":")
    range_str = range_str.strip()
    if not range_str:
        return chain_id, None, None
    if "-" in range_str.lstrip("-"):
        lo, _, hi = range_str.partition("-")
        try:
            start = int(lo) if lo.strip() else None
            end = int(hi) if hi.strip() else None
        except ValueError:
            raise ValueError(
                f"Invalid residue range in chain selector {selector!r}; "
                "expected 'CHAIN:start-end' with integer bounds.") from None
        return chain_id, start, end
    try:
        only = int(range_str)
    except ValueError:
        raise ValueError(
            f"Invalid residue range in chain selector {selector!r}; "
            "expected 'CHAIN:start-end' with integer bounds.") from None
    return chain_id, only, only


def _chain_ids(struct: gemmi.Structure) -> List[str]:
    return [ch.name for ch in struct[0]]


def _resolve_selectors(struct: gemmi.Structure,
                       sel: Optional[Sequence[Union[str, int]]]) -> Optional[List[str]]:
    """Normalise a chain selection to a list of selector strings.

    Integers are 1-based chain indices. Chain names are validated against the
    structure; residue ranges are preserved verbatim for
    :func:`select_residues` to apply.
    """
    if sel is None:
        return None
    ids = _chain_ids(struct)
    out: List[str] = []
    for item in sel:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idx = int(item)
            if not 1 <= idx <= len(ids):
                raise ValueError(f"Chain index {idx} out of range 1..{len(ids)}")
            out.append(ids[idx - 1])
        else:
            cid, _, _ = _parse_chain_selector(item)
            if cid not in ids:
                raise ValueError(
                    f"Chain {cid!r} (from {item!r}) not in {ids}")
            out.append(str(item))
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _icode_of(res: gemmi.Residue) -> str:
    icode = res.seqid.icode or ""
    return icode.strip() if icode.strip() != "" else ""


def select_residues(
    struct: gemmi.Structure,
    selectors: Optional[Sequence[Union[str, int]]] = None,
    min_b_factor: float = 0.0,
    min_plddt: float = 0.0,
    source: str = "",
    with_atoms: bool = True,
) -> Selection:
    """Select protein residues and build the matching sequence in one pass.

    Parameters
    ----------
    selectors
        Chain names, 1-based chain indices, or ``"CHAIN:start-end"`` residue
        ranges. ``None`` selects every chain.
    min_b_factor, min_plddt
        Lower bounds on the CA ``b_iso``. Both are lower bounds on the same
        column, so the effective cutoff is the larger; see
        :meth:`pdb_align.PDBAligner.align` for why ``min_plddt`` is applied
        only to predicted structures.
    with_atoms
        Collect every heavy atom per residue (needed for ``atoms="backbone"``
        / ``"all_heavy"``). ``False`` keeps CA only, which is cheaper.

    Raises
    ------
    ValueError
        If a named chain is absent, a residue range selects nothing, or the
        selection ends up empty. Silence here would mean comparing something
        other than what the caller asked for.
    """
    if len(struct) == 0:
        raise ValueError(f"Structure {source or struct.name!r} has no models.")
    model = struct[0]
    available = [ch.name for ch in model]

    ranges: Dict[str, List[Tuple[Optional[int], Optional[int]]]] = {}
    order: List[str] = []
    if selectors is None:
        order = list(available)
        ranges = {c: [] for c in available}
    else:
        for item in selectors:
            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                idx = int(item)
                if not 1 <= idx <= len(available):
                    raise ValueError(
                        f"Chain index {idx} out of range 1..{len(available)} "
                        f"in {source or 'structure'}")
                cid, start, end = available[idx - 1], None, None
            else:
                cid, start, end = _parse_chain_selector(item)
            if cid not in available:
                raise ValueError(
                    f"Chain {cid!r} not found in {source or 'structure'} "
                    f"(available: {available})")
            if cid not in ranges:
                ranges[cid] = []
                order.append(cid)
            if start is not None or end is not None:
                ranges[cid].append((start, end))

    cutoff = max(float(min_b_factor), float(min_plddt))
    residues: List[ResidueSel] = []
    lens: Dict[str, int] = {}
    chain_order: List[str] = []
    dropped_by_cutoff = 0

    for cid in order:
        chain = model.find_chain(cid)
        if chain is None:
            continue
        bounds = ranges.get(cid, [])
        n_in_range = 0
        for res in chain:
            letter = residue_letter(res.name)
            if letter is None:
                continue
            seqid = int(res.seqid.num)
            if bounds:
                if not any((lo is None or seqid >= lo) and (hi is None or seqid <= hi)
                           for lo, hi in bounds):
                    continue
            n_in_range += 1
            ca = None
            atoms: Dict[str, np.ndarray] = {}
            for atom in res:
                if atom.element == gemmi.Element("H"):
                    continue
                name = atom.name
                if name == "CA" and ca is None:
                    ca = atom
                if with_atoms and name not in atoms:
                    atoms[name] = np.array(atom.pos.tolist(), dtype=float)
            # Residues without a CA cannot anchor a superposition or a
            # sequence position; keeping them would desynchronise the two.
            if ca is None:
                continue
            if cutoff > 0.0 and ca.b_iso < cutoff:
                dropped_by_cutoff += 1
                continue
            coord = np.array(ca.pos.tolist(), dtype=float)
            if not with_atoms:
                atoms = {"CA": coord}
            residues.append(ResidueSel(
                chain_id=cid, seqid=seqid, icode=_icode_of(res), name=res.name,
                letter=letter, ca=coord, b_iso=float(ca.b_iso), atoms=atoms))
            lens[cid] = lens.get(cid, 0) + 1
            if cid not in chain_order:
                chain_order.append(cid)
        if bounds and n_in_range == 0:
            rng = ", ".join(f"{lo}-{hi}" for lo, hi in bounds)
            raise ValueError(
                f"Residue range {cid}:{rng} selected no residues in "
                f"{source or 'structure'}; chain {cid} covers "
                f"{_chain_residue_span(chain)}.")

    if not residues:
        detail = (f" ({dropped_by_cutoff} residues were removed by the "
                  f"B-factor/pLDDT cutoff of {cutoff:g})" if dropped_by_cutoff else "")
        raise ValueError(
            f"Selection {list(selectors) if selectors else 'all chains'} "
            f"matched no protein residues with a CA atom in "
            f"{source or 'structure'}{detail}.")

    return Selection(residues=residues,
                     sequence="".join(r.letter for r in residues),
                     chain_order=chain_order, lens=lens, source=source)


def _chain_residue_span(chain: gemmi.Chain) -> str:
    nums = [int(r.seqid.num) for r in chain if residue_letter(r.name) is not None]
    return f"residues {min(nums)}-{max(nums)}" if nums else "no protein residues"


# ---------------------------------------------------------------------------
# back-compatible views over a Selection
# ---------------------------------------------------------------------------

@dataclass
class ResidueInfo:
    idx: int
    chain_id: str
    resseq: int
    icode: str
    resname: str
    coord: np.ndarray


def selection_to_infos(sel: Selection) -> List[ResidueInfo]:
    return [ResidueInfo(idx=i, chain_id=r.chain_id, resseq=r.seqid, icode=r.icode,
                        resname=r.name, coord=r.ca)
            for i, r in enumerate(sel.residues)]


def extract_sequences_and_lengths(struct: gemmi.Structure, fname: str = ""):
    """Per-chain sequences and CA counts (one entry per chain with residues).

    Kept as a convenience for chain-level work (similarity matrices, chain
    matching). Anything that pairs residues with coordinates must use
    :func:`select_residues` instead, so the two can never diverge.
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    seqs: Dict[str, SeqRecord] = {}
    lens: Dict[str, int] = {}
    if len(struct) == 0:
        return {}, {}
    for chain in struct[0]:
        try:
            sel = select_residues(struct, [chain.name], source=fname, with_atoms=False)
        except ValueError:
            continue
        if not sel.residues:
            continue
        seqs[chain.name] = SeqRecord(Seq(sel.sequence),
                                     id=f"{fname}_{chain.name}",
                                     description=f"Chain {chain.name}")
        lens[chain.name] = sel.n_residues
    return seqs, lens


def _extract_ca_infos(struct: gemmi.Structure, chain_filter=None,
                      min_b_factor: float = 0.0, min_plddt: float = 0.0,
                      source: str = "") -> List[ResidueInfo]:
    """CA-level residue records for a selection (thin view over Selection)."""
    sel = select_residues(struct, chain_filter, min_b_factor=min_b_factor,
                          min_plddt=min_plddt, source=source, with_atoms=False)
    return selection_to_infos(sel)


# ---------------------------------------------------------------------------
# metrics computed here (single-superposition GDT, contact-map overlap)
# ---------------------------------------------------------------------------

GDT_CUTOFFS = (1.0, 2.0, 4.0, 8.0)


def compute_gdt_ts(dists: np.ndarray, n_total: Optional[int] = None,
                   cutoffs: Tuple[float, ...] = GDT_CUTOFFS) -> float:
    """GDT_TS of a single superposition.

    ``dists`` must be the distances of ALL matched residue pairs under the
    final superposition (never an inlier subset — normalizing by survivors of
    outlier rejection inflates the score). ``n_total`` is the number of
    residues in the reference selection; CASP-style, residues that could not be
    aligned count as failures at every cutoff. When ``n_total`` is None the
    matched-pair count is used (coverage is then NOT penalized — label such
    values accordingly).

    Note: CASP's GDT_TS additionally maximizes each cutoff's fraction over many
    superpositions (LGA); this single-superposition value is a lower bound.
    """
    dists = np.asarray(dists, dtype=float)
    if dists.size == 0:
        return 0.0
    denom = int(n_total) if n_total else dists.size
    if denom <= 0:
        return 0.0
    fractions = [np.count_nonzero(dists <= c) / denom for c in cutoffs]
    return float(np.mean(fractions)) * 100.0


def compute_contact_overlap(ref_coords: np.ndarray, mob_coords: np.ndarray,
                            contact_dist: float = 8.0, chunk: int = 1024) -> float:
    """Jaccard index of the two CA contact maps (self and i,i+1 pairs excluded).

    Superposition-invariant. Computed in row blocks so a large complex does not
    allocate several N x N matrices (at N = 6000 the unchunked version peaked
    near 900 MB).

    This is a contact-map overlap, NOT the CAD-score of Olechnovic & Venclovas,
    which is defined on Voronoi contact *areas* over heavy atoms.
    """
    P = np.asarray(ref_coords, dtype=float)
    Q = np.asarray(mob_coords, dtype=float)
    n = len(P)
    if n == 0 or len(Q) != n:
        return 0.0
    cut2 = contact_dist * contact_dist
    intersection = 0
    union = 0
    idx = np.arange(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        rows = idx[s:e, None]
        d2_ref = np.sum((P[s:e, None, :] - P[None, :, :]) ** 2, axis=-1)
        d2_mob = np.sum((Q[s:e, None, :] - Q[None, :, :]) ** 2, axis=-1)
        valid = np.abs(rows - idx[None, :]) > 1
        c_ref = (d2_ref <= cut2) & valid
        c_mob = (d2_mob <= cut2) & valid
        intersection += int(np.count_nonzero(c_ref & c_mob))
        union += int(np.count_nonzero(c_ref | c_mob))
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _pairwise_dists(coords: np.ndarray) -> np.ndarray:
    x = np.asarray(coords, dtype=float)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares superposition: returns (R, t, rmsd) with ``R @ Q + t ~ P``."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Kabsch expects matched (K, 3) coordinate arrays")
    K = P.shape[0]
    if K == 0:
        raise ValueError("Kabsch needs at least one point")
    if K < 3:
        cP = P.mean(axis=0)
        cQ = Q.mean(axis=0)
        R = np.eye(3)
        t = cP - cQ
        rmsd = float(np.sqrt(np.mean(np.sum((P - (Q + t)) ** 2, axis=1))))
        return R, t, rmsd
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (Q - cQ).T @ (P - cP)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cP - R @ cQ
    rmsd = float(np.sqrt(np.mean(np.sum((P - ((R @ Q.T).T + t)) ** 2, axis=1))))
    return R, t, rmsd


def _transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ np.asarray(coords, dtype=float).T).T + t


def _iterative_kabsch(P: np.ndarray, Q: np.ndarray, recycles: int,
                      keep_fraction: float):
    """Kabsch with optional iterative outlier rejection.

    Returns ``(R, t, rmsd, mask)`` where ``mask`` marks the pairs that survived.
    """
    R, t, rmsd = _kabsch(P, Q)
    N = P.shape[0]
    min_keep = max(3, int(round(N * keep_fraction)))
    mask = np.ones(N, dtype=bool)

    for _ in range(recycles):
        d = np.linalg.norm(P - ((R @ Q.T).T + t), axis=1)
        active_d = d[mask]
        current = float(np.sqrt(np.mean(active_d ** 2))) if active_d.size else 0.0
        cut = max(2.0, 1.5 * current)
        new_mask = (d <= cut) & mask
        if new_mask.sum() < min_keep:
            active = np.where(mask)[0]
            if len(active) <= min_keep:
                new_mask = mask
            else:
                best = active[np.argsort(d[active])][:min_keep]
                new_mask = np.zeros(N, dtype=bool)
                new_mask[best] = True
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
        if mask.sum() < 3:
            break
        R, t, rmsd = _kabsch(P[mask], Q[mask])

    return R, t, rmsd, mask


# ---------------------------------------------------------------------------
# hinge detection
# ---------------------------------------------------------------------------

@lazy_jit(cache=True)
def _sliding_window_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Per-element sliding-window mean (edge windows are truncated)."""
    N = len(arr)
    half = window // 2
    out = np.zeros(N)
    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)
        s = 0.0
        count = 0
        for j in range(start, end):
            s += arr[j]
            count += 1
        out[i] = s / count
    return out


def _detect_hinges_1d(per_residue_rmsd: np.ndarray, window: int, threshold: float,
                      min_segment: int) -> List[int]:
    N = len(per_residue_rmsd)
    if N < 2 * min_segment:
        return []
    smoothed = _sliding_window_mean(np.ascontiguousarray(per_residue_rmsd,
                                                         dtype=float), window)
    hinge_mask = smoothed > threshold

    splits: List[int] = []
    in_hinge = False
    hinge_start = 0
    for i in range(N):
        if hinge_mask[i] and not in_hinge:
            in_hinge = True
            hinge_start = i
        elif not hinge_mask[i] and in_hinge:
            in_hinge = False
            splits.append((hinge_start + i) // 2)
    if in_hinge:
        splits.append((hinge_start + N) // 2)

    filtered: List[int] = []
    prev = 0
    for s in splits:
        if s - prev >= min_segment:
            filtered.append(s)
            prev = s
    while filtered and (N - filtered[-1]) < min_segment:
        filtered.pop()
    return filtered


def _detect_hinges(per_residue_rmsd: np.ndarray, window: int = 15,
                   threshold: float = 3.0, min_segment: int = 30,
                   chain_starts: Optional[Sequence[int]] = None) -> List[int]:
    """0-based split indices for a per-residue RMSD array.

    A "split at index s" means the preceding segment is ``[..s-1]`` and a new
    one begins at ``s``. Consecutive above-threshold positions merge into one
    hinge whose midpoint becomes the split; splits that would leave a segment
    shorter than *min_segment* are dropped.

    ``chain_starts`` lists the indices at which a new chain begins. Those
    indices are always splits, and hinge detection runs inside each chain
    independently — a chain boundary is not a hinge, and a "domain" that spans
    one is not a rigid body. Without this, fitting such a pseudo-domain mixes
    two independent rigid motions and reports a meaningless per-domain RMSD.
    """
    arr = np.asarray(per_residue_rmsd, dtype=float)
    N = len(arr)
    if N == 0:
        return []
    starts = sorted({0, *(int(s) for s in (chain_starts or []) if 0 < int(s) < N)})
    boundaries = starts + [N]
    splits: List[int] = [s for s in starts if s > 0]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        inner = _detect_hinges_1d(arr[lo:hi], window, threshold, min_segment)
        splits.extend(lo + s for s in inner)
    return sorted(set(splits))


# ---------------------------------------------------------------------------
# sequence-free pairing kernels
# ---------------------------------------------------------------------------

@lazy_jit(cache=True, parallel=True, fastmath=True)
def _window_pairs_jit(A: np.ndarray, B: np.ndarray, aN: int, bN: int):
    """Best diagonal offset of A inside B by L1 distance-matrix agreement.

    Explicit loops rather than ``np.sum(np.abs(A - B[o:o+aN, o:o+aN]))``: the
    vectorised form allocated an aN x aN temporary per offset inside the JIT,
    which dominated the runtime. Offsets are independent, so they run in
    parallel.
    """
    n_off = bN - aN + 1
    scores = np.zeros(n_off)
    for offset in prange(n_off):
        total = 0.0
        for i in range(aN):
            bi = offset + i
            for j in range(aN):
                total -= abs(A[i, j] - B[bi, offset + j])
        scores[offset] = total
    best_offset = 0
    best_score = scores[0]
    for o in range(1, n_off):
        if scores[o] > best_score:
            best_score = scores[o]
            best_offset = o
    return best_score, best_offset, scores


def _window_pairs(D1: np.ndarray, D2: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    n1, n2 = D1.shape[0], D2.shape[0]
    if n1 == 0 or n2 == 0:
        return [], np.array([])
    swapped = False
    A, B, aN, bN = D1, D2, n1, n2
    if aN > bN:
        A, B, aN, bN = D2, D1, n2, n1
        swapped = True
    _best_score, best_offset, scores = _window_pairs_jit(
        np.ascontiguousarray(A), np.ascontiguousarray(B), aN, bN)
    if best_offset < 0:
        return [], scores
    if swapped:
        return [(best_offset + i, i) for i in range(aN)], scores
    return [(i, best_offset + i) for i in range(aN)], scores


def _radial_histograms(D: np.ndarray, nbins: int = 24, rmax_mode: str = "p98"):
    """Row-wise normalised histograms of each residue's distances to all others.

    One ``bincount`` per row rather than ``np.histogram`` per row: identical
    output, ~1.7x faster, and no N x N temporaries (a fully "vectorised"
    version that digitises the whole matrix at once needs several N x N index
    arrays and ends up slower than the loop it replaces).
    """
    N = D.shape[0]
    if N == 0:
        return np.zeros((0, nbins)), np.linspace(0, 1, nbins + 1)
    vals = D[np.triu_indices(N, k=1)]
    if vals.size == 0:
        rmax = 1.0
    else:
        rmax = float(np.max(vals)) if rmax_mode == "max" else float(np.quantile(vals, 0.98))
        rmax = max(rmax, 1.0)
    edges = np.linspace(0.0, rmax, nbins + 1)
    scale = nbins / rmax
    H = np.zeros((N, nbins))
    for i in range(N):
        row = D[i]
        # Self-distances (0) carry no shape information; distances past rmax
        # are outside the histogram, and one exactly at rmax belongs to the
        # last bin (np.histogram's closed right edge).
        keep = (row > 0.0) & (row <= rmax)
        if not keep.any():
            continue
        idx = np.minimum((row[keep] * scale).astype(np.intp), nbins - 1)
        counts = np.bincount(idx, minlength=nbins).astype(float)
        total = counts.sum()
        H[i] = counts / total if total > 0 else counts
    return H, edges


@lazy_jit(cache=True, parallel=True, fastmath=True)
def _chi2_distance_jit(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    N = X.shape[0]
    M = Y.shape[0]
    K = X.shape[1]
    res = np.zeros((N, M))
    for i in prange(N):
        for j in range(M):
            s = 0.0
            for k in range(K):
                num = (X[i, k] - Y[j, k]) ** 2
                den = X[i, k] + Y[j, k] + eps
                s += num / den
            res[i, j] = 0.5 * s
    return res


def _chi2_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return _chi2_distance_jit(np.ascontiguousarray(X, dtype=float),
                              np.ascontiguousarray(Y, dtype=float), eps)


@lazy_jit(cache=True)
def _banded_dp_maxscore_jit(S: np.ndarray, gap: float, band: int):
    """Needleman-Wunsch restricted to a diagonal band, banded storage.

    Row ``i`` only stores columns ``[i-band, i+band]``, so memory is
    O(N * band) instead of O(N * M) — the full matrix defeated the point of
    banding (40 MB at 2000 x 2200).
    """
    N, M = S.shape
    width = 2 * band + 2
    neg = -1e18
    dp = np.full((N + 1, width), neg)
    bt = np.zeros((N + 1, width), dtype=np.int8)

    def_off = band  # column j of row i lives at j - i + band

    dp[0, def_off] = 0.0
    for j in range(1, min(M, band) + 1):
        k = j - 0 + def_off
        if 0 <= k < width:
            dp[0, k] = dp[0, k - 1] - gap
            bt[0, k] = 3

    for i in range(1, N + 1):
        jmin = max(0, i - band)
        jmax = min(M, i + band)
        for j in range(jmin, jmax + 1):
            k = j - i + def_off
            best = neg
            move = 0
            if j > 0:
                kd = (j - 1) - (i - 1) + def_off
                if 0 <= kd < width and dp[i - 1, kd] > neg:
                    cand = dp[i - 1, kd] + S[i - 1, j - 1]
                    if cand > best:
                        best = cand
                        move = 1
            ku = j - (i - 1) + def_off
            if 0 <= ku < width and dp[i - 1, ku] > neg:
                cand = dp[i - 1, ku] - gap
                if cand > best:
                    best = cand
                    move = 2
            if j > 0:
                kl = k - 1
                if 0 <= kl < width and dp[i, kl] > neg:
                    cand = dp[i, kl] - gap
                    if cand > best:
                        best = cand
                        move = 3
            dp[i, k] = best
            bt[i, k] = move

    kN = M - N + def_off
    final = dp[N, kN] if 0 <= kN < width else neg
    return bt, final, def_off


def _banded_dp_maxscore(S: np.ndarray, gap: float, band: int):
    N, M = S.shape
    if N == 0 or M == 0:
        return [], 0.0
    band = int(max(band, abs(N - M) + 1))
    bt, max_score, off = _banded_dp_maxscore_jit(
        np.ascontiguousarray(S, dtype=float), float(gap), band)
    width = bt.shape[1]
    i, j = N, M
    pairs: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        k = j - i + off
        if not (0 <= k < width):
            break
        move = bt[i, k]
        if move == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break
    pairs.reverse()
    return pairs, float(max_score)


def _shape_pairs(coords1: np.ndarray, coords2: np.ndarray, nbins: int = 24,
                 gap_penalty: float = 2.0, band_frac: float = 0.20):
    D1 = _pairwise_dists(coords1)
    D2 = _pairwise_dists(coords2)
    H1, _ = _radial_histograms(D1, nbins=nbins, rmax_mode="p98")
    H2, _ = _radial_histograms(D2, nbins=nbins, rmax_mode="p98")
    S = -_chi2_distance(H1, H2)
    N, M = S.shape
    band = max(3, int(band_frac * max(N, M)))
    pairs, _ = _banded_dp_maxscore(S, gap=gap_penalty, band=band)
    return pairs, S, -S


# ---------------------------------------------------------------------------
# sequence alignment and index-based pairing
# ---------------------------------------------------------------------------

class PairwiseAlignment:
    """Two gapped strings plus the raw score (the only thing callers need)."""

    __slots__ = ("seqA", "seqB", "score")

    def __init__(self, seqA: str, seqB: str, score: float):
        self.seqA = seqA
        self.seqB = seqB
        self.score = score

    @property
    def n_identical(self) -> int:
        return sum(1 for a, b in zip(self.seqA, self.seqB) if a == b and a != "-")

    @property
    def n_aligned(self) -> int:
        return sum(1 for a, b in zip(self.seqA, self.seqB) if a != "-" and b != "-")

    def identity(self, normalize: str = "aligned") -> float:
        """Percent identity. ``normalize``: 'aligned' (over aligned columns),
        'shorter' (over the shorter sequence) or 'alignment' (over all columns,
        terminal gaps included)."""
        n_id = self.n_identical
        if normalize == "alignment":
            denom = len(self.seqA)
        elif normalize == "shorter":
            la = sum(1 for c in self.seqA if c != "-")
            lb = sum(1 for c in self.seqB if c != "-")
            denom = min(la, lb)
        else:
            denom = self.n_aligned
        return 100.0 * n_id / denom if denom else 0.0


def _build_aligner(gap_open: float, gap_extend: float, mode: str = "global",
                   free_end_gaps: bool = True):
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.substitution_matrix = blosum62()
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    aligner.mode = mode
    if free_end_gaps and mode == "global":
        # Semi-global: a domain or a truncated construct should align inside a
        # longer chain without paying for the overhang.
        aligner.target_end_gap_score = 0.0
        aligner.query_end_gap_score = 0.0
    return aligner


def perform_sequence_alignment(seq1: str, seq2: str, gap_open: float = -10.0,
                               gap_extend: float = -0.5,
                               free_end_gaps: bool = True) -> Optional[PairwiseAlignment]:
    """Semi-global BLOSUM62 alignment of two sequences.

    Returns ``None`` only when an input is empty; a genuine alignment failure
    raises, because silently returning ``None`` sends callers down a fallback
    path with no explanation.
    """
    if not seq1 or not seq2:
        return None
    aligner = _build_aligner(gap_open, gap_extend, free_end_gaps=free_end_gaps)
    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return None
    best = alignments[0]
    ia, ib = best.indices
    out_a: List[str] = []
    out_b: List[str] = []
    for k in range(len(ia)):
        out_a.append(seq1[ia[k]] if ia[k] != -1 else "-")
        out_b.append(seq2[ib[k]] if ib[k] != -1 else "-")
    return PairwiseAlignment("".join(out_a), "".join(out_b), float(best.score))


def pairs_from_alignment(alignment: Optional[PairwiseAlignment]) -> List[Tuple[int, int]]:
    """Index pairs ``(i, j)`` into the two *selections* the alignment came from.

    Pairing is positional: the k-th non-gap character of ``seqA`` is residue k
    of the reference selection. No residue-name matching, no resynchronisation
    — the selection guarantees the correspondence.
    """
    if alignment is None:
        return []
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    for a, b in zip(alignment.seqA, alignment.seqB):
        if a != "-" and b != "-":
            pairs.append((i, j))
        if a != "-":
            i += 1
        if b != "-":
            j += 1
    return pairs


BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


class AtomRef:
    """A matched atom, carrying enough identity to label and group it.

    ``res_index`` is the position of the residue inside its Selection, which is
    what lets per-atom superposition results collapse back to per-residue
    reporting without guessing.
    """

    __slots__ = ("coord", "name", "chain_name", "res_seq", "res_icode",
                 "resname", "res_index")

    def __init__(self, coord, name, chain_name, res_seq, res_icode, resname,
                 res_index):
        self.coord = coord
        self.name = name
        self.chain_name = chain_name
        self.res_seq = res_seq
        self.res_icode = res_icode
        self.resname = resname
        self.res_index = res_index

    def get_coord(self):
        return self.coord

    def get_name(self):
        return self.name

    @property
    def label(self) -> str:
        ic = str(self.res_icode).strip()
        return f"{self.chain_name}:{self.res_seq}{ic}" if ic else \
            f"{self.chain_name}:{self.res_seq}"

    @property
    def residue_key(self):
        return (self.chain_name, self.res_seq, str(self.res_icode).strip())


def _atom_names_for(mode: str, like: bool) -> Optional[Tuple[str, ...]]:
    """Atom names to pair, given the requested mode and whether the two
    residues are of the same type.

    Side chains of different residue types share atom names without sharing
    chemistry (an ALA CB and a TRP CB point into different environments), so
    unlike pairs contribute backbone only. The sequence-free path already did
    this; the sequence-guided path did not, which made ``atoms="all_heavy"``
    quietly fit chemically unrelated atoms onto each other.
    """
    if mode == "CA":
        return ("CA",)
    if mode == "backbone":
        return BACKBONE_ATOM_NAMES
    if mode == "all_heavy":
        return None if like else BACKBONE_ATOM_NAMES
    raise ValueError(f"atoms must be 'CA', 'backbone' or 'all_heavy', got {mode!r}")


def paired_atoms(ref_sel: Selection, mob_sel: Selection,
                 pairs: Sequence[Tuple[int, int]], atoms: str = "CA"):
    """Matched atom lists for residue index pairs, honouring the atom mode."""
    ref_out: List[AtomRef] = []
    mob_out: List[AtomRef] = []
    for i, j in pairs:
        r = ref_sel.residues[i]
        m = mob_sel.residues[j]
        names = _atom_names_for(atoms, r.name == m.name)
        candidates = names if names is not None else tuple(r.atoms.keys())
        for name in candidates:
            rc = r.atoms.get(name)
            mc = m.atoms.get(name)
            if rc is None or mc is None:
                continue
            ref_out.append(AtomRef(rc, name, r.chain_id, r.seqid, r.icode, r.name, i))
            mob_out.append(AtomRef(mc, name, m.chain_id, m.seqid, m.icode, m.name, j))
    return ref_out, mob_out


def superimpose_atoms(ref_atoms: Sequence[AtomRef], mob_atoms: Sequence[AtomRef],
                      recycles: int = 0, keep_fraction: float = 1.0,
                      n_total: Optional[int] = None) -> Optional[dict]:
    """Superpose matched atoms and report at residue level.

    The returned ``per_residue_rmsd`` has one entry per *residue* (the RMS over
    that residue's matched atoms), so ``atoms="backbone"``/``"all_heavy"`` no
    longer inflate residue counts, coverage or the per-residue plot.

    ``n_total`` is the reference selection's residue count, used to normalize
    GDT_TS with CASP semantics (unaligned residues fail every cutoff).
    """
    if not ref_atoms or not mob_atoms or len(ref_atoms) != len(mob_atoms):
        return None
    ref_coords = np.array([a.coord for a in ref_atoms], dtype=float)
    mob_coords = np.array([a.coord for a in mob_atoms], dtype=float)

    R, t, rmsd, mask = _iterative_kabsch(ref_coords, mob_coords, recycles, keep_fraction)
    mob_aligned = _transform(mob_coords, R, t)
    sq = np.sum((ref_coords - mob_aligned) ** 2, axis=1)

    # Collapse atom-level deviations onto residues, preserving first-seen order.
    # Grouped on the residue *key* (chain, number, insertion code), not on the
    # index inside a Selection: the multi-chain path builds one selection per
    # chain, so indices restart at 0 for every chain and collapsing on them
    # merged chain A's residue 0 with chain B's (574 residues became 146).
    res_order: List[tuple] = []
    seen: Dict[tuple, int] = {}
    for a in ref_atoms:
        key = a.residue_key
        if key not in seen:
            seen[key] = len(res_order)
            res_order.append(key)
    n_res = len(res_order)
    sums = np.zeros(n_res)
    counts = np.zeros(n_res)
    ca_ref = np.full((n_res, 3), np.nan)
    ca_mob = np.full((n_res, 3), np.nan)
    labels: List[str] = [""] * n_res
    chains: List[str] = [""] * n_res
    keys: List[tuple] = [()] * n_res
    mob_labels: List[str] = [""] * n_res
    mob_chains: List[str] = [""] * n_res
    mob_keys: List[tuple] = [()] * n_res
    for k, a in enumerate(ref_atoms):
        slot = seen[a.residue_key]
        sums[slot] += sq[k]
        counts[slot] += 1
        if not labels[slot]:
            labels[slot] = a.label
            chains[slot] = a.chain_name
            keys[slot] = a.residue_key
            b = mob_atoms[k]
            mob_labels[slot] = b.label
            mob_chains[slot] = b.chain_name
            mob_keys[slot] = b.residue_key
        if a.name == "CA":
            ca_ref[slot] = ref_coords[k]
            ca_mob[slot] = mob_aligned[k]
    per_residue = np.sqrt(sums / np.maximum(counts, 1.0))

    has_ca = ~np.isnan(ca_ref[:, 0])
    if has_ca.any():
        ca_dists = np.linalg.norm(ca_ref[has_ca] - ca_mob[has_ca], axis=1)
    else:
        ca_dists = per_residue
    gdt_ts = compute_gdt_ts(ca_dists, n_total=n_total)

    active_ref = [a for k, a in enumerate(ref_atoms) if mask[k]]
    active_mob = [a for k, a in enumerate(mob_atoms) if mask[k]]

    return dict(rmsd=float(rmsd), rotation=R, translation=t,
                ref_coords=ref_coords, mob_coords_transformed=mob_aligned,
                per_residue_rmsd=per_residue, residue_labels=labels,
                residue_chains=chains, residue_keys=keys,
                residue_order=res_order,
                mob_residue_labels=mob_labels, mob_residue_chains=mob_chains,
                mob_residue_keys=mob_keys,
                ca_ref=ca_ref[has_ca], ca_mob=ca_mob[has_ca],
                n_residues=n_res,
                active_ref_atoms=active_ref, active_mob_atoms=active_mob,
                mask=mask, gdt_ts=gdt_ts)


# ---------------------------------------------------------------------------
# sequence-free alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignSummary:
    method: str
    rmsd: float
    inliers: int
    total_pairs: int
    iterations: int


@dataclass
class AlignmentResultSF:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    iterations: int
    kept_pairs: int
    method: str
    pairs: List[Tuple[int, int]]
    ref_subset_infos: List[ResidueInfo]
    mob_subset_infos: List[ResidueInfo]
    ref_subset_ca_coords: np.ndarray
    mob_subset_ca_coords_aligned: np.ndarray
    summaries: Dict[str, AlignSummary]
    shift_matrix: Optional[np.ndarray] = None
    shift_scores: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    gdt_ts: Optional[float] = None
    ref_selection: Optional[Selection] = None
    mob_selection: Optional[Selection] = None


def sequence_independent_alignment_joined_v2(
    file_ref: Union[str, gemmi.Structure],
    file_mob: Union[str, gemmi.Structure],
    chains_ref: Optional[Sequence[Union[str, int]]] = None,
    chains_mob: Optional[Sequence[Union[str, int]]] = None,
    method: str = "auto",
    shape_nbins: int = 24, shape_gap_penalty: float = 2.0,
    shape_band_frac: float = 0.20,
    recycles: int = 0, keep_fraction: float = 1.0,
    atoms: str = "CA", min_b_factor: float = 0.0, min_plddt: float = 0.0,
    ref_selection: Optional[Selection] = None,
    mob_selection: Optional[Selection] = None,
) -> AlignmentResultSF:
    """Superpose two structures without using their sequences.

    Accepts parsed structures (or pre-built selections) as well as paths, so
    callers that already hold a structure do not pay for a second parse — the
    old path-only signature re-read both files from disk on every call, which
    re-parsed the reference once per model of an ensemble.
    """
    if ref_selection is None:
        ref_struct = _as_structure(file_ref)
        ref_selection = select_residues(
            ref_struct, chains_ref, min_b_factor=min_b_factor,
            min_plddt=min_plddt, source=_name_of(file_ref))
    if mob_selection is None:
        mob_struct = _as_structure(file_mob)
        mob_selection = select_residues(
            mob_struct, chains_mob, min_b_factor=min_b_factor,
            min_plddt=min_plddt, source=_name_of(file_mob))

    n_ref, n_mob = ref_selection.n_residues, mob_selection.n_residues
    if max(n_ref, n_mob) > MAX_SEQFREE_RESIDUES:
        raise ValueError(
            f"Sequence-free alignment needs O(N^2) distance matrices and this "
            f"selection has {max(n_ref, n_mob)} residues (limit "
            f"{MAX_SEQFREE_RESIDUES}). Use mode='seq_guided', or restrict the "
            f"selection with chain/residue-range selectors.")

    ref_infos = selection_to_infos(ref_selection)
    mob_infos = selection_to_infos(mob_selection)
    ref_subset = ref_selection.ca_coords
    mob_subset = mob_selection.ca_coords

    summaries: Dict[str, AlignSummary] = {}
    candidates: Dict[str, dict] = {}
    shift_matrix = None
    shift_scores = None

    if method in ("shape", "auto"):
        pairs_s, S, _C = _shape_pairs(ref_subset, mob_subset, nbins=shape_nbins,
                                      gap_penalty=shape_gap_penalty,
                                      band_frac=shape_band_frac)
        shift_matrix = S
        R_s, t_s, rmsd_s = np.eye(3), np.zeros(3), float("inf")
        mask_s = np.ones(len(pairs_s), dtype=bool)
        if len(pairs_s) >= 3:
            P = ref_subset[[i for i, _ in pairs_s]]
            Q = mob_subset[[j for _, j in pairs_s]]
            R_s, t_s, rmsd_s, mask_s = _iterative_kabsch(P, Q, recycles, keep_fraction)
        summaries["shape"] = AlignSummary("shape", float(rmsd_s), int(np.sum(mask_s)),
                                          len(pairs_s), 1 + recycles)
        candidates["shape"] = dict(pairs=pairs_s, rmsd=rmsd_s, R=R_s, t=t_s, mask=mask_s)

    if method in ("window", "auto"):
        pairs_w, scores = _window_pairs(_pairwise_dists(ref_subset),
                                        _pairwise_dists(mob_subset))
        shift_scores = scores
        R_w, t_w, rmsd_w = np.eye(3), np.zeros(3), float("inf")
        mask_w = np.ones(len(pairs_w), dtype=bool)
        if len(pairs_w) >= 3:
            P = ref_subset[[i for i, _ in pairs_w]]
            Q = mob_subset[[j for _, j in pairs_w]]
            R_w, t_w, rmsd_w, mask_w = _iterative_kabsch(P, Q, recycles, keep_fraction)
        summaries["window"] = AlignSummary("window", float(rmsd_w), int(np.sum(mask_w)),
                                           len(pairs_w), 1 + recycles)
        candidates["window"] = dict(pairs=pairs_w, rmsd=rmsd_w, R=R_w, t=t_w, mask=mask_w)

    if not candidates:
        raise ValueError(f"Unknown sequence-free method {method!r}")
    if method in ("shape", "window"):
        chosen = method
    else:
        chosen = _select_seqfree_method({k: summaries[k] for k in candidates})

    R = candidates[chosen]["R"]
    t = candidates[chosen]["t"]
    final_pairs = candidates[chosen]["pairs"]
    final_rmsd = float(candidates[chosen]["rmsd"])
    final_mask = candidates[chosen]["mask"]

    # Refit on the requested atom set using the inlier residue pairs.
    if atoms != "CA" and len(final_pairs):
        inlier_pairs = [p for k, p in enumerate(final_pairs) if final_mask[k]]
        ref_atoms_l, mob_atoms_l = paired_atoms(ref_selection, mob_selection,
                                                inlier_pairs, atoms=atoms)
        if len(ref_atoms_l) >= 3:
            R, t, final_rmsd = _kabsch(
                np.array([a.coord for a in ref_atoms_l]),
                np.array([a.coord for a in mob_atoms_l]))

    mob_subset_aligned = _transform(mob_subset, R, t)
    gdt_ts = 0.0
    if final_pairs:
        dists = np.linalg.norm(
            ref_subset[[i for i, _ in final_pairs]]
            - mob_subset_aligned[[j for _, j in final_pairs]], axis=1)
        gdt_ts = compute_gdt_ts(dists, n_total=n_ref)

    logger.info("Seq-free alignment (%s): RMSD = %.3f, GDT_TS = %.2f",
                chosen, final_rmsd, gdt_ts)

    return AlignmentResultSF(
        rotation=R, translation=t, rmsd=final_rmsd, iterations=1,
        kept_pairs=int(np.sum(final_mask)), method=chosen, pairs=final_pairs,
        ref_subset_infos=ref_infos, mob_subset_infos=mob_infos,
        ref_subset_ca_coords=ref_subset,
        mob_subset_ca_coords_aligned=mob_subset_aligned,
        summaries=summaries,
        shift_matrix=shift_matrix if chosen == "shape" else None,
        shift_scores=shift_scores if chosen == "window" else None,
        active_mask=final_mask, gdt_ts=gdt_ts,
        ref_selection=ref_selection, mob_selection=mob_selection)


def _as_structure(x: Union[str, gemmi.Structure]) -> gemmi.Structure:
    if isinstance(x, gemmi.Structure):
        return x
    return _parse_path(str(x))


def _name_of(x: Union[str, gemmi.Structure]) -> str:
    import os
    if isinstance(x, gemmi.Structure):
        return x.name or "structure"
    return os.path.basename(str(x))


def _parse_path(path: str) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


# ---------------------------------------------------------------------------
# chain-level sequence comparison
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8192)
def _pair_identity(seq_a: str, seq_b: str) -> Tuple[float, float, float]:
    """(% identity over aligned columns, % over shorter chain, mean BLOSUM62).

    Memoised on the sequence pair: a homomultimer repeats the same comparison
    once per chain pair (576 identical alignments for a 24-mer).
    """
    if not seq_a or not seq_b:
        return 0.0, 0.0, 0.0
    if seq_a == seq_b:
        mat = blosum62()
        mean_score = float(np.mean([mat[(c, c)] for c in seq_a])) if seq_a else 0.0
        return 100.0, 100.0, mean_score
    aln = perform_sequence_alignment(seq_a, seq_b, -10.0, -0.5)
    if aln is None:
        return 0.0, 0.0, 0.0
    mat = blosum62()
    total = 0.0
    n = 0
    for a, b in zip(aln.seqA, aln.seqB):
        if a != "-" and b != "-":
            try:
                total += float(mat[(a, b)])
            except (KeyError, IndexError):
                pass
            n += 1
    return (aln.identity("aligned"), aln.identity("shorter"),
            total / n if n else 0.0)


def compute_chain_similarity_matrix(seqsA, seqsB, normalize: str = "shorter"):
    """Pairwise chain identity and mean-BLOSUM62 matrices as DataFrames.

    ``normalize="shorter"`` (the default) divides identities by the shorter
    chain's length. The previous normalisation — alignment length including
    terminal gaps — reported a perfectly matching 120-residue domain against
    its 600-residue parent chain as 20% identical, which is a sequence-length
    ratio dressed up as an identity and wrecked chain matching for truncated
    constructs, Fv fragments and single-domain models.
    """
    import pandas as pd
    chainsA = list(seqsA.keys())
    chainsB = list(seqsB.keys())
    if not chainsA or not chainsB:
        return pd.DataFrame(), pd.DataFrame()
    id_mat = np.full((len(chainsA), len(chainsB)), np.nan)
    sc_mat = np.full((len(chainsA), len(chainsB)), np.nan)
    col = 0 if normalize == "aligned" else 1
    for i, chA in enumerate(chainsA):
        sA = str(seqsA[chA].seq)
        for j, chB in enumerate(chainsB):
            sB = str(seqsB[chB].seq)
            if not sA or not sB:
                continue
            stats = _pair_identity(sA, sB)
            id_mat[i, j] = stats[col]
            sc_mat[i, j] = stats[2]
    return (pd.DataFrame(id_mat, index=chainsA, columns=chainsB),
            pd.DataFrame(sc_mat, index=chainsA, columns=chainsB))


# ---------------------------------------------------------------------------
# candidate selection
# ---------------------------------------------------------------------------

# Length scale (A) for the coverage-weighted selection score. Deviations much
# smaller than this barely change the score; larger ones are penalised.
_SELECTION_RMSD_SCALE = 3.0


def _coverage_score(rmsd: float, pairs: int) -> float:
    """``n_pairs / (1 + (rmsd / 3 A)^2)`` — higher is better.

    Rewards aligning more residues while still penalising deviation, so a
    strategy matching a handful of residues at near-zero RMSD cannot beat one
    that superimposes the whole protein well. At equal coverage it reduces to
    preferring the lower RMSD.
    """
    if not np.isfinite(rmsd) or pairs <= 0:
        return -np.inf
    return pairs / (1.0 + (rmsd / _SELECTION_RMSD_SCALE) ** 2)


def _select_seqfree_method(summaries: Dict[str, AlignSummary]) -> str:
    return max(summaries.keys(),
               key=lambda m: (_coverage_score(summaries[m].rmsd, summaries[m].inliers),
                              -summaries[m].rmsd))


def pick_best_overall(seqguided, seqfree, min_pairs: int = 3):
    """Choose between the sequence-guided and sequence-free candidates."""
    cands = []
    if seqguided is not None:
        si = seqguided["si"]
        n_res = si.get("n_residues")
        if not n_res:
            # Count residues, not atoms: with atoms="backbone"/"all_heavy" the
            # atom list holds several entries per residue, which would inflate
            # the coverage term and bias the choice toward the seq-guided side.
            ref_atoms = seqguided.get("ref_atoms") or []
            n_res = sum(1 for a in ref_atoms
                        if getattr(a, "get_name", lambda: "CA")() == "CA") \
                or len(ref_atoms)
        cands.append(dict(name="Sequence-guided", rmsd=float(si["rmsd"]),
                          pairs=int(n_res), kind="seqguided"))
    if seqfree is not None:
        cands.append(dict(name=f"Sequence-free ({seqfree.method})",
                          rmsd=float(seqfree.rmsd), pairs=int(seqfree.kept_pairs),
                          kind="seqfree"))
    if not cands:
        return None, "No candidates available."

    for c in cands:
        c["score"] = _coverage_score(c["rmsd"], c["pairs"])

    valid = [c for c in cands if np.isfinite(c["rmsd"]) and c["pairs"] >= min_pairs]
    if not valid:
        valid = [c for c in cands if np.isfinite(c["rmsd"])]
    if not valid:
        best = min(cands, key=lambda c: (not math.isfinite(c["rmsd"]), c["rmsd"]))
        return best, "Chose the only available candidate."
    best = max(valid, key=lambda c: (c["score"], -c["rmsd"]))
    others = [c for c in valid if c is not best]
    if others:
        alt = max(others, key=lambda c: (c["score"], -c["rmsd"]))
        reason = (f"Higher coverage-weighted score ({best['score']:.1f}: "
                  f"{best['pairs']} pairs @ {best['rmsd']:.2f} Å) vs {alt['name']} "
                  f"({alt['score']:.1f}: {alt['pairs']} pairs @ {alt['rmsd']:.2f} Å).")
    else:
        reason = "Single valid candidate."
    return best, reason


# ---------------------------------------------------------------------------
# legacy shim
# ---------------------------------------------------------------------------

def get_aligned_atoms_by_alignment(ref_struct, ref_chains, mob_struct, mob_chains,
                                   alignment, atoms: str = "CA",
                                   min_b_factor: float = 0.0, min_plddt: float = 0.0):
    """Deprecated. Build selections and use :func:`paired_atoms` instead.

    Retained for external callers. It re-derives both selections, so an
    alignment produced from *different* selections than the ones implied by the
    arguments would mis-pair; the selection-based API makes that impossible,
    which is why it is the one the package uses internally.
    """
    import warnings
    warnings.warn(
        "get_aligned_atoms_by_alignment() is deprecated; build a Selection with "
        "select_residues() and pair with pairs_from_alignment()/paired_atoms(), "
        "which cannot desynchronise the sequence from the coordinates.",
        DeprecationWarning, stacklevel=2)
    ref_sel = select_residues(ref_struct, ref_chains, min_b_factor=min_b_factor,
                              min_plddt=min_plddt)
    mob_sel = select_residues(mob_struct, mob_chains, min_b_factor=min_b_factor,
                              min_plddt=min_plddt)
    return paired_atoms(ref_sel, mob_sel, pairs_from_alignment(alignment), atoms=atoms)
