"""Chain correspondence and multi-chain superposition strategy selection.

Two decisions live here:

1. **Which reference chain corresponds to which model chain.** Optimal 1:1
   assignment on percent sequence identity (Hungarian), refined geometrically
   when sequence cannot tell copies of the same chain apart — the homomultimer
   case, where identity is tied by construction and only geometry decides.
2. **What to superpose on.** A *global* fit over every mapped chain, or a
   *local* fit on the single best chain pair, chosen by the same
   coverage-weighted score used elsewhere in the package.

Both decisions are reported on the result (``strategy``, ``chain_mapping``,
``per_chain``) rather than left implicit, and a correspondence that rests on
near-random sequence identity is flagged instead of being presented as fact.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    Selection,
    _coverage_score,
    _kabsch,
    compute_chain_similarity_matrix,
    paired_atoms,
    pairs_from_alignment,
    perform_sequence_alignment,
    select_residues,
)

MAX_PERMUTE_CHAINS = 24
_TIE_TOL = 5.0          # % identity within which chains are indistinguishable
_REFINE_MAX_ROUNDS = 10  # ICP rounds; it converges in 2-3 on real assemblies
# Added to the centroid distance of a pairing that sequence rules out, so the
# geometric reassignment can only permute interchangeable chains.
_INCOMPATIBLE_PENALTY = 1.0e6

# A correspondence resting on this little sequence identity is not evidence of
# homology: unrelated protein chains align at roughly 10-20% identity by
# chance. Pairs below the floor are still reported (dropping them silently
# would be worse) but carry an explicit warning.
WEAK_IDENTITY = 25.0
# A pair whose alignment covers less of the shorter chain than this is a
# fragment match, not a chain correspondence.
MIN_PAIR_COVERAGE = 20.0


@dataclass
class ChainMapping:
    pairs: List[Tuple[str, str, float, float]] = field(default_factory=list)
    unmatched_ref: List[str] = field(default_factory=list)
    unmatched_mob: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    refined: bool = False

    @property
    def min_identity(self) -> Optional[float]:
        return min((p[2] for p in self.pairs), default=None)


def match_chains(ref_seqs, mob_seqs, ref_struct=None, mob_struct=None,
                 ref_chains=None, mob_chains=None) -> ChainMapping:
    """Optimal 1:1 chain correspondence, geometrically refined when tied."""
    from scipy.optimize import linear_sum_assignment

    ref_chains = list(ref_chains) if ref_chains is not None else list(ref_seqs)
    mob_chains = list(mob_chains) if mob_chains is not None else list(mob_seqs)
    r_seqs = {c: ref_seqs[c] for c in ref_chains if c in ref_seqs}
    m_seqs = {c: mob_seqs[c] for c in mob_chains if c in mob_seqs}
    r_ids, m_ids = list(r_seqs), list(m_seqs)
    if not r_ids or not m_ids:
        return ChainMapping(unmatched_ref=r_ids, unmatched_mob=m_ids)

    id_mat, _ = compute_chain_similarity_matrix(r_seqs, m_seqs, normalize="shorter")
    mat = np.nan_to_num(np.asarray(id_mat, dtype=float), nan=0.0)

    row_idx, col_idx = linear_sum_assignment(-mat)
    pairs: List[Tuple[str, str, float, float]] = []
    used_r, used_m = set(), set()
    for ri, ci in zip(row_idx, col_idx):
        ident = float(mat[ri, ci])
        if ident <= 0.0:
            continue
        pairs.append((r_ids[ri], m_ids[ci], ident, ident))
        used_r.add(r_ids[ri])
        used_m.add(m_ids[ci])

    mapping = ChainMapping(
        pairs=pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
    )

    if _needs_permutation_refinement(mat) and ref_struct is not None and mob_struct is not None:
        mapping = _refine_by_superposition(mapping, ref_struct, mob_struct,
                                           r_ids, m_ids, mat)

    mapping.warnings = _mapping_warnings(mapping, r_seqs, m_seqs)
    # Only the scientifically material notes are raised as Python warnings.
    # Leftover chains are routine (comparing a two-chain receptor against every
    # chain of a model always leaves some), so that note stays on the mapping
    # and reaches the user through the report instead of the warning stream.
    for msg in mapping.warnings:
        if msg.startswith(("Chain correspondence rests", "Chain pair")):
            warnings.warn(msg, UserWarning, stacklevel=2)
    return mapping


def _mapping_warnings(mapping: ChainMapping, r_seqs, m_seqs) -> List[str]:
    msgs: List[str] = []
    weak = [(a, b, i) for a, b, i, _ in mapping.pairs if i < WEAK_IDENTITY]
    if weak:
        detail = ", ".join(f"{a}->{b} ({i:.0f}%)" for a, b, i in weak)
        msgs.append(
            f"Chain correspondence rests on weak sequence identity ({detail}); "
            f"unrelated chains reach ~10-20% by chance, so treat the mapping — "
            f"and every number derived from it — as unverified.")
    short = []
    for a, b, _i, _s in mapping.pairs:
        la = len(str(r_seqs[a].seq)) if a in r_seqs else 0
        lb = len(str(m_seqs[b].seq)) if b in m_seqs else 0
        if la and lb:
            ratio = 100.0 * min(la, lb) / max(la, lb)
            if ratio < MIN_PAIR_COVERAGE:
                short.append(f"{a}({la} aa)->{b}({lb} aa)")
    if short:
        msgs.append(
            f"Chain pair(s) {', '.join(short)} differ in length by more than "
            f"5x; the shorter chain may be a fragment or a different entity.")
    if mapping.unmatched_ref or mapping.unmatched_mob:
        msgs.append(
            f"Unmatched chains — reference: {mapping.unmatched_ref or 'none'}, "
            f"mobile: {mapping.unmatched_mob or 'none'}; they contribute to "
            f"coverage but not to the superposition.")
    return msgs


def _needs_permutation_refinement(mat: np.ndarray) -> bool:
    """True if any reference chain has >=2 candidates within _TIE_TOL identity."""
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return False
    for row in mat:
        top = np.sort(row)[::-1]
        if top[0] > 0 and (top[0] - top[1]) <= _TIE_TOL:
            return True
    return False


def _chain_centroids(struct, ids) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ids:
        try:
            sel = select_residues(struct, [c], with_atoms=False)
        except ValueError:
            continue
        out[c] = sel.ca_coords.mean(axis=0)
    return out


def _refine_by_superposition(mapping, ref_struct, mob_struct, r_ids, m_ids, mat):
    """Iterative centroid ICP: superpose on the current mapping, reassign by
    post-superposition centroid proximity, repeat until the mapping stops
    changing.

    Sequence identity is exactly tied between copies of the same chain, so the
    Hungarian assignment picks among them arbitrarily and can pair chain A of
    the reference with the *wrong* copy in the model — a correspondence error
    that inflates RMSD without any other symptom. Geometry is the only signal
    that distinguishes them. A single round is not always enough: the first fit
    is computed on a possibly-wrong mapping, so the reassignment it implies can
    itself be improved. The loop is capped and returns the best-scoring round.
    """
    from scipy.optimize import linear_sum_assignment

    if len(r_ids) > MAX_PERMUTE_CHAINS:
        return mapping
    r_cen = _chain_centroids(ref_struct, r_ids)
    m_cen = _chain_centroids(mob_struct, m_ids)
    if len(mapping.pairs) < 2:
        return mapping

    id_lookup = {(r_ids[i], m_ids[j]): float(mat[i, j])
                 for i in range(len(r_ids)) for j in range(len(m_ids))}
    common_r = [c for c in r_ids if c in r_cen]
    common_m = [c for c in m_ids if c in m_cen]
    if len(common_r) < 2 or len(common_m) < 2:
        return mapping

    # Geometry breaks ties; it does not overrule sequence. A pair whose
    # identity is well below the best available for that reference chain is
    # forbidden, so a chain can only be reassigned among candidates sequence
    # says are interchangeable. Without this the reassignment cost is pure
    # centroid distance, and on a dimer of heterodimers it happily paired
    # α-globin with β-globin because their centroids happened to be closer —
    # which drove fnat to 0 while every other number still looked plausible.
    r_index = {c: i for i, c in enumerate(r_ids)}
    m_index = {c: j for j, c in enumerate(m_ids)}

    def _compatible(a: str, b: str) -> bool:
        i, j = r_index.get(a), m_index.get(b)
        if i is None or j is None:
            return True
        row = mat[i]
        best = float(np.max(row)) if row.size else 0.0
        return float(mat[i, j]) >= max(best - _TIE_TOL, 0.0)

    current = [(a, b) for a, b, *_ in mapping.pairs if a in r_cen and b in m_cen]
    best_pairs, best_cost, refined = current, np.inf, False
    seen = set()

    for _round in range(_REFINE_MAX_ROUNDS):
        key = tuple(current)
        if key in seen:
            break
        seen.add(key)
        P = np.array([r_cen[a] for a, b in current])
        Q = np.array([m_cen[b] for a, b in current])
        if len(P) < 2:
            break
        R, t, _ = _kabsch(P, Q)
        moved = {b: R @ m_cen[b] + t for b in common_m}
        # A large finite penalty rather than inf: the assignment must stay
        # solvable even when no compatible pairing exists for some chain.
        cost = np.array([[np.linalg.norm(r_cen[a] - moved[b])
                          + (0.0 if _compatible(a, b) else _INCOMPATIBLE_PENALTY)
                          for b in common_m]
                         for a in common_r])
        ri, ci = linear_sum_assignment(cost)
        total = float(cost[ri, ci].sum())
        proposal = [(common_r[i], common_m[j]) for i, j in zip(ri, ci)]
        if total < best_cost - 1e-9:
            best_cost, best_pairs = total, proposal
            refined = refined or (proposal != current)
        if proposal == current:
            break
        current = proposal

    new_pairs = [(a, b, id_lookup.get((a, b), 0.0), id_lookup.get((a, b), 0.0))
                 for a, b in best_pairs]
    used_r = {p[0] for p in new_pairs}
    used_m = {p[1] for p in new_pairs}
    return ChainMapping(
        pairs=new_pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
        refined=refined,
    )


# ---------------------------------------------------------------------------
# multi-chain superposition
# ---------------------------------------------------------------------------

@dataclass
class MultiChainResult:
    strategy: str
    mapping: ChainMapping
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    pairs: list
    ref_coords: np.ndarray
    mob_coords_aligned: np.ndarray
    per_chain: list
    ref_infos: list
    mob_infos: list
    ref_selection: Optional[Selection] = None
    mob_selection: Optional[Selection] = None


def _chain_selections(struct, chain_ids, min_b_factor, min_plddt, source=""):
    """One Selection per chain, built once and reused by both strategies."""
    out: Dict[str, Selection] = {}
    for c in chain_ids:
        try:
            out[c] = select_residues(struct, [c], min_b_factor=min_b_factor,
                                     min_plddt=min_plddt, source=source)
        except ValueError:
            continue
    return out


def _matched_atoms_per_chain(ref_sels, mob_sels, mapping, atoms):
    """Per mapped chain pair, residues paired by that pair's own sequence
    alignment (so unmodelled loops shift nothing downstream)."""
    out = []
    for a, b, *_ in mapping.pairs:
        ref_sel, mob_sel = ref_sels.get(a), mob_sels.get(b)
        if ref_sel is None or mob_sel is None:
            continue
        aln = perform_sequence_alignment(ref_sel.sequence, mob_sel.sequence,
                                         -10.0, -0.5)
        pairs = pairs_from_alignment(aln)
        if not pairs:
            continue
        ratoms, matoms = paired_atoms(ref_sel, mob_sel, pairs, atoms=atoms)
        if not ratoms:
            continue
        out.append((a, b, ratoms, matoms, len(pairs)))
    return out


def _per_chain_rows(per_chain_matches, R, t):
    rows = []
    for a, b, ra, ma, n_res in per_chain_matches:
        if not ra:
            continue
        P = np.array([x.coord for x in ra])
        Q = np.array([x.coord for x in ma])
        Qa = (R @ Q.T).T + t
        rows.append({"chain_ref": a, "chain_mob": b, "n_residues": int(n_res),
                     "rmsd": float(np.sqrt(np.mean(np.sum((P - Qa) ** 2, axis=1))))})
    return rows


def align_multichain(ref_struct, mob_struct, mapping, strategy: str = "auto",
                     atoms: str = "CA", min_b_factor: float = 0.0,
                     min_plddt: float = 0.0,
                     ref_selections=None, mob_selections=None) -> MultiChainResult:
    """Build the global and local superpositions and pick one.

    ``strategy="auto"`` takes the higher coverage-weighted score; ``"global"``
    and ``"local"`` force the choice. Selections are built once and shared by
    both candidates — rebuilding them per candidate meant parsing sequences and
    re-running every chain's alignment twice.
    """
    ref_ids = [a for a, _b, *_ in mapping.pairs]
    mob_ids = [b for _a, b, *_ in mapping.pairs]
    if ref_selections is None:
        ref_selections = _chain_selections(ref_struct, ref_ids, min_b_factor, min_plddt)
    if mob_selections is None:
        mob_selections = _chain_selections(mob_struct, mob_ids, min_b_factor, min_plddt)

    def build(sub_mapping, name):
        matches = _matched_atoms_per_chain(ref_selections, mob_selections,
                                           sub_mapping, atoms)
        ref_atoms, mob_atoms = [], []
        for _a, _b, ra, ma, _n in matches:
            ref_atoms.extend(ra)
            mob_atoms.extend(ma)
        if len(ref_atoms) < 3:
            return None
        P = np.array([x.coord for x in ref_atoms])
        Q = np.array([x.coord for x in mob_atoms])
        R, t, rmsd = _kabsch(P, Q)
        return MultiChainResult(
            strategy=name, mapping=sub_mapping, rotation=R, translation=t,
            rmsd=rmsd, pairs=list(range(len(P))), ref_coords=P,
            mob_coords_aligned=(R @ Q.T).T + t,
            per_chain=_per_chain_rows(matches, R, t),
            ref_infos=ref_atoms, mob_infos=mob_atoms)

    global_res = build(mapping, "global")
    local_res = None
    if mapping.pairs:
        best = max(mapping.pairs, key=lambda p: p[2])
        local_res = build(ChainMapping(pairs=[best], warnings=mapping.warnings),
                          "local")

    if strategy == "global":
        if global_res is None:
            raise ValueError("strategy='global' requested but no viable "
                             "multi-chain superposition could be built.")
        return global_res
    if strategy == "local":
        if local_res is None:
            raise ValueError("strategy='local' requested but no viable single "
                             "chain pair could be superposed.")
        return local_res

    candidates = [c for c in (global_res, local_res) if c is not None]
    if not candidates:
        raise ValueError("No viable multi-chain superposition could be produced.")
    return max(candidates,
               key=lambda c: _coverage_score(c.rmsd, len(c.ref_coords)))
