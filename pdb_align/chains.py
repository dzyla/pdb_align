"""Chain correspondence and multi-chain superposition strategy selection."""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .core import compute_chain_similarity_matrix, _extract_ca_infos, _kabsch

MAX_PERMUTE_CHAINS = 12
_TIE_TOL = 5.0  # % identity within which chains are treated as indistinguishable


@dataclass
class ChainMapping:
    pairs: List[Tuple[str, str, float, float]] = field(default_factory=list)
    unmatched_ref: List[str] = field(default_factory=list)
    unmatched_mob: List[str] = field(default_factory=list)


def match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct,
                 ref_chains, mob_chains) -> ChainMapping:
    """Optimal 1:1 chain correspondence via Hungarian assignment on % identity,
    refined by superposition for near-identical (homomultimer) chains."""
    from scipy.optimize import linear_sum_assignment

    r_seqs = {c: ref_seqs[c] for c in ref_chains if c in ref_seqs}
    m_seqs = {c: mob_seqs[c] for c in mob_chains if c in mob_seqs}
    r_ids, m_ids = list(r_seqs.keys()), list(m_seqs.keys())
    if not r_ids or not m_ids:
        return ChainMapping(unmatched_ref=r_ids, unmatched_mob=m_ids)

    id_mat, _ = compute_chain_similarity_matrix(r_seqs, m_seqs)
    mat = np.nan_to_num(np.asarray(id_mat, dtype=float), nan=0.0)

    # Hungarian maximises identity -> minimise negative identity.
    row_idx, col_idx = linear_sum_assignment(-mat)
    pairs = []
    used_r, used_m = set(), set()
    for ri, ci in zip(row_idx, col_idx):
        ident = float(mat[ri, ci])
        if ident <= 0.0:
            continue
        pairs.append((r_ids[ri], m_ids[ci], ident, ident))
        used_r.add(r_ids[ri]); used_m.add(m_ids[ci])

    mapping = ChainMapping(
        pairs=pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
    )

    # Refine homomultimers where sequence identity ties across candidates.
    if _needs_permutation_refinement(mat) and ref_struct is not None:
        mapping = _refine_by_superposition(
            mapping, ref_struct, mob_struct, r_ids, m_ids, mat)
    return mapping


def _needs_permutation_refinement(mat: np.ndarray) -> bool:
    """True if any ref chain has >=2 mob candidates within _TIE_TOL % identity."""
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return False
    for row in mat:
        top = np.sort(row)[::-1]
        if top[0] > 0 and (top[0] - top[1]) <= _TIE_TOL:
            return True
    return False


def _refine_by_superposition(mapping, ref_struct, mob_struct, r_ids, m_ids, mat):
    """Centroid-ICP refinement: superpose on current mapping, then reassign
    chains by post-superposition centroid proximity. Capped by MAX_PERMUTE_CHAINS."""
    from scipy.optimize import linear_sum_assignment
    if len(r_ids) > MAX_PERMUTE_CHAINS:
        return mapping  # keep Hungarian result on very large complexes

    def centroids(struct, ids):
        out = {}
        for c in ids:
            infos = _extract_ca_infos(struct, [c])
            if infos:
                out[c] = np.mean([i.coord for i in infos], axis=0)
        return out

    r_cen = centroids(ref_struct, r_ids)
    m_cen = centroids(mob_struct, m_ids)
    # Need >=2 pairs for the reassignment step to be meaningful (with a single
    # pair there is nothing to permute). Note: _kabsch already has a safe
    # translation-only fallback for K<3 points, so we don't need to require
    # 3 chains here -- 2-chain homodimers are refined fine via that fallback.
    if len(mapping.pairs) < 2:
        return mapping

    # Superpose using current pairing's centroids.
    P = np.array([r_cen[a] for a, b, *_ in mapping.pairs if a in r_cen and b in m_cen])
    Q = np.array([m_cen[b] for a, b, *_ in mapping.pairs if a in r_cen and b in m_cen])
    if len(P) < 2:
        return mapping
    R, t, _ = _kabsch(P, Q)

    common_r = [c for c in r_ids if c in r_cen]
    common_m = [c for c in m_ids if c in m_cen]
    cost = np.zeros((len(common_r), len(common_m)))
    for i, a in enumerate(common_r):
        for j, b in enumerate(common_m):
            moved = R @ m_cen[b] + t
            cost[i, j] = np.linalg.norm(r_cen[a] - moved)
    ri, ci = linear_sum_assignment(cost)
    id_lookup = {
        (r_ids[i], m_ids[j]): float(mat[i, j])
        for i in range(len(r_ids)) for j in range(len(m_ids))
    }
    new_pairs = []
    for i, j in zip(ri, ci):
        a, b = common_r[i], common_m[j]
        ident = id_lookup.get((a, b), 0.0)
        new_pairs.append((a, b, ident, ident))
    used_r = {p[0] for p in new_pairs}
    used_m = {p[1] for p in new_pairs}
    return ChainMapping(
        pairs=new_pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
    )


@dataclass
class MultiChainResult:
    strategy: str
    mapping: "ChainMapping"
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    pairs: list
    ref_coords: np.ndarray
    mob_coords_aligned: np.ndarray
    per_chain: list
    ref_infos: list
    mob_infos: list


def _coverage_score(n_pairs: int, rmsd: float) -> float:
    return n_pairs / (1.0 + (rmsd / 3.0) ** 2)


def _paired_ca(ref_struct, mob_struct, mapping, atoms, min_b_factor, min_plddt):
    """Return matched CA coords + infos across all mapped chains (by residue order)."""
    ref_infos, mob_infos = [], []
    for a, b, *_ in mapping.pairs:
        ri = _extract_ca_infos(ref_struct, [a], min_b_factor, min_plddt)
        mi = _extract_ca_infos(mob_struct, [b], min_b_factor, min_plddt)
        n = min(len(ri), len(mi))
        ref_infos.extend(ri[:n]); mob_infos.extend(mi[:n])
    P = np.array([i.coord for i in ref_infos]) if ref_infos else np.empty((0, 3))
    Q = np.array([i.coord for i in mob_infos]) if mob_infos else np.empty((0, 3))
    return P, Q, ref_infos, mob_infos


def _per_chain_rmsd(mapping, ref_struct, mob_struct, R, t, min_b_factor, min_plddt):
    rows = []
    for a, b, *_ in mapping.pairs:
        ri = _extract_ca_infos(ref_struct, [a], min_b_factor, min_plddt)
        mi = _extract_ca_infos(mob_struct, [b], min_b_factor, min_plddt)
        n = min(len(ri), len(mi))
        if n == 0:
            continue
        P = np.array([x.coord for x in ri[:n]])
        Q = np.array([x.coord for x in mi[:n]])
        Qa = (R @ Q.T).T + t
        rmsd = float(np.sqrt(np.mean(np.sum((P - Qa) ** 2, axis=1))))
        rows.append({"chain_ref": a, "chain_mob": b, "n_residues": n, "rmsd": rmsd})
    return rows


def align_multichain(ref_struct, mob_struct, mapping, strategy="auto",
                     atoms="CA", min_b_factor=0.0, min_plddt=0.0) -> MultiChainResult:
    """Build global and/or local superpositions and pick per coverage-weighted score."""
    def build(sub_mapping, name):
        P, Q, ri, mi = _paired_ca(ref_struct, mob_struct, sub_mapping,
                                  atoms, min_b_factor, min_plddt)
        if len(P) < 3:
            return None
        R, t, rmsd = _kabsch(P, Q)
        Qa = (R @ Q.T).T + t
        per_chain = _per_chain_rmsd(sub_mapping, ref_struct, mob_struct, R, t,
                                    min_b_factor, min_plddt)
        return MultiChainResult(
            strategy=name, mapping=sub_mapping, rotation=R, translation=t,
            rmsd=rmsd, pairs=list(range(len(P))), ref_coords=P,
            mob_coords_aligned=Qa, per_chain=per_chain,
            ref_infos=ri, mob_infos=mi)

    global_res = build(mapping, "global")

    # local candidate = single best-identity chain pair
    local_res = None
    if mapping.pairs:
        best = max(mapping.pairs, key=lambda p: p[2])
        local_map = ChainMapping(pairs=[best])
        local_res = build(local_map, "local")

    if strategy == "global":
        if global_res is None:
            raise ValueError("global strategy requested but no viable multi-chain superposition.")
        return global_res
    if strategy == "local":
        if local_res is None:
            raise ValueError("local strategy requested but no viable single chain pair.")
        return local_res

    # auto: pick by coverage-weighted score
    candidates = [c for c in (global_res, local_res) if c is not None]
    if not candidates:
        raise ValueError("No viable multi-chain superposition could be produced.")
    return max(candidates, key=lambda c: _coverage_score(len(c.ref_coords), c.rmsd))
