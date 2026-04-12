import os
import gemmi
import io
import math
import json
import tempfile
import datetime
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# BioPython
from Bio.PDB import (
    PDBParser, MMCIFParser, PDBIO, Select,
    Structure, Atom, Superimposer
)
from Bio.PDB.Polypeptide import protein_letters_3to1, standard_aa_names, is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from Bio.PDB import Structure as _Structure

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

VALID_AA_3 = set(standard_aa_names)
def _aa_dict() -> Dict[str, str]:
    d = {k: protein_letters_3to1[k] for k in VALID_AA_3}
    d['MSE'] = 'M'
    return d
AA_DICT = _aa_dict()

def extract_sequences_and_lengths(struct: gemmi.Structure, fname: str):
    seqs: Dict[str, SeqRecord] = {}
    lens: Dict[str, int] = {}
    try:
        model = struct[0]
    except Exception:
        return {}, {}
    for chain in model:
        seq = []
        ca_count = 0
        for res in chain:
            resname = res.name
            if resname in AA_DICT:
                seq.append(AA_DICT[resname])
                has_ca = False
                for atom in res:
                    if atom.name == "CA":
                        has_ca = True
                        break
                if has_ca:
                    ca_count += 1
        if seq:
            seqs[chain.name] = SeqRecord(
                Seq("".join(seq)),
                id=f"{fname}_{chain.name}",
                description=f"Chain {chain.name}",
            )
            lens[chain.name] = int(ca_count)
    return seqs, lens

@dataclass
class ResidueInfo:
    idx: int
    chain_id: str
    resseq: int
    icode: str
    resname: str
    coord: np.ndarray

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
    pairs: List[Tuple[int,int]]
    ref_subset_infos: List[ResidueInfo]
    mob_subset_infos: List[ResidueInfo]
    ref_subset_ca_coords: np.ndarray
    mob_subset_ca_coords_aligned: np.ndarray
    mob_all_infos: List[ResidueInfo]
    mob_all_ca_coords_aligned: np.ndarray
    summaries: Dict[str, AlignSummary]
    shift_matrix: Optional[np.ndarray] = None
    shift_scores: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    gdt_ts: Optional[float] = None
    cad_score: Optional[float] = None

def compute_gdt_ts(dists: np.ndarray, cutoffs: List[float] = [1.0, 2.0, 4.0, 8.0]) -> float:
    if len(dists) == 0: return 0.0
    fractions = [np.mean(dists <= c) for c in cutoffs]
    return float(np.mean(fractions)) * 100.0

def compute_cad_score_approx(ref_coords: np.ndarray, mob_coords: np.ndarray, contact_dist: float = 8.0) -> float:
    # Pseudo-CAD using purely C-alpha contact maps (O(N^2) naive). Real CAD requires Voronoi, but this captures local environment change.
    if len(ref_coords) == 0: return 0.0
    d_ref = _pairwise_dists(ref_coords)
    d_mob = _pairwise_dists(mob_coords)
    # Mask out self-contacts and adjacent residues
    N = len(ref_coords)
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    for i in range(N-1):
        mask[i, i+1] = False
        mask[i+1, i] = False

    c_ref = (d_ref <= contact_dist) & mask
    c_mob = (d_mob <= contact_dist) & mask
    
    # Jaccard index of contacts
    intersection = np.sum(c_ref & c_mob)
    union = np.sum(c_ref | c_mob)
    return float(intersection / union) if union > 0 else 0.0

def _parse_path(path: str) -> gemmi.Structure:
    return gemmi.read_structure(path)

def _chain_ids(struct: gemmi.Structure) -> List[str]:
    return [ch.name for ch in struct[0]]

def _parse_chain_selector(selector: str) -> Tuple[str, Optional[int], Optional[int]]:
    if ":" in selector:
        parts = selector.split(":")
        chain_id = parts[0]
        range_str = parts[1]
        if "-" in range_str:
            bounds = range_str.split("-")
            return chain_id, int(bounds[0]) if bounds[0] else None, int(bounds[1]) if bounds[1] else None
        else:
            return chain_id, int(range_str), int(range_str)
    return selector, None, None

def _resolve_selectors(struct:_Structure.Structure, sel: Optional[List[Union[str,int]]]) -> Optional[List[str]]:
    if sel is None: return None
    ids = _chain_ids(struct)
    out=[]
    for x in sel:
        if isinstance(x,int):
            if not (1<=x<=len(ids)): raise ValueError(f"Chain index {x} out of range 1..{len(ids)}")
            out.append(ids[x-1])
        else:
            # x could be "A:10-150". Check if the pure chain is in ids.
            cid, _, _ = _parse_chain_selector(x)
            if cid not in ids: raise ValueError(f"Chain '{cid}' (from '{x}') not in {ids}")
            out.append(x)
    seen=set(); uniq=[c for c in out if not (c in seen or seen.add(c))]
    return uniq

def _extract_ca_infos(
    struct: gemmi.Structure, chain_filter: Optional[List[str]], min_b_factor: float = 0.0, min_plddt: float = 0.0
) -> List[ResidueInfo]:
    model = struct[0]
    infos = []
    idx = 0

    # Parse bounds
    bounds_map = {}
    valid_chain_ids = set()
    if chain_filter is not None:
        for c in chain_filter:
            cid, start, end = _parse_chain_selector(c)
            valid_chain_ids.add(cid)
            if cid not in bounds_map:
                bounds_map[cid] = []
            if start is not None or end is not None:
                bounds_map[cid].append((start, end))

    for chain in model:
        if chain_filter is not None and chain.name not in valid_chain_ids:
            continue

        c_bounds = bounds_map.get(chain.name, [])

        for res in chain:
            if res.name not in AA_DICT:
                continue

            resseq = res.seqid.num
            icode = res.seqid.icode if hasattr(res.seqid, 'has_icode') and res.seqid.has_icode() else ""
            if not icode and hasattr(res.seqid, 'icode') and res.seqid.icode != ' ':
                icode = res.seqid.icode

            # Sub-domain filtering
            if c_bounds:
                in_bounds = False
                for start, end in c_bounds:
                    if (start is None or resseq >= start) and (
                        end is None or resseq <= end
                    ):
                        in_bounds = True
                        break
                if not in_bounds:
                    continue

            ca = None
            for atom in res:
                if atom.name == "CA":
                    ca = atom
                    break

            if ca is None:
                continue

            if min_b_factor > 0.0 and ca.b_iso < min_b_factor:
                continue
            
            # Since AF models store pLDDT in B-factor column
            if min_plddt > 0.0 and ca.b_iso < min_plddt:
                continue

            coord = np.array(ca.pos.tolist(), dtype=float)
            infos.append(
                ResidueInfo(
                    idx=idx,
                    chain_id=chain.name,
                    resseq=int(resseq),
                    icode=icode.strip() if hasattr(icode, 'strip') else "",
                    resname=res.name,
                    coord=coord,
                )
            )
            idx += 1

    if not infos:
        raise ValueError(f"No C-alpha atoms found for selected chains.")
    return infos
def _pairwise_dists(coords: np.ndarray) -> np.ndarray:
    x=coords; x2=np.sum(x*x, axis=1, keepdims=True)
    d2=x2+x2.T-2.0*np.dot(x,x.T); np.maximum(d2,0.0,out=d2)
    return np.sqrt(d2, out=d2)

@jit(nopython=True, cache=True)
def _sliding_window_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute per-element sliding-window mean. JIT-compiled for speed."""
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


def _detect_hinges(
    per_residue_rmsd: np.ndarray,
    window: int = 15,
    threshold: float = 3.0,
    min_segment: int = 30,
) -> List[int]:
    """
    Return 0-based split indices for a per-residue RMSD array.

    A "split at index s" means the first segment is [0..s-1] and the next
    begins at [s..]. Consecutive above-threshold positions are merged into one
    hinge; the split is placed at the midpoint of the hinge region.
    Splits that would produce segments shorter than *min_segment* are dropped.
    """
    N = len(per_residue_rmsd)
    if N < 2 * min_segment:
        return []

    smoothed = _sliding_window_mean(per_residue_rmsd.astype(float), window)
    hinge_mask = smoothed > threshold

    # Identify contiguous hinge regions, pick midpoint of each as the split
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

    # Drop splits that leave segments shorter than min_segment
    filtered: List[int] = []
    prev = 0
    for s in splits:
        if s - prev >= min_segment:
            filtered.append(s)
            prev = s
    if filtered and (N - filtered[-1]) < min_segment:
        filtered.pop()

    return filtered

def _kabsch(P:np.ndarray, Q:np.ndarray)->Tuple[np.ndarray,np.ndarray,float]:
    if P.shape != Q.shape or P.shape[1]!=3: raise ValueError("Kabsch expects matched (K,3)")
    K=P.shape[0]
    if K<3:
        cP=P.mean(axis=0); cQ=Q.mean(axis=0); R=np.eye(3); t=cP - cQ
        rmsd=float(np.sqrt(np.mean(np.sum((P-(Q+t))**2, axis=1)))); return R,t,rmsd
    cP=P.mean(axis=0); cQ=Q.mean(axis=0)
    P0=P-cP; Q0=Q-cQ; H=Q0.T@P0
    U,S,Vt=np.linalg.svd(H); R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1,:]*=-1.0; R=Vt.T@U.T
    t=cP - R@cQ
    Q_aln=(R@Q.T).T + t
    rmsd=float(np.sqrt(np.mean(np.sum((P-Q_aln)**2, axis=1))))
    return R,t,rmsd

def _iterative_kabsch(P: np.ndarray, Q: np.ndarray, recycles: int, keep_fraction: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    R, t, rmsd = _kabsch(P, Q)
    N = P.shape[0]
    min_keep = max(3, int(round(N * keep_fraction)))
    mask = np.ones(N, dtype=bool)

    for _ in range(recycles):
        Q_aln = (R @ Q.T).T + t
        d = np.linalg.norm(P - Q_aln, axis=1)

        # Only consider currently active pairs
        active_d = d[mask]
        current_rmsd = float(np.sqrt(np.mean(active_d**2))) if len(active_d) > 0 else 0.0

        cut = max(2.0, 1.5 * current_rmsd)
        
        # Determine which pairs are good enough to keep
        new_mask = (d <= cut) & mask

        # If we dropped below min_keep, forcibly keep the best min_keep distances (respecting previous mask)
        if new_mask.sum() < min_keep:
            # Sort the distances of the *previously active* set
            active_indices = np.where(mask)[0]
            if len(active_indices) <= min_keep:
                new_mask = mask # Can't drop anything
            else:
                sorted_by_dist = active_indices[np.argsort(d[active_indices])]
                new_mask = np.zeros(N, dtype=bool)
                new_mask[sorted_by_dist[:min_keep]] = True

        if np.array_equal(mask, new_mask):
            break

        mask = new_mask
        if mask.sum() < 3:
            break
        
        R, t, rmsd = _kabsch(P[mask], Q[mask])

    return R, t, rmsd, mask

def _transform(coords:np.ndarray, R:np.ndarray, t:np.ndarray)->np.ndarray:
    return (R@coords.T).T + t

def _robust_inlier_mask(d:np.ndarray, hard_cut:float, q_keep:float)->np.ndarray:
    if d.size==0: return np.zeros(0, dtype=bool)
    qthr=float(np.quantile(d, q_keep)); return (d <= max(hard_cut, qthr))

@jit(nopython=True)
def _window_pairs_jit(A: np.ndarray, B: np.ndarray, aN: int, bN: int) -> Tuple[float, int, np.ndarray]:
    best_score = -np.inf
    best_offset = -1
    scores = np.zeros(bN-aN+1)
    for offset in range(bN-aN+1):
        subB = B[offset:offset+aN, offset:offset+aN]
        score = -np.sum(np.abs(A - subB))
        scores[offset] = score
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_score, best_offset, scores

def _window_pairs(D1:np.ndarray, D2:np.ndarray)->Tuple[List[Tuple[int,int]], np.ndarray]:
    n1,n2=D1.shape[0], D2.shape[0]
    if n1==0 or n2==0: return [], np.array([])
    swapped=False; A,B,aN,bN=D1,D2,n1,n2
    if aN>bN: A,B,aN,bN=D2,D1,n2,n1; swapped=True
    best_score, best_offset, scores = _window_pairs_jit(A, B, aN, bN)
    if best_offset<0: return [], scores
    if swapped: return [(best_offset+i, i) for i in range(aN)], scores
    else: return [(i, best_offset+i) for i in range(aN)], scores

def _radial_histograms(D:np.ndarray, nbins:int=24, rmax_mode:str="p98"):
    N=D.shape[0]
    if N==0: return np.zeros((0,nbins)), np.linspace(0,1,nbins+1)
    vals=D[np.triu_indices(N,k=1)]
    if vals.size==0: rmax=1.0
    else:
        rmax=float(np.quantile(vals,0.98)) if rmax_mode!="max" else float(np.max(vals))
        rmax=max(rmax,1.0)
    edges=np.linspace(0.0,rmax,nbins+1)
    H=np.zeros((N,nbins), dtype=np.float64)
    for i in range(N):
        row=D[i,:]; row=row[row>0.0]
        hist,_=np.histogram(row, bins=edges); s=hist.sum()
        H[i,:]=hist/s if s>0 else hist
    return H,edges

@jit(nopython=True)
def _chi2_distance_jit(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    N = X.shape[0]
    M = Y.shape[0]
    K = X.shape[1]
    res = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            s = 0.0
            for k in range(K):
                num = (X[i, k] - Y[j, k])**2
                den = X[i, k] + Y[j, k] + eps
                s += num / den
            res[i, j] = 0.5 * s
    return res

def _chi2_distance(X:np.ndarray, Y:np.ndarray, eps:float=1e-12)->np.ndarray:
    return _chi2_distance_jit(X, Y, eps)

@jit(nopython=True)
def _banded_dp_maxscore_jit(S: np.ndarray, gap: float, band: int) -> Tuple[np.ndarray, float]:
    N, M = S.shape
    neg = -1e18
    dp = np.full((N+1, M+1), neg)
    bt = np.zeros((N+1, M+1), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(0, N+1):
        jmin = max(0, i-band)
        jmax = min(M, i+band)
        if i > 0:
            j0 = max(0, i-band)
            if dp[i-1, j0] > neg:
                val = dp[i-1, j0] - gap
                if val > dp[i, j0]:
                    dp[i, j0] = val
                    bt[i, j0] = 2
        for j in range(jmin, jmax+1):
            if i == 0 and j == 0: continue
            best = neg
            move = 0
            if i > 0 and j > 0:
                cand = dp[i-1, j-1] + S[i-1, j-1]
                if cand > best:
                    best = cand
                    move = 1
            if i > 0:
                cand = dp[i-1, j] - gap
                if cand > best:
                    best = cand
                    move = 2
            if j > 0:
                cand = dp[i, j-1] - gap
                if cand > best:
                    best = cand
                    move = 3
            dp[i, j] = best
            bt[i, j] = move
    return bt, dp[N, M]

def _banded_dp_maxscore(S:np.ndarray, gap:float, band:int):
    N,M=S.shape
    if N==0 or M==0: return [], 0.0
    bt, max_score = _banded_dp_maxscore_jit(S, gap, band)
    i,j=N,M; pairs=[]
    while i>0 or j>0:
        move=bt[i,j]
        if move==1: pairs.append((i-1,j-1)); i-=1; j-=1
        elif move==2: i-=1
        elif move==3: j-=1
        else: break
    pairs.reverse()
    return pairs, float(max_score)

def _shape_pairs(coords1:np.ndarray, coords2:np.ndarray, nbins:int=24, gap_penalty:float=2.0, band_frac:float=0.20)->Tuple[List[Tuple[int,int]], np.ndarray, np.ndarray]:
    D1=_pairwise_dists(coords1); D2=_pairwise_dists(coords2)
    H1,_=_radial_histograms(D1, nbins=nbins, rmax_mode="p98")
    H2,_=_radial_histograms(D2, nbins=nbins, rmax_mode="p98")
    C=_chi2_distance(H1,H2); S=-C
    N,M=S.shape; band=max(3, int(band_frac*max(N,M)))
    pairs,_=_banded_dp_maxscore(S, gap=gap_penalty, band=band)
    return pairs, S, C

class _AllAtomsSelect(Select):
    def __init__(self, R:np.ndarray, t:np.ndarray, keep_heteroatoms:bool=True):
        super().__init__(); self.R=R; self.t=t; self.keep_heteroatoms=keep_heteroatoms
    def accept_residue(self, residue)->bool:
        if not self.keep_heteroatoms and residue.id[0] != ' ':
            return False
        return True
    def accept_atom(self, atom:Atom.Atom)->bool:
        coord=atom.get_coord().astype(float); atom.set_coord((self.R@coord)+self.t); return True

def sequence_independent_alignment_joined_v2(
    file_ref: str, file_mob: str,
    chains_ref: Optional[List[Union[str,int]]]=None,
    chains_mob: Optional[List[Union[str,int]]]=None,
    method:str="auto",
    shape_nbins:int=24, shape_gap_penalty:float=2.0, shape_band_frac:float=0.20,
    inlier_rmsd_cut:float=3.0, inlier_quantile:float=0.85, 
    recycles: int = 0, keep_fraction: float = 1.0,
    atoms: str = "CA", min_b_factor: float = 0.0, min_plddt: float = 0.0
)->AlignmentResultSF:
    logger.info(f"Running sequence_independent_alignment_joined_v2: ref={file_ref}, mob={file_mob}, method={method}")
    ref_struct=_parse_path(file_ref); mob_struct=_parse_path(file_mob)
    ref_ids=_resolve_selectors(ref_struct, chains_ref)
    mob_ids=_resolve_selectors(mob_struct, chains_mob)
    if (ref_ids is None) or (mob_ids is None):
        raise ValueError("Specify chains_ref and chains_mob for sequence-independent alignment.")

    # We must ALWAYS run the structure DP matrix on C-alphas to keep size (N) = number of residues.
    ref_infos=_extract_ca_infos(ref_struct, ref_ids, min_b_factor, min_plddt)
    mob_infos=_extract_ca_infos(mob_struct, mob_ids, min_b_factor, min_plddt)
    ref_subset=np.vstack([ri.coord for ri in ref_infos])
    mob_subset=np.vstack([mi.coord for mi in mob_infos])
    D1=_pairwise_dists(ref_subset); D2=_pairwise_dists(mob_subset)

    summaries={}; candidates={}
    shift_matrix = None
    shift_scores = None

    # shape
    if method in ("shape","auto"):
        pairs_s, S, C = _shape_pairs(ref_subset, mob_subset, nbins=shape_nbins, gap_penalty=shape_gap_penalty, band_frac=shape_band_frac)
        shift_matrix = S
        R_s,t_s,rmsd_s = np.eye(3), np.zeros(3), float("inf")
        mask_s = np.ones(len(pairs_s), dtype=bool)
        if len(pairs_s)>=3:
            P=np.vstack([ref_subset[i] for (i,j) in pairs_s]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_s])
            R_s, t_s, rmsd_s, mask_s = _iterative_kabsch(P, Q, recycles, keep_fraction)
        active_len_s = int(np.sum(mask_s))
        summaries["shape"]=AlignSummary("shape", float(rmsd_s), active_len_s, len(pairs_s), 1+recycles)
        candidates["shape"]=dict(pairs=pairs_s, rmsd=rmsd_s, R=R_s, t=t_s, mask=mask_s)

    # window
    if method in ("window","auto"):
        pairs_w, scores = _window_pairs(D1,D2)
        shift_scores = scores
        R_w,t_w,rmsd_w = np.eye(3), np.zeros(3), float("inf")
        mask_w = np.ones(len(pairs_w), dtype=bool)
        if len(pairs_w)>=3:
            P=np.vstack([ref_subset[i] for (i,j) in pairs_w]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_w])
            R_w, t_w, rmsd_w, mask_w = _iterative_kabsch(P, Q, recycles, keep_fraction)
        active_len_w = int(np.sum(mask_w))
        summaries["window"]=AlignSummary("window", float(rmsd_w), active_len_w, len(pairs_w), 1+recycles)
        candidates["window"]=dict(pairs=pairs_w, rmsd=rmsd_w, R=R_w, t=t_w, mask=mask_w)

    # choose by lowest RMSD; pairs only tie-break
    if method=="shape": chosen="shape"
    elif method=="window": chosen="window"
    else:
        def keyfun(mname):
            s=summaries[mname]; return (s.rmsd, -s.inliers)
        chosen=min(candidates.keys(), key=keyfun)

    R=candidates[chosen]["R"]; t=candidates[chosen]["t"]
    final_pairs=candidates[chosen]["pairs"]; final_rmsd=float(candidates[chosen]["rmsd"])
    final_mask = candidates[chosen]["mask"]

    # If the user requested backbone or all_heavy, we recalculate the final Kabsch superposition on the extended atoms
    if atoms != "CA":
        # We only use active pairs to compute the Kabsch alignment if extended atoms are requested
        seqfree_res_pairs = [(ref_infos[i], mob_infos[j]) for idx, (i, j) in enumerate(final_pairs) if final_mask[idx]]

        if atoms == "backbone":
            target_atoms = {"N", "CA", "C", "O"}
        else:
            target_atoms = None # All non-hydrogen

        final_ref_atoms = []
        final_mob_atoms = []

        ref_model = ref_struct[0]
        mob_model = mob_struct[0]

        for (ri, mi) in seqfree_res_pairs:
            r_res = None
            m_res = None

            # Find matching residue in reference
            try:
                r_chain = ref_model[ri.chain_id]
                for res in r_chain:
                    if res.seqid.num == ri.resseq and (res.seqid.icode == ri.icode or (not res.seqid.icode and not ri.icode)):
                        r_res = res
                        break
            except Exception:
                pass

            # Find matching residue in mobile
            try:
                m_chain = mob_model[mi.chain_id]
                for res in m_chain:
                    if res.seqid.num == mi.resseq and (res.seqid.icode == mi.icode or (not res.seqid.icode and not mi.icode)):
                        m_res = res
                        break
            except Exception:
                pass

            if not r_res or not m_res:
                continue

            for atA in r_res:
                if atA.element.name == "H": continue
                if target_atoms is not None and atA.name not in target_atoms: continue
                for atB in m_res:
                    if atB.name == atA.name:
                        # Construct a light pseudo-atom with coordinate interface
                        # Since _kabsch just needs a numpy array, we append PseudoAtom here
                        class PseudoAtom:
                            def __init__(self, pos):
                                self.pos = pos
                            def get_coord(self):
                                return np.array(self.pos.tolist(), dtype=float)
                        
                        final_ref_atoms.append(PseudoAtom(atA.pos))
                        final_mob_atoms.append(PseudoAtom(atB.pos))
                        break

        if final_ref_atoms and final_mob_atoms and len(final_ref_atoms) == len(final_mob_atoms):
            ref_c = np.array([a.get_coord() for a in final_ref_atoms])
            mob_c = np.array([a.get_coord() for a in final_mob_atoms])
            R, t, final_rmsd = _kabsch(ref_c, mob_c)

    mob_all_infos=_extract_ca_infos(_parse_path(file_mob), chain_filter=None, min_b_factor=min_b_factor, min_plddt=min_plddt)
    mob_all_ca=np.vstack([mi.coord for mi in mob_all_infos]); mob_all_ca_aligned=_transform(mob_all_ca, R, t)
    
    mob_subset_aligned = _transform(mob_subset, R, t)
    # Calculate GDT_TS and pseudo-CAD for the active fraction
    active_ref_idx = [i for idx, (i,j) in enumerate(final_pairs) if final_mask[idx]]
    active_mob_idx = [j for idx, (i,j) in enumerate(final_pairs) if final_mask[idx]]
    
    gdt_ts, cad_score = 0.0, 0.0
    if len(active_ref_idx) > 0:
        dists = np.linalg.norm(ref_subset[active_ref_idx] - mob_subset_aligned[active_mob_idx], axis=1)
        gdt_ts = compute_gdt_ts(dists)
        cad_score = compute_cad_score_approx(ref_subset[active_ref_idx], mob_subset_aligned[active_mob_idx])

    logger.info(f"Seq-free alignment ({chosen}) finished. RMSD = {final_rmsd:.3f}, GDT_TS = {gdt_ts:.2f}")

    return AlignmentResultSF(
        rotation=R, translation=t, rmsd=final_rmsd, iterations=1,
        kept_pairs=int(np.sum(final_mask)), method=chosen, pairs=final_pairs,
        ref_subset_infos=ref_infos, mob_subset_infos=mob_infos,
        ref_subset_ca_coords=ref_subset,
        mob_subset_ca_coords_aligned=mob_subset_aligned,
        mob_all_infos=mob_all_infos, mob_all_ca_coords_aligned=mob_all_ca_aligned,
        summaries=summaries,
        shift_matrix=shift_matrix if chosen == "shape" else None,
        shift_scores=shift_scores if chosen == "window" else None,
        active_mask=final_mask, gdt_ts=gdt_ts, cad_score=cad_score
    )

def perform_sequence_alignment(seq1:str, seq2:str, gap_open:float, gap_extend:float):
    if not seq1 or not seq2: return None
    try:
        blosum62=substitution_matrices.load("BLOSUM62")
        aligner = PairwiseAligner()
        aligner.substitution_matrix = blosum62
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend
        aligner.mode = 'global'
        # Semi-global alignment to allow individual chains to map to multi-chain references freely
        try:
            aligner.target_end_gap_score = 0.0
            aligner.query_end_gap_score = 0.0
        except Exception:
            pass
        alns = aligner.align(seq1, seq2)
        if not alns: return None
        a = alns[0]

        # For PairwiseAligner, formatting can vary based on exact match vs mismatch
        # It's safer to extract it from the alignment path indices
        seqA_aln = ""
        seqB_aln = ""
        # The coordinates are provided as lists of start/end indices
        # a.coordinates gives the path
        if hasattr(a, "indices"):
            ref_idx = a.indices[0]
            mob_idx = a.indices[1]

            p1, p2 = 0, 0
            for i in range(len(ref_idx)):
                if ref_idx[i] != -1 and mob_idx[i] != -1:
                    seqA_aln += seq1[ref_idx[i]]
                    seqB_aln += seq2[mob_idx[i]]
                elif ref_idx[i] != -1:
                    seqA_aln += seq1[ref_idx[i]]
                    seqB_aln += "-"
                elif mob_idx[i] != -1:
                    seqA_aln += "-"
                    seqB_aln += seq2[mob_idx[i]]
            seqA = seqA_aln
            seqB = seqB_aln
        else:
            # Fallback for Biopython > 1.80
            seqA = str(a[0])
            seqB = str(a[1])

        class Wrap:
            def __init__(self, seqA, seqB, score):
                self.seqA = seqA
                self.seqB = seqB
                self.score = score
        return Wrap(seqA, seqB, a.score)
    except Exception as e:
        # Avoid print here, log handles it in script
        return None

def get_aligned_atoms_by_alignment(ref_struct: gemmi.Structure, ref_chains, mob_struct: gemmi.Structure, mob_chains, alignment, atoms: str = "CA", min_b_factor: float = 0.0, min_plddt: float = 0.0):
    if not alignment: return [], []
    seqA, seqB = alignment.seqA, alignment.seqB

    if atoms == "backbone":
        target_atoms = {"N", "CA", "C", "O"}
    elif atoms == "all_heavy":
        target_atoms = None # All non-hydrogen
    else:
        target_atoms = {"CA"}

    class ResidueWrapper:
        def __init__(self, res, cid):
            self.res = res
            self._chain_id = cid
        
        def __iter__(self):
            return iter(self.res)
            
        def __getattr__(self, attr):
            return getattr(self.res, attr)

    def get_res_list(struct, chains):
        residues=[]
        model=struct[0]
        bounds_map = {}
        for c in chains:
            cid, start, end = _parse_chain_selector(c)
            if cid not in bounds_map: bounds_map[cid] = []
            if start is not None or end is not None: bounds_map[cid].append((start, end))

        for ch in chains:
            cid, _, _ = _parse_chain_selector(ch)
            chain = model.find_chain(cid)
            if chain:
                c_bounds = bounds_map.get(cid, [])
                for res in chain:
                    if res.name in AA_DICT:
                        resseq = res.seqid.num

                        if c_bounds:
                            in_bounds = False
                            for (start, end) in c_bounds:
                                if (start is None or resseq >= start) and (end is None or resseq <= end):
                                    in_bounds = True
                                    break
                            if not in_bounds:
                                continue

                        has_ca = False
                        for atom in res:
                            if atom.name == "CA":
                                has_ca = True
                                break
                        if atoms == "CA" and not has_ca:
                            continue

                        # Filter by B-factor for CA atoms if requested
                        if min_b_factor > 0.0:
                            b_factor_ok = False
                            for atom in res:
                                if atom.name == "CA" and atom.b_iso >= min_b_factor:
                                    b_factor_ok = True
                                    break
                            if not b_factor_ok:
                                continue
                        
                        # Filter by pLDDT
                        if min_plddt > 0.0:
                            plddt_ok = False
                            for atom in res:
                                if atom.name == "CA" and atom.b_iso >= min_plddt:
                                    plddt_ok = True
                                    break
                            if not plddt_ok:
                                continue

                        # Wrap the immutable C++ object to attach the chain ID dynamically
                        residues.append(ResidueWrapper(res, cid))
        return residues

    ref_res=get_res_list(ref_struct, ref_chains); mob_res=get_res_list(mob_struct, mob_chains)
    ref_idx=0; mob_idx=0; ref_atoms=[]; mob_atoms=[]

    class PseudoResidue:
        def __init__(self, resname, res_seq, res_icode=' '):
            self.resname = resname
            self._id = (' ', res_seq, res_icode)
        def get_resname(self): return self.resname
        def get_id(self): return self._id

    class PseudoAtom:
        def __init__(self, coord, name, chain_name, res_seq, res_icode, resname="UNK"):
            self.coord = coord
            self.name = name
            self.chain_name = chain_name
            self.res_seq = res_seq
            self.res_icode = res_icode
            self.parent = PseudoResidue(resname, res_seq, res_icode)
        def get_coord(self):
            return self.coord
        def get_name(self):
            return self.name
        def get_parent(self):
            return self.parent

    for a,b in zip(seqA, seqB):
        r_match=None; m_match=None
        if a!='-':
            while ref_idx<len(ref_res):
                r=ref_res[ref_idx]
                if AA_DICT.get(r.name)==a:
                    r_match=r; ref_idx+=1; break
                ref_idx+=1
        if b!='-':
            while mob_idx<len(mob_res):
                m=mob_res[mob_idx]
                if AA_DICT.get(m.name)==b:
                    m_match=m; mob_idx+=1; break
                mob_idx+=1

        if r_match is not None and m_match is not None:
            r_parent_chain = r_match._chain_id if hasattr(r_match, '_chain_id') else "A"
            m_parent_chain = m_match._chain_id if hasattr(m_match, '_chain_id') else "A"
            for atA in r_match:
                if atA.element.name == "H": continue
                if target_atoms is not None and atA.name not in target_atoms: continue
                for atB in m_match:
                    if atB.name == atA.name:
                        ref_atoms.append(PseudoAtom(np.array(atA.pos.tolist(), dtype=float), atA.name, r_parent_chain, r_match.seqid.num, r_match.seqid.icode if hasattr(r_match.seqid, 'has_icode') and r_match.seqid.has_icode() else "", r_match.name))
                        mob_atoms.append(PseudoAtom(np.array(atB.pos.tolist(), dtype=float), atB.name, m_parent_chain, m_match.seqid.num, m_match.seqid.icode if hasattr(m_match.seqid, 'has_icode') and m_match.seqid.has_icode() else "", m_match.name))
                        break

    return ref_atoms, mob_atoms

def superimpose_atoms(ref_atoms, mob_atoms, recycles: int = 0, keep_fraction: float = 1.0):
    if not ref_atoms or not mob_atoms or len(ref_atoms)!=len(mob_atoms): return None
    ref_coords=np.array([a.get_coord() for a in ref_atoms])
    mob_coords=np.array([a.get_coord() for a in mob_atoms])

    R, t, rmsd, mask = _iterative_kabsch(ref_coords, mob_coords, recycles, keep_fraction)
    
    mob_aligned=_transform(mob_coords, R, t)
    per_res=np.sqrt(np.sum((ref_coords - mob_aligned)**2, axis=1))

    # We return the active mask as well
    res_labels=[]
    active_ref_atoms = []
    active_mob_atoms = []
    
    for i, a in enumerate(ref_atoms):
        chain = a.chain_name
        lbl = f"{chain}:{a.res_seq}{a.res_icode.strip()}" if str(a.res_icode).strip() else f"{chain}:{a.res_seq}"
        res_labels.append(lbl)
        if mask[i]:
            active_ref_atoms.append(a)
            active_mob_atoms.append(mob_atoms[i])

    gdt_ts, cad_score = 0.0, 0.0
    if len(active_ref_atoms) > 0:
        active_ref_c = np.array([a.get_coord() for a in active_ref_atoms])
        active_mob_c = np.array([a.get_coord() for a in active_mob_atoms])
        
        # Calculate active distances
        dists = np.linalg.norm(active_ref_c - _transform(active_mob_c, R, t), axis=1)
        gdt_ts = compute_gdt_ts(dists)
        cad_score = compute_cad_score_approx(active_ref_c, _transform(active_mob_c, R, t))
        logger.info(f"Superimpose resulted in RMSD = {rmsd:.3f}, GDT_TS = {gdt_ts:.2f}")

    return dict(rmsd=float(rmsd), rotation=R, translation=t,
                ref_coords=ref_coords, mob_coords_transformed=mob_aligned,
                per_residue_rmsd=per_res, residue_labels=res_labels,
                active_ref_atoms=active_ref_atoms, active_mob_atoms=active_mob_atoms,
                mask=mask, gdt_ts=gdt_ts, cad_score=cad_score)

def compute_chain_similarity_matrix(seqsA, seqsB)->Tuple[pd.DataFrame,pd.DataFrame]:
    chainsA=list(seqsA.keys()); chainsB=list(seqsB.keys())
    if not chainsA or not chainsB: return pd.DataFrame(), pd.DataFrame()
    blosum62=substitution_matrices.load("BLOSUM62")
    id_mat=np.zeros((len(chainsA), len(chainsB))) * np.nan
    sc_mat=np.zeros((len(chainsA), len(chainsB))) * np.nan
    for i,chA in enumerate(chainsA):
        sA=str(seqsA[chA].seq)
        for j,chB in enumerate(chainsB):
            sB=str(seqsB[chB].seq)
            if not sA or not sB: continue

            aligner = PairwiseAligner()
            aligner.mode = 'global'
            aligner.substitution_matrix = blosum62
            aligner.open_gap_score = -10.0
            aligner.extend_gap_score = -0.5

            alns = aligner.align(sA, sB)
            if not alns: continue
            a = alns[0]

            if hasattr(a, "indices"):
                ref_idx = a.indices[0]
                mob_idx = a.indices[1]
                seqA_aln = ""
                seqB_aln = ""
                for k in range(len(ref_idx)):
                    if ref_idx[k] != -1 and mob_idx[k] != -1:
                        seqA_aln += sA[ref_idx[k]]
                        seqB_aln += sB[mob_idx[k]]
                    elif ref_idx[k] != -1:
                        seqA_aln += sA[ref_idx[k]]
                        seqB_aln += "-"
                    elif mob_idx[k] != -1:
                        seqA_aln += "-"
                        seqB_aln += sB[mob_idx[k]]
            else:
                seqA_aln = str(a[0])
                seqB_aln = str(a[1])

            matches=sum(1 for aa,bb in zip(seqA_aln, seqB_aln) if aa==bb and aa!='-')
            ident=100.0 * matches / max(1, len(seqA_aln)); id_mat[i,j]=ident
            sc=0.0; L=0
            for aa,bb in zip(seqA_aln,seqB_aln):
                if aa!='-' and bb!='-':
                    sc += blosum62.get((aa,bb), blosum62.get((bb,aa),0.0)); L+=1
            sc_mat[i,j]= sc / max(1,L)
    return pd.DataFrame(id_mat, index=chainsA, columns=chainsB), pd.DataFrame(sc_mat, index=chainsA, columns=chainsB)

def structure_based_alignment_strings(ref_infos: List[ResidueInfo], mob_infos: List[ResidueInfo],
                                      pairs: List[Tuple[int,int]]) -> Tuple[str,str,str]:
    if not pairs: return "", "", ""
    def letter(info: ResidueInfo) -> str: return AA_DICT.get(info.resname, "X")
    outA,outB=[],[]
    i_prev,j_prev=pairs[0]
    outA.append(letter(ref_infos[i_prev])); outB.append(letter(mob_infos[j_prev]))
    for (i,j) in pairs[1:]:
        di, dj = i - i_prev, j - j_prev
        while di>1 or dj>1:
            if di>dj: outA.append(letter(ref_infos[i_prev+1])); outB.append("-"); i_prev+=1; di-=1
            elif dj>di: outA.append("-"); outB.append(letter(mob_infos[j_prev+1])); j_prev+=1; dj-=1
            else: outA.append(letter(ref_infos[i_prev+1])); outB.append(letter(mob_infos[j_prev+1])); i_prev+=1; j_prev+=1; di-=1; dj-=1
        outA.append(letter(ref_infos[i])); outB.append(letter(mob_infos[j])); i_prev,j_prev=i,j
    blosum62=substitution_matrices.load("BLOSUM62")
    match=[]
    for a,b in zip(outA,outB):
        if a==b and a!='-': match.append("|")
        elif a!='-' and b!='-' and (blosum62.get((a,b), blosum62.get((b,a),0))>0): match.append(":")
        elif a=='-' or b=='-': match.append(" ")
        else: match.append(".")
    return "".join(outA), "".join(outB), "".join(match)

def pick_best_overall(seqguided, seqfree, min_pairs:int=3):
    cands=[]
    if seqguided is not None:
        cands.append(dict(name="Sequence-guided", rmsd=float(seqguided["si"]["rmsd"]), pairs=len(seqguided["ref_atoms"]), kind="seqguided"))
    if seqfree is not None:
        cands.append(dict(name=f"Sequence-free ({seqfree.method})", rmsd=float(seqfree.rmsd), pairs=int(seqfree.kept_pairs), kind="seqfree"))
    if not cands: return None, "No candidates available."

    valid=[c for c in cands if np.isfinite(c["rmsd"]) and c["pairs"]>=min_pairs]
    if not valid: valid=[c for c in cands if np.isfinite(c["rmsd"])]
    if not valid:
        best=min(cands, key=lambda c: (math.isfinite(c["rmsd"])==False, c["rmsd"]))
        return best, "Chose the only available candidate."
    best=min(valid, key=lambda c: (c["rmsd"], -c["pairs"]))
    others=[c for c in valid if c is not best]
    if others:
        alt=min(others, key=lambda c: (c["rmsd"], -c["pairs"]))
        reason=f"Lower RMSD ({best['rmsd']:.2f} Å) vs {alt['name']} ({alt['rmsd']:.2f} Å)."
        if abs(best["rmsd"]-alt["rmsd"])<1e-6 and best["pairs"]!=alt["pairs"]:
            reason+=f" Tie on RMSD; chose higher pairs ({best['pairs']} vs {alt['pairs']})."
    else:
        reason="Single valid candidate."
    return best, reason

def progressive_align_ensemble(
    files: List[str], 
    chains_list: Optional[List[Optional[List[Union[str,int]]]]] = None,
    method: str = "shape", 
    recycles: int = 0, keep_fraction: float = 1.0, min_plddt: float = 0.0
) -> Dict[str, Any]:
    """
    Computes a progressive ensemble alignment by picking a central "medoid" structure 
    and aligning all others to it to achieve a Multiple Structure Alignment (MSA).
    Returns mapping parameters and overall RMSD metrics.
    """
    if len(files) < 2:
        return {"error": "Need at least 2 structures for ensemble alignment"}
    
    if chains_list is None:
        chains_list = [None] * len(files)
        
    logger.info(f"Starting progressive ensemble alignment for {len(files)} structures")
    
    # 1. Pairwise shape/window alignments to target the Medoid
    N = len(files)
    pairwise_rmsds = np.zeros((N, N))
    
    # For a full medoid we need N*(N-1)/2, but for speed we can approximate.
    # To be robust, if N<=10 we do all pairs. If larger, we pick the first file as reference.
    if N <= 10:
        for i in range(N):
            for j in range(i+1, N):
                res = sequence_independent_alignment_joined_v2(
                    files[i], files[j], chains_ref=chains_list[i], chains_mob=chains_list[j],
                    method=method, recycles=recycles, keep_fraction=keep_fraction, min_plddt=min_plddt
                )
                pairwise_rmsds[i, j] = res.rmsd
                pairwise_rmsds[j, i] = res.rmsd
        
        avg_rmsds = np.sum(pairwise_rmsds, axis=1) / (N - 1)
        medoid_idx = int(np.argmin(avg_rmsds))
    else:
        medoid_idx = 0
        
    logger.info(f"Selected structure {medoid_idx} ({files[medoid_idx]}) as the Medoid.")
    
    aligned_results = []
    medoid_file = files[medoid_idx]
    
    for i in range(N):
        if i == medoid_idx:
            # Identity matrix for medoid
            aligned_results.append({
                "mob_index": i, "file": files[i], 
                "R": np.eye(3), "t": np.zeros(3), "rmsd": 0.0, "is_medoid": True
            })
            continue
            
        res = sequence_independent_alignment_joined_v2(
            medoid_file, files[i], chains_ref=chains_list[medoid_idx], chains_mob=chains_list[i],
            method=method, recycles=recycles, keep_fraction=keep_fraction, min_plddt=min_plddt
        )
        logger.info(f"Aligned {files[i]} to Medoid, RMSD: {res.rmsd:.3f}")
        
        aligned_results.append({
            "mob_index": i, "file": files[i], 
            "R": res.rotation, "t": res.translation, "rmsd": res.rmsd, "gdt_ts": res.gdt_ts, "is_medoid": False
        })
        
    return {
        "medoid_idx": medoid_idx,
        "medoid_file": medoid_file,
        "results": aligned_results,
        "avg_ensemble_rmsd": float(np.mean([r["rmsd"] for r in aligned_results if r["rmsd"] > 0]))
    }

