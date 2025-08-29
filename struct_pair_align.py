# app.py
# =========================================================
# Structure Alignment Workhorse
# - Auto logic picks lowest RMSD (pairs as tie-breaker)
# - Sequence-guided alignment (pairwise seq + Kabsch)
# - Sequence-independent alignment: "shape", "window", "auto"
# - 3D superposition highlights RMSD peaks, draws connectors,
#   zooms to a selected peak (unique widget keys)
# - Side-panel: chain similarity matrix, chain lengths
# - Diagnostics, structure-based sequence alignment, exports
# - NEW: In Seq-free results, toggle between matched subset vs FULL selected chains
#
# Run: streamlit run app.py
# =========================================================

from __future__ import annotations

import os
import io
import math
import json
import tempfile
import datetime
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# BioPython
from Bio.PDB import (
    PDBParser, MMCIFParser, PDBIO, Select,
    Structure, Atom, Superimposer
)
from Bio.PDB.Polypeptide import protein_letters_3to1, standard_aa_names, is_aa
from Bio.Seq import Seq
    # noqa
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
from Bio.Align import substitution_matrices

import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Structure Alignment Workhorse", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
code, pre { font-size: 12px !important; line-height: 1.25; }
</style>
""", unsafe_allow_html=True)

# ===========================================
# Helpers and constants
# ===========================================
VALID_AA_3 = set(standard_aa_names)
def _aa_dict() -> Dict[str, str]:
    d = {k: protein_letters_3to1[k] for k in VALID_AA_3}
    d['MSE'] = 'M'
    return d
AA_DICT = _aa_dict()

def init_state():
    defaults = dict(
        structures={}, sequences={}, chain_lengths={},
        selected_ref_file=None, selected_mobile_file=None,
        selected_ref_chains=[], selected_mobile_chains=[],
        align_mode="Auto (best RMSD)",
        seq_gap_open=-10, seq_gap_extend=-0.5,
        results_cache={}, last_run_summary=None, logs=[]
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
init_state()

# ===========================================
# IO / parsing
# ===========================================
def parse_structure(uploaded_file):
    try:
        raw = uploaded_file.getvalue()
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw); tmp_path = tmp.name
        parser = MMCIFParser(QUIET=True) if suffix in (".cif",".mmcif") else PDBParser(QUIET=True)
        s = parser.get_structure(os.path.splitext(uploaded_file.name)[0], tmp_path)
        os.unlink(tmp_path)
        return s, None
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, f"Error parsing {uploaded_file.name}: {e}"

def extract_sequences_and_lengths(struct: Structure.Structure, fname: str):
    seqs: Dict[str, SeqRecord] = {}
    lens: Dict[str, int] = {}
    try:
        model = list(struct.get_models())[0]
    except Exception:
        return {}, {}
    for chain in model:
        seq = []
        ca_count = 0
        for res in chain:
            if res.id[0] == ' ' and res.get_resname() in AA_DICT:
                seq.append(AA_DICT[res.get_resname()])
                if 'CA' in res:
                    ca_count += 1
        if seq:
            seqs[chain.id] = SeqRecord(Seq("".join(seq)), id=f"{fname}_{chain.id}", description=f"Chain {chain.id}")
            lens[chain.id] = int(ca_count)
    return seqs, lens

def write_uploads_to_temp_files(fnameA: str, fnameB: str) -> Tuple[str,str]:
    ref_obj = st.session_state.uploaded_blob_map[fnameA]
    mob_obj = st.session_state.uploaded_blob_map[fnameB]
    sufxA = os.path.splitext(fnameA)[1].lower()
    sufxB = os.path.splitext(fnameB)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufxA) as t1:
        t1.write(ref_obj.getvalue()); pathA = t1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufxB) as t2:
        t2.write(mob_obj.getvalue()); pathB = t2.name
    return pathA, pathB

# ===========================================
# Seq-free core (robust "shape" + "window")
# ===========================================
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

from Bio.PDB import Structure as _Structure
def _parse_path(path:str) -> _Structure.Structure:
    parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif",".mmcif")) else PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)

def _chain_ids(struct:_Structure.Structure) -> List[str]:
    return [ch.id for ch in list(struct)[0]]

def _resolve_selectors(struct:_Structure.Structure, sel: Optional[List[Union[str,int]]]) -> Optional[List[str]]:
    if sel is None: return None
    ids = _chain_ids(struct)
    out=[]
    for x in sel:
        if isinstance(x,int):
            if not (1<=x<=len(ids)): raise ValueError(f"Chain index {x} out of range 1..{len(ids)}")
            out.append(ids[x-1])
        else:
            if x not in ids: raise ValueError(f"Chain '{x}' not in {ids}")
            out.append(x)
    seen=set(); uniq=[c for c in out if not (c in seen or seen.add(c))]
    return uniq

def _extract_ca_infos(struct:_Structure.Structure, chain_filter:Optional[List[str]])->List[ResidueInfo]:
    model = list(struct)[0]
    infos=[]; idx=0
    for chain in model:
        if chain_filter is not None and chain.id not in chain_filter: continue
        for res in chain:
            if not is_aa(res, standard=True): continue
            if "CA" not in res: continue
            ca: Atom.Atom = res["CA"]
            coord = ca.get_coord().astype(float)
            hetflag, resseq, icode = res.get_id()
            infos.append(ResidueInfo(idx=idx, chain_id=chain.id, resseq=int(resseq),
                                     icode=(icode or "").strip() if isinstance(icode,str) else "",
                                     resname=str(res.get_resname()), coord=coord))
            idx += 1
    if not infos: raise ValueError("No Cα atoms found for selected chains.")
    return infos

def _pairwise_dists(coords: np.ndarray) -> np.ndarray:
    x=coords; x2=np.sum(x*x, axis=1, keepdims=True)
    d2=x2+x2.T-2.0*np.dot(x,x.T); np.maximum(d2,0.0,out=d2)
    return np.sqrt(d2, out=d2)

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

def _transform(coords:np.ndarray, R:np.ndarray, t:np.ndarray)->np.ndarray:
    return (R@coords.T).T + t

def _robust_inlier_mask(d:np.ndarray, hard_cut:float, q_keep:float)->np.ndarray:
    if d.size==0: return np.zeros(0, dtype=bool)
    qthr=float(np.quantile(d, q_keep)); return (d <= max(hard_cut, qthr))

def _window_pairs(D1:np.ndarray, D2:np.ndarray)->List[Tuple[int,int]]:
    n1,n2=D1.shape[0], D2.shape[0]
    if n1==0 or n2==0: return []
    swapped=False; A,B,aN,bN=D1,D2,n1,n2
    if aN>bN: A,B,aN,bN=D2,D1,n2,n1; swapped=True
    best_score, best_offset = -np.inf, -1
    for offset in range(bN-aN+1):
        subB=B[offset:offset+aN, offset:offset+aN]
        score=-np.sum(np.abs(A - subB))
        if score>best_score: best_score, best_offset = score, offset
    if best_offset<0: return []
    if swapped: return [(best_offset+i, i) for i in range(aN)]
    else: return [(i, best_offset+i) for i in range(aN)]

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

def _chi2_distance(X:np.ndarray, Y:np.ndarray, eps:float=1e-12)->np.ndarray:
    Xe=X[:,None,:]; Ye=Y[None,:,:]; num=(Xe-Ye)**2; den=(Xe+Ye)+eps
    return 0.5*np.sum(num/den, axis=2)

def _banded_dp_maxscore(S:np.ndarray, gap:float, band:int):
    N,M=S.shape
    if N==0 or M==0: return [], 0.0
    neg=-1e18
    dp=np.full((N+1,M+1), neg); bt=np.zeros((N+1,M+1), dtype=np.int8)
    dp[0,0]=0.0
    for i in range(0,N+1):
        jmin=max(0, i-band); jmax=min(M, i+band)
        if i>0:
            j0=max(0, i-band)
            if dp[i-1,j0]>neg:
                dp[i,j0]=max(dp[i,j0], dp[i-1,j0]-gap); bt[i,j0]=2
        for j in range(jmin, jmax+1):
            if i==0 and j==0: continue
            best,move=neg,0
            if i>0 and j>0:
                cand=dp[i-1,j-1]+S[i-1,j-1]
                if cand>best: best,move=cand,1
            if i>0:
                cand=dp[i-1,j]-gap
                if cand>best: best,move=cand,2
            if j>0:
                cand=dp[i,j-1]-gap
                if cand>best: best,move=cand,3
            dp[i,j]=best; bt[i,j]=move
    i,j=N,M; pairs=[]
    while i>0 or j>0:
        move=bt[i,j]
        if move==1: pairs.append((i-1,j-1)); i-=1; j-=1
        elif move==2: i-=1
        elif move==3: j-=1
        else: break
    pairs.reverse(); total=float(dp[N,M])
    return pairs,total

def _shape_pairs(coords1:np.ndarray, coords2:np.ndarray, nbins:int=24, gap_penalty:float=2.0, band_frac:float=0.20):
    D1=_pairwise_dists(coords1); D2=_pairwise_dists(coords2)
    H1,_=_radial_histograms(D1, nbins=nbins, rmax_mode="p98")
    H2,_=_radial_histograms(D2, nbins=nbins, rmax_mode="p98")
    C=_chi2_distance(H1,H2); S=-C
    N,M=S.shape; band=max(3, int(band_frac*max(N,M)))
    pairs,_=_banded_dp_maxscore(S, gap=gap_penalty, band=band)
    return pairs

class _AllAtomsSelect(Select):
    def __init__(self, R:np.ndarray, t:np.ndarray):
        super().__init__(); self.R=R; self.t=t
    def accept_atom(self, atom:Atom.Atom)->bool:
        coord=atom.get_coord().astype(float); atom.set_coord((self.R@coord)+self.t); return True

def sequence_independent_alignment_joined_v2(
    file_ref: str, file_mob: str,
    chains_ref: Optional[List[Union[str,int]]]=None,
    chains_mob: Optional[List[Union[str,int]]]=None,
    method:str="auto",
    shape_nbins:int=24, shape_gap_penalty:float=2.0, shape_band_frac:float=0.20,
    inlier_rmsd_cut:float=3.0, inlier_quantile:float=0.85, refinement_iters:int=2
)->AlignmentResultSF:
    ref_struct=_parse_path(file_ref); mob_struct=_parse_path(file_mob)
    ref_ids=_resolve_selectors(ref_struct, chains_ref)
    mob_ids=_resolve_selectors(mob_struct, chains_mob)
    if (ref_ids is None) or (mob_ids is None):
        raise ValueError("Specify chains_ref and chains_mob for sequence-independent alignment.")
    ref_infos=_extract_ca_infos(ref_struct, ref_ids)
    mob_infos=_extract_ca_infos(mob_struct, mob_ids)
    ref_subset=np.vstack([ri.coord for ri in ref_infos])
    mob_subset=np.vstack([mi.coord for mi in mob_infos])
    D1=_pairwise_dists(ref_subset); D2=_pairwise_dists(mob_subset)

    summaries={}; candidates={}
    # shape
    if method in ("shape","auto"):
        pairs_s=_shape_pairs(ref_subset, mob_subset, nbins=shape_nbins, gap_penalty=shape_gap_penalty, band_frac=shape_band_frac)
        R_s,t_s,rmsd_s=np.eye(3), np.zeros(3), float("inf")
        mob_aln_s=mob_subset.copy()
        if len(pairs_s)>=3:
            P=np.vstack([ref_subset[i] for (i,j) in pairs_s]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_s])
            R_s,t_s,rmsd_s=_kabsch(P,Q); mob_aln_s=_transform(mob_subset, R_s, t_s)
        pairs_ref=pairs_s; rmsd_ref=rmsd_s
        for _ in range(refinement_iters):
            if len(pairs_ref)<3: break
            d=np.array([np.linalg.norm(ref_subset[i]-mob_aln_s[j]) for (i,j) in pairs_ref])
            mask=_robust_inlier_mask(d, hard_cut=inlier_rmsd_cut, q_keep=inlier_quantile)
            pairs_ref=[p for k,p in enumerate(pairs_ref) if mask[k]]
            if len(pairs_ref)<3: break
            P=np.vstack([ref_subset[i] for (i,j) in pairs_ref]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_ref])
            R_s,t_s,rmsd_ref=_kabsch(P,Q); mob_aln_s=_transform(mob_subset, R_s, t_s)
        summaries["shape"]=AlignSummary("shape", float(rmsd_ref), len(pairs_ref), len(pairs_ref), 1)
        candidates["shape"]=dict(pairs=pairs_ref, rmsd=rmsd_ref, R=R_s, t=t_s)

    # window
    if method in ("window","auto"):
        pairs_w=_window_pairs(D1,D2)
        R_w,t_w,rmsd_w=np.eye(3), np.zeros(3), float("inf")
        mob_aln_w=mob_subset.copy()
        if len(pairs_w)>=3:
            P=np.vstack([ref_subset[i] for (i,j) in pairs_w]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_w])
            R_w,t_w,rmsd_w=_kabsch(P,Q); mob_aln_w=_transform(mob_subset, R_w, t_w)
        pairs_ref=pairs_w; rmsd_ref=rmsd_w
        for _ in range(refinement_iters):
            if len(pairs_ref)<3: break
            d=np.array([np.linalg.norm(ref_subset[i]-mob_aln_w[j]) for (i,j) in pairs_ref])
            mask=_robust_inlier_mask(d, hard_cut=inlier_rmsd_cut, q_keep=inlier_quantile)
            pairs_ref=[p for k,p in enumerate(pairs_ref) if mask[k]]
            if len(pairs_ref)<3: break
            P=np.vstack([ref_subset[i] for (i,j) in pairs_ref]); Q=np.vstack([mob_subset[j] for (i,j) in pairs_ref])
            R_w,t_w,rmsd_ref=_kabsch(P,Q); mob_aln_w=_transform(mob_subset, R_w, t_w)
        summaries["window"]=AlignSummary("window", float(rmsd_ref), len(pairs_ref), len(pairs_ref), 1)
        candidates["window"]=dict(pairs=pairs_ref, rmsd=rmsd_ref, R=R_w, t=t_w)

    # choose by lowest RMSD; pairs only tie-break
    if method=="shape": chosen="shape"
    elif method=="window": chosen="window"
    else:
        def keyfun(mname):
            s=summaries[mname]; return (s.rmsd, -s.inliers)
        chosen=min(candidates.keys(), key=keyfun)

    R=candidates[chosen]["R"]; t=candidates[chosen]["t"]
    final_pairs=candidates[chosen]["pairs"]; final_rmsd=float(candidates[chosen]["rmsd"])
    mob_all_infos=_extract_ca_infos(_parse_path(file_mob), chain_filter=None)
    mob_all_ca=np.vstack([mi.coord for mi in mob_all_infos]); mob_all_ca_aligned=_transform(mob_all_ca, R, t)

    return AlignmentResultSF(
        rotation=R, translation=t, rmsd=final_rmsd, iterations=1,
        kept_pairs=len(final_pairs), method=chosen, pairs=final_pairs,
        ref_subset_infos=ref_infos, mob_subset_infos=mob_infos,
        ref_subset_ca_coords=ref_subset,
        mob_subset_ca_coords_aligned=_transform(mob_subset, R, t),
        mob_all_infos=mob_all_infos, mob_all_ca_coords_aligned=mob_all_ca_aligned,
        summaries=summaries
    )

# ===========================================
# Sequence-guided utilities
# ===========================================
def perform_sequence_alignment(seq1:str, seq2:str, gap_open:float, gap_extend:float):
    if not seq1 or not seq2: return None
    try:
        blosum62=substitution_matrices.load("BLOSUM62")
        alns=pairwise2.align.globalds(seq1, seq2, blosum62, gap_open, gap_extend, one_alignment_only=True)
        if not alns: return None
        a=alns[0]
        class Wrap:
            def __init__(self, tup): self.seqA, self.seqB, self.score = tup[0], tup[1], tup[2]
        return Wrap(a)
    except Exception as e:
        log(f"Sequence alignment failed: {e}")
        return None

def get_aligned_atoms_by_alignment(ref_struct, ref_chains, mob_struct, mob_chains, alignment):
    if not alignment: return [], []
    seqA, seqB = alignment.seqA, alignment.seqB
    def get_res_list(struct, chains):
        residues=[]
        model=list(struct.get_models())[0]
        for ch in chains:
            if ch in model:
                for res in model[ch]:
                    if res.id[0]==' ' and res.get_resname() in AA_DICT and 'CA' in res:
                        residues.append(res)
        return residues
    ref_res=get_res_list(ref_struct, ref_chains); mob_res=get_res_list(mob_struct, mob_chains)
    ref_idx=0; mob_idx=0; ref_atoms=[]; mob_atoms=[]
    for a,b in zip(seqA, seqB):
        ref_atom=None; mob_atom=None
        if a!='-':
            while ref_idx<len(ref_res):
                r=ref_res[ref_idx]
                if AA_DICT.get(r.get_resname())==a and 'CA' in r:
                    ref_atom=r['CA']; ref_idx+=1; break
                ref_idx+=1
        if b!='-':
            while mob_idx<len(mob_res):
                m=mob_res[mob_idx]
                if AA_DICT.get(m.get_resname())==b and 'CA' in m:
                    mob_atom=m['CA']; mob_idx+=1; break
                mob_idx+=1
        if ref_atom is not None and mob_atom is not None:
            ref_atoms.append(ref_atom); mob_atoms.append(mob_atom)
    return ref_atoms, mob_atoms

def superimpose_atoms(ref_atoms, mob_atoms):
    if not ref_atoms or not mob_atoms or len(ref_atoms)!=len(mob_atoms): return None
    SI=Superimposer(); SI.set_atoms(ref_atoms, mob_atoms); R,t=SI.rotran
    ref_coords=np.array([a.get_coord() for a in ref_atoms])
    mob_coords=np.array([a.get_coord() for a in mob_atoms])
    mob_aligned=(mob_coords@R) + t
    per_res=np.sqrt(np.sum((ref_coords - mob_aligned)**2, axis=1))
    res_ids=[a.get_parent().get_id() for a in ref_atoms]
    res_labels=[f"{rid[1]}{rid[2].strip()}" if str(rid[2]).strip() else f"{rid[1]}" for rid in res_ids]
    return dict(rmsd=float(SI.rms), rotation=R, translation=t,
                ref_coords=ref_coords, mob_coords_transformed=mob_aligned,
                per_residue_rmsd=per_res, residue_labels=res_labels)

# ===========================================
# Chain similarity matrix
# ===========================================
def compute_chain_similarity_matrix(fileA:str, fileB:str)->Tuple[pd.DataFrame,pd.DataFrame]:
    seqsA=st.session_state.sequences.get(fileA, {})
    seqsB=st.session_state.sequences.get(fileB, {})
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
            aln=pairwise2.align.globalxx(sA, sB, one_alignment_only=True)
            if not aln: continue
            a=aln[0]
            matches=sum(1 for aa,bb in zip(a.seqA, a.seqB) if aa==bb and aa!='-')
            ident=100.0 * matches / max(1, len(a.seqA)); id_mat[i,j]=ident
            sc=0.0; L=0
            for aa,bb in zip(a.seqA,a.seqB):
                if aa!='-' and bb!='-':
                    sc += blosum62.get((aa,bb), blosum62.get((bb,aa),0.0)); L+=1
            sc_mat[i,j]= sc / max(1,L)
    return pd.DataFrame(id_mat, index=chainsA, columns=chainsB), pd.DataFrame(sc_mat, index=chainsA, columns=chainsB)

# ===========================================
# Pretty alignment text
# ===========================================
def formatted_alignment_text(id1, aligned1, id2, aligned2, score:float, interval:int=10)->str:
    def numline(aln: str, interval=10):
        line=[' ']*len(aln); c=0; nxt=interval
        for i,ch in enumerate(aln):
            if ch!='-':
                c+=1
                if c==nxt:
                    s=str(nxt); start=max(0, i-len(s)+1)
                    for k,d in enumerate(s):
                        if start+k < len(aln): line[start+k]=d
                    nxt += interval
        return ''.join(line)
    pad=max(len(id1), len(id2)); id1p=id1.ljust(pad); id2p=id2.ljust(pad); mp="Match".ljust(pad)
    blosum62=substitution_matrices.load("BLOSUM62")
    match=""
    for a,b in zip(aligned1, aligned2):
        if a==b and a!='-': match+="|"
        elif a!='-' and b!='-' and (blosum62.get((a,b), blosum62.get((b,a),0))>0): match+=":"
        elif a=='-' or b=='-': match+=" "
        else: match+="."
    loc1,loc2=numline(aligned1,interval), numline(aligned2,interval); padsp=" "*(pad+2)
    return ( "Pairwise Alignment:\n"
             f"{padsp}{loc1}\n"
             f"{id1p}: {aligned1}\n"
             f"{mp}: {match}\n"
             f"{id2p}: {aligned2}\n"
             f"{padsp}{loc2}\n"
             f"{'Score'.ljust(pad)}: {score:.2f}\n" )

# ===========================================
# 3D superposition with peaks (unique widget keys)
# ===========================================
def _san_key(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)

def plot_superposition_3d(ref_coords: np.ndarray,
                          mob_coords: np.ndarray,
                          title: str,
                          labels: Optional[List[str]] = None,
                          pairs: Optional[List[Tuple[int,int]]] = None,
                          default_top_k: int = 10,
                          key_prefix: Optional[str] = None):
    kp = _san_key(key_prefix if key_prefix else f"k_{title}_{id(ref_coords)}_{id(mob_coords)}")

    # Build arrays and labels consistent with "pairs" indexing (for distances/peaks)
    if pairs is not None:
        ref_pts_for_peaks = np.vstack([ref_coords[i] for (i, j) in pairs])
        mob_pts_for_peaks = np.vstack([mob_coords[j] for (i, j) in pairs])
        if labels is None:
            labels = [f"{i}" for (i, _) in pairs]
    else:
        ref_pts_for_peaks = np.array(ref_coords, dtype=float)
        mob_pts_for_peaks = np.array(mob_coords, dtype=float)
        if labels is None:
            labels = [str(i) for i in range(len(ref_pts_for_peaks))]

    # Compute per-pair distances on the peaks arrays
    if ref_pts_for_peaks.shape != mob_pts_for_peaks.shape or ref_pts_for_peaks.shape[1] != 3:
        st.warning("Cannot compute per-pair 3D distances: shapes do not match.")
        ref_draw = np.array(ref_coords, dtype=float)
        mob_draw = np.array(mob_coords, dtype=float)
        _plot_basic_3d(ref_draw, mob_draw, title)
        return

    d = np.linalg.norm(ref_pts_for_peaks - mob_pts_for_peaks, axis=1)
    N = len(d)
    if N == 0:
        st.info("No matched pairs to visualize.")
        return

    # Peak detection
    try:
        from scipy.signal import argrelextrema
        idx_peaks = argrelextrema(d, np.greater, order=3)[0]
        if idx_peaks.size == 0:
            idx_peaks = np.argsort(-d)[:min(default_top_k, N)]
        else:
            idx_peaks = idx_peaks[np.argsort(-d[idx_peaks])]
    except Exception:
        idx_peaks = np.argsort(-d)[:min(default_top_k, N)]

    # Controls (unique keys)
    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.2, 1.2])
    top_k = c1.slider("Top peaks to highlight",
                      min_value=1, max_value=min(50, N), value=min(default_top_k, N), step=1,
                      key=f"{kp}_topk")
    show_connectors = c2.checkbox("Show connectors", value=True, key=f"{kp}_conn")
    show_all_points = c3.checkbox("Show all Cα points", value=True, key=f"{kp}_showall")
    focus_idx = c4.selectbox(
        "Focus on peak",
        options=[None] + list(idx_peaks[:top_k]),
        format_func=lambda i: "None" if i is None else f"#{int(i)}  {labels[int(i)]}  ({d[int(i)]:.2f} Å)",
        key=f"{kp}_focusidx"
    )
    focus_radius = st.slider("Focus radius (Å)",
                             min_value=5.0, max_value=60.0, value=12.0, step=1.0,
                             key=f"{kp}_radius")

    # For drawing, we allow passing either the subset or the full arrays
    ref_draw = np.array(ref_coords, dtype=float)
    mob_draw = np.array(mob_coords, dtype=float)

    # Build 3D figure
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=ref_draw[:,0], y=ref_draw[:,1], z=ref_draw[:,2],
        mode='lines+markers' if show_all_points else 'lines',
        name='Reference Cα',
        marker=dict(size=2),
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter3d(
        x=mob_draw[:,0], y=mob_draw[:,1], z=mob_draw[:,2],
        mode='lines+markers' if show_all_points else 'lines',
        name='Mobile Cα (Aligned)',
        marker=dict(size=2),
        line=dict(width=2)
    ))

    # Highlight peaks (indexed in the "peaks arrays")
    pk = np.array(idx_peaks[:top_k], dtype=int)
    if pk.size > 0:
        fig.add_trace(go.Scatter3d(
            x=ref_pts_for_peaks[pk,0], y=ref_pts_for_peaks[pk,1], z=ref_pts_for_peaks[pk,2],
            mode='markers+text',
            name='Peaks (ref)',
            marker=dict(size=5, symbol='diamond'),
            text=[f"{labels[i]} ({d[i]:.2f} Å)" for i in pk],
            textposition="top center",
            hovertemplate="Ref %{text}<extra></extra>"
        ))
        fig.add_trace(go.Scatter3d(
            x=mob_pts_for_peaks[pk,0], y=mob_pts_for_peaks[pk,1], z=mob_pts_for_peaks[pk,2],
            mode='markers',
            name='Peaks (mob)',
            marker=dict(size=5, symbol='x'),
            hovertemplate="Mob distance: %{customdata:.2f} Å<extra></extra>",
            customdata=d[pk]
        ))
        if show_connectors:
            for i in pk:
                fig.add_trace(go.Scatter3d(
                    x=[ref_pts_for_peaks[i,0], mob_pts_for_peaks[i,0]],
                    y=[ref_pts_for_peaks[i,1], mob_pts_for_peaks[i,1]],
                    z=[ref_pts_for_peaks[i,2], mob_pts_for_peaks[i,2]],
                    mode='lines',
                    name=f"Δ {labels[i]}",
                    showlegend=False
                ))

    # Focus window
    if focus_idx is not None:
        i = int(focus_idx)
        center = 0.5*(ref_pts_for_peaks[i] + mob_pts_for_peaks[i])
        xr = (center[0]-focus_radius, center[0]+focus_radius)
        yr = (center[1]-focus_radius, center[1]+focus_radius)
        zr = (center[2]-focus_radius, center[2]+focus_radius)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=xr),
                yaxis=dict(range=yr),
                zaxis=dict(range=zr),
                aspectmode='cube'
            )
        )
        st.info(f"Focused on peak #{i}: {labels[i]} — distance {d[i]:.2f} Å")

    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data', xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
        height=600, margin=dict(l=0,r=0,t=50,b=0),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

def _plot_basic_3d(ref_coords, mob_coords, title):
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=ref_coords[:,0], y=ref_coords[:,1], z=ref_coords[:,2],
                               mode='lines+markers', name='Reference Cα',
                               marker=dict(size=2), line=dict(width=2)))
    fig.add_trace(go.Scatter3d(x=mob_coords[:,0], y=mob_coords[:,1], z=mob_coords[:,2],
                               mode='lines+markers', name='Mobile Cα (Aligned)',
                               marker=dict(size=2), line=dict(width=2)))
    fig.update_layout(title=title, scene=dict(aspectmode='data', xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                      height=560, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ===========================================
# Distance/cost plots
# ===========================================
def make_dual_heat(A: np.ndarray, B: np.ndarray, nameA: str, nameB: str, title: str):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06)
    fig.add_trace(go.Heatmap(z=A, coloraxis="coloraxis"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=B, coloraxis="coloraxis"), row=1, col=2)
    fig.update_layout(coloraxis={'colorscale':'Magma'}, title=title, height=360,
                      margin=dict(l=10,r=10,t=40,b=10))
    fig.update_xaxes(title_text="Index", row=1,col=1); fig.update_yaxes(title_text="Index", row=1,col=1)
    fig.update_xaxes(title_text="Index", row=1,col=2); fig.update_yaxes(title_text="Index", row=1,col=2)
    fig.update_layout(annotations=[
        dict(text=nameA, x=0.22, xref="paper", y=1.12, yref="paper", showarrow=False),
        dict(text=nameB, x=0.78, xref="paper", y=1.12, yref="paper", showarrow=False)
    ])
    return fig

def plot_distance_matrices(D1: np.ndarray, D2: np.ndarray, name1: str, name2: str):
    """
    Displays two Cα distance matrices side-by-side using Matplotlib for performance.
    This replaces the original plotly-based `make_dual_heat` and `plot_distance_matrices`.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True, sharey=True)
    fig.suptitle("Distance Matrices (Cα–Cα)", fontsize=14)

    # Find shared min/max values for a consistent color scale
    vmin = min(np.min(D1) if D1.size > 0 else 0, np.min(D2) if D2.size > 0 else 0)
    vmax = max(np.max(D1) if D1.size > 0 else 1, np.max(D2) if D2.size > 0 else 1)

    # Plot the first matrix (Reference)
    im = axes[0].imshow(D1, cmap='magma', vmin=vmin, vmax=vmax, interpolation='none')
    axes[0].set_title(name1)
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")

    # Plot the second matrix (Mobile)
    axes[1].imshow(D2, cmap='magma', vmin=vmin, vmax=vmax, interpolation='none')
    axes[1].set_title(name2)
    axes[1].set_xlabel("Residue Index")

    # Add a single colorbar for both plots
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Distance (Å)")

    # Display the plot in the Streamlit app
    st.pyplot(fig)

def plot_pair_distance_hist(dists: np.ndarray, title: str):
    if dists is None or len(dists)==0: st.info("No pairs to plot."); return
    fig = go.Figure(); fig.add_trace(go.Histogram(x=dists, nbinsx=30))
    fig.update_layout(title=title, xaxis_title="Per-pair distance after superposition (Å)",
                      yaxis_title="Count", height=300, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ===========================================
# Structure-based sequence alignment
# ===========================================
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

# ===========================================
# Exports
# ===========================================
def export_zip_seqguided(results: dict, aln_text: str, ref_atoms, mob_atoms) -> Tuple[io.BytesIO, str]:
    df = pd.DataFrame({
        "Residue Label": results["residue_labels"],
        "RMSD": results["per_residue_rmsd"],
        "Ref_X": results["ref_coords"][:,0], "Ref_Y": results["ref_coords"][:,1], "Ref_Z": results["ref_coords"][:,2],
        "Mob_X": results["mob_coords_transformed"][:,0], "Mob_Y": results["mob_coords_transformed"][:,1], "Mob_Z": results["mob_coords_transformed"][:,2],
    })
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)

    def pdb_from_atoms(atoms, chain_letter, override=None):
        lines=["MODEL        1"]; serial=1
        for idx, atom in enumerate(atoms):
            res=atom.get_parent(); resname=res.get_resname()
            rid=res.get_id(); resnum=int(rid[1])
            if override is not None: x,y,z=override[idx]
            else: x,y,z=atom.get_coord()
            lines.append(f"ATOM  {serial:5d}  CA  {resname:>3} {chain_letter}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"); serial+=1
        lines.append("ENDMDL"); return "\n".join(lines)

    ref_pdb=pdb_from_atoms(ref_atoms,"A",None)
    mob_pdb=pdb_from_atoms(mob_atoms,"B",results["mob_coords_transformed"])

    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,"w") as z:
        z.writestr("per_residue_data.csv", csv_buf.getvalue())
        z.writestr("aligned_ref_CA_only.pdb", ref_pdb)
        z.writestr("aligned_mobile_CA_only.pdb", mob_pdb)
        z.writestr("alignment.txt", aln_text)
    zbuf.seek(0); name=f"seqguided_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

def export_zip_seqfree(ref_file: str, mob_file: str, R: np.ndarray, t: np.ndarray,
                       ref_subset_pairs_df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    pathA, pathB = write_uploads_to_temp_files(ref_file, mob_file)
    parser = MMCIFParser(QUIET=True) if pathB.lower().endswith((".cif",".mmcif")) else PDBParser(QUIET=True)
    mob_struct = parser.get_structure(os.path.basename(pathB), pathB)
    io_obj = PDBIO(); io_obj.set_structure(mob_struct)
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb"); out_path=out.name; out.close()
    io_obj.save(out_path, _AllAtomsSelect(R=R, t=t))
    with open(out_path,"rb") as fh: aligned_mobile_full = fh.read()
    for p in (out_path, pathA, pathB):
        try: os.unlink(p)
        except Exception: pass

    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,"w") as z:
        z.writestr("aligned_mobile_fullatom_on_reference.pdb", aligned_mobile_full)
        z.writestr("matched_pairs.csv", ref_subset_pairs_df.to_csv(index=False))
    zbuf.seek(0); name=f"seqfree_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

# ===========================================
# UI — sidebar
# ===========================================
st.title("🧬 Structure Alignment Workhorse")
st.write("Upload two or more structures. Select reference and mobile. Choose mode. Run and review results, peaks, and exports.")

with st.sidebar:
    st.header("📤 Upload")
    uploads = st.file_uploader("Select PDB or mmCIF files", type=["pdb","cif","mmcif"], accept_multiple_files=True)
    if uploads:
        st.session_state.uploaded_blob_map = {f.name: f for f in uploads}
        new_names = sorted([f.name for f in uploads]); old_names = sorted(list(st.session_state.structures.keys()))
        if new_names != old_names:
            st.session_state.structures.clear(); st.session_state.sequences.clear(); st.session_state.chain_lengths.clear()
            with st.spinner("Parsing structures..."):
                for f in uploads:
                    s, err = parse_structure(f)
                    if s:
                        st.session_state.structures[f.name] = s
                        seqs, lens = extract_sequences_and_lengths(s, f.name)
                        st.session_state.sequences[f.name] = seqs
                        st.session_state.chain_lengths[f.name] = lens
                    else:
                        log(err or f"Failed to parse {f.name}")

    if len(st.session_state.structures) >= 2:
        st.header("🧭 Choose files")
        files = list(st.session_state.structures.keys())
        ref_default = files.index(st.session_state.selected_ref_file) if st.session_state.selected_ref_file in files else 0
        ref_file = st.selectbox("Reference file", options=files, index=ref_default, key="ref_file_box")
        mob_options = [f for f in files if f != ref_file]
        mob_default = (mob_options.index(st.session_state.selected_mobile_file)
                       if st.session_state.selected_mobile_file in mob_options else 0)
        mob_file = st.selectbox("Mobile file (to align)", options=mob_options, index=mob_default, key="mob_file_box")

        st.subheader("🔗 Chain similarity matrix")
        df_id, df_sc = compute_chain_similarity_matrix(ref_file, mob_file)
        if not df_id.empty:
            tab1, tab2 = st.tabs(["% identity","BLOSUM62 score/len"])
            with tab1:
                st.plotly_chart(px.imshow(df_id, text_auto=".1f", color_continuous_scale="Viridis", aspect="auto", origin="upper")
                                .update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10)),
                                use_container_width=True)
            with tab2:
                st.plotly_chart(px.imshow(df_sc, text_auto=".2f", color_continuous_scale="Plasma", aspect="auto", origin="upper")
                                .update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10)),
                                use_container_width=True)
        else:
            st.info("Sequences not available to compute similarity matrix.")

        st.subheader("🧯 Chain selection")
        def with_len(fname, ch): return f"{ch} ({st.session_state.chain_lengths.get(fname, {}).get(ch, 0)} aa)"
        ref_chs_all = list(st.session_state.sequences.get(ref_file, {}).keys())
        mob_chs_all = list(st.session_state.sequences.get(mob_file, {}).keys())
        ref_labels = [with_len(ref_file,c) for c in ref_chs_all]
        mob_labels = [with_len(mob_file,c) for c in mob_chs_all]
        parse_lbl = lambda s: s.split()[0]

        prev_ref = st.session_state.selected_ref_chains
        prev_mob = st.session_state.selected_mobile_chains
        ref_sel = st.multiselect("Reference chains", options=ref_labels,
                                 default=[with_len(ref_file,c) for c in prev_ref if c in ref_chs_all] or (ref_labels[:1] if ref_labels else []))
        mob_sel = st.multiselect("Mobile chains", options=mob_labels,
                                 default=[with_len(mob_file,c) for c in prev_mob if c in mob_chs_all] or (mob_labels[:1] if mob_labels else []))
        st.session_state.selected_ref_file = ref_file
        st.session_state.selected_mobile_file = mob_file
        st.session_state.selected_ref_chains = [parse_lbl(x) for x in ref_sel]
        st.session_state.selected_mobile_chains = [parse_lbl(x) for x in mob_sel]

        st.divider()
        st.header("⚙️ Alignment mode")
        st.session_state.seq_gap_open = st.slider("Sequence gap open penalty", -20, -1, st.session_state.seq_gap_open, key="gap_open_slider")
        st.session_state.seq_gap_extend = st.slider("Sequence gap extend penalty", -20.0, -0.1, st.session_state.seq_gap_extend, step=0.1, key="gap_extend_slider")
        mode = st.selectbox("Choose alignment mode",
                            ["Auto (best RMSD)","Sequence-guided","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"],
                            index=["Auto (best RMSD)","Sequence-guided","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"].index(st.session_state.align_mode),
                            key="mode_select")
        st.session_state.align_mode = mode
        run = st.button("🚀 Run alignment", use_container_width=True, key="run_button")
    else:
        st.info("Upload at least two structures to proceed.")
        run = False

# ===========================================
# Selection logic
# ===========================================
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

# ===========================================
# Main run
# ===========================================
if run:
    ref_file = st.session_state.selected_ref_file
    mob_file = st.session_state.selected_mobile_file
    ref_chs = st.session_state.selected_ref_chains
    mob_chs = st.session_state.selected_mobile_chains
    if not (ref_file and mob_file and ref_chs and mob_chs):
        st.error("Select both files and at least one chain per file."); st.stop()

    key = json.dumps(dict(ref=ref_file, mob=mob_file, ref_chs=ref_chs, mob_chs=mob_chs,
                          mode=st.session_state.align_mode,
                          gap_open=st.session_state.seq_gap_open, gap_ext=st.session_state.seq_gap_extend), sort_keys=True)
    if key in st.session_state.results_cache:
        log("Cache hit for identical parameters.")
        result_bundle = st.session_state.results_cache[key]
    else:
        log(f"Running mode: {st.session_state.align_mode}")
        ref_struct = st.session_state.structures[ref_file]
        mob_struct = st.session_state.structures[mob_file]

        # Sequence-guided (if needed)
        seqguided=None
        if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-guided"):
            seqA="".join(str(st.session_state.sequences[ref_file][c].seq) for c in ref_chs if c in st.session_state.sequences[ref_file])
            seqB="".join(str(st.session_state.sequences[mob_file][c].seq) for c in mob_chs if c in st.session_state.sequences[mob_file])
            aln=perform_sequence_alignment(seqA, seqB, st.session_state.seq_gap_open, st.session_state.seq_gap_extend)
            if aln:
                ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(ref_struct, ref_chs, mob_struct, mob_chs, aln)
                if ref_atoms and mob_atoms:
                    si = superimpose_atoms(ref_atoms, mob_atoms)
                    if si:
                        seqguided = dict(aln=aln, ref_atoms=ref_atoms, mob_atoms=mob_atoms, si=si)
                        log(f"Sequence-guided RMSD = {si['rmsd']:.3f} Å with {len(ref_atoms)} pairs.")
            else:
                log("Sequence-guided alignment could not be built.")

        # Sequence-free (if needed)
        seqfree=None
        if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"):
            pathA, pathB = write_uploads_to_temp_files(ref_file, mob_file)
            try:
                method = "auto" if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-free (auto)") else (
                         "shape" if st.session_state.align_mode=="Sequence-free (shape)" else "window")
                res = sequence_independent_alignment_joined_v2(
                    file_ref=pathA, file_mob=pathB,
                    chains_ref=ref_chs, chains_mob=mob_chs,
                    method=method, shape_nbins=48, shape_gap_penalty=2.0, shape_band_frac=0.30,
                    inlier_rmsd_cut=5.0, inlier_quantile=0.85, refinement_iters=50
                )
                seqfree=res
                log(f"Sequence-free [{res.method}] RMSD = {res.rmsd:.3f} Å with {res.kept_pairs} pairs.")
            finally:
                for p in (pathA, pathB):
                    try: os.unlink(p)
                    except Exception: pass

        # Decide best
        if st.session_state.align_mode == "Auto (best RMSD)":
            best, reason = pick_best_overall(seqguided, seqfree, min_pairs=3)
            if best is None:
                st.error("No alignment could be produced. Try different chains or mode."); st.stop()
            chosen_name = best["name"]
            chosen = dict(name=chosen_name, reason=reason,
                          seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                          seqfree=seqfree if "Sequence-free" in chosen_name else None)
            log(f"AUTO chose: {chosen_name}. Reason: {reason}")
        else:
            chosen = dict(name=st.session_state.align_mode, reason="Manual mode.",
                          seqguided=seqguided if st.session_state.align_mode=="Sequence-guided" else None,
                          seqfree=seqfree if "Sequence-free" in st.session_state.align_mode else None)

        result_bundle = dict(seqguided=seqguided, seqfree=seqfree, chosen=chosen)
        st.session_state.results_cache[key] = result_bundle

    st.session_state.last_run_summary = result_bundle

# ===========================================
# Results display
# ===========================================
if st.session_state.last_run_summary:
    seqguided = st.session_state.last_run_summary["seqguided"]
    seqfree = st.session_state.last_run_summary["seqfree"]
    chosen = st.session_state.last_run_summary["chosen"]

    st.success(f"Alignment complete. Chosen mode: **{chosen['name']}**")
    st.caption(f"Selection criterion: lowest RMSD (pairs only to break ties). Reason: {chosen.get('reason','—')}")

    cols = st.columns(3)
    with cols[0]:
        if seqguided:
            st.metric("Sequence-guided RMSD (Å)", f"{seqguided['si']['rmsd']:.3f}")
            st.caption(f"Pairs: {len(seqguided['ref_atoms'])}")
        else:
            st.metric("Sequence-guided RMSD (Å)", "—")
    with cols[1]:
        if seqfree:
            st.metric(f"Seq-free RMSD (Å) [{seqfree.method}]", f"{seqfree.rmsd:.3f}")
            st.caption(f"Pairs kept: {seqfree.kept_pairs}")
        else:
            st.metric("Seq-free RMSD (Å)", "—")
    with cols[2]:
        st.metric("Mode selected", chosen["name"])

    st.divider()

    # ===== A) Sequence-guided =====
    if seqguided:
        st.header("A. Sequence-guided alignment")
        txt_block = formatted_alignment_text(
            st.session_state.selected_ref_file, seqguided["aln"].seqA,
            st.session_state.selected_mobile_file, seqguided["aln"].seqB,
            seqguided["aln"].score, interval=10
        )
        with st.expander("Show alignment text", expanded=False):
            st.code(txt_block)

        id_pairs = [(i,i) for i in range(len(seqguided["si"]["residue_labels"]))]
        sg_key_prefix = f"sg3d_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}"
        plot_superposition_3d(
            ref_coords=seqguided["si"]["ref_coords"],
            mob_coords=seqguided["si"]["mob_coords_transformed"],
            title="Sequence-guided superposition with RMSD peaks",
            labels=seqguided["si"]["residue_labels"],
            pairs=id_pairs,
            default_top_k=10,
            key_prefix=sg_key_prefix
        )

        zbuf, zname = export_zip_seqguided(seqguided["si"], txt_block, seqguided["ref_atoms"], seqguided["mob_atoms"])
        st.download_button("⬇️ Download sequence-guided ZIP (CA, CSV, TXT)", data=zbuf, file_name=zname, mime="application/zip")
        st.divider()

    # ===== B) Sequence-independent =====
    if seqfree:
        st.header("B. Sequence-independent alignment")
        with st.expander("Diagnostics: distance and distributions", expanded=False):
            plot_distance_matrices(_pairwise_dists(seqfree.ref_subset_ca_coords),
                                   _pairwise_dists(seqfree.mob_subset_ca_coords_aligned),
                                   "Reference subset", "Mobile subset (aligned)")
            if len(seqfree.pairs) >= 3:
                mob_sub_aln = seqfree.mob_subset_ca_coords_aligned
                dists = np.array([np.linalg.norm(seqfree.ref_subset_ca_coords[i] - mob_sub_aln[j])
                                  for (i,j) in seqfree.pairs])
                plot_pair_distance_hist(dists, "Per-pair distances after superposition")

        if len(seqfree.pairs) >= 1:
            # NEW: toggle full chains vs matched subset
            show_full = st.checkbox(
                "Show FULL selected chains (Cα) transformed by the seq-free alignment",
                value=True, key="sf_show_full"
            )

            # Labels for peaks come from the matched pairs (ref side)
            labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}" for (i,_) in seqfree.pairs]

            if show_full:
                # Draw FULL selected chains; compute peaks over matched pairs
                sf_key_prefix = f"sf3d_full_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}_{seqfree.method}"
                plot_superposition_3d(
                    ref_coords=seqfree.ref_subset_ca_coords,                 # all CA of selected ref chains
                    mob_coords=seqfree.mob_subset_ca_coords_aligned,         # all CA of selected mob chains (after R,t)
                    title=f"Seq-free superposition [{seqfree.method}] • FULL selected chains • RMSD {seqfree.rmsd:.3f} Å",
                    labels=labels,
                    pairs=seqfree.pairs,                                     # peaks computed on matched pairs
                    default_top_k=10,
                    key_prefix=sf_key_prefix
                )
            else:
                # Draw only matched subset for both structures (as before)
                ref_pts = np.vstack([seqfree.ref_subset_ca_coords[i] for (i,j) in seqfree.pairs])
                mob_pts = np.vstack([seqfree.mob_subset_ca_coords_aligned[j] for (i,j) in seqfree.pairs])
                id_pairs = [(k,k) for k in range(len(labels))]
                sf_key_prefix = f"sf3d_pairs_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}_{seqfree.method}"
                plot_superposition_3d(
                    ref_coords=ref_pts,
                    mob_coords=mob_pts,
                    title=f"Seq-free superposition [{seqfree.method}] • MATCHED subset • RMSD {seqfree.rmsd:.3f} Å",
                    labels=labels,
                    pairs=id_pairs,
                    default_top_k=10,
                    key_prefix=sf_key_prefix
                )

            # Table of matched pairs
            rows=[]
            for k,(i_ref,j_mob) in enumerate(seqfree.pairs):
                ri = seqfree.ref_subset_infos[i_ref]
                mi = seqfree.mob_subset_infos[j_mob]
                d = float(np.linalg.norm(seqfree.ref_subset_ca_coords[i_ref] - seqfree.mob_subset_ca_coords_aligned[j_mob]))
                rows.append(dict(rank=k, ref_chain=ri.chain_id, ref_resseq=ri.resseq, ref_resname=ri.resname,
                                 mob_chain=mi.chain_id, mob_resseq=mi.resseq, mob_resname=mi.resname,
                                 pair_distance_A=d))
            df_pairs = pd.DataFrame(rows)
            st.dataframe(df_pairs, use_container_width=True, hide_index=True)

            # Structure-based sequence alignment (derived from matched pairs path)
            a1,a2,_ = structure_based_alignment_strings(seqfree.ref_subset_infos, seqfree.mob_subset_infos, seqfree.pairs)
            with st.expander("Show structure-based sequence alignment", expanded=False):
                idA = f"{st.session_state.selected_ref_file} ({','.join(st.session_state.selected_ref_chains)})"
                idB = f"{st.session_state.selected_mobile_file} ({','.join(st.session_state.selected_mobile_chains)})"
                st.code(formatted_alignment_text(idA, a1, idB, a2, score=0.0, interval=10))

            # Export full-atom transformed + pairs.csv
            zbuf2, zname2 = export_zip_seqfree(st.session_state.selected_ref_file,
                                               st.session_state.selected_mobile_file,
                                               seqfree.rotation, seqfree.translation,
                                               df_pairs)
            st.download_button("⬇️ Download seq-free ZIP (full-atom aligned PDB + pairs.csv)",
                               data=zbuf2, file_name=zname2, mime="application/zip")
            st.divider()

    with st.expander("📜 Log", expanded=False):
        if not st.session_state.logs: st.write("No log messages.")
        else: st.text("\n".join(st.session_state.logs))
