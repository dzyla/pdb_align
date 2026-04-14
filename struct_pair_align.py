# app.py
# =========================================================
# Structure Alignment Suite
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
from pdb_align.core import (
    VALID_AA_3, AA_DICT, _aa_dict, ResidueInfo, AlignSummary, AlignmentResultSF, _AllAtomsSelect,
    extract_sequences_and_lengths, _parse_path, _chain_ids,
    _resolve_selectors, _extract_ca_infos, _pairwise_dists, _kabsch, _transform,
    _robust_inlier_mask, _window_pairs, _radial_histograms, _chi2_distance,
    _banded_dp_maxscore, _shape_pairs, sequence_independent_alignment_joined_v2,
    perform_sequence_alignment, get_aligned_atoms_by_alignment, superimpose_atoms,
    compute_chain_similarity_matrix, structure_based_alignment_strings, pick_best_overall
)



# Page config
st.set_page_config(page_title="Structure Alignment Suite", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
code, pre { font-size: 12px !important; line-height: 1.25; }
</style>
""", unsafe_allow_html=True)

# ===========================================
# Helpers and constants
# ===========================================
def init_state():
    defaults = dict(
        structures={}, sequences={}, chain_lengths={},
        selected_ref_file=None, selected_mobile_file=None,
        selected_ref_chains=[], selected_mobile_chains=[],
        align_mode="Auto (best RMSD)",
        seq_gap_open=-10, seq_gap_extend=-0.5,
        results_cache={}, last_run_summary=None, logs=[],
        fetched_blobs={}
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
def fetch_structure_to_blob(file_or_id: str):
    import requests
    if file_or_id.lower().startswith("pdb:"):
        pdb_id = file_or_id[4:].strip().upper()
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif")
        if r.status_code == 200:
            blob = io.BytesIO(r.content)
            blob.name = f"{pdb_id}.cif"
            return blob, None
        return None, f"Could not fetch PDB {pdb_id}"
    elif file_or_id.lower().startswith("af:"):
        af_id = file_or_id[3:].strip().upper()
        r = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v6.pdb")
        if r.status_code == 200:
            blob = io.BytesIO(r.content)
            blob.name = f"AF-{af_id}.pdb"
            return blob, None
        return None, f"Could not fetch AlphaFold model {af_id}"
    return None, "Invalid ID format. Use pdb:XXXX or af:XXXX"

def parse_structure(uploaded_file):
    try:
        raw = uploaded_file.getvalue()
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw); tmp_path = tmp.name
        s = _parse_path(tmp_path)
        os.unlink(tmp_path)
        return s, None
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, f"Error parsing {uploaded_file.name}: {e}"

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
# Chain similarity matrix
# ===========================================
def compute_chain_similarity_matrix_ui(fileA:str, fileB:str)->Tuple[pd.DataFrame,pd.DataFrame]:
    seqsA=st.session_state.sequences.get(fileA, {})
    seqsB=st.session_state.sequences.get(fileB, {})
    return compute_chain_similarity_matrix(seqsA, seqsB)

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
        _plot_basic_3d(ref_draw, mob_draw, title, key_suffix=kp)
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

    ref_draw = np.array(ref_coords, dtype=float)
    mob_draw = np.array(mob_coords, dtype=float)

    # Build 3D figure (Plotly overview)
    st.write("### Analytical Distance Plotly Overview")
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
    st.plotly_chart(fig, use_container_width=True, key=f"{kp}_main_plot")

def _plot_basic_3d(ref_coords, mob_coords, title, key_suffix=""):
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=ref_coords[:,0], y=ref_coords[:,1], z=ref_coords[:,2],
                               mode='lines+markers', name='Reference Cα',
                               marker=dict(size=2), line=dict(width=2)))
    fig.add_trace(go.Scatter3d(x=mob_coords[:,0], y=mob_coords[:,1], z=mob_coords[:,2],
                               mode='lines+markers', name='Mobile Cα (Aligned)',
                               marker=dict(size=2), line=dict(width=2)))
    fig.update_layout(title=title, scene=dict(aspectmode='data', xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                      height=560, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, key=f"basic3d_{key_suffix}")

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
    import matplotlib.pyplot as plt
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

def plot_pair_distance_hist(dists: np.ndarray, title: str, key_suffix=""):
    if dists is None or len(dists)==0: st.info("No pairs to plot."); return
    fig = go.Figure(); fig.add_trace(go.Histogram(x=dists, nbinsx=30))
    fig.update_layout(title=title, xaxis_title="Per-pair distance after superposition (Å)",
                      yaxis_title="Count", height=300, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{key_suffix}")

# ===========================================
# Exports
# ===========================================
class _ChainSelect(Select):
    def __init__(self, chains):
        self.chains = set(chains)
    def accept_chain(self, chain):
        if chain.id in self.chains: return 1
        return 0

def _remap_long_chain_ids(structure):
    """Rename chains whose IDs exceed 1 character (PDB format limit) to single-char IDs."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    used = {chain.id for model in structure for chain in model if len(chain.id) == 1}
    replacements = {}
    for model in structure:
        for chain in model:
            if len(chain.id) > 1:
                if chain.id not in replacements:
                    for c in alphabet:
                        if c not in used:
                            replacements[chain.id] = c
                            used.add(c)
                            break
                chain.id = replacements.get(chain.id, chain.id[0])

def _generate_4_pdbs(ref_file: str, mob_file: str, ref_chains: list, mob_chains: list, R: np.ndarray, t: np.ndarray):
    pathA, pathB = write_uploads_to_temp_files(ref_file, mob_file)
    parserA = MMCIFParser(QUIET=True) if pathA.lower().endswith((".cif",".mmcif")) else PDBParser(QUIET=True)
    parserB = MMCIFParser(QUIET=True) if pathB.lower().endswith((".cif",".mmcif")) else PDBParser(QUIET=True)
    
    ref_struct = parserA.get_structure("ref", pathA)
    mob_struct = parserB.get_structure("mob", pathB)

    _remap_long_chain_ids(ref_struct)
    _remap_long_chain_ids(mob_struct)

    for model in mob_struct:
        for atom in model.get_atoms():
            c = atom.get_coord()
            atom.set_coord(np.dot(c, R.T) + t)

    io_obj = PDBIO()

    io_obj.set_structure(ref_struct)
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb"); pR_full=out.name; out.close()
    io_obj.save(pR_full, Select())
    with open(pR_full,"rb") as fh: ref_full = fh.read()

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb"); pR_sel=out.name; out.close()
    io_obj.save(pR_sel, _ChainSelect(ref_chains))
    with open(pR_sel,"rb") as fh: ref_sel = fh.read()

    io_obj.set_structure(mob_struct)
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb"); pM_full=out.name; out.close()
    io_obj.save(pM_full, Select())
    with open(pM_full,"rb") as fh: mob_full = fh.read()

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb"); pM_sel=out.name; out.close()
    io_obj.save(pM_sel, _ChainSelect(mob_chains))
    with open(pM_sel,"rb") as fh: mob_sel = fh.read()

    for p in (pR_full, pR_sel, pM_full, pM_sel, pathA, pathB):
        try: os.unlink(p)
        except Exception: pass

    return ref_full, ref_sel, mob_full, mob_sel

def _generate_pymol_script(ref_name, mob_name, rmsd_df):
    script = f"# PyMOL alignment visualization\\n"
    script += f"load {ref_name}\\n"
    script += f"load {mob_name}\\n"
    script += "hide everything\\nshow cartoon\\n"
    script += "color white, ref_full\\ncolor cyan, mob_full\\n"
    if rmsd_df is not None and not rmsd_df.empty:
        script += "alter all, b=0.0\\n"
        # B-factor injection
        for _, row in rmsd_df.iterrows():
            lbl = row['Residue Label']
            val = row['RMSD']
            if ':' in lbl:
                ch, res = lbl.split(':')
                resseq = "".join([c for c in res if c.isdigit() or c == '-'])
                script += f"alter (mob_full and chain {ch} and resi {resseq}), b={val:.3f}\\n"
        script += "spectrum b, blue_white_red, mob_full, minimum=0, maximum=10\\n"
    script += "zoom\\n"
    return script.encode()

def export_zip_batch(results: list, summary_df: pd.DataFrame, ref_file: str, ref_chains: list, mob_chs_dict: dict) -> Tuple[io.BytesIO, str]:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("batch_summary.csv", summary_df.to_csv(index=False))
        
        for r in results:
            mob = r["mob_file"]
            chsn = r["chosen"]
            
            if "Sequence-guided" in chsn["name"] and r["seqguided"]:
                sg = r["seqguided"]
                rmsd_df = pd.DataFrame({"Residue Label": sg["si"]["residue_labels"], "RMSD": sg["si"]["per_residue_rmsd"]})
                z.writestr(f"{mob}/seq_guided_rmsd_per_position.csv", rmsd_df.to_csv(index=False))
                
                rf, rs, mf, ms = _generate_4_pdbs(ref_file, mob, ref_chains, mob_chs_dict.get(mob, []), sg["si"]["rotation"], sg["si"]["translation"])
                
                r_base = os.path.splitext(os.path.basename(ref_file))[0]
                m_base = os.path.splitext(os.path.basename(mob))[0]
                
                z.writestr(f"{mob}/{r_base}_aligned_ref_full.pdb", rf)
                z.writestr(f"{mob}/{r_base}_aligned_ref_selected.pdb", rs)
                z.writestr(f"{mob}/{m_base}_aligned_mobile_full.pdb", mf)
                z.writestr(f"{mob}/{m_base}_aligned_mobile_selected.pdb", ms)
                z.writestr(f"{mob}/visualize.pml", _generate_pymol_script(f"{r_base}_aligned_ref_full.pdb", f"{m_base}_aligned_mobile_full.pdb", rmsd_df))
                
            elif "Sequence-free" in chsn["name"] and r.get("seqfree") is not None:
                sf = r["seqfree"]
                sf_rmsd_labels = [f"{sf.ref_subset_infos[i].chain_id}:{sf.ref_subset_infos[i].resseq}{(sf.ref_subset_infos[i].icode or '').strip()}" for (i, _) in sf.pairs]
                sf_per_res_rmsd = [np.linalg.norm(sf.ref_subset_ca_coords[i] - sf.mob_subset_ca_coords_aligned[j]) for (i, j) in sf.pairs]
                sf_rmsd_df = pd.DataFrame({"Residue Label": sf_rmsd_labels, "RMSD": sf_per_res_rmsd})
                z.writestr(f"{mob}/seq_free_rmsd_per_position.csv", sf_rmsd_df.to_csv(index=False))
                
                rf, rs, mf, ms = _generate_4_pdbs(ref_file, mob, ref_chains, mob_chs_dict.get(mob, []), sf.rotation, sf.translation)
                
                r_base = os.path.splitext(os.path.basename(ref_file))[0]
                m_base = os.path.splitext(os.path.basename(mob))[0]
                
                z.writestr(f"{mob}/{r_base}_aligned_ref_full.pdb", rf)
                z.writestr(f"{mob}/{r_base}_aligned_ref_selected.pdb", rs)
                z.writestr(f"{mob}/{m_base}_aligned_mobile_full.pdb", mf)
                z.writestr(f"{mob}/{m_base}_aligned_mobile_selected.pdb", ms)
                z.writestr(f"{mob}/visualize.pml", _generate_pymol_script(f"{r_base}_aligned_ref_full.pdb", f"{m_base}_aligned_mobile_full.pdb", sf_rmsd_df))

    zbuf.seek(0)
    name = f"batch_alignment_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

def export_zip_seqguided(ref_file: str, mob_file: str, ref_chains: list, mob_chains: list, 
                         results: dict, aln_text: str) -> Tuple[io.BytesIO, str]:
    df = pd.DataFrame({
        "Residue Label": results["residue_labels"],
        "RMSD": results["per_residue_rmsd"],
        "Ref_X": results["ref_coords"][:,0], "Ref_Y": results["ref_coords"][:,1], "Ref_Z": results["ref_coords"][:,2],
        "Mob_X": results["mob_coords_transformed"][:,0], "Mob_Y": results["mob_coords_transformed"][:,1], "Mob_Z": results["mob_coords_transformed"][:,2],
    })
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)

    rmsd_df = pd.DataFrame({
        "Residue Label": results["residue_labels"],
        "RMSD": results["per_residue_rmsd"]
    })
    rmsd_csv_buf = io.StringIO(); rmsd_df.to_csv(rmsd_csv_buf, index=False)

    rf, rs, mf, ms = _generate_4_pdbs(ref_file, mob_file, ref_chains, mob_chains, results["rotation"], results["translation"])

    r_base = os.path.splitext(os.path.basename(ref_file))[0]
    m_base = os.path.splitext(os.path.basename(mob_file))[0]

    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,"w") as z:
        z.writestr("per_residue_data.csv", csv_buf.getvalue())
        z.writestr("rmsd_per_position.csv", rmsd_csv_buf.getvalue())
        z.writestr(f"{r_base}_aligned_ref_full.pdb", rf)
        z.writestr(f"{r_base}_aligned_ref_selected.pdb", rs)
        z.writestr(f"{m_base}_aligned_mobile_full.pdb", mf)
        z.writestr(f"{m_base}_aligned_mobile_selected.pdb", ms)
        z.writestr("alignment.txt", aln_text)
    zbuf.seek(0); name=f"seqguided_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

def export_zip_seqfree(ref_file: str, mob_file: str, ref_chains: list, mob_chains: list, 
                       seqfree: AlignmentResultSF,
                       ref_subset_pairs_df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    
    rf, rs, mf, ms = _generate_4_pdbs(ref_file, mob_file, ref_chains, mob_chains, seqfree.rotation, seqfree.translation)

    sf_rmsd_labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}{(seqfree.ref_subset_infos[i].icode or '').strip()}" for (i, _) in seqfree.pairs]
    sf_per_res_rmsd = [np.linalg.norm(seqfree.ref_subset_ca_coords[i] - seqfree.mob_subset_ca_coords_aligned[j]) for (i, j) in seqfree.pairs]
    sf_rmsd_df = pd.DataFrame({
        "Residue Label": sf_rmsd_labels,
        "RMSD": sf_per_res_rmsd
    })

    r_base = os.path.splitext(os.path.basename(ref_file))[0]
    m_base = os.path.splitext(os.path.basename(mob_file))[0]

    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,"w") as z:
        z.writestr(f"{r_base}_aligned_ref_full.pdb", rf)
        z.writestr(f"{r_base}_aligned_ref_selected.pdb", rs)
        z.writestr(f"{m_base}_aligned_mobile_full.pdb", mf)
        z.writestr(f"{m_base}_aligned_mobile_selected.pdb", ms)
        z.writestr("matched_pairs.csv", ref_subset_pairs_df.to_csv(index=False))
        z.writestr("rmsd_per_position.csv", sf_rmsd_df.to_csv(index=False))
        if seqfree.method == "shape" and seqfree.shift_matrix is not None:
            z.writestr("shift_matrix.csv", pd.DataFrame(seqfree.shift_matrix).to_csv(index=False))
        elif seqfree.method == "window" and seqfree.shift_scores is not None:
            z.writestr("shift_scores.csv", pd.DataFrame({"Score": seqfree.shift_scores}).to_csv(index=False))

    zbuf.seek(0); name=f"seqfree_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

# ===========================================
# UI — sidebar
# ===========================================
st.title("🧬 Structure Alignment Workhorse")
st.write("Upload two or more structures. Select reference and mobile. Choose mode. Run and review results, peaks, and exports.")

with st.sidebar:
    st.header("📤 Upload & Fetch")
    uploads = st.file_uploader("Select PDB or mmCIF files", type=["pdb","cif","mmcif"], accept_multiple_files=True)
    fetch_id = st.text_input("Or fetch (e.g., pdb:8UUP, af:P00533, pdb:1ABC)")
    fetch_btn = st.button("Fetch Structure(s)", use_container_width=True)

    if fetch_btn and fetch_id:
        targets = [t.strip() for t in fetch_id.split(",") if t.strip()]
        if targets:
            with st.spinner(f"Fetching {len(targets)} structures concurrently..."):
                from concurrent.futures import ThreadPoolExecutor
                def _fetch(t): return (t, fetch_structure_to_blob(t))
                
                try:
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        results = list(executor.map(_fetch, targets))
                    
                    for t, (blob, err) in results:
                        if blob:
                            st.session_state.fetched_blobs[blob.name] = blob
                        else:
                            st.error(f"{err} for {t}")
                except Exception as e:
                    st.error(f"Concurrent fetch error: {e}")

    if st.session_state.fetched_blobs:
        st.write("Fetched structures:")
        # We will use pills if available in this streamlit version, or fallback to small buttons
        for fname in list(st.session_state.fetched_blobs.keys()):
            if st.button(f"✕ {fname}", key=f"del_{fname}", help="Click to remove"):
                del st.session_state.fetched_blobs[fname]
                st.rerun()

    current_blobs = list(uploads) if uploads else []
    current_blobs.extend(st.session_state.fetched_blobs.values())

    if current_blobs:
        st.session_state.uploaded_blob_map = {f.name: f for f in current_blobs}
        new_names = sorted([f.name for f in current_blobs])
        old_names = sorted(list(st.session_state.structures.keys()))
        if new_names != old_names:
            st.session_state.structures.clear(); st.session_state.sequences.clear(); st.session_state.chain_lengths.clear()
            st.session_state.last_run_summary = [r for r in (st.session_state.last_run_summary or []) if r["mob_file"] in new_names and st.session_state.get("selected_ref_file") in new_names]
            with st.spinner("Parsing structures..."):
                for f in current_blobs:
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
        
        # Change mobile selection to multiselect
        prev_mob_files = st.session_state.get("selected_mobile_files", [])
        valid_prev = [m for m in prev_mob_files if m in mob_options]
        mob_files = st.multiselect("Mobile structures (to align against reference)", options=mob_options, default=valid_prev, key="mob_files_box")

        st.subheader("🧯 Chain selection")
        def with_len(fname, ch): return f"{ch} ({st.session_state.chain_lengths.get(fname, {}).get(ch, 0)} aa)"
        ref_chs_all = list(st.session_state.sequences.get(ref_file, {}).keys())
        ref_labels = [with_len(ref_file,c) for c in ref_chs_all]
        parse_lbl = lambda s: s.split()[0]

        prev_ref = st.session_state.selected_ref_chains

        with st.form("alignment_settings_form"):
            ref_sel = st.multiselect("Reference chains", options=ref_labels,
                                     default=[with_len(ref_file,c) for c in prev_ref if c in ref_chs_all] or (ref_labels[:1] if ref_labels else []))
            
            mob_sel_dict = {}
            for m_file in mob_files:
                m_chs_all = list(st.session_state.sequences.get(m_file, {}).keys())
                m_labels = [with_len(m_file, c) for c in m_chs_all]
                prev_m_chs = st.session_state.get(f"selected_mob_chains_{m_file}", [])
                mob_sel_dict[m_file] = st.multiselect(
                    f"Selected chains for {m_file}", options=m_labels,
                    default=[with_len(m_file, c) for c in prev_m_chs if c in m_chs_all] or m_labels
                )

            st.divider()
            st.header("⚙️ Alignment mode")
            seq_gap_open = st.slider("Sequence gap open penalty", -20, -1, st.session_state.seq_gap_open)
            seq_gap_extend = st.slider("Sequence gap extend penalty", -20.0, -0.1, st.session_state.seq_gap_extend, step=0.1)
            keep_fraction = st.slider("Keep Fraction (minimum aligned)", 0.1, 1.0, 1.0, step=0.01, 
                                      help="Minimum fraction of mapped residues to keep after outlier rejection.")
            recycles = st.number_input("Recycles (outlier rejection passes)", 0, 10, 1, step=1,
                                       help="How many times to remove high-distance outliers and re-align.")

            mode = st.selectbox("Choose alignment mode",
                                ["Auto (best RMSD)","Sequence-guided","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"],
                                index=["Auto (best RMSD)","Sequence-guided","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"].index(st.session_state.align_mode))
            
            run = st.form_submit_button("🚀 Run alignment", use_container_width=True)

        if run:
            st.session_state.selected_ref_file = ref_file
            st.session_state.selected_mobile_files = mob_files
            st.session_state.selected_ref_chains = [parse_lbl(x) for x in ref_sel]
            st.session_state.selected_mobile_chains_dict = {m: [parse_lbl(x) for x in sel] for m, sel in mob_sel_dict.items()}
            for m, sel in mob_sel_dict.items():
                st.session_state[f"selected_mob_chains_{m}"] = [parse_lbl(x) for x in sel]
            st.session_state.seq_gap_open = seq_gap_open
            st.session_state.seq_gap_extend = seq_gap_extend
            st.session_state.keep_fraction = keep_fraction
            st.session_state.recycles = recycles
            st.session_state.align_mode = mode
    else:
        st.info("Upload/Fetch at least two structures to proceed.")
        run = False

# ===========================================
# Selection logic
# ===========================================
# ===========================================
# Main run
# ===========================================
if run:
    ref_file = st.session_state.selected_ref_file
    mob_files = st.session_state.selected_mobile_files
    ref_chs = st.session_state.selected_ref_chains
    
    if not (ref_file and mob_files and ref_chs):
        st.error("Select reference file, at least one mobile file, and reference chain(s)."); st.stop()

    mob_chs_dict = st.session_state.get("selected_mobile_chains_dict", {})
    key = json.dumps(dict(ref=ref_file, mobs=mob_files, ref_chs=ref_chs,
                          mob_chs=mob_chs_dict,
                          mode=st.session_state.align_mode,
                          gap_open=st.session_state.seq_gap_open, gap_ext=st.session_state.seq_gap_extend,
                          keep_fraction=st.session_state.get("keep_fraction", 1.0),
                          recycles=st.session_state.get("recycles", 0)), sort_keys=True)
    
    if key in st.session_state.results_cache:
        log("Cache hit for identical batch parameters.")
        results_bundle_list = st.session_state.results_cache[key]
    else:
        log(f"Running batch mode: {st.session_state.align_mode}")
        ref_struct = st.session_state.structures[ref_file]

        results_bundle_list = []
        progress_bar = st.progress(0)
        
        for idx, mob_file in enumerate(mob_files):
            mob_struct = st.session_state.structures[mob_file]
            
            # User-selected mobile chains, fallback to all available
            mob_chs = st.session_state.get("selected_mobile_chains_dict", {}).get(mob_file, list(st.session_state.sequences.get(mob_file, {}).keys()))

            try:
                # Sequence-guided (if needed)
                seqguided=None
                if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-guided"):
                    try:
                        ref_infos = _extract_ca_infos(ref_struct, chain_filter=ref_chs)
                        mob_infos = _extract_ca_infos(mob_struct, chain_filter=mob_chs)
                        seqA = "".join(AA_DICT[ri.resname] for ri in ref_infos if ri.resname in AA_DICT)
                        seqB = "".join(AA_DICT[mi.resname] for mi in mob_infos if mi.resname in AA_DICT)
                    except Exception:
                        seqA = ""
                        seqB = ""
                    aln=perform_sequence_alignment(seqA, seqB, st.session_state.seq_gap_open, st.session_state.seq_gap_extend)
                if aln:
                    ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(ref_struct, ref_chs, mob_struct, mob_chs, aln)
                    if ref_atoms and mob_atoms:
                        kf = st.session_state.get("keep_fraction", 1.0)
                        recs = st.session_state.get("recycles", 0)
                        si = superimpose_atoms(ref_atoms, mob_atoms, recycles=recs, keep_fraction=kf)
                        if si:
                            # Re-map ref_atoms and mob_atoms to the ACTIVE matched set that passed iterative rejection
                            active_ref = si.pop("active_ref_atoms")
                            active_mob = si.pop("active_mob_atoms")
                            seqguided = dict(aln=aln, ref_atoms=active_ref, mob_atoms=active_mob, si=si)
                            log(f"Sequence-guided RMSD = {si['rmsd']:.3f} Å with {len(active_ref)} pairs.")
                else:
                    log("Sequence-guided alignment could not be built.")
            except Exception as e:
                log(f"Sequence-guided alignment failed: {e}")

            # Sequence-free (if needed)
            seqfree=None
            if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-free (auto)","Sequence-free (shape)","Sequence-free (window)"):
                pathA, pathB = write_uploads_to_temp_files(ref_file, mob_file)
                try:
                    method = "auto" if st.session_state.align_mode in ("Auto (best RMSD)","Sequence-free (auto)") else (
                             "shape" if st.session_state.align_mode=="Sequence-free (shape)" else "window")
                    kf = st.session_state.get("keep_fraction", 1.0)
                    recs = st.session_state.get("recycles", 0)
                    res = sequence_independent_alignment_joined_v2(
                        file_ref=pathA, file_mob=pathB,
                        chains_ref=ref_chs, chains_mob=mob_chs,
                        method=method, shape_nbins=48, shape_gap_penalty=2.0, shape_band_frac=0.30,
                        inlier_rmsd_cut=5.0, inlier_quantile=0.85, 
                        recycles=recs, keep_fraction=kf
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
                    st.warning(f"No alignment could be produced for {mob_file}. Skipping.")
                    continue
                
                chosen_name = best["name"]
                chosen = dict(name=chosen_name, reason=reason,
                              seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                              seqfree=seqfree if "Sequence-free" in chosen_name else None)
                log(f"AUTO chose: {chosen_name} for {mob_file}. Reason: {reason}")
            else:
                chosen = dict(name=st.session_state.align_mode, reason="Manual mode.",
                              seqguided=seqguided if st.session_state.align_mode=="Sequence-guided" else None,
                              seqfree=seqfree if "Sequence-free" in st.session_state.align_mode else None)

            results_bundle_list.append(dict(mob_file=mob_file, seqguided=seqguided, seqfree=seqfree, chosen=chosen))
            progress_bar.progress((idx + 1) / len(mob_files))

        st.session_state.results_cache[key] = results_bundle_list

    st.session_state.last_run_summary = results_bundle_list

# ===========================================
# Results display
# ===========================================
if st.session_state.last_run_summary:
    
    st.success(f"Batch Alignment complete for {len(st.session_state.last_run_summary)} targets.")

    # 1. Generate Summary Dataframe
    summary_data = []
    for r in st.session_state.last_run_summary:
        mob_file = r["mob_file"]
        chosen = r["chosen"]
        mode = chosen["name"]
        
        if "Sequence-guided" in mode and r.get("seqguided") is not None:
            rmsd = r["seqguided"]["si"]["rmsd"]
            pairs = len(r["seqguided"]["ref_atoms"])
        elif "Sequence-free" in mode and r.get("seqfree") is not None:
            rmsd = r["seqfree"].rmsd
            pairs = r["seqfree"].kept_pairs
        else:
            rmsd = None; pairs = 0
            
        summary_data.append({"Mobile Structure": mob_file, "Mode Used": mode, "RMSD (Å)": round(rmsd, 3) if rmsd else None, "Mapped Pairs": pairs})
    
    df_summary = pd.DataFrame(summary_data).sort_values(by="RMSD (Å)", ascending=True)
    st.subheader("Batch Results Summary")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # 1.5 NEW: Combined Data Line Plots if > 1 mobile
    if len(st.session_state.last_run_summary) > 1:
        with st.expander("📈 Combined Batch Data (RMSD vs Reference)", expanded=True):
            st.write("Compare per-position RMSD across all aligned structures mapped to the reference sequence.")
            
            # Extract all potential reference labels strictly in order to fix Plotly categorical x-axis
            ref_struct = st.session_state.structures[st.session_state.selected_ref_file]
            ref_chs = st.session_state.selected_ref_chains
            from pdb_align.core import _extract_ca_infos
            try:
                ref_full_infos = _extract_ca_infos(ref_struct, chain_filter=ref_chs)
                def make_lbl(ri):
                    ic = (ri.icode or '').strip()
                    return f"{ri.chain_id}:{ri.resseq}{ic}" if ic else f"{ri.chain_id}:{ri.resseq}"
                full_ref_labels = [make_lbl(ri) for ri in ref_full_infos]
            except Exception:
                full_ref_labels = []

            combined_data = []
            for r in st.session_state.last_run_summary:
                mob = r["mob_file"]
                chsn = r["chosen"]
                
                # Extract reference sequence labels and RMSD based on mode
                if "Sequence-guided" in chsn["name"] and r["seqguided"]:
                    sg = r["seqguided"]["si"]
                    mask = sg.get("mask", [True]*len(sg["residue_labels"]))
                    for idx, lbl in enumerate(sg["residue_labels"]):
                        combined_data.append({"Reference Label": lbl, "RMSD (Å)": sg["per_residue_rmsd"][idx], "Structure": mob})
                elif "Sequence-free" in chsn["name"] and r["seqfree"]:
                    sf = r["seqfree"]
                    mask = sf.active_mask
                    def make_lbl_sf(ri):
                        ic = (ri.icode or '').strip()
                        return f"{ri.chain_id}:{ri.resseq}{ic}" if ic else f"{ri.chain_id}:{ri.resseq}"
                    labels = [make_lbl_sf(sf.ref_subset_infos[i]) for (i, _) in sf.pairs]
                    rmsds = [np.linalg.norm(sf.ref_subset_ca_coords[i] - sf.mob_subset_ca_coords_aligned[j]) for (i, j) in sf.pairs]
                    for idx, lbl in enumerate(labels):
                        combined_data.append({"Reference Label": lbl, "RMSD (Å)": rmsds[idx], "Structure": mob})
            
            # To preserve categorical ordering and pad missing residues with NaNs:
            if combined_data:
                df_combined = pd.DataFrame(combined_data)
                if full_ref_labels:
                    # Create a complete combination lattice
                    structures = df_combined["Structure"].unique()
                    lattice = pd.DataFrame([(lbl, st) for lbl in full_ref_labels for st in structures], columns=["Reference Label", "Structure"])
                    # Merge left to embed NaNs at missing matching labels while preserving duplicate labels if they exist
                    # (Note: duplicated ref labels in full_ref_labels are rare but technically possible, handle gracefully)
                    df_combined = pd.merge(lattice, df_combined, on=["Reference Label", "Structure"], how="left")
                
                # Plotly line chart
                if full_ref_labels:
                    fig_combined = px.line(df_combined, x="Reference Label", y="RMSD (Å)", color="Structure",
                                           title="RMSD of Mapped Pairs by Reference Residue",
                                           category_orders={"Reference Label": full_ref_labels})
                else:
                    fig_combined = px.line(df_combined, x="Reference Label", y="RMSD (Å)", color="Structure",
                                           title="RMSD of Mapped Pairs by Reference Residue")
                
                fig_combined.update_traces(mode='lines+markers', marker=dict(size=2), line=dict(width=1.5), connectgaps=False)
                fig_combined.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", height=500)
                st.plotly_chart(fig_combined, use_container_width=True, key="fig_combined_batch_rmsd")
                
                # CSV Export (Clean NaNs for valid structural data only)
                df_export = df_combined.dropna(subset=["RMSD (Å)"])
                csv_combined = df_export.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Combined Data (CSV)", data=csv_combined, file_name="combined_batch_rmsd.csv", mime="text/csv", key="dl_combined_csv")
                
                # NEW: 3D structure showing average RMSD
                st.write("### Average RMSD on Reference 3D Structure")
                if full_ref_labels:
                    df_avg = df_combined.groupby("Reference Label")["RMSD (Å)"].mean().reset_index()
                    avg_rmsd_dict = dict(zip(df_avg["Reference Label"], df_avg["RMSD (Å)"]))
                    
                    ref_x_list, ref_y_list, ref_z_list = [], [], []
                    avg_rmsd_list, text_list = [], []
                    
                    for ri in ref_full_infos:
                        lbl = make_lbl(ri)
                        rmsd_val = avg_rmsd_dict.get(lbl, np.nan)
                        ref_x_list.append(ri.coord[0])
                        ref_y_list.append(ri.coord[1])
                        ref_z_list.append(ri.coord[2])
                        avg_rmsd_list.append(rmsd_val)
                        if pd.isna(rmsd_val):
                            text_list.append(f"{lbl} (No alignment)")
                        else:
                            text_list.append(f"{lbl}: {rmsd_val:.3f} Å")
                    
                    fig_3d = go.Figure()
                    # Backbone
                    fig_3d.add_trace(go.Scatter3d(
                        x=ref_x_list, y=ref_y_list, z=ref_z_list,
                        mode='lines',
                        line=dict(color='lightgray', width=2),
                        name='Reference Backbone',
                        hoverinfo='skip'
                    ))
                    
                    valid_mask = ~np.isnan(avg_rmsd_list)
                    if any(valid_mask):
                        x_valid = np.array(ref_x_list)[valid_mask]
                        y_valid = np.array(ref_y_list)[valid_mask]
                        z_valid = np.array(ref_z_list)[valid_mask]
                        rmsds_valid = np.array(avg_rmsd_list)[valid_mask]
                        texts_valid = np.array(text_list)[valid_mask]
                        
                        # Amplified difference between min and max radius
                        scaled_sizes = np.clip(1 + rmsds_valid * 5.0, 1, 35)
                        
                        fig_3d.add_trace(go.Scatter3d(
                            x=x_valid, y=y_valid, z=z_valid,
                            mode='markers',
                            marker=dict(
                                size=scaled_sizes,
                                color=rmsds_valid,
                                colorscale='OrRd',
                                colorbar=dict(title='Avg RMSD (Å)', x=0.85, thickness=20),
                                showscale=True,
                                opacity=0.9
                            ),
                            text=texts_valid,
                            hovertemplate="%{text}<extra></extra>",
                            name='Avg RMSD'
                        ))
                    
                    fig_3d.update_layout(
                        scene=dict(
                            aspectmode='data',
                            xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'
                        ),
                        height=600,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_3d, use_container_width=True, key="fig_3d_avg_rmsd")
            else:
                st.info("Not enough mapped data for a combined plot.")
    
    # NEW: Master ZIP Export
    if len(st.session_state.last_run_summary) > 0:
        zbuf_master, zname_master = export_zip_batch(st.session_state.last_run_summary, df_summary, st.session_state.selected_ref_file, st.session_state.selected_ref_chains, st.session_state.get("selected_mobile_chains_dict", {}))
        st.download_button("⬇️ Download Batch Bundle (ZIP)", data=zbuf_master, file_name=zname_master, mime="application/zip", type="primary")

    st.divider()
    # 2. Detailed Loop
    for target_result in st.session_state.last_run_summary:
        mob_file = target_result["mob_file"]
        seqguided = target_result["seqguided"]
        seqfree = target_result["seqfree"]
        chosen = target_result["chosen"]
        
        with st.expander(f"Detailed Alignment: {mob_file}", expanded=False):
            st.caption(f"Selection criterion: {chosen.get('reason','—')}")
        
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
            
            st.write("### Mol* Interactive Structure Alignment")
            import tempfile
            from streamlit_molstar import st_molstar
            
            rot = seqguided["si"]["rotation"] if "Sequence-guided" in chosen["name"] and seqguided else (seqfree.rotation if seqfree else None)
            trans = seqguided["si"]["translation"] if "Sequence-guided" in chosen["name"] and seqguided else (seqfree.translation if seqfree else None)
            if rot is not None and trans is not None:
                with st.spinner("Generating full structures for Mol*..."):
                    rf, rs, mf, ms = _generate_4_pdbs(
                        st.session_state.selected_ref_file, mob_file,
                        st.session_state.selected_ref_chains,
                        st.session_state.get("selected_mobile_chains_dict", {}).get(mob_file, []),
                        rot, trans
                    )
                    tdir = tempfile.mkdtemp()
                    merged_path = os.path.join(tdir, "merged_aligned.pdb")
                    
                    # Clean out any stray 'END' strings from individual saves before merging into models
                    rf_clean = rf.replace(b"END   \\n", b"").replace(b"END\\n", b"")
                    mf_clean = mf.replace(b"END   \\n", b"").replace(b"END\\n", b"")
                    
                    merged_pdb = b"MODEL        1\\n" + rf_clean + b"ENDMDL\\nMODEL        2\\n" + mf_clean + b"ENDMDL\\nEND   \\n"
                    
                    with open(merged_path, "wb") as f:
                        f.write(merged_pdb)
                    
                    st_molstar(merged_path, key=f"molstar_{mob_file}", height=600)
            else:
                st.warning("Cannot render Mol* because no valid transformation was found.")
                
            st.divider()
        
            # ===== A) Sequence-guided =====
            if seqguided:
                st.header("A. Sequence-guided alignment")
                txt_block = formatted_alignment_text(
                    st.session_state.selected_ref_file, seqguided["aln"].seqA,
                    mob_file, seqguided["aln"].seqB,
                    seqguided["aln"].score, interval=10
                )
                st.subheader("Show alignment text")
                st.code(txt_block)

                # NEW: toggle full chains vs matched subset
                sg_show_full = st.checkbox(
                    f"Show whole structure (all chains) for {mob_file}",
                    value=False, key=f"sg_show_full_{mob_file}"
                )
        
                id_pairs = [(i,i) for i in range(len(seqguided["si"]["residue_labels"]))]
        
                if sg_show_full:
                    # Extract full structures for reference and mobile
                    ref_full_infos = _extract_ca_infos(st.session_state.structures[st.session_state.selected_ref_file], chain_filter=None)
                    mob_full_infos = _extract_ca_infos(st.session_state.structures[mob_file], chain_filter=None)

                    ref_full_coords = np.vstack([ri.coord for ri in ref_full_infos])
                    mob_full_coords = np.vstack([mi.coord for mi in mob_full_infos])
        
                    # Align full mobile coords
                    mob_full_coords_aligned = _transform(mob_full_coords, seqguided["si"]["rotation"], seqguided["si"]["translation"])
        
                    # To highlight matching pairs within full structures, we need to map matched pairs to indices in the full structure arrays
                    ref_res_to_full_idx = {f"{ri.chain_id}{ri.resseq}{ri.icode.strip()}": i for i, ri in enumerate(ref_full_infos)}
                    mob_res_to_full_idx = {f"{mi.chain_id}{mi.resseq}{mi.icode.strip()}": j for j, mi in enumerate(mob_full_infos)}
        
                    full_pairs = []
                    # Labels for ALL residues in the full structure
                    all_labels = [f"{ri.chain_id}:{ri.resseq}{ri.icode.strip()}" for ri in ref_full_infos]
        
                    # `seqguided["ref_atoms"]` have ids like (' ', 42, ' ')
                    # map the parent residue back to `ref_full_infos`
                    for i, (ref_atom, mob_atom) in enumerate(zip(seqguided["ref_atoms"], seqguided["mob_atoms"])):
                        r_res = ref_atom.get_parent()
                        m_res = mob_atom.get_parent()
        
                        if hasattr(ref_atom, 'chain_name'):
                            r_chain = ref_atom.chain_name
                            r_key = f"{r_chain}{ref_atom.res_seq}{(ref_atom.res_icode or '').strip()}"
                        else:
                            # Biopython get_id() returns (het, resseq, icode)
                            # `res_labels` in `superimpose_atoms` gives us chain information via `get_parent().get_parent().id`
                            r_chain = r_res.get_parent().id
                            r_key = f"{r_chain}{r_res.get_id()[1]}{(r_res.get_id()[2] or '').strip()}"
        
                        if hasattr(mob_atom, 'chain_name'):
                            m_chain = mob_atom.chain_name
                            m_key = f"{m_chain}{mob_atom.res_seq}{(mob_atom.res_icode or '').strip()}"
                        else:
                            m_chain = m_res.get_parent().id
                            m_key = f"{m_chain}{m_res.get_id()[1]}{(m_res.get_id()[2] or '').strip()}"
        
                        if r_key in ref_res_to_full_idx and m_key in mob_res_to_full_idx:
                            idx_ref = ref_res_to_full_idx[r_key]
                            idx_mob = mob_res_to_full_idx[m_key]
                            full_pairs.append((idx_ref, idx_mob))
        
                    sg_key_prefix = f"sg3d_full_{_san_key(st.session_state.selected_ref_file)}_{_san_key(mob_file)}"
                    plot_superposition_3d(
                        ref_coords=ref_full_coords,
                        mob_coords=mob_full_coords_aligned,
                        title="Sequence-guided superposition (FULL structure)",
                        labels=all_labels,
                        pairs=full_pairs,
                        default_top_k=10,
                        key_prefix=sg_key_prefix
                    )
                else:
                    sg_key_prefix = f"sg3d_pairs_{_san_key(st.session_state.selected_ref_file)}_{_san_key(mob_file)}"
                    plot_superposition_3d(
                        ref_coords=seqguided["si"]["ref_coords"],
                        mob_coords=seqguided["si"]["mob_coords_transformed"],
                        title="Sequence-guided superposition (MATCHED subset)",
                        labels=seqguided["si"]["residue_labels"],
                        pairs=id_pairs,
                        default_top_k=10,
                        key_prefix=sg_key_prefix
                    )
        
                st.subheader("Per-Position Cα RMSD")
                colors = ['#1f77b4' if m else '#d62728' for m in seqguided["si"]["mask"]]
                rmsd_fig = go.Figure(data=[go.Bar(
                    x=seqguided["si"]["residue_labels"], y=seqguided["si"]["per_residue_rmsd"],
                    marker_color=colors
                )])
                rmsd_fig.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", yaxis_title="RMSD (Å)", height=400)
                st.plotly_chart(rmsd_fig, use_container_width=True, key=f"sg_rmsd_fig_{_san_key(mob_file)}")
        
                excluded = [lbl for lbl, m in zip(seqguided["si"]["residue_labels"], seqguided["si"]["mask"]) if not m]
                if excluded:
                    st.warning(f"Residues excluded from alignment calculation (outliers): {', '.join(excluded)}")
        
                # Prepare CSV for per position RMSD
                rmsd_df = pd.DataFrame({
                    "Residue Label": seqguided["si"]["residue_labels"],
                    "RMSD": seqguided["si"]["per_residue_rmsd"]
                })
                csv_data = rmsd_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download RMSD per position (CSV)", data=csv_data, file_name=f"rmsd_per_position_seqguided_{mob_file}.csv", mime="text/csv", key=f"dl_sg_rmsd_{mob_file}")
        
                mob_chs = st.session_state.get(f"selected_mob_chains_{mob_file}", [])
                zbuf, zname = export_zip_seqguided(
                    st.session_state.selected_ref_file, mob_file,
                    st.session_state.selected_ref_chains, mob_chs,
                    seqguided["si"], txt_block
                )
                st.download_button("⬇️ Download sequence-guided ZIP (4x PDBs, CSV, TXT)", data=zbuf, file_name=zname, mime="application/zip", key=f"dl_sg_zip_{mob_file}")
                st.divider()

            # ===== B) Sequence-independent =====
            if seqfree:
                st.header("B. Sequence-independent alignment")
                st.subheader("Diagnostics: distance and distributions")
                plot_distance_matrices(_pairwise_dists(seqfree.ref_subset_ca_coords),
                                       _pairwise_dists(seqfree.mob_subset_ca_coords_aligned),
                                       "Reference subset", "Mobile subset (aligned)")
                if len(seqfree.pairs) >= 3:
                    mob_sub_aln = seqfree.mob_subset_ca_coords_aligned
                    dists = np.array([np.linalg.norm(seqfree.ref_subset_ca_coords[i] - mob_sub_aln[j])
                                      for (i,j) in seqfree.pairs])
                    plot_pair_distance_hist(dists, "Per-pair distances after superposition", key_suffix=_san_key(mob_file))

                if len(seqfree.pairs) >= 1:
                    if seqfree.method == "shape" and seqfree.shift_matrix is not None:
                        if st.checkbox(f"Show shape alignment shift matrix for {mob_file}", key=f"show_shift_{mob_file}_shape"):
                            st.subheader("Alignment shift diagnostics (Shape Method Matrix)")
                            st.write("Heatmap of matching similarities between residues based on 3D neighborhood context. The alignment follows a path of highest similarity.")
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = ax.imshow(seqfree.shift_matrix, cmap='viridis', origin='lower')
                            ax.set_xlabel("Mobile Residue Index")
                            ax.set_ylabel("Reference Residue Index")
                            fig.colorbar(im, ax=ax, label="Similarity Score")
                            st.pyplot(fig)
                    elif seqfree.method == "window" and seqfree.shift_scores is not None:
                        if st.checkbox(f"Show window alignment scores for {mob_file}", key=f"show_shift_{mob_file}_window"):
                            st.subheader("Alignment shift diagnostics (Window Method Scores)")
                            st.write("Score plot of alignment across multiple sliding window offsets. The peak score indicates the best matching structural regions.")
                            fig = go.Figure(data=go.Scatter(x=np.arange(len(seqfree.shift_scores)), y=seqfree.shift_scores, mode='lines+markers'))
                            fig.update_layout(xaxis_title="Shift offset", yaxis_title="Correlation Score", height=400)
                            st.plotly_chart(fig, use_container_width=True, key=f"sf_window_scores_{_san_key(mob_file)}")

                    # NEW: toggle full chains vs matched subset
                    sf_show_mode = st.radio(
                        f"Superposition view mode for {mob_file}",
                        options=["Matched Subset", "Selected Chains", "Whole Structure (All Chains)"],
                        index=1,
                        key=f"sf_show_mode_{mob_file}",
                        horizontal=True
                    )

                    if sf_show_mode == "Whole Structure (All Chains)":
                        ref_full_infos = _extract_ca_infos(st.session_state.structures[st.session_state.selected_ref_file], chain_filter=None)
                        mob_full_infos = seqfree.mob_all_infos

                        ref_full_coords = np.vstack([ri.coord for ri in ref_full_infos])
                        mob_full_coords_aligned = seqfree.mob_all_ca_coords_aligned

                        ref_res_to_full_idx = {f"{ri.chain_id}{ri.resseq}{ri.icode.strip()}": i for i, ri in enumerate(ref_full_infos)}
                        mob_res_to_full_idx = {f"{mi.chain_id}{mi.resseq}{mi.icode.strip()}": j for j, mi in enumerate(mob_full_infos)}

                        full_pairs = []
                        for (i, j) in seqfree.pairs:
                            ri = seqfree.ref_subset_infos[i]
                            mi = seqfree.mob_subset_infos[j]
                            r_key = f"{ri.chain_id}{ri.resseq}{ri.icode.strip()}"
                            m_key = f"{mi.chain_id}{mi.resseq}{mi.icode.strip()}"
                            if r_key in ref_res_to_full_idx and m_key in mob_res_to_full_idx:
                                full_pairs.append((ref_res_to_full_idx[r_key], mob_res_to_full_idx[m_key]))

                        all_labels = [f"{ri.chain_id}:{ri.resseq}{ri.icode.strip()}" for ri in ref_full_infos]
                        sf_key_prefix = f"sf3d_whole_{_san_key(st.session_state.selected_ref_file)}_{_san_key(mob_file)}_{seqfree.method}"
                        plot_superposition_3d(
                            ref_coords=ref_full_coords,
                            mob_coords=mob_full_coords_aligned,
                            title=f"Seq-free superposition [{seqfree.method}] • WHOLE structure • RMSD {seqfree.rmsd:.3f} Å",
                            labels=all_labels,
                            pairs=full_pairs,
                            default_top_k=10,
                            key_prefix=sf_key_prefix
                        )
                    elif sf_show_mode == "Selected Chains":
                        labels = [f"{ri.chain_id}:{ri.resseq}{ri.icode.strip()}" for ri in seqfree.ref_subset_infos]
                        sf_key_prefix = f"sf3d_full_{_san_key(st.session_state.selected_ref_file)}_{_san_key(mob_file)}_{seqfree.method}"
                        plot_superposition_3d(
                            ref_coords=seqfree.ref_subset_ca_coords,
                            mob_coords=seqfree.mob_subset_ca_coords_aligned,
                            title=f"Seq-free superposition [{seqfree.method}] • SELECTED chains • RMSD {seqfree.rmsd:.3f} Å",
                            labels=labels,
                            pairs=seqfree.pairs,
                            default_top_k=10,
                            key_prefix=sf_key_prefix
                        )
                    else:
                        ref_pts = np.vstack([seqfree.ref_subset_ca_coords[i] for (i,j) in seqfree.pairs])
                        mob_pts = np.vstack([seqfree.mob_subset_ca_coords_aligned[j] for (i,j) in seqfree.pairs])
                        labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}{(seqfree.ref_subset_infos[i].icode or '').strip()}" for (i,_) in seqfree.pairs]
                        id_pairs = [(k,k) for k in range(len(labels))]
                        sf_key_prefix = f"sf3d_pairs_{_san_key(st.session_state.selected_ref_file)}_{_san_key(mob_file)}_{seqfree.method}"
                        plot_superposition_3d(
                            ref_coords=ref_pts,
                            mob_coords=mob_pts,
                            title=f"Seq-free superposition [{seqfree.method}] • MATCHED subset • RMSD {seqfree.rmsd:.3f} Å",
                            labels=labels,
                            pairs=id_pairs,
                            default_top_k=10,
                            key_prefix=sf_key_prefix
                        )

                        st.subheader("Per-Position Cα RMSD (Mapped Subset)")
                        dists = np.linalg.norm(ref_pts - mob_pts, axis=1)
                        colors = ['#1f77b4' if m else '#d62728' for m in seqfree.active_mask]
                        rmsd_fig = go.Figure(data=[go.Bar(x=labels, y=dists, marker_color=colors)])
                        rmsd_fig.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", yaxis_title="Distance (Å)", height=400)
                        st.plotly_chart(rmsd_fig, use_container_width=True, key=f"sf_mapped_rmsd_{_san_key(mob_file)}")

                        excluded = [lbl for i, lbl in enumerate(labels) if not seqfree.active_mask[i]]
                        if excluded:
                            st.warning(f"Residues excluded from alignment calculation (outliers): {', '.join(excluded)}")

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

                    st.subheader("Per-Position Cα RMSD (Matched Subset)")
                    # Labels and RMSD for seqfree
                    sf_rmsd_labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}{(seqfree.ref_subset_infos[i].icode or '').strip()}" for (i, _) in seqfree.pairs]
                    sf_per_res_rmsd = [np.linalg.norm(seqfree.ref_subset_ca_coords[i] - seqfree.mob_subset_ca_coords_aligned[j]) for (i, j) in seqfree.pairs]

                    sf_rmsd_fig = go.Figure(data=[go.Bar(x=sf_rmsd_labels, y=sf_per_res_rmsd)])
                    sf_rmsd_fig.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", yaxis_title="RMSD (Å)", height=400)
                    st.plotly_chart(sf_rmsd_fig, use_container_width=True, key=f"sf_matched_rmsd_{_san_key(mob_file)}")

                    sf_rmsd_df = pd.DataFrame({
                        "Residue Label": sf_rmsd_labels,
                        "RMSD": sf_per_res_rmsd
                    })
                    sf_csv_data = sf_rmsd_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download RMSD per position (CSV)", data=sf_csv_data, file_name=f"rmsd_per_position_seqfree_{mob_file}.csv", mime="text/csv", key=f"dl_sf1_{mob_file}")

                    # Structure-based sequence alignment (derived from matched pairs path)
                    a1,a2,_ = structure_based_alignment_strings(seqfree.ref_subset_infos, seqfree.mob_subset_infos, seqfree.pairs)
                    st.subheader("Structure-based sequence alignment")
                    idA = f"{st.session_state.selected_ref_file} ({','.join(st.session_state.selected_ref_chains)})"
                    idB = f"{mob_file} (auto)"
                    st.code(formatted_alignment_text(idA, a1, idB, a2, score=0.0, interval=10))

                    mob_chs = st.session_state.get(f"selected_mob_chains_{mob_file}", [])
                    zbuf2, zname2 = export_zip_seqfree(
                        st.session_state.selected_ref_file, mob_file,
                        st.session_state.selected_ref_chains, mob_chs,
                        seqfree, df_pairs
                    )
                    st.download_button("⬇️ Download seq-free ZIP (4x PDBs, CSV)",
                                       data=zbuf2, file_name=zname2, mime="application/zip", key=f"dl_sf2_{mob_file}")

    with st.expander("📜 Log", expanded=False):
        if not st.session_state.logs: st.write("No log messages.")
        else: st.text("\n".join(st.session_state.logs))
