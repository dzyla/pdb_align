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
from pdb_align.core import (
    VALID_AA_3, AA_DICT, _aa_dict, ResidueInfo, AlignSummary, AlignmentResultSF, _AllAtomsSelect,
    extract_sequences_and_lengths, _parse_path, _chain_ids,
    _resolve_selectors, _extract_ca_infos, _pairwise_dists, _kabsch, _transform,
    _robust_inlier_mask, _window_pairs, _radial_histograms, _chi2_distance,
    _banded_dp_maxscore, _shape_pairs, sequence_independent_alignment_joined_v2,
    perform_sequence_alignment, get_aligned_atoms_by_alignment, superimpose_atoms,
    compute_chain_similarity_matrix, structure_based_alignment_strings, pick_best_overall
)

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
                          key_prefix: Optional[str] = None,
                          pdb_str: Optional[str] = None):

    if pdb_str:
        import py3Dmol
        from stmol import showmol
        st.subheader(title)
        st.markdown("*Note: Rendering uses Py3Dmol cartoon representations. Mobile chain colored by B-factor (RMSD gradient).*")
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_str, "pdb")
        # Color first model (reference) blue, second model (mobile) by b-factor
        view.setStyle({'model': 0}, {"cartoon": {'color': 'blue'}})
        view.setStyle({'model': 1}, {"cartoon": {'colorscheme': {'prop': 'b', 'gradient': 'rwb', 'min': 0, 'max': 10}}})
        view.zoomTo()
        showmol(view, height=600, width=800)
        return

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

    rmsd_df = pd.DataFrame({
        "Residue Label": results["residue_labels"],
        "RMSD": results["per_residue_rmsd"]
    })
    rmsd_csv_buf = io.StringIO(); rmsd_df.to_csv(rmsd_csv_buf, index=False)

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
        z.writestr("rmsd_per_position.csv", rmsd_csv_buf.getvalue())
        z.writestr("aligned_ref_CA_only.pdb", ref_pdb)
        z.writestr("aligned_mobile_CA_only.pdb", mob_pdb)
        z.writestr("alignment.txt", aln_text)
    zbuf.seek(0); name=f"seqguided_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return zbuf, name

def export_zip_seqfree(ref_file: str, mob_file: str, seqfree: AlignmentResultSF,
                       ref_subset_pairs_df: pd.DataFrame) -> Tuple[io.BytesIO, str]:
    R = seqfree.rotation
    t = seqfree.translation
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

    sf_rmsd_labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}{(seqfree.ref_subset_infos[i].icode or '').strip()}" for (i, _) in seqfree.pairs]
    sf_per_res_rmsd = [np.linalg.norm(seqfree.ref_subset_ca_coords[i] - seqfree.mob_subset_ca_coords_aligned[j]) for (i, j) in seqfree.pairs]
    sf_rmsd_df = pd.DataFrame({
        "Residue Label": sf_rmsd_labels,
        "RMSD": sf_per_res_rmsd
    })

    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,"w") as z:
        z.writestr("aligned_mobile_fullatom_on_reference.pdb", aligned_mobile_full)
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
        df_id, df_sc = compute_chain_similarity_matrix_ui(ref_file, mob_file)
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

        try:
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
        except Exception as e:
            st.error(f"Alignment failed: {str(e)}")
            log(f"Error during alignment: {str(e)}")
            st.stop()

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

        # NEW: toggle full chains vs matched subset
        sg_show_full = st.checkbox(
            "Show whole structure (all chains)",
            value=False, key="sg_show_full"
        )

        id_pairs = [(i,i) for i in range(len(seqguided["si"]["residue_labels"]))]

        if sg_show_full:
            # Extract full structures for reference and mobile
            ref_full_infos = _extract_ca_infos(st.session_state.structures[st.session_state.selected_ref_file], chain_filter=None)
            mob_full_infos = _extract_ca_infos(st.session_state.structures[st.session_state.selected_mobile_file], chain_filter=None)

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

                # Biopython get_id() returns (het, resseq, icode)
                # `res_labels` in `superimpose_atoms` gives us chain information via `get_parent().get_parent().id`
                r_chain = r_res.get_parent().id
                r_key = f"{r_chain}{r_res.get_id()[1]}{(r_res.get_id()[2] or '').strip()}"

                m_chain = m_res.get_parent().id
                m_key = f"{m_chain}{m_res.get_id()[1]}{(m_res.get_id()[2] or '').strip()}"

                if r_key in ref_res_to_full_idx and m_key in mob_res_to_full_idx:
                    idx_ref = ref_res_to_full_idx[r_key]
                    idx_mob = mob_res_to_full_idx[m_key]
                    full_pairs.append((idx_ref, idx_mob))

            sg_key_prefix = f"sg3d_full_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}"
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
            sg_key_prefix = f"sg3d_pairs_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}"
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
        rmsd_fig = go.Figure(data=[go.Bar(x=seqguided["si"]["residue_labels"], y=seqguided["si"]["per_residue_rmsd"])])
        rmsd_fig.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", yaxis_title="RMSD (Å)", height=400)
        st.plotly_chart(rmsd_fig, use_container_width=True)

        # Prepare CSV for per position RMSD
        rmsd_df = pd.DataFrame({
            "Residue Label": seqguided["si"]["residue_labels"],
            "RMSD": seqguided["si"]["per_residue_rmsd"]
        })
        csv_data = rmsd_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download RMSD per position (CSV)", data=csv_data, file_name="rmsd_per_position_seqguided.csv", mime="text/csv")

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
            if seqfree.method == "shape" and seqfree.shift_matrix is not None:
                with st.expander("Alignment shift diagnostics (Shape Method Matrix)", expanded=False):
                    st.write("Heatmap of matching similarities between residues based on 3D neighborhood context. The alignment follows a path of highest similarity.")
                    fig = go.Figure(data=go.Heatmap(z=seqfree.shift_matrix, colorscale='Viridis'))
                    fig.update_layout(xaxis_title="Mobile Residue Index", yaxis_title="Reference Residue Index", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            elif seqfree.method == "window" and seqfree.shift_scores is not None:
                with st.expander("Alignment shift diagnostics (Window Method Scores)", expanded=False):
                    st.write("Score plot of alignment across multiple sliding window offsets. The peak score indicates the best matching structural regions.")
                    fig = go.Figure(data=go.Scatter(x=np.arange(len(seqfree.shift_scores)), y=seqfree.shift_scores, mode='lines+markers'))
                    fig.update_layout(xaxis_title="Shift offset", yaxis_title="Correlation Score", height=400)
                    st.plotly_chart(fig, use_container_width=True)

            # NEW: toggle full chains vs matched subset
            sf_show_mode = st.radio(
                "Superposition view mode",
                options=["Matched Subset", "Selected Chains", "Whole Structure (All Chains)"],
                index=1,
                key="sf_show_mode",
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
                sf_key_prefix = f"sf3d_whole_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}_{seqfree.method}"
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
                sf_key_prefix = f"sf3d_full_{_san_key(st.session_state.selected_ref_file)}_{_san_key(st.session_state.selected_mobile_file)}_{seqfree.method}"
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

            st.subheader("Per-Position Cα RMSD (Matched Subset)")
            # Labels and RMSD for seqfree
            sf_rmsd_labels = [f"{seqfree.ref_subset_infos[i].chain_id}:{seqfree.ref_subset_infos[i].resseq}{(seqfree.ref_subset_infos[i].icode or '').strip()}" for (i, _) in seqfree.pairs]
            sf_per_res_rmsd = [np.linalg.norm(seqfree.ref_subset_ca_coords[i] - seqfree.mob_subset_ca_coords_aligned[j]) for (i, j) in seqfree.pairs]

            sf_rmsd_fig = go.Figure(data=[go.Bar(x=sf_rmsd_labels, y=sf_per_res_rmsd)])
            sf_rmsd_fig.update_layout(xaxis_title="Reference Residue (Chain:ResSeq)", yaxis_title="RMSD (Å)", height=400)
            st.plotly_chart(sf_rmsd_fig, use_container_width=True)

            sf_rmsd_df = pd.DataFrame({
                "Residue Label": sf_rmsd_labels,
                "RMSD": sf_per_res_rmsd
            })
            sf_csv_data = sf_rmsd_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download RMSD per position (CSV)", data=sf_csv_data, file_name="rmsd_per_position_seqfree.csv", mime="text/csv")

            # Structure-based sequence alignment (derived from matched pairs path)
            a1,a2,_ = structure_based_alignment_strings(seqfree.ref_subset_infos, seqfree.mob_subset_infos, seqfree.pairs)
            with st.expander("Show structure-based sequence alignment", expanded=False):
                idA = f"{st.session_state.selected_ref_file} ({','.join(st.session_state.selected_ref_chains)})"
                idB = f"{st.session_state.selected_mobile_file} ({','.join(st.session_state.selected_mobile_chains)})"
                st.code(formatted_alignment_text(idA, a1, idB, a2, score=0.0, interval=10))

            # Export full-atom transformed + pairs.csv
            zbuf2, zname2 = export_zip_seqfree(st.session_state.selected_ref_file,
                                               st.session_state.selected_mobile_file,
                                               seqfree,
                                               df_pairs)
            st.download_button("⬇️ Download seq-free ZIP (full-atom aligned PDB + pairs.csv)",
                               data=zbuf2, file_name=zname2, mime="application/zip")
            st.divider()

    with st.expander("📜 Log", expanded=False):
        if not st.session_state.logs: st.write("No log messages.")
        else: st.text("\n".join(st.session_state.logs))
