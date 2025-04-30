import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, PDBIO, Select
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2, SeqIO
from Bio.PDB.Polypeptide import protein_letters_3to1, standard_aa_names
import io
import os
import tempfile
from scipy.signal import argrelextrema
import py3Dmol
from stmol import showmol
import traceback
import datetime
import zipfile
from Bio.Align import substitution_matrices

# --- Constants and Helper Functions ---

VALID_AA_3 = set(standard_aa_names)

def get_aa_dict():
    """
    Create a mapping from three-letter amino acid codes to one-letter codes.
    """
    d = {k: protein_letters_3to1[k] for k in VALID_AA_3}
    d['MSE'] = 'M'
    return d

AA_DICT = get_aa_dict()

# --- Session State Initialization ---
def init_session_state():
    """
    Initialize session state variables.
    """
    defaults = {
        "uploaded_files": None,
        "structures": {},
        "sequences": {},
        "selected_ref_file": None,
        "selected_mobile_file": None,
        "selected_ref_chains": [],
        "selected_mobile_chains": [],
        "current_alignment": None,
        "alignment_confirmed": False,
        "ref_atoms": None,
        "mobile_atoms": None,
        "superimposition_results": None,
        "display_key": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Core Logic Functions ---

def parse_structure(uploaded_file):
    """
    Parses a PDB or mmCIF file using Biopython.
    """
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
        structure_id = os.path.splitext(uploaded_file.name)[0]
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=file_suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        if file_suffix == ".pdb":
            parser = PDBParser(QUIET=True)
        elif file_suffix == ".cif":
            parser = MMCIFParser(QUIET=True)
        else:
            return None, f"Unsupported file format: {file_suffix}"
        structure = parser.get_structure(structure_id, tmp_path)
        os.unlink(tmp_path)
        return structure, None
    except Exception as e:
        print(traceback.format_exc())
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, f"Error parsing {uploaded_file.name}: {e}"

def extract_sequences_from_structure(_structure, structure_id):
    """
    Extracts protein sequences from a Biopython structure object.
    """
    sequences = {}
    try:
        models = list(_structure.get_models())
        if not models:
            raise ValueError("Structure contains no models.")
        model = models[0]
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error accessing model in structure {structure_id}: {e}")
        return {}
    for chain in model:
        seq = ""
        valid_residues = 0
        for residue in chain:
            if residue.id[0] == ' ' and residue.get_resname() in AA_DICT:
                seq += AA_DICT[residue.get_resname()]
                valid_residues += 1
            elif residue.id[0].startswith("H_"):
                continue
            else:
                pass
        if valid_residues > 5:
            seq_record = SeqRecord(Seq(seq), id=f"{structure_id}_{chain.id}", description=f"Chain {chain.id}")
            sequences[chain.id] = seq_record
    return sequences

def get_combined_sequence(structure_id, chain_ids):
    """
    Combines sequences from selected chains for a structure.
    """
    combined_seq = ""
    if structure_id in st.session_state.sequences:
        for chain_id in chain_ids:
            if chain_id in st.session_state.sequences[structure_id]:
                combined_seq += str(st.session_state.sequences[structure_id][chain_id].seq)
            else:
                print(f"Warning: Chain {chain_id} not found in {structure_id}.")
                return None
    else:
        print(f"Error: Sequences for {structure_id} not found in session state.")
        return None
    return combined_seq if combined_seq else None

def plot_pairwise_alignment_heatmap(alignment, seq1_id, seq2_id):
    """
    Plots pairwise alignment as an interactive heatmap using Plotly.
    Abbreviates long IDs for the y-axis while preserving full names in hover text.
    """
    if not alignment:
        print("Warning: No alignment data to plot.")
        return
    seq1_aligned = str(alignment.seqA)
    seq2_aligned = str(alignment.seqB)
    alignment_length = len(seq1_aligned)
    AA_CODES = {
        "-": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
        "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14,
        "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21
    }
    msa_image = np.zeros((2, alignment_length), dtype=int)
    msa_letters = np.empty((2, alignment_length), dtype=object)
    for j, aa in enumerate(seq1_aligned):
        msa_image[0, j] = AA_CODES.get(aa.upper(), AA_CODES['-'])
        msa_letters[0, j] = aa.upper()
    for j, aa in enumerate(seq2_aligned):
        msa_image[1, j] = AA_CODES.get(aa.upper(), AA_CODES['-'])
        msa_letters[1, j] = aa.upper()
    hover_text = [
        [f"Position: {j+1}<br>Sequence: {seq1_id}<br>Amino Acid: {msa_letters[0, j]}" for j in range(alignment_length)],
        [f"Position: {j+1}<br>Sequence: {seq2_id}<br>Amino Acid: {msa_letters[1, j]}" for j in range(alignment_length)]
    ]
    max_label_len = 10
    short_seq1 = seq1_id if len(seq1_id) <= max_label_len else seq1_id[:max_label_len] + '…'
    short_seq2 = seq2_id if len(seq2_id) <= max_label_len else seq2_id[:max_label_len] + '…'
    fig = go.Figure(
        data=go.Heatmap(
            z=msa_image.tolist(),
            text=hover_text,
            hoverinfo="text",
            colorscale="Portland",
            showscale=False,
            xgap=1, ygap=1
        )
    )
    fig.update_traces(hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title="Sequence Alignment Heatmap",
        xaxis_title="Alignment Position",
        yaxis_title="Sequence",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=[short_seq1, short_seq2],
            showgrid=False,
            zeroline=False,
            autorange="reversed"
        ),
        xaxis=dict(showgrid=False, zeroline=False, side='top'),
        plot_bgcolor='white',
        height=200,
        margin=dict(l=100, r=20, t=50, b=20)
    )
    with st.container():
        st.plotly_chart(fig, use_container_width=False)

def generate_location_line(aligned_seq, interval):
    """
    Generate a location (number) line for an aligned sequence.
    """
    location = [' '] * len(aligned_seq)
    residue_count = 0
    next_mark = interval
    for i, char in enumerate(aligned_seq):
        if char != '-':
            residue_count += 1
            if residue_count == next_mark:
                mark_str = str(next_mark)
                start = max(0, i - len(mark_str) + 1)
                for j, digit in enumerate(mark_str):
                    if start + j < len(aligned_seq):
                        location[start + j] = digit
                next_mark += interval
    return ''.join(location)

def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10):
    """
    Create a formatted string for alignment display, including location markers.
    """
    pad_length = max(len(id1), len(id2))
    id1_padded = id1.ljust(pad_length)
    match_label = "Match".ljust(pad_length)
    id2_padded = id2.ljust(pad_length)
    loc1 = generate_location_line(aligned1, interval)
    loc2 = generate_location_line(aligned2, interval)
    padding = ' ' * (pad_length + 2)
    loc1_padded = padding + loc1
    loc2_padded = padding + loc2
    alignment_text = (
        "Pairwise Alignment:\n" +
        f"{loc1_padded}\n" +
        f"{id1_padded}: {aligned1}\n" +
        f"{match_label}: {match_line}\n" +
        f"{id2_padded}: {aligned2}\n" +
        f"{loc2_padded}\n" +
        f"{'Score'.ljust(pad_length)}: {score:.2f}\n"
    )
    return alignment_text

def display_alignment_text(alignment, seq1_id, seq2_id):
    """
    Formats and displays the pairwise alignment text in a scrollable code block.
    """
    if not alignment:
        print("Warning: No alignment to display.")
        return
    blosum62 = substitution_matrices.load("BLOSUM62")
    seq1_aligned = str(alignment.seqA)
    seq2_aligned = str(alignment.seqB)
    score = alignment.score
    match_line = ""
    for a, b in zip(seq1_aligned, seq2_aligned):
        if a == b and a != '-':
            match_line += "|"
        elif a != '-' and b != '-' and ((a, b) in blosum62 and blosum62[(a, b)] > 0 or (b, a) in blosum62 and blosum62[(b, a)] > 0):
            match_line += ":"
        elif a == '-' or b == '-':
            match_line += " "
        else:
            match_line += "."
    formatted_text = format_alignment_display(seq1_id, seq1_aligned, match_line, seq2_id, seq2_aligned, score, interval=10)
    st.code(formatted_text, language="")

def pdb_string_from_atoms(atoms, model_chain, override_coords=None):
    """
    Generates a PDB-formatted string for a list of Cα atoms.
    """
    lines = []
    lines.append("MODEL        1")
    serial = 1
    for i, atom in enumerate(atoms):
        residue = atom.get_parent()
        resname = residue.get_resname()
        chain = model_chain  
        resnum = residue.get_id()[1]
        if override_coords is not None:
            x, y, z = override_coords[i]
        else:
            x, y, z = atom.get_coord()
        line = f"ATOM  {serial:5d}  CA  {resname:>3} {chain}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        lines.append(line)
        serial += 1
    lines.append("ENDMDL")
    return "\n".join(lines)

def perform_sequence_alignment(seq1_str, seq2_str, _seq1_id, _seq2_id):
    """
    Performs pairwise sequence alignment.
    """
    gap_open = st.sidebar.slider("Gap Open Penalty", -20, -1, -10)
    gap_extend = st.sidebar.slider("Gap Extend Penalty", -20.0, -1.0, -0.5)
    if not seq1_str or not seq2_str:
        print("Error: One or both sequences are empty.")
        return None
    try:
        
        blosum62 = substitution_matrices.load("BLOSUM62")
        alignments = pairwise2.align.globalds(seq1_str, seq2_str, blosum62, gap_open, gap_extend, one_alignment_only=True)
        if not alignments:
            print("Warning: No alignment could be generated.")
            return None
        alignment_tuple = alignments[0]
        class AlignmentWrapper:
            def __init__(self, align_tuple, idA, idB):
                self.seqA = align_tuple[0]
                self.seqB = align_tuple[1]
                self.score = align_tuple[2]
                self.idA = idA
                self.idB = idB
        return AlignmentWrapper(alignment_tuple, _seq1_id, _seq2_id)
    except Exception as e:
        print(traceback.format_exc())
        print(f"Sequence alignment failed: {e}")
        return None

def get_aligned_atoms(_ref_struct, _ref_chains, _mobile_struct, _mobile_chains, alignment):
    """
    Extracts corresponding Cα atoms based on the sequence alignment.
    """
    ref_atoms = []
    mobile_atoms = []
    ref_res_indices_in_alignment = []
    if not alignment:
        print("Error: Cannot extract atoms without a valid alignment.")
        return [], [], []
    seqA_aligned = alignment.seqA
    seqB_aligned = alignment.seqB
    ref_id = alignment.idA
    mobile_id = alignment.idB
    def get_residues(structure, chain_ids):
        residues = []
        model = list(structure.get_models())[0]
        for chain_id in chain_ids:
            if chain_id in model:
                chain = model[chain_id]
                for residue in chain:
                    if residue.id[0] == ' ' and residue.get_resname() in AA_DICT and 'CA' in residue:
                        residues.append(residue)
            else:
                print(f"Warning: Chain {chain_id} not found in structure {structure.id}.")
        return residues
    ref_all_residues = get_residues(_ref_struct, _ref_chains)
    mobile_all_residues = get_residues(_mobile_struct, _mobile_chains)
    ref_res_idx = 0
    mobile_res_idx = 0
    for i, (res1_align, res2_align) in enumerate(zip(seqA_aligned, seqB_aligned)):
        if res1_align != '-':
            found_ref = False
            while ref_res_idx < len(ref_all_residues):
                current_ref_res = ref_all_residues[ref_res_idx]
                if AA_DICT.get(current_ref_res.get_resname()) == res1_align:
                    if 'CA' in current_ref_res:
                        ref_atom = current_ref_res['CA']
                        found_ref = True
                        break
                    else:
                        ref_res_idx += 1
                else:
                    ref_res_idx += 1
            if not found_ref:
                print(f"Warning: Alignment mismatch: Could not find residue '{res1_align}' in reference {ref_id} at position {i+1}.")
                if res2_align != '-':
                    mobile_res_idx += 1
                continue
            ref_res_idx += 1
        else:
            ref_atom = None
        if res2_align != '-':
            found_mobile = False
            while mobile_res_idx < len(mobile_all_residues):
                current_mobile_res = mobile_all_residues[mobile_res_idx]
                if AA_DICT.get(current_mobile_res.get_resname()) == res2_align:
                    if 'CA' in current_mobile_res:
                        mobile_atom = current_mobile_res['CA']
                        found_mobile = True
                        break
                    else:
                        mobile_res_idx += 1
                else:
                    mobile_res_idx += 1
            if not found_mobile:
                print(f"Warning: Alignment mismatch: Could not find residue '{res2_align}' in mobile {mobile_id} at position {i+1}.")
                if res1_align != '-':
                    ref_res_idx += 1
                continue
            mobile_res_idx += 1
        else:
            mobile_atom = None
        if ref_atom and mobile_atom:
            ref_atoms.append(ref_atom)
            mobile_atoms.append(mobile_atom)
            ref_res_indices_in_alignment.append(i)
    if not ref_atoms or not mobile_atoms:
        print("Error: Could not extract any corresponding Cα atoms based on the alignment. Check chain selections and alignment quality.")
        return [], [], []
    print(f"Successfully extracted {len(ref_atoms)} pairs of aligned Cα atoms.")
    return ref_atoms, mobile_atoms, ref_res_indices_in_alignment

def perform_superimposition(ref_atoms, mobile_atoms):
    """
    Superimposes mobile_atoms onto ref_atoms.
    """
    if not ref_atoms or not mobile_atoms or len(ref_atoms) != len(mobile_atoms):
        print("Error: Invalid atom lists for superimposition.")
        return None
    try:
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, mobile_atoms)
        rotation, translation = super_imposer.rotran
        mobile_coords = np.array([atom.get_coord() for atom in mobile_atoms])
        mobile_coords_transformed = np.dot(mobile_coords.copy(), rotation) + translation
        ref_coords = np.array([atom.get_coord() for atom in ref_atoms])
        rmsd_per_res = np.sqrt(np.sum((ref_coords - mobile_coords_transformed) ** 2, axis=1))
        residue_ids = [atom.get_parent().get_id() for atom in ref_atoms]
        residue_numbers = [res_id[1] for res_id in residue_ids]
        residue_labels = [f"{res_id[1]}{res_id[2]}" if res_id[2].strip() else str(res_id[1]) for res_id in residue_ids]
        results = {
            "rmsd": super_imposer.rms,
            "rotation": rotation,
            "translation": translation,
            "ref_coords": ref_coords,
            "mobile_coords_original": mobile_coords,
            "mobile_coords_transformed": mobile_coords_transformed,
            "per_residue_rmsd": rmsd_per_res,
            "residue_numbers": residue_numbers,
            "residue_labels": residue_labels
        }
        return results
    except Exception as e:
        print(traceback.format_exc())
        print(f"Structural superimposition failed: {e}")
        return None

def plot_calphas_alignment_plotly(ref_coords, mobile_coords, residue_labels, n_peaks=10, show_peaks=False):
    """
    Plots the aligned Cα atom positions in 3D for both structures using Plotly.
    
    Parameters:
        ref_coords (ndarray): Array of reference Cα coordinates (n x 3).
        mobile_coords (ndarray): Array of mobile Cα coordinates (n x 3) after transformation.
        residue_labels (list): List of residue labels corresponding to each pair.
        n_peaks (int): Number of RMSD peaks to annotate.
        show_peaks (bool): If True, annotate the peaks on the 3D plot and enable focusing.
        
    Extra functionality:
        If peaks are available and show_peaks is True, a selectbox is provided.
        Selecting a peak will update the camera to focus (with a 7 Å neighborhood) on that region.
        The default selection is "Show All".
    """
    import plotly.graph_objects as go

    # Extra option: checkbox to show connecting dashed lines between corresponding atoms.
    show_connections = st.checkbox("Show connecting dashed lines", value=False)
    
    # Extract coordinate arrays.
    ref_x, ref_y, ref_z = ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2]
    mobile_x, mobile_y, mobile_z = mobile_coords[:, 0], mobile_coords[:, 1], mobile_coords[:, 2]
    
    fig = go.Figure()
    
    # Option for RMSD-based coloring.
    color_by_rmsd = st.checkbox("Color 3D traces by RMSD", value=False)
    if color_by_rmsd:
        rmsd = np.sqrt(np.sum((ref_coords - mobile_coords)**2, axis=1))
        cmin, cmax = float(np.min(rmsd)), float(np.max(rmsd))
        ref_marker = dict(
            color=rmsd,
            colorscale='thermal',
            cmin=cmin,
            cmax=cmax,
            size=2,
            colorbar=dict(title='RMSD')
        )
        mobile_marker = dict(
            color=rmsd,
            colorscale='thermal',
            cmin=cmin,
            cmax=cmax,
            size=2,
            showscale=False
        )
        ref_hover = [f"Residue {lbl}<br>RMSD: {r:.2f}Å<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                     for lbl, r, x, y, z in zip(residue_labels, rmsd, ref_x, ref_y, ref_z)]
        mobile_hover = [f"Residue {lbl}<br>RMSD: {r:.2f}Å<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                        for lbl, r, x, y, z in zip(residue_labels, rmsd, mobile_x, mobile_y, mobile_z)]
        ref_line = dict(color='gray', width=2)
        mobile_line = dict(color='gray', width=2)
    else:
        ref_marker = dict(color='blue', size=2)
        mobile_marker = dict(color='red', size=2)
        ref_hover = [f"Residue {lbl}<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                     for lbl, x, y, z in zip(residue_labels, ref_x, ref_y, ref_z)]
        mobile_hover = [f"Residue {lbl}<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                        for lbl, x, y, z in zip(residue_labels, mobile_x, mobile_y, mobile_z)]
        ref_line = dict(color='blue', width=2)
        mobile_line = dict(color='red', width=2)
    
    # Add traces for reference and mobile coordinates.
    fig.add_trace(go.Scatter3d(
         x=ref_x, y=ref_y, z=ref_z,
         mode='markers+lines',
         name='Reference Cα',
         marker=ref_marker,
         line=ref_line,
         text=ref_hover,
         hoverinfo='text'
    ))
    fig.add_trace(go.Scatter3d(
         x=mobile_x, y=mobile_y, z=mobile_z,
         mode='markers+lines',
         name='Mobile Cα (Transformed)',
         marker=mobile_marker,
         line=mobile_line,
         text=mobile_hover,
         hoverinfo='text'
    ))
    
    # Add connecting dashed lines between corresponding atoms if selected.
    if show_connections:
        line_x, line_y, line_z = [], [], []
        for i in range(len(ref_coords)):
            line_x.extend([ref_x[i], mobile_x[i], None])
            line_y.extend([ref_y[i], mobile_y[i], None])
            line_z.extend([ref_z[i], mobile_z[i], None])
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines',
            name='Alignment Connections',
            line=dict(color='black', dash='dash', width=1),
            hoverinfo='skip'
        ))
    
    # Define a dictionary to store peak information.
    peak_dict = {}
    if show_peaks:
        rmsd = np.sqrt(np.sum((ref_coords - mobile_coords)**2, axis=1))
        local_max_indices = argrelextrema(rmsd, np.greater, order=3)[0]
        if len(local_max_indices) > 0:
            df_peaks = pd.DataFrame({'index': local_max_indices, 'rmsd': rmsd[local_max_indices]})
            df_peaks = df_peaks.nlargest(n_peaks, 'rmsd')
            peak_indices = df_peaks['index'].values
            peak_x = mobile_x[peak_indices]
            peak_y = mobile_y[peak_indices]
            peak_z = mobile_z[peak_indices]
            peak_labels = [f"Peak {i+1}: {r:.2f}Å" for i, r in enumerate(df_peaks['rmsd'].values)]
            for label, idx in zip(peak_labels, peak_indices):
                peak_dict[label] = idx
            fig.add_trace(go.Scatter3d(
                   x=peak_x, y=peak_y, z=peak_z,
                   mode='markers+text',
                   name='RMSD Peaks',
                   marker=dict(color='black', size=7, symbol='circle-open'),
                   text=peak_labels,
                   textposition='top center',
                   hoverinfo='text',
                   textfont=dict(size=14, color='black')
            ))
    
    # Provide a selectbox that lets the user choose a focus region.
    focus_option = "Show All"
    
    cc1, cc2 = st.columns(2)
    
    if show_peaks and peak_dict:
        peak_options = ["Show All"] + list(peak_dict.keys())
        focus_option = cc1.selectbox("Focus region:", options=peak_options, index=0)
    
    if focus_option != "Show All":
        # Get the index of the selected peak.
        focus_idx = peak_dict[focus_option]
        # Use the mobile coordinate at the peak as the focus point.
        focus_x, focus_y, focus_z = mobile_x[focus_idx], mobile_y[focus_idx], mobile_z[focus_idx]
        # Set the camera so that it centers on this point and shows a neighborhood of ~7 Å.
        new_camera = dict(
            center=dict(x=focus_x, y=focus_y, z=focus_z),
            eye=dict(x=focus_x + 7, y=focus_y + 7, z=focus_z + 7),
            up=dict(x=0, y=0, z=1)
        )
        fig.update_layout(scene_camera=new_camera)
    
    # Standard layout settings.
    fig.update_layout(
         title="3D Cα Atom Alignment",
         scene=dict(
             xaxis_title="X Coordinate",
             yaxis_title="Y Coordinate",
             zaxis_title="Z Coordinate"
         ),
         legend=dict(x=0.01, y=0.99),
         height=600,
         margin=dict(l=0, r=0, t=0, b=0),
         showlegend=True
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_calphas_alignment_plotly(ref_coords, mobile_coords, residue_labels, n_peaks=10, show_peaks=False):
    """
    Plots the aligned Cα atom positions in 3D for both structures using Plotly.
    
    Parameters:
        ref_coords (ndarray): Array of reference Cα coordinates (n x 3).
        mobile_coords (ndarray): Array of mobile Cα coordinates (n x 3) after transformation.
        residue_labels (list): List of residue labels corresponding to each pair.
        n_peaks (int): Number of RMSD peaks to annotate.
        show_peaks (bool): If True, annotate the peaks on the 3D plot and enable focusing.
        
    Extra functionality:
        A selectbox is provided with "Show All" (default) plus the detected peaks.
        If a peak is selected, the scene’s x, y, and z axis ranges are updated to show only the 14 Å cube 
        (i.e. 7 Å in each direction) around the mobile coordinate at the selected peak.
    """
    import plotly.graph_objects as go
    
    # Option to show connecting dashed lines between corresponding atoms.
    show_connections = st.checkbox("Show connecting dashed lines", value=False)
    
    # Extract coordinate arrays.
    ref_x, ref_y, ref_z = ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2]
    mobile_x, mobile_y, mobile_z = mobile_coords[:, 0], mobile_coords[:, 1], mobile_coords[:, 2]
    
    fig = go.Figure()
    
    # Option for RMSD-based coloring.
    color_by_rmsd = st.checkbox("Color 3D traces by RMSD", value=False)
    if color_by_rmsd:
        rmsd = np.sqrt(np.sum((ref_coords - mobile_coords)**2, axis=1))
        cmin, cmax = float(np.min(rmsd)), float(np.max(rmsd))
        ref_marker = dict(
            color=rmsd,
            colorscale='thermal',
            cmin=cmin,
            cmax=cmax,
            size=2,
            colorbar=dict(title='RMSD')
        )
        mobile_marker = dict(
            color=rmsd,
            colorscale='thermal',
            cmin=cmin,
            cmax=cmax,
            size=2,
            showscale=False
        )
        ref_hover = [f"Residue {lbl}<br>RMSD: {r:.2f}Å<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                     for lbl, r, x, y, z in zip(residue_labels, rmsd, ref_x, ref_y, ref_z)]
        mobile_hover = [f"Residue {lbl}<br>RMSD: {r:.2f}Å<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                        for lbl, r, x, y, z in zip(residue_labels, rmsd, mobile_x, mobile_y, mobile_z)]
        ref_line = dict(color='gray', width=2)
        mobile_line = dict(color='gray', width=2)
    else:
        ref_marker = dict(color='blue', size=2)
        mobile_marker = dict(color='red', size=2)
        ref_hover = [f"Residue {lbl}<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                     for lbl, x, y, z in zip(residue_labels, ref_x, ref_y, ref_z)]
        mobile_hover = [f"Residue {lbl}<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
                        for lbl, x, y, z in zip(residue_labels, mobile_x, mobile_y, mobile_z)]
        ref_line = dict(color='blue', width=2)
        mobile_line = dict(color='red', width=2)
    
    # Plot the main traces.
    fig.add_trace(go.Scatter3d(
         x=ref_x,
         y=ref_y,
         z=ref_z,
         mode='markers+lines',
         name='Reference Cα',
         marker=ref_marker,
         line=ref_line,
         text=ref_hover,
         hoverinfo='text'
    ))
    fig.add_trace(go.Scatter3d(
         x=mobile_x,
         y=mobile_y,
         z=mobile_z,
         mode='markers+lines',
         name='Mobile Cα (Transformed)',
         marker=mobile_marker,
         line=mobile_line,
         text=mobile_hover,
         hoverinfo='text'
    ))
    
    # Optionally draw connecting dashed lines.
    if show_connections:
        line_x, line_y, line_z = [], [], []
        for i in range(len(ref_coords)):
            line_x.extend([ref_x[i], mobile_x[i], None])
            line_y.extend([ref_y[i], mobile_y[i], None])
            line_z.extend([ref_z[i], mobile_z[i], None])
        fig.add_trace(go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            name='Alignment Connections',
            line=dict(color='black', dash='dash', width=1),
            hoverinfo='skip'
        ))
    
    # Peak detection.
    peak_dict = {}  # mapping: peak label -> index
    if show_peaks:
        rmsd = np.sqrt(np.sum((ref_coords - mobile_coords)**2, axis=1))
        local_max_indices = argrelextrema(rmsd, np.greater, order=3)[0]
        if len(local_max_indices) > 0:
            df_peaks = pd.DataFrame({'index': local_max_indices, 'rmsd': rmsd[local_max_indices]})
            df_peaks = df_peaks.nlargest(n_peaks, 'rmsd')
            peak_indices = df_peaks['index'].values
            peak_x = mobile_x[peak_indices]
            peak_y = mobile_y[peak_indices]
            peak_z = mobile_z[peak_indices]
            peak_labels = [f"Peak {i+1}: {r:.2f}Å" for i, r in enumerate(df_peaks['rmsd'].values)]
            for label, idx in zip(peak_labels, peak_indices):
                peak_dict[label] = idx
            fig.add_trace(go.Scatter3d(
                   x=peak_x,
                   y=peak_y,
                   z=peak_z,
                   mode='markers+text',
                   name='RMSD Peaks',
                   marker=dict(color='black', size=7, symbol='circle-open'),
                   text=peak_labels,
                   textposition='top center',
                   hoverinfo='text',
                   textfont=dict(size=14, color='black')
            ))
    
    # Provide a selectbox for focus region.
    focus_options = ["Show All"]
    if show_peaks and peak_dict:
        focus_options += list(peak_dict.keys())
    selected_focus = st.selectbox("Focus region:", options=focus_options, index=0)
    view_distance = st.slider("View distance (Å)", min_value=5, max_value=50, value=10)
    
    if selected_focus != "Show All":
        focus_idx = peak_dict[selected_focus]
        # Get the mobile coordinate for the selected peak.
        fx, fy, fz = mobile_x[focus_idx], mobile_y[focus_idx], mobile_z[focus_idx]
        # Instead of merely shifting the camera eye, update scene axis ranges to display a region of ±7 Å around the focus.
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[fx - view_distance, fx + view_distance], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[fy - view_distance, fy + view_distance], showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(range=[fz - view_distance, fz + view_distance], showgrid=False, zeroline=False, showticklabels=False)
            )
        )
    else:
        # Otherwise, use automatic axis ranges.
        fig.update_layout(
            scene=dict(
                xaxis=dict(autorange=True, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(autorange=True, showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(autorange=True, showgrid=False, zeroline=False, showticklabels=False)
            )
        )
    
    # Final layout settings.
    fig.update_layout(
         title="3D Cα Atom Alignment",
         scene=dict(
             xaxis_title="X Coordinate",
             yaxis_title="Y Coordinate",
             zaxis_title="Z Coordinate"
         ),
         legend=dict(x=0.01, y=0.99),
         height=800,
         margin=dict(l=0, r=0, t=0, b=0),
         showlegend=True
    )
    
    # hide axes and do not show their labels
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="")
        ))
    
    st.plotly_chart(fig, use_container_width=True)


def export_results(results, alignment_text):
    """
    Exports per-residue data (CSV), aligned structures (PDB), and alignment text as a ZIP file.
    """
    df = pd.DataFrame({
        'Residue Label': results['residue_labels'],
        'RMSD': results['per_residue_rmsd'],
        'Ref_X': results['ref_coords'][:,0],
        'Ref_Y': results['ref_coords'][:,1],
        'Ref_Z': results['ref_coords'][:,2],
        'Mobile_X': results['mobile_coords_transformed'][:,0],
        'Mobile_Y': results['mobile_coords_transformed'][:,1],
        'Mobile_Z': results['mobile_coords_transformed'][:,2]
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    ref_pdb = pdb_string_from_atoms(st.session_state.ref_atoms, "A")
    mobile_pdb = pdb_string_from_atoms(st.session_state.mobile_atoms, "B", override_coords=results['mobile_coords_transformed'])
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("per_residue_data.csv", csv_data)
        zip_file.writestr("aligned_ref.pdb", ref_pdb)
        zip_file.writestr("aligned_mobile.pdb", mobile_pdb)
        zip_file.writestr("alignment.txt", alignment_text)
    zip_buffer.seek(0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"export_{timestamp}.zip"
    return zip_buffer, filename


def plot_rmsd_plotly(residue_labels, rmsd_values, ref_name, mobile_name, n_peaks=10):
    """
    Generates an interactive RMSD plot using Plotly.
    
    Parameters:
        residue_labels (list): List of residue labels.
        rmsd_values (ndarray): Per-residue RMSD values.
        ref_name (str): Identifier for the reference structure.
        mobile_name (str): Identifier for the mobile structure.
        n_peaks (int): Number of top RMSD peaks to annotate.
    """
    if not residue_labels or rmsd_values is None or len(residue_labels) != len(rmsd_values):
        print("Warning: Insufficient data for RMSD plot.")
        return

    df = pd.DataFrame({
        'Residue Label': residue_labels,
        'Residue Number': [int(''.join(filter(str.isdigit, lbl))) for lbl in residue_labels],
        'RMSD': rmsd_values
    })
    df['Residue Index'] = range(len(df))
    overall_rmsd = np.mean(rmsd_values)
    mean_per_res_rmsd = np.mean(rmsd_values)
    local_max_indices = argrelextrema(df['RMSD'].values, np.greater, order=3)[0]
    df['Local Max'] = False
    df.loc[local_max_indices, 'Local Max'] = True
    top_n = n_peaks
    top_maxima_df = df[df['Local Max']].nlargest(top_n, 'RMSD')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Residue Index'],
        y=df['RMSD'],
        mode='lines',
        name='Per-Residue RMSD',
        line=dict(color='steelblue', width=2),
        customdata=df[['Residue Label', 'RMSD']],
        hovertemplate='Residue: %{customdata[0]}<br>RMSD: %{customdata[1]:.3f} Å<extra></extra>'
    ))
    if not top_maxima_df.empty:
        fig.add_trace(go.Scatter(
            x=top_maxima_df['Residue Index'],
            y=top_maxima_df['RMSD'],
            mode='markers+text',
            name=f'Top {top_n} Peaks',
            marker=dict(color='red', size=8, symbol='circle-open'),
            text=[f"Peak {i+1}" for i in range(len(top_maxima_df))],
            textposition="top right",
            textfont=dict(size=10, color='red'),
            customdata=top_maxima_df[['Residue Label', 'RMSD']],
            hovertemplate='<b>Top Peak</b><br>Residue: %{customdata[0]}<br>RMSD: %{customdata[1]:.3f} Å<extra></extra>'
        ))
    fig.add_hline(
        y=mean_per_res_rmsd,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Mean RMSD: {mean_per_res_rmsd:.3f} Å",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="grey"
    )
    fig.update_layout(
        title=f'Per-Residue Cα RMSD: {ref_name} (Ref) vs {mobile_name} (Aligned)<br>Overall RMSD: {overall_rmsd:.3f} Å',
        xaxis_title='Residue Position (Reference Structure)',
        yaxis_title='RMSD (Å)',
        hovermode='closest',
        xaxis=dict(
            tickmode='array',
            tickvals=df['Residue Index'][::max(1, len(df)//20)],
            ticktext=df['Residue Label'][::max(1, len(df)//20)]
        ),
        yaxis=dict(range=[0, max(df['RMSD'])*1.1]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit App Layout ---

st.set_page_config(page_title="Protein RMSD Analysis", layout="wide")
st.title("🔬 Protein Structure Alignment and RMSD Analysis")
init_session_state()

st.write("""
Upload two or more protein structures (PDB or mmCIF).
Select two structures and the chains you want to align.
The app will perform sequence alignment, structural superposition based on Cα atoms,
and plot the per-residue RMSD.
""")

with st.sidebar:
    st.header("📤 Upload Structures")
    uploaded_files = st.file_uploader("Select PDB or mmCIF files", type=["pdb", "cif"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        current_filenames = sorted([f.name for f in uploaded_files])
        previous_filenames = sorted(st.session_state.structures.keys()) if st.session_state.structures else []
        if current_filenames != previous_filenames:
            print("Info: New file upload detected. Parsing structures...")
            init_session_state()
            st.session_state.uploaded_files = uploaded_files
            new_structures = {}
            new_sequences = {}
            errors = []
            with st.spinner("Parsing uploaded files..."):
                for file in uploaded_files:
                    structure, error = parse_structure(file)
                    if structure:
                        new_structures[file.name] = structure
                        struct_seqs = extract_sequences_from_structure(structure, file.name)
                        if struct_seqs:
                            new_sequences[file.name] = struct_seqs
                        else:
                            print(f"Warning: Could not extract sequences from {file.name}")
                    elif error:
                        errors.append(error)
            st.session_state.structures = new_structures
            st.session_state.sequences = new_sequences
            if errors:
                for e in errors:
                    print(e)
            if st.session_state.structures:
                print(f"Successfully parsed {len(st.session_state.structures)} structure(s).")
            if not st.session_state.structures:
                print("Error: No valid structures could be parsed.")
        if len(st.session_state.structures) >= 2:
            st.header("⚙️ Select Structures & Chains")
            available_files = list(st.session_state.structures.keys())
            selected_ref_file = st.selectbox("Reference Structure:", options=available_files,
                                             index=available_files.index(st.session_state.selected_ref_file) if st.session_state.selected_ref_file in available_files else 0,
                                             key="select_ref")
            available_mobile_files = [f for f in available_files if f != selected_ref_file]
            if not available_mobile_files:
                print("Warning: Please upload at least two different structure files.")
            else:
                current_mobile_selection = st.session_state.selected_mobile_file
                default_mobile_index = 0
                if current_mobile_selection in available_mobile_files:
                    default_mobile_index = available_mobile_files.index(current_mobile_selection)
                selected_mobile_file = st.selectbox("Mobile Structure (to align):", options=available_mobile_files,
                                                      index=default_mobile_index, key="select_mobile")
                if selected_ref_file and selected_mobile_file:
                    selections_changed = (selected_ref_file != st.session_state.selected_ref_file or
                                          selected_mobile_file != st.session_state.selected_mobile_file)
                    if selections_changed:
                        st.session_state.selected_ref_chains = []
                        st.session_state.selected_mobile_chains = []
                        st.session_state.current_alignment = None
                        st.session_state.alignment_confirmed = False
                        st.session_state.ref_atoms = None
                        st.session_state.mobile_atoms = None
                        st.session_state.superimposition_results = None
                    st.session_state.selected_ref_file = selected_ref_file
                    st.session_state.selected_mobile_file = selected_mobile_file
                    ref_chains = list(st.session_state.sequences.get(selected_ref_file, {}).keys())
                    mobile_chains = list(st.session_state.sequences.get(selected_mobile_file, {}).keys())
                    if not ref_chains:
                        print(f"Warning: No usable chains found in {selected_ref_file}")
                    if not mobile_chains:
                        print(f"Warning: No usable chains found in {selected_mobile_file}")
                    if ref_chains and mobile_chains:
                        valid_prev_ref_chains = [c for c in st.session_state.selected_ref_chains if c in ref_chains]
                        valid_prev_mobile_chains = [c for c in st.session_state.selected_mobile_chains if c in mobile_chains]
                        selected_ref_chains = st.multiselect(f"Reference Chains ({selected_ref_file}):", options=ref_chains,
                                                             default=valid_prev_ref_chains if valid_prev_ref_chains else (ref_chains[0] if ref_chains else []),
                                                             key="select_ref_chains")
                        selected_mobile_chains = st.multiselect(f"Mobile Chains ({selected_mobile_file}):", options=mobile_chains,
                                                                default=valid_prev_mobile_chains if valid_prev_mobile_chains else (mobile_chains[0] if mobile_chains else []),
                                                                key="select_mobile_chains")
                        chains_changed = (sorted(selected_ref_chains) != sorted(st.session_state.selected_ref_chains) or
                                          sorted(selected_mobile_chains) != sorted(st.session_state.selected_mobile_chains))
                        if chains_changed:
                            st.session_state.current_alignment = None
                            st.session_state.alignment_confirmed = False
                            st.session_state.ref_atoms = None
                            st.session_state.mobile_atoms = None
                            st.session_state.superimposition_results = None
                        st.session_state.selected_ref_chains = selected_ref_chains
                        st.session_state.selected_mobile_chains = selected_mobile_chains
        elif st.session_state.uploaded_files:
            print("Warning: Please upload at least two structure files.")
    else:
        st.info("Upload PDB or mmCIF files to begin.")

if (st.session_state.selected_ref_file and st.session_state.selected_mobile_file and
    st.session_state.selected_ref_chains and st.session_state.selected_mobile_chains and
    not st.session_state.alignment_confirmed):
    
    st.header("1. Sequence Alignment")
    ref_seq_str = get_combined_sequence(st.session_state.selected_ref_file, st.session_state.selected_ref_chains)
    mobile_seq_str = get_combined_sequence(st.session_state.selected_mobile_file, st.session_state.selected_mobile_chains)
    if ref_seq_str and mobile_seq_str:
        if st.session_state.current_alignment is None:
            st.session_state.current_alignment = perform_sequence_alignment(
                ref_seq_str,
                mobile_seq_str,
                st.session_state.selected_ref_file,
                st.session_state.selected_mobile_file
            )
        if st.session_state.current_alignment:
            alignment = st.session_state.current_alignment
            st.subheader(f"Alignment: {alignment.idA} vs {alignment.idB}")
            col1, col2 = st.columns([1, 1])
            plot_pairwise_alignment_heatmap(alignment, alignment.idA, alignment.idB)
            st.write("**Alignment Text:**")
            display_alignment_text(alignment, alignment.idA, alignment.idB)
            if st.button("✅ Confirm Alignment & Proceed to Structural Analysis"):
                st.session_state.alignment_confirmed = True
                st.session_state.ref_atoms = None
                st.session_state.mobile_atoms = None
                st.session_state.superimposition_results = None
                st.rerun()
        else:
            print("Warning: Could not generate sequence alignment. Check sequences and chain selections.")
    else:
        print("Warning: Select valid chains with sequences for both structures.")

if st.session_state.alignment_confirmed:
    st.header("2. Structural Superposition & RMSD Analysis")
    ref_struct = st.session_state.structures.get(st.session_state.selected_ref_file)
    mobile_struct = st.session_state.structures.get(st.session_state.selected_mobile_file)
    alignment = st.session_state.current_alignment
    if not ref_struct or not mobile_struct or not alignment:
        print("Error: Missing data required for structural analysis. Please restart the process.")
    else:
        if st.session_state.ref_atoms is None or st.session_state.mobile_atoms is None:
            st.session_state.ref_atoms, st.session_state.mobile_atoms, _ = get_aligned_atoms(
                ref_struct,
                st.session_state.selected_ref_chains,
                mobile_struct,
                st.session_state.selected_mobile_chains,
                alignment
            )
        if st.session_state.ref_atoms and st.session_state.mobile_atoms:
            if st.session_state.superimposition_results is None:
                st.session_state.superimposition_results = perform_superimposition(
                    st.session_state.ref_atoms,
                    st.session_state.mobile_atoms
                )
            results = st.session_state.superimposition_results
            if results:
                st.success(f"Superimposition complete! Overall RMSD: {results['rmsd']:.3f} Å")
                n_peaks = st.number_input("Number of RMSD Peaks to Annotate", value=10, min_value=1, step=1)
                show_peaks = st.checkbox("Show RMSD Peaks on 3D Plot", value=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("3D Cα Alignment")
                    try:
                        plot_calphas_alignment_plotly(
                            results['ref_coords'],
                            results['mobile_coords_transformed'],
                            results['residue_labels'],
                            n_peaks=n_peaks,
                            show_peaks=show_peaks
                        )
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f"Failed to render 3D view: {e}")
                        print("Ensure py3Dmol and stmol are installed correctly.")
                with col2:
                    st.subheader("Per-Residue RMSD Plot")
                    plot_rmsd_plotly(
                        results['residue_labels'],
                        results['per_residue_rmsd'],
                        st.session_state.selected_ref_file,
                        st.session_state.selected_mobile_file,
                        n_peaks=n_peaks
                    )
                st.subheader("Export Results")
                alignment_text = format_alignment_display(
                    alignment.idA, str(alignment.seqA), "", alignment.idB, str(alignment.seqB), alignment.score, interval=10
                )
                zip_buffer, zip_filename = export_results(results, alignment_text)
                st.download_button("Download Exported Data", data=zip_buffer, file_name=zip_filename, mime="application/zip")
                if st.button("↩️ Change Selections or Alignment"):
                    st.session_state.alignment_confirmed = False
                    st.session_state.current_alignment = None
                    st.session_state.ref_atoms = None
                    st.session_state.mobile_atoms = None
                    st.session_state.superimposition_results = None
                    st.rerun()
            else:
                print("Error: Structural superimposition failed.")
        else:
            print("Error: Failed to extract aligned atoms based on the sequence alignment.")
            if st.button("↩️ Change Selections or Alignment"):
                st.session_state.alignment_confirmed = False
                st.session_state.current_alignment = None
                st.rerun()

st.markdown("---")
