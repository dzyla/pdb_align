# pdb_align
Align two PDB structures and explore local differences.

This package provides a robust Python library for scripting high-throughput structural bioinformatics tasks, as well as an interactive Streamlit frontend application.

## Installation

Install the package via `pip` locally. For the core Python API only:

```bash
pip install -e .
```

To install with full App and Visualization extras (required for Streamlit and Py3Dmol plotting):

```bash
pip install -e .[app]
```

## Streamlit App

We provide an interactive web application built with Streamlit and Py3Dmol to visualize the alignment.

To launch the app locally:
```bash
streamlit run struct_pair_align.py
```

**Features of the Streamlit App:**
- File uploads for reference and mobile structures
- Interactive Py3Dmol rendering (cartoon views colored by B-factor/RMSD)
- Sequence-guided and Sequence-free (shape/window) alignment modes
- Downloadable alignment data (CSVs, Fasta, ZIPs)

## Python Library Usage

You can use the alignment core logic programmatically as a Python library.

### Usage

The package provides a robust interface via the `PDBAligner` class allowing structural alignment, data extractions, and batch processing. It can be initialized with `verbose=True` to print detailed runtime information, sequence similarity matrices, and alignment strategies.

## API Documentation

Below is a detailed guide ("wiki") covering the functionalities available within the `PDBAligner` API.

```python
from pdb_align.aligner import PDBAligner

# 1. Initialization
# =================
# Enable verbose logging to see step-by-step processing, matrix outputs, and reasoning.
aligner = PDBAligner(verbose=True)

# 2. Loading Structures and Chains
# ================================
# The API automatically handles remote fetching from the RCSB PDB or AlphaFold DB
aligner.add_reference("pdb:8UUP", chains=["A:10-150", "B"]) # Subdomain selection supported
aligner.add_mobile("af:P00533", chains=["A"])

# Need to update selected chains later? No problem:
aligner.set_reference_chains(["A"])
aligner.set_mobile_chains(["A"])

# 3. Alignment Execution
# ======================
# mode: "auto", "seq_guided", "seq_free_shape", "seq_free_window"
# atoms: "CA" (default), "backbone", or "all_heavy"
# Returns a state-free `AlignmentResult` object.
result = aligner.align(mode="auto", atoms="backbone", seq_gap_open=-10.0, seq_gap_extend=-0.5)

# 4. Accessing Data
# =================
print(f"RMSD: {result.rmsd}")                   # Properties evaluated dynamically
print(f"TM-Score: {result.tm_score}")

ref_coords, mob_coords = result.get_aligned_coords()
matched_pairs = result.get_matched_pairs()
rotation = result.rotation_matrix               # Rotation matrix (numpy)
translation = result.translation_vector         # Translation vector (numpy)

# 5. Sequence Alignments
# ======================
# Structure-based Sequence Alignment (based on 3D proximity):
seqA, seqB, score = result.get_structure_based_sequence_alignment()
result.print_sequence_alignment() # Prints formatted structure-based alignment

# Export structural alignment directly to FASTA:
result.save_sequence_alignment_fasta("alignment.fasta")

# General/Classic Sequence Alignment (pure 1D string matching between specified chains):
# Note: general alignment lives on the base aligner class since it is structure-independent
aligner.print_general_sequence_alignment(ref_chain="A", mob_chain="A")

# 6. Exploring Local Structural Deviations
# ========================================
# Report the highest C-alpha RMSD peaks ("on" specifies numbering format: 'reference' or 'mobile')
peaks = result.report_peaks(on="reference", top_n=5)

# Retrieve per-residue RMSD as a pandas DataFrame:
df = result.get_rmsd_df(on="reference")
# Or directly save it to CSV:
result.save_rmsd_csv("rmsd.csv", on="reference")

# Plot per-residue structural deviation (lines separated by chain hue)
# Supports publication-ready 'scientific' styling natively.
result.plot_rmsd("rmsd_plot.pdf", style="scientific", on="reference")

# 7. Batch Processing & Exports
# =============================
result.save_aligned_pdb("aligned_mobile.pdb")
result.save_log("alignment_log.txt") # Save formatted summary of run

# Generate a PyMOL or ChimeraX script that automatically loads both structures,
# applies the transformation, and visualizes the mobile chain colored by local RMSD.
result.save_pymol_script("visualization.pml", aligned_mobile_filename="aligned_mobile.pdb")
result.save_chimerax_script("visualization.cxc", aligned_mobile_filename="aligned_mobile.pdb")

# Batch process a whole directory across multiple cores natively
# Using the iterative generator allows tracking progress natively
for fname, res in aligner.batch_align_iter(mob_dir="path/", out_dir="out/", mode="auto", workers=4):
    print(f"Processed {fname} with TM-Score: {res.get('tm_score')}")

# Or get a pandas DataFrame immediately
df_batch = aligner.batch_align(mob_dir="path/", out_dir="out/", mode="auto", workers=4)

# And then calculate ensemble-wide metrics across all targets
stats = aligner.get_ensemble_statistics(df_batch)
print(stats) # Mean RMSD, Median TM-score, etc.
```
