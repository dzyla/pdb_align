# pdb_align
Align two PDB structures and explore local differences

## Python Library Usage

You can use the alignment core logic programmatically as a python library.

### Installation

Install the package via `pip` locally:

```bash
pip install -e .
```

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
aligner.add_reference("8UUP.pdb", chains=["A", "B"])
aligner.add_mobile("9DRQ.pdb", chains=["A"])

# Need to update selected chains later? No problem:
aligner.set_reference_chains(["A"])
aligner.set_mobile_chains(["A"])

# 3. Alignment Execution
# ======================
# mode: "auto", "seq_guided", "seq_free_shape", "seq_free_window"
# Gap opening and extension parameters are supported as kwargs for sequence-dependent logic.
result = aligner.align(mode="auto", seq_gap_open=-10.0, seq_gap_extend=-0.5)

# 4. Accessing Data
# =================
rmsd = aligner.get_rmsd()                       # Overall RMSD
ref_coords, mob_coords = aligner.get_aligned_coords()
matched_pairs = aligner.get_matched_pairs()
rotation = aligner.get_rotation()               # Rotation matrix (numpy)
translation = aligner.get_translation()         # Translation vector (numpy)

# 5. Sequence Alignments
# ======================
# Structure-based Sequence Alignment (based on 3D proximity):
seqA, seqB, score = aligner.get_structure_based_sequence_alignment()
aligner.print_sequence_alignment() # Prints formatted structure-based alignment

# Export structural alignment directly to FASTA:
aligner.save_sequence_alignment_fasta("alignment.fasta")

# General/Classic Sequence Alignment (pure 1D string matching between specified chains):
aligner.print_general_sequence_alignment(ref_chain="A", mob_chain="A")

# 6. Exploring Local Structural Deviations
# ========================================
# Report the highest C-alpha RMSD peaks ("on" specifies numbering format: 'reference' or 'mobile')
peaks = aligner.report_peaks(on="reference", top_n=5)

# Retrieve per-residue RMSD as a pandas DataFrame:
df = aligner.get_rmsd_df(on="reference")
# Or directly save it to CSV:
aligner.save_rmsd_csv("rmsd.csv", on="reference")

# Plot per-residue structural deviation (lines separated by chain hue)
# Supports publication-ready 'scientific' styling natively.
aligner.plot_rmsd("rmsd_plot.pdf", style="scientific", on="reference")

# 7. Batch Processing & Exports
# =============================
aligner.save_aligned_pdb("aligned_mobile.pdb")
aligner.save_log("alignment_log.txt") # Save formatted summary of run

# Batch process a whole directory using the same initialized parameters
batch_results = aligner.batch_align(mob_dir="path/to/mobiles", out_dir="path/to/output", mode="auto")
```
