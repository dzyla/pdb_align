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
# You can also supply residue ranges to restrict the alignment domain
aligner.add_reference("8UUP.pdb", chains=["A:10-150", "B"])
aligner.add_mobile("9DRQ.pdb", chains=["A"])

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

# Batch process a whole directory across multiple cores natively
batch_results = aligner.batch_align(
    mob_dir="path/to/mobiles",
    out_dir="path/to/output",
    mode="auto",
    workers=4 # Multi-processing
)
```
