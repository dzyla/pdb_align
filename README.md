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

```python
from pdb_align.aligner import PDBAligner

# 1. Initialize Aligner with verbose logging
aligner = PDBAligner(verbose=True)

# 2. Add Reference and Mobile structures (specify subsets)
aligner.add_reference("1CRN.pdb", chains=["A"])
aligner.add_mobile("2CRN.pdb", chains=["A"])

# 3. Align (options for mode: "auto", "seq_guided", "seq_free_shape", "seq_free_window")
# Supports gap scoring kwargs: seq_gap_open, seq_gap_extend
result = aligner.align(mode="auto", seq_gap_open=-10.0, seq_gap_extend=-0.5)

# 4. Access standard alignment data
rmsd = aligner.get_rmsd()
ref_coords, mob_coords = aligner.get_aligned_coords()
matched_pairs = aligner.get_matched_pairs()
print(f"RMSD: {rmsd}")

# 5. Get Transformation matrices
rotation = aligner.get_rotation()
translation = aligner.get_translation()

# 6. Retrieve Logs and Sequence Alignments
text_log = aligner.get_log()
aligner.print_sequence_alignment() # Formatted pairwise alignment view
fasta_str = aligner.get_sequence_alignment_fasta()

# 7. Identify structural peaks (highest deviations)
# "on" allows specifying whether to output reference or mobile numbering
peaks = aligner.report_peaks(on="reference", top_n=5)

# 8. Generate publication-ready structural deviation plots
aligner.plot_rmsd("structural_deviation.pdf", style="scientific", on="reference")

# 9. Export aligned PDB or Logs
aligner.save_aligned_pdb("aligned_mobile.pdb")
aligner.save_log("alignment_log.txt")

# 10. Batch alignment (align entire directory against the reference)
batch_results = aligner.batch_align(mob_dir="path/to/mobiles", out_dir="path/to/output", mode="auto", seq_gap_open=-10)
print(batch_results)
```
