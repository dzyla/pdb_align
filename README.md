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

The package provides a robust interface via the `PDBAligner` class allowing structural alignment, data extractions, and batch processing.

```python
from pdb_align.aligner import PDBAligner

# 1. Initialize Aligner
aligner = PDBAligner(ref_file="1CRN.pdb", chains_ref=["A"])

# 2. Add Mobile structure
aligner.add_mobile("2CRN.pdb", chains=["A"])

# 3. Align (options for mode: "auto", "seq_guided", "seq_free_shape", "seq_free_window")
result = aligner.align(mode="auto")

# 4. Access data
rmsd = aligner.get_rmsd()
ref_coords, mob_coords = aligner.get_aligned_coords()
matched_pairs = aligner.get_matched_pairs()
print(f"RMSD: {rmsd}")

# 5. Export aligned PDB or Logs
aligner.save_aligned_pdb("aligned_mobile.pdb")
aligner.save_log("alignment_log.txt")

# 6. Batch alignment (align entire directory against the reference)
batch_results = aligner.batch_align(mob_dir="path/to/mobiles", out_dir="path/to/output", mode="auto")
print(batch_results)
```
