# pdb_align
Align protein structures and explore local differences — pairwise, domain-flexible, or across full ensembles.

This package provides a Python library for structural bioinformatics scripting and an interactive Streamlit web app.

## Installation

Core library only:
```bash
pip install -e .
```

With Streamlit app and visualization extras (Py3Dmol, Plotly):
```bash
pip install -e .[app]
```

## Streamlit App

Launch locally:
```bash
streamlit run struct_pair_align.py
```

**Features:**
- File uploads for reference and mobile structures (PDB, mmCIF)
- Remote fetch from RCSB PDB (`pdb:XXXX`) and AlphaFold DB (`af:UniProtID`)
- Sequence-guided and shape/window alignment modes
- Interactive Py3Dmol 3D views colored by RMSD or B-factor
- Per-residue RMSD plots and downloadable outputs (CSV, FASTA, ZIP, PyMOL/ChimeraX scripts)

---

## Python Library

### One-liner alignment

```python
import pdb_align

result = pdb_align.align("pdb:8UUP", "af:P00533", chains_ref=["A"], chains_mob=["A"])
print(result.rmsd, result.tm_score)
```

### Full API via `PDBAligner`

```python
from pdb_align import PDBAligner

aligner = PDBAligner(verbose=True)

# Load structures — local files, PDB IDs, or AlphaFold IDs
aligner.add_reference("pdb:8UUP", chains=["A:10-150", "B"])
aligner.add_mobile("af:P00533", chains=["A"])

# Align (mode: "auto", "seq_guided", "seq_free_shape", "seq_free_window", "flexible")
result = aligner.align(mode="auto", atoms="CA")

print(f"RMSD:     {result.rmsd:.3f} Å")
print(f"TM-score: {result.tm_score:.3f}")
```

### Domain-flexible alignment

For structures with large conformational changes (e.g., multi-domain proteins, hinge motions):

```python
result = aligner.align(
    mode="flexible",
    hinge_threshold=3.0,   # local RMSD (Å) above which a position is a hinge
    hinge_window=15,        # sliding-window width for RMSD smoothing
    domain_min_residues=30, # minimum residues per domain
)

for domain in result.domains:
    print(f"Domain {domain.domain_id}: residues {domain.residue_start}–{domain.residue_end}, RMSD={domain.rmsd:.2f} Å")

print(f"Weighted RMSD: {result.rmsd:.3f} Å")  # residue-count-weighted average
```

### Ensemble alignment

Align 10–50 models against a common reference (e.g., cryo-EM heterogeneous refinement):

```python
aligner.add_reference("reference.pdb")

ens = aligner.align_ensemble(
    mob_list=["model_001.pdb", "model_002.pdb", ...],
    mode="auto",
    out_dir="aligned/",   # optional: save aligned PDBs
)

# Summary table (model, rmsd, tm_score, gdt_ts, n_aligned)
print(ens.summary())

# Conformational landscape via PCA
ens.cluster(n_clusters=3)
fig = ens.plot_pca(color_by="cluster")
fig.savefig("pca.png")

# Hierarchical clustering dendrogram
fig = ens.plot_dendrogram()
fig.savefig("dendrogram.png")

# Pairwise RMSD matrix
mat = ens.rmsd_matrix()
```

### Per-residue analysis and exports

```python
# Per-residue RMSD as DataFrame
df = result.get_rmsd_df(on="reference")
result.save_rmsd_csv("rmsd.csv")
result.plot_rmsd("rmsd_plot.pdf", style="scientific")

# Top deviation hotspots
peaks = result.report_peaks(on="reference", top_n=5)

# Sequence alignment from structure
result.print_sequence_alignment()
result.save_sequence_alignment_fasta("alignment.fasta")

# Save aligned coordinates
result.save_aligned_pdb("aligned_mobile.pdb")

# Visualization scripts
result.save_pymol_script("view.pml", aligned_mobile_filename="aligned_mobile.pdb")
result.save_chimerax_script("view.cxc", aligned_mobile_filename="aligned_mobile.pdb")
```

### Batch processing

```python
# Align all PDBs in a directory, returns a DataFrame
df_batch = aligner.batch_align(mob_dir="models/", out_dir="out/", mode="auto", workers=4)

# Or iterate for progress tracking
for fname, res in aligner.batch_align_iter(mob_dir="models/", out_dir="out/", workers=4):
    print(f"{fname}: RMSD={res.get('rmsd'):.3f}")

# Ensemble statistics across all models
stats = aligner.get_ensemble_statistics(df_batch)
print(stats)  # mean RMSD, median TM-score, etc.
```
