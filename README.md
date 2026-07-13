# pdb_align
Align protein structures and explore local differences — pairwise, domain-flexible, or across full ensembles.

This package provides a Python library for structural bioinformatics scripting, a CLI, and an interactive Streamlit web app.

## Installation

Core library only:
```bash
pip install -e .
```

With Streamlit app and visualization extras (Py3Dmol, Plotly):
```bash
pip install -e .[app]
```

## Command-line quickstart

`pdb_align` is zero-config by default: give it two structures and it prints a stats report to the terminal — no files are written unless you ask for them.

```bash
pdb_align ref.pdb mob.pdb                 # stats only
pdb_align pdb:1ABC pdb:2XYZ --plot        # + per-residue RMSD plot (rmsd.png)
pdb_align a.cif b.cif -o aligned.cif --summary-plot   # + aligned structure + summary figure
```

`REF`/`MOB` accept local file paths or remote IDs (`pdb:XXXX`, `af:UniProtID`) directly. Other useful flags:

```bash
pdb_align ref.pdb mob.pdb --strategy global      # force global vs. local multi-chain superposition (default: auto)
pdb_align ref.pdb mob.pdb --ref-chains A --mob-chains A
pdb_align ref.pdb mob.pdb --json                 # machine-readable report to stdout
pdb_align ref.pdb mob.pdb --csv rmsd.csv --report report.txt --save result.npz
```

Multi-chain complexes (e.g. antibody-antigen, homomultimers) are matched automatically — chain correspondence is found via optimal 1:1 sequence-identity assignment, refined geometrically for near-identical chains — and `--strategy` picks between a global (all-chains) or local (best single chain pair) superposition.

Run `pdb_align --help` for the full flag list.

### Python: saving and reloading results

An `AlignmentResult` can be persisted and later reloaded without the original structure files:

```python
result.save("result.npz")
loaded = pdb_align.AlignmentResult.load("result.npz")   # gemmi-free LoadedResult
print(loaded.report())
loaded.plot_summary("summary.png")
```

`result.report(fmt="text"|"json")` gives a human-readable or machine-readable summary at any time; `result.plot_summary()` renders a compact Nature-style multi-panel figure (per-residue RMSD + per-chain/score panel).

### Python: quality verdict, 3D structure, and one-call export

Every `AlignmentResult` carries a plain-language quality assessment and can hand
you a ready-to-view structure or a complete output bundle:

```python
r = pdb_align.align("ref.pdb", "mob.pdb")

# Plain-language interpretation (band / verdict / confidence / flagged regions)
print(r.quality.verdict)          # e.g. "Same fold: 97% of residues within 1.4 A, TM=0.82"
print(r.quality.band, r.quality.confidence)
for region in r.quality.flagged_regions:
    print(region.chain, region.start_label, region.end_label, region.kind)

# In-memory transformed mobile structure, B-factors = per-residue RMSD (for 3D)
struct = r.aligned_structure(color_by="rmsd")   # gemmi.Structure

# One reproducible bundle: aligned coords, RMSD CSV, plots, PyMOL/ChimeraX, report
r.export_bundle("result.zip")                    # or fmt="dir" for a folder
```

The verdict also appears in `result.report()` and `result.to_json()["quality"]`,
so the CLI (`pdb_align ref.pdb mob.pdb`, or `--json`) shows it too. For ensembles,
`EnsembleResult.export_bundle("ensemble.zip")` writes the RMSD matrix, cluster
labels, PCA, and dendrogram.

## Streamlit App

Launch locally:
```bash
streamlit run struct_pair_align.py
```

The app is built entirely on the public `pdb_align` API (the same alignment the
CLI and library use — no separate code path). Inputs live in a compact sidebar;
results open with a plain-language **quality verdict header** (band, one-line
verdict, RMSD/TM-score/GDT-TS/coverage, confidence, warnings) followed by tabs:

- **Overview** — method/strategy, chain mapping, per-chain RMSD, flagged regions
- **3D** — interactive Py3Dmol superposition coloured by per-residue RMSD / pLDDT / chain
- **Per-residue** — RMSD plot with flagged regions shaded, plus sequence-alignment and distance-matrix expanders
- **Ensemble** (when you pick more than one mobile) — summary table, RMSD-matrix heatmap, clustering, PCA, dendrogram
- **Export** — one-click reproducible bundle (aligned structure, CSV, plots, PyMOL/ChimeraX, report)

**Inputs:**
- File uploads (PDB, mmCIF) and remote fetch (`pdb:XXXX`, `af:UniProtID`)
- Reference + one-or-more mobile selection with per-structure chain pickers
- Mode (Auto / sequence-guided / sequence-free / flexible) and multi-chain Strategy, with gap penalties and pLDDT filtering under **Advanced**

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
