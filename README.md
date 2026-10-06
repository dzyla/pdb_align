# pdb_align

Compare protein structures and score predicted models against experimental
references — pairwise, domain-flexible, across ensembles, or as a ranked table
of N models.

A Python library, a command-line tool, and a Streamlit web app. Metric
implementations are cross-validated against the reference programs (official
`DockQ`, TM-align via `tmtools`) in the test suite, and every reported number
states what it was normalized by. See **[docs/METHODS.md](docs/METHODS.md)**
for what each number means and where the implementation deviates from the
published method.

[![CI](https://github.com/dzyla/pdb_align/actions/workflows/ci.yml/badge.svg)](https://github.com/dzyla/pdb_align/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)

## What it computes

| | |
|---|---|
| **Fold similarity** | RMSD, TM-score (TM-optimal superposition, with p-value), GDT_TS, lDDT-Cα, coverage |
| **Correspondence** | sequence-guided, sequence-free (shape/window), automatic multi-chain matching, flexible multi-domain |
| **Interfaces** | DockQ + CAPRI class, fnat/fnonnat, iRMSD, LRMSD, epitope/paratope precision–recall–F1 |
| **Prediction confidence** | pDockQ, pDockQ2 (from PAE), ipTM ingestion for AlphaFold 2/3 and Boltz |
| **Antibodies** | IMGT CDR annotation and per-CDR backbone RMSD after framework superposition |
| **Interpretation** | plain-language verdict, confidence, flagged flexible regions, explicit warnings |

## Installation

```bash
pip install pdb_align                 # core library + CLI
pip install 'pdb_align[speed]'        # + numba JIT on the sequence-free kernels
pip install 'pdb_align[app]'          # + Streamlit web app
pip install 'pdb_align[antibody]'     # + ANARCI for CDR annotation
pip install 'pdb_align[dev]'          # everything, including the test suite
```

Requires Python 3.10+. Everything outside the core dependencies is optional:
the whole test suite passes on a core install, and `numba` changes no result,
only speed.

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
pdb_align ref.pdb mob.pdb --ref-chains "A:10-150 B"   # chains, or residue ranges within them
pdb_align ref.pdb mob.pdb --ref-chains A --mob-chains A
pdb_align ref.pdb mob.pdb --json                 # machine-readable report to stdout
pdb_align ref.pdb mob.pdb --csv rmsd.csv --report report.txt --save result.npz
```

Multi-chain complexes (e.g. antibody-antigen, homomultimers) are matched automatically — chain correspondence is found via optimal 1:1 sequence-identity assignment, refined geometrically for near-identical chains — and `--strategy` picks between a global (all-chains) or local (best single chain pair) superposition.

Run `pdb_align --help` for the full flag list.

## Model evaluation: rank predictions against a reference (DockQ, epitope metrics, pDockQ)

Given one experimental structure (or template) and N predicted models
(AlphaFold/Boltz/...), `--models` ranks them:

```bash
# fold metrics only: TM-score, RMSD, GDT_TS, lDDT-Ca, coverage
pdb_align native.pdb --models model1.cif model2.cif model3.cif

# + interface metrics (DockQ/fnat/iRMSD/LRMSD/CAPRI class + pDockQ)
pdb_align native.pdb --models m*.cif --receptor-chains A --ligand-chains B

# immune complexes: antibody H+L merged as receptor, plus epitope/paratope
# precision/recall/F1 (right-epitope-wrong-pose vs wrong-surface diagnostics)
# and per-CDR RMSD after framework superposition (cdr_h3 column; needs ANARCI)
pdb_align native.pdb --models m*.cif --antibody-chains H L --antigen-chains G

# AF2/AF3/Boltz confidence files (ipTM + PAE -> pDockQ2) are auto-discovered
# next to each model, or given explicitly ('-' skips a model):
pdb_align native.pdb --models m1.cif m2.cif --receptor-chains A --ligand-chains B \
    --confidence m1_scores.json -
```

Or from Python:

```python
from pdb_align import evaluate_models, compute_dockq, evaluate_antibody_complex, compute_pdockq

ev = evaluate_models("native.pdb", ["m1.cif", "m2.cif"],
                     antibody_chains=["H", "L"], antigen_chains=["G"])
print(ev.report()); print(ev.best)

dq = compute_dockq("native.pdb", "model.cif", ["H", "L"], ["G"])   # DockQResult
ab = evaluate_antibody_complex("native.pdb", "model.cif", ["H", "L"], ["G"])
pq = compute_pdockq("model.cif", ["H", "L"], ["G"])   # reference-free, needs pLDDT
```

### Metric definitions (and their sources)

Every reported number follows the published definition and is cross-validated
in the test suite against the reference implementation where one exists:

- **DockQ** (Basu & Wallner 2016, PLoS ONE; validated against the official
  `DockQ` package to <1e-3 on decoys): fnat = fraction of native interface
  contacts (any heavy-atom pair < 5 Å) recovered; iRMSD = backbone RMSD over
  the native 10 Å interface residues; LRMSD = ligand backbone RMSD after
  superposing the receptor; DockQ = (fnat + 1/(1+(iRMSD/1.5)²) +
  1/(1+(LRMSD/8.5)²))/3 with CAPRI classes (incorrect/acceptable/medium/high).
  Residue correspondence is established by per-chain sequence alignment, so
  mismatched numbering or chain naming cannot mis-pair residues; sequence-
  identical chains are assigned by an fnat-maximizing permutation search
  (symmetric homomultimers).
- **Epitope/paratope metrics**: precision/recall/F1/Jaccard of the model's
  4.5 Å contact residue sets against the native ones — separates "right
  epitope, mis-oriented pose" from "wrong antigen surface", which one DockQ
  number conflates.
- **pDockQ** (Bryant, Pozzati & Elofsson 2022, Nat Commun): reference-free
  interface confidence from interface pLDDT × log10(contacts); exact
  published sigmoid constants.
- **pDockQ2** (Zhu, Shenoy, Kundrotas & Elofsson 2023, Bioinformatics):
  PAE-aware per-interface confidence — X = ⟨1/(1+(PAE/10)²)⟩ · ⟨pLDDT⟩ over
  the interface, with the reference implementation's sigmoid constants.
  PAE/ipTM are ingested from AF2/ColabFold scores JSON, AF3
  `*_summary_confidences.json`/`*_confidences.json`, Boltz
  `confidence_*.json`/`pae_*.npz`, or bare PAE `.npy`/`.npz` (auto-discovered
  next to each model, `pdb_align.load_confidence` / `find_confidence_files`).
- **Per-CDR RMSD** (IMGT CDR definitions; ANARCI numbering, Dunbar & Deane
  2016): backbone RMSD of each CDR after superposing the model's framework
  onto the reference framework — CDR-H3 RMSD being the standard antibody-
  modelling headline number. Install with `pip install anarci` plus an HMMER
  3.3.x `hmmscan` (`conda install -c bioconda 'hmmer=3.3*'`; HMMER ≥ 3.4
  output is not parsed correctly by ANARCI). Any custom numbering callable
  can be injected instead (`cdr_rmsd(..., numberer=...)`).
- **TM-score** (Zhang & Skolnick 2004): reported as the maximum over rigid
  superpositions for the matched correspondence (as TM-align reports it),
  not the TM of the RMSD-optimal frame; validated against TM-align (tmtools).
- **TM-score p-value** (Xu & Zhang 2010): extreme-value distribution with the
  published parameters (μ=0.1512, σ=0.0242); P(TM≥0.5) = 5.5e-7 reproduces
  the paper's value.
- **lDDT-Cα** (Mariani et al. 2013): superposition-free, 15 Å inclusion
  radius, 0.5/1/2/4 Å thresholds, computed over matched residues (read with
  coverage).
- **GDT_TS**: single-superposition GDT normalized by the reference selection
  length (unaligned residues count as failures) — a lower bound on CASP's
  multi-superposition GDT, and never normalized by an inlier subset.
- **Contact overlap**: Jaccard index of Cα contact maps (this metric was
  previously mislabeled "CAD-score"; it is not CAD).

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


## Reproducibility

```python
res = pdb_align.align("ref.cif", "model.cif")

res.export_bundle("run1.zip")   # aligned structure + reference + per-residue CSV
                                # + both plots + PyMOL/ChimeraX scripts + report
res.save("run1.npz")            # versioned, gemmi-free; reload and re-plot later
pdb_align.AlignmentResult.load("run1.npz").report()

print(res.report(fmt="json"))   # every number, plus what it was normalized by
```

Parallel and serial execution give identical numbers, as does running with or
without `numba`; both are asserted in the test suite
(`tests/test_parallel.py`, `tests/test_optional_numba.py`).

## Running the tests

```bash
pip install -e '.[dev]'
pytest                      # 251 tests
pytest -k crossvalidation   # agreement with the official DockQ and TM-align
ruff check pdb_align tests
```

## Citing

If you use `pdb_align` in published work, please cite it (see
[CITATION.cff](CITATION.cff)) **and** the original papers for whichever metrics
you report — they are listed with DOIs in `CITATION.cff` and against each
metric in [docs/METHODS.md](docs/METHODS.md).

## Licence

AGPL-3.0-or-later. Modified versions stay under the same licence, and §13
extends that to network use: anyone who offers a modified version as a hosted
service must offer its users the source. See [COPYRIGHT](COPYRIGHT) for why
this licence was chosen.
