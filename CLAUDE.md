# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
# Core library only
pip install -e .

# With Streamlit app and visualization extras (py3Dmol, plotly, etc.)
pip install -e .[app]
```

### Run tests
```bash
pytest tests/
# Single test:
pytest tests/test_core.py::test_compute_gdt_ts
```

### Run the Streamlit app
```bash
streamlit run struct_pair_align.py
```

## Architecture

The package has two layers: a **Python library** and a **Streamlit frontend**.

### Library (`pdb_align/`)

- **`core.py`** — The computational heart. Contains all alignment algorithms, data structures (`ResidueInfo`, `AlignSummary`, `AlignmentResultSF`), and low-level functions:
  - `_kabsch()` — Kabsch SVD superposition
  - `perform_sequence_alignment()` / `get_aligned_atoms_by_alignment()` — sequence-guided alignment path
  - `sequence_independent_alignment_joined_v2()` — sequence-free shape/window alignment
  - `pick_best_overall()` — selects the best alignment result from all strategies
  - `compute_gdt_ts()`, `compute_cad_score_approx()` — scoring functions
  - `progressive_align_ensemble()` — multi-structure ensemble alignment
  - Uses **gemmi** for structure parsing and **BioPython** for sequence alignment; **numba** JIT is used on hot paths (gracefully falls back if unavailable)

- **`aligner.py`** — High-level public API. `PDBAligner` orchestrates loading, aligning, and batch processing. `AlignmentResult` is the stateless result object with properties (`rmsd`, `tm_score`), export methods (`save_aligned_pdb`, `save_pymol_script`, `save_chimerax_script`, `save_rmsd_csv`), and plotting (`plot_rmsd`). Structures can be loaded from local files, `pdb:<ID>`, or `af:<UniProt>` (remote fetch).

- **`structure.py`** — `StructureBase` wraps `gemmi.Structure` with chain selection and subdomain range support (e.g., `"A:10-150"`).

- **`metrics.py`** — Standalone metric functions: `calculate_tm_score()`, `calculate_lddt()`.

- **`exceptions.py`** — Custom exceptions: `ParsingError`, `ChainNotFoundError`.

- **`__main__.py`** — CLI entry point.

### Streamlit App (`struct_pair_align.py`)

Single-file app that exposes the `PDBAligner` API via file uploads, interactive Py3Dmol 3D views (cartoon colored by B-factor/RMSD), per-residue RMSD plots, and downloadable outputs (CSV, FASTA, ZIP). All Streamlit `plotly_chart` calls must use unique `key=` arguments.

### Key design patterns

- `AlignmentResult` is **stateless** — all properties are computed lazily from the raw alignment dictionaries stored at construction time (`_chosen`, `_seqguided`, `_seqfree`).
- Alignment mode `"auto"` runs both sequence-guided and sequence-free paths, then calls `pick_best_overall()` to select the winner.
- `gemmi` is used for parsing; `BioPython` `Superimposer` / `PairwiseAligner` is used for the actual alignment math.
- `numba` JIT decorates performance-critical inner loops in `core.py`; the fallback no-op decorator ensures the code runs without numba installed.
