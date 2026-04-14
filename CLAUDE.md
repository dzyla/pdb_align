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
pytest tests/test_aligner.py::test_align_ensemble_returns_ensemble_result
```

### Run the Streamlit app
```bash
streamlit run struct_pair_align.py
```

## Architecture

The package has two layers: a **Python library** and a **Streamlit frontend**.

### Library (`pdb_align/`)

- **`__init__.py`** — Public API surface. Exports `PDBAligner`, `AlignmentResult`, `AlignmentFailedError`, `EnsembleResult`, `DomainResult`, `ParsingError`, `ChainNotFoundError`, and the top-level `align()` convenience function. The `align(ref, mob, chains_ref, chains_mob, **kwargs)` function is a zero-setup one-liner that wraps `PDBAligner`.

- **`core.py`** — The computational heart. Low-level functions:
  - `_kabsch()` — Kabsch SVD superposition; returns `(R, t, rmsd)`
  - `perform_sequence_alignment()` / `get_aligned_atoms_by_alignment()` — sequence-guided alignment path
  - `sequence_independent_alignment_joined_v2()` — sequence-free shape/window alignment
  - `pick_best_overall()` — selects best result across all strategies
  - `compute_gdt_ts()`, `compute_cad_score_approx()` — scoring functions
  - `progressive_align_ensemble()` — multi-structure ensemble alignment
  - `_sliding_window_mean()` — numba JIT-compiled sliding-window mean (used for hinge detection)
  - `_detect_hinges()` — returns split indices from a per-residue RMSD array; used by flexible mode
  - Uses **gemmi** for structure parsing; **BioPython** for sequence alignment; **numba** JIT on hot paths (graceful fallback if unavailable)

- **`aligner.py`** — High-level public API:
  - `DomainResult` — dataclass for a single rigid domain from flexible alignment (`domain_id`, `chain_id`, `residue_start`, `residue_end`, `n_residues`, `rmsd`, `rotation`, `translation`)
  - `AlignmentResult` — stateless result object. All properties computed lazily from `_chosen`/`_seqguided`/`_seqfree` dicts. Has `rmsd` (weighted-average when `domains` is set), `tm_score`, `domains` (List[DomainResult] or None), export methods, and plotting.
  - `EnsembleResult` — holds a list of `AlignmentResult` objects from an ensemble run. Methods: `summary()` → DataFrame, `rmsd_matrix()` → NxN DataFrame, `cluster(n_clusters)` → K-means labels, `plot_pca(color_by)` → matplotlib Figure, `plot_dendrogram()` → matplotlib Figure.
  - `PDBAligner` — orchestrates loading, aligning, and batch processing:
    - `add_reference()` / `add_mobile()` — parse and cache structures; `_struct_cache` (instance-scoped dict, keyed by `os.path.abspath`) avoids re-parsing the same file; always returns `.clone()` on cache hit
    - `align(mode, atoms, ...)` — modes: `"auto"`, `"seq_guided"`, `"seq_free_shape"`, `"seq_free_window"`, `"flexible"`. Flexible mode runs auto first, then detects hinges via `_detect_hinges`, runs per-domain Kabsch, returns domains in `result.domains`.
    - `align_ensemble(mob_list, mode, atoms, out_dir)` — iterates a list of mobile paths, returns `EnsembleResult`; emits `UserWarning` per failed model
    - `batch_align()` / `batch_align_iter()` — directory-level batch with `ProcessPoolExecutor`

- **`structure.py`** — `StructureBase` wraps `gemmi.Structure` with chain selection and subdomain range support (e.g., `"A:10-150"`).

- **`metrics.py`** — Standalone metric functions: `calculate_tm_score()`, `calculate_lddt()`.

- **`exceptions.py`** — Custom exceptions: `ParsingError`, `ChainNotFoundError`.

- **`__main__.py`** — CLI entry point.

### Streamlit App (`struct_pair_align.py`)

Single-file app that exposes the `PDBAligner` API via file uploads, interactive Py3Dmol 3D views (cartoon colored by B-factor/RMSD), per-residue RMSD plots, and downloadable outputs (CSV, FASTA, ZIP, PyMOL/ChimeraX scripts). All `st.plotly_chart` calls must use unique `key=` arguments. Multi-character chain IDs (e.g., from mmCIF files) are remapped to single characters before `PDBIO.save()` via `_remap_long_chain_ids()`.

### Key design patterns

- `AlignmentResult` is **stateless** — all properties are computed lazily from the raw alignment dicts stored at construction time.
- `mode="auto"` runs both sequence-guided and sequence-free paths, then calls `pick_best_overall()` to select the winner.
- `mode="flexible"` first runs `mode="auto"`, then detects hinges on per-residue CA RMSD, and re-runs `_kabsch()` independently per domain.
- The structure cache is **instance-scoped** (not shared across `PDBAligner` instances) and returns `.clone()` on every hit to prevent in-place mutation from corrupting cached structures.
- `numba` JIT decorates `_sliding_window_mean` in `core.py`; the fallback no-op decorator ensures the code runs without numba installed.
