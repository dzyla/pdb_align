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
  - `pick_best_overall()` — selects best result across all strategies using a **coverage-weighted score** (`n_pairs / (1 + (rmsd/3Å)²)`), so a strategy matching only a few residues at low RMSD cannot beat one that superimposes the whole protein
  - `compute_gdt_ts()`, `compute_cad_score_approx()` — scoring functions
  - `progressive_align_ensemble()` — multi-structure ensemble alignment
  - `_sliding_window_mean()` — numba JIT-compiled sliding-window mean (used for hinge detection)
  - `_detect_hinges()` — returns split indices from a per-residue RMSD array; used by flexible mode
  - Uses **gemmi** for structure parsing; **BioPython** for sequence alignment; **numba** JIT on hot paths (graceful fallback if unavailable)

- **`aligner.py`** — High-level public API:
  - `DomainResult` — dataclass for a single rigid domain from flexible alignment (`domain_id`, `chain_id`, `residue_start`, `residue_end`, `n_residues`, `rmsd`, `rotation`, `translation`)
  - `AlignmentResult` — stateless result object. All properties computed lazily from `_chosen`/`_seqguided`/`_seqfree` dicts. Has `rmsd` (weighted-average when `domains` is set), `tm_score`, `domains` (List[DomainResult] or None), export methods, and plotting. Now data-rich:
    - `.strategy` — `"single"` (one chain pair), `"global"`, or `"local"` (set by the multi-chain path; see `align_multichain()` below)
    - `.chain_mapping` — the `ChainMapping` used, or `None` for single-chain results
    - `.per_chain` — DataFrame (`chain_ref`, `chain_mob`, `n_residues`, `rmsd`) of per-chain-pair RMSD; empty for single-chain results
    - `summary_stats()` — dict of method/strategy/rmsd/tm_score/gdt_ts/coverage/chain_mapping
    - `report(fmt="text"|"json")` — human-readable or JSON report string (`to_dict()`/`to_json()` back `report(fmt="json")`)
    - `save(path)` — persists RMSD table, per-chain table, coords, and `summary_stats()` to a versioned `.npz` (`_SAVE_VERSION`); static `load(path)` reloads it as a `LoadedResult`, a gemmi-free object that can still `report()`, `plot_rmsd()`, and `plot_summary()` (methods reused via delegation) without the original structure files
    - `plot_summary(filename=None, show=False)` — compact two-panel Nature-style figure (per-residue RMSD by chain + per-chain RMSD bar, or a text summary when there's only one chain pair), via `plotstyle`
  - `LoadedResult` — returned by `AlignmentResult.load()`; wraps the saved meta/RMSD/per-chain data with no dependency on gemmi/the original files.
  - `EnsembleResult` — holds a list of `AlignmentResult` objects from an ensemble run. Methods: `summary()` → DataFrame, `rmsd_matrix()` → NxN DataFrame, `cluster(n_clusters)` → K-means labels, `plot_pca(color_by)` → matplotlib Figure, `plot_dendrogram()` → matplotlib Figure.
  - `PDBAligner` — orchestrates loading, aligning, and batch processing:
    - `add_reference()` / `add_mobile()` — parse and cache structures via `_load_cached_structure()`; `_struct_cache` (instance-scoped dict, keyed by `os.path.abspath`, storing `(structure, (mtime, size))`) avoids re-parsing an unchanged file and re-parses when the file changes on disk; always returns `.clone()` on cache hit. Remote IDs (`pdb:XXXX`, `af:UniProtID`) are downloaded to `self._fetch_cache_dir` (default `~/.cache/pdb_align`, override with `PDB_ALIGN_CACHE_DIR`), written atomically; AlphaFold fetches fall back across model versions (v6→v5→v4).
    - `align(mode, atoms, strategy="auto", ...)` — modes: `"auto"`, `"seq_guided"`, `"seq_free_shape"`, `"seq_free_window"`, `"flexible"`. Flexible mode runs auto first, then detects hinges via `_detect_hinges`, runs per-domain Kabsch, returns domains in `result.domains`. When `mode="auto"` and **both** structures have more than one active chain, `align()` auto-dispatches to the multi-chain path (`chains.match_chains()` + `chains.align_multichain()`) instead of the single-chain seq-guided/seq-free comparison; single-chain behavior is unchanged. `strategy` (`"auto"`/`"global"`/`"local"`) is forwarded to `align_multichain()`.
    - `align_ensemble(mob_list, mode, atoms, out_dir)` — iterates a list of mobile paths, returns `EnsembleResult`; emits `UserWarning` per failed model
    - `batch_align()` / `batch_align_iter()` — directory-level batch with `ProcessPoolExecutor`

- **`chains.py`** — Chain correspondence and multi-chain superposition strategy, used when both structures have multiple chains:
  - `ChainMapping` — dataclass: `pairs` (list of `(ref_chain, mob_chain, identity, score)`), `unmatched_ref`, `unmatched_mob`
  - `match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct, ref_chains, mob_chains)` — optimal 1:1 chain correspondence via Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the pairwise % sequence identity matrix (`compute_chain_similarity_matrix`). When multiple candidate chains are within `_TIE_TOL` (5%) identity of each other (the homomultimer case), refines the assignment geometrically: superposes on the tied mapping's chain centroids, then reassigns by post-superposition centroid proximity (iterative centroid ICP). Refinement is capped at `MAX_PERMUTE_CHAINS=12` chains (skipped above that, keeping the plain Hungarian result).
  - `align_multichain(ref_struct, mob_struct, mapping, strategy="auto", ...)` — builds a **global** superposition (Kabsch over all mapped chains' CA atoms) and a **local** superposition (Kabsch over just the single highest-identity chain pair), then picks the winner by the same coverage-weighted score used elsewhere (`n_pairs / (1 + (rmsd/3Å)²)`) unless `strategy` forces `"global"` or `"local"`. Returns a `MultiChainResult` (`strategy`, `mapping`, `rotation`, `translation`, `rmsd`, `per_chain`, coords).

- **`plotstyle.py`** — Nature-journal matplotlib style, shared by `plot_rmsd` and `plot_summary`:
  - `apply_nature_style()` — context manager applying Nature-style `rcParams` (Helvetica/Arial sans-serif, small fonts, no top/right spines, 300 dpi) via `plt.rc_context`, restored on exit
  - `nature_figure(width="single"|"double", height=None)` — returns `(fig, ax)` sized to a Nature column width (89mm/183mm) at 300 dpi
  - `panel_label(ax, letter)` — places a bold panel label (e.g. "a", "b") at the axis top-left, outside the frame
  - `PALETTE` — Okabe–Ito colorblind-safe categorical color list

- **`structure.py`** — `StructureBase` wraps `gemmi.Structure` with chain selection and subdomain range support (e.g., `"A:10-150"`).

- **`metrics.py`** — Standalone metric functions: `compute_d0()` (TM-score normalization distance, clamped to ≥0.5 Å), `calculate_tm_score()`, `calculate_lddt()`, and `calculate_tm_pvalue()` (approximate Gumbel-model significance; random-pair TM≈0.17 → p≈1).

- **`exceptions.py`** — Custom exceptions: `ParsingError`, `ChainNotFoundError`.

- **`__main__.py`** — CLI entry point, positional-first: `pdb_align REF MOB [options]`. By default it only prints a stats report to the terminal (`res.report(fmt="text")`, or `res.to_json()` with `--json`) — no files are written unless requested. Opt-in outputs: `-o/--out` (aligned structure via `save_aligned_pdb`), `--plot [FILE]` (per-residue RMSD plot, default `rmsd.png`), `--summary-plot [FILE]` (`plot_summary`, default `summary.png`), `--report FILE` (text or JSON, by extension), `--csv FILE` (per-residue RMSD table), `--save FILE.npz` (`AlignmentResult.save`), `--json` (machine-readable report to stdout), plus `--strategy {auto,global,local}`, `--mode`, `--ref-chains`/`--mob-chains`, `--atoms`, `--min-plddt`, `--show` (open plot windows instead of headless `Agg`), `-v/--verbose`. Legacy `--ref`/`--mob` flags remain accepted alongside the positionals. `REF`/`MOB` accept local file paths or remote IDs (`pdb:XXXX`, `af:UniProtID`) directly as positionals.

### Streamlit App (`struct_pair_align.py`)

Single-file app that exposes the `PDBAligner` API via file uploads, interactive Py3Dmol 3D views (cartoon colored by B-factor/RMSD), per-residue RMSD plots, and downloadable outputs (CSV, FASTA, ZIP, PyMOL/ChimeraX scripts). All `st.plotly_chart` calls must use unique `key=` arguments. Multi-character chain IDs (e.g., from mmCIF files) are remapped to single characters before `PDBIO.save()` via `_remap_long_chain_ids()`.

### Key design patterns

- `AlignmentResult` is **stateless** — all properties are computed lazily from the raw alignment dicts stored at construction time.
- `mode="auto"` runs both sequence-guided and sequence-free paths, then calls `pick_best_overall()` to select the winner.
- `mode="flexible"` first runs `mode="auto"`, then detects hinges on per-residue CA RMSD, and re-runs `_kabsch()` independently per domain.
- The structure cache is **instance-scoped** (not shared across `PDBAligner` instances) and returns `.clone()` on every hit to prevent in-place mutation from corrupting cached structures. It invalidates when the file's mtime/size changes.
- Both the internal shape-vs-window choice (`_select_seqfree_method`) and the overall seq-guided-vs-seq-free choice (`pick_best_overall`) rank candidates by the same coverage-weighted score, counting residues (CA atoms), not raw atoms. `chains.align_multichain()`'s global-vs-local choice uses the identical `n_pairs / (1 + (rmsd/3Å)²)` formula for consistency.
- `numba` JIT decorates `_sliding_window_mean` in `core.py`; the fallback no-op decorator ensures the code runs without numba installed.
- Multi-chain dispatch is automatic and narrow: `PDBAligner.align(mode="auto", ...)` only takes the `chains.py` path when *both* the reference and mobile selections have more than one active chain; anything with a single chain per side (including the common one-chain-per-structure case) goes through the original seq-guided/seq-free comparison unchanged.
- `AlignmentResult.save()`/`load()` round-trip is deliberately gemmi-free: `LoadedResult` only carries plain numpy/pandas/JSON data plus a `_SAVE_VERSION` check, so saved results can be replotted/reported later without the original structure files or even gemmi installed.
