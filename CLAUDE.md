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
  - `compute_gdt_ts(dists, n_total)` — single-superposition GDT_TS over ALL matched pairs, normalized by the reference selection length (`n_total`; CASP semantics — never by an inlier subset). `compute_contact_overlap()` — Cα contact-map Jaccard (formerly mislabeled `compute_cad_score_approx`, kept as a deprecated alias)
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
    - `quality` — lazy property returning an `AlignmentQuality` (from `interpretation.py`): plain-language `band`/`verdict`/`confidence`, `flagged_regions` (contiguous high-RMSD or hinge runs), and `warnings`. Pure function of numbers already on the result (tm_score, rmsd, coverage, per-residue RMSD, both candidate RMSDs, domains). Surfaced in `report()`, `to_dict()["quality"]`, and the CLI verdict line
    - `aligned_structure(color_by="rmsd")` — in-memory transformed mobile `gemmi.Structure` for 3D viewing/export. `color_by="rmsd"` writes per-residue deviation into B-factors; `"bfactor"`/`"plddt"` preserve input B-factors. Shares the private `_build_aligned_structure()` helper with `save_aligned_pdb()`, so file output and 3D view are identical
    - `export_bundle(path, include=None, fmt="zip")` — one reproducible bundle (`.zip` or `fmt="dir"` folder): aligned structure, per-residue RMSD CSV, summary/RMSD plots, PyMOL `.pml` + ChimeraX `.cxc` scripts, and text/JSON report (both carrying the quality verdict). Replaces the Streamlit app's `export_zip_*` helpers
  - `LoadedResult` — returned by `AlignmentResult.load()`; wraps the saved meta/RMSD/per-chain data with no dependency on gemmi/the original files.
  - `EnsembleResult` — holds a list of `AlignmentResult` objects from an ensemble run. Methods: `summary()` → DataFrame, `rmsd_matrix()` → NxN DataFrame, `cluster(n_clusters)` → K-means labels, `plot_pca(color_by)` → matplotlib Figure, `plot_dendrogram()` → matplotlib Figure, `export_bundle(path, fmt="zip")` → bundle of `summary.csv`, `rmsd_matrix.csv`, `clusters.csv`, `pca.png`, `dendrogram.png`.
  - `PDBAligner` — orchestrates loading, aligning, and batch processing:
    - `add_reference()` / `add_mobile()` — parse and cache structures via `_load_cached_structure()`; `_struct_cache` (instance-scoped dict, keyed by `os.path.abspath`, storing `(structure, (mtime, size))`) avoids re-parsing an unchanged file and re-parses when the file changes on disk; always returns `.clone()` on cache hit. Remote IDs (`pdb:XXXX`, `af:UniProtID`) are downloaded to `self._fetch_cache_dir` (default `~/.cache/pdb_align`, override with `PDB_ALIGN_CACHE_DIR`), written atomically; AlphaFold fetches fall back across model versions (v6→v5→v4).
    - `align(mode, atoms, strategy="auto", ...)` — modes: `"auto"`, `"seq_guided"`, `"seq_free_shape"`, `"seq_free_window"`, `"flexible"`. Flexible mode runs auto first, then detects hinges via `_detect_hinges`, runs per-domain Kabsch, returns domains in `result.domains`. When `mode="auto"` and **both** structures have more than one active chain, `align()` auto-dispatches to the multi-chain path (`chains.match_chains()` + `chains.align_multichain()`) instead of the single-chain seq-guided/seq-free comparison; single-chain behavior is unchanged. `strategy` (`"auto"`/`"global"`/`"local"`) is forwarded to `align_multichain()`.
    - `align_ensemble(mob_list, mode, atoms, out_dir)` — iterates a list of mobile paths, returns `EnsembleResult`; emits `UserWarning` per failed model
    - `batch_align()` / `batch_align_iter()` — directory-level batch with `ProcessPoolExecutor`
  - `inspect_structure(path_or_id, cache_dir=None)` — module-level helper returning `{"chains": {chain: n_residues}, "sequences": {chain: seq_str}}` for a local file or remote ID. Public so callers (e.g. the GUI chain pickers) never need `pdb_align.core`.

- **`chains.py`** — Chain correspondence and multi-chain superposition strategy, used when both structures have multiple chains:
  - `ChainMapping` — dataclass: `pairs` (list of `(ref_chain, mob_chain, identity, score)`), `unmatched_ref`, `unmatched_mob`
  - `match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct, ref_chains, mob_chains)` — optimal 1:1 chain correspondence via Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the pairwise % sequence identity matrix (`compute_chain_similarity_matrix`). When multiple candidate chains are within `_TIE_TOL` (5%) identity of each other (the homomultimer case), refines the assignment geometrically: superposes on the tied mapping's chain centroids, then reassigns by post-superposition centroid proximity (iterative centroid ICP). Refinement is capped at `MAX_PERMUTE_CHAINS=12` chains (skipped above that, keeping the plain Hungarian result).
  - `align_multichain(ref_struct, mob_struct, mapping, strategy="auto", ...)` — builds a **global** superposition (Kabsch over all mapped chains' CA atoms) and a **local** superposition (Kabsch over just the single highest-identity chain pair), then picks the winner by the same coverage-weighted score used elsewhere (`n_pairs / (1 + (rmsd/3Å)²)`) unless `strategy` forces `"global"` or `"local"`. Returns a `MultiChainResult` (`strategy`, `mapping`, `rotation`, `translation`, `rmsd`, `per_chain`, coords).

- **`plotstyle.py`** — Nature-journal matplotlib style, shared by `plot_rmsd` and `plot_summary`:
  - `apply_nature_style()` — context manager applying Nature-style `rcParams` (Helvetica/Arial sans-serif, small fonts, no top/right spines, 300 dpi) via `plt.rc_context`, restored on exit
  - `nature_figure(width="single"|"double", height=None)` — returns `(fig, ax)` sized to a Nature column width (89mm/183mm) at 300 dpi
  - `panel_label(ax, letter)` — places a bold panel label (e.g. "a", "b") at the axis top-left, outside the frame
  - `PALETTE` — Okabe–Ito colorblind-safe categorical color list

- **`interpretation.py`** — Pure, gemmi-free interpretation layer backing `AlignmentResult.quality`. `assess(...)` turns numbers already on a result into an `AlignmentQuality` (band/verdict/confidence/`flagged_regions`/warnings); `FlaggedRegion` is a contiguous high-RMSD or hinge run. All thresholds are named module constants (`TM_EXCELLENT`, `RMSD_FLAG_ABS`, `LOW_COVERAGE`, `CANDIDATE_DISAGREE`, …) so they are reviewable in one place. No I/O, unit-testable without structures.

- **`structure.py`** — `StructureBase` wraps `gemmi.Structure` with chain selection and subdomain range support (e.g., `"A:10-150"`).

- **`metrics.py`** — Standalone, dependency-free metric functions: `compute_d0()` (TM-score normalization distance, clamped to ≥0.5 Å), `calculate_tm_score()` (TM of a *given* superposition), `tm_optimal_superposition()` (TM-maximizing superposition for a fixed correspondence, TMscore-program style — this is what `AlignmentResult.get_tm_score()` reports, validated against TM-align via `tmtools`), `calculate_lddt()` (lDDT-Cα, chunked, wired into `summary_stats()["lddt_ca"]`), and `calculate_tm_pvalue()` (Xu & Zhang 2010 EVD, μ=0.1512/σ=0.0242; golden value P(TM≥0.5)=5.5e-7).

- **`interface.py`** — DockQ-family interface metrics, all constants from the published sources (see module docstring): `compute_dockq()` (fnat 5 Å heavy-atom contacts / iRMSD over native 10 Å interface backbone / LRMSD after receptor superposition / DockQ formula + CAPRI class; residue correspondence by per-chain sequence alignment; fnat-maximizing permutation search over sequence-identical chains for homomultimers; validated against the official `DockQ` package in `tests/test_golden_crossvalidation.py`), `epitope_metrics()` (epitope/paratope precision/recall/F1/Jaccard at 4.5 Å), `evaluate_antibody_complex()` (H+L merged as receptor + site diagnostics), `compute_pdockq()` (Bryant 2022 sigmoid on interface pLDDT × log10 contacts, reference-free).

- **`evaluate.py`** — `evaluate_models(ref, models, ...)` ranks N predicted models against a reference: fold metrics (TM/RMSD/GDT/lDDT-Cα/coverage) + optional DockQ columns (`receptor_chains`/`ligand_chains`) + epitope F1 and `cdr_h3` (`antibody_chains`/`antigen_chains`) + pDockQ + `iptm`/`pdockq2` when confidence files are given (`confidence_files=`, one entry per model) or auto-discovered as siblings. Returns `ModelEvaluation` (`.table` DataFrame ranked by DockQ else TM, `.best`, `.report()`, `.details`). Failing models warn and appear with NaNs, never silently dropped. CLI: `pdb_align REF --models M1 M2 ... [--receptor-chains/--ligand-chains | --antibody-chains/--antigen-chains] [--confidence F1 - F3]`.

- **`confidence.py`** — Prediction-confidence ingestion + pDockQ2. `load_confidence(paths)` parses AF2/ColabFold scores JSON, AF3 `*_summary_confidences.json` + `*_confidences.json`, Boltz `confidence_*.json` + `pae_*.npz`, and bare PAE `.npy`/`.npz` into a `ModelConfidence` (iptm/ptm/ranking_score/pae/plddt/chain_pair_iptm); `find_confidence_files(model_path)` auto-discovers siblings by the tools' naming conventions. `compute_pdockq2(struct, chains_a, chains_b, pae)` implements Zhu et al. 2023 (X = ⟨1/(1+(PAE/10)²)⟩·⟨pLDDT⟩ over CB≤8 Å interface contacts; sigmoid L=1.31034849, x0=84.7326239, k=0.0747157696, b=0.00501886443 from the reference implementation); PAE is asymmetric so both directions plus their mean are returned, and a PAE/residue-count mismatch raises rather than guessing.

- **`cdr.py`** — IMGT CDR annotation + per-CDR RMSD. `annotate_cdrs(seq, numberer=None)` (IMGT CDR1 27–38, CDR2 56–65, CDR3 105–117); default numberer is ANARCI (`pip install anarci` + HMMER 3.3.x `hmmscan` — HMMER ≥3.4 output breaks ANARCI's Biopython parser; a clear RuntimeError explains this), and any `numberer(seq) -> (numbering, start, chain_type)` callable can be injected (used by the unit tests, which run without ANARCI). `cdr_rmsd(ref, model, antibody_chains, ...)` superposes the combined framework backbone with one Kabsch fit and reports each CDR's backbone RMSD in that frame (`CDRResult`, `.h3` = CDR-H3 RMSD); kappa chains are typed "K" but labeled L1–L3.

- **`exceptions.py`** — Custom exceptions: `ParsingError`, `ChainNotFoundError`.

- **`__main__.py`** — CLI entry point, positional-first: `pdb_align REF MOB [options]`. By default it only prints a stats report to the terminal (`res.report(fmt="text")`, or `res.to_json()` with `--json`) — no files are written unless requested. Opt-in outputs: `-o/--out` (aligned structure via `save_aligned_pdb`), `--plot [FILE]` (per-residue RMSD plot, default `rmsd.png`), `--summary-plot [FILE]` (`plot_summary`, default `summary.png`), `--report FILE` (text or JSON, by extension), `--csv FILE` (per-residue RMSD table), `--save FILE.npz` (`AlignmentResult.save`), `--json` (machine-readable report to stdout), plus `--strategy {auto,global,local}`, `--mode`, `--ref-chains`/`--mob-chains`, `--atoms`, `--min-plddt`, `--show` (open plot windows instead of headless `Agg`), `-v/--verbose`. Legacy `--ref`/`--mob` flags remain accepted alongside the positionals. `REF`/`MOB` accept local file paths or remote IDs (`pdb:XXXX`, `af:UniProtID`) directly as positionals.

### Streamlit App (`struct_pair_align.py` + `webapp/` package)

The app is a **thin entry** (`struct_pair_align.py`, ~130 lines: page config, sidebar inputs, tab dispatch) on top of a small **`webapp/` package**, and imports **only the public API** — never `pdb_align.core`. This is enforced by `tests/test_convergence_guard.py` (scans the entry + every `webapp/*.py`). The `webapp/` package is app-level (like the entry script), not part of the installed `pdb_align` package; `pyproject.toml` sets `pythonpath = ["."]` so tests can import it.

- **`webapp/data.py`** — IO + orchestration: `save_upload_to_temp`, `list_chains` (via `pdb_align.inspect_structure`), `run_pairwise`/`run_ensemble` (thin wrappers over `PDBAligner.align`/`align_ensemble`), and `input_key` (stable hash of inputs for `st.session_state` result caching). No `st.*`, unit-tested.
- **`webapp/figures.py`** — pure Plotly builders (`per_residue_figure`, `distance_matrix_figure`, `pair_distance_hist`, `ensemble_rmsd_heatmap`), return figures, no `st.*`.
- **`webapp/viewer.py`** — Py3Dmol superposition from `AlignmentResult.aligned_structure()`; `remap_long_chain_ids()` (gemmi, >1-char chain names → unique single chars) and `structure_to_pdb_string()`.
- **`webapp/sections.py`** — thin `st.*` tab renderers: `render_header` (quality verdict), `render_overview`, `render_3d`, `render_per_residue`, `render_ensemble`, `render_evaluation` (ranked model table + best-model metrics + CSV download; fed by `data.run_evaluation`, shown as an "Evaluation" tab when the sidebar's "Model evaluation (interface)" expander is enabled), `render_export` (one-click `export_bundle`). The reference PDB text for the 3D view is stashed in `st.session_state["_ref_pdb_str"]` by the entry after a run. All `st.plotly_chart` calls use unique `key=` arguments.

Layout: compact sidebar (upload/fetch, ref + mobile(s), chain pickers, Mode/Strategy, Advanced) → persistent verdict header → tabs (Overview, 3D, Per-residue, [Ensemble when >1 mobile], Export). `tests/test_app_render.py` drives the real renderers end-to-end via Streamlit's `AppTest` (skipped without the `[app]` extras).

### Key design patterns

- `AlignmentResult` is **stateless** — all properties are computed lazily from the raw alignment dicts stored at construction time.
- `mode="auto"` runs both sequence-guided and sequence-free paths, then calls `pick_best_overall()` to select the winner.
- `mode="flexible"` first runs `mode="auto"`, then detects hinges on per-residue CA RMSD, and re-runs `_kabsch()` independently per domain.
- The structure cache is **instance-scoped** (not shared across `PDBAligner` instances) and returns `.clone()` on every hit to prevent in-place mutation from corrupting cached structures. It invalidates when the file's mtime/size changes.
- Both the internal shape-vs-window choice (`_select_seqfree_method`) and the overall seq-guided-vs-seq-free choice (`pick_best_overall`) rank candidates by the same coverage-weighted score, counting residues (CA atoms), not raw atoms. `chains.align_multichain()`'s global-vs-local choice uses the identical `n_pairs / (1 + (rmsd/3Å)²)` formula for consistency.
- `numba` JIT decorates `_sliding_window_mean` in `core.py`; the fallback no-op decorator ensures the code runs without numba installed.
- Multi-chain dispatch is automatic and narrow: `PDBAligner.align(mode="auto", ...)` only takes the `chains.py` path when *both* the reference and mobile selections have more than one active chain; anything with a single chain per side (including the common one-chain-per-structure case) goes through the original seq-guided/seq-free comparison unchanged.
- `AlignmentResult.save()`/`load()` round-trip is deliberately gemmi-free: `LoadedResult` only carries plain numpy/pandas/JSON data plus a `_SAVE_VERSION` check, so saved results can be replotted/reported later without the original structure files or even gemmi installed.
