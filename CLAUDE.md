# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e .                 # core library + CLI (no numba, no seaborn)
pip install -e '.[speed]'        # + numba JIT (optional; changes no result)
pip install -e '.[app]'          # + Streamlit app (py3Dmol, plotly, ...)
pip install -e '.[validation]'   # + official DockQ and tmtools for the golden tests
pip install -e '.[dev]'          # everything + ruff
```
Requires Python >= 3.10. `numpy` is deliberately unpinned (the old `<2.0` cap
existed only for numba, now optional). There are no `requirements*.txt` files —
they duplicated and contradicted `pyproject.toml`.

### Run tests / lint / build
```bash
pytest                      # 232 tests; the whole suite must also pass on a core install
pytest -k crossvalidation   # agreement with the official DockQ and TM-align
pytest tests/test_core.py::test_compute_gdt_ts
ruff check pdb_align webapp struct_pair_align.py tests   # must be clean
python -m build && python -m twine check dist/*
```
Tests needing an optional extra skip themselves (`pytest.importorskip`), so a
core install reports passes and skips but never a failure. CI
(`.github/workflows/ci.yml`) runs the matrix 3.10-3.13, a core-install job, a
macOS job (spawn start method), lint and a distribution check.

### Run the Streamlit app
```bash
streamlit run struct_pair_align.py
```

## Architecture

The package has two layers: a **Python library** and a **Streamlit frontend**.

### Library (`pdb_align/`)

- **`__init__.py`** — Public API surface. Exports `PDBAligner`, `AlignmentResult`, `AlignmentFailedError`, `EnsembleResult`, `DomainResult`, `ParsingError`, `ChainNotFoundError`, and the top-level `align()` convenience function. The `align(ref, mob, chains_ref, chains_mob, **kwargs)` function is a zero-setup one-liner that wraps `PDBAligner`.

- **`core.py`** — The computational heart. **Read its module docstring before touching it**: it states the invariant the whole package rests on.
  - **`select_residues(struct, selectors, min_b_factor, min_plddt)` → `Selection`** — the *only* place that decides which residues take part in a comparison. A `Selection` holds `residues` (list of `ResidueSel`: chain/seqid/icode/name/letter/ca/b_iso/atoms) and the `sequence` describing exactly those residues, with `len(sequence) == len(residues)` asserted in `__post_init__`. Handles chain names, 1-based chain indices and `"A:10-150"` residue ranges, and raises (naming what the structure contains) for a missing chain, an empty range or an empty selection.
    - **Never build a sequence and a residue list separately.** That was the bug class fixed in 0.4.0: a filter applied to one and not the other silently mis-paired residues and reported 5.62 Å RMSD between a structure and itself.
  - `pairs_from_alignment(aln)` → index pairs; `paired_atoms(ref_sel, mob_sel, pairs, atoms)` → matched `AtomRef` lists. Pairing is **positional** — no residue-name rescanning. `paired_atoms` also enforces that side chains are only paired between residues of the same type.
  - `residue_letter(resname)` — one-letter code via gemmi's CCD tables, so modified residues resolve to their parent (MSE→M, SEP→S, …) for the whole CCD rather than a hand-written map. `AA_DICT` is gone.
  - `superimpose_atoms()` — Kabsch + optional iterative outlier rejection, returning **residue-level** `per_residue_rmsd` / `residue_labels` / `residue_keys` / `ca_ref` / `ca_mob` / `n_residues` (grouped on the residue *key*, not an index — the multi-chain path builds one selection per chain, so indices restart per chain).
  - `_kabsch()` — SVD superposition with a reflection guard; returns `(R, t, rmsd)`
  - `sequence_independent_alignment_joined_v2()` — sequence-free shape/window alignment. Accepts parsed structures or prebuilt selections (`ref_selection=`/`mob_selection=`) so it never re-reads files; refuses a selection above `MAX_SEQFREE_RESIDUES` (20000) because the method is O(N²) in memory.
  - `pick_best_overall()` / `_select_seqfree_method()` — rank candidates by the **coverage-weighted score** (`n_pairs / (1 + (rmsd/3Å)²)`), so a strategy matching a few residues at low RMSD cannot beat one that superimposes the whole protein. The same formula picks global vs local in `chains.py`.
  - `compute_gdt_ts(dists, n_total)` — single-superposition GDT_TS over ALL matched pairs, normalized by the reference selection length (CASP semantics — never an inlier subset). `compute_contact_overlap()` — chunked Cα contact-map Jaccard, now opt-in via `AlignmentResult.contact_overlap()` and no longer computed on every alignment.
  - `_detect_hinges(..., chain_starts=)` — split indices from a per-residue RMSD array; **chain starts are always splits and detection runs per chain** (a chain boundary is not a hinge).
  - `compute_chain_similarity_matrix(..., normalize="shorter")` — identity normalized by the shorter chain (not alignment length), memoised per sequence pair via `_pair_identity`.
  - `perform_sequence_alignment()` — semi-global BLOSUM62; configures free end gaps by probing the **class** for `end_insertion_score` (Biopython ≥1.86) or `target_end_gap_score` (older), because those properties' getters raise on an instance.
  - **numba is optional and lazy**: `lazy_jit` compiles on first call, `prange` falls back to `range`, and `numba_available()` reports status. All kernels pass `cache=True`. Import of `pdb_align` must not import numba (asserted in `tests/test_optional_numba.py`).
  - Uses **gemmi** for parsing and residue typing and **Biopython** only for sequence alignment (`Bio.PDB` is no longer imported anywhere).

- **`aligner.py`** — High-level public API:
  - `DomainResult` — dataclass for a single rigid domain from flexible alignment (`domain_id`, `chain_id`, `residue_start`, `residue_end`, `n_residues`, `rmsd`, `rotation`, `translation`)
  - `AlignmentResult` — stateless result object. All properties computed lazily from `_chosen`/`_seqguided`/`_seqfree` dicts. Has `rmsd` (**combined in quadrature** over domains when `domains` is set — RMSDs are root-mean-square quantities), `tm_score`, `domains` (List[DomainResult] or None), export methods, and plotting. Also carries `ref_selection`/`mob_selection`: the post-filter selections the comparison was actually made on, which every reported length, coverage and normalization refers to.
    - `.strategy` — `"single"` (one chain pair), `"global"`, or `"local"` (set by the multi-chain path; see `align_multichain()` below)
    - `.chain_mapping` — the `ChainMapping` used, or `None` for single-chain results
    - `.per_chain` — DataFrame (`chain_ref`, `chain_mob`, `n_residues`, `rmsd`) of per-chain-pair RMSD; empty for single-chain results
    - `summary_stats()` — method/strategy/rmsd/tm_score/gdt_ts/lddt_ca/coverage/chain_mapping, plus `tm_scope` (`"chain"`/`"complex"`), `tm_normalization_length`, `chain_mapping_warnings`, and `tm_score_per_chain` for multi-chain selections. `n_aligned` and `coverage_pct` are **residue** counts in every atom mode.
    - `tm_score_per_chain()` — per-chain TM, each normalized by its own reference chain. TM-score is a single-chain measure, so a complex-level value is labelled `TM-score (cplx)` and reported next to these.
    - `contact_overlap()` — opt-in, cached; O(N²) and in no report, so it is not computed during `align()`
    - `report(fmt="text"|"json")` — human-readable or JSON report string (`to_dict()`/`to_json()` back `report(fmt="json")`)
    - `save(path)` — persists RMSD table, per-chain table, coords, and `summary_stats()` to a versioned `.npz` (`_SAVE_VERSION`); static `load(path)` reloads it as a `LoadedResult`, a gemmi-free object that can still `report()`, `plot_rmsd()`, and `plot_summary()` (methods reused via delegation) without the original structure files
    - `plot_summary(filename=None, show=False)` — compact two-panel Nature-style figure (per-residue RMSD by chain + per-chain RMSD bar, or a text summary when there's only one chain pair), via `plotstyle`
    - `quality` — lazy property returning an `AlignmentQuality` (from `interpretation.py`): plain-language `band`/`verdict`/`confidence`, `flagged_regions` (contiguous high-RMSD or hinge runs), and `warnings`. Pure function of numbers already on the result (tm_score, rmsd, coverage, per-residue RMSD, both candidate RMSDs, domains). Surfaced in `report()`, `to_dict()["quality"]`, and the CLI verdict line
    - `get_rmsd_df(on=...)` — **one row per residue in every atom mode**; with `backbone`/`all_heavy` the value is the RMS over that residue's matched atoms
    - `aligned_structure(color_by="rmsd")` — in-memory transformed mobile `gemmi.Structure` for 3D viewing/export. `color_by="rmsd"` writes per-residue deviation into B-factors; `"bfactor"`/`"plddt"` preserve input B-factors. Shares the private `_build_aligned_structure()` helper with `save_aligned_pdb()`, so file output and 3D view are identical
    - `export_bundle(path, include=None, fmt="zip")` — one reproducible bundle (`.zip` or `fmt="dir"` folder): aligned structure, per-residue RMSD CSV, summary/RMSD plots, PyMOL `.pml` + ChimeraX `.cxc` scripts, and text/JSON report (both carrying the quality verdict). Replaces the Streamlit app's `export_zip_*` helpers
  - `LoadedResult` — returned by `AlignmentResult.load()`; wraps the saved meta/RMSD/per-chain data with no dependency on gemmi/the original files.
  - `EnsembleResult` — holds a list of `AlignmentResult` objects from an ensemble run. Methods: `summary()` → DataFrame, `rmsd_matrix()` → NxN DataFrame, `cluster(n_clusters)` → K-means labels, `plot_pca(color_by)` → matplotlib Figure, `plot_dendrogram()` → matplotlib Figure, `export_bundle(path, fmt="zip")` → bundle of `summary.csv`, `rmsd_matrix.csv`, `clusters.csv`, `pca.png`, `dendrogram.png`.
  - `_resolve_workers()` / `_process_pool()` / `_map_parallel()` — process-pool helpers. Pools use `forkserver`→`spawn`→`fork` in that order (plain `fork` can deadlock a child that inherited a BLAS thread lock) and `_map_parallel` **falls back to in-process execution with a warning** when a pool cannot start (forkserver/spawn need an importable `__main__`). `_ensemble_worker` returns plain data and `_RemoteAlignmentResult` rebuilds a read-only view, because `AlignmentResult` holds a `gemmi.Structure` and cannot be pickled.
  - `PDBAligner` — orchestrates loading, aligning, and batch processing:
    - `add_reference()` / `add_mobile()` — parse and cache structures via `_load_cached_structure()`; `_struct_cache` (instance-scoped dict, keyed by `os.path.abspath`, storing `(structure, (mtime, size))`) avoids re-parsing an unchanged file and re-parses when the file changes on disk; always returns `.clone()` on cache hit. Remote IDs (`pdb:XXXX`, `af:UniProtID`) are downloaded to `self._fetch_cache_dir` (default `~/.cache/pdb_align`, override with `PDB_ALIGN_CACHE_DIR`), written atomically; AlphaFold fetches fall back across model versions (v6→v5→v4).
    - `align(mode, atoms, strategy="auto", ...)` — modes: `"auto"`, `"seq_guided"`, `"seq_free_shape"`, `"seq_free_window"`, `"flexible"`. Builds both selections once via `_build_selections()` and passes them to every path. **Raises `AlignmentFailedError` (a `ValueError` subclass) rather than returning a result whose metrics are `None` or infinite.** `min_plddt` is applied only to a side whose B-factor column looks like pLDDT (`_looks_like_plddt`) and warns about any side it skipped — a crystallographic B-factor means the opposite of a confidence; `min_b_factor` stays symmetric.
    - `_align_flexible()` — runs auto, detects hinges **per chain**, then `_merge_rigid_segments()` merges adjacent segments that still fit as one rigid body (hinges are found in a compromise frame, so a rigidly-moved chain otherwise picks up spurious splits). A hinged haemoglobin gives 4 rigid domains at 0.000 Å, not 9. When `mode="auto"` and **both** structures have more than one active chain, `align()` auto-dispatches to the multi-chain path (`chains.match_chains()` + `chains.align_multichain()`) instead of the single-chain seq-guided/seq-free comparison; single-chain behavior is unchanged. `strategy` (`"auto"`/`"global"`/`"local"`) is forwarded to `align_multichain()`.
    - `align_ensemble(mob_list, mode, atoms, workers, out_dir)` — returns `EnsembleResult`; emits `UserWarning` per failed model. `workers > 1` (or `-1`) spreads models over a process pool and gives identical numbers to serial (asserted)
    - `batch_align()` / `batch_align_iter()` — directory-level batch with `ProcessPoolExecutor`
  - `inspect_structure(path_or_id, cache_dir=None)` — module-level helper returning `{"chains": {chain: n_residues}, "sequences": {chain: seq_str}}` for a local file or remote ID. Public so callers (e.g. the GUI chain pickers) never need `pdb_align.core`.

- **`chains.py`** — Chain correspondence and multi-chain superposition strategy, used when both structures have multiple chains:
  - `ChainMapping` — dataclass: `pairs` (list of `(ref_chain, mob_chain, identity, score)`), `unmatched_ref`, `unmatched_mob`, `warnings` (surfaced in the report and in `quality.warnings`), `refined` (whether geometry changed the mapping)
  - `match_chains(...)` — optimal 1:1 chain correspondence via Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on percent identity **normalized by the shorter chain** (normalizing by alignment length turned a perfect 50-residue domain match inside a 200-residue chain into "25% identity"). When candidates are within `_TIE_TOL` (5%) of each other (the homomultimer case, where identity is tied by construction and only geometry can decide), the mapping is refined by **iterated** centroid ICP: superpose on the current mapping's chain centroids, reassign by post-superposition proximity, repeat until it stops changing (`_REFINE_MAX_ROUNDS=10`, best-scoring round wins), skipped above `MAX_PERMUTE_CHAINS=24` chains.
  - A correspondence below `WEAK_IDENTITY` (25%) or between chains differing in length by more than 5x produces a `UserWarning`, lands in `mapping.warnings`, and **caps the reported confidence at `low`** — unrelated chains reach ~10-20% identity by chance. Unmatched-chain notes stay on the mapping only (routine when matching a two-chain receptor against every model chain).
  - `align_multichain(..., ref_selections=, mob_selections=)` — builds a **global** superposition (Kabsch over all mapped chains) and a **local** one (the single highest-identity chain pair), then picks by the same coverage-weighted score unless `strategy` forces `"global"`/`"local"`. Per-chain selections are built once and shared by both candidates (they used to be rebuilt, re-running every chain's sequence alignment twice). Returns a `MultiChainResult`.

- **`plotstyle.py`** — Nature-journal matplotlib style, shared by `plot_rmsd` and `plot_summary`:
  - `apply_nature_style()` — context manager applying Nature-style `rcParams` (Helvetica/Arial sans-serif, small fonts, no top/right spines, 300 dpi) via `plt.rc_context`, restored on exit
  - `nature_figure(width="single"|"double", height=None)` — returns `(fig, ax)` sized to a Nature column width (89mm/183mm) at 300 dpi
  - `panel_label(ax, letter)` — places a bold panel label (e.g. "a", "b") at the axis top-left, outside the frame
  - `PALETTE` — Okabe–Ito colorblind-safe categorical color list

- **`interpretation.py`** — Pure, gemmi-free interpretation layer backing `AlignmentResult.quality`. `assess(...)` turns numbers already on a result into an `AlignmentQuality` (band/verdict/confidence/`flagged_regions`/warnings); `FlaggedRegion` is a contiguous high-RMSD or hinge run. All thresholds are named module constants (`TM_EXCELLENT`, `RMSD_FLAG_ABS`, `LOW_COVERAGE`, `CANDIDATE_DISAGREE`, …) so they are reviewable in one place. No I/O, unit-testable without structures.

- **`metrics.py`** — Standalone, dependency-free metric functions: `compute_d0()` (TM-score normalization distance, clamped to ≥0.5 Å), `calculate_tm_score()` (TM of a *given* superposition), `tm_optimal_superposition()` (TM-maximizing superposition for a fixed correspondence, following the reference TM-score search: fragment seeds at L, L/2, L/4 ... >= 4 across start positions, `d0_search` clamped to [4.5, 8] Å and grown when too few residues qualify; `max_starts` caps the seeds, which cannot lower the score since only the best seed is kept. Validated against TM-align via `tmtools` to within 0.011), `calculate_lddt()` (lDDT-Cα, chunked, wired into `summary_stats()["lddt_ca"]`), and `calculate_tm_pvalue()` (Xu & Zhang 2010 EVD, μ=0.1512/σ=0.0242; golden value P(TM≥0.5)=5.5e-7).

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

**The invariant everything rests on.** One selection, one sequence. `select_residues()` is the only thing that decides which residues take part, and the `Selection` it returns guarantees `len(sequence) == len(residues)`. Alignment columns map onto coordinates **by index**. Never derive a sequence in one place and a coordinate list in another, and never "resynchronise" a pairing by matching residue names — that is exactly how eight residue-pairing bugs reached 0.3.0, including 5.62 Å RMSD between a structure and an identical copy of itself.

**Fail loudly.** An empty selection, an unknown chain, a range that selects nothing, or a mode that cannot superpose raises (`AlignmentFailedError` subclasses `ValueError`). A result object handed to a caller always has usable numbers. A plausible wrong number costs far more than an exception.

**Say what a number is normalized by.** GDT_TS by the reference selection length (CASP semantics, a lower bound on LGA's GDT); TM-score by the reference length, with `tm_scope`/`tm_normalization_length` and per-chain values when the selection spans chains; lDDT over matched residues only, so it is read with `coverage_pct`. `docs/METHODS.md` is the contract — **update it when a metric changes.**

**Residue-level reporting.** `n_aligned`, `coverage_pct` and `get_rmsd_df()` are per residue in every atom mode; only the Kabsch fit itself is per atom.

**Other invariants**
- `AlignmentResult` is **stateless** — properties computed lazily from the raw alignment dicts stored at construction.
- `mode="auto"` runs both paths and calls `pick_best_overall()`. The internal shape-vs-window choice, the seq-guided-vs-seq-free choice, and `align_multichain`'s global-vs-local choice all use the identical coverage-weighted score, counting residues.
- Multi-chain dispatch is automatic and narrow: only when *both* selections have more than one chain. Single-chain-per-side goes through the unchanged seq-guided/seq-free comparison.
- `mode="flexible"` detects hinges **per chain** and merges adjacent segments that still fit as one rigid body.
- The structure cache is instance-scoped, returns `.clone()` on every hit, invalidates on mtime/size change, and is **bounded** (`_CACHE_MAX = 4`) so evaluating N models does not retain N structures.
- `numba` is optional and lazy (`lazy_jit`); it must change no result, and `import pdb_align` must not import it.
- Parallel execution must be an optimisation only: identical numbers to serial, errors carried back from workers and re-emitted (a `warnings.warn` in a worker never reaches the parent), and a graceful in-process fallback when a pool cannot start.
- `AlignmentResult.save()`/`load()` is deliberately gemmi-free: `LoadedResult` carries plain numpy/pandas/JSON plus a `_SAVE_VERSION` check.

### Working on this repo

- **TDD.** Every fix here landed as a failing test first. The bugs that survived to 0.3.0 did so because the fixtures were straight-line poly-alanine rods: collinear CAs make a Kabsch fit rank-deficient about the chain axis, so a decoy can be rotated around it for free. Use the real fixtures in `tests/data/` (1UBQ, 1CRN, trimmed 4HHB; add new ones with `git add -f`, since `*.pdb` is otherwise ignored) and keep synthetic complexes helical (`tests/_synthetic.py`).
- **Don't mock what you can build.** A test double that subclasses `AlignmentResult` without running its `__init__` breaks the moment production code touches a new field; write a small stub implementing only what the consumer needs.
- **Changing a reported number** means: a regression test, a `docs/METHODS.md` update, and a `CHANGELOG.md` entry marked **[affects results]**.
- `ruff check pdb_align webapp struct_pair_align.py tests` must stay clean, and the whole suite must pass on a core install (optional-extra tests skip themselves).
