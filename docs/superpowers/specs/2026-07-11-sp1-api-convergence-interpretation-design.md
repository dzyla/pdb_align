# SP1 — API Convergence + Interpretation Core

**Date:** 2026-07-11
**Status:** Approved design, pending implementation plan
**Parent goal:** Turn `pdb_align` into a publishable tool. This is sub-project 1 of 2.
SP2 (Streamlit GUI redesign) builds on the API this sub-project delivers.

## Motivation

The library API (`PDBAligner`, `AlignmentResult`, `core.py`) is mature and
well-tested. The Streamlit app, however, is a **parallel reimplementation**: it
calls `core` internals directly (`_extract_ca_infos`, `perform_sequence_alignment`,
`_kabsch`, `AlignmentResultSF`) and never goes through the intelligent
`PDBAligner.align()` path (chain matching, strategy selection, coverage-weighted
scoring). Consequences:

1. The app and the library can produce **different answers** for the same inputs.
2. Alignment logic is **maintained twice**.
3. Alignment quality is presented as raw numbers with **no interpretation**, so a
   user cannot tell whether a result is trustworthy.

SP1 fixes the foundation so SP2 can be a thin, correct GUI:
- Make the library the **single source of truth** (close accessor gaps so the GUI
  never needs `core` internals).
- Add an **interpretation layer** that turns numbers into a plain-language,
  confidence-scored verdict — shared by CLI, GUI, and scripts.
- Consolidate exports into a **single reproducible bundle** call.

## Scope

**In scope:** additive changes to `aligner.py` (and a small new module for the
interpretation layer); new tests; documentation of the GUI→API accessor mapping.

**Non-goals (SP1):**
- No GUI changes (that is SP2).
- No `core.py` algorithm changes — the computational core is preserved as-is.
- No new alignment modes or scoring formulas.

## Section A — Convergence contract (single source of truth)

**Rule:** every value the GUI renders must be reachable from `PDBAligner` /
`AlignmentResult` / `EnsembleResult`. The GUI must never import from
`pdb_align.core`.

### A1. `AlignmentResult.aligned_structure(color_by="rmsd")`
New method returning an **in-memory** transformed `gemmi.Structure` (the mobile
structure moved onto the reference), with a per-residue value written into each
atom's B-factor:
- `color_by="rmsd"` (default): per-residue alignment deviation (Å), the same
  values `get_rmsd_df()` reports and `save_aligned_pdb()` already writes.
- `color_by="bfactor"` / `"plddt"`: leave original B-factors untouched.

Refactor: extract the coordinate-transform + B-factor-mapping loop currently
inside `save_aligned_pdb()` into a private helper
`_build_aligned_structure(color_by)`. Both `save_aligned_pdb()` and
`aligned_structure()` call it, so file output and 3D view are guaranteed
identical. `save_aligned_pdb()` behavior is unchanged (regression-tested).

### A2. Accessor-coverage audit
Produce (in this spec's appendix during implementation, or as a docstring table)
a mapping from **each quantity the current GUI computes itself** to the API
accessor that now provides it. Every row must resolve to one of:
`get_rmsd_df`, `get_aligned_coords`, `get_matched_pairs`, `aligned_structure`,
`summary_stats`, `per_chain`, `domains`, `quality`, `get_sequence_alignment`.
Any quantity with no accessor is a gap to close in SP1. This table becomes SP2's
build checklist.

### A3. No `core` leakage
Add a test asserting the GUI module (once SP2 exists) imports nothing from
`pdb_align.core`. For SP1, add the same guard as a documented contract and a test
scaffold that scans `struct_pair_align.py` for `from pdb_align.core` /
`import pdb_align.core` and is currently expected to fail (xfail) — it flips to
passing in SP2. This makes the convergence goal executable, not aspirational.

## Section B — Interpretation layer (`AlignmentQuality`)

New module `pdb_align/interpretation.py` with a frozen dataclass `AlignmentQuality`,
exposed as a lazy property `AlignmentResult.quality`. It is a **pure function of
data already on the result** (rmsd, tm_score, coverage, per-residue RMSD DataFrame,
domains, per_chain, tm_pvalue, and both candidate RMSDs) — no new alignment.

### Fields
- `band: Literal["excellent","good","moderate","poor"]`
  - `excellent`: TM-score > 0.9
  - `good`: TM-score > 0.5 (same fold)
  - `moderate`: TM-score > 0.3
  - `poor`: otherwise
  - When TM-score is unavailable, fall back to RMSD thresholds
    (`<1`, `<2.5`, `<5`, else) documented in code.
- `verdict: str` — one plain-language sentence, e.g.
  `"Near-identical fold: 98% of residues superimpose within 1.2 Å (TM=0.94)."`
- `confidence: Literal["high","medium","low"]` — from a small rubric:
  coverage (fraction of residues aligned), `tm_pvalue`, and agreement between the
  seq-guided and seq-free candidate RMSDs (large disagreement lowers confidence).
- `flagged_regions: list[FlaggedRegion]` — contiguous residue runs where
  per-residue RMSD exceeds `max(2.0 Å, 2 × median RMSD)`. Each `FlaggedRegion`
  carries `chain`, `start_label`, `end_label`, `n_residues`, `max_rmsd`,
  `mean_rmsd`, and `kind` (`"deviation"`; or `"hinge"` when it corresponds to a
  domain boundary from flexible mode).
- `warnings: list[str]` — human-readable flags: low coverage (<50%), large length
  mismatch between structures, "aligned only k of N chains," missing TM-score, etc.

### Thresholds
All thresholds live as named module constants (`TM_EXCELLENT=0.9`,
`RMSD_FLAG_ABS=2.0`, `RMSD_FLAG_REL=2.0`, `LOW_COVERAGE=0.5`, …) so they are
reviewable and tunable in one place.

### Surfacing
- `AlignmentResult.report(fmt="text")` gains a **Quality** section
  (band, verdict, confidence, top flagged regions, warnings).
- `to_dict()` / `to_json()` include a `quality` object.
- CLI prints the verdict line by default (it is the headline a user reads first);
  full quality block shown under `-v` and always in `--json`.

## Section C — Reproducible export bundle

Move the GUI's `export_zip_*` and `_generate_pymol_script` logic into the API.

### `AlignmentResult.export_bundle(path, include=None, fmt="zip")`
`path` is a directory or `.zip`. `include` selects components (default = all):
- `aligned` — aligned structure (`aligned_structure()` written to PDB/CIF)
- `rmsd_csv` — `get_rmsd_df()` as CSV
- `plots` — `plot_summary()` and `plot_rmsd()` PNGs
- `pymol` — `.pml` script that loads ref + aligned mobile and colors by B-factor
- `chimerax` — `.cxc` equivalent
- `report` — `report.txt` and `report.json` (both include the quality verdict)

Deterministic filenames; returns the bundle path. `fmt="dir"` writes a folder
instead of a zip.

### `EnsembleResult.export_bundle(path, ...)`
Analog for ensembles: RMSD matrix CSV, cluster labels, PCA and dendrogram PNGs,
and a combined `summary.csv` (from `EnsembleResult.summary()`).

## Section D — Testing (TDD) & acceptance

Write tests first, per component:

1. **`aligned_structure`**: B-factor of each residue equals the corresponding
   `get_rmsd_df()` RMSD (within float tolerance); `color_by="bfactor"` preserves
   input B-factors; transformed coords match `save_aligned_pdb()` output.
2. **`quality`**: crafted fixtures — identical structures → `band=="excellent"`,
   no flagged regions, `confidence=="high"`; a two-domain hinge fixture → at least
   one `flagged_regions` entry of `kind=="hinge"`; a partial/low-coverage case →
   a coverage warning and reduced confidence.
3. **`export_bundle`**: all requested components present in the zip; `report.json`
   parses and contains `quality`; PyMOL/ChimeraX scripts reference the written
   files; round-trips through `AlignmentResult.load()` where applicable.
4. **Regression**: existing `save_aligned_pdb`, `report`, `summary_stats`, and CLI
   tests still pass unchanged.
5. **Convergence guard**: xfail test scanning the GUI for `pdb_align.core` imports
   (Section A3), to be flipped in SP2.

**Acceptance:** all new + existing tests pass; `report()`/`--json` show the quality
verdict; a single `export_bundle()` call produces a complete, reloadable bundle;
no `core.py` changes; the GUI→API accessor table is complete with no open gaps.

## Deliverables

- `pdb_align/interpretation.py` (`AlignmentQuality`, `FlaggedRegion`, thresholds)
- `AlignmentResult.aligned_structure()`, `.quality`, `.export_bundle()`
- `EnsembleResult.export_bundle()`
- CLI wiring for the quality verdict
- Tests for all of the above; GUI→API accessor mapping table
- Updated `CLAUDE.md` / `README.md` API docs
