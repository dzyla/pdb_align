# SP2 — Streamlit GUI Redesign (on the converged API)

**Date:** 2026-07-12
**Status:** Approved design, pending implementation plan
**Parent goal:** Turn `pdb_align` into a publishable tool. Sub-project 2 of 2.
Builds directly on SP1 (API convergence + interpretation core), which is merged.

## Motivation

`struct_pair_align.py` is a 1340-line single file that reimplements alignment by
importing `pdb_align.core` internals (`_extract_ca_infos`,
`perform_sequence_alignment`, `_kabsch`, `AlignmentResultSF`). It therefore never
uses the intelligent `PDBAligner.align()` path and can disagree with the library.
Its UX is form-heavy and exposes raw knobs (gap penalties, `recycles`,
`keep_fraction`) that a publishable tool should not put front-and-center.

SP1 added everything the GUI needs on the public API: `aligned_structure()`,
`quality`, `export_bundle()`, and the ensemble methods. SP2 rebuilds the app on
that API as a clean, guided, publish-quality tool.

## Scope

**In scope:** rewrite of the Streamlit app onto the public API; split into a small
testable `webapp/` package; unit tests for the extracted pure logic; flip the
convergence guard to strict.

**Non-goals:**
- No new alignment science — all computation stays in the library.
- No `core.py` changes.
- No API changes unless a genuine gap appears mid-build; if one does, add the
  small missing accessor to the **library** (with a test), not to the app.

## Section A — The convergence rule

The app imports **only** the public API: `from pdb_align import PDBAligner,
AlignmentResult, EnsembleResult, align` (plus `AlignmentQuality`/`FlaggedRegion`
for typing). **Zero** `pdb_align.core` imports anywhere in the app or `webapp/`.

Every rendered value comes from public accessors:
`PDBAligner.align()`, `AlignmentResult.{get_rmsd_df, aligned_structure, quality,
summary_stats, per_chain, domains, get_sequence_alignment, report_peaks,
export_bundle}`, and `EnsembleResult.{summary, rmsd_matrix, cluster, plot_pca,
plot_dendrogram, export_bundle}`.

The old GUI-only knobs `recycles` and `keep_fraction` are removed — they only
existed to drive the private path; `mode="auto"` and the library's own outlier
handling replace them.

**Guard:** `tests/test_convergence_guard.py` flips from `xfail` to a strict pass
asserting no `pdb_align.core` import in `struct_pair_align.py` **and** in every
`webapp/*.py`.

## Section B — File structure

Split the monolith into a thin entry plus a small, testable package. Files that
change together live together; pure logic is separated from `st.*` rendering so it
can be unit-tested.

- `struct_pair_align.py` — thin entry (~150 lines): `st.set_page_config`, sidebar
  inputs, orchestration, tab dispatch. Kept as the launch target
  (`streamlit run struct_pair_align.py`) per existing convention.
- `webapp/__init__.py` — package marker.
- `webapp/data.py` — upload/fetch/parse, temp-file handling, chain-length
  extraction, and cached align/ensemble wrappers (`st.cache_data`). Pure-ish; the
  cached wrappers take plain args (paths, chain lists, options) and return
  `AlignmentResult`/`EnsembleResult`. Unit-testable without a running server.
- `webapp/figures.py` — pure figure builders returning Plotly/Matplotlib figures,
  **no `st.*`**: `per_residue_figure(rmsd_df, flagged)`,
  `distance_matrix_figure(...)`, `pair_distance_hist(...)`,
  `ensemble_rmsd_heatmap(matrix_df)`, plus thin wrappers over
  `EnsembleResult.plot_pca`/`plot_dendrogram`. Unit-testable.
- `webapp/viewer.py` — Py3Dmol view from a `gemmi.Structure` + color mode; the
  multi-character chain-ID remap (`_remap_long_chain_ids`) lives here.
- `webapp/sections.py` — tab renderers (`render_overview`, `render_3d`,
  `render_per_residue`, `render_ensemble`, `render_export`). Call `st.*`; kept
  thin, delegating data/figures to `data.py`/`figures.py`.

## Section C — Layout

**Sidebar (inputs):**
- Upload (PDB/mmCIF, multiple) + fetch-by-ID (`pdb:XXXX`, `af:UniProtID`,
  comma-separated).
- Reference selectbox.
- Mobile multiselect: one selected → pairwise; more than one → ensemble.
- Chain pickers for reference and each mobile, labelled with residue counts.
- Mode selectbox (default **Auto**; also `seq_guided`, `seq_free_shape`,
  `seq_free_window`, `flexible`) and Strategy (`auto`/`global`/`local`).
- **Advanced** expander: gap-open/gap-extend penalties, `atoms`
  (CA/backbone/all_heavy), `min_plddt`.
- Run button.

**Main area:**
- **Persistent verdict header** (always visible after a run): band badge, one-line
  `quality.verdict`, key metrics (RMSD, TM-score, GDT-TS, coverage), confidence,
  and any `quality.warnings`. For an ensemble, show aggregate stats (mean RMSD,
  n models) and the best/worst model.
- **Tabs:**
  - **Overview** — method/strategy/reason, chain mapping, `per_chain` RMSD table,
    `quality.flagged_regions` table.
  - **3D** — Py3Dmol superposition built from `aligned_structure(color_by)`;
    color-by selector (RMSD / pLDDT / B-factor / chain); chain-visibility toggles;
    "jump to flagged region" buttons; download the aligned structure.
  - **Per-residue** — interactive per-residue RMSD plot with flagged regions
    shaded and top peaks marked; an expander for the sequence-alignment view
    (`get_sequence_alignment`), and an expander for distance matrices +
    pairwise-distance histograms.
  - **Ensemble** — shown only when >1 mobile: `summary()` table, RMSD-matrix
    heatmap, clustering (n-clusters slider → `cluster()`), PCA (`plot_pca`),
    dendrogram (`plot_dendrogram`).
  - **Export** — one-click `export_bundle` (zip) with component checkboxes, plus
    individual downloads (aligned PDB, RMSD CSV, plots, PyMOL/ChimeraX, report).
    Ensemble uses `EnsembleResult.export_bundle`.

## Section D — Reliability & state

- Results cached in `st.session_state` keyed by a stable hash of
  (ref path, sorted mobile paths, ref chains, per-mobile chains, mode, strategy,
  advanced options). Identical re-runs return instantly with no recompute.
- Parsing/alignment wrapped in `st.cache_data` in `webapp/data.py` so Streamlit
  reruns don't re-parse unchanged files (complements `PDBAligner`'s own cache).
- `AlignmentFailedError` and parse errors caught and shown as friendly `st.error`
  messages; per-model ensemble failures surfaced as `st.warning` (the API emits
  `UserWarning` per failed model).
- The 3D viewer is fed a temp PDB written from `aligned_structure()`, never
  hand-built coordinates, so the view cannot disagree with the reported numbers.

## Section E — Testing

Streamlit rendering is not unit-tested directly; the extracted pure logic is:
- `webapp/data.py`: cached align wrapper returns an `AlignmentResult`; ensemble
  wrapper returns an `EnsembleResult`; chain-length extraction; input-hash
  stability (same inputs → same key; changed inputs → different key).
- `webapp/figures.py`: each builder returns a non-None figure for pairwise and
  ensemble inputs without raising.
- `webapp/viewer.py`: chain-ID remap maps multi-char IDs to unique single chars.
- **Convergence guard** flipped to strict, scanning `struct_pair_align.py` and all
  `webapp/*.py`.
- **Import smoke**: importing `struct_pair_align` and `webapp.*` under a stubbed/
  headless Streamlit does not raise at import time.
- **Optional Playwright E2E** (best-effort, skipped if the environment can't run
  `streamlit` + a browser): launch the app, upload the two bundled test PDBs, click
  Run, assert the verdict header text appears.

## Acceptance

- App runs (`streamlit run struct_pair_align.py`) and produces a verdict header,
  all tabs, working 3D, and a downloadable bundle for a pairwise run; ensemble tab
  appears and works for >1 mobile.
- No `pdb_align.core` import anywhere in the app/`webapp/`; convergence guard
  passes strict.
- All new + existing tests pass (full `pytest tests/`).
- No `core.py` changes; any API gap discovered is fixed in the library with a test.

## Deliverables

- `struct_pair_align.py` (rewritten thin entry) + `webapp/` package
  (`data.py`, `figures.py`, `viewer.py`, `sections.py`, `__init__.py`).
- Tests for `data.py`, `figures.py`, `viewer.py`; strict convergence guard;
  import smoke; optional Playwright E2E.
- Updated `README.md` (Streamlit section: new layout, tabs, one-click bundle) and
  `CLAUDE.md` (Streamlit App section: new structure and `webapp/` modules).
