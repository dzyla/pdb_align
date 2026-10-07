# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); entries that change a reported
number are marked **[affects results]**.

## [0.4.1] — 2026-10-07

A follow-up to the 0.4.0 audit: the paths that silently did the wrong thing,
or nothing at all. One change affects a reported number (DockQ on near-
identical chain copies); the rest make failures visible or unblock exports
that used to crash.

### Fixed — chain correspondence for DockQ **[affects results]**

- **Interchangeable chains were found by exact sequence equality.** Copies of
  one chain in a deposited structure almost always differ by a disordered
  terminus or loop, so the fnat-maximizing permutation search added in 0.4.0
  switched itself off on precisely the structures it exists for. On 4HHB,
  removing three modelled residues from one α chain dropped the candidate
  mappings from four to two and lost the α/α swap. Two chains now count as
  copies at ≥95% identity over the shorter chain with a length ratio ≥0.9;
  the length guard keeps a short fragment out of the class of a long chain it
  matches perfectly over its own length. Agreement with the official `DockQ`
  package is unchanged (`tests/test_golden_crossvalidation.py`).

### Fixed — exports

- **Every PDB export died on chain names longer than one character** — raw
  `RuntimeError: chain name too long for the PDB format` out of
  `save_aligned_pdb()`, `export_bundle()` and the CLI's `-o`, on exactly the
  large assemblies and RCSB mmCIF downloads they are most useful for. Such a
  structure is now written as mmCIF under the same basename, with a warning;
  `save_aligned_pdb()` returns the path actually written, and the bundle's
  PyMOL/ChimeraX scripts name the files it really contains.
- **`save_aligned_pdb(subset_only=True)` was accepted and ignored**, writing
  the whole mobile structure. It now writes only the residues that were
  matched.
- **`export_bundle()` swallowed a plot failure silently**, shipping an archive
  missing the figures it promised. It warns.
- **One implementation of the viewer scripts.** `save_pymol_script()` /
  `save_chimerax_script()` were a second, untested pair that had drifted from
  the ones the bundle ships: a different colour scale, and one `alter` command
  per residue re-injecting a B-factor column the aligned file already carries.
  They now delegate to the bundle's writers.

### Fixed — silent degradation

- **`mode="auto"` could compare one candidate and still call itself auto.** A
  sequence-free failure was logged and forgotten; the report, the JSON and the
  quality verdict all described a two-strategy comparison that never happened.
  The reasons are now carried as `summary_stats()["candidate_failures"]`,
  printed in the report, added to `quality.warnings`, and they cap confidence.
- **A mistyped keyword silently disabled the sequence-free path.**
  `align(**kwargs)` forwarded everything, so `align(recycle=5)` raised a
  `TypeError` inside that path, which was caught and logged. `align()` now
  declares `recycles`, `keep_fraction`, `shape_nbins`, `shape_gap_penalty` and
  `shape_band_frac` explicitly, and an unknown keyword raises.
- **A multi-model file was truncated to model 1 in silence.** An NMR ensemble,
  a multi-model prediction or MD snapshots now raise a `UserWarning` naming
  the model count.
- **`LoadedResult.get_rmsd_df(on="mobile")` ignored `on`** and returned the
  reference-numbered table, labelling every residue with the wrong structure's
  numbering. It raises and explains instead.

### Fixed — other

- **`LoadedResult.report()` raised `AttributeError`** — the one method a saved
  result exists for. It now carries the quality verdict, recomputed from the
  saved numbers.
- Ensemble figures labelled every point with the full model path; they use the
  model name (the tables keep the path).
- The example notebook imported `pdb_align.structure`, a module that has not
  existed for several releases, and died on its first cell. A test now checks
  every module the notebook imports.
- `pdb_align/aligner.py.bak`, a stale 37 KB copy, was tracked inside the
  package directory. Removed.
- Dead parameters removed (`_joint_mapping_options(model_struct)`,
  `_mapping_permutations` in full — it had no callers,
  `render_export(is_ensemble)`).

### Added

- **`--min-b-factor`** on the CLI and a **Min B-factor** input in the app. The
  library has filtered on B-factors since 0.4.0 and the pLDDT warning tells
  users to use it, but neither front end exposed it.
- **`--export-bundle FILE.zip`** on the CLI: the reproducible bundle that was
  previously API- and GUI-only.
- `--mode` and `--atoms` validate their values (`--mode seqguided` was reaching
  the aligner and coming back as "produced no alignment. unknown mode"), and
  `-v` prints a traceback on failure instead of one opaque line.
- The quality layer warns below 30 matched residues, where TM-score's d0 is
  clamped and the band stops meaning what it says, and caps confidence there.
- README figures: app screenshots, the summary and per-residue plots, and the
  ensemble PCA, all generated by the code they document.

## [0.4.0] — 2026-10-05

A correctness and maturity release. Several fixes change numbers that earlier
versions reported, in some cases substantially. If you have published or saved
results from 0.3.x, re-run them — in particular anything that used
`--atoms backbone`/`all_heavy`, `min_plddt`, `min_b_factor`, residue-range
selectors, or `mode="flexible"`.

### Fixed — residue pairing **[affects results]**

The sequence handed to the aligner and the residue list providing coordinates
were built by two independent passes over the structure, so any filter applied
to one and not the other shifted the pairing silently. `core.select_residues()`
is now the single source of both, with `len(sequence) == len(residues)`
asserted, and alignment columns map onto residues by index instead of by
re-scanning for a matching one-letter code.

- **B-factor / pLDDT / residue-range filters mis-paired residues.** Two
  identical copies of 1UBQ with two residues filtered out of one side gave 24
  pairs, four of them joining different residues, and 5.62 Å RMSD between a
  structure and itself. Now 74 pairs at 0 Å.
- **`"A:10-150"` residue ranges were rejected** ("not present in the
  structure") although documented in the README, the CLI help and CLAUDE.md.
- **`n_aligned` and `coverage_pct` counted atoms, not residues.** With
  `--atoms all_heavy`, 76 residues were reported as 602 and coverage as 792%.
  `get_rmsd_df()` is now one row per residue in every atom mode, with the
  per-residue value the RMS over that residue's matched atoms. This also fixes
  the per-residue plot and the B-factor colouring of the aligned output.
- **`atoms="all_heavy"` paired side chains across unlike residues** (an ALA CB
  onto a TRP CB, by name). Unlike pairs now contribute backbone only.
- **`min_plddt` was applied to experimental references.** A crystallographic
  B-factor means the opposite of a pLDDT, so `--min-plddt 70` on a crystal
  structure discarded every residue and aborted the run. It now applies only
  to a side whose B-factor column looks like pLDDT, and warns, naming any side
  it skipped. `min_b_factor` keeps its symmetric semantics.

### Fixed — flexible mode **[affects results]**

- **Domains spanned chain boundaries.** Hinge detection ran on a concatenated
  per-residue array, producing domains such as "chain A 42–21" fitted with a
  single rotation. Detection is now per chain.
- **Domain RMSDs were averaged arithmetically.** RMSDs combine in quadrature;
  a hinged haemoglobin reported 1.22 Å where the deviation is 2.23 Å.
- **Spurious splits are merged.** Adjacent segments that still fit as one
  rigid body are merged back, since hinges are detected in a compromise frame.
  The hinged haemoglobin now gives 4 rigid domains at 0.000 Å instead of 9
  with a 4.55 Å artefact.

### Fixed — chain correspondence **[affects results]**

Found by cross-validating DockQ against the official implementation on a
*real* multi-chain complex; the previous cross-validation used only synthetic
single-chain-per-side decoys, where neither defect can appear.

- **The geometric tie-break could overrule sequence.** The homomultimer
  refinement reassigned chains by centroid proximity alone, so on 4HHB
  (A, C identical α-globin; B, D identical β) it paired reference α-globin
  with model β-globin — `fnat` went to 0 while every other number still looked
  plausible. Geometry may now only permute chains sequence says are
  interchangeable (within the 5% tie tolerance).
- **The receptor group could steal the ligand's chains.** Matching the
  receptor first against every model chain and the ligand among the leftovers
  inverted the interface on a symmetric assembly: asking for receptor C+D gave
  receptor C→A, D→B and ligand A→C, B→D, and an LRMSD of 14.8 Å where the
  correct answer is 20.6 Å. Both groups are now assigned in one search, and
  candidate mappings are enumerated across both groups and ranked by fnat —
  the criterion that actually defines the correct mapping for DockQ.

### Fixed — other

- `export_bundle()` shipped no reference structure, and its PyMOL/ChimeraX
  scripts loaded the aligned mobile twice, so the side-by-side view they
  promise could not be reproduced. The bundle now contains `reference.pdb`.
- `plot_rmsd()` imported seaborn, which is declared only in the `[app]` extra,
  so `pdb_align --plot` raised `ImportError` on a core install. It now uses
  matplotlib only.
- Explicit modes (`seq_guided`, `seq_free_*`) returned an `AlignmentResult`
  whose every metric was `None`, or an infinite RMSD, instead of failing.
  They raise `AlignmentFailedError`, which now subclasses `ValueError` so one
  `except ValueError` covers every unanswerable request.
- A selection that matches nothing, a missing chain, or a residue range
  outside the chain raises with a message naming what the structure contains.

### Changed — metrics **[affects results]**

- **TM-score search** now follows the reference TM-score program (fragment
  seeds at L, L/2, L/4 … ≥ 4 across start positions; `d0_search` clamped to
  [4.5, 8] Å and grown when too few residues qualify) instead of seven
  non-overlapping seeds with a fixed cutoff. Agreement with TM-align is within
  0.011 on decoys where the correspondence is correct, and the search is 30×
  faster than the exhaustive version (0.18 s vs 5.47 s at L = 3000).
- **Chain identity is normalized by the shorter chain**, not by alignment
  length including terminal gaps. A domain matching its parent chain perfectly
  now reads ~100% instead of a sequence-length ratio, which fixes chain
  matching for truncated constructs, Fv fragments and single-domain models.
- **Modified residues** resolve to their parent one-letter code through
  gemmi's CCD tables rather than a 22-entry hand-written map, so
  phosphoresidues, methylated lysines, selenomethionine and the rest keep
  their place in the sequence.
- **Homomultimer chain refinement iterates** to convergence (capped at 10
  rounds) instead of running a single pass, and keeps the best-scoring round.

### Added

- **Per-chain TM-scores** and explicit TM normalization scope. A complex-level
  TM-score is labelled `TM-score (cplx)`, carries `tm_scope` and
  `tm_normalization_length` in `summary_stats()`, and is reported next to the
  per-chain values — TM-score is defined for one chain against one chain.
- **Weak chain correspondence is flagged.** A mapping below 25% identity, or
  between chains differing in length by more than 5×, warns, appears in the
  report and in `quality.warnings`, and caps confidence at `low`.
- **`workers` works.** `align_ensemble(workers=…)` was documented as
  "reserved for future parallel execution" and ignored; `evaluate_models()`
  had no such option. Both now use a process pool (6× on 8 workers for 16
  four-chain complexes with DockQ), with identical results to serial.
- `AlignmentResult.contact_overlap()` — opt-in, cached.
- `docs/METHODS.md` — what every number means, what it is normalized by, and
  every deviation from the reference implementations.
- `CITATION.cff`, `LICENSE` (AGPL-3.0-or-later), `COPYRIGHT`, `py.typed`,
  a ruff configuration, and GitHub Actions CI across Python 3.10–3.13.

### Performance

- numba is **optional** (`pip install pdb_align[speed]`) and imported lazily;
  results are identical without it. `import pdb_align` dropped from 374 ms to
  255 ms, and all JIT kernels now cache their compilation (1.50 s → 0.16 s per
  process, previously paid again in every worker).
- The sequence-free path no longer re-reads both files from disk on every
  call; a 200-model ensemble re-parsed the reference 200 times.
- Window pairing 8.4× faster at N = 2000; radial histograms 1.7× faster and
  bit-identical; banded DP stores only its band; contact overlap chunked (900
  MB → 310 MB at N = 6000) and no longer computed on every alignment.
- Chain identities memoised per sequence pair (a 24-mer ran 576 identical
  alignments); BLOSUM62 loaded once; structure cache bounded at 4 entries.
- Process pools use `forkserver`/`spawn` rather than `fork`, which can deadlock
  a child that inherited a BLAS thread lock, and fall back to in-process
  execution with a warning when a pool cannot start.

### Removed

- `core.progressive_align_ensemble()` — unreachable from the public API.
- `core.compute_cad_score_approx()` — it never computed CAD-score. Use
  `AlignmentResult.contact_overlap()`.
- `pdb_align.structure.StructureBase`, `core._AllAtomsSelect`,
  `core.structure_based_alignment_strings`, `core._robust_inlier_mask` — dead
  code; removing them also drops the `Bio.PDB` import entirely.
- `requirements*.txt` — they duplicated and contradicted `pyproject.toml`
  (one omitted numba; seaborn was needed by core code but declared only in
  `[app]`). Use `pip install -e .[dev]`.
- The `numpy<2.0` pin, which existed only for numba.

### Testing

- Real-structure fixtures (1UBQ, 1CRN, trimmed 4HHB). The previous suite was
  entirely straight-line poly-alanine rods — collinear Cα positions make a
  Kabsch fit rank-deficient about the chain axis — which is why none of the
  pairing bugs were caught. Synthetic complexes are now ideal α-helices.
- DockQ cross-validation extended to six synthetic decoys spanning DockQ
  0.02–1.00 and three CAPRI classes, **and** to the merged multi-chain path on
  a real complex (4HHB, receptor A+B vs ligand C+D at rotations of 0–40°),
  with tolerances tightened to 1e-4 Å (iRMSD/LRMSD) and exact agreement on
  fnat, plus a guard test that the decoys really do span the range.
- New suites: residue-selection integrity, flexible-domain decomposition,
  failure modes, parallel-equals-serial, numba-optional equivalence, kernel
  equivalence against the obvious implementations, reporting claims, and
  **known biology** — haemoglobin's identical α copies (0.3 Å, TM 0.996), the
  homologous α/β pair (1.57 Å, TM 0.895 against TM-align's 0.904, literature
  ~1.5–2 Å), the swapped-identical-chain correspondence recovered by geometry,
  and ubiquitin vs crambin as a negative control (TM 0.19, p = 0.18).
- 251 tests (from 170).

## [0.3.0] — 2026-08-30

PAE/ipTM ingestion (pDockQ2), CDR metrics, GUI evaluation tab.

## [0.2.0]

Scientific-correctness overhaul; DockQ, epitope and model-evaluation layer;
Streamlit app rewritten on the public API.

## [0.1.0]

Initial release: sequence-guided and sequence-free alignment, ensembles,
Streamlit front-end.
