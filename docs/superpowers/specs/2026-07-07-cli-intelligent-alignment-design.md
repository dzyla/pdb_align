# Design: Go-to structure comparison — intelligent multi-chain alignment, rich results, Nature-style plots

Date: 2026-07-07

## Goal

Make `pdb_align` the go-to tool for comparing two protein structures, from both the
command line and Python. Three thrusts:

1. **Intelligent multi-chain alignment** — when structures have many chains, automatically
   determine chain correspondence (optimal, not just greedy) and automatically choose
   between a global (all-chains) and local (best single chain-pair) superposition.
2. **Zero-config CLI** with minimal default output — print stats to the terminal, write
   files only when explicitly requested.
3. **Data-rich Python result object** that holds all computed data (so it can be plotted,
   saved, reloaded, and re-reported without the original files) and produces
   **Nature-journal-style** plots.

## Non-goals (YAGNI)

- No HTML report, interactive dashboard, or new 3D rendering. Plots stay matplotlib.
- No bundled Helvetica font file — Nature style is rcParams + palette + spine/width
  conventions, with graceful fallback to Arial/DejaVu.
- Optimal chain permutation is **capped** so large complexes stay fast.
- Single-chain alignment behavior is **unchanged**.

## Current state (baseline)

- `PDBAligner` (aligner.py) supports modes `auto`, `seq_guided`, `seq_free_*`, `flexible`,
  plus `align_ensemble`, `batch_align`, chain-similarity matrix, binder-target finding.
- `AlignmentResult` is **stateless** — lazily recomputes from `_chosen`/`_seqguided`/
  `_seqfree` dicts and holds a live `gemmi` structure. Already has `plot_rmsd`,
  `get_rmsd_df`, TM-score, PyMOL/ChimeraX export.
- `EnsembleResult` has `plot_pca`, `plot_dendrogram`, `cluster`, `summary`.
- CLI (`__main__.py`) is thin: required `--ref/--mob` flags, prints RMSD/TM-score only.
- `core.py` has `compute_chain_similarity_matrix`, `_extract_ca_infos`, `_kabsch`,
  `pick_best_overall` (coverage-weighted score `n_pairs / (1 + (rmsd/3)^2)`).
- In today's `auto`, multi-chain is handled crudely: sequences are concatenated in file
  order with no correspondence/permutation logic.

## Architecture — six components

### 1. Chain correspondence — `core.match_chains()` (new, pure)

**Signature (intent):**
`match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct, ref_chains, mob_chains, ...) -> ChainMapping`

`ChainMapping` is a small dataclass: `pairs: list[(ref_chain, mob_chain, identity, score)]`,
`unmatched_ref: list[str]`, `unmatched_mob: list[str]`.

Algorithm:
- Build the ref×mob similarity matrix via `compute_chain_similarity_matrix`.
- **Heteromers:** solve the assignment problem with `scipy.optimize.linear_sum_assignment`
  (Hungarian) on the identity matrix → globally optimal 1:1 pairing.
- **Homomultimers / near-identical chains** (sequence cannot disambiguate — detected when
  multiple candidate pairings tie within a small identity tolerance): superposition-based
  permutation refinement. Seed from the Hungarian pairing, superimpose globally with
  `_kabsch`, then iteratively reassign chains by post-superposition proximity of chain
  centroids (ICP-style), keeping the mapping with the best coverage-weighted RMSD.
- **Combinatorial cap:** if `n_chains > MAX_PERMUTE_CHAINS` (default 12), skip exhaustive
  permutation enumeration and keep only the iterative centroid refinement.
- Pure function: no file I/O, fully unit-testable. Depends on `compute_chain_similarity_matrix`,
  `_extract_ca_infos`, `_kabsch`.

### 2. Multi-chain alignment — `core.align_multichain()` (new orchestration)

Uses the `ChainMapping` to build two candidate superpositions and pick the winner with the
same coverage-weighted score used by `pick_best_overall`:

- **global** — concatenate CA atoms across all mapped chain-pairs; one `_kabsch` transform
  (whole-complex fit).
- **local** — the single best-scoring chain-pair aligned alone.

Returns a structured result recording:
- `strategy`: `"global"` | `"local"` | `"single"` (single-chain inputs).
- `chain_mapping`: the `ChainMapping`.
- `per_chain`: per-mapped-chain RMSD under the chosen transform.
- The rotation/translation, matched pairs, and per-residue distances (same shape the rest of
  the code already consumes, so downstream `AlignmentResult` plumbing is reused).

Single-chain inputs bypass this entirely and preserve today's exact code path.

### 3. `align()` integration

`mode="auto"` becomes chain-aware:
- If both sides have exactly one active chain → existing behavior (unchanged).
- If multi-chain → run `match_chains` then `align_multichain`.
- New `strategy` parameter on `align(...)`: `"auto"` (default), `"global"`, `"local"`.
  `"auto"` lets `align_multichain` choose; `"global"`/`"local"` force the candidate.

Existing explicit modes (`seq_guided`, `seq_free_*`, `flexible`) are untouched.

### 4. Data-rich result object — `AlignmentResult`

Make the result carry all computed data as populated attributes (keeping existing lazy
properties for back-compat):

- **Attributes:** `stats` (full metrics dict), `chain_mapping`, `strategy`,
  `per_chain` (DataFrame), `rmsd_df` (per-residue DataFrame), `ref_coords` /
  `mob_coords_aligned` (numpy), `rotation` / `translation`, `sequence_alignment`.
- **Reporting:** `summary_stats() -> dict` (chosen method, strategy, chain mapping, overall
  RMSD, TM-score for ref/mob/min, GDT-TS, n_aligned, coverage %, per-chain table, sequence
  identity, domains if flexible). `report(fmt="text"|"json") -> str` — text form is the
  pretty terminal block; JSON for scripting. `to_dict()` / `to_json()`.
- **Persistence:** `result.save("run.npz")` / `AlignmentResult.load("run.npz")` persists all
  numeric data + metadata so the result can be replotted / re-reported **without the original
  files or gemmi**. Aligned-structure export stays separate via `save_aligned_pdb`.
- Plot/export methods read from held attributes, so a loaded result plots identically.
- `__repr__` shows strategy + RMSD + TM.

### 5. Nature-journal plot style — `pdb_align/plotstyle.py` (new)

Shared style applied to every figure:
- Sans-serif (Helvetica → Arial → DejaVu fallback), ~7pt base text, thin axes (0.75pt),
  **only left + bottom spines**, outward ticks, no gridlines.
- Nature column widths: single ≈ 89 mm, double ≈ 183 mm; 300+ DPI; tight layout.
- Colorblind-safe categorical palette (Okabe–Ito) for chains; perceptually-uniform
  sequential map (viridis/magma) for RMSD coloring; panel-label helper (**a**, **b**).
- `apply_nature_style()` context manager + `palette` constants.
- Refactor existing `plot_rmsd`, `plot_pca`, `plot_dendrogram` to use it.
- New `plot_summary()` — compact multi-panel figure (per-residue RMSD + per-chain RMSD bar +
  score box) for reports.
- Follows the repo `dataviz` design guidance when writing chart code.

### 6. CLI rewrite — `__main__.py`

```
pdb_align REF MOB [options]
```
- Positional `ref` `mob` (files or `pdb:1ABC` / `af:P12345`). `--ref` / `--mob` kept as
  optional aliases for back-compat.
- **Default = stats only, no files written.** Prints `report()` text (auto strategy, chain
  mapping, RMSD/TM/GDT, top RMSD peaks).
- Opt-in outputs (nothing written unless a flag is present):
  - `-o/--out FILE` — aligned structure (`.pdb`/`.cif`).
  - `--plot [FILE]` — per-residue RMSD plot; saves FILE, or with `--show` opens a window;
    headless-safe (`Agg` backend unless `--show`).
  - `--report FILE` — write the text/JSON report.
  - `--json` — emit machine-readable report to stdout.
  - `--csv FILE` — per-residue RMSD table.
- Passthrough flags: `--ref-chains`, `--mob-chains`, `--mode`, `--strategy {auto,global,local}`,
  `--atoms`, `--min-plddt`, `-v/--verbose`.

## Data flow

```
CLI (positional ref/mob)
  → PDBAligner.add_reference / add_mobile
  → align(mode="auto", strategy=...)
      → single-chain: existing path
      → multi-chain: match_chains → align_multichain
  → AlignmentResult (carries stats / chain_mapping / strategy / per_chain / coords)
  → report()  → terminal (always)
  → files only if -o / --plot / --report / --csv given
```

Python parallel: `r = pdb_align.align("a.pdb", "b.pdb"); r.plot_rmsd(); r.save("r.npz")`.

## Public API additions

- `pdb_align.core.match_chains`, `pdb_align.core.align_multichain`, `ChainMapping`.
- `AlignmentResult`: `.stats`, `.chain_mapping`, `.strategy`, `.per_chain`,
  `.summary_stats()`, `.report()`, `.to_dict()`, `.to_json()`, `.save()`, `.load()`,
  `.plot_summary()`.
- `PDBAligner.align(strategy=...)`.
- `pdb_align.plotstyle`: `apply_nature_style()`, `palette`.
- CLI: positional invocation + reporting/plot flags.

## Error handling

- No chain correspondence found → clear error listing the similarity matrix, suggesting
  explicit `--ref-chains/--mob-chains`.
- `--strategy local` with no viable single pair, or `global` with no mapping → explicit error.
- Headless plotting without a display and without `--show` → save silently; `--show` with no
  display → warn and fall back to saving.
- `AlignmentResult.load` on a mismatched/old file → clear version error.

## Testing

- `match_chains`: heteromer Hungarian correctness; homodimer with swapped chains picks the
  right permutation; unmatched-chain handling; combinatorial cap path.
- `align_multichain`: global-vs-local selection on a synthetic 2-chain case where local
  clearly wins; per-chain RMSD values; strategy override.
- `AlignmentResult`: `save`/`load` round-trip reproduces `stats` and replots byte-stable data;
  `report(fmt="json")` shape; `summary_stats` keys.
- `plotstyle`: applies headless without error; `plot_summary` returns a Figure.
- CLI: default writes no files; `-o`/`--plot`/`--report`/`--csv` produce exactly the
  requested artifacts; positional + remote-ID parsing; `--json` stdout shape; `--ref/--mob`
  back-compat aliases.
- Back-compat: single-chain `auto` output unchanged; existing tests still pass.

## Rollout / phasing (single plan, ordered)

1. `plotstyle.py` + refactor existing plots (low risk, independent).
2. `AlignmentResult` data-rich attributes + `summary_stats`/`report`/`save`/`load`.
3. `core.match_chains` + `core.align_multichain` (pure, test-first).
4. `align(strategy=...)` integration for multi-chain auto.
5. CLI rewrite wiring it all together.
