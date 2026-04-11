# pdb_align Expansion Design

**Date:** 2026-04-11  
**Scope:** Core alignment capabilities (B) + Python API ergonomics (C)  
**Approach:** Incremental enhancement — all additions are additive, no breaking changes

---

## Goals

1. Make the Python API as easy to use as PyMOL `align` or ChimeraX `matchmaker` — one-liner for the common case
2. Support multi-structure (ensemble) alignment for cryo-EM heterogeneous refinement datasets (10–50 models)
3. Add flexible/domain alignment mode for structures that are too divergent for rigid-body alignment
4. Enable clustering and PCA of ensembles to reveal conformational landscapes
5. Improve performance via structure caching and extended numba JIT coverage

---

## Architecture

All changes are additive layers on top of existing `PDBAligner` + `AlignmentResult`. No existing public API changes.

### New public surface

| Addition | Where |
|---|---|
| `pdb_align.align(ref, mob, **kwargs)` | `pdb_align/__init__.py` |
| `PDBAligner.align(mode="flexible", ...)` | `pdb_align/aligner.py` |
| `PDBAligner.align_ensemble(mob_list, ...)` | `pdb_align/aligner.py` |
| `DomainResult` dataclass | `pdb_align/aligner.py` |
| `EnsembleResult` class | `pdb_align/aligner.py` |
| `_detect_hinges()` function | `pdb_align/core.py` |

---

## Section 1: Top-Level Convenience Function

Add `pdb_align.align()` to `__init__.py` as a zero-setup one-liner:

```python
import pdb_align

result = pdb_align.align(
    "pdb:8UUP", "af:P00533",
    chains_ref=["A"], chains_mob=["A"],
    mode="auto",
    atoms="CA"
)
print(result.rmsd, result.tm_score)
```

Internally instantiates a `PDBAligner`, calls `add_reference(ref, chains=chains_ref)`, `add_mobile(mob, chains=chains_mob)`, and `align(**kwargs)`, returns the `AlignmentResult`. `chains_ref` and `chains_mob` are consumed by the facade; all other kwargs are forwarded to `align()`.

---

## Section 2: Flexible Domain Alignment

New `mode="flexible"` option in `PDBAligner.align()`.

### Algorithm

1. Run a full rigid-body alignment using existing `auto` mode to get an initial superposition
2. Compute per-residue Cα RMSD
3. Apply a sliding window (width 15 residues) across the chain; flag positions where the local mean RMSD exceeds a threshold (default 3.0 Å) as hinge candidates
4. Split the chain at each hinge; merge segments shorter than `domain_min_residues` (default 30 residues) into their nearest neighbor
5. Re-run Kabsch independently on each segment
6. Store each segment as a `DomainResult`

### `DomainResult` dataclass

```python
@dataclass
class DomainResult:
    domain_id: int
    chain_id: str
    residue_start: int       # sequence number of first residue
    residue_end: int         # sequence number of last residue
    n_residues: int
    rmsd: float
    rotation: np.ndarray     # (3,3)
    translation: np.ndarray  # (3,)
```

### `AlignmentResult` additions (flexible mode only)

```python
result.domains          # List[DomainResult] — None when mode != "flexible"
result.rmsd             # weighted-average RMSD across all domains (by n_residues)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `domain_min_residues` | 30 | Minimum residues for a segment to be kept as its own domain |
| `hinge_threshold` | 3.0 | Local RMSD (Å) above which a position is a hinge candidate |
| `hinge_window` | 15 | Sliding-window width for local RMSD smoothing |

---

## Section 3: Ensemble Alignment

New method `PDBAligner.align_ensemble()`:

```python
aligner.add_reference("pdb:8UUP", chains=["A"])

ens = aligner.align_ensemble(
    mob_list=["model_001.pdb", "model_002.pdb", ...],  # paths or pdb:/af: IDs
    mode="auto",        # same modes as align()
    atoms="CA",
    workers=4,          # parallel workers
    out_dir=None,       # optional: save aligned PDBs here
)
```

- Reuses the existing `ProcessPoolExecutor` infrastructure from `batch_align`
- Reference structure is parsed once and cached before workers are spawned
- Returns an `EnsembleResult`

### `EnsembleResult` class

```python
class EnsembleResult:
    results: List[AlignmentResult]   # one per model, in input order
    labels: List[str]                # filenames / IDs

    def summary(self) -> pd.DataFrame:
        """DataFrame with columns: model, rmsd, tm_score, gdt_ts, n_aligned"""

    def rmsd_matrix(self) -> pd.DataFrame:
        """NxN pairwise RMSD DataFrame (models vs models)"""

    def cluster(self, n_clusters: int = None) -> np.ndarray:
        """
        K-means clustering on per-residue RMSD vectors.
        If n_clusters is None, auto-selects via elbow method (k=2..8).
        Returns integer label array of length N.
        Stores labels internally for use by plot_pca().
        """

    def plot_pca(
        self,
        color_by: str = "cluster",  # "cluster" | "rmsd" | "tm_score"
        save_path: str = None,
    ) -> "matplotlib.figure.Figure":
        """
        2D PCA of per-residue RMSD vectors.
        Points colored by cluster label, RMSD, or TM-score.
        If color_by="cluster" but cluster() has not been called, falls back to
        coloring by RMSD and emits a warning.
        """

    def plot_dendrogram(self, save_path: str = None) -> "matplotlib.figure.Figure":
        """Hierarchical clustering dendrogram on pairwise RMSD matrix."""
```

---

## Section 4: Performance

### Structure cache

`PDBAligner` gains an internal `_struct_cache: Dict[str, gemmi.Structure]` dict. `_parse_path()` in `core.py` checks the cache before parsing. The reference is always cached after `add_reference()`, so ensemble runs never re-parse it.

Cache is instance-scoped (cleared when `PDBAligner` is garbage-collected). No cross-instance sharing — this avoids stale state issues in long-running scripts.

### Extended numba JIT

Apply `@jit(nopython=True, cache=True)` to:
- `_pairwise_dists()` in `core.py` (used by CAD score, already a hot path)
- The new `_detect_hinges()` sliding-window loop

---

## Section 5: Streamlit App (deferred — Priority A, after B+C)

Once the library additions are stable, the Streamlit app will be updated to:
- Expose the flexible alignment mode via a toggle
- Add an ensemble upload widget (multi-file) feeding `align_ensemble()`
- Show PCA scatter and dendrogram plots in new tabs
- Verify all dependencies are Streamlit Cloud compatible (especially numba — may need a fallback build)

---

## Files Changed

| File | Change type |
|---|---|
| `pdb_align/__init__.py` | Add `align()` convenience function |
| `pdb_align/aligner.py` | Add `align_ensemble()`, flexible domain logic, `DomainResult`, `EnsembleResult` |
| `pdb_align/core.py` | Add `_detect_hinges()`, extend numba coverage |
| `pyproject.toml` | No new deps needed (scikit-learn, scipy already present) |
| `tests/test_core.py` | Add tests for `_detect_hinges()` |
| `tests/test_aligner.py` | New file: tests for `align_ensemble()`, `EnsembleResult`, flexible mode |

---

## Out of Scope

- New remote structure sources (ESMFold, ModelArchive) — deferred
- Full Streamlit Cloud deployment — deferred to Phase A
- Fluent/pipeline API rewrite — explicitly rejected in design
- RNA / nucleic acid support
