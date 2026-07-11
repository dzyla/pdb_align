# Intelligent Alignment CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pdb_align` the go-to structure-comparison tool: intelligent multi-chain alignment (optimal chain correspondence + auto global/local strategy), a data-rich self-contained result object, Nature-journal-style plots, and a zero-config CLI with minimal default output.

**Architecture:** Add two pure `core` functions (`match_chains`, `align_multichain`) that `align(mode="auto")` dispatches to when structures have >1 chain. Enrich `AlignmentResult` to hold all computed data as attributes with `report()`/`summary_stats()`/`save()`/`load()`. Add a shared `plotstyle` module used by all figures. Rewrite the CLI as a positional, stats-only-by-default front end.

**Tech Stack:** Python 3.12, numpy, pandas, scipy (`linear_sum_assignment`), gemmi, BioPython, matplotlib/seaborn, numba (optional), pytest.

## Global Constraints

- Single-chain alignment behavior MUST remain byte-for-byte unchanged (existing tests pass).
- No new required dependencies beyond what's already used; `scipy` is already a dependency.
- Default CLI run writes NO files — stats to terminal only. Files only on explicit flags.
- Nature style = matplotlib rcParams + palette + spine/width conventions; font fallback Helvetica → Arial → DejaVu. No bundled font file.
- Chain permutation search capped by `MAX_PERMUTE_CHAINS = 12`.
- Coverage-weighted score for strategy/candidate selection is `n_pairs / (1 + (rmsd/3.0)**2)` — the same formula `pick_best_overall` uses.
- All plotting must work headless (`matplotlib.use("Agg")` unless `--show`).

## File Structure

- Create `pdb_align/plotstyle.py` — Nature-style rcParams, palette, `apply_nature_style()`, panel-label helper.
- Create `pdb_align/chains.py` — `ChainMapping` dataclass, `match_chains()`, `align_multichain()`, `MultiChainResult` dataclass. (Kept out of the already-large `core.py`; imported by `core`/`aligner`.)
- Modify `pdb_align/aligner.py` — enrich `AlignmentResult` (attributes, `summary_stats`, `report`, `to_dict`, `to_json`, `save`, `load`, `plot_summary`), refactor plots to use `plotstyle`, add `strategy` param + multi-chain dispatch in `PDBAligner.align`.
- Modify `pdb_align/__main__.py` — full CLI rewrite.
- Modify `pdb_align/__init__.py` — export `match_chains`, `align_multichain`, `ChainMapping`, `apply_nature_style`.
- Tests: `tests/test_plotstyle.py`, `tests/test_result_data.py`, `tests/test_chain_matching.py`, `tests/test_multichain_align.py`, `tests/test_cli.py`.

---

### Task 1: Nature-style plot module

**Files:**
- Create: `pdb_align/plotstyle.py`
- Test: `tests/test_plotstyle.py`

**Interfaces:**
- Consumes: nothing (matplotlib only).
- Produces:
  - `PALETTE: list[str]` — 8 Okabe–Ito hex colors.
  - `apply_nature_style() -> contextlib.AbstractContextManager` — context manager applying rcParams.
  - `nature_figure(width="single"|"double", height=None) -> (fig, ax)` — figure sized to Nature column widths (single=89mm, double=183mm) at 300 dpi.
  - `panel_label(ax, letter)` — places a bold panel label (**a**) at the top-left.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plotstyle.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pdb_align import plotstyle


def test_palette_is_colorblind_safe_hexes():
    assert len(plotstyle.PALETTE) >= 8
    assert all(c.startswith("#") and len(c) == 7 for c in plotstyle.PALETTE)


def test_apply_nature_style_sets_rcparams():
    with plotstyle.apply_nature_style():
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["xtick.direction"] == "out"


def test_nature_figure_single_column_width_mm():
    fig, ax = plotstyle.nature_figure(width="single")
    w_in, _ = fig.get_size_inches()
    assert abs(w_in - 89 / 25.4) < 0.05
    plt.close(fig)


def test_panel_label_adds_text():
    fig, ax = plotstyle.nature_figure()
    plotstyle.panel_label(ax, "a")
    assert any(t.get_text() == "a" for t in ax.texts)
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plotstyle.py -v`
Expected: FAIL — `ModuleNotFoundError: pdb_align.plotstyle`.

- [ ] **Step 3: Write minimal implementation**

```python
# pdb_align/plotstyle.py
"""Nature-journal-style matplotlib defaults for pdb_align figures."""
from contextlib import contextmanager

# Okabe-Ito colorblind-safe categorical palette.
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

# Nature column widths in millimetres.
_WIDTH_MM = {"single": 89.0, "double": 183.0}
_MM_PER_INCH = 25.4

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@contextmanager
def apply_nature_style():
    """Context manager applying Nature-style rcParams, restoring on exit."""
    import matplotlib.pyplot as plt
    with plt.rc_context(_RCPARAMS):
        yield


def nature_figure(width: str = "single", height: float = None):
    """Return (fig, ax) sized to a Nature column width at 300 dpi."""
    import matplotlib.pyplot as plt
    w_in = _WIDTH_MM.get(width, _WIDTH_MM["single"]) / _MM_PER_INCH
    h_in = height if height is not None else w_in * 0.62
    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(w_in, h_in))
    return fig, ax


def panel_label(ax, letter: str):
    """Place a bold panel label at the axis top-left (outside the frame)."""
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plotstyle.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Refactor `plot_rmsd` to use the style**

In `pdb_align/aligner.py`, replace the body of `AlignmentResult.plot_rmsd` (currently ~lines 458-487) so the `style == "scientific"` branch wraps drawing in `plotstyle.apply_nature_style()` and uses `plotstyle.PALETTE` for the `hue="Chain"` colors. Keep the same signature and file-saving behavior. Add near the top of the method:

```python
from . import plotstyle
```
and wrap the plotting block:
```python
with plotstyle.apply_nature_style():
    fig, ax = plt.subplots(figsize=(89/25.4*2, 89/25.4*1.1))
    sns.lineplot(data=df, x=df.index, y="RMSD", hue="Chain",
                 palette=plotstyle.PALETTE[:df["Chain"].nunique()],
                 marker='o', markersize=3, linewidth=0.9, ax=ax)
    # ... keep existing tick / label / save logic ...
```

- [ ] **Step 6: Run existing plot tests + new tests**

Run: `pytest tests/test_plotstyle.py tests/ -k "plot" -v`
Expected: PASS; no regressions.

- [ ] **Step 7: Commit**

```bash
git add pdb_align/plotstyle.py tests/test_plotstyle.py pdb_align/aligner.py
git commit -m "feat: add Nature-style plotstyle module and apply to plot_rmsd"
```

---

### Task 2: Data-rich AlignmentResult — stats, report, dict/json

**Files:**
- Modify: `pdb_align/aligner.py` (class `AlignmentResult`)
- Test: `tests/test_result_data.py`

**Interfaces:**
- Consumes: existing `AlignmentResult` properties (`rmsd`, `tm_score`, `get_tm_score`, `get_rmsd_df`, `_chosen`).
- Produces on `AlignmentResult`:
  - `.strategy: str` — `"single"` by default (set by Task 6 for multi-chain).
  - `.chain_mapping` — `None` by default (set by Task 6).
  - `.per_chain -> pandas.DataFrame` — columns `chain_ref, chain_mob, n_residues, rmsd`; empty DF when unavailable.
  - `.summary_stats() -> dict`
  - `.report(fmt="text") -> str` (`fmt` in `{"text","json"}`)
  - `.to_dict() -> dict`, `.to_json(indent=2) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_result_data.py
import json
import pytest
import pdb_align

REF = "tests/data/ref.pdb"
MOB = "tests/data/mob.pdb"


@pytest.fixture(scope="module")
def result():
    return pdb_align.align(REF, MOB)


def test_summary_stats_has_core_keys(result):
    s = result.summary_stats()
    for key in ("method", "strategy", "rmsd", "tm_score", "n_aligned"):
        assert key in s
    assert isinstance(s["rmsd"], (float, type(None)))


def test_strategy_defaults_to_single_chain(result):
    assert result.strategy in ("single", "global", "local")


def test_per_chain_is_dataframe(result):
    df = result.per_chain
    assert list(df.columns) == ["chain_ref", "chain_mob", "n_residues", "rmsd"]


def test_report_text_mentions_rmsd(result):
    text = result.report(fmt="text")
    assert "RMSD" in text


def test_report_json_roundtrips(result):
    payload = json.loads(result.report(fmt="json"))
    assert payload["strategy"] == result.strategy
```

Note: create small `tests/data/ref.pdb` and `tests/data/mob.pdb` if not present — two single-chain structures (reuse any fixture already in `tests/`; check `tests/` first and point `REF`/`MOB` at existing fixtures rather than adding files).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_result_data.py -v`
Expected: FAIL — `AttributeError: 'AlignmentResult' object has no attribute 'summary_stats'`.

- [ ] **Step 3: Implement the methods**

In `AlignmentResult.__init__`, after existing assignments add:
```python
self.strategy = "single"
self.chain_mapping = None
self._per_chain = None  # optional DataFrame set by multi-chain path
```

Add methods to the class:
```python
@property
def per_chain(self):
    import pandas as pd
    if self._per_chain is not None:
        return self._per_chain
    return pd.DataFrame(columns=["chain_ref", "chain_mob", "n_residues", "rmsd"])

def summary_stats(self) -> dict:
    gdt = None
    sg = self._chosen.get("seqguided")
    sf = self._chosen.get("seqfree")
    if sg and sg.get("si"):
        gdt = sg["si"].get("gdt_ts")
    elif sf:
        gdt = getattr(sf, "gdt_ts", None)
    try:
        n_aligned = len(self.get_rmsd_df())
    except Exception:
        n_aligned = None
    l_ref = sum(self.ref_lens.values()) or None
    coverage = (n_aligned / l_ref * 100.0) if (n_aligned and l_ref) else None
    mapping = None
    if self.chain_mapping is not None:
        mapping = [
            {"ref": p[0], "mob": p[1], "identity": round(float(p[2]), 1)}
            for p in self.chain_mapping.pairs
        ]
    return {
        "method": self._chosen.get("name"),
        "strategy": self.strategy,
        "reason": self._chosen.get("reason"),
        "rmsd": self.rmsd,
        "tm_score": self.tm_score,
        "tm_score_min": self.get_tm_score("min"),
        "gdt_ts": gdt,
        "n_aligned": n_aligned,
        "coverage_pct": coverage,
        "chain_mapping": mapping,
        "ref_file": self.ref_file,
        "mob_file": self.mob_file,
    }

def to_dict(self) -> dict:
    d = self.summary_stats()
    d["per_chain"] = self.per_chain.to_dict(orient="records")
    return d

def to_json(self, indent: int = 2) -> str:
    import json
    return json.dumps(self.to_dict(), indent=indent, default=float)

def report(self, fmt: str = "text") -> str:
    if fmt == "json":
        return self.to_json()
    if fmt != "text":
        raise ValueError("fmt must be 'text' or 'json'")
    s = self.summary_stats()
    def fmt_num(v, spec):
        return format(v, spec) if v is not None else "n/a"
    lines = []
    lines.append("=" * 52)
    lines.append(" pdb_align — structural comparison")
    lines.append("=" * 52)
    import os
    lines.append(f" Reference : {os.path.basename(s['ref_file'])}")
    lines.append(f" Mobile    : {os.path.basename(s['mob_file'])}")
    lines.append(f" Method    : {s['method']}  (strategy: {s['strategy']})")
    lines.append("-" * 52)
    lines.append(f" RMSD          : {fmt_num(s['rmsd'], '.3f')} A")
    lines.append(f" TM-score      : {fmt_num(s['tm_score'], '.4f')}")
    lines.append(f" GDT-TS        : {fmt_num(s['gdt_ts'], '.2f')}")
    lines.append(f" Aligned res   : {s['n_aligned'] if s['n_aligned'] is not None else 'n/a'}")
    lines.append(f" Coverage      : {fmt_num(s['coverage_pct'], '.1f')} %")
    if s["chain_mapping"]:
        lines.append("-" * 52)
        lines.append(" Chain mapping (ref -> mob, %id):")
        for m in s["chain_mapping"]:
            lines.append(f"   {m['ref']} -> {m['mob']}  ({m['identity']:.1f}%)")
    df = self.per_chain
    if not df.empty:
        lines.append("-" * 52)
        lines.append(" Per-chain RMSD:")
        for _, r in df.iterrows():
            lines.append(f"   {r['chain_ref']}->{r['chain_mob']}: "
                         f"{r['rmsd']:.3f} A ({int(r['n_residues'])} res)")
    lines.append("=" * 52)
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_result_data.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py tests/test_result_data.py
git commit -m "feat: data-rich AlignmentResult with summary_stats/report/to_dict"
```

---

### Task 3: AlignmentResult persistence — save / load

**Files:**
- Modify: `pdb_align/aligner.py` (class `AlignmentResult`)
- Test: `tests/test_result_data.py` (append)

**Interfaces:**
- Consumes: `.summary_stats()`, `.get_rmsd_df()`, `.get_aligned_coords()`, `.rotation_matrix`, `.translation_vector`, `.per_chain`.
- Produces:
  - `AlignmentResult.save(path)` — writes an `.npz` with a versioned schema (`_SAVE_VERSION = 1`).
  - `AlignmentResult.load(path) -> LoadedResult` — a lightweight object exposing `.summary_stats()`, `.report()`, `.get_rmsd_df()`, `.plot_rmsd()`, `.plot_summary()`, `.rmsd`, `.tm_score`, `.strategy` — WITHOUT needing gemmi or the original files.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_result_data.py
def test_save_load_roundtrip_reproduces_stats(result, tmp_path):
    p = tmp_path / "run.npz"
    result.save(str(p))
    from pdb_align import AlignmentResult
    loaded = AlignmentResult.load(str(p))
    assert loaded.strategy == result.strategy
    a, b = result.summary_stats(), loaded.summary_stats()
    for key in ("rmsd", "tm_score", "n_aligned", "strategy"):
        assert (a[key] is None and b[key] is None) or \
               (abs((a[key] or 0) - (b[key] or 0)) < 1e-6) or a[key] == b[key]


def test_loaded_result_replots_without_original_files(result, tmp_path):
    p = tmp_path / "run.npz"
    result.save(str(p))
    from pdb_align import AlignmentResult
    loaded = AlignmentResult.load(str(p))
    out = tmp_path / "r.png"
    loaded.plot_rmsd(filename=str(out))
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_result_data.py -k save_load -v`
Expected: FAIL — `AttributeError: type object 'AlignmentResult' has no attribute 'load'`.

- [ ] **Step 3: Implement save/load + LoadedResult**

Add a class attribute and methods to `AlignmentResult`:
```python
_SAVE_VERSION = 1

def save(self, path: str):
    """Persist all computed data to a versioned .npz (no gemmi needed to reload)."""
    import numpy as np, json
    df = self.get_rmsd_df()
    per_chain = self.per_chain
    meta = self.summary_stats()
    meta["_save_version"] = self._SAVE_VERSION
    coords = self.get_aligned_coords()
    ref_c, mob_c = (coords if coords is not None else (np.empty((0, 3)), np.empty((0, 3))))
    np.savez_compressed(
        path,
        meta_json=json.dumps(meta, default=float),
        rmsd_residues=np.array(df["Residue"].tolist(), dtype=object),
        rmsd_chains=np.array(df["Chain"].tolist(), dtype=object),
        rmsd_values=df["RMSD"].to_numpy(dtype=float),
        per_chain_json=per_chain.to_json(orient="records"),
        ref_coords=np.asarray(ref_c, dtype=float),
        mob_coords=np.asarray(mob_c, dtype=float),
    )

@staticmethod
def load(path: str) -> "LoadedResult":
    return LoadedResult._from_npz(path)
```

Add a module-level `LoadedResult` class (near `AlignmentResult`):
```python
class LoadedResult:
    """A replayable, gemmi-free view of a saved AlignmentResult."""

    def __init__(self, meta, rmsd_df, per_chain, ref_coords, mob_coords):
        self._meta = meta
        self._rmsd_df = rmsd_df
        self._per_chain = per_chain
        self.ref_coords = ref_coords
        self.mob_coords_aligned = mob_coords
        self.strategy = meta.get("strategy")
        self.rmsd = meta.get("rmsd")
        self.tm_score = meta.get("tm_score")

    @classmethod
    def _from_npz(cls, path):
        import numpy as np, json, pandas as pd
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        if meta.get("_save_version") != AlignmentResult._SAVE_VERSION:
            raise ValueError(
                f"Unsupported save version {meta.get('_save_version')}; "
                f"expected {AlignmentResult._SAVE_VERSION}."
            )
        rmsd_df = pd.DataFrame({
            "Residue": list(z["rmsd_residues"]),
            "Chain": list(z["rmsd_chains"]),
            "RMSD": z["rmsd_values"],
        })
        per_chain = pd.read_json(str(z["per_chain_json"]), orient="records")
        return cls(meta, rmsd_df, per_chain, z["ref_coords"], z["mob_coords"])

    def summary_stats(self):
        return dict(self._meta)

    def get_rmsd_df(self, on="reference"):
        return self._rmsd_df.copy()

    @property
    def per_chain(self):
        return self._per_chain

    # Reuse the exact rendering logic from AlignmentResult by delegation.
    report = AlignmentResult.report
    plot_rmsd = AlignmentResult.plot_rmsd
    plot_summary = AlignmentResult.plot_summary  # defined in Task 4
```

Note: `report` on `LoadedResult` calls `self.per_chain` and `self.summary_stats()` only, both provided — so delegation works. `plot_summary` is added in Task 4; if executing Task 3 before Task 4, temporarily drop the `plot_summary` line and add it in Task 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_result_data.py -k save_load -v` then `pytest tests/test_result_data.py -k replots -v`
Expected: PASS.

- [ ] **Step 5: Export and commit**

Add `LoadedResult` to `pdb_align/__init__.py` imports/`__all__`.
```bash
git add pdb_align/aligner.py pdb_align/__init__.py tests/test_result_data.py
git commit -m "feat: AlignmentResult.save/load with gemmi-free LoadedResult"
```

---

### Task 4: plot_summary multi-panel figure

**Files:**
- Modify: `pdb_align/aligner.py` (class `AlignmentResult`)
- Test: `tests/test_plotstyle.py` (append)

**Interfaces:**
- Consumes: `plotstyle`, `.get_rmsd_df()`, `.per_chain`, `.summary_stats()`.
- Produces: `AlignmentResult.plot_summary(filename=None, show=False) -> matplotlib.figure.Figure`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotstyle.py
def test_plot_summary_returns_figure(tmp_path):
    import pdb_align
    r = pdb_align.align("tests/data/ref.pdb", "tests/data/mob.pdb")
    out = tmp_path / "summary.png"
    fig = r.plot_summary(filename=str(out))
    assert fig is not None
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plotstyle.py -k summary -v`
Expected: FAIL — `AttributeError: ... no attribute 'plot_summary'`.

- [ ] **Step 3: Implement plot_summary**

```python
def plot_summary(self, filename: str = None, show: bool = False):
    """Compact multi-panel Nature-style summary: per-residue RMSD + per-chain bar + scores."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plotstyle
    df = self.get_rmsd_df()
    stats = self.summary_stats()
    with plotstyle.apply_nature_style():
        fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 183/25.4*0.4))
        ax0, ax1 = axes
        if not df.empty:
            for i, (chain, g) in enumerate(df.groupby("Chain")):
                ax0.plot(range(len(g)), g["RMSD"], lw=0.9,
                         color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)], label=str(chain))
            ax0.set_xlabel("Residue index"); ax0.set_ylabel(r"C$\alpha$ deviation ($\AA$)")
            if df["Chain"].nunique() > 1:
                ax0.legend(frameon=False)
        plotstyle.panel_label(ax0, "a")
        pc = self.per_chain
        if not pc.empty:
            labels = [f"{a}->{b}" for a, b in zip(pc["chain_ref"], pc["chain_mob"])]
            ax1.bar(labels, pc["rmsd"], color=plotstyle.PALETTE[0])
            ax1.set_ylabel(r"RMSD ($\AA$)"); ax1.tick_params(axis="x", rotation=45)
        else:
            txt = (f"RMSD {stats['rmsd']:.2f} A\n"
                   f"TM {stats['tm_score']:.3f}" if stats['rmsd'] is not None else "n/a")
            ax1.text(0.5, 0.5, txt, ha="center", va="center", transform=ax1.transAxes)
            ax1.axis("off")
        plotstyle.panel_label(ax1, "b")
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
        if show:
            plt.show()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plotstyle.py -k summary -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py tests/test_plotstyle.py
git commit -m "feat: add plot_summary multi-panel Nature-style figure"
```

---

### Task 5: Chain correspondence — match_chains

**Files:**
- Create: `pdb_align/chains.py`
- Test: `tests/test_chain_matching.py`

**Interfaces:**
- Consumes: `core.compute_chain_similarity_matrix`, `core._extract_ca_infos`, `core._kabsch`.
- Produces:
  - `@dataclass ChainMapping`: `pairs: list[tuple[str, str, float, float]]` (ref_chain, mob_chain, identity_pct, coverage_score), `unmatched_ref: list[str]`, `unmatched_mob: list[str]`.
  - `MAX_PERMUTE_CHAINS = 12`.
  - `match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct, ref_chains, mob_chains) -> ChainMapping`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chain_matching.py
import numpy as np
from types import SimpleNamespace
from pdb_align.chains import match_chains, ChainMapping


def _seqrec(seq):
    return SimpleNamespace(seq=seq)


def test_heteromer_pairs_by_best_identity(monkeypatch):
    import pdb_align.chains as ch
    import pandas as pd
    # ref A~mob Y (identical), ref B~mob X (identical); file order is crossed.
    ref_seqs = {"A": _seqrec("AAAAKKKK"), "B": _seqrec("DDDDEEEE")}
    mob_seqs = {"X": _seqrec("DDDDEEEE"), "Y": _seqrec("AAAAKKKK")}
    id_mat = pd.DataFrame([[0.0, 100.0], [100.0, 0.0]],
                          index=["A", "B"], columns=["X", "Y"])
    monkeypatch.setattr(ch, "compute_chain_similarity_matrix",
                        lambda a, b: (id_mat, id_mat))
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A", "B"], ["X", "Y"])
    pairs = {(p[0], p[1]) for p in mapping.pairs}
    assert pairs == {("A", "Y"), ("B", "X")}


def test_unmatched_chains_reported(monkeypatch):
    import pdb_align.chains as ch
    import pandas as pd
    ref_seqs = {"A": _seqrec("AAAAKKKK")}
    mob_seqs = {"X": _seqrec("AAAAKKKK"), "Z": _seqrec("WWWWWWWW")}
    id_mat = pd.DataFrame([[100.0, 0.0]], index=["A"], columns=["X", "Z"])
    monkeypatch.setattr(ch, "compute_chain_similarity_matrix",
                        lambda a, b: (id_mat, id_mat))
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A"], ["X", "Z"])
    assert mapping.pairs[0][0] == "A" and mapping.pairs[0][1] == "X"
    assert "Z" in mapping.unmatched_mob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chain_matching.py -v`
Expected: FAIL — `ModuleNotFoundError: pdb_align.chains`.

- [ ] **Step 3: Implement match_chains (Hungarian core)**

```python
# pdb_align/chains.py
"""Chain correspondence and multi-chain superposition strategy selection."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from .core import compute_chain_similarity_matrix, _extract_ca_infos, _kabsch

MAX_PERMUTE_CHAINS = 12
_TIE_TOL = 5.0  # % identity within which chains are treated as indistinguishable


@dataclass
class ChainMapping:
    pairs: List[Tuple[str, str, float, float]] = field(default_factory=list)
    unmatched_ref: List[str] = field(default_factory=list)
    unmatched_mob: List[str] = field(default_factory=list)


def match_chains(ref_seqs, mob_seqs, ref_struct, mob_struct,
                 ref_chains, mob_chains) -> ChainMapping:
    """Optimal 1:1 chain correspondence via Hungarian assignment on % identity,
    refined by superposition for near-identical (homomultimer) chains."""
    from scipy.optimize import linear_sum_assignment

    r_seqs = {c: ref_seqs[c] for c in ref_chains if c in ref_seqs}
    m_seqs = {c: mob_seqs[c] for c in mob_chains if c in mob_seqs}
    r_ids, m_ids = list(r_seqs.keys()), list(m_seqs.keys())
    if not r_ids or not m_ids:
        return ChainMapping(unmatched_ref=r_ids, unmatched_mob=m_ids)

    id_mat, _ = compute_chain_similarity_matrix(r_seqs, m_seqs)
    mat = np.nan_to_num(np.asarray(id_mat, dtype=float), nan=0.0)

    # Hungarian maximises identity -> minimise negative identity.
    row_idx, col_idx = linear_sum_assignment(-mat)
    pairs = []
    used_r, used_m = set(), set()
    for ri, ci in zip(row_idx, col_idx):
        ident = float(mat[ri, ci])
        if ident <= 0.0:
            continue
        pairs.append((r_ids[ri], m_ids[ci], ident, ident))
        used_r.add(r_ids[ri]); used_m.add(m_ids[ci])

    mapping = ChainMapping(
        pairs=pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
    )

    # Refine homomultimers where sequence identity ties across candidates.
    if _needs_permutation_refinement(mat) and ref_struct is not None:
        mapping = _refine_by_superposition(
            mapping, ref_struct, mob_struct, r_ids, m_ids)
    return mapping


def _needs_permutation_refinement(mat: np.ndarray) -> bool:
    """True if any ref chain has >=2 mob candidates within _TIE_TOL % identity."""
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return False
    for row in mat:
        top = np.sort(row)[::-1]
        if top[0] > 0 and (top[0] - top[1]) <= _TIE_TOL:
            return True
    return False


def _refine_by_superposition(mapping, ref_struct, mob_struct, r_ids, m_ids):
    """Centroid-ICP refinement: superpose on current mapping, then reassign
    chains by post-superposition centroid proximity. Capped by MAX_PERMUTE_CHAINS."""
    from scipy.optimize import linear_sum_assignment
    if len(r_ids) > MAX_PERMUTE_CHAINS:
        return mapping  # keep Hungarian result on very large complexes

    def centroids(struct, ids):
        out = {}
        for c in ids:
            infos = _extract_ca_infos(struct, [c])
            if infos:
                out[c] = np.mean([i.coord for i in infos], axis=0)
        return out

    r_cen = centroids(ref_struct, r_ids)
    m_cen = centroids(mob_struct, m_ids)
    if len(mapping.pairs) < 3:
        return mapping

    # Superpose using current pairing's centroids.
    P = np.array([r_cen[a] for a, b, *_ in mapping.pairs if a in r_cen and b in m_cen])
    Q = np.array([m_cen[b] for a, b, *_ in mapping.pairs if a in r_cen and b in m_cen])
    if len(P) < 3:
        return mapping
    R, t, _ = _kabsch(P, Q)

    common_r = [c for c in r_ids if c in r_cen]
    common_m = [c for c in m_ids if c in m_cen]
    cost = np.zeros((len(common_r), len(common_m)))
    for i, a in enumerate(common_r):
        for j, b in enumerate(common_m):
            moved = R @ m_cen[b] + t
            cost[i, j] = np.linalg.norm(r_cen[a] - moved)
    ri, ci = linear_sum_assignment(cost)
    id_lookup = {(p[0], p[1]): p[2] for p in mapping.pairs}
    new_pairs = []
    for i, j in zip(ri, ci):
        a, b = common_r[i], common_m[j]
        ident = id_lookup.get((a, b), 0.0)
        new_pairs.append((a, b, ident, ident))
    used_r = {p[0] for p in new_pairs}
    used_m = {p[1] for p in new_pairs}
    return ChainMapping(
        pairs=new_pairs,
        unmatched_ref=[c for c in r_ids if c not in used_r],
        unmatched_mob=[c for c in m_ids if c not in used_m],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chain_matching.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add homomultimer refinement test**

```python
# append to tests/test_chain_matching.py
def test_homodimer_swapped_chains_refined_by_geometry():
    """Two identical chains; correct mapping must come from geometry, not sequence."""
    import gemmi
    from pdb_align.core import extract_sequences_and_lengths

    def _struct(coords_by_chain):
        st = gemmi.Structure(); model = gemmi.Model("1"); 
        for cname, coords in coords_by_chain.items():
            chain = gemmi.Chain(cname)
            for k, (x, y, z) in enumerate(coords, start=1):
                res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k, " ")
                at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
                res.add_atom(at); chain.add_residue(res)
            model.add_chain(chain)
        st.add_model(model); return st
    base = [(i, 0.0, 0.0) for i in range(5)]
    ref = _struct({"A": base, "B": [(x, 10.0, 0.0) for x, _, _ in base]})
    # mobile chains carry swapped names relative to geometry
    mob = _struct({"P": [(x, 10.0, 0.0) for x, _, _ in base], "Q": base})
    rs, _ = extract_sequences_and_lengths(ref, "ref")
    ms, _ = extract_sequences_and_lengths(mob, "mob")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["P", "Q"])
    pairs = {(p[0], p[1]) for p in mapping.pairs}
    # A (y=0) should map to Q (y=0); B (y=10) to P (y=10)
    assert pairs == {("A", "Q"), ("B", "P")}
```

Run: `pytest tests/test_chain_matching.py -v`
Expected: PASS (3 tests). If the homodimer sequences are too short for `compute_chain_similarity_matrix`, lengthen `base` to ~15 residues.

- [ ] **Step 6: Commit**

```bash
git add pdb_align/chains.py tests/test_chain_matching.py
git commit -m "feat: optimal chain correspondence via Hungarian + geometric refinement"
```

---

### Task 6: Multi-chain superposition — align_multichain + align() dispatch

**Files:**
- Modify: `pdb_align/chains.py` (add `align_multichain`, `MultiChainResult`)
- Modify: `pdb_align/aligner.py` (`PDBAligner.align` dispatch, `AlignmentResult` wiring)
- Test: `tests/test_multichain_align.py`

**Interfaces:**
- Consumes: `match_chains`, `ChainMapping`, `core._extract_ca_infos`, `core._kabsch`, `core.compute_gdt_ts`.
- Produces:
  - `@dataclass MultiChainResult`: `strategy: str`, `mapping: ChainMapping`, `rotation`, `translation`, `rmsd: float`, `pairs`, `ref_coords`, `mob_coords_aligned`, `per_chain: list[dict]` (keys `chain_ref, chain_mob, n_residues, rmsd`), `ref_infos`, `mob_infos`.
  - `align_multichain(ref_struct, mob_struct, mapping, strategy="auto", atoms="CA", min_b_factor=0.0, min_plddt=0.0) -> MultiChainResult`.
  - `PDBAligner.align(strategy="auto", ...)` dispatches to multi-chain when both sides have >1 active chain and `mode` is auto.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multichain_align.py
import numpy as np
import gemmi
from pdb_align.core import extract_sequences_and_lengths
from pdb_align.chains import match_chains, align_multichain


def _struct(coords_by_chain):
    st = gemmi.Structure(); model = gemmi.Model("1")
    for cname, coords in coords_by_chain.items():
        chain = gemmi.Chain(cname)
        for k, (x, y, z) in enumerate(coords, start=1):
            res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k, " ")
            at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
            res.add_atom(at); chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model); return st


def _two_chain(offset_b=0.0):
    a = [(i, 0.0, 0.0) for i in range(15)]
    b = [(i, 10.0 + offset_b, 0.0) for i in range(15)]
    return {"A": a, "B": b}


def test_global_strategy_superimposes_all_chains():
    ref = _struct(_two_chain(0.0))
    # mobile = ref rigidly translated by +5 in x
    mob = _struct({k: [(x + 5, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="global")
    assert res.strategy == "global"
    assert res.rmsd < 1e-6  # rigid translation recovered exactly
    assert len(res.per_chain) == 2


def test_local_beats_global_when_one_chain_diverges():
    ref = _struct(_two_chain(0.0))
    # chain B of mobile is badly displaced; chain A matches after rigid move
    mob = _struct({"A": [(x + 5, 0.0, 0.0) for x in range(15)],
                   "B": [(x + 5, 40.0, 30.0) for x in range(15)]})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="auto")
    assert res.strategy == "local"
    assert res.rmsd < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multichain_align.py -v`
Expected: FAIL — `ImportError: cannot import name 'align_multichain'`.

- [ ] **Step 3: Implement align_multichain**

```python
# append to pdb_align/chains.py
from dataclasses import dataclass as _dataclass

@_dataclass
class MultiChainResult:
    strategy: str
    mapping: "ChainMapping"
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    pairs: list
    ref_coords: np.ndarray
    mob_coords_aligned: np.ndarray
    per_chain: list
    ref_infos: list
    mob_infos: list


def _coverage_score(n_pairs: int, rmsd: float) -> float:
    return n_pairs / (1.0 + (rmsd / 3.0) ** 2)


def _paired_ca(ref_struct, mob_struct, mapping, atoms, min_b_factor, min_plddt):
    """Return matched CA coords + infos across all mapped chains (by residue order)."""
    ref_infos, mob_infos = [], []
    for a, b, *_ in mapping.pairs:
        ri = _extract_ca_infos(ref_struct, [a], min_b_factor, min_plddt)
        mi = _extract_ca_infos(mob_struct, [b], min_b_factor, min_plddt)
        n = min(len(ri), len(mi))
        ref_infos.extend(ri[:n]); mob_infos.extend(mi[:n])
    P = np.array([i.coord for i in ref_infos]) if ref_infos else np.empty((0, 3))
    Q = np.array([i.coord for i in mob_infos]) if mob_infos else np.empty((0, 3))
    return P, Q, ref_infos, mob_infos


def _per_chain_rmsd(mapping, ref_struct, mob_struct, R, t, min_b_factor, min_plddt):
    rows = []
    for a, b, *_ in mapping.pairs:
        ri = _extract_ca_infos(ref_struct, [a], min_b_factor, min_plddt)
        mi = _extract_ca_infos(mob_struct, [b], min_b_factor, min_plddt)
        n = min(len(ri), len(mi))
        if n == 0:
            continue
        P = np.array([x.coord for x in ri[:n]])
        Q = np.array([x.coord for x in mi[:n]])
        Qa = (R @ Q.T).T + t
        rmsd = float(np.sqrt(np.mean(np.sum((P - Qa) ** 2, axis=1))))
        rows.append({"chain_ref": a, "chain_mob": b, "n_residues": n, "rmsd": rmsd})
    return rows


def align_multichain(ref_struct, mob_struct, mapping, strategy="auto",
                     atoms="CA", min_b_factor=0.0, min_plddt=0.0) -> MultiChainResult:
    """Build global and/or local superpositions and pick per coverage-weighted score."""
    def build(sub_mapping, name):
        P, Q, ri, mi = _paired_ca(ref_struct, mob_struct, sub_mapping,
                                  atoms, min_b_factor, min_plddt)
        if len(P) < 3:
            return None
        R, t, rmsd = _kabsch(P, Q)
        Qa = (R @ Q.T).T + t
        per_chain = _per_chain_rmsd(sub_mapping, ref_struct, mob_struct, R, t,
                                    min_b_factor, min_plddt)
        return MultiChainResult(
            strategy=name, mapping=sub_mapping, rotation=R, translation=t,
            rmsd=rmsd, pairs=list(range(len(P))), ref_coords=P,
            mob_coords_aligned=Qa, per_chain=per_chain,
            ref_infos=ri, mob_infos=mi)

    global_res = build(mapping, "global")

    # local candidate = single best-identity chain pair
    local_res = None
    if mapping.pairs:
        best = max(mapping.pairs, key=lambda p: p[2])
        local_map = ChainMapping(pairs=[best])
        local_res = build(local_map, "local")

    if strategy == "global":
        if global_res is None:
            raise ValueError("global strategy requested but no viable multi-chain superposition.")
        return global_res
    if strategy == "local":
        if local_res is None:
            raise ValueError("local strategy requested but no viable single chain pair.")
        return local_res

    # auto: pick by coverage-weighted score
    candidates = [c for c in (global_res, local_res) if c is not None]
    if not candidates:
        raise ValueError("No viable multi-chain superposition could be produced.")
    return max(candidates, key=lambda c: _coverage_score(len(c.ref_coords), c.rmsd))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multichain_align.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire align_multichain into PDBAligner.align + AlignmentResult**

In `pdb_align/aligner.py`, change the `align` signature to add `strategy: str = "auto"`. Immediately after the `flexible` block and after `ref_chs`/`mob_chs` are computed (around line 1110), insert a multi-chain dispatch that runs only for auto mode with multiple chains on both sides:

```python
if mode in ("auto", "Auto (best RMSD)") and len(ref_chs) > 1 and len(mob_chs) > 1:
    from .chains import match_chains, align_multichain
    mapping = match_chains(self.ref_seqs, self.mob_seqs,
                           self.ref_struct, self.mob_struct, ref_chs, mob_chs)
    if mapping.pairs:
        mc = align_multichain(self.ref_struct, self.mob_struct, mapping,
                              strategy=strategy, atoms=atoms,
                              min_b_factor=min_b_factor, min_plddt=min_plddt)
        result_obj = self._multichain_to_result(mc, ref_chs, mob_chs)
        self.last_result = {"seqguided": None, "seqfree": None,
                            "chosen": result_obj._chosen}
        if self.verbose:
            print(result_obj.report())
        return result_obj
```

Add a helper on `PDBAligner` that adapts a `MultiChainResult` into an `AlignmentResult` by constructing the `seqguided`-shaped dict the result object already understands. The matched-atom list uses lightweight pseudo-atoms carrying `chain_name`/`res_seq`/`res_icode`/`get_name()` so `get_rmsd_df` and `save_aligned_pdb` work unchanged:

```python
def _multichain_to_result(self, mc, ref_chs, mob_chs):
    import numpy as np
    from .core import compute_gdt_ts

    class _PA:
        def __init__(self, info):
            self.chain_name = info.chain_id
            self.res_seq = info.resseq
            self.res_icode = info.icode
            self._coord = info.coord
        def get_name(self): return "CA"
        def get_coord(self): return self._coord

    ref_atoms = [_PA(i) for i in mc.ref_infos]
    mob_atoms = [_PA(i) for i in mc.mob_infos]
    diff = mc.ref_coords - mc.mob_coords_aligned
    per_res = np.sqrt(np.sum(diff ** 2, axis=1)) if len(diff) else np.array([])
    gdt = compute_gdt_ts(per_res) if len(per_res) else None
    si = {"rotation": mc.rotation, "translation": mc.translation,
          "rmsd": mc.rmsd, "per_residue_rmsd": per_res,
          "ref_coords": mc.ref_coords, "mob_coords_transformed": mc.mob_coords_aligned,
          "gdt_ts": gdt}
    seqguided = {"aln": None, "ref_atoms": ref_atoms, "mob_atoms": mob_atoms, "si": si}
    chosen = {"name": f"Multi-chain ({mc.strategy})",
              "reason": f"Chain-aware {mc.strategy} superposition over "
                        f"{len(mc.mapping.pairs)} chain pair(s).",
              "seqguided": seqguided, "seqfree": None}
    active_ref_lens = {c: self.ref_lens[c] for c in ref_chs if c in self.ref_lens}
    active_mob_lens = {c: self.mob_lens[c] for c in mob_chs if c in self.mob_lens}
    res = AlignmentResult(chosen=chosen, seqguided=seqguided, seqfree=None,
                          ref_file=self.ref_file, mob_file=self.mob_file,
                          mob_struct=self.mob_struct,
                          ref_lens=active_ref_lens, mob_lens=active_mob_lens,
                          verbose=self.verbose)
    res.strategy = mc.strategy
    res.chain_mapping = mc.mapping
    import pandas as pd
    res._per_chain = pd.DataFrame(mc.per_chain,
        columns=["chain_ref", "chain_mob", "n_residues", "rmsd"])
    return res
```

Note: `AlignmentResult.get_sequence_alignment` handles `aln=None` via its seqfree branch — since seqfree is also None here, guard `get_sequence_alignment` to return `None` when both `aln` is None and seqfree is None (add `if self._chosen["seqguided"] and self._chosen["seqguided"].get("aln") is None and not self._chosen["seqfree"]: return None` at the top). Also thread `strategy` through the `align(strategy=...)` recursive call in the `flexible` block (pass `strategy=strategy`).

- [ ] **Step 6: Write the integration test**

```python
# append to tests/test_multichain_align.py
def test_pdbaligner_auto_uses_multichain(tmp_path):
    import gemmi, pdb_align
    ref = _struct(_two_chain(0.0))
    mob = _struct({k: [(x + 3, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rp, mp = tmp_path / "ref.pdb", tmp_path / "mob.pdb"
    ref.write_pdb(str(rp)); mob.write_pdb(str(mp))
    r = pdb_align.align(str(rp), str(mp))
    assert r.strategy in ("global", "local")
    assert not r.per_chain.empty
    assert r.rmsd < 1e-5
```

- [ ] **Step 7: Run all multi-chain tests**

Run: `pytest tests/test_multichain_align.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Run full suite for regressions**

Run: `pytest tests/ -q`
Expected: all pass (single-chain behavior unchanged).

- [ ] **Step 9: Commit**

```bash
git add pdb_align/chains.py pdb_align/aligner.py tests/test_multichain_align.py
git commit -m "feat: chain-aware auto alignment with global/local strategy selection"
```

---

### Task 7: CLI rewrite — positional, stats-first, opt-in outputs

**Files:**
- Modify: `pdb_align/__main__.py` (full rewrite)
- Modify: `pdb_align/__init__.py` (exports)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `PDBAligner`, `AlignmentResult.report`, `.plot_rmsd`, `.plot_summary`, `.save_aligned_pdb`, `.save_rmsd_csv`, `.to_json`.
- Produces: `pdb_align.__main__.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import os, json, gemmi
from pdb_align.__main__ import main


def _write_single(path, x0=0.0):
    st = gemmi.Structure(); model = gemmi.Model("1"); chain = gemmi.Chain("A")
    for k in range(20):
        res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k + 1, " ")
        at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(k + x0, 0.0, 0.0)
        res.add_atom(at); chain.add_residue(res)
    model.add_chain(chain); st.add_model(model); st.write_pdb(path)


def test_default_run_writes_no_files(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    before = set(os.listdir(tmp_path))
    rc = main([str(ref), str(mob)])
    assert rc == 0
    assert set(os.listdir(tmp_path)) == before  # nothing written
    assert "RMSD" in capsys.readouterr().out


def test_json_flag_emits_valid_json(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main([str(ref), str(mob), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "strategy" in payload


def test_out_and_plot_flags_write_requested_files(tmp_path):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    out, plot = tmp_path / "aligned.pdb", tmp_path / "rmsd.png"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main([str(ref), str(mob), "-o", str(out), "--plot", str(plot)])
    assert rc == 0
    assert out.exists() and plot.exists()


def test_legacy_ref_mob_flags_still_work(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main(["--ref", str(ref), "--mob", str(mob)])
    assert rc == 0
    assert "RMSD" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `main()` takes no argv / positional args unsupported.

- [ ] **Step 3: Rewrite the CLI**

```python
# pdb_align/__main__.py
import argparse
import sys
from .aligner import PDBAligner, AlignmentFailedError


def build_parser():
    p = argparse.ArgumentParser(
        prog="pdb_align",
        description="Compare two protein structures. Prints stats; writes files only when asked.")
    p.add_argument("ref", nargs="?", help="Reference structure (file, pdb:XXXX, af:UniProtID)")
    p.add_argument("mob", nargs="?", help="Mobile structure (file, pdb:XXXX, af:UniProtID)")
    p.add_argument("--ref", dest="ref_flag", help="Reference (alias for positional)")
    p.add_argument("--mob", dest="mob_flag", help="Mobile (alias for positional)")
    p.add_argument("--ref-chains", "--ref_chains", dest="ref_chains",
                   help="Reference chains, e.g. 'A' or 'A:10-150 B'")
    p.add_argument("--mob-chains", "--mob_chains", dest="mob_chains",
                   help="Mobile chains, e.g. 'A'")
    p.add_argument("--mode", default="auto",
                   help="auto | seq_guided | seq_free_shape | seq_free_window | flexible")
    p.add_argument("--strategy", default="auto", choices=["auto", "global", "local"],
                   help="Multi-chain superposition strategy (default auto)")
    p.add_argument("--atoms", default="CA", help="CA | backbone | all_heavy")
    p.add_argument("--min-plddt", "--min_plddt", dest="min_plddt", type=float, default=0.0)
    p.add_argument("-o", "--out", help="Write aligned mobile structure to this file")
    p.add_argument("--plot", nargs="?", const="rmsd.png",
                   help="Write per-residue RMSD plot (default rmsd.png)")
    p.add_argument("--summary-plot", nargs="?", const="summary.png",
                   help="Write multi-panel summary figure")
    p.add_argument("--show", action="store_true", help="Open plots in a window")
    p.add_argument("--report", help="Write the text/JSON report to this file")
    p.add_argument("--csv", help="Write per-residue RMSD table to this CSV")
    p.add_argument("--save", help="Save full result to a .npz for later replotting")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ref = args.ref or args.ref_flag
    mob = args.mob or args.mob_flag
    if not ref or not mob:
        print("error: need a reference and a mobile structure "
              "(positional REF MOB or --ref/--mob).", file=sys.stderr)
        return 2

    ref_chains = args.ref_chains.split() if args.ref_chains else None
    mob_chains = args.mob_chains.split() if args.mob_chains else None

    aligner = PDBAligner(verbose=args.verbose)
    try:
        aligner.add_reference(ref, chains=ref_chains)
        aligner.add_mobile(mob, chains=mob_chains)
        res = aligner.align(mode=args.mode, strategy=args.strategy,
                            atoms=args.atoms, min_plddt=args.min_plddt)
    except (AlignmentFailedError, Exception) as e:
        print(f"Alignment failed: {e}", file=sys.stderr)
        return 1

    # Default: stats to terminal. Nothing written unless a flag asks.
    if args.json:
        print(res.to_json())
    else:
        print(res.report(fmt="text"))

    if args.out:
        res.save_aligned_pdb(args.out)
        if not args.json: print(f"Wrote aligned structure: {args.out}")
    if args.csv:
        res.save_rmsd_csv(args.csv)
        if not args.json: print(f"Wrote per-residue RMSD: {args.csv}")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(res.report(fmt="json" if args.report.endswith(".json") else "text"))
        if not args.json: print(f"Wrote report: {args.report}")
    if args.save:
        res.save(args.save)
        if not args.json: print(f"Saved result: {args.save}")
    if args.plot is not None:
        res.plot_rmsd(filename=args.plot)
        if args.show:
            import matplotlib.pyplot as plt; plt.show()
        if not args.json: print(f"Wrote RMSD plot: {args.plot}")
    if args.summary_plot is not None:
        res.plot_summary(filename=args.summary_plot, show=args.show)
        if not args.json: print(f"Wrote summary plot: {args.summary_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Export new public symbols**

In `pdb_align/__init__.py` add to imports and `__all__`: `LoadedResult` (from `.aligner`), and from `.chains` import `match_chains, align_multichain, ChainMapping`; from `.plotstyle` import `apply_nature_style`. Add each to `__all__`.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Manual smoke test**

Run: `python -m pdb_align pdb:1CRN pdb:1CRN`
Expected: stats block with RMSD ≈ 0.000 Å, TM-score ≈ 1.0, no files written.

- [ ] **Step 8: Commit**

```bash
git add pdb_align/__main__.py pdb_align/__init__.py tests/test_cli.py
git commit -m "feat: positional stats-first CLI with opt-in outputs"
```

---

### Task 8: Docs update

**Files:**
- Modify: `CLAUDE.md`, `README.md` (if present)

- [ ] **Step 1: Update CLAUDE.md**

Add to the architecture section: `chains.py` (chain correspondence + multi-chain strategy), `plotstyle.py` (Nature style), `AlignmentResult.save/load/report/summary_stats/plot_summary`, `align(strategy=...)`, and the new positional CLI usage.

- [ ] **Step 2: Update README with the CLI quickstart**

```
pdb_align ref.pdb mob.pdb                 # stats only
pdb_align pdb:1ABC pdb:2XYZ --plot        # + RMSD plot
pdb_align a.cif b.cif -o aligned.cif --summary-plot
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document intelligent multi-chain CLI, rich results, plotstyle"
```

---

## Self-Review

**Spec coverage:**
- Chain correspondence (Hungarian + permutation) → Task 5. ✓
- Global/local auto strategy → Task 6. ✓
- `align(strategy=...)` integration, single-chain unchanged → Task 6 (dispatch guarded by `len(ref_chs)>1 and len(mob_chs)>1`). ✓
- Data-rich result + summary_stats/report/to_dict/to_json → Task 2. ✓
- save/load without gemmi → Task 3. ✓
- Nature-style plotstyle + refactor plots + plot_summary → Tasks 1, 4. ✓
- Zero-config positional CLI, stats-only default, opt-in outputs, legacy flags → Task 7. ✓
- Public API exports → Tasks 3, 7. ✓
- Error handling (strategy with no viable candidate, load version mismatch) → Tasks 3, 6. ✓
- Testing across all units → each task. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain concrete code.

**Type consistency:** `ChainMapping.pairs` is `(ref, mob, identity, score)` used consistently in Tasks 5–6; `per_chain` columns `chain_ref, chain_mob, n_residues, rmsd` consistent across Tasks 2, 4, 6; `MultiChainResult` fields consumed by `_multichain_to_result` match. `match_chains` signature identical in Tasks 5 and 6. CLI `main(argv=None)->int` matches tests.

**Known follow-up:** `MAX_PERMUTE_CHAINS` currently gates only the geometric-refinement path (exhaustive enumeration was intentionally not implemented — iterative centroid ICP covers the homomultimer case within the cap). This matches the spec's "capped permutation search."
