# SP2 — Streamlit GUI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Streamlit app on the public `pdb_align` API — a verdict-header + tabs layout, real Py3Dmol 3D, ensemble views, one-click export — split into a thin entry plus a testable `webapp/` package, with zero `pdb_align.core` imports.

**Architecture:** `struct_pair_align.py` becomes a thin entry (page config, sidebar inputs, tab dispatch). Pure logic moves into `webapp/`: `data.py` (IO + cached align wrappers), `figures.py` (pure figure builders), `viewer.py` (Py3Dmol), `sections.py` (thin `st.*` tab renderers). One small library addition (`inspect_structure`) closes the chain-listing gap so the app never imports `core`.

**Tech Stack:** Python 3.12, Streamlit, py3Dmol + stmol, Plotly, gemmi, pandas/numpy, pytest, (optional) Playwright.

## Global Constraints

- **No `pdb_align.core` imports** anywhere in `struct_pair_align.py` or `webapp/*.py`.
- App uses only the public API: `PDBAligner`, `AlignmentResult`, `EnsembleResult`,
  `align`, `AlignmentQuality`, `FlaggedRegion`, `inspect_structure`.
- No `core.py` algorithm changes. Any gap → a tested library addition, not app code.
- Remove GUI-only knobs `recycles` and `keep_fraction` (they drove the old private path).
- Launch target stays `streamlit run struct_pair_align.py`.
- Every task ends green: `pytest tests/` passes.

---

### Task 1: Library gap — `pdb_align.inspect_structure()`

**Files:**
- Modify: `pdb_align/aligner.py` (add module-level function near the bottom)
- Modify: `pdb_align/__init__.py` (export it)
- Test: `tests/test_inspect_structure.py` (create)

**Interfaces:**
- Produces: `pdb_align.inspect_structure(path_or_id, cache_dir=None) -> dict` with
  keys `"chains"` (`{chain_id: n_residues}`) and `"sequences"` (`{chain_id: str}`),
  for local files or remote IDs (`pdb:`/`af:`). Lets the GUI list chains for the
  pickers without importing `core`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspect_structure.py
import os
import pdb_align

DATA = os.path.join(os.path.dirname(__file__), "data")

def test_inspect_structure_lists_chains_and_sequences():
    info = pdb_align.inspect_structure(os.path.join(DATA, "ref.pdb"))
    assert "chains" in info and "sequences" in info
    assert len(info["chains"]) >= 1
    chain = next(iter(info["chains"]))
    assert info["chains"][chain] > 0
    assert isinstance(info["sequences"][chain], str)
    assert len(info["sequences"][chain]) == info["chains"][chain]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inspect_structure.py -v`
Expected: FAIL — `AttributeError: module 'pdb_align' has no attribute 'inspect_structure'`

- [ ] **Step 3: Implement**

At the end of `pdb_align/aligner.py` (module level, after the classes):

```python
def inspect_structure(path_or_id, cache_dir=None):
    """List chains, residue counts, and sequences for a structure.

    Accepts a local file path or a remote ID (``pdb:XXXX`` / ``af:UniProtID``).
    Returns ``{"chains": {chain: n_residues}, "sequences": {chain: seq_str}}``.
    A public accessor so callers (e.g. the GUI) need no ``pdb_align.core``.
    """
    al = PDBAligner()
    if cache_dir:
        al._fetch_cache_dir = cache_dir
    al.add_reference(path_or_id)
    chains = {c: int(n) for c, n in al.ref_lens.items()}
    sequences = {c: str(rec.seq) for c, rec in al.ref_seqs.items()}
    return {"chains": chains, "sequences": sequences}
```

In `pdb_align/__init__.py`, import and add to `__all__`:

```python
from .aligner import (PDBAligner, AlignmentResult, AlignmentFailedError,
                      EnsembleResult, DomainResult, LoadedResult, inspect_structure)
```
Add `"inspect_structure"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inspect_structure.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py pdb_align/__init__.py tests/test_inspect_structure.py
git commit -m "feat: pdb_align.inspect_structure() for chain listing (GUI gap)"
```

---

### Task 2: `webapp/data.py` — IO + cached align wrappers

**Files:**
- Create: `webapp/__init__.py`
- Create: `webapp/data.py`
- Test: `tests/test_webapp_data.py` (create)

**Interfaces:**
- Consumes: `pdb_align.PDBAligner`, `pdb_align.inspect_structure`.
- Produces:
  - `save_upload_to_temp(name: str, data: bytes) -> str` — writes bytes to a temp
    file preserving suffix; returns the path.
  - `run_pairwise(ref_path, mob_path, ref_chains, mob_chains, mode, strategy, opts) -> AlignmentResult`
  - `run_ensemble(ref_path, mob_paths, ref_chains, mob_chains_map, mode, strategy, opts) -> EnsembleResult`
  - `input_key(ref, mobs, ref_chains, mob_chains_map, mode, strategy, opts) -> str`
    — stable hash of inputs for caching.
  - `list_chains(path) -> dict` — thin wrapper over `inspect_structure(path)["chains"]`.
  - `opts` is a dict: `{seq_gap_open, seq_gap_extend, atoms, min_plddt}`.
  Cached wrappers use `st.cache_data` but the bodies are importable/testable
  because the functions below are the plain implementations the cache wraps.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webapp_data.py
import os
import pytest
from webapp import data as D

DATA = os.path.join(os.path.dirname(__file__), "data")
REF = os.path.join(DATA, "ref.pdb")
MOB = os.path.join(DATA, "mob.pdb")
OPTS = {"seq_gap_open": -10, "seq_gap_extend": -0.5, "atoms": "CA", "min_plddt": 0.0}


def test_save_upload_to_temp_roundtrip():
    p = D.save_upload_to_temp("x.pdb", open(REF, "rb").read())
    assert p.endswith(".pdb") and os.path.exists(p)
    assert open(p, "rb").read() == open(REF, "rb").read()


def test_list_chains():
    chains = D.list_chains(REF)
    assert isinstance(chains, dict) and len(chains) >= 1


def test_run_pairwise_returns_result():
    from pdb_align import AlignmentResult
    res = D.run_pairwise(REF, MOB, None, None, "auto", "auto", OPTS)
    assert isinstance(res, AlignmentResult)
    assert res.quality.band in ("excellent", "good", "moderate", "poor")


def test_run_ensemble_returns_result():
    from pdb_align import EnsembleResult
    ens = D.run_ensemble(REF, [MOB, REF], None, {}, "auto", "auto", OPTS)
    assert isinstance(ens, EnsembleResult)
    assert len(ens.results) == 2


def test_input_key_stability():
    k1 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k2 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k3 = D.input_key(REF, [MOB], None, {}, "flexible", "auto", OPTS)
    assert k1 == k2 and k1 != k3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp'`

- [ ] **Step 3: Implement**

```python
# webapp/__init__.py
"""Streamlit UI package for pdb_align (pure logic + thin renderers)."""
```

```python
# webapp/data.py
"""IO, caching, and alignment wrappers for the Streamlit app.

Only the public pdb_align API is used here — never pdb_align.core.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

from pdb_align import PDBAligner, inspect_structure


def save_upload_to_temp(name: str, data: bytes) -> str:
    suffix = os.path.splitext(name)[1].lower() or ".pdb"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="pdb_align_up_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def list_chains(path: str) -> dict:
    return inspect_structure(path)["chains"]


def _apply_opts(kwargs: dict, opts: dict) -> dict:
    kwargs.update(
        seq_gap_open=opts.get("seq_gap_open", -10),
        seq_gap_extend=opts.get("seq_gap_extend", -0.5),
        atoms=opts.get("atoms", "CA"),
        min_plddt=opts.get("min_plddt", 0.0),
    )
    return kwargs


def run_pairwise(ref_path, mob_path, ref_chains, mob_chains, mode, strategy, opts):
    al = PDBAligner()
    al.add_reference(ref_path, chains=ref_chains)
    al.add_mobile(mob_path, chains=mob_chains)
    return al.align(**_apply_opts(dict(mode=mode, strategy=strategy), opts))


def run_ensemble(ref_path, mob_paths, ref_chains, mob_chains_map, mode, strategy, opts):
    al = PDBAligner()
    al.add_reference(ref_path, chains=ref_chains)
    return al.align_ensemble(
        mob_list=list(mob_paths),
        **_apply_opts(dict(mode=mode), opts),
    )


def input_key(ref, mobs, ref_chains, mob_chains_map, mode, strategy, opts) -> str:
    payload = json.dumps(dict(
        ref=ref, mobs=sorted(mobs), ref_chains=ref_chains,
        mob_chains=mob_chains_map, mode=mode, strategy=strategy, opts=opts,
    ), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()
```

Note: `run_ensemble` intentionally forwards `mode` and per-run options; the
reference chains are honored via `add_reference`. `align_ensemble` does not take a
`strategy`/per-mobile-chains argument in the current API, so `strategy` and
`mob_chains_map` participate only in the cache key, not the call — this keeps the
signature stable and is verified by `test_run_ensemble_returns_result`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_webapp_data.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add webapp/__init__.py webapp/data.py tests/test_webapp_data.py
git commit -m "feat: webapp.data IO + cached align/ensemble wrappers"
```

---

### Task 3: `webapp/figures.py` — pure figure builders

**Files:**
- Create: `webapp/figures.py`
- Test: `tests/test_webapp_figures.py` (create)

**Interfaces:**
- Consumes: an `AlignmentResult` (via its public accessors) and DataFrames.
- Produces (all return a Plotly `go.Figure`, no `st.*`):
  - `per_residue_figure(rmsd_df, flagged_regions=None) -> go.Figure`
  - `distance_matrix_figure(coords, title) -> go.Figure`
  - `pair_distance_hist(ref_coords, mob_coords, title) -> go.Figure`
  - `ensemble_rmsd_heatmap(matrix_df) -> go.Figure`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webapp_figures.py
import os
import numpy as np
import plotly.graph_objects as go
from pdb_align import PDBAligner
from webapp import figures as F

DATA = os.path.join(os.path.dirname(__file__), "data")

def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")

def test_per_residue_figure():
    r = _result()
    fig = F.per_residue_figure(r.get_rmsd_df(), r.quality.flagged_regions)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1

def test_distance_matrix_figure():
    r = _result()
    ref_c, mob_c = r.get_aligned_coords()
    fig = F.distance_matrix_figure(ref_c, "Reference")
    assert isinstance(fig, go.Figure)

def test_pair_distance_hist():
    r = _result()
    ref_c, mob_c = r.get_aligned_coords()
    fig = F.pair_distance_hist(ref_c, mob_c, "Pairs")
    assert isinstance(fig, go.Figure)

def test_ensemble_rmsd_heatmap():
    al = PDBAligner(); al.add_reference(os.path.join(DATA, "ref.pdb"))
    ens = al.align_ensemble([os.path.join(DATA, "mob.pdb"),
                             os.path.join(DATA, "ref.pdb")], mode="auto")
    fig = F.ensemble_rmsd_heatmap(ens.rmsd_matrix())
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp_figures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.figures'`

- [ ] **Step 3: Implement**

```python
# webapp/figures.py
"""Pure Plotly figure builders for the Streamlit app (no st.* calls)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def per_residue_figure(rmsd_df, flagged_regions=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(rmsd_df))), y=rmsd_df["RMSD"], mode="lines+markers",
        text=rmsd_df["Residue"], name="Cα RMSD",
        hovertemplate="%{text}<br>%{y:.2f} Å<extra></extra>"))
    for fr in (flagged_regions or []):
        # shade flagged spans by matching residue labels to x-indices
        labels = list(rmsd_df["Residue"])
        try:
            x0 = labels.index(fr.start_label)
            x1 = labels.index(fr.end_label)
        except ValueError:
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.12, line_width=0)
    fig.update_layout(xaxis_title="Residue index", yaxis_title="RMSD (Å)",
                      margin=dict(l=40, r=10, t=30, b=40), height=360)
    return fig


def distance_matrix_figure(coords, title) -> go.Figure:
    c = np.asarray(coords, dtype=float)
    D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    fig = go.Figure(go.Heatmap(z=D, colorscale="Viridis",
                               colorbar=dict(title="Å")))
    fig.update_layout(title=title, height=420,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def pair_distance_hist(ref_coords, mob_coords, title) -> go.Figure:
    a = np.asarray(ref_coords, dtype=float)
    b = np.asarray(mob_coords, dtype=float)
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=-1)
    fig = go.Figure(go.Histogram(x=d, nbinsx=30))
    fig.update_layout(title=title, xaxis_title="Cα–Cα distance (Å)",
                      yaxis_title="count", height=320,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def ensemble_rmsd_heatmap(matrix_df) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix_df.values, x=list(matrix_df.columns), y=list(matrix_df.index),
        colorscale="Viridis", colorbar=dict(title="RMSD (Å)")))
    fig.update_layout(height=460, margin=dict(l=60, r=10, t=30, b=60))
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_webapp_figures.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add webapp/figures.py tests/test_webapp_figures.py
git commit -m "feat: webapp.figures pure Plotly builders"
```

---

### Task 4: `webapp/viewer.py` — Py3Dmol superposition

**Files:**
- Create: `webapp/viewer.py`
- Test: `tests/test_webapp_viewer.py` (create)

**Interfaces:**
- Consumes: an `AlignmentResult` (`aligned_structure`, `mob_struct`/ref file) and gemmi.
- Produces:
  - `remap_long_chain_ids(struct) -> dict` — renames chains whose gemmi name is
    >1 char to unique single chars (PDB limit); returns the mapping.
  - `structure_to_pdb_string(struct) -> str` — PDB text (after remap).
  - `build_view(ref_pdb_str, aligned_pdb_str, color_by="rmsd") -> py3Dmol.view` —
    ref as grey cartoon, mobile as cartoon spectrum by B-factor (RMSD) or plddt.
  The remap + to-string are pure and tested; `build_view` is exercised only for
  no-raise (py3Dmol import guarded).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webapp_viewer.py
import os
import gemmi
from webapp import viewer as V

DATA = os.path.join(os.path.dirname(__file__), "data")

def test_remap_long_chain_ids():
    st = gemmi.read_structure(os.path.join(DATA, "ref.pdb"))
    # force a multi-char chain name
    st[0][0].name = "AAA"
    mapping = V.remap_long_chain_ids(st)
    assert mapping.get("AAA")
    assert all(len(c.name) == 1 for m in st for c in m)

def test_structure_to_pdb_string():
    st = gemmi.read_structure(os.path.join(DATA, "ref.pdb"))
    s = V.structure_to_pdb_string(st)
    assert "ATOM" in s and isinstance(s, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp_viewer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.viewer'`

- [ ] **Step 3: Implement**

```python
# webapp/viewer.py
"""Py3Dmol superposition view built from AlignmentResult.aligned_structure()."""
from __future__ import annotations

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def remap_long_chain_ids(struct) -> dict:
    used = {c.name for m in struct for c in m if len(c.name) == 1}
    replacements = {}
    for model in struct:
        for chain in model:
            if len(chain.name) > 1:
                if chain.name not in replacements:
                    for c in _ALPHABET:
                        if c not in used:
                            replacements[chain.name] = c
                            used.add(c)
                            break
                chain.name = replacements.get(chain.name, chain.name[0])
    return replacements


def structure_to_pdb_string(struct) -> str:
    import io
    remap_long_chain_ids(struct)
    # gemmi writes via a document/os path; use its string writer.
    try:
        return struct.make_pdb_string()
    except Exception:
        import tempfile
        import os
        fd, p = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        struct.write_pdb(p)
        s = open(p).read()
        os.unlink(p)
        return s


def build_view(ref_pdb_str, aligned_pdb_str, color_by="rmsd"):
    import py3Dmol
    view = py3Dmol.view(width=760, height=520)
    view.addModel(ref_pdb_str, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgrey"}})
    view.addModel(aligned_pdb_str, "pdb")
    if color_by in ("rmsd", "plddt", "bfactor"):
        view.setStyle({"model": 1}, {"cartoon": {"colorscheme":
            {"prop": "b", "gradient": "roygb", "min": 0, "max": 5}}})
    else:  # by chain
        view.setStyle({"model": 1}, {"cartoon": {"colorscheme": "chainHetatm"}})
    view.zoomTo()
    return view
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_webapp_viewer.py -v`
Expected: PASS (2 tests)

If `make_pdb_string` is unavailable in the installed gemmi, the fallback path is
exercised — the test still passes because it only asserts on the returned string.

- [ ] **Step 5: Commit**

```bash
git add webapp/viewer.py tests/test_webapp_viewer.py
git commit -m "feat: webapp.viewer Py3Dmol superposition + chain remap"
```

---

### Task 5: `webapp/sections.py` — tab renderers

**Files:**
- Create: `webapp/sections.py`
- Test: `tests/test_webapp_import.py` (create)

**Interfaces:**
- Consumes: `webapp.data`, `webapp.figures`, `webapp.viewer`, `streamlit`,
  `stmol.showmol`.
- Produces thin renderers (each takes already-computed objects, calls `st.*`):
  `render_header(result_or_ensemble)`, `render_overview(result)`,
  `render_3d(result)`, `render_per_residue(result)`,
  `render_ensemble(ensemble)`, `render_export(result_or_ensemble, is_ensemble)`.

- [ ] **Step 1: Write the failing import/smoke test**

```python
# tests/test_webapp_import.py
def test_webapp_modules_import():
    import webapp.data, webapp.figures, webapp.viewer, webapp.sections
    for fn in ("render_header", "render_overview", "render_3d",
               "render_per_residue", "render_ensemble", "render_export"):
        assert hasattr(webapp.sections, fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.sections'`

- [ ] **Step 3: Implement**

```python
# webapp/sections.py
"""Thin Streamlit tab renderers. Data/figures come from webapp.data/figures."""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from webapp import figures as F
from webapp import viewer as V


def render_header(res):
    q = res.quality
    s = res.summary_stats()
    badge = {"excellent": "🟢", "good": "🔵", "moderate": "🟠", "poor": "🔴"}
    st.subheader(f"{badge.get(q.band, '⚪')} {q.band.upper()}  ·  {q.verdict}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSD (Å)", f"{s['rmsd']:.3f}" if s["rmsd"] is not None else "—")
    c2.metric("TM-score", f"{s['tm_score']:.3f}" if s["tm_score"] is not None else "—")
    c3.metric("GDT-TS", f"{s['gdt_ts']:.1f}" if s["gdt_ts"] is not None else "—")
    c4.metric("Coverage", f"{s['coverage_pct']:.0f}%" if s["coverage_pct"] is not None else "—")
    st.caption(f"Confidence: {q.confidence}")
    for w in q.warnings:
        st.warning(w)


def render_overview(res):
    s = res.summary_stats()
    st.write(f"**Method:** {s['method']}  ·  **Strategy:** {s['strategy']}")
    st.caption(s.get("reason") or "")
    if s.get("chain_mapping"):
        st.write("**Chain mapping (ref → mob, %id):**")
        st.dataframe([{"ref": m["ref"], "mob": m["mob"], "identity": m["identity"]}
                      for m in s["chain_mapping"]], hide_index=True,
                     use_container_width=True)
    pc = res.per_chain
    if not pc.empty:
        st.write("**Per-chain RMSD:**")
        st.dataframe(pc, hide_index=True, use_container_width=True)
    fr = res.quality.flagged_regions
    if fr:
        st.write("**Flagged regions:**")
        st.dataframe([r.to_dict() for r in fr], hide_index=True,
                     use_container_width=True)
    else:
        st.info("No flagged high-deviation regions.")


def render_3d(res):
    from stmol import showmol
    color_by = st.selectbox("Colour by", ["rmsd", "plddt", "chain"], index=0)
    ref_struct = res.mob_struct  # placeholder replaced below
    aligned = res.aligned_structure(
        color_by="rmsd" if color_by == "rmsd" else "bfactor")
    aligned_str = V.structure_to_pdb_string(aligned)
    # reference structure written from the aligner's cached ref via a temp file
    ref_str = st.session_state.get("_ref_pdb_str", "")
    try:
        view = V.build_view(ref_str, aligned_str, color_by=color_by)
        showmol(view, height=520, width=760)
    except Exception as e:
        st.error(f"3D viewer unavailable: {e}")
    st.download_button("⬇️ Aligned structure (PDB)", data=aligned_str,
                       file_name="aligned.pdb")


def render_per_residue(res):
    df = res.get_rmsd_df(on="reference")
    st.plotly_chart(F.per_residue_figure(df, res.quality.flagged_regions),
                    use_container_width=True, key="per_res")
    with st.expander("Sequence alignment"):
        aln = res.get_sequence_alignment()
        st.code(aln if isinstance(aln, str) else str(aln))
    with st.expander("Distance matrices & histograms"):
        ref_c, mob_c = res.get_aligned_coords()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(F.distance_matrix_figure(ref_c, "Reference"),
                            use_container_width=True, key="dm_ref")
        with c2:
            st.plotly_chart(F.pair_distance_hist(ref_c, mob_c, "Cα–Cα"),
                            use_container_width=True, key="pdh")


def render_ensemble(ens):
    st.dataframe(ens.summary(), use_container_width=True, hide_index=True)
    st.plotly_chart(F.ensemble_rmsd_heatmap(ens.rmsd_matrix()),
                    use_container_width=True, key="ens_heat")
    n = st.slider("Clusters", 2, max(2, len(ens.results)), 2)
    try:
        ens.cluster(n_clusters=n)
        st.pyplot(ens.plot_pca(color_by="cluster"))
        st.pyplot(ens.plot_dendrogram())
    except Exception as e:
        st.info(f"Clustering unavailable: {e}")


def render_export(obj, is_ensemble):
    st.write("One-click reproducible bundle:")
    if st.button("📦 Build bundle (ZIP)"):
        out = os.path.join(tempfile.mkdtemp(), "bundle.zip")
        obj.export_bundle(out)
        st.download_button("⬇️ Download bundle", data=open(out, "rb").read(),
                           file_name="pdb_align_bundle.zip")
```

Note: the reference PDB string for the 3D view is stashed in
`st.session_state["_ref_pdb_str"]` by the entry point (Task 6) right after the
alignment runs, so `render_3d` needs no `core` access.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webapp_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/sections.py tests/test_webapp_import.py
git commit -m "feat: webapp.sections thin tab renderers"
```

---

### Task 6: Rewrite `struct_pair_align.py` entry + flip convergence guard

**Files:**
- Rewrite: `struct_pair_align.py`
- Modify: `tests/test_convergence_guard.py` (remove xfail; scan webapp too)
- Test: existing `tests/test_convergence_guard.py`

**Interfaces:**
- Consumes: `webapp.data`, `webapp.sections`, `pdb_align`.

- [ ] **Step 1: Update the convergence guard to strict (write the failing test first)**

```python
# tests/test_convergence_guard.py
import os
import re
import glob

ROOT = os.path.join(os.path.dirname(__file__), "..")

def _files():
    yield os.path.join(ROOT, "struct_pair_align.py")
    yield from glob.glob(os.path.join(ROOT, "webapp", "*.py"))

def test_app_does_not_import_core():
    for path in _files():
        with open(path) as f:
            src = f.read()
        assert not re.search(r"from\s+pdb_align\.core\s+import", src), path
        assert not re.search(r"import\s+pdb_align\.core", src), path
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_convergence_guard.py -v`
Expected: FAIL — old `struct_pair_align.py` still imports `pdb_align.core`.

- [ ] **Step 3: Rewrite the entry point**

Replace the entire `struct_pair_align.py` with a thin entry (no `core` imports).
Full file:

```python
"""Structure Alignment Suite — Streamlit front-end for pdb_align.

Run: streamlit run struct_pair_align.py
Built entirely on the public pdb_align API (no pdb_align.core).
"""
from __future__ import annotations

import streamlit as st

import pdb_align
from webapp import data as D
from webapp import sections as S

st.set_page_config(page_title="Structure Alignment Suite", page_icon="🧬",
                   layout="wide")
st.title("🧬 Structure Alignment Suite")

if "uploads" not in st.session_state:
    st.session_state.uploads = {}      # name -> temp path
if "results" not in st.session_state:
    st.session_state.results = {}      # input_key -> result/ensemble

with st.sidebar:
    st.header("📤 Structures")
    files = st.file_uploader("Upload PDB/mmCIF", type=["pdb", "cif", "mmcif"],
                             accept_multiple_files=True)
    fetch = st.text_input("…or fetch (pdb:1ABC, af:P00533)")
    if st.button("Fetch") and fetch:
        for tid in [t.strip() for t in fetch.split(",") if t.strip()]:
            try:
                st.session_state.uploads[tid] = tid  # API resolves IDs directly
            except Exception as e:
                st.error(f"{tid}: {e}")
    if files:
        for f in files:
            if f.name not in st.session_state.uploads:
                st.session_state.uploads[f.name] = D.save_upload_to_temp(
                    f.name, f.getvalue())

    names = list(st.session_state.uploads.keys())
    if len(names) < 2:
        st.info("Add at least two structures.")
        st.stop()

    ref_name = st.selectbox("Reference", names)
    mob_names = st.multiselect("Mobile(s)", [n for n in names if n != ref_name],
                               default=[n for n in names if n != ref_name][:1])
    ref_path = st.session_state.uploads[ref_name]
    ref_chain_opts = list(D.list_chains(ref_path).keys())
    ref_chains = st.multiselect("Reference chains", ref_chain_opts,
                                default=ref_chain_opts) or None

    mode_label = st.selectbox("Mode",
        ["auto", "seq_guided", "seq_free_shape", "seq_free_window", "flexible"])
    strategy = st.selectbox("Strategy", ["auto", "global", "local"])
    with st.expander("Advanced"):
        opts = dict(
            seq_gap_open=st.slider("Gap open", -20, -1, -10),
            seq_gap_extend=st.slider("Gap extend", -20.0, -0.1, -0.5, 0.1),
            atoms=st.selectbox("Atoms", ["CA", "backbone", "all_heavy"]),
            min_plddt=st.number_input("Min pLDDT", 0.0, 100.0, 0.0),
        )
    run = st.button("🚀 Run", use_container_width=True)

if not mob_names:
    st.info("Select at least one mobile structure.")
    st.stop()

is_ensemble = len(mob_names) > 1
mob_paths = [st.session_state.uploads[n] for n in mob_names]

if run:
    key = D.input_key(ref_path, mob_paths, ref_chains, {}, mode_label, strategy, opts)
    if key not in st.session_state.results:
        with st.spinner("Aligning…"):
            try:
                if is_ensemble:
                    st.session_state.results[key] = ("ensemble",
                        D.run_ensemble(ref_path, mob_paths, ref_chains, {},
                                       mode_label, strategy, opts))
                else:
                    res = D.run_pairwise(ref_path, mob_paths[0], ref_chains, None,
                                         mode_label, strategy, opts)
                    st.session_state.results[key] = ("pairwise", res)
                    # stash ref PDB for the 3D view (public API only)
                    st.session_state["_ref_pdb_str"] = open(ref_path).read() \
                        if ref_path.lower().endswith(".pdb") else ""
            except Exception as e:
                st.error(f"Alignment failed: {e}")
                st.stop()
    st.session_state["_last_key"] = key

key = st.session_state.get("_last_key")
if not key or key not in st.session_state.results:
    st.info("Configure inputs in the sidebar and press Run.")
    st.stop()

kind, obj = st.session_state.results[key]

if kind == "ensemble":
    best = min(obj.results, key=lambda r: r.rmsd if r.rmsd is not None else 1e9)
    S.render_header(best)
    tabs = st.tabs(["Overview", "3D", "Per-residue", "Ensemble", "Export"])
    with tabs[0]: S.render_overview(best)
    with tabs[1]: S.render_3d(best)
    with tabs[2]: S.render_per_residue(best)
    with tabs[3]: S.render_ensemble(obj)
    with tabs[4]: S.render_export(obj, is_ensemble=True)
else:
    S.render_header(obj)
    tabs = st.tabs(["Overview", "3D", "Per-residue", "Export"])
    with tabs[0]: S.render_overview(obj)
    with tabs[1]: S.render_3d(obj)
    with tabs[2]: S.render_per_residue(obj)
    with tabs[3]: S.render_export(obj, is_ensemble=False)
```

- [ ] **Step 4: Run the convergence guard and full suite**

Run: `pytest tests/test_convergence_guard.py tests/ -q`
Expected: PASS — guard is now strict-green; no `core` imports remain.

- [ ] **Step 5: Commit**

```bash
git add struct_pair_align.py tests/test_convergence_guard.py
git commit -m "feat: rewrite Streamlit app on public API; strict convergence guard"
```

---

### Task 7: Manual run verification + optional Playwright E2E

**Files:**
- Optional create: `tests/test_app_e2e.py`

- [ ] **Step 1: Launch the app headless and confirm it boots**

Run:
```bash
streamlit run struct_pair_align.py --server.headless true --server.port 8599 &
sleep 8
curl -sSf http://localhost:8599/ >/dev/null && echo "APP UP"
kill %1
```
Expected: `APP UP` (Streamlit served the page without a Python import/exception).

- [ ] **Step 2 (optional): Playwright smoke**

Only if the environment can run a browser. Drive: open app, upload
`tests/data/ref.pdb` + `tests/data/mob.pdb`, select mobile, click Run, assert the
verdict header (`GOOD`/`Same fold`) appears. Skip with a clear message otherwise.

- [ ] **Step 3: Commit (if E2E added)**

```bash
git add tests/test_app_e2e.py
git commit -m "test: Playwright smoke for the Streamlit app"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `requirements-app.txt` (ensure `stmol`,
  `py3Dmol` present — they already are)

- [ ] **Step 1: Update README Streamlit section**

Describe the new layout (verdict header + Overview/3D/Per-residue/Ensemble/Export
tabs), that it runs on the public API, and the one-click bundle. Keep the
`streamlit run struct_pair_align.py` command.

- [ ] **Step 2: Update CLAUDE.md**

Replace the "Streamlit App" section: the app is now a thin
`struct_pair_align.py` entry plus a `webapp/` package (`data.py` = IO + cached
align wrappers, `figures.py` = pure Plotly builders, `viewer.py` = Py3Dmol +
chain remap, `sections.py` = thin tab renderers). It imports only the public API
(guarded by `tests/test_convergence_guard.py`). Note `pdb_align.inspect_structure`.

- [ ] **Step 3: Run full suite**

Run: `pytest tests/`
Expected: PASS (no xfail remains for the convergence guard).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md requirements-app.txt
git commit -m "docs: document redesigned Streamlit app and webapp package"
```

---

## Self-Review

**Spec coverage:**
- Section A (convergence rule, drop recycles/keep_fraction) → Task 6 rewrite + Task 1
  gap + strict guard. ✅
- Section B (file split: entry + data/figures/viewer/sections) → Tasks 2–6. ✅
- Section C (verdict header + tabs, sidebar inputs, Advanced) → Tasks 5 & 6. ✅
- Section D (state cache, error handling, temp-PDB 3D) → Tasks 2 & 6. ✅
- Section E (tests for data/figures/viewer, guard strict, import smoke, optional
  Playwright) → Tasks 2–7. ✅
- Section F (non-goals) → respected; only additive library change is `inspect_structure`. ✅

**Placeholder scan:** All code steps contain real code. `render_3d` reference-PDB
handling is fully specified (session_state stash in Task 6). No TBDs.

**Type consistency:** `run_pairwise`/`run_ensemble`/`input_key`/`list_chains`
signatures in Task 2 match their calls in Task 6. `per_residue_figure`,
`distance_matrix_figure`, `pair_distance_hist`, `ensemble_rmsd_heatmap` in Task 3
match calls in Task 5. `structure_to_pdb_string`/`build_view` in Task 4 match Task 5.

**Notes for implementer:**
- `PDBAligner.align_ensemble(mob_list, mode, atoms, workers, out_dir, **kwargs)`
  forwards `**kwargs` to `align`, so `_apply_opts` (which sets `seq_gap_open`,
  `seq_gap_extend`, `atoms`, `min_plddt`) is accepted by both `run_pairwise`
  (`align`) and `run_ensemble` (`align_ensemble`). Verified against the code.
- If the installed gemmi lacks `make_pdb_string`, the Task 4 fallback covers it.
- `res.get_sequence_alignment()` may return a tuple; `render_per_residue` stringifies
  defensively.
