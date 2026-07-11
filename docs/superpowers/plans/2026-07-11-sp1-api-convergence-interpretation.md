# SP1 — API Convergence + Interpretation Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `pdb_align` library the single source of truth for the GUI by closing accessor gaps, adding a plain-language interpretation layer, and consolidating exports into one reproducible bundle call.

**Architecture:** Additive changes to `pdb_align/aligner.py` plus a new pure-Python module `pdb_align/interpretation.py`. The computational core (`core.py`) is untouched. The interpretation layer is a pure function of data already present on `AlignmentResult`, so it is unit-testable without gemmi. Export logic moves out of the Streamlit app into the API.

**Tech Stack:** Python 3.12, gemmi (structures), numpy/pandas, matplotlib, pytest.

## Global Constraints

- No changes to `pdb_align/core.py` algorithms or public functions.
- No changes to the Streamlit app in this sub-project (that is SP2).
- `save_aligned_pdb()` external behavior must remain identical (regression-tested).
- All interpretation thresholds are named module constants in `interpretation.py`.
- Follow existing code style: lazy properties, `import` inside methods where the
  file already does so, gemmi for structures.
- Every task ends green: `pytest tests/` passes.

---

### Task 1: Factor out aligned-structure builder + `aligned_structure()`

**Files:**
- Modify: `pdb_align/aligner.py` (`save_aligned_pdb`, ~lines 274-377; add helper + method)
- Test: `tests/test_aligned_structure.py` (create)

**Interfaces:**
- Produces:
  - `AlignmentResult._build_aligned_structure(color_by: str = "rmsd") -> gemmi.Structure`
    — returns a transformed clone of the mobile structure. `color_by="rmsd"`
    writes per-residue deviation (Å) into every atom's `b_iso`; `color_by` in
    `{"bfactor","plddt"}` leaves original B-factors untouched.
  - `AlignmentResult.aligned_structure(color_by: str = "rmsd") -> gemmi.Structure`
    — thin public wrapper over the helper.
  - `save_aligned_pdb(filename, subset_only=False, preserve_bfactor=False)` keeps
    its signature; internally calls `_build_aligned_structure` with
    `color_by="bfactor" if preserve_bfactor else "rmsd"` and writes to disk.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligned_structure.py
import os
import numpy as np
import pytest
from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture
def result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")

def test_aligned_structure_bfactor_matches_rmsd_df(result):
    struct = result.aligned_structure(color_by="rmsd")
    df = result.get_rmsd_df(on="mobile")
    # Build residue->rmsd lookup from the df labels "CHAIN:RESSEQ[icode]"
    want = {row["Residue"]: row["RMSD"] for _, row in df.iterrows()}
    seen = 0
    for model in struct:
        for chain in model:
            for res in chain:
                label = f"{chain.name}:{res.seqid.num}"
                if label in want:
                    b = res[0].b_iso
                    assert b == pytest.approx(want[label], abs=1e-3)
                    seen += 1
    assert seen > 0

def test_aligned_structure_preserve_bfactor(result):
    struct = result.aligned_structure(color_by="bfactor")
    # coords are moved but at least one non-zero original b-factor survives
    bvals = [a.b_iso for m in struct for c in m for r in c for a in r]
    assert any(b != 0.0 for b in bvals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aligned_structure.py -v`
Expected: FAIL — `AttributeError: 'AlignmentResult' object has no attribute 'aligned_structure'`

- [ ] **Step 3: Refactor `save_aligned_pdb` and add the methods**

In `pdb_align/aligner.py`, extract the body of `save_aligned_pdb` that computes
`R, t, per_res_rmsd, ref_atoms, mob_atoms` and mutates a cloned structure into a
new private method. Keep the existing distance-mapping logic verbatim; only
change where the structure comes from and whether B-factors are overwritten.

```python
def _build_aligned_structure(self, color_by: str = "rmsd"):
    """Return a transformed clone of the mobile structure.

    color_by="rmsd": write per-residue deviation (A) into every atom b_iso.
    color_by in {"bfactor","plddt"}: keep original B-factors.
    """
    import numpy as np
    if not self._chosen:
        raise ValueError("No alignment results available. Run align() first.")
    chosen = self._chosen
    R = t = per_res_rmsd = None
    ref_atoms, mob_atoms = [], []

    if chosen["seqguided"]:
        R = chosen["seqguided"]["si"]["rotation"]
        t = chosen["seqguided"]["si"]["translation"]
        per_res_rmsd = chosen["seqguided"]["si"]["per_residue_rmsd"]
        ref_atoms = chosen["seqguided"]["ref_atoms"]
        mob_atoms = chosen["seqguided"]["mob_atoms"]
    elif chosen["seqfree"]:
        R = chosen["seqfree"].rotation
        t = chosen["seqfree"].translation
        ref_subset = chosen["seqfree"].ref_subset_ca_coords
        mob_subset = chosen["seqfree"].mob_subset_ca_coords_aligned
        pairs = chosen["seqfree"].pairs
        ref_infos = chosen["seqfree"].ref_subset_infos
        mob_infos = chosen["seqfree"].mob_subset_infos
        per_res_rmsd = [float(np.linalg.norm(ref_subset[i] - mob_subset[j]))
                        for (i, j) in pairs]

        class PseudoAtom:
            def __init__(self, c_name, r_seq, r_ico):
                self.chain_name = c_name
                self.res_seq = r_seq
                self.res_icode = r_ico
        mob_atoms = [PseudoAtom(mob_infos[j].chain_id, mob_infos[j].resseq,
                                mob_infos[j].icode) for (i, j) in pairs]

    if R is None or t is None:
        raise ValueError("Chosen alignment has no transform.")

    out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, "clone") \
        else self.mob_struct.copy()

    dist_map = {}
    write_rmsd = (color_by == "rmsd")
    if write_rmsd and mob_atoms and per_res_rmsd is not None:
        for k in range(min(len(mob_atoms), len(per_res_rmsd))):
            ma = mob_atoms[k]
            c_name = getattr(ma, "chain_name", getattr(ma, "last_chain_name", "A"))
            if hasattr(ma, "get_id"):
                het, r_seq, r_ico = ma.get_parent().get_id()
            else:
                r_seq, r_ico = ma.res_seq, ma.res_icode
            key = (c_name, r_seq, r_ico.strip() if hasattr(r_ico, "strip") else "")
            dist_map[key] = float(per_res_rmsd[k])

    for model in out_struct:
        for chain in model:
            for residue in chain:
                resseq = residue.seqid.num
                icode = residue.seqid.icode if getattr(residue.seqid, "icode", " ") != " " else ""
                key = (chain.name, resseq, icode.strip() if hasattr(icode, "strip") else "")
                mapped = dist_map.get(key, 0.0)
                for atom in residue:
                    coord = np.array(atom.pos.tolist(), dtype=float)
                    new_coord = (R @ coord) + t
                    atom.pos.x = float(new_coord[0])
                    atom.pos.y = float(new_coord[1])
                    atom.pos.z = float(new_coord[2])
                    if write_rmsd:
                        atom.b_iso = mapped
    return out_struct

def aligned_structure(self, color_by: str = "rmsd"):
    """In-memory transformed mobile structure for 3D viewing/export."""
    return self._build_aligned_structure(color_by=color_by)
```

Then replace the body of `save_aligned_pdb` after its docstring with:

```python
    color_by = "bfactor" if preserve_bfactor else "rmsd"
    out_struct = self._build_aligned_structure(color_by=color_by)
    if filename.lower().endswith((".cif", ".mmcif")):
        out_struct.make_mmcif_document().write_file(filename)
    else:
        out_struct.write_pdb(filename)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aligned_structure.py tests/test_save_pdb.py -v`
Expected: PASS (new tests pass; existing `test_save_pdb.py` regression still passes)

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py tests/test_aligned_structure.py
git commit -m "feat: AlignmentResult.aligned_structure() via shared builder"
```

---

### Task 2: Interpretation module (`interpretation.py`)

**Files:**
- Create: `pdb_align/interpretation.py`
- Test: `tests/test_interpretation.py` (create)

**Interfaces:**
- Produces (pure, gemmi-free):
  - `FlaggedRegion` dataclass: `chain: str`, `start_label: str`, `end_label: str`,
    `n_residues: int`, `max_rmsd: float`, `mean_rmsd: float`, `kind: str`.
  - `AlignmentQuality` dataclass (frozen): `band: str`, `verdict: str`,
    `confidence: str`, `flagged_regions: list[FlaggedRegion]`, `warnings: list[str]`.
    Method `to_dict() -> dict`.
  - `assess(*, tm_score, rmsd, coverage_pct, n_aligned, per_residue, chain_mapping, candidate_rmsds, hinge_regions=None) -> AlignmentQuality`
    where `per_residue` is a list of `(chain, residue_label, rmsd)` tuples,
    `candidate_rmsds` is a list of the seq-guided/seq-free RMSDs (may contain None),
    `hinge_regions` is an optional list of `(chain, start_label, end_label)`.
  - Module constants: `TM_EXCELLENT=0.9`, `TM_GOOD=0.5`, `TM_MODERATE=0.3`,
    `RMSD_EXCELLENT=1.0`, `RMSD_GOOD=2.5`, `RMSD_MODERATE=5.0`,
    `RMSD_FLAG_ABS=2.0`, `RMSD_FLAG_REL=2.0`, `LOW_COVERAGE=50.0`,
    `CANDIDATE_DISAGREE=1.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpretation.py
from pdb_align.interpretation import assess, AlignmentQuality, FlaggedRegion

def _uniform(chain, n, val):
    return [(chain, f"{chain}:{i}", val) for i in range(1, n + 1)]

def test_identical_is_excellent_high_confidence():
    q = assess(tm_score=0.99, rmsd=0.2, coverage_pct=100.0, n_aligned=100,
               per_residue=_uniform("A", 100, 0.2), chain_mapping=None,
               candidate_rmsds=[0.2, 0.2])
    assert q.band == "excellent"
    assert q.confidence == "high"
    assert q.flagged_regions == []
    assert not q.warnings

def test_flags_contiguous_high_rmsd_region():
    pr = _uniform("A", 20, 0.5)
    for k in (9, 10, 11, 12):  # residues A:10..A:13, well above max(2.0, 2*median)
        pr[k] = ("A", f"A:{k+1}", 6.0)
    q = assess(tm_score=0.7, rmsd=1.5, coverage_pct=100.0, n_aligned=20,
               per_residue=pr, chain_mapping=None, candidate_rmsds=[1.5, 1.6])
    assert len(q.flagged_regions) == 1
    fr = q.flagged_regions[0]
    assert fr.chain == "A" and fr.n_residues == 4
    assert fr.max_rmsd == 6.0 and fr.kind == "deviation"

def test_hinge_region_kind():
    q = assess(tm_score=0.7, rmsd=2.0, coverage_pct=100.0, n_aligned=50,
               per_residue=_uniform("A", 50, 1.0), chain_mapping=None,
               candidate_rmsds=[2.0, 2.1],
               hinge_regions=[("A", "A:20", "A:30")])
    assert any(fr.kind == "hinge" for fr in q.flagged_regions)

def test_low_coverage_warns_and_lowers_confidence():
    q = assess(tm_score=0.6, rmsd=2.0, coverage_pct=30.0, n_aligned=30,
               per_residue=_uniform("A", 30, 2.0), chain_mapping=None,
               candidate_rmsds=[2.0, 2.1])
    assert any("coverage" in w.lower() for w in q.warnings)
    assert q.confidence in ("low", "medium")

def test_candidate_disagreement_lowers_confidence():
    q = assess(tm_score=0.6, rmsd=1.0, coverage_pct=90.0, n_aligned=90,
               per_residue=_uniform("A", 90, 1.0), chain_mapping=None,
               candidate_rmsds=[1.0, 4.0])
    assert q.confidence in ("low", "medium")

def test_band_falls_back_to_rmsd_without_tm():
    q = assess(tm_score=None, rmsd=0.5, coverage_pct=100.0, n_aligned=50,
               per_residue=_uniform("A", 50, 0.5), chain_mapping=None,
               candidate_rmsds=[0.5, 0.5])
    assert q.band == "excellent"

def test_to_dict_roundtrips():
    q = assess(tm_score=0.99, rmsd=0.2, coverage_pct=100.0, n_aligned=100,
               per_residue=_uniform("A", 100, 0.2), chain_mapping=None,
               candidate_rmsds=[0.2, 0.2])
    d = q.to_dict()
    assert d["band"] == "excellent" and "verdict" in d and "flagged_regions" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpretation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdb_align.interpretation'`

- [ ] **Step 3: Implement the module**

```python
# pdb_align/interpretation.py
"""Plain-language interpretation of an alignment result.

Pure functions of numbers already produced by the aligner: no gemmi, no I/O.
Thresholds are module constants so they are reviewable and tunable in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

TM_EXCELLENT = 0.9
TM_GOOD = 0.5
TM_MODERATE = 0.3
RMSD_EXCELLENT = 1.0
RMSD_GOOD = 2.5
RMSD_MODERATE = 5.0
RMSD_FLAG_ABS = 2.0
RMSD_FLAG_REL = 2.0
LOW_COVERAGE = 50.0
CANDIDATE_DISAGREE = 1.0


@dataclass
class FlaggedRegion:
    chain: str
    start_label: str
    end_label: str
    n_residues: int
    max_rmsd: float
    mean_rmsd: float
    kind: str = "deviation"

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "start": self.start_label,
            "end": self.end_label,
            "n_residues": self.n_residues,
            "max_rmsd": round(self.max_rmsd, 3),
            "mean_rmsd": round(self.mean_rmsd, 3),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AlignmentQuality:
    band: str
    verdict: str
    confidence: str
    flagged_regions: List[FlaggedRegion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "flagged_regions": [fr.to_dict() for fr in self.flagged_regions],
            "warnings": list(self.warnings),
        }


def _band(tm_score: Optional[float], rmsd: Optional[float]) -> str:
    if tm_score is not None:
        if tm_score > TM_EXCELLENT:
            return "excellent"
        if tm_score > TM_GOOD:
            return "good"
        if tm_score > TM_MODERATE:
            return "moderate"
        return "poor"
    if rmsd is not None:
        if rmsd < RMSD_EXCELLENT:
            return "excellent"
        if rmsd < RMSD_GOOD:
            return "good"
        if rmsd < RMSD_MODERATE:
            return "moderate"
        return "poor"
    return "moderate"


def _flag_deviation_regions(per_residue) -> List[FlaggedRegion]:
    vals = [r for (_c, _l, r) in per_residue if r is not None]
    if not vals:
        return []
    thr = max(RMSD_FLAG_ABS, RMSD_FLAG_REL * median(vals))
    regions: List[FlaggedRegion] = []
    run: List[Tuple[str, str, float]] = []

    def flush():
        if not run:
            return
        rs = [x[2] for x in run]
        regions.append(FlaggedRegion(
            chain=run[0][0], start_label=run[0][1], end_label=run[-1][1],
            n_residues=len(run), max_rmsd=max(rs), mean_rmsd=sum(rs) / len(rs),
            kind="deviation"))

    prev_chain = None
    for chain, label, r in per_residue:
        if r is not None and r > thr:
            if prev_chain is not None and chain != prev_chain:
                flush(); run = []
            run.append((chain, label, r))
        else:
            flush(); run = []
        prev_chain = chain
    flush()
    return regions


def _confidence(coverage_pct, tm_pvalue, candidate_rmsds) -> str:
    score = 2  # start "high"
    if coverage_pct is not None and coverage_pct < LOW_COVERAGE:
        score -= 1
    rs = [r for r in (candidate_rmsds or []) if r is not None]
    if len(rs) >= 2 and (max(rs) - min(rs)) > CANDIDATE_DISAGREE:
        score -= 1
    if tm_pvalue is not None and tm_pvalue > 0.05:
        score -= 1
    return {2: "high", 1: "medium"}.get(max(score, 0), "low")


def _verdict(band, tm_score, rmsd, coverage_pct) -> str:
    phrases = {
        "excellent": "Near-identical structures",
        "good": "Same fold",
        "moderate": "Partial structural similarity",
        "poor": "Little structural similarity",
    }
    parts = [phrases[band]]
    if coverage_pct is not None and rmsd is not None:
        parts.append(f"{coverage_pct:.0f}% of residues superimpose within "
                     f"{rmsd:.1f} A")
    if tm_score is not None:
        parts.append(f"TM={tm_score:.2f}")
    return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def assess(*, tm_score, rmsd, coverage_pct, n_aligned, per_residue,
           chain_mapping, candidate_rmsds, hinge_regions=None,
           tm_pvalue=None) -> AlignmentQuality:
    band = _band(tm_score, rmsd)
    regions = _flag_deviation_regions(per_residue)
    if hinge_regions:
        for (chain, start, end) in hinge_regions:
            regions.append(FlaggedRegion(chain=chain, start_label=start,
                                         end_label=end, n_residues=0,
                                         max_rmsd=0.0, mean_rmsd=0.0, kind="hinge"))
    warnings: List[str] = []
    if coverage_pct is not None and coverage_pct < LOW_COVERAGE:
        warnings.append(f"Low coverage: only {coverage_pct:.0f}% of residues aligned.")
    if tm_score is None:
        warnings.append("TM-score unavailable; quality band derived from RMSD.")
    if chain_mapping is not None:
        n_pairs = len(chain_mapping)
        if n_pairs == 1:
            warnings.append("Only one chain pair aligned; multi-chain agreement not assessed.")
    confidence = _confidence(coverage_pct, tm_pvalue, candidate_rmsds)
    verdict = _verdict(band, tm_score, rmsd, coverage_pct)
    return AlignmentQuality(band=band, verdict=verdict, confidence=confidence,
                            flagged_regions=regions, warnings=warnings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpretation.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add pdb_align/interpretation.py tests/test_interpretation.py
git commit -m "feat: AlignmentQuality interpretation layer (pure, tested)"
```

---

### Task 3: Wire `AlignmentResult.quality` + report/JSON + CLI verdict

**Files:**
- Modify: `pdb_align/aligner.py` (add `quality` property; extend `report()` before
  its final `"=" * 52` line ~149; extend `to_dict()` ~106)
- Modify: `pdb_align/__main__.py` (print verdict line)
- Modify: `pdb_align/__init__.py` (export `AlignmentQuality`, `FlaggedRegion`)
- Test: `tests/test_quality_wiring.py` (create); `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `assess()` from Task 2; `summary_stats()`, `get_rmsd_df()`,
  `domains`, `tm_pvalue`, `_seqguided`, `_seqfree` from `AlignmentResult`.
- Produces: `AlignmentResult.quality -> AlignmentQuality` (lazy property);
  `to_dict()["quality"]`; a "Quality" block in `report(fmt="text")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality_wiring.py
import os
from pdb_align import PDBAligner
from pdb_align.interpretation import AlignmentQuality

DATA = os.path.join(os.path.dirname(__file__), "data")

def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")

def test_quality_property_returns_assessment():
    q = _result().quality
    assert isinstance(q, AlignmentQuality)
    assert q.band in ("excellent", "good", "moderate", "poor")

def test_report_text_contains_quality_block():
    txt = _result().report(fmt="text")
    assert "Quality" in txt and "Verdict" in txt

def test_to_dict_contains_quality():
    d = _result().to_dict()
    assert "quality" in d and "band" in d["quality"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quality_wiring.py -v`
Expected: FAIL — `AttributeError: 'AlignmentResult' object has no attribute 'quality'`

- [ ] **Step 3: Add the property and wiring**

In `pdb_align/aligner.py`, add near the other properties:

```python
@property
def quality(self):
    from pdb_align.interpretation import assess
    s = self.summary_stats()
    try:
        df = self.get_rmsd_df(on="reference")
        per_residue = list(zip(df["Chain"], df["Residue"], df["RMSD"]))
    except Exception:
        per_residue = []
    cand = []
    if self._seqguided:
        cand.append(self._seqguided.get("si", {}).get("rmsd")
                    if isinstance(self._seqguided.get("si"), dict) else None)
    if self._seqfree is not None:
        cand.append(getattr(self._seqfree, "rmsd", None))
    hinge = None
    if self.domains:
        hinge = [(d.chain_id, f"{d.chain_id}:{d.residue_start}",
                  f"{d.chain_id}:{d.residue_end}") for d in self.domains]
    return assess(
        tm_score=s.get("tm_score"), rmsd=s.get("rmsd"),
        coverage_pct=s.get("coverage_pct"), n_aligned=s.get("n_aligned"),
        per_residue=per_residue, chain_mapping=s.get("chain_mapping"),
        candidate_rmsds=cand, hinge_regions=hinge,
        tm_pvalue=self.tm_pvalue,
    )
```

In `to_dict()` (after `d["per_chain"] = ...`):

```python
        d["quality"] = self.quality.to_dict()
```

In `report()`, immediately before the final `lines.append("=" * 52)`:

```python
        q = self.quality
        lines.append("-" * 52)
        lines.append(f" Quality   : {q.band.upper()}  (confidence: {q.confidence})")
        lines.append(f" Verdict   : {q.verdict}")
        if q.flagged_regions:
            lines.append(" Flagged regions:")
            for fr in q.flagged_regions[:5]:
                lines.append(f"   {fr.chain} {fr.start_label}..{fr.end_label} "
                             f"({fr.kind}, max {fr.max_rmsd:.1f} A)")
        for w in q.warnings:
            lines.append(f" ! {w}")
```

In `pdb_align/__init__.py`, add to the imports/exports:

```python
from pdb_align.interpretation import AlignmentQuality, FlaggedRegion
```
and add `"AlignmentQuality"`, `"FlaggedRegion"` to `__all__`.

- [ ] **Step 4: Add CLI verdict line and test**

In `pdb_align/__main__.py`, after the text report is printed (non-JSON path),
print the verdict headline first. Locate where `res.report(...)` is printed and,
for the default text path, ensure the verdict is visible (it now lives inside
`report()`, so no extra print is required — but add a `-v` guarded full-quality
note). Add to `tests/test_cli.py`:

```python
def test_cli_text_output_shows_quality(tmp_path, capsys):
    import subprocess, sys, os
    DATA = os.path.join(os.path.dirname(__file__), "data")
    out = subprocess.run(
        [sys.executable, "-m", "pdb_align",
         os.path.join(DATA, "ref.pdb"), os.path.join(DATA, "mob.pdb")],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "Verdict" in out.stdout

def test_cli_json_contains_quality():
    import subprocess, sys, os, json
    DATA = os.path.join(os.path.dirname(__file__), "data")
    out = subprocess.run(
        [sys.executable, "-m", "pdb_align",
         os.path.join(DATA, "ref.pdb"), os.path.join(DATA, "mob.pdb"), "--json"],
        capture_output=True, text=True)
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert "quality" in payload and "band" in payload["quality"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_quality_wiring.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pdb_align/aligner.py pdb_align/__main__.py pdb_align/__init__.py \
        tests/test_quality_wiring.py tests/test_cli.py
git commit -m "feat: surface AlignmentQuality in result, report, JSON, and CLI"
```

---

### Task 4: `AlignmentResult.export_bundle()`

**Files:**
- Modify: `pdb_align/aligner.py` (add `export_bundle` + private script writers)
- Test: `tests/test_export_bundle.py` (create)

**Interfaces:**
- Consumes: `aligned_structure()`, `get_rmsd_df()`, `plot_summary()`,
  `plot_rmsd()`, `report()`, `to_json()` from earlier tasks.
- Produces: `AlignmentResult.export_bundle(path, include=None, fmt="zip") -> str`.
  `include` defaults to `["aligned","rmsd_csv","plots","pymol","chimerax","report"]`.
  Returns the path written. `fmt="dir"` writes a folder; `fmt="zip"` a `.zip`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_bundle.py
import os, zipfile, json
from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")

def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")

def test_export_bundle_zip_contains_all(tmp_path):
    out = _result().export_bundle(str(tmp_path / "bundle.zip"))
    assert os.path.exists(out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.endswith("aligned.pdb") for n in names)
        assert any(n.endswith("rmsd.csv") for n in names)
        assert any(n.endswith(".pml") for n in names)
        assert any(n.endswith(".cxc") for n in names)
        assert any(n.endswith("report.txt") for n in names)
        assert any(n.endswith("report.json") for n in names)
        with z.open([n for n in names if n.endswith("report.json")][0]) as f:
            payload = json.load(f)
            assert "quality" in payload

def test_export_bundle_dir_subset(tmp_path):
    out = _result().export_bundle(str(tmp_path / "b"), include=["rmsd_csv"], fmt="dir")
    assert os.path.isdir(out)
    assert os.path.exists(os.path.join(out, "rmsd.csv"))
    assert not os.path.exists(os.path.join(out, "aligned.pdb"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_bundle.py -v`
Expected: FAIL — `AttributeError: 'AlignmentResult' object has no attribute 'export_bundle'`

- [ ] **Step 3: Implement `export_bundle` and script writers**

```python
def _write_pymol_script(self, path, ref_name, aligned_name):
    lines = [
        f"load {ref_name}, ref",
        f"load {aligned_name}, mob",
        "hide everything",
        "show cartoon",
        "color grey70, ref",
        "spectrum b, blue_white_red, mob",
        "set cartoon_transparency, 0.1",
        "zoom",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def _write_chimerax_script(self, path, ref_name, aligned_name):
    lines = [
        f"open {ref_name}",
        f"open {aligned_name}",
        "hide atoms",
        "show cartoons",
        "color #1 grey",
        "color byattribute bfactor #2 palette blue:white:red",
        "view",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def export_bundle(self, path, include=None, fmt="zip"):
    """Write a reproducible bundle of alignment outputs.

    include components: aligned, rmsd_csv, plots, pymol, chimerax, report.
    fmt="zip" writes a .zip; fmt="dir" writes a folder. Returns the path.
    """
    import os, tempfile, zipfile, shutil
    components = include or ["aligned", "rmsd_csv", "plots", "pymol",
                             "chimerax", "report"]
    workdir = tempfile.mkdtemp(prefix="pdb_align_bundle_")
    ref_name = "aligned.pdb"
    try:
        if "aligned" in components:
            self.aligned_structure(color_by="rmsd").write_pdb(
                os.path.join(workdir, "aligned.pdb"))
        if "rmsd_csv" in components:
            self.get_rmsd_df().to_csv(os.path.join(workdir, "rmsd.csv"),
                                      index=False)
        if "plots" in components:
            try:
                self.plot_summary(os.path.join(workdir, "summary.png"))
                self.plot_rmsd(os.path.join(workdir, "rmsd.png"))
            except Exception:
                pass
        if "pymol" in components:
            self._write_pymol_script(os.path.join(workdir, "view.pml"),
                                     "aligned.pdb", ref_name)
        if "chimerax" in components:
            self._write_chimerax_script(os.path.join(workdir, "view.cxc"),
                                        "aligned.pdb", ref_name)
        if "report" in components:
            with open(os.path.join(workdir, "report.txt"), "w") as f:
                f.write(self.report(fmt="text") + "\n")
            with open(os.path.join(workdir, "report.json"), "w") as f:
                f.write(self.to_json())

        if fmt == "dir":
            if os.path.isdir(path):
                shutil.rmtree(path)
            shutil.copytree(workdir, path)
            return path
        zpath = path if path.endswith(".zip") else path + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(workdir)):
                z.write(os.path.join(workdir, fn), fn)
        return zpath
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export_bundle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py tests/test_export_bundle.py
git commit -m "feat: AlignmentResult.export_bundle reproducible outputs"
```

---

### Task 5: `EnsembleResult.export_bundle()`

**Files:**
- Modify: `pdb_align/aligner.py` (`EnsembleResult`, ~lines 820-1040)
- Test: `tests/test_ensemble_bundle.py` (create)

**Interfaces:**
- Consumes: `EnsembleResult.summary()`, `rmsd_matrix()`, `cluster()`,
  `plot_pca()`, `plot_dendrogram()`.
- Produces: `EnsembleResult.export_bundle(path, fmt="zip") -> str` writing
  `summary.csv`, `rmsd_matrix.csv`, `clusters.csv`, `pca.png`, `dendrogram.png`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ensemble_bundle.py
import os, zipfile
from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")

def _ensemble():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    return al.align_ensemble([os.path.join(DATA, "mob.pdb"),
                              os.path.join(DATA, "ref.pdb")], mode="auto")

def test_ensemble_bundle_contains_tables(tmp_path):
    out = _ensemble().export_bundle(str(tmp_path / "ens.zip"))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.endswith("summary.csv") for n in names)
        assert any(n.endswith("rmsd_matrix.csv") for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ensemble_bundle.py -v`
Expected: FAIL — `AttributeError: 'EnsembleResult' object has no attribute 'export_bundle'`

- [ ] **Step 3: Implement**

Add to `EnsembleResult`:

```python
def export_bundle(self, path, fmt="zip"):
    import os, tempfile, zipfile, shutil
    workdir = tempfile.mkdtemp(prefix="pdb_align_ens_")
    try:
        self.summary().to_csv(os.path.join(workdir, "summary.csv"), index=False)
        self.rmsd_matrix().to_csv(os.path.join(workdir, "rmsd_matrix.csv"))
        try:
            import pandas as pd
            labels = self.cluster()
            pd.DataFrame({"model": self.labels, "cluster": labels}).to_csv(
                os.path.join(workdir, "clusters.csv"), index=False)
        except Exception:
            pass
        for meth, fn in ((self.plot_pca, "pca.png"),
                         (self.plot_dendrogram, "dendrogram.png")):
            try:
                meth(save_path=os.path.join(workdir, fn))
            except Exception:
                pass
        if fmt == "dir":
            if os.path.isdir(path):
                shutil.rmtree(path)
            shutil.copytree(workdir, path)
            return path
        zpath = path if path.endswith(".zip") else path + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(workdir)):
                z.write(os.path.join(workdir, fn), fn)
        return zpath
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ensemble_bundle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pdb_align/aligner.py tests/test_ensemble_bundle.py
git commit -m "feat: EnsembleResult.export_bundle"
```

---

### Task 6: Convergence guard + accessor mapping doc

**Files:**
- Create: `tests/test_convergence_guard.py`
- Create: `docs/superpowers/specs/sp1-accessor-mapping.md`

**Interfaces:**
- Produces: an xfail test asserting the GUI does not import `pdb_align.core`
  (flips to strict in SP2), and a table mapping GUI-computed quantities to API
  accessors.

- [ ] **Step 1: Write the (xfail) guard test**

```python
# tests/test_convergence_guard.py
import os, re
import pytest

GUI = os.path.join(os.path.dirname(__file__), "..", "struct_pair_align.py")

@pytest.mark.xfail(reason="GUI still imports core; converged in SP2", strict=False)
def test_gui_does_not_import_core():
    with open(GUI) as f:
        src = f.read()
    assert not re.search(r"from\s+pdb_align\.core\s+import", src)
    assert not re.search(r"import\s+pdb_align\.core", src)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_convergence_guard.py -v`
Expected: XFAIL (expected failure — GUI still imports core)

- [ ] **Step 3: Write the accessor mapping doc**

```markdown
# SP1 Accessor Mapping (GUI-computed → API accessor)

This is SP2's build checklist: every quantity the current Streamlit app derives
from `pdb_align.core` must be read from these API accessors instead.

| GUI need                         | API accessor                                  |
|----------------------------------|-----------------------------------------------|
| Run alignment (any mode)         | `PDBAligner.align(mode, strategy, atoms, ...)` |
| Per-residue RMSD table           | `AlignmentResult.get_rmsd_df(on=...)`         |
| Top deviation peaks              | `AlignmentResult.report_peaks(...)`           |
| Aligned coords (ref/mob)         | `AlignmentResult.get_aligned_coords()`        |
| Transformed structure for 3D     | `AlignmentResult.aligned_structure(color_by)` |
| Matched residue pairs            | `AlignmentResult.get_matched_pairs()`         |
| Summary numbers (rmsd/tm/cov)    | `AlignmentResult.summary_stats()`             |
| Per-chain RMSD                   | `AlignmentResult.per_chain`                   |
| Flexible domains / hinges        | `AlignmentResult.domains`                     |
| Sequence alignment text          | `AlignmentResult.get_sequence_alignment()`    |
| Quality verdict / flags          | `AlignmentResult.quality`                     |
| Downloads (zip, scripts, report) | `AlignmentResult.export_bundle(...)`          |
| Ensemble tables/plots            | `EnsembleResult.*` + `.export_bundle(...)`    |

No remaining GUI quantity requires importing `pdb_align.core`.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_convergence_guard.py docs/superpowers/specs/sp1-accessor-mapping.md
git commit -m "test: convergence guard (xfail) + accessor mapping doc"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Update README**

Add under the Python section: `result.quality` (verdict/band/confidence/flags),
`result.aligned_structure(color_by=...)`, and `result.export_bundle("out.zip")`.
Show a 4-line example ending in `print(result.quality.verdict)`.

- [ ] **Step 2: Update CLAUDE.md**

In the `aligner.py` section, document the new `quality` property (backed by
`interpretation.py`), `aligned_structure()`, and `export_bundle()`. Add a bullet
for the new `pdb_align/interpretation.py` module (pure, threshold constants).

- [ ] **Step 3: Run full suite**

Run: `pytest tests/`
Expected: PASS (xfail on convergence guard is expected)

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document quality, aligned_structure, export_bundle"
```

---

## Self-Review

**Spec coverage:**
- Section A1 (`aligned_structure`) → Task 1. ✅
- Section A2 (accessor audit) → Task 6 mapping doc. ✅
- Section A3 (no-core-leakage guard) → Task 6 xfail test. ✅
- Section B (`AlignmentQuality`, thresholds, fields, surfacing) → Tasks 2 & 3. ✅
- Section C (`export_bundle` for result + ensemble) → Tasks 4 & 5. ✅
- Section D (tests, regression, acceptance) → tests in every task; full suite in Task 7. ✅

**Placeholder scan:** All code steps contain real code; no TBD/TODO. Plot steps
wrapped in try/except because plotting deps are optional (`[app]` extra). ✅

**Type consistency:** `assess(...)` signature in Task 2 matches its call in Task 3.
`aligned_structure(color_by=...)` defined in Task 1, consumed in Task 4.
`export_bundle(path, include, fmt)` consistent across Tasks 4/5 and doc in Task 6. ✅

**Note for implementer:** Verify the exact key holding each candidate RMSD
(`self._seqguided["si"]["rmsd"]` and `self._seqfree.rmsd`) against the code before
relying on it in Task 3; if a key differs, adjust the `cand` assembly — the
`assess()` contract (a list possibly containing `None`) is unchanged.
