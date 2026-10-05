"""Tests for robustness/correctness fixes.

Covers:
- PDBAligner.get_log/save_log working off last_result (previously AttributeError)
- pick_best_overall counting residues, not atoms, for backbone/all_heavy
- coverage-weighted internal shape-vs-window selection
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pdb_align.aligner import PDBAligner
from pdb_align.core import AlignSummary, _select_seqfree_method, pick_best_overall

_PDB = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""


def _make_aligner(tmp_path):
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(_PDB)
    mob.write_text(_PDB)
    aligner = PDBAligner()
    aligner.set_reference(str(ref))
    aligner.add_mobile(str(mob))
    return aligner


def test_pdbaligner_get_log_after_align(tmp_path):
    """PDBAligner.get_log() must work after align() (was AttributeError)."""
    aligner = _make_aligner(tmp_path)
    aligner.align()
    log = aligner.get_log()
    assert "PDB Aligner Result Log" in log
    assert "RMSD" in log


def test_pdbaligner_get_log_without_result_raises(tmp_path):
    """get_log() before align() should raise a clear ValueError, not AttributeError."""
    aligner = _make_aligner(tmp_path)
    try:
        aligner.get_log()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError before alignment")


def _seqguided(rmsd, n_atoms):
    return {"si": {"rmsd": rmsd}, "ref_atoms": [SimpleNamespace(get_name=lambda: "CA")] * n_atoms}


def test_pick_best_counts_residues_not_atoms(tmp_path):
    """With backbone atoms, seqguided ref_atoms hold ~4 atoms/residue.

    seqguided covers only 30 residues (120 backbone atoms) at 2.0 A; seqfree
    covers 60 residues at 1.5 A. Counting atoms inflates seqguided's coverage
    enough to win; counting residues (the correct unit) makes seqfree win.
    """
    # 30 residues -> 120 backbone atoms (N, CA, C, O each)
    ref_atoms = []
    for _ in range(30):
        for name in ("N", "CA", "C", "O"):
            ref_atoms.append(SimpleNamespace(get_name=(lambda n=name: n)))
    seqguided = {"si": {"rmsd": 2.0}, "ref_atoms": ref_atoms}
    seqfree = SimpleNamespace(rmsd=1.5, method="shape", kept_pairs=60)

    best, _ = pick_best_overall(seqguided, seqfree)
    assert best["kind"] == "seqfree", (
        "seqguided coverage should be counted in residues (30), not atoms (120)"
    )


def _pdb_from_coords(coords):
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_seqguided_recycles_forwarded_rejects_outlier(tmp_path):
    """recycles/keep_fraction must reach the seq-guided superposition so that
    outlier rejection lowers the reported RMSD, just as it does for seq-free."""
    base = [(float(i) * 3.8, 0.0, 0.0) for i in range(6)]
    mob = list(base)
    mob[-1] = (base[-1][0], 20.0, 0.0)  # displace last CA by 20 A

    ref = tmp_path / "ref.pdb"
    mobf = tmp_path / "mob.pdb"
    ref.write_text(_pdb_from_coords(base))
    mobf.write_text(_pdb_from_coords(mob))

    aligner = PDBAligner()
    aligner.set_reference(str(ref))
    aligner.add_mobile(str(mobf))

    r_no_recycle = aligner.align(mode="seq_guided").rmsd
    r_recycle = aligner.align(mode="seq_guided", recycles=5, keep_fraction=0.5).rmsd

    assert r_recycle < r_no_recycle - 1.0, (
        f"recycles should reject the outlier: {r_recycle} vs {r_no_recycle}"
    )


def test_select_seqfree_prefers_coverage_over_tiny_rmsd():
    """A shape match of 10 residues @0.3 A must not beat a window match of
    200 residues @1.2 A — the internal selection must be coverage-weighted too."""
    summaries = {
        "shape": AlignSummary("shape", rmsd=0.3, inliers=10, total_pairs=10, iterations=1),
        "window": AlignSummary("window", rmsd=1.2, inliers=200, total_pairs=200, iterations=1),
    }
    assert _select_seqfree_method(summaries) == "window"


def test_fetch_writes_to_cache_dir_not_cwd(tmp_path, monkeypatch):
    """Fetched structures must land in the configured cache dir, not the CWD."""
    aligner = PDBAligner()
    aligner._fetch_cache_dir = str(tmp_path / "cache")

    written = {}

    def fake_download(url, dest, what):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write("dummy")
        written["dest"] = dest

    monkeypatch.setattr(aligner, "_download", fake_download)

    path = aligner._fetch_structure("pdb:1abc")
    assert path.startswith(str(tmp_path / "cache"))
    assert os.path.exists(path)


def test_fetch_alphafold_version_fallback(tmp_path, monkeypatch):
    """If the newest AlphaFold model version 404s, fall back to older ones."""
    aligner = PDBAligner()
    aligner._fetch_cache_dir = str(tmp_path / "cache")

    attempted = []

    def fake_download(url, dest, what):
        attempted.append(url)
        # Fail on the newest version, succeed on an older one.
        if "model_v6" in url or "model_v5" in url:
            raise ValueError(f"Could not fetch {what}: 404")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write("dummy")

    monkeypatch.setattr(aligner, "_download", fake_download)

    path = aligner._fetch_structure("af:P12345")
    assert os.path.exists(path)
    assert any("model_v6" in u for u in attempted), "should try newest first"
    assert len(attempted) >= 2, "should fall back after the newest version fails"


def test_struct_cache_invalidates_on_file_change(tmp_path):
    """Editing a reference file on disk must cause a re-parse, not a stale hit."""
    from unittest.mock import patch

    import gemmi

    p = tmp_path / "ref.pdb"
    coords_a = [(float(i) * 3.8, 0.0, 0.0) for i in range(4)]
    p.write_text(_pdb_from_coords(coords_a))

    aligner = PDBAligner()
    aligner.set_reference(str(p))
    n_first = sum(aligner.ref_lens.values())

    # Rewrite the file with a different residue count and a newer mtime.
    coords_b = [(float(i) * 3.8, 0.0, 0.0) for i in range(8)]
    p.write_text(_pdb_from_coords(coords_b))
    os.utime(str(p), (os.path.getmtime(str(p)) + 10, os.path.getmtime(str(p)) + 10))

    with patch("pdb_align.aligner._parse_path", wraps=lambda pp: gemmi.read_structure(pp)) as mock_parse:
        aligner.set_reference(str(p))
        mock_parse.assert_called_once()
    assert sum(aligner.ref_lens.values()) == n_first * 2


def test_seqfree_failure_is_logged_not_printed(tmp_path, caplog):
    """A seq-free failure in align() should be logged, not dumped to stdout."""
    import logging
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(_PDB)
    mob.write_text(_PDB)

    aligner = PDBAligner()
    aligner.set_reference(str(ref))
    aligner.add_mobile(str(mob))

    from unittest.mock import patch
    with patch("pdb_align.aligner.sequence_independent_alignment_joined_v2",
               side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING, logger="pdb_align.aligner"):
            # seq_guided still succeeds, so align() returns a result
            aligner.align(mode="auto")

    assert any("boom" in rec.getMessage() or "seq" in rec.getMessage().lower()
               for rec in caplog.records), "seq-free failure should be logged"


def _backbone_residue(r, ca_xyz, present=("N", "CA", "C", "O")):
    """Emit ATOM lines (as list) for one all-ALA residue r with the given CA."""
    cx, cy, cz = ca_xyz
    coords = {"N": (cx - 1.0, cy, cz), "CA": (cx, cy, cz),
              "C": (cx + 1.0, cy, cz), "O": (cx + 1.0, cy + 1.0, cz)}
    out = []
    for name in ("N", "CA", "C", "O"):
        if name not in present:
            continue
        x, y, z = coords[name]
        out.append((r, name, x, y, z))
    return out


def _write_pdb(records, path):
    lines = []
    for serial, (r, name, x, y, z) in enumerate(records, start=1):
        lines.append(
            f"ATOM  {serial:>5}  {name:<3} ALA A{r:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_backbone_mode_handles_ca_less_residue(tmp_path):
    """A reference residue lacking CA must be dropped from the residue list so it
    stays in lockstep with the CA-only alignment sequence. Otherwise the CA-less
    residue is wrongly paired and RMSD blows up."""
    # CA x-positions widely spaced so any off-by-one pairing is huge.
    xs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    # Mobile: 7 normal residues at those CA positions.
    mob_records = []
    for i, x in enumerate(xs, start=1):
        mob_records += _backbone_residue(i, (x, 0.0, 0.0))
    _write_pdb(mob_records, tmp_path / "mob.pdb")

    # Reference: same 7 CA-bearing residues, plus a CA-less residue (only N/C/O)
    # inserted as residue 4 between the CA=30 and CA=40 residues.
    ref_records = []
    ref_records += _backbone_residue(1, (xs[0], 0.0, 0.0))
    ref_records += _backbone_residue(2, (xs[1], 0.0, 0.0))
    ref_records += _backbone_residue(3, (xs[2], 0.0, 0.0))
    ref_records += _backbone_residue(4, (35.0, 5.0, 0.0), present=("N", "C", "O"))  # no CA
    ref_records += _backbone_residue(5, (xs[3], 0.0, 0.0))
    ref_records += _backbone_residue(6, (xs[4], 0.0, 0.0))
    ref_records += _backbone_residue(7, (xs[5], 0.0, 0.0))
    ref_records += _backbone_residue(8, (xs[6], 0.0, 0.0))
    _write_pdb(ref_records, tmp_path / "ref.pdb")

    aligner = PDBAligner()
    aligner.set_reference(str(tmp_path / "ref.pdb"))
    aligner.add_mobile(str(tmp_path / "mob.pdb"))
    res = aligner.align(mode="seq_guided", atoms="backbone")
    assert res.rmsd is not None
    assert res.rmsd < 1e-6, f"CA-less residue should be skipped; got RMSD {res.rmsd}"


def test_chain_index_selection_is_1_based(tmp_path):
    """A numeric chain selector is a 1-based index, consistently everywhere.

    (The dead StructureBase wrapper this used to test was removed in 0.4.0;
    select_residues() is now the single implementation.)"""
    from pdb_align.core import _parse_path, select_residues

    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C",
        "END",
    ]
    p = tmp_path / "two.pdb"
    p.write_text("\n".join(lines) + "\n")
    st = _parse_path(str(p))

    assert select_residues(st, [1]).chain_order == ["A"]
    assert select_residues(st, [2]).chain_order == ["B"]

    for bad in (0, 3):
        with pytest.raises(ValueError, match="out of range"):
            select_residues(st, [bad])


def test_pdbaligner_save_aligned_pdb_preserve_bfactor(tmp_path):
    """PDBAligner.save_aligned_pdb must honour preserve_bfactor (delegates to
    the AlignmentResult implementation instead of duplicating it)."""
    import gemmi
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    # Non-zero input B-factors (e.g. pLDDT) to check preservation.
    content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 55.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00 66.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00 77.00           C
END
"""
    ref.write_text(content)
    mob.write_text(content)

    aligner = PDBAligner()
    aligner.set_reference(str(ref))
    aligner.add_mobile(str(mob))
    aligner.align()

    out = tmp_path / "aligned.pdb"
    aligner.save_aligned_pdb(str(out), preserve_bfactor=True)

    st = gemmi.read_structure(str(out))
    bvals = sorted(round(a.b_iso) for ch in st[0] for r in ch for a in r)
    assert bvals == [55, 66, 77], f"input B-factors should be preserved, got {bvals}"
