"""Tests for pick_best_overall selection logic.

Selecting purely by lowest RMSD is a known pitfall: a strategy that aligns only
a handful of residues can post a near-zero RMSD and beat a strategy that
correctly superimposes the whole protein. Selection must reward coverage as
well as accuracy.
"""
from types import SimpleNamespace

from pdb_align.core import pick_best_overall


def _seqguided(rmsd, n_pairs):
    return {"si": {"rmsd": rmsd}, "ref_atoms": [None] * n_pairs}


def _seqfree(rmsd, n_pairs, method="shape"):
    return SimpleNamespace(rmsd=rmsd, method=method, kept_pairs=n_pairs)


def test_prefers_full_coverage_over_tiny_fragment():
    # seqfree nails 5 residues; seqguided aligns 200 residues well.
    seqfree = _seqfree(0.2, 5)
    seqguided = _seqguided(1.5, 200)
    best, _ = pick_best_overall(seqguided, seqfree)
    assert best["kind"] == "seqguided"


def test_prefers_lower_rmsd_when_coverage_equal():
    # Equal coverage → the tighter fit should win.
    seqfree = _seqfree(2.0, 100)
    seqguided = _seqguided(1.0, 100)
    best, _ = pick_best_overall(seqguided, seqfree)
    assert best["kind"] == "seqguided"


def test_single_candidate_returned():
    best, _ = pick_best_overall(None, _seqfree(1.0, 50))
    assert best["kind"] == "seqfree"
