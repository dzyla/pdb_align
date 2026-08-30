"""Golden-value tests for the corrected metrics layer."""
import numpy as np
import pytest

from pdb_align.metrics import (
    calculate_tm_pvalue, calculate_tm_score, calculate_lddt,
    tm_optimal_superposition, compute_d0,
)
from pdb_align.core import compute_gdt_ts, compute_contact_overlap


# --- TM p-value: Xu & Zhang 2010 EVD (mu=0.1512, sigma=0.0242) --------------

def test_tm_pvalue_published_golden_value():
    # The paper states P(TM >= 0.5) = 5.5e-7 for its fitted EVD.
    p = calculate_tm_pvalue(0.5, 200)
    assert p == pytest.approx(5.5e-7, rel=0.15)


def test_tm_pvalue_at_random_mean_is_large():
    # Random pairs have TM ~ 0.15-0.17; the p-value there must be large.
    assert calculate_tm_pvalue(0.1512, 200) > 0.3


def test_tm_pvalue_same_fold_threshold_significant():
    # TM = 0.5 is the classic same-fold threshold: strongly significant.
    assert calculate_tm_pvalue(0.5, 200) < 1e-5


# --- TM-optimal superposition ------------------------------------------------

def _line(n, spacing=3.8):
    xs = np.arange(n) * spacing
    return np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)


def test_tm_optimal_identity_is_one():
    P = _line(50)
    tm, R, t = tm_optimal_superposition(P, P.copy(), 50)
    assert tm == pytest.approx(1.0, abs=1e-9)


def test_tm_optimal_beats_kabsch_frame_on_hinge():
    # 60 residues: first 30 identical, last 30 displaced far away. The
    # RMSD-optimal (Kabsch) frame compromises both halves; the TM-optimal
    # superposition must fit the conserved half essentially perfectly.
    N = 60
    P = _line(N)
    Q = P.copy()
    Q[30:, 1] += 25.0  # move the second half 25 A off
    from pdb_align.core import _kabsch
    R, t, _ = _kabsch(P, Q)
    Q_kabsch = (R @ Q.T).T + t
    tm_fixed = calculate_tm_score(P, Q_kabsch, N)
    tm_opt, _, _ = tm_optimal_superposition(P, Q, N)
    assert tm_opt > tm_fixed
    # conserved half fit perfectly: each of the 30 residues contributes ~1/N
    assert tm_opt >= 30 / N * 0.99


def test_tm_optimal_invariant_to_input_frame():
    # Feeding Q in a rotated/translated frame must not change the score.
    P = _line(40)
    Q = P.copy()
    Q[20:, 1] += 10.0
    theta = 0.7
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Q_moved = (Rz @ Q.T).T + np.array([5.0, -3.0, 8.0])
    tm1, _, _ = tm_optimal_superposition(P, Q, 40)
    tm2, _, _ = tm_optimal_superposition(P, Q_moved, 40)
    assert tm1 == pytest.approx(tm2, abs=1e-6)


# --- GDT_TS normalization ----------------------------------------------------

def test_gdt_normalized_by_reference_length():
    # 4 matched residues at distance 0, but the reference has 8 residues:
    # every cutoff fraction is 4/8, so GDT_TS = 50 (CASP semantics), not 100.
    dists = np.zeros(4)
    assert compute_gdt_ts(dists, n_total=8) == pytest.approx(50.0)
    # backward-compatible default: normalize by the matched count
    assert compute_gdt_ts(dists) == pytest.approx(100.0)


def test_gdt_never_normalizes_by_inlier_subset():
    # Half the residues fit perfectly, half are 20 A off, full-length target:
    # GDT_TS must be 50, not 100 (which the old inlier normalization gave).
    dists = np.concatenate([np.zeros(10), np.full(10, 20.0)])
    assert compute_gdt_ts(dists, n_total=20) == pytest.approx(50.0)


# --- lDDT chunking ------------------------------------------------------------

def _lddt_reference_impl(P, Q, threshold=15.0):
    n = len(P)
    rd = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    md = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
    mask = (rd < threshold) & ~np.eye(n, dtype=bool)
    diffs = np.abs(rd - md)
    tot = mask.sum()
    pres = sum(int(((diffs < t) & mask).sum()) for t in (0.5, 1.0, 2.0, 4.0))
    return pres / (4 * tot)


def test_lddt_chunked_matches_full_matrix():
    rng = np.random.default_rng(7)
    P = rng.normal(size=(300, 3)) * 15
    Q = P + rng.normal(size=(300, 3)) * 0.8
    assert calculate_lddt(P, Q, chunk=64) == pytest.approx(
        _lddt_reference_impl(P, Q), abs=1e-12)


def test_lddt_identity_is_one():
    P = _line(80)
    assert calculate_lddt(P, P.copy()) == pytest.approx(1.0)


# --- contact overlap rename ----------------------------------------------------

def test_contact_overlap_alias_warns():
    from pdb_align.core import compute_cad_score_approx
    P = _line(10)
    with pytest.warns(DeprecationWarning):
        v = compute_cad_score_approx(P, P.copy())
    assert v == pytest.approx(compute_contact_overlap(P, P.copy()))
