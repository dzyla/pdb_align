import numpy as np
import pytest

from pdb_align.metrics import calculate_tm_pvalue, calculate_tm_score, compute_d0


def test_compute_d0_normal_length():
    # Standard TM-align formula for a long chain.
    L = 300
    expected = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    assert compute_d0(L) == pytest.approx(expected)


def test_compute_d0_clamped_for_short_chains():
    # For L=16 the raw formula gives d0 = 1.24*1 - 1.8 = -0.56, which is
    # physically invalid. TM-align clamps d0 to a minimum of 0.5.
    assert compute_d0(16) == pytest.approx(0.5)
    # Also clamped through the 16..~20 danger zone where raw d0 < 0.5.
    assert compute_d0(20) == pytest.approx(0.5)


def test_tm_pvalue_random_level_is_not_significant():
    # A random pair of structures has TM ~ 0.17; its p-value must be near 1
    # (clearly NOT significant), not a vanishingly small number.
    p = calculate_tm_pvalue(0.17, 200)
    assert 0.2 < p <= 1.0


def test_tm_pvalue_strong_match_is_significant():
    # A strong structural match (TM=0.9) must be highly significant.
    assert calculate_tm_pvalue(0.9, 200) < 1e-4


def test_tm_pvalue_monotonic_decreasing():
    # Higher TM-score → smaller (more significant) p-value.
    assert calculate_tm_pvalue(0.3, 200) > calculate_tm_pvalue(0.6, 200)


def test_tm_pvalue_bounds():
    assert calculate_tm_pvalue(-0.1, 200) == 1.0
    assert calculate_tm_pvalue(0.5, 10) == 1.0  # too short to be meaningful
    assert 0.0 <= calculate_tm_pvalue(0.99, 200) <= 1.0


def test_tm_score_uses_clamped_d0():
    # 16 residues each displaced 1.0 A. With clamped d0=0.5:
    #   per-term = 1/(1+(1/0.5)^2) = 0.2  ->  score = 0.2
    L = 16
    ref = np.zeros((L, 3))
    mob = np.zeros((L, 3))
    mob[:, 0] = 1.0
    assert calculate_tm_score(ref, mob, L) == pytest.approx(0.2, abs=1e-9)
