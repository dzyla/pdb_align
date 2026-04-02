import os
import pytest
import numpy as np

from pdb_align.core import (
    compute_gdt_ts,
    compute_cad_score_approx,
    progressive_align_ensemble,
    _extract_ca_infos,
    _parse_path,
    _kabsch
)

def test_compute_gdt_ts():
    # If distance is perfectly 0, all cutoffs are satisfied
    dists = np.array([0.0, 0.5, 1.5, 3.0, 5.0, 10.0])
    # Cutoffs: 1.0, 2.0, 4.0, 8.0
    # <= 1.0: 2/6
    # <= 2.0: 3/6
    # <= 4.0: 4/6
    # <= 8.0: 5/6
    # Mean fractions: (2/6 + 3/6 + 4/6 + 5/6) / 4 = 14 / 24 = 0.5833
    score = compute_gdt_ts(dists)
    assert np.isclose(score, 58.333, atol=0.1)

def test_compute_cad_score_approx():
    # Mock some points. 1, 2, 3 in a line
    ref = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [6.0, 0.0, 0.0]
    ])
    # Exact same mob
    cad = compute_cad_score_approx(ref, ref, contact_dist=8.0)
    # Jaccard index for self should be 1.0
    assert cad == 1.0

    # If completely far apart, cad is 0
    mob = np.array([
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
        [300.0, 0.0, 0.0]
    ])
    cad_far = compute_cad_score_approx(ref, mob, contact_dist=8.0)
    assert cad_far == 0.0

def test_kabsch():
    ref = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    # Rotate 90 deg around Z
    mob = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    
    R, t, rmsd = _kabsch(ref, mob)
    assert np.isclose(rmsd, 0.0, atol=1e-6)
    
def test_progressive_align_error():
    res = progressive_align_ensemble(["file.pdb"])
    assert "error" in res
