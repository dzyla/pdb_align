"""Cross-validation against reference implementations.

These are the tests that make the numbers citable: agreement with the
published implementations on non-trivial decoys, to tight tolerances, across
the whole range of each score — not just on identity cases.

Skipped when the reference packages are absent, so the suite still runs on a
core install:

- DockQ components vs the official Wallner-lab implementation (``pip DockQ``).
- TM-score vs the real TM-align code (``pip tmtools``).
"""
import numpy as np
import pytest
from _synthetic import two_chain_complex

dockq_pkg = pytest.importorskip("DockQ.DockQ", reason="official DockQ not installed")
from pdb_align.interface import capri_class, compute_dockq


def _official_dockq(model_path, native_path, chain_map):
    # NB: DockQ.load_PDB caches by path, so every decoy needs its own filename.
    m = dockq_pkg.load_PDB(str(model_path))
    n = dockq_pkg.load_PDB(str(native_path))
    res, _total = dockq_pkg.run_on_all_native_interfaces(m, n, chain_map=chain_map)
    assert len(res) == 1
    return next(iter(res.values()))


# Decoys spanning DockQ 0.02 -> 1.00 and every CAPRI class. Each perturbs the
# ligand helix differently: along the interface, away from it, by rotating its
# axis, and by rotating its helical phase (which changes which atoms face the
# receptor without moving the chain's centre of mass).
DECOYS = {
    "native": {},
    "slid_and_lifted": dict(lig_origin=(1.5, 9.5, 0.0)),
    "pushed_off": dict(lig_origin=(0.0, 11.0, 0.0)),
    "axis_rotated": dict(lig_origin=(0.0, 8.0, 0.0), lig_axis=(0.9, 0.0, 0.44)),
    "phase_rotated": dict(lig_origin=(0.0, 8.0, 0.0), lig_phase=np.pi),
    "far_away": dict(lig_origin=(0.0, 40.0, 0.0)),
}


@pytest.mark.parametrize("name", list(DECOYS))
def test_dockq_components_match_official_implementation(tmp_path, name):
    native = two_chain_complex()
    model = two_chain_complex(**DECOYS[name])
    p_native = tmp_path / "native.pdb"
    p_model = tmp_path / f"model_{name}.pdb"
    native.write_pdb(str(p_native))
    model.write_pdb(str(p_model))

    ours = compute_dockq(str(p_native), str(p_model), ["A"], ["B"])
    ref = _official_dockq(p_model, p_native, {"A": "A", "B": "B"})

    # fnat is a ratio of contact counts, so it must agree exactly. The RMSDs
    # go through a different SVD in each implementation, which leaves ~1e-6 of
    # floating-point noise (visible as 1.7e-14 vs 1.4e-6 on an exact match).
    assert ours.fnat == pytest.approx(ref["fnat"], abs=1e-9)
    assert ours.irmsd == pytest.approx(ref["iRMSD"], abs=1e-4)
    assert ours.lrmsd == pytest.approx(ref["LRMSD"], abs=1e-4)
    assert ours.dockq == pytest.approx(ref["DockQ"], abs=1e-5)
    assert ours.capri == capri_class(ref["DockQ"])


def test_dockq_decoys_span_the_whole_score_range(tmp_path):
    """Guards the cross-validation itself: if every decoy scored ~1.0 the
    comparison above would pass while testing nothing."""
    native = two_chain_complex()
    p_native = tmp_path / "native.pdb"
    native.write_pdb(str(p_native))
    scores = []
    for name, kw in DECOYS.items():
        p_model = tmp_path / f"m_{name}.pdb"
        two_chain_complex(**kw).write_pdb(str(p_model))
        scores.append(compute_dockq(str(p_native), str(p_model), ["A"], ["B"]).dockq)
    assert min(scores) < 0.10 and max(scores) > 0.95
    assert len({capri_class(s) for s in scores}) >= 3


# --- TM-score -----------------------------------------------------------------

def _tmtools():
    return pytest.importorskip("tmtools", reason="tmtools not installed")


def _tm_cases():
    rng = np.random.default_rng(3)
    cases = {}
    N = 120
    P = np.cumsum(rng.normal(scale=2.2, size=(N, 3)), axis=0)
    Q = P + rng.normal(scale=0.7, size=(N, 3))
    Q[90:] += np.array([18.0, 5.0, -7.0])  # swung-out tail
    cases["hinged_tail"] = (P, Q)

    N = 200
    P = np.cumsum(rng.normal(scale=2.0, size=(N, 3)), axis=0)
    Q = P.copy()
    Q[:120] += rng.normal(scale=6.0, size=(120, 3))  # only the C-term matches
    cases["subdomain_only"] = (P, Q)

    N = 150
    P = np.cumsum(rng.normal(scale=2.0, size=(N, 3)), axis=0)
    cases["near_identical"] = (P, P + rng.normal(scale=0.4, size=(N, 3)))
    return cases


@pytest.mark.parametrize("name", list(_tm_cases()))
def test_tm_score_matches_tmalign_for_a_correct_correspondence(name):
    """Our TM-score maximizes over superpositions for a *fixed* correspondence.

    Where the given correspondence is the right one, that must land on
    TM-align's value. The agreement is within 0.01 on these decoys; the
    seven-seed search this replaced drifted further, and a plain Kabsch frame
    underestimates TM whenever a flexible tail drags the least-squares fit.
    """
    tmtools = _tmtools()
    from pdb_align.metrics import tm_optimal_superposition

    P, Q = _tm_cases()[name]
    seq = "A" * len(P)
    ref = tmtools.tm_align(P, Q, seq, seq).tm_norm_chain1
    ours, _R, _t = tm_optimal_superposition(P, Q, len(P))
    assert ours == pytest.approx(ref, abs=0.011)


def test_tm_score_of_identical_structures_is_one():
    tmtools = _tmtools()
    from pdb_align.metrics import tm_optimal_superposition

    rng = np.random.default_rng(5)
    P = np.cumsum(rng.normal(scale=2.0, size=(100, 3)), axis=0)
    seq = "A" * len(P)
    ours, _R, _t = tm_optimal_superposition(P, P.copy(), len(P))
    assert ours == pytest.approx(1.0, abs=1e-6)
    assert tmtools.tm_align(P, P.copy(), seq, seq).tm_norm_chain1 == \
        pytest.approx(1.0, abs=1e-4)


def test_raising_the_seed_cap_cannot_lower_the_tm_score():
    """The search only ever keeps its best seed, so a wider search is monotone.

    This is what licenses the default seed cap: it trades a little of the
    reference program's exhaustive O(L^2) search for speed (5.5 s -> 0.2 s at
    L = 3000) without being able to overshoot.
    """
    from pdb_align.metrics import tm_optimal_superposition

    P, Q = _tm_cases()["subdomain_only"]
    narrow, _, _ = tm_optimal_superposition(P, Q, len(P), max_starts=8)
    wide, _, _ = tm_optimal_superposition(P, Q, len(P), max_starts=256)
    assert wide >= narrow - 1e-12
