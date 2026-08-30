"""Cross-validation against reference implementations (skipped when the
reference packages are not installed).

- DockQ components vs the official Wallner-lab implementation (pip DockQ).
- TM-score vs the real TM-align code (pip tmtools).

These are the tests that make the numbers citable: agreement is asserted to
tight tolerances on non-trivial decoys, not just on identity cases.
"""
import numpy as np
import pytest

from _synthetic import two_chain_complex

dockq_pkg = pytest.importorskip("DockQ.DockQ", reason="official DockQ not installed")
from pdb_align.interface import compute_dockq


def _official_dockq(model_path, native_path, chain_map):
    m = dockq_pkg.load_PDB(str(model_path))
    n = dockq_pkg.load_PDB(str(native_path))
    res, _total = dockq_pkg.run_on_all_native_interfaces(m, n, chain_map=chain_map)
    assert len(res) == 1
    return next(iter(res.values()))


@pytest.mark.parametrize("lig_origin", [
    (3.8, 5.5, 0.0),     # slid one residue + lifted: partial fnat
    (0.0, 7.0, 0.0),     # pushed off: low fnat, moderate LRMSD
    (7.6, 4.0, 0.0),     # slid two residues: register-shifted contacts
])
def test_dockq_components_match_official_implementation(tmp_path, lig_origin):
    native = two_chain_complex(lig_origin=(0.0, 4.0, 0.0))
    model = two_chain_complex(lig_origin=lig_origin)
    p_native = tmp_path / "native.pdb"
    p_model = tmp_path / "model.pdb"
    native.write_pdb(str(p_native))
    model.write_pdb(str(p_model))

    ours = compute_dockq(str(p_native), str(p_model), ["A"], ["B"])
    ref = _official_dockq(p_model, p_native, {"A": "A", "B": "B"})

    assert ours.fnat == pytest.approx(ref["fnat"], abs=1e-3)
    assert ours.irmsd == pytest.approx(ref["iRMSD"], abs=1e-3)
    assert ours.lrmsd == pytest.approx(ref["LRMSD"], abs=1e-3)
    assert ours.dockq == pytest.approx(ref["DockQ"], abs=1e-3)


def test_tm_score_close_to_tmalign():
    tmtools = pytest.importorskip("tmtools", reason="tmtools not installed")
    from pdb_align.metrics import tm_optimal_superposition

    rng = np.random.default_rng(3)
    N = 120
    P = np.cumsum(rng.normal(scale=2.2, size=(N, 3)), axis=0)
    Q = P + rng.normal(scale=0.7, size=(N, 3))
    Q[90:] += np.array([18.0, 5.0, -7.0])  # swung-out tail (hinge decoy)
    seq = "A" * N

    ref = tmtools.tm_align(P, Q, seq, seq).tm_norm_chain1
    ours, _, _ = tm_optimal_superposition(P, Q, N)
    # TM-align also optimizes the residue alignment (a superset search), and
    # both searches are heuristic; for a correct fixed correspondence our
    # TM-optimal superposition must land within a few percent, never below
    # TM-align by much (scoring the full correspondence can only add terms).
    assert ours == pytest.approx(ref, abs=0.03)

    ident = tmtools.tm_align(P, P.copy(), seq, seq).tm_norm_chain1
    ours_ident, _, _ = tm_optimal_superposition(P, P.copy(), N)
    assert ours_ident == pytest.approx(1.0, abs=1e-6)
    assert ident == pytest.approx(1.0, abs=1e-4)
