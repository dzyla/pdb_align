import os

from webapp import data as D

DATA = os.path.join(os.path.dirname(__file__), "data")
REF = os.path.join(DATA, "ref.pdb")
MOB = os.path.join(DATA, "mob.pdb")
OPTS = {"seq_gap_open": -10, "seq_gap_extend": -0.5, "atoms": "CA", "min_plddt": 0.0}


def test_save_upload_to_temp_roundtrip():
    with open(REF, "rb") as f:
        raw = f.read()
    p = D.save_upload_to_temp("x.pdb", raw)
    assert p.endswith(".pdb") and os.path.exists(p)
    with open(p, "rb") as f:
        assert f.read() == raw


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
    ens = D.run_ensemble(REF, [MOB, REF], None, "auto", "auto", OPTS)
    assert isinstance(ens, EnsembleResult)
    assert len(ens.results) == 2


def test_input_key_stability():
    k1 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k2 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k3 = D.input_key(REF, [MOB], None, {}, "flexible", "auto", OPTS)
    assert k1 == k2 and k1 != k3


def test_run_ensemble_forwards_strategy_and_workers():
    """Both were accepted and dropped, so the sidebar's Strategy selector had
    no effect on an ensemble run."""
    import inspect

    sig = inspect.signature(D.run_ensemble)
    assert "strategy" in sig.parameters
    assert "workers" in sig.parameters
    src = inspect.getsource(D.run_ensemble)
    assert "strategy=strategy" in src
    assert "workers=workers" in src


def test_run_pairwise_applies_the_b_factor_floor(tmp_path):
    """The library has filtered on B-factors since 0.4.0 and the pLDDT warning
    points users at it, but the app forwarded min_plddt only."""
    from _synthetic import make_chain, make_structure

    def write(path, bfactors):
        chain = make_chain("A", len(bfactors), (0.0, 0.0, 0.0))
        for res, b in zip(chain, bfactors):
            for atom in res:
                atom.b_iso = float(b)
        st = make_structure([chain])
        st.setup_entities()
        st.write_pdb(str(path))

    bfs = [90.0] * 12 + [10.0] * 8
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    write(ref, bfs); write(mob, bfs)

    opts = dict(OPTS, min_b_factor=50.0)
    res = D.run_pairwise(str(ref), str(mob), None, None, "auto", "auto", opts)

    assert res.summary_stats()["n_aligned"] == 12
