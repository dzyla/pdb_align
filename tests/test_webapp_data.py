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
    ens = D.run_ensemble(REF, [MOB, REF], None, {}, "auto", "auto", OPTS)
    assert isinstance(ens, EnsembleResult)
    assert len(ens.results) == 2


def test_input_key_stability():
    k1 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k2 = D.input_key(REF, [MOB], None, {}, "auto", "auto", OPTS)
    k3 = D.input_key(REF, [MOB], None, {}, "flexible", "auto", OPTS)
    assert k1 == k2 and k1 != k3
