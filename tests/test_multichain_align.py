import numpy as np
import gemmi
from pdb_align.core import extract_sequences_and_lengths
from pdb_align.chains import match_chains, align_multichain


def _struct(coords_by_chain):
    st = gemmi.Structure(); model = gemmi.Model("1")
    for cname, coords in coords_by_chain.items():
        chain = gemmi.Chain(cname)
        for k, (x, y, z) in enumerate(coords, start=1):
            res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k, " ")
            at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
            res.add_atom(at); chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model); return st


def _two_chain(offset_b=0.0):
    a = [(i, 0.0, 0.0) for i in range(15)]
    b = [(i, 10.0 + offset_b, 0.0) for i in range(15)]
    return {"A": a, "B": b}


def test_global_strategy_superimposes_all_chains():
    ref = _struct(_two_chain(0.0))
    # mobile = ref rigidly translated by +5 in x
    mob = _struct({k: [(x + 5, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="global")
    assert res.strategy == "global"
    assert res.rmsd < 1e-6  # rigid translation recovered exactly
    assert len(res.per_chain) == 2


def test_local_beats_global_when_one_chain_diverges():
    ref = _struct(_two_chain(0.0))
    # chain B of mobile is badly displaced; chain A matches after rigid move
    mob = _struct({"A": [(x + 5, 0.0, 0.0) for x in range(15)],
                   "B": [(x + 5, 40.0, 30.0) for x in range(15)]})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="auto")
    assert res.strategy == "local"
    assert res.rmsd < 1e-6


def test_pdbaligner_auto_uses_multichain(tmp_path):
    import gemmi, pdb_align
    ref = _struct(_two_chain(0.0))
    mob = _struct({k: [(x + 3, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rp, mp = tmp_path / "ref.pdb", tmp_path / "mob.pdb"
    ref.write_pdb(str(rp)); mob.write_pdb(str(mp))
    r = pdb_align.align(str(rp), str(mp))
    assert r.strategy in ("global", "local")
    assert not r.per_chain.empty
    assert r.rmsd < 1e-5
