import numpy as np
from types import SimpleNamespace
from pdb_align.chains import match_chains, ChainMapping


def _seqrec(seq):
    return SimpleNamespace(seq=seq)


def test_heteromer_pairs_by_best_identity(monkeypatch):
    import pdb_align.chains as ch
    import pandas as pd
    # ref A~mob Y (identical), ref B~mob X (identical); file order is crossed.
    ref_seqs = {"A": _seqrec("AAAAKKKK"), "B": _seqrec("DDDDEEEE")}
    mob_seqs = {"X": _seqrec("DDDDEEEE"), "Y": _seqrec("AAAAKKKK")}
    id_mat = pd.DataFrame([[0.0, 100.0], [100.0, 0.0]],
                          index=["A", "B"], columns=["X", "Y"])
    monkeypatch.setattr(ch, "compute_chain_similarity_matrix",
                        lambda a, b: (id_mat, id_mat))
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A", "B"], ["X", "Y"])
    pairs = {(p[0], p[1]) for p in mapping.pairs}
    assert pairs == {("A", "Y"), ("B", "X")}


def test_unmatched_chains_reported(monkeypatch):
    import pdb_align.chains as ch
    import pandas as pd
    ref_seqs = {"A": _seqrec("AAAAKKKK")}
    mob_seqs = {"X": _seqrec("AAAAKKKK"), "Z": _seqrec("WWWWWWWW")}
    id_mat = pd.DataFrame([[100.0, 0.0]], index=["A"], columns=["X", "Z"])
    monkeypatch.setattr(ch, "compute_chain_similarity_matrix",
                        lambda a, b: (id_mat, id_mat))
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A"], ["X", "Z"])
    assert mapping.pairs[0][0] == "A" and mapping.pairs[0][1] == "X"
    assert "Z" in mapping.unmatched_mob


def test_homodimer_swapped_chains_refined_by_geometry():
    """Two identical chains; correct mapping must come from geometry, not sequence."""
    import gemmi
    from pdb_align.core import extract_sequences_and_lengths

    def _struct(coords_by_chain):
        st = gemmi.Structure(); model = gemmi.Model("1");
        for cname, coords in coords_by_chain.items():
            chain = gemmi.Chain(cname)
            for k, (x, y, z) in enumerate(coords, start=1):
                res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k, " ")
                at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
                res.add_atom(at); chain.add_residue(res)
            model.add_chain(chain)
        st.add_model(model); return st
    base = [(i, 0.0, 0.0) for i in range(15)]
    ref = _struct({"A": base, "B": [(x, 10.0, 0.0) for x, _, _ in base]})
    # mobile chains carry swapped names relative to geometry
    mob = _struct({"P": [(x, 10.0, 0.0) for x, _, _ in base], "Q": base})
    rs, _ = extract_sequences_and_lengths(ref, "ref")
    ms, _ = extract_sequences_and_lengths(mob, "mob")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["P", "Q"])
    pairs = {(p[0], p[1]) for p in mapping.pairs}
    # A (y=0) should map to Q (y=0); B (y=10) to P (y=10)
    assert pairs == {("A", "Q"), ("B", "P")}


def test_homodimer_swapped_chains_keep_true_identity():
    """Chains reassigned by geometric refinement must carry their real
    sequence identity, not a 0.0 fallback from the stale Hungarian pairing."""
    import gemmi
    from pdb_align.core import extract_sequences_and_lengths

    def _struct(coords_by_chain):
        st = gemmi.Structure(); model = gemmi.Model("1");
        for cname, coords in coords_by_chain.items():
            chain = gemmi.Chain(cname)
            for k, (x, y, z) in enumerate(coords, start=1):
                res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k, " ")
                at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
                res.add_atom(at); chain.add_residue(res)
            model.add_chain(chain)
        st.add_model(model); return st
    base = [(i, 0.0, 0.0) for i in range(15)]
    ref = _struct({"A": base, "B": [(x, 10.0, 0.0) for x, _, _ in base]})
    # mobile chains carry swapped names relative to geometry
    mob = _struct({"P": [(x, 10.0, 0.0) for x, _, _ in base], "Q": base})
    rs, _ = extract_sequences_and_lengths(ref, "ref")
    ms, _ = extract_sequences_and_lengths(mob, "mob")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["P", "Q"])
    assert all(p[2] > 0 for p in mapping.pairs)
    # Identical (all-ALA) sequences -> ~100% identity, not the stale 0.0 fallback.
    assert all(p[2] > 50 for p in mapping.pairs)
