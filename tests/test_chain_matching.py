from types import SimpleNamespace

import numpy as np

from pdb_align.chains import ChainMapping, match_chains


def _seqrec(seq):
    return SimpleNamespace(seq=seq)


# Two unrelated 40-mers; real BLOSUM62 identities, no mocking of the matrix.
_SEQ1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
_SEQ2 = "GSHMLEDPRVWQDFLSRAKEIVAGNCYTWPDGVKLHFNEAMSRYLTPDQQ"


def test_heteromer_pairs_by_sequence_identity_not_file_order():
    """File order is crossed, so only sequence can give the correspondence."""
    ref_seqs = {"A": _seqrec(_SEQ1), "B": _seqrec(_SEQ2)}
    mob_seqs = {"X": _seqrec(_SEQ2), "Y": _seqrec(_SEQ1)}
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A", "B"], ["X", "Y"])
    assert {(p[0], p[1]) for p in mapping.pairs} == {("A", "Y"), ("B", "X")}


def test_unmatched_chains_reported():
    ref_seqs = {"A": _seqrec(_SEQ1)}
    mob_seqs = {"X": _seqrec(_SEQ1), "Z": _seqrec(_SEQ2)}
    mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A"], ["X", "Z"])
    assert (mapping.pairs[0][0], mapping.pairs[0][1]) == ("A", "X")
    assert "Z" in mapping.unmatched_mob


def test_identity_is_normalised_by_the_shorter_chain():
    """A domain that matches its parent chain perfectly is 100% identical to it.

    Normalising by alignment length (including terminal gaps) turned a perfect
    50-residue match inside a 200-residue chain into "25% identity" — a length
    ratio wearing an identity's name, which broke chain matching for truncated
    constructs, Fv fragments and single-domain models.
    """
    from pdb_align.core import compute_chain_similarity_matrix
    full = _SEQ1 + _SEQ2 + _SEQ1 + _SEQ2
    domain = _SEQ2
    id_mat, _ = compute_chain_similarity_matrix({"A": _seqrec(full)},
                                                {"d": _seqrec(domain)})
    assert float(id_mat.iloc[0, 0]) > 95.0


def test_weak_correspondence_is_warned_about():
    """A mapping at chance-level identity must not be presented as a fact."""
    import pytest
    ref_seqs = {"A": _seqrec(_SEQ1)}
    mob_seqs = {"X": _seqrec(_SEQ2)}
    with pytest.warns(UserWarning, match="weak sequence identity"):
        mapping = match_chains(ref_seqs, mob_seqs, None, None, ["A"], ["X"])
    assert mapping.warnings


def test_homodimer_swapped_chains_refined_by_geometry():
    """Two identical chains; correct mapping must come from geometry, not sequence."""
    import gemmi

    from pdb_align.core import extract_sequences_and_lengths

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
        st = gemmi.Structure(); model = gemmi.Model("1")
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
