"""What the result object claims about itself must be defensible.

These tests pin the reporting decisions that a reader (or a reviewer) would
otherwise have to take on trust: what a TM-score was normalized by, whether a
chain correspondence is supported by sequence, and what GDT_TS counts as a
failure.
"""
import gemmi
import numpy as np
import pytest

import pdb_align

UBQ = "tests/data/1ubq.pdb"
HHB = "tests/data/4hhb_bb.pdb"


def test_complex_tm_score_is_labelled_and_accompanied_by_per_chain_values():
    """TM-score is defined per chain. A complex-level value must say so and
    carry the per-chain numbers, because one well-placed large chain otherwise
    hides a badly placed small one."""
    res = pdb_align.align(HHB, HHB)
    s = res.summary_stats()
    assert s["tm_scope"] == "complex"
    assert s["tm_normalization_length"] == 574
    per_chain = s["tm_score_per_chain"]
    assert set(per_chain) == {"A", "B", "C", "D"}
    assert all(v == pytest.approx(1.0, abs=1e-6) for v in per_chain.values())
    assert any("normalized by the whole reference selection" in w
               for w in res.quality.warnings)


def test_single_chain_tm_score_is_not_labelled_as_a_complex():
    res = pdb_align.align(UBQ, UBQ, chains_ref=["A"], chains_mob=["A"])
    s = res.summary_stats()
    assert s["tm_scope"] == "chain"
    assert "tm_score_per_chain" not in s


def test_per_chain_tm_matches_aligning_that_chain_alone():
    """The per-chain value must be a real per-chain TM-score, not a slice of
    the complex-level one."""
    whole = pdb_align.align(HHB, HHB).summary_stats()["tm_score_per_chain"]
    alone = pdb_align.align(HHB, HHB, chains_ref=["B"], chains_mob=["B"]).tm_score
    assert whole["B"] == pytest.approx(alone, abs=1e-6)


def test_weak_mapping_warning_reaches_the_quality_verdict(tmp_path):
    """Two unrelated two-chain complexes: the mapping is forced, and the user
    must be told, in the report, that it is not supported by sequence."""
    st = gemmi.read_structure(UBQ)
    st.setup_entities()
    # Build a 2-chain structure out of ubiquitin so both sides are multi-chain.
    model = st[0]
    chain_b = gemmi.Chain("B")
    for res in model["A"]:
        copy = gemmi.Residue()
        copy.name = res.name
        copy.seqid = res.seqid
        for atom in res:
            moved = gemmi.Atom()
            moved.name = atom.name
            moved.element = atom.element
            moved.b_iso = atom.b_iso
            moved.occ = atom.occ
            moved.pos = gemmi.Position(atom.pos.x + 40.0, atom.pos.y, atom.pos.z)
            copy.add_atom(moved)
        chain_b.add_residue(copy)
    model.add_chain(chain_b)
    two_chain_ubq = tmp_path / "ubq2.pdb"
    st.setup_entities()
    st.write_pdb(str(two_chain_ubq))

    with pytest.warns(UserWarning, match="weak sequence identity"):
        res = pdb_align.align(HHB, str(two_chain_ubq),
                              chains_ref=["A", "B"], chains_mob=["A", "B"])
    assert res.quality.confidence == "low"
    assert any("weak sequence identity" in w for w in res.quality.warnings)
    assert "weak sequence identity" in res.report()


def test_gdt_ts_counts_unaligned_reference_residues_as_failures():
    """CASP semantics: GDT_TS is normalized by the reference length, so a model
    covering half the reference cannot score above ~50 even if that half is
    perfect."""
    res = pdb_align.align(UBQ, UBQ, chains_ref=["A"], chains_mob=["A:1-38"],
                          mode="seq_guided")
    gdt = res.summary_stats()["gdt_ts"]
    assert 45.0 <= gdt <= 52.0


def test_gdt_normalization_follows_the_filtered_selection():
    """When a filter removes reference residues they are no longer part of the
    target, so they must not be counted as GDT failures."""
    st = gemmi.read_structure(UBQ)
    st.setup_entities()
    for i, res in enumerate(st[0]["A"]):
        for atom in res:
            atom.b_iso = 95.0 if i < 40 else 20.0
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    st.write_pdb(path)
    try:
        res = pdb_align.align(path, path, mode="seq_guided", min_b_factor=50.0)
        assert res.summary_stats()["gdt_ts"] == pytest.approx(100.0)
        assert res.summary_stats()["coverage_pct"] == pytest.approx(100.0)
    finally:
        os.unlink(path)


def test_contact_overlap_is_opt_in_and_superposition_free():
    """It is an O(N^2) comparison (peaking near 900 MB at 6000 residues) that
    no report shows, so it must not run as part of every alignment — but it
    must still be available."""
    res = pdb_align.align(UBQ, UBQ, chains_ref=["A"], chains_mob=["A"])
    assert "contact_overlap" not in res.summary_stats()
    assert res.contact_overlap() == pytest.approx(1.0)
