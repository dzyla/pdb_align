"""End-to-end checks against structural facts, not against our own output.

Unit tests prove the code does what it was written to do; these prove the
answers are biologically right. Haemoglobin is the fixture because one file
contains three known relationships: two identical copies of the α chain, two
identical copies of β, and the α/β pair — homologous globins at ~43% sequence
identity that share the globin fold and are documented to superpose at
~1.5–2 Å. Ubiquitin and crambin supply the negative control.

If a refactor breaks the science while keeping the unit tests green, it breaks
here.
"""
import warnings

import pytest

import pdb_align

HHB = "tests/data/4hhb_bb.pdb"
UBQ = "tests/data/1ubq.pdb"
CRN = "tests/data/1crn.pdb"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def test_identical_chain_copies_superpose_almost_exactly():
    """Chains A and C of 4HHB are two copies of the same α-globin."""
    res = pdb_align.align(HHB, HHB, chains_ref=["A"], chains_mob=["C"])
    assert res.rmsd < 0.5
    assert res.tm_score > 0.99
    assert res.quality.band == "excellent"


def test_homologous_globins_are_recognised_as_the_same_fold():
    """α (chain A) vs β (chain B): ~43% identity, same fold, ~1.5-2 A."""
    res = pdb_align.align(HHB, HHB, chains_ref=["A"], chains_mob=["B"])
    stats = res.summary_stats()
    assert 1.0 < stats["rmsd"] < 2.5
    assert stats["tm_score"] > 0.7
    assert stats["coverage_pct"] > 90.0
    assert stats["lddt_ca"] > 0.7
    assert stats["tm_pvalue"] < 1e-6      # unambiguously significant
    assert res.quality.band in ("good", "excellent")


def test_homologous_globin_tm_score_matches_tmalign():
    """The headline number on a real homologous pair, against TM-align.

    TM-align also optimises the residue correspondence, so it can only score
    at least as high; agreement to a few thousandths means our fixed
    sequence-based correspondence is the right one here.
    """
    import gemmi
    tmtools = pytest.importorskip("tmtools", reason="tmtools not installed")
    from pdb_align.core import select_residues

    ours = pdb_align.align(HHB, HHB, chains_ref=["A"], chains_mob=["B"]).tm_score
    a = select_residues(gemmi.read_structure(HHB), ["A"], with_atoms=False)
    b = select_residues(gemmi.read_structure(HHB), ["B"], with_atoms=False)
    ref = tmtools.tm_align(a.ca_coords, b.ca_coords, a.sequence, b.sequence)
    assert ours == pytest.approx(ref.tm_norm_chain1, abs=0.02)
    assert ours <= ref.tm_norm_chain1 + 1e-9


def test_unrelated_folds_are_not_called_similar():
    """Ubiquitin (β-grasp) vs crambin: different folds, and the p-value must
    say the TM-score is not significant."""
    res = pdb_align.align(UBQ, CRN)
    stats = res.summary_stats()
    assert stats["tm_score"] < 0.3
    assert stats["tm_pvalue"] > 0.01
    assert res.quality.band == "poor"


def test_the_whole_tetramer_matches_itself_chain_for_chain():
    """4HHB has two α and two β chains; the correspondence must pair like with
    like, which sequence alone cannot do for the identical copies."""
    res = pdb_align.align(HHB, HHB)
    pairs = {(a, b) for a, b, *_ in res.chain_mapping.pairs}
    assert pairs == {("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")}
    assert res.rmsd == pytest.approx(0.0, abs=1e-6)
    assert all(v == pytest.approx(1.0, abs=1e-6)
               for v in res.summary_stats()["tm_score_per_chain"].values())


def test_swapped_identical_chains_are_matched_by_geometry(tmp_path):
    """Rename 4HHB's A→C and C→A. Sequence identity is tied between the two
    α copies, so only geometry can recover the correct correspondence; getting
    it wrong inflates RMSD with no other symptom."""
    import gemmi

    st = gemmi.read_structure(HHB)
    st.setup_entities()
    rename = {"A": "C", "C": "A"}
    for chain in st[0]:
        chain.name = rename.get(chain.name, chain.name) + "_tmp"
    for chain in st[0]:
        chain.name = chain.name[:-4]
    swapped = tmp_path / "swapped.pdb"
    st.write_pdb(str(swapped))

    res = pdb_align.align(HHB, str(swapped))
    pairs = {(a, b) for a, b, *_ in res.chain_mapping.pairs}
    assert pairs == {("A", "C"), ("C", "A"), ("B", "B"), ("D", "D")}
    assert res.rmsd == pytest.approx(0.0, abs=1e-6)
