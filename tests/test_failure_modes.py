"""How the package behaves when it cannot do what was asked.

A scientific tool must fail loudly. The alternative — returning a result
object whose every number is None, or quietly filtering a reference structure
into nothing — is worse than an exception.
"""
import gemmi
import pytest

import pdb_align
from pdb_align.aligner import AlignmentFailedError, PDBAligner

UBQ = "tests/data/1ubq.pdb"
CRN = "tests/data/1crn.pdb"


def test_a_selection_that_filters_everything_away_raises():
    """A cutoff that removes every residue is an error, not an empty result."""
    al = PDBAligner()
    al.add_reference(UBQ)
    al.add_mobile(UBQ)
    with pytest.raises(ValueError, match="cutoff"):
        al.align(mode="seq_guided", min_b_factor=1e6)


def test_explicit_mode_that_cannot_superpose_raises():
    """A selection too small to superpose must raise, not return a result whose
    RMSD is inf and whose every derived metric is meaningless."""
    with pytest.raises(AlignmentFailedError, match="could not superpose"):
        pdb_align.align(UBQ, UBQ, chains_ref=["A:1-2"], chains_mob=["A:1-2"],
                        mode="seq_free_shape")


def test_alignment_failed_error_is_a_value_error():
    """One `except ValueError` catches every unanswerable request."""
    assert issubclass(AlignmentFailedError, ValueError)


def test_result_never_carries_none_rmsd():
    """Any AlignmentResult handed to a caller has a usable RMSD."""
    res = pdb_align.align(UBQ, CRN)
    assert res.rmsd is not None


def test_min_plddt_on_an_experimental_reference_warns_and_does_not_filter_it():
    """pLDDT is a *confidence* (high = good) while a crystallographic B-factor
    is a *disorder* measure (low = good). Applying a pLDDT cutoff to an
    experimental reference used to discard every residue and abort the run."""
    with pytest.warns(UserWarning, match="pLDDT"):
        res = pdb_align.align(UBQ, UBQ, min_plddt=70.0)
    assert res.rmsd == pytest.approx(0.0, abs=1e-6)
    assert len(res.get_rmsd_df()) == 76


def test_min_plddt_filters_a_predicted_model(tmp_path):
    """On a model whose B-factor column really is pLDDT, the cutoff applies."""
    st = gemmi.read_structure(UBQ)
    st.setup_entities()
    for i, res in enumerate(st[0]["A"]):
        for atom in res:
            atom.b_iso = 30.0 if i < 20 else 95.0
    model = tmp_path / "model.pdb"
    st.write_pdb(str(model))

    res = pdb_align.align(UBQ, str(model), min_plddt=70.0)
    assert len(res.get_rmsd_df()) == 56


def test_min_b_factor_filters_both_sides_symmetrically():
    """min_b_factor is the explicit, symmetric knob: it applies to reference and
    mobile alike (unlike min_plddt), and the residues it keeps still pair
    exactly, so an identical pair superposes at 0 A over the filtered subset."""
    al = PDBAligner()
    al.add_reference(UBQ)
    al.add_mobile(UBQ)
    unfiltered = len(al.align().get_rmsd_df())
    res = al.align(min_b_factor=5.0)
    n = len(res.get_rmsd_df())
    assert 0 < n < unfiltered
    assert res.rmsd == pytest.approx(0.0, abs=1e-6)
    assert res.summary_stats()["coverage_pct"] == pytest.approx(100.0, abs=0.01)


def test_unrelated_structures_warn_about_weak_chain_correspondence():
    """A chain mapping at random-sequence identity must be flagged."""
    res = pdb_align.align("tests/data/4hhb_bb.pdb", UBQ)
    q = res.quality
    assert q.band == "poor"
