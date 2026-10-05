"""Residue-selection and residue-pairing integrity.

These are the invariants that make every downstream number trustworthy:

1. The sequence handed to the aligner and the residue list used to build
   coordinates must describe *the same residues, in the same order*. Any
   divergence silently mis-pairs residues and produces a plausible but wrong
   RMSD (observed: 5.6 A between a structure and an identical copy of itself).
2. A selector with a residue range ("A:10-150") must be honoured, not rejected.
3. Residue-level reporting (n_aligned, coverage) must count residues even when
   the superposition uses several atoms per residue.
4. Side-chain atoms must never be paired across residues of different type.
"""
import gemmi
import numpy as np
import pytest

import pdb_align
from pdb_align.core import (
    residue_letter, select_residues, pairs_from_alignment, paired_atoms,
    perform_sequence_alignment,
)

REF = "tests/data/1ubq.pdb"


def _with_bfactors(path, low_indices, low=10.0, high=90.0):
    """Copy of *path* whose residues at *low_indices* carry a low B-factor."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    for chain in st[0]:
        for i, res in enumerate(chain):
            for atom in res:
                atom.b_iso = low if i in low_indices else high
    return st


def test_bfactor_filter_keeps_sequence_and_residues_in_sync():
    """A filtered residue must disappear from the sequence too.

    Otherwise the alignment is computed over residues that are absent from the
    coordinate list, and the pairing silently drifts.
    """
    st = _with_bfactors(REF, {20, 40})
    sel = select_residues(st, ["A"], min_b_factor=50.0)
    assert len(sel.sequence) == len(sel.residues)
    assert len(sel.residues) == 74  # 76 residues minus the two filtered out


def test_filtered_mobile_still_pairs_identical_structure_exactly():
    """Filtering residues out of one side must not mis-pair the rest.

    Identical coordinates, two residues dropped from the mobile side (chosen so
    the gap placement is unambiguous — see the degenerate-run test below): every
    surviving pair must join the *same* residue number, and the RMSD must be 0.
    Before the single-source-of-truth selection this produced 24 pairs, four of
    them mis-paired, and an RMSD of 5.62 A between a structure and itself.
    """
    ref_st = _with_bfactors(REF, set())
    mob_st = _with_bfactors(REF, {2, 16})  # residues 3 (I in QIF) and 17 (V in EVE)

    ref_sel = select_residues(ref_st, ["A"], min_b_factor=50.0)
    mob_sel = select_residues(mob_st, ["A"], min_b_factor=50.0)
    aln = perform_sequence_alignment(ref_sel.sequence, mob_sel.sequence, -10.0, -0.5)
    pairs = pairs_from_alignment(aln)
    ref_atoms, mob_atoms = paired_atoms(ref_sel, mob_sel, pairs, atoms="CA")

    assert len(ref_atoms) == 74
    mispaired = [(a.res_seq, b.res_seq) for a, b in zip(ref_atoms, mob_atoms)
                 if a.res_seq != b.res_seq]
    assert mispaired == []

    P = np.array([a.get_coord() for a in ref_atoms])
    Q = np.array([b.get_coord() for b in mob_atoms])
    assert float(np.sqrt(((P - Q) ** 2).sum(1).mean())) == pytest.approx(0.0, abs=1e-9)


def test_deletion_inside_a_repeated_residue_run_is_a_documented_tie():
    """Sequence alignment cannot localise a deletion inside a homopolymer run.

    Ubiquitin has Q40-Q41; deleting either leaves the same sequence, so the gap
    may be placed at 40 or 41 with identical score. The pairing stays
    positional and complete — exactly one residue is offset, and only inside
    the run — which is a property of sequence-guided alignment, not drift. A
    structure-based correspondence (mode='seq_free_*') has no such ambiguity.
    """
    ref_sel = select_residues(_with_bfactors(REF, set()), ["A"], min_b_factor=50.0)
    mob_sel = select_residues(_with_bfactors(REF, {40}), ["A"], min_b_factor=50.0)
    aln = perform_sequence_alignment(ref_sel.sequence, mob_sel.sequence, -10.0, -0.5)
    ref_atoms, mob_atoms = paired_atoms(ref_sel, mob_sel,
                                        pairs_from_alignment(aln), atoms="CA")
    assert len(ref_atoms) == 75
    offset = [(a.res_seq, b.res_seq) for a, b in zip(ref_atoms, mob_atoms)
              if a.res_seq != b.res_seq]
    assert len(offset) <= 1
    for ref_num, mob_num in offset:
        assert abs(ref_num - mob_num) == 1
        assert 39 <= ref_num <= 42


def test_residue_range_selector_is_accepted_and_applied():
    """'A:2-10' must select exactly residues 2..10, not raise."""
    sel = select_residues(gemmi.read_structure(REF), ["A:2-10"])
    assert [r.seqid for r in sel.residues] == list(range(2, 11))


def test_align_accepts_residue_range_selector():
    """The documented 'A:start-end' syntax must work through the public API."""
    res = pdb_align.align(REF, REF, chains_ref=["A:2-40"], chains_mob=["A:2-40"])
    assert res.rmsd == pytest.approx(0.0, abs=1e-6)
    assert len(res.get_rmsd_df()) == 39


def test_align_rejects_range_outside_the_chain():
    with pytest.raises(ValueError, match="no residues"):
        pdb_align.align(REF, REF, chains_ref=["A:900-999"], chains_mob=["A"])


def test_n_aligned_counts_residues_not_atoms():
    """With atoms='all_heavy' the superposition uses ~8 atoms/residue, but
    n_aligned and coverage are residue-level quantities (coverage was 792%)."""
    res = pdb_align.align(REF, REF, chains_ref=["A"], chains_mob=["A"],
                          mode="seq_guided", atoms="all_heavy")
    stats = res.summary_stats()
    assert stats["n_aligned"] == 76
    assert stats["coverage_pct"] == pytest.approx(100.0, abs=0.01)
    assert len(res.get_rmsd_df()) == 76


def test_per_residue_table_has_one_row_per_residue_for_multi_atom_modes():
    df = pdb_align.align(REF, REF, chains_ref=["A"], chains_mob=["A"],
                         mode="seq_guided", atoms="backbone").get_rmsd_df()
    assert len(df) == 76
    assert df["Residue"].is_unique


def test_sidechain_atoms_are_not_paired_across_unlike_residues():
    """An ALA CB and an ARG CB point into different chemistry; pairing them by
    name is meaningless, so unlike residue pairs contribute backbone only."""
    st_a = gemmi.read_structure(REF)
    st_b = gemmi.read_structure(REF)
    # Mutate one mobile residue's identity (keep its atoms) so the pair is unlike.
    target = st_b[0]["A"][10]
    assert residue_letter(target.name) != "G"
    original = target.name
    target.name = "TRP" if original != "TRP" else "PHE"

    sel_a = select_residues(st_a, ["A"])
    sel_b = select_residues(st_b, ["A"])
    aln = perform_sequence_alignment(sel_a.sequence, sel_b.sequence, -10.0, -0.5)
    pairs = pairs_from_alignment(aln)
    ref_atoms, mob_atoms = paired_atoms(sel_a, sel_b, pairs, atoms="all_heavy")

    unlike = [(a, b) for a, b in zip(ref_atoms, mob_atoms)
              if a.res_seq == target.seqid.num]
    assert unlike, "the mutated residue should still be paired"
    assert {a.name for a, _ in unlike} <= {"N", "CA", "C", "O"}
