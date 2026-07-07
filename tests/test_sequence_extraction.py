"""Regression tests for sequence/residue-list consistency.

The alignment sequence built by ``extract_sequences_and_lengths`` must contain
exactly the residues that ``get_aligned_atoms_by_alignment`` will iterate over
(the ones carrying a CA atom). If a standard amino acid without a CA leaks into
the sequence, the alignment-to-residue walk desynchronises and every downstream
pair is shifted, silently corrupting the superposition.
"""
from pdb_align.core import extract_sequences_and_lengths, _parse_path


def test_sequence_excludes_residues_without_ca(tmp_path):
    # Residue 2 is a standard AA but has only an N atom (no CA).
    pdb = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N   ALA A   2       3.800   0.000   0.000  1.00  0.00           N
ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00  0.00           C
END
"""
    p = tmp_path / "gap.pdb"
    p.write_text(pdb)
    struct = _parse_path(str(p))
    seqs, lens = extract_sequences_and_lengths(struct, "gap.pdb")

    # Only two residues carry a CA, so both the sequence and the length must be 2.
    assert lens["A"] == 2
    assert len(str(seqs["A"].seq)) == 2
    assert str(seqs["A"].seq) == "AA"
