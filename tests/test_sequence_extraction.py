"""Regression tests for sequence/residue-list consistency.

The alignment sequence built by ``extract_sequences_and_lengths`` must contain
exactly the residues that ``get_aligned_atoms_by_alignment`` will iterate over
(the ones carrying a CA atom). If a standard amino acid without a CA leaks into
the sequence, the alignment-to-residue walk desynchronises and every downstream
pair is shifted, silently corrupting the superposition.
"""
from pdb_align.core import _parse_path, extract_sequences_and_lengths


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


def test_semiglobal_alignment_does_not_penalise_terminal_overhang():
    """A domain must align inside its parent chain without paying for the
    overhang, on every supported Biopython.

    Biopython 1.86 renamed the end-gap attributes and deprecated the old
    names. Setting neither silently charges for the overhang, which changes
    every alignment, so the configuration is asserted by its effect rather
    than by which attribute exists.
    """
    from pdb_align.core import pairs_from_alignment, perform_sequence_alignment

    domain = "MKTAYIAKQRQISFVKSHFSRQ"
    full = "GSHMGSHM" + domain + "LEDPRVWQDFLSRAKEIVAGNC"

    aln = perform_sequence_alignment(full, domain, -10.0, -0.5)
    pairs = pairs_from_alignment(aln)
    # Every domain residue pairs, contiguously, at its true offset in `full`.
    assert len(pairs) == len(domain)
    assert pairs[0] == (full.index(domain), 0)
    assert [j for _i, j in pairs] == list(range(len(domain)))


def test_no_biopython_deprecation_warnings_escape():
    """A deprecation warning from a dependency is a maintenance alarm; it must
    not be part of normal output."""
    import warnings

    from pdb_align.core import perform_sequence_alignment

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        perform_sequence_alignment("MKTAYIAKQR", "MKTAYIAKQR", -10.0, -0.5)
