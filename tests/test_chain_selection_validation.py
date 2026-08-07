"""A duplicate chain in a selection must raise, not silently produce a wrong RMSD.

Motivation: callers commonly build a mobile-chain list by mapping reference chains through
a correspondence table, falling back to the identity for entries the table omits. On a
multi-copy assembly where only one copy was modelled, two distinct reference chains then
land on the SAME mobile chain. Before this validation the aligner accepted that and fitted
one chain against two partners, reporting 23.65 A for a structure that superposes at
4.82 A. The failure is invisible in the result object -- the only tell is a per-chain
breakdown with identical values for chains that are not identical -- so it has to be
rejected at the point of entry.
"""
import os

import pytest

from pdb_align import PDBAligner


def _write_two_chain_pdb(path, offset=0.0):
    """Two short chains, A and B, separated along x so a mis-pairing is detectable."""
    lines = []
    serial = 1
    for chain, base in (("A", 0.0), ("B", 30.0)):
        for i in range(6):
            x = base + i * 3.8 + offset
            lines.append(
                "ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                % (serial, chain, i + 1, x, 0.0, 0.0)
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def two_chain_pair(tmp_path):
    ref = _write_two_chain_pdb(str(tmp_path / "ref.pdb"))
    mob = _write_two_chain_pdb(str(tmp_path / "mob.pdb"), offset=1.0)
    return ref, mob


def test_duplicate_mobile_chain_raises(two_chain_pair):
    ref, mob = two_chain_pair
    al = PDBAligner(ref, chains_ref=["A", "B"])
    al.add_mobile(mob)
    with pytest.raises(ValueError, match="duplicate"):
        al.set_mobile_chains(["A", "A"])


def test_duplicate_reference_chain_raises(two_chain_pair):
    ref, _ = two_chain_pair
    with pytest.raises(ValueError, match="duplicate"):
        PDBAligner(ref, chains_ref=["A", "A"])


def test_duplicate_in_add_mobile_raises(two_chain_pair):
    ref, mob = two_chain_pair
    al = PDBAligner(ref, chains_ref=["A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        al.add_mobile(mob, chains=["B", "B"])


def test_unknown_chain_raises(two_chain_pair):
    ref, mob = two_chain_pair
    al = PDBAligner(ref, chains_ref=["A"])
    al.add_mobile(mob)
    with pytest.raises(ValueError, match="not present"):
        al.set_mobile_chains(["Z"])


def test_valid_selection_still_works(two_chain_pair):
    """Validation must not break the ordinary path."""
    ref, mob = two_chain_pair
    al = PDBAligner(ref, chains_ref=["A", "B"])
    al.add_mobile(mob)
    al.set_mobile_chains(["A", "B"])
    res = al.align(mode="seq_guided", atoms="CA")
    assert res.rmsd is not None
    assert res.rmsd < 5.0


def test_none_selection_means_all_chains(two_chain_pair):
    """chains=None is not a selection and must remain untouched by validation."""
    ref, mob = two_chain_pair
    al = PDBAligner(ref)
    al.add_mobile(mob)
    assert al.chains_ref is None
    assert al.chains_mob is None
    res = al.align(mode="seq_guided", atoms="CA")
    assert res.rmsd is not None
