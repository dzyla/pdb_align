"""Flexible (multi-domain) alignment correctness.

Ground truth used throughout: 4HHB with chains B and D rotated 25 degrees as
one rigid body. The correct decomposition is one rigid domain per chain, each
with RMSD ~ 0.
"""
import math

import gemmi
import numpy as np
import pytest

import pdb_align

REF = "tests/data/4hhb_bb.pdb"


@pytest.fixture(scope="module")
def hinged(tmp_path_factory):
    """4HHB with chains B and D moved as one rigid body."""
    st = gemmi.read_structure(REF)
    st.setup_entities()
    ang = math.radians(25.0)
    R = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                  [math.sin(ang), math.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    shift = np.array([5.0, 0.0, 0.0])
    for chain in st[0]:
        if chain.name in ("B", "D"):
            for res in chain:
                for atom in res:
                    v = np.array(atom.pos.tolist())
                    w = R @ v + shift
                    atom.pos = gemmi.Position(*w)
    path = tmp_path_factory.mktemp("flex") / "4hhb_hinged.pdb"
    st.write_pdb(str(path))
    return str(path)


def test_domains_never_span_a_chain_boundary(hinged):
    """A domain is a contiguous stretch of ONE chain.

    Hinge detection used to run on the concatenated per-residue array, which
    produced domains like 'chain A 42-21' that fused the tail of chain A to the
    head of chain B and then fitted the two with a single rotation.
    """
    res = pdb_align.align(REF, hinged, mode="flexible")
    assert res.domains, "the hinged complex must decompose into domains"
    for d in res.domains:
        assert d.residue_start <= d.residue_end, (
            f"domain {d.domain_id} on chain {d.chain_id} has start "
            f"{d.residue_start} > end {d.residue_end} (spans chains)")


def test_rigid_body_chain_motion_is_recovered_per_chain(hinged):
    """Each chain moves rigidly, so every domain must fit at ~0 A."""
    res = pdb_align.align(REF, hinged, mode="flexible")
    worst = max(d.rmsd for d in res.domains)
    assert worst < 0.5, f"domains should be rigid; worst domain RMSD {worst:.2f} A"


def test_flexible_rmsd_combines_domain_rmsds_in_quadrature():
    """RMSDs add in quadrature, never as an arithmetic mean.

    Two equal-sized domains at 1 A and 5 A give sqrt((1+25)/2) = 3.606 A,
    not (1+5)/2 = 3 A.
    """
    from pdb_align.aligner import AlignmentResult, DomainResult

    def dom(i, rmsd, n):
        return DomainResult(domain_id=i, chain_id="A", residue_start=1,
                            residue_end=n, n_residues=n, rmsd=rmsd,
                            rotation=np.eye(3), translation=np.zeros(3))

    res = AlignmentResult.__new__(AlignmentResult)
    res.domains = [dom(0, 1.0, 50), dom(1, 5.0, 50)]
    assert res.rmsd == pytest.approx(math.sqrt((1.0 + 25.0) / 2.0), abs=1e-9)


def test_flexible_domains_are_reported_per_chain_for_a_rigid_complex():
    """No hinge anywhere -> one domain per chain, not one domain overall."""
    res = pdb_align.align(REF, REF, mode="flexible")
    if res.domains:
        assert {d.chain_id for d in res.domains} == {"A", "B", "C", "D"}


def test_rigidly_moved_chains_yield_one_domain_each(hinged):
    """Four chains, two of them moved as one rigid body -> four rigid domains.

    Hinges are detected on deviations measured in the initial compromise frame,
    where even a rigidly-moved chain shows a deviation ramp and so collects
    spurious splits (this case produced nine "domains"). Adjacent segments that
    still fit together as one rigid body must be merged back.
    """
    res = pdb_align.align(REF, hinged, mode="flexible")
    assert len(res.domains) == 4
    assert sorted(d.chain_id for d in res.domains) == ["A", "B", "C", "D"]
    assert max(d.rmsd for d in res.domains) < 0.1


def test_a_real_hinge_is_still_split(tmp_path):
    """Merging must not erase genuine hinges: half a chain rotated in place
    cannot be fitted as one rigid body and must stay two domains."""
    st = gemmi.read_structure("tests/data/1ubq.pdb")
    st.setup_entities()
    ang = math.radians(40.0)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, math.cos(ang), -math.sin(ang)],
                  [0.0, math.sin(ang), math.cos(ang)]])
    for i, resi in enumerate(st[0]["A"]):
        if i >= 38:
            for atom in resi:
                atom.pos = gemmi.Position(*(R @ np.array(atom.pos.tolist())))
    path = tmp_path / "ubq_hinge.pdb"
    st.write_pdb(str(path))

    res = pdb_align.align("tests/data/1ubq.pdb", str(path), mode="flexible",
                          domain_min_residues=15)
    assert res.domains is not None and len(res.domains) >= 2
    assert max(d.rmsd for d in res.domains) < 1.0
