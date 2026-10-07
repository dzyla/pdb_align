"""Chain correspondence must never contradict sequence, or cross a group.

4HHB is the fixture because it is the hard case that occurs constantly in real
work: a dimer of heterodimers, where chains A and C are identical α-globin and
B and D identical β-globin. Sequence alone cannot tell the two α copies apart
(their identity is exactly tied), so geometry has to break that tie — but
geometry must only choose among candidates sequence says are interchangeable.
An α chain must never be paired with a β chain, however convenient its
centroid.
"""
import math
import warnings

import gemmi
import numpy as np
import pytest

from pdb_align.interface import compute_dockq

HHB = "tests/data/4hhb_bb.pdb"


@pytest.fixture(scope="module")
def native_and_rotated(tmp_path_factory):
    """4HHB, and a copy with chains C+D rotated 40° as one rigid body."""
    out = tmp_path_factory.mktemp("hhb")
    nat = gemmi.read_structure(HHB)
    nat.setup_entities()
    p_native = out / "native.pdb"
    nat.write_pdb(str(p_native))

    mod = gemmi.read_structure(HHB)
    mod.setup_entities()
    a = math.radians(40.0)
    R = np.array([[math.cos(a), -math.sin(a), 0.0],
                  [math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    for chain in mod[0]:
        if chain.name in ("C", "D"):
            for res in chain:
                for atom in res:
                    atom.pos = gemmi.Position(*(R @ np.array(atom.pos.tolist())))
    p_model = out / "model.pdb"
    mod.write_pdb(str(p_model))
    return str(p_native), str(p_model)


@pytest.mark.parametrize("receptor,ligand", [
    (["A", "B"], ["C", "D"]),
    (["C", "D"], ["A", "B"]),   # the same interface, groups named the other way
])
def test_chain_mapping_is_the_identity_for_an_unrelabelled_model(
        native_and_rotated, receptor, ligand):
    """The model's chains carry their own names, so every chain must map to
    itself — in either group order.

    Giving the receptor group only part of the symmetric chains used to let its
    matching search the *whole* model, steal the chains belonging to the ligand
    group, and (because the geometric tie-break ignored sequence) pair α with
    β: receptor C->B, D->A, which drove fnat to 0.
    """
    p_native, p_model = native_and_rotated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = compute_dockq(p_native, p_model, receptor, ligand)
    assert res.receptor_mapping == [(c, c) for c in receptor]
    assert res.ligand_mapping == [(c, c) for c in ligand]


def test_dockq_is_the_same_interface_whichever_group_is_called_the_receptor(
        native_and_rotated):
    """fnat and iRMSD describe the interface itself, so naming the groups the
    other way round cannot change them. (LRMSD legitimately differs: it is
    measured after superposing on whichever group is the receptor.)"""
    p_native, p_model = native_and_rotated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ab = compute_dockq(p_native, p_model, ["A", "B"], ["C", "D"])
        cd = compute_dockq(p_native, p_model, ["C", "D"], ["A", "B"])
    assert ab.fnat == pytest.approx(cd.fnat)
    assert ab.n_native_contacts == cd.n_native_contacts
    assert ab.irmsd == pytest.approx(cd.irmsd, abs=1e-6)


def test_lrmsd_matches_a_direct_computation(native_and_rotated):
    """LRMSD is the ligand backbone RMSD after superposing on the receptor.

    Checked against plain numpy so the value cannot drift with the chain
    mapping: a wrong mapping gave 35.8 A where the correct answer is 20.6 A.
    """
    p_native, p_model = native_and_rotated
    backbone = ("N", "CA", "C", "O")

    def coords(path, chains):
        st = gemmi.read_structure(path)
        st.setup_entities()
        out = []
        for cname in chains:
            for res in st[0][cname]:
                for name in backbone:
                    for atom in res:
                        if atom.name == name:
                            out.append(atom.pos.tolist())
                            break
        return np.array(out)

    def kabsch(P, Q):
        cP, cQ = P.mean(0), Q.mean(0)
        U, _S, Vt = np.linalg.svd((Q - cQ).T @ (P - cP))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return R, cP - R @ cQ

    for receptor, ligand in ((["A", "B"], ["C", "D"]), (["C", "D"], ["A", "B"])):
        R, t = kabsch(coords(p_native, receptor), coords(p_model, receptor))
        Pl, Ql = coords(p_native, ligand), coords(p_model, ligand)
        expected = float(np.sqrt((((R @ Ql.T).T + t - Pl) ** 2).sum(1).mean()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            got = compute_dockq(p_native, p_model, receptor, ligand).lrmsd
        assert got == pytest.approx(expected, abs=1e-6), (receptor, ligand)


def test_geometry_may_not_pair_chains_of_different_sequence():
    """Geometry breaks ties between interchangeable chains; it does not get a
    vote when sequence already decides."""
    from pdb_align.chains import match_chains
    from pdb_align.core import extract_sequences_and_lengths

    st = gemmi.read_structure(HHB)
    st.setup_entities()
    seqs, _ = extract_sequences_and_lengths(st, "hhb")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mapping = match_chains(seqs, seqs, st, st, ["A", "B"], ["A", "B", "C", "D"])
    for ref, mob, identity, _score in mapping.pairs:
        assert identity > 90.0, (ref, mob, identity)


def _chain_residues(struct, chains, trim=0, trim_chain=None):
    from pdb_align.core import select_residues
    out = {}
    for c in chains:
        residues = list(select_residues(struct, [c]).residues)
        if trim and c == trim_chain:
            residues = residues[trim:]
        out[c] = residues
    return out


def test_mapping_search_treats_near_identical_copies_as_interchangeable():
    """Copies of one chain in a real native rarely have identical sequences.

    One copy is always missing a terminus or a disordered loop. Requiring
    byte-identical sequences before two chains may be swapped switched the
    fnat-maximising search off on exactly the structures it exists for: on
    4HHB, removing three modelled residues from one alpha chain dropped the
    candidate mappings from four to two, and the alpha/alpha swap — the whole
    point of the search — was no longer among them.
    """
    from pdb_align.interface import _joint_mapping_options

    st = gemmi.read_structure(HHB)
    st.setup_entities()
    rec_pairs = [("A", "A"), ("B", "B")]
    lig_pairs = [("C", "C"), ("D", "D")]
    ref_lig = _chain_residues(st, ["C", "D"])

    identical = _joint_mapping_options(rec_pairs, lig_pairs,
                                       _chain_residues(st, ["A", "B"]), ref_lig)
    near = _joint_mapping_options(
        rec_pairs, lig_pairs,
        _chain_residues(st, ["A", "B"], trim=3, trim_chain="A"), ref_lig)

    assert len(identical) == 4          # alpha swap x beta swap
    assert len(near) == len(identical)
    assert any(dict(rec)["A"] == "C" for rec, _lig in near)


def test_mapping_search_never_swaps_chains_of_different_sequence():
    """Regression guard: an alpha-globin chain is not interchangeable with a
    beta-globin one, however the grouping is computed."""
    from pdb_align.interface import _joint_mapping_options

    st = gemmi.read_structure(HHB)
    st.setup_entities()
    opts = _joint_mapping_options([("A", "A")], [("B", "B")],
                                  _chain_residues(st, ["A"]),
                                  _chain_residues(st, ["B"]))
    assert opts == [([("A", "A")], [("B", "B")])]
