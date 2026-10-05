import gemmi
import numpy as np

from pdb_align.chains import align_multichain, match_chains
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


def _two_chain(offset_b=0.0):
    a = [(i, 0.0, 0.0) for i in range(15)]
    b = [(i, 10.0 + offset_b, 0.0) for i in range(15)]
    return {"A": a, "B": b}


def test_global_strategy_superimposes_all_chains():
    ref = _struct(_two_chain(0.0))
    # mobile = ref rigidly translated by +5 in x
    mob = _struct({k: [(x + 5, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="global")
    assert res.strategy == "global"
    assert res.rmsd < 1e-6  # rigid translation recovered exactly
    assert len(res.per_chain) == 2


def test_local_beats_global_when_one_chain_diverges():
    ref = _struct(_two_chain(0.0))
    # chain B of mobile is badly displaced; chain A matches after rigid move
    mob = _struct({"A": [(x + 5, 0.0, 0.0) for x in range(15)],
                   "B": [(x + 5, 40.0, 30.0) for x in range(15)]})
    rs, _ = extract_sequences_and_lengths(ref, "r")
    ms, _ = extract_sequences_and_lengths(mob, "m")
    mapping = match_chains(rs, ms, ref, mob, ["A", "B"], ["A", "B"])
    res = align_multichain(ref, mob, mapping, strategy="auto")
    assert res.strategy == "local"
    assert res.rmsd < 1e-6


def test_pdbaligner_auto_uses_multichain(tmp_path):
    import gemmi

    import pdb_align
    ref = _struct(_two_chain(0.0))
    mob = _struct({k: [(x + 3, y, z) for x, y, z in v] for k, v in _two_chain(0.0).items()})
    rp, mp = tmp_path / "ref.pdb", tmp_path / "mob.pdb"
    ref.write_pdb(str(rp)); mob.write_pdb(str(mp))
    r = pdb_align.align(str(rp), str(mp))
    assert r.strategy in ("global", "local")
    assert not r.per_chain.empty
    assert r.rmsd < 1e-5


# --- FIX 1 regression: missing internal residue must not corrupt pairing ---

_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def _struct_with_seq(chain_seqs_coords):
    """chain_seqs_coords: {chain_name: [(one_letter_aa, x, y, z), ...]}"""
    st = gemmi.Structure(); model = gemmi.Model("1")
    for cname, entries in chain_seqs_coords.items():
        chain = gemmi.Chain(cname)
        for k, (aa, x, y, z) in enumerate(entries, start=1):
            res = gemmi.Residue(); res.name = _AA3[aa]; res.seqid = gemmi.SeqId(k, " ")
            at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(x, y, z)
            res.add_atom(at); chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model); return st


def test_missing_internal_residue_does_not_corrupt_rmsd(tmp_path):
    """Heteromeric two-chain case: mobile chain B is coordinate-identical to
    reference except one internal residue is unmodeled (dropped). Index-based
    pairing shifts everything after the gap by one and corrupts the RMSD;
    sequence-alignment-based pairing must correctly skip the gap and recover
    RMSD ~ 0."""
    import pdb_align

    seqA = "ACDEFGHIKLMNPQR"  # 15 unique residues -> unambiguous chain A
    seqB = "RQPNMLKIHGFEDCA"  # 15 unique residues, distinct from A -> unambiguous chain B

    ref_a = [(aa, i, 0.0, 0.0) for i, aa in enumerate(seqA)]
    ref_b = [(aa, i, 10.0, 0.0) for i, aa in enumerate(seqB)]
    ref = _struct_with_seq({"A": ref_a, "B": ref_b})

    # Mobile: chain A untouched; chain B missing its internal residue at
    # index 7 ('I'), coordinates otherwise identical (same absolute x, y, z).
    mob_a = list(ref_a)
    mob_b = [entry for idx, entry in enumerate(ref_b) if idx != 7]
    mob = _struct_with_seq({"A": mob_a, "B": mob_b})

    rp, mp = tmp_path / "ref.pdb", tmp_path / "mob.pdb"
    ref.write_pdb(str(rp)); mob.write_pdb(str(mp))

    r = pdb_align.align(str(rp), str(mp))
    assert r.rmsd is not None
    assert r.rmsd < 0.1, f"expected ~0 RMSD after correct sequence-based pairing, got {r.rmsd}"
