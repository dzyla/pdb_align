"""Tests for CDR annotation and per-CDR RMSD.

The geometry tests use an injected deterministic numberer so they run without
ANARCI; a separate integration test exercises real ANARCI numbering on the
D1.3 antibody sequence and is skipped when ANARCI/HMMER are not functional.
"""
import gemmi
import numpy as np
import pytest
from _synthetic import make_chain, make_structure

from pdb_align.cdr import IMGT_CDR_RANGES, annotate_cdrs, cdr_rmsd

# D1.3 anti-lysozyme VH/VL (from PDB 1VFB chains B/A)
D13_VH = ("QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSR"
          "LSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS")
D13_VL = ("DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSG"
          "SGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK")


def _identity_numberer(seq):
    """Fake IMGT numbering: residue i -> IMGT position i+1, chain type H.

    Covers up to IMGT 128, which is enough to place CDR1/2/3 at fixed,
    analytically known sequence positions.
    """
    n = min(len(seq), 128)
    numbering = [(((i + 1), " "), seq[i]) for i in range(n)]
    return numbering, 0, "H"


def _anarci_works():
    try:
        from pdb_align.cdr import anarci_numberer
        out = anarci_numberer(D13_VH)
        return out is not None and out[2] == "H"
    except Exception:
        return False


# --- annotation with injected numberer ---------------------------------------

def test_annotate_cdrs_with_injected_numberer():
    seq = "A" * 128
    ann = annotate_cdrs(seq, numberer=_identity_numberer)
    assert ann.chain_type == "H"
    # identity numbering: IMGT position p sits at sequence index p-1
    for name, (lo, hi) in IMGT_CDR_RANGES.items():
        assert ann.regions[name] == list(range(lo - 1, hi))
    # framework = numbered positions not in any CDR
    all_cdr = {i for idxs in ann.regions.values() for i in idxs}
    assert set(ann.framework).isdisjoint(all_cdr)
    assert len(ann.framework) == 128 - len(all_cdr)


def test_annotate_cdrs_not_an_antibody_returns_none():
    assert annotate_cdrs("AAAA", numberer=lambda s: None) is None


# --- per-CDR RMSD geometry (no ANARCI needed) --------------------------------

def _displace_region(struct, chain_name, seq_positions, dy):
    for chain in struct[0]:
        if chain.name != chain_name:
            continue
        for i, res in enumerate(chain):
            if i in seq_positions:
                for atom in res:
                    atom.pos = gemmi.Position(atom.pos.x, atom.pos.y + dy,
                                              atom.pos.z)


def test_cdr_rmsd_identity_all_zero():
    ref = make_structure([make_chain("H", 128, (0, 0, 0))])
    mod = make_structure([make_chain("H", 128, (0, 0, 0))])
    r = cdr_rmsd(ref, mod, ["H"], model_antibody_chains=["H"],
                 numberer=_identity_numberer)
    assert r.framework_rmsd == pytest.approx(0.0, abs=1e-9)
    for name in ("H1", "H2", "H3"):
        assert r.cdr_rmsd[name] == pytest.approx(0.0, abs=1e-9)


def test_cdr_rmsd_displaced_h3_exact():
    # Displace exactly the CDR3 residues (IMGT 105-117 = seq idx 104..116) by
    # 5 A. The framework superposition is then the identity, so H3 RMSD = 5
    # exactly while H1/H2/framework stay at 0.
    shift = 5.0
    ref = make_structure([make_chain("H", 128, (0, 0, 0))])
    mod = make_structure([make_chain("H", 128, (0, 0, 0))])
    cdr3_idx = set(range(104, 117))
    _displace_region(mod, "H", cdr3_idx, shift)
    r = cdr_rmsd(ref, mod, ["H"], model_antibody_chains=["H"],
                 numberer=_identity_numberer)
    assert r.framework_rmsd == pytest.approx(0.0, abs=1e-9)
    assert r.cdr_rmsd["H1"] == pytest.approx(0.0, abs=1e-9)
    assert r.cdr_rmsd["H2"] == pytest.approx(0.0, abs=1e-9)
    assert r.cdr_rmsd["H3"] == pytest.approx(shift, abs=1e-9)
    assert r.h3 == pytest.approx(shift, abs=1e-9)
    assert r.cdr_n_residues["H3"] == 13


def test_cdr_rmsd_numbering_offset_irrelevant():
    # Model numbered from 501: residue correspondence comes from sequence
    # alignment, so results are identical.
    ref = make_structure([make_chain("H", 128, (0, 0, 0))])
    mod = make_structure([make_chain("H", 128, (0, 0, 0), start_num=501)])
    _displace_region(mod, "H", set(range(104, 117)), 3.0)
    r = cdr_rmsd(ref, mod, ["H"], model_antibody_chains=["H"],
                 numberer=_identity_numberer)
    assert r.cdr_rmsd["H3"] == pytest.approx(3.0, abs=1e-9)


def test_cdr_rmsd_no_antibody_chain_raises():
    ref = make_structure([make_chain("X", 40, (0, 0, 0))])
    mod = make_structure([make_chain("X", 40, (0, 0, 0))])
    with pytest.raises(ValueError, match="antibody variable domain"):
        with pytest.warns(UserWarning):
            cdr_rmsd(ref, mod, ["X"], model_antibody_chains=["X"],
                     numberer=lambda s: None)


# --- real ANARCI integration --------------------------------------------------

@pytest.mark.skipif(not _anarci_works(),
                    reason="ANARCI/HMMER not installed or not functional")
def test_real_anarci_d13_annotation_and_rmsd():
    ann = annotate_cdrs(D13_VH)
    assert ann.chain_type == "H"
    assert ann.sequences["CDR3"] == "ARERDYRLDY"   # D1.3 CDR-H3 (IMGT 105-117)

    ann_l = annotate_cdrs(D13_VL)
    assert ann_l.chain_type in ("K", "L")
    assert ann_l.sequences["CDR3"] == "QHFWSTPRT"

    # Fv with both chains; identity model -> all CDR RMSDs ~ 0
    ref = make_structure([make_chain("H", 0, (0, 0, 0), sequence=D13_VH),
                          make_chain("L", 0, (0, 300, 0), sequence=D13_VL)])
    mod = make_structure([make_chain("H", 0, (0, 0, 0), sequence=D13_VH),
                          make_chain("L", 0, (0, 300, 0), sequence=D13_VL)])
    r = cdr_rmsd(ref, mod, ["H", "L"], model_antibody_chains=["H", "L"])
    assert r.chain_types["H"] == "H"
    assert r.chain_types["L"] in ("K", "L")
    assert set(r.cdr_rmsd) == {"H1", "H2", "H3", "L1", "L2", "L3"}
    for v in r.cdr_rmsd.values():
        assert v == pytest.approx(0.0, abs=1e-6)
    assert r.cdr_sequences["H3"] == "ARERDYRLDY"
