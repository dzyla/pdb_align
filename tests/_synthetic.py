"""Synthetic gemmi complex builders shared by the interface/evaluate tests."""
import numpy as np
import gemmi

SPACING = 3.8
# per-residue heavy atoms, offsets from the CA position
ATOM_OFFSETS = {
    "N":  np.array([-0.6, 0.5, 0.0]),
    "CA": np.array([0.0, 0.0, 0.0]),
    "C":  np.array([0.6, 0.5, 0.0]),
    "O":  np.array([0.6, 1.5, 0.0]),
    "CB": np.array([0.0, -1.2, 0.8]),
}
_ELEMENT = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C"}

AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def make_chain(name, n_res, origin, resname="ALA", b_iso=50.0, start_num=1,
               direction=np.array([1.0, 0.0, 0.0]), sequence=None):
    """Straight-line chain; `sequence` (one-letter) overrides `resname`/`n_res`."""
    chain = gemmi.Chain(name)
    origin = np.asarray(origin, dtype=float)
    if sequence is not None:
        n_res = len(sequence)
    for i in range(n_res):
        res = gemmi.Residue()
        res.name = AA_1TO3[sequence[i]] if sequence is not None else resname
        res.seqid = gemmi.SeqId(str(start_num + i))
        ca = origin + direction * (SPACING * i)
        for aname, off in ATOM_OFFSETS.items():
            if res.name == "GLY" and aname == "CB":
                continue
            atom = gemmi.Atom()
            atom.name = aname
            atom.element = gemmi.Element(_ELEMENT[aname])
            atom.pos = gemmi.Position(*(ca + off))
            atom.b_iso = b_iso
            atom.occ = 1.0
            res.add_atom(atom)
        chain.add_residue(res)
    return chain


def make_structure(chains):
    st = gemmi.Structure()
    st.name = "synthetic"
    model = gemmi.Model("1")
    for ch in chains:
        model.add_chain(ch)
    st.add_model(model)
    return st


def two_chain_complex(lig_origin=(0.0, 4.0, 0.0), lig_name="B", lig_resname="VAL",
                      rec_name="A", n_rec=30, n_lig=10, b_iso=50.0,
                      rec_start=1, lig_start=1):
    """Receptor along x at y=0; ligand parallel at the given origin.

    With |dy| = 4.0 the CA-CA cross-chain distance is 4.0 < 5 A, so the
    chains form a genuine interface.
    """
    rec = make_chain(rec_name, n_rec, (0.0, 0.0, 0.0), resname="ALA",
                     b_iso=b_iso, start_num=rec_start)
    lig = make_chain(lig_name, n_lig, lig_origin, resname=lig_resname,
                     b_iso=b_iso, start_num=lig_start)
    return make_structure([rec, lig])


