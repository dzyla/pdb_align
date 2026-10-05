"""Synthetic gemmi complex builders shared by the interface/evaluate tests.

Chains are built as ideal alpha helices, not straight lines. Collinear CA
positions make a Kabsch fit rank-deficient about the chain axis, so a decoy
can be rotated around that axis for free: an interface test on rods validates
the arithmetic of a metric while saying nothing about whether it responds to
real geometry. The helix also gives side-chain atoms (CB) directions that
differ between residues, which is what contact-based metrics actually count.
"""
import gemmi
import numpy as np

SPACING = 3.8

# Ideal right-handed alpha helix: 1.5 A rise and 100 degrees of twist per
# residue on a 2.3 A radius (Pauling).
HELIX_RISE = 1.5
HELIX_RADIUS = 2.3
HELIX_TWIST_DEG = 100.0
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


def helix_ca(n_res, origin, axis=np.array([1.0, 0.0, 0.0]), phase=0.0):
    """CA positions of an ideal alpha helix of *n_res* residues along *axis*."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Two unit vectors spanning the plane perpendicular to the axis.
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    origin = np.asarray(origin, dtype=float)
    twist = np.radians(HELIX_TWIST_DEG)
    out = np.empty((n_res, 3))
    for i in range(n_res):
        ang = phase + twist * i
        out[i] = (origin + axis * (HELIX_RISE * i)
                  + HELIX_RADIUS * (np.cos(ang) * u + np.sin(ang) * v))
    return out


def make_chain(name, n_res, origin, resname="ALA", b_iso=50.0, start_num=1,
               direction=np.array([1.0, 0.0, 0.0]), sequence=None, phase=0.0):
    """Alpha-helical chain; `sequence` (one-letter) overrides `resname`/`n_res`."""
    chain = gemmi.Chain(name)
    if sequence is not None:
        n_res = len(sequence)
    cas = helix_ca(n_res, origin, axis=direction, phase=phase)
    for i in range(n_res):
        res = gemmi.Residue()
        res.name = AA_1TO3[sequence[i]] if sequence is not None else resname
        res.seqid = gemmi.SeqId(str(start_num + i))
        ca = cas[i]
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


def two_chain_complex(lig_origin=(0.0, 8.0, 0.0), lig_name="B", lig_resname="VAL",
                      rec_name="A", n_rec=30, n_lig=10, b_iso=50.0,
                      rec_start=1, lig_start=1, lig_axis=(1.0, 0.0, 0.0),
                      lig_phase=0.0):
    """Receptor helix along x at y=0; ligand helix offset to *lig_origin*.

    The two helices have a 2.3 A radius, so the default 8 A axis separation
    puts facing atoms around 3-4 A apart: a genuine interface with contacts
    that depend on the helical phase, and one that disappears as the ligand is
    moved away. Moving *lig_origin* is how the tests build decoys.
    """
    rec = make_chain(rec_name, n_rec, (0.0, 0.0, 0.0), resname="ALA",
                     b_iso=b_iso, start_num=rec_start)
    lig = make_chain(lig_name, n_lig, lig_origin, resname=lig_resname,
                     b_iso=b_iso, start_num=lig_start,
                     direction=np.asarray(lig_axis, dtype=float),
                     phase=lig_phase)
    return make_structure([rec, lig])


