"""save_aligned_pdb B-factor handling.

By default the aligned output encodes per-residue deviation in the B-factor
column (for heat-map visualisation). That is destructive for inputs whose
B-factor column carries meaning (e.g. AlphaFold pLDDT), so callers can opt to
preserve the original values instead.
"""
import gemmi
import pdb_align


_PDB = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 50.00           C
ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00 50.00           C
ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00 50.00           C
END
"""


def _bfactors(path):
    st = gemmi.read_structure(str(path))
    return [a.b_iso for chain in st[0] for res in chain for a in res]


def test_preserve_bfactor_keeps_original(tmp_path):
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(_PDB)
    mob.write_text(_PDB)

    result = pdb_align.align(str(ref), str(mob))

    out = tmp_path / "aligned.pdb"
    result.save_aligned_pdb(str(out), preserve_bfactor=True)
    assert all(abs(b - 50.0) < 1e-3 for b in _bfactors(out))


def test_default_writes_rmsd_into_bfactor(tmp_path):
    ref = tmp_path / "ref.pdb"
    mob = tmp_path / "mob.pdb"
    ref.write_text(_PDB)
    mob.write_text(_PDB)

    result = pdb_align.align(str(ref), str(mob))

    out = tmp_path / "aligned.pdb"
    result.save_aligned_pdb(str(out))
    # Identical structures → zero deviation written into B-factor column.
    assert all(abs(b) < 1e-3 for b in _bfactors(out))
