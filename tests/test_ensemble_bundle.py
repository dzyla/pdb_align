import os
import zipfile

from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")


def _ensemble():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    return al.align_ensemble([os.path.join(DATA, "mob.pdb"),
                              os.path.join(DATA, "ref.pdb")], mode="auto")


def test_ensemble_bundle_contains_tables(tmp_path):
    out = _ensemble().export_bundle(str(tmp_path / "ens.zip"))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.endswith("summary.csv") for n in names)
        assert any(n.endswith("rmsd_matrix.csv") for n in names)
