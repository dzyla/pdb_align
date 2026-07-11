import os
import zipfile
import json

from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")


def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")


def test_export_bundle_zip_contains_all(tmp_path):
    out = _result().export_bundle(str(tmp_path / "bundle.zip"))
    assert os.path.exists(out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.endswith("aligned.pdb") for n in names)
        assert any(n.endswith("rmsd.csv") for n in names)
        assert any(n.endswith(".pml") for n in names)
        assert any(n.endswith(".cxc") for n in names)
        assert any(n.endswith("report.txt") for n in names)
        assert any(n.endswith("report.json") for n in names)
        jname = [n for n in names if n.endswith("report.json")][0]
        with z.open(jname) as f:
            payload = json.load(f)
            assert "quality" in payload


def test_export_bundle_dir_subset(tmp_path):
    out = _result().export_bundle(str(tmp_path / "b"), include=["rmsd_csv"], fmt="dir")
    assert os.path.isdir(out)
    assert os.path.exists(os.path.join(out, "rmsd.csv"))
    assert not os.path.exists(os.path.join(out, "aligned.pdb"))
