import json
import os
import zipfile

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
        jname = next(n for n in names if n.endswith("report.json"))
        with z.open(jname) as f:
            payload = json.load(f)
            assert "quality" in payload


def test_export_bundle_dir_subset(tmp_path):
    out = _result().export_bundle(str(tmp_path / "b"), include=["rmsd_csv"], fmt="dir")
    assert os.path.isdir(out)
    assert os.path.exists(os.path.join(out, "rmsd.csv"))
    assert not os.path.exists(os.path.join(out, "aligned.pdb"))


def test_bundle_ships_the_reference_and_scripts_load_both(tmp_path):
    """A bundle whose viewer scripts load the aligned mobile twice is useless.

    The point of the bundle is a side-by-side view, so the reference must be in
    it and the .pml/.cxc must load reference + aligned mobile, not one file
    twice.
    """
    out = _result().export_bundle(str(tmp_path / "bundle.zip"))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "reference.pdb" in names
        pml = z.read("view.pml").decode()
        cxc = z.read("view.cxc").decode()
    assert "reference.pdb" in pml and "aligned.pdb" in pml
    assert pml.count("load ") == 2
    assert "reference.pdb" in cxc and "aligned.pdb" in cxc
    assert cxc.count("open ") == 2
