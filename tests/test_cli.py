import os, json, gemmi
from pdb_align.__main__ import main


def _write_single(path, x0=0.0):
    st = gemmi.Structure(); model = gemmi.Model("1"); chain = gemmi.Chain("A")
    for k in range(20):
        res = gemmi.Residue(); res.name = "ALA"; res.seqid = gemmi.SeqId(k + 1, " ")
        at = gemmi.Atom(); at.name = "CA"; at.pos = gemmi.Position(k + x0, 0.0, 0.0)
        res.add_atom(at); chain.add_residue(res)
    model.add_chain(chain); st.add_model(model); st.write_pdb(path)


def test_default_run_writes_no_files(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    before = set(os.listdir(tmp_path))
    rc = main([str(ref), str(mob)])
    assert rc == 0
    assert set(os.listdir(tmp_path)) == before  # nothing written
    assert "RMSD" in capsys.readouterr().out


def test_json_flag_emits_valid_json(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main([str(ref), str(mob), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "strategy" in payload


def test_out_and_plot_flags_write_requested_files(tmp_path):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    out, plot = tmp_path / "aligned.pdb", tmp_path / "rmsd.png"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main([str(ref), str(mob), "-o", str(out), "--plot", str(plot)])
    assert rc == 0
    assert out.exists() and plot.exists()


def test_legacy_ref_mob_flags_still_work(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main(["--ref", str(ref), "--mob", str(mob)])
    assert rc == 0
    assert "RMSD" in capsys.readouterr().out
