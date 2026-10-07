import json
import os

import gemmi

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


def test_json_with_verbose_still_valid_json(tmp_path, capsys):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    rc = main([str(ref), str(mob), "--json", "-v"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)   # must not raise
    assert "strategy" in payload


def test_missing_args_returns_2(capsys):
    rc = main([])
    assert rc == 2


def test_alignment_failure_returns_1(tmp_path):
    mob = tmp_path / "m.pdb"; _write_single(str(mob))
    rc = main([str(tmp_path / "does_not_exist.pdb"), str(mob)])
    assert rc == 1


def _write_helix_with_bfactors(path, bfactors, x0=0.0):
    """One helical chain, one B-factor per residue."""
    from _synthetic import make_chain, make_structure
    chain = make_chain("A", len(bfactors), (x0, 0.0, 0.0))
    for res, b in zip(chain, bfactors):
        for atom in res:
            atom.b_iso = float(b)
    st = make_structure([chain])
    st.setup_entities()
    st.write_pdb(str(path))


def test_min_b_factor_flag_filters_disordered_residues(tmp_path, capsys):
    """The CLI's own pLDDT warning tells users to filter on B-factors instead,
    and there was no flag to do it with."""
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    bfs = [90.0] * 10 + [10.0] * 10
    _write_helix_with_bfactors(ref, bfs)
    _write_helix_with_bfactors(mob, bfs)

    rc = main([str(ref), str(mob), "--min-b-factor", "50", "--json"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["n_aligned"] == 10


def test_export_bundle_flag_writes_a_bundle(tmp_path):
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    bundle = tmp_path / "bundle.zip"

    rc = main([str(ref), str(mob), "--export-bundle", str(bundle)])

    assert rc == 0
    assert bundle.exists()


def test_an_unknown_mode_is_rejected_by_the_parser(tmp_path):
    """`--mode seqguided` (a plausible typo) used to reach the aligner and come
    back as 'produced no alignment. unknown mode'."""
    import pytest
    ref, mob = tmp_path / "r.pdb", tmp_path / "m.pdb"
    _write_single(str(ref)); _write_single(str(mob), x0=2.0)
    with pytest.raises(SystemExit) as exc:
        main([str(ref), str(mob), "--mode", "seqguided"])
    assert exc.value.code == 2


def test_verbose_failure_shows_a_traceback(tmp_path, capsys):
    """Without it, every failure — including a bug — is one opaque line."""
    mob = tmp_path / "m.pdb"; _write_single(str(mob))
    rc = main([str(tmp_path / "missing.pdb"), str(mob), "-v"])
    assert rc == 1
    assert "Traceback" in capsys.readouterr().err
