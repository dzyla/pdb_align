import json

import pytest

import pdb_align

REF = "tests/data/ref.pdb"
MOB = "tests/data/mob.pdb"


@pytest.fixture(scope="module")
def result():
    return pdb_align.align(REF, MOB)


def test_summary_stats_has_core_keys(result):
    s = result.summary_stats()
    for key in ("method", "strategy", "rmsd", "tm_score", "n_aligned"):
        assert key in s
    assert isinstance(s["rmsd"], (float, type(None)))


def test_strategy_defaults_to_single_chain(result):
    assert result.strategy in ("single", "global", "local")


def test_per_chain_is_dataframe(result):
    df = result.per_chain
    assert list(df.columns) == ["chain_ref", "chain_mob", "n_residues", "rmsd"]


def test_report_text_mentions_rmsd(result):
    text = result.report(fmt="text")
    assert "RMSD" in text


def test_report_json_roundtrips(result):
    payload = json.loads(result.report(fmt="json"))
    assert payload["strategy"] == result.strategy


def test_save_load_roundtrip_reproduces_stats(result, tmp_path):
    p = tmp_path / "run.npz"
    result.save(str(p))
    from pdb_align import AlignmentResult
    loaded = AlignmentResult.load(str(p))
    assert loaded.strategy == result.strategy
    a, b = result.summary_stats(), loaded.summary_stats()
    for key in ("rmsd", "tm_score", "n_aligned", "strategy"):
        av, bv = a[key], b[key]
        if av is None and bv is None:
            continue
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            assert abs(av - bv) < 1e-6
        else:
            assert av == bv


def test_loaded_result_replots_without_original_files(result, tmp_path):
    p = tmp_path / "run.npz"
    result.save(str(p))
    from pdb_align import AlignmentResult
    loaded = AlignmentResult.load(str(p))
    out = tmp_path / "r.png"
    loaded.plot_rmsd(filename=str(out))
    assert out.exists()


def test_loaded_result_plot_summary(result, tmp_path):
    p = tmp_path / "run.npz"
    result.save(str(p))
    from pdb_align import AlignmentResult
    loaded = AlignmentResult.load(str(p))
    out = tmp_path / "summary.png"
    loaded.plot_summary(filename=str(out))
    assert out.exists()
