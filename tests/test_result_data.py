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
