"""Tests for evaluate_models: ranking predicted models against a reference."""
import os

import numpy as np
import pytest

from pdb_align.evaluate import evaluate_models
from _synthetic import two_chain_complex, make_chain, make_structure

DATA = os.path.join(os.path.dirname(__file__), "data")
REF = os.path.join(DATA, "ref.pdb")
MOB = os.path.join(DATA, "mob.pdb")


def test_evaluate_models_ranks_perfect_model_first(tmp_path):
    ev = evaluate_models(REF, [MOB, REF], labels=["decoy", "perfect"])
    assert not ev.interface_scored
    assert ev.best == "perfect"
    row = ev.table.iloc[0]
    assert row["tm_score"] == pytest.approx(1.0, abs=1e-3)
    assert row["lddt_ca"] == pytest.approx(1.0, abs=1e-3)
    assert {"model", "rmsd", "tm_score", "gdt_ts", "lddt_ca",
            "coverage_pct"} <= set(ev.table.columns)
    rep = ev.report()
    assert "Best model: perfect" in rep
    assert "TM-score" in rep


def test_evaluate_models_interface_mode(tmp_path):
    native = two_chain_complex(b_iso=90.0)
    good = two_chain_complex(b_iso=90.0)
    bad = two_chain_complex(lig_origin=(0.0, 20.0, 0.0), b_iso=90.0)
    paths = {}
    for name, st in [("native", native), ("good", good), ("bad", bad)]:
        p = tmp_path / f"{name}.pdb"
        st.write_pdb(str(p))
        paths[name] = str(p)

    ev = evaluate_models(paths["native"], [paths["bad"], paths["good"]],
                         labels=["bad", "good"],
                         receptor_chains=["A"], ligand_chains=["B"])
    assert ev.interface_scored
    assert ev.best == "good"
    good_row = ev.table[ev.table["model"] == "good"].iloc[0]
    bad_row = ev.table[ev.table["model"] == "bad"].iloc[0]
    assert good_row["dockq"] == pytest.approx(1.0, abs=1e-6)
    assert good_row["capri"] == "high"
    assert bad_row["dockq"] < 0.23
    assert good_row["pdockq"] > bad_row["pdockq"]
    d = ev.to_dict()
    assert d["ranked_by"] == "dockq"
    assert d["best_model"] == "good"


def test_evaluate_models_antibody_mode(tmp_path):
    native = two_chain_complex(b_iso=90.0)
    slid = two_chain_complex(lig_origin=(4 * 3.8, 4.0, 0.0), b_iso=90.0)
    p_native = tmp_path / "native.pdb"
    p_slid = tmp_path / "slid.pdb"
    native.write_pdb(str(p_native))
    slid.write_pdb(str(p_slid))

    ev = evaluate_models(str(p_native), [str(p_slid)], labels=["slid"],
                         antibody_chains=["A"], antigen_chains=["B"])
    assert ev.antibody_mode
    row = ev.table.iloc[0]
    assert row["epitope_f1"] > 0.5      # right epitope
    assert row["dockq"] < 0.23          # wrong pose
    assert "Epitope F1" in ev.report()


def test_evaluate_models_bad_model_warns_not_crashes(tmp_path):
    broken = tmp_path / "broken.pdb"
    broken.write_text("not a structure\n")
    with pytest.warns(UserWarning, match="failed"):
        ev = evaluate_models(REF, [str(broken), REF], labels=["broken", "ok"])
    assert ev.best == "ok"
    broken_row = ev.table[ev.table["model"] == "broken"].iloc[0]
    assert np.isnan(broken_row["tm_score"])


def test_evaluate_models_argument_validation():
    with pytest.raises(ValueError, match="together"):
        evaluate_models(REF, [MOB], receptor_chains=["A"])
    with pytest.raises(ValueError, match="antigen_chains"):
        evaluate_models(REF, [MOB], antibody_chains=["H"])
    with pytest.raises(ValueError, match="not both"):
        evaluate_models(REF, [MOB], receptor_chains=["A"], ligand_chains=["B"],
                        antibody_chains=["H"], antigen_chains=["G"])
