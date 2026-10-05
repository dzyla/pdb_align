"""Tests for confidence-file ingestion (PAE/ipTM) and pDockQ2."""
import json
import math

import numpy as np
import pytest
from _synthetic import two_chain_complex

from pdb_align.confidence import (
    PDOCKQ2_B,
    PDOCKQ2_K,
    PDOCKQ2_L,
    PDOCKQ2_X0,
    compute_pdockq2,
    find_confidence_files,
    load_confidence,
)


def _sigmoid(x):
    return PDOCKQ2_L / (1 + math.exp(-PDOCKQ2_K * (x - PDOCKQ2_X0))) + PDOCKQ2_B


# --- format parsing ---------------------------------------------------------

def test_load_af2_colabfold_json(tmp_path):
    n = 6
    d = {"plddt": [90.0] * n, "pae": [[1.0] * n for _ in range(n)],
         "max_pae": 31.75, "ptm": 0.85, "iptm": 0.80}
    p = tmp_path / "model_scores.json"
    p.write_text(json.dumps(d))
    c = load_confidence(str(p))
    assert c.format == "af2/colabfold"
    assert c.iptm == pytest.approx(0.80)
    assert c.ptm == pytest.approx(0.85)
    assert c.pae.shape == (n, n)
    assert c.plddt.shape == (n,)


def test_load_af3_summary_and_full(tmp_path):
    summary = {"iptm": 0.9, "ptm": 0.92, "ranking_score": 0.88,
               "chain_pair_iptm": [[0.9, 0.8], [0.8, 0.95]],
               "chain_iptm": [0.9, 0.9], "fraction_disordered": 0.0}
    full = {"pae": [[0.5] * 4 for _ in range(4)],
            "token_chain_ids": ["A", "A", "B", "B"]}
    ps = tmp_path / "foo_summary_confidences.json"
    pf = tmp_path / "foo_confidences.json"
    ps.write_text(json.dumps(summary))
    pf.write_text(json.dumps(full))
    c = load_confidence([str(ps), str(pf)])
    assert c.format == "af3"
    assert c.iptm == pytest.approx(0.9)
    assert c.ranking_score == pytest.approx(0.88)
    assert c.pae.shape == (4, 4)
    assert c.token_chain_ids == ["A", "A", "B", "B"]
    assert c.chain_pair_iptm.shape == (2, 2)


def test_load_boltz_json_and_pae_npz(tmp_path):
    d = {"confidence_score": 0.8, "iptm": 0.77, "ptm": 0.81,
         "complex_plddt": 0.9,
         "pair_chains_iptm": {"0": {"0": 0.9, "1": 0.7},
                              "1": {"0": 0.7, "1": 0.95}}}
    pj = tmp_path / "confidence_model_0.json"
    pj.write_text(json.dumps(d))
    pn = tmp_path / "pae_model_0.npz"
    np.savez(pn, pae=np.full((5, 5), 2.0))
    c = load_confidence([str(pj), str(pn)])
    assert c.format == "boltz"
    assert c.iptm == pytest.approx(0.77)
    assert c.chain_pair_iptm[0, 1] == pytest.approx(0.7)
    assert c.pae.shape == (5, 5)


def test_find_confidence_files_conventions(tmp_path):
    model = tmp_path / "pred_model.cif"
    model.write_text("")
    for name in ("pred_summary_confidences.json", "pred_confidences.json",
                 "confidence_pred_model.json", "pae_pred_model.npz"):
        (tmp_path / name).write_text("{}")
    found = {p.split("/")[-1] for p in find_confidence_files(str(model))}
    assert "pred_summary_confidences.json" in found
    assert "pred_confidences.json" in found
    assert "confidence_pred_model.json" in found
    assert "pae_pred_model.npz" in found


def test_find_confidence_files_none(tmp_path):
    model = tmp_path / "lonely.pdb"
    model.write_text("")
    assert find_confidence_files(str(model)) == []


# --- pDockQ2 ----------------------------------------------------------------

def test_pdockq2_analytic_uniform_pae():
    # Uniform PAE and pLDDT make X analytic:
    # X = (1/(1+(pae/10)^2)) * plddt for both directions.
    st = two_chain_complex(b_iso=90.0)   # 30 + 10 residues
    n = 40
    pae_val = 5.0
    pae = np.full((n, n), pae_val)
    r = compute_pdockq2(st, ["A"], ["B"], pae)
    assert r.n_contacts > 0
    x = (1.0 / (1.0 + (pae_val / 10.0) ** 2)) * 90.0
    assert r.pdockq2 == pytest.approx(_sigmoid(x), abs=1e-12)
    assert r.pdockq2_ab == pytest.approx(r.pdockq2_ba)   # symmetric PAE
    assert r.mean_interface_pae == pytest.approx(pae_val)
    assert r.mean_interface_plddt == pytest.approx(90.0)


def test_pdockq2_low_pae_beats_high_pae():
    st = two_chain_complex(b_iso=90.0)
    good = compute_pdockq2(st, ["A"], ["B"], np.full((40, 40), 1.0))
    bad = compute_pdockq2(st, ["A"], ["B"], np.full((40, 40), 25.0))
    assert good.pdockq2 > bad.pdockq2


def test_pdockq2_asymmetric_pae_directions_differ():
    st = two_chain_complex(b_iso=90.0)
    pae = np.full((40, 40), 20.0)
    pae[:30, 30:] = 2.0   # A->B confident, B->A not
    r = compute_pdockq2(st, ["A"], ["B"], pae)
    assert r.pdockq2_ab > r.pdockq2_ba
    assert r.pdockq2 == pytest.approx((r.pdockq2_ab + r.pdockq2_ba) / 2)


def test_pdockq2_wrong_pae_size_raises():
    st = two_chain_complex()
    with pytest.raises(ValueError, match="cannot map PAE"):
        compute_pdockq2(st, ["A"], ["B"], np.zeros((7, 7)))


def test_pdockq2_no_contacts_zero():
    st = two_chain_complex(lig_origin=(0.0, 100.0, 0.0))
    r = compute_pdockq2(st, ["A"], ["B"], np.zeros((40, 40)))
    assert r.pdockq2 == 0.0
    assert r.n_contacts == 0
