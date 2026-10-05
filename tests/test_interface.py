"""Analytic golden tests for the DockQ / epitope / pDockQ interface layer.

Synthetic complexes are built in gemmi with exactly controlled geometry so
several metric values are known in closed form (no external files, no network).
"""
import numpy as np
import pytest
from _synthetic import SPACING, make_chain, make_structure, two_chain_complex

from pdb_align.interface import (
    DOCKQ_D1_IRMSD,
    DOCKQ_D2_LRMSD,
    FNAT_CONTACT_CUTOFF,
    capri_class,
    compute_dockq,
    compute_pdockq,
    dockq_formula,
    epitope_metrics,
    evaluate_antibody_complex,
)

# --------------------------------------------------------------------------
# DockQ
# --------------------------------------------------------------------------

def test_dockq_identity_is_perfect():
    native = two_chain_complex()
    model = two_chain_complex()
    r = compute_dockq(native, model, ["A"], ["B"])
    assert r.fnat == pytest.approx(1.0)
    assert r.fnonnat == pytest.approx(0.0)
    assert r.irmsd == pytest.approx(0.0, abs=1e-6)
    assert r.lrmsd == pytest.approx(0.0, abs=1e-6)
    assert r.dockq == pytest.approx(1.0, abs=1e-6)
    assert r.capri == "high"
    assert r.n_native_contacts > 0


def test_dockq_pure_ligand_translation_exact_lrmsd():
    # Ligand rigidly translated 12 A away: the receptor superposition is the
    # identity, so LRMSD equals the translation exactly; all native contacts
    # are lost (they were at ~4-5 A, now >= 12 A), so fnat = 0.
    shift = 12.0
    native = two_chain_complex(lig_origin=(0.0, 4.0, 0.0))
    model = two_chain_complex(lig_origin=(0.0, 4.0 + shift, 0.0))
    r = compute_dockq(native, model, ["A"], ["B"])
    assert r.fnat == pytest.approx(0.0)
    assert r.lrmsd == pytest.approx(shift, abs=1e-6)
    expected = dockq_formula(0.0, r.irmsd, shift)
    assert r.dockq == pytest.approx(expected, abs=1e-9)
    assert r.capri == "incorrect"


def test_dockq_formula_and_capri_bands():
    assert dockq_formula(1.0, 0.0, 0.0) == pytest.approx(1.0)
    # published scaling constants
    assert dockq_formula(0.0, DOCKQ_D1_IRMSD, DOCKQ_D2_LRMSD) == pytest.approx(
        (0.0 + 0.5 + 0.5) / 3.0)
    assert capri_class(0.10) == "incorrect"
    assert capri_class(0.23) == "acceptable"
    assert capri_class(0.49) == "medium"
    assert capri_class(0.80) == "high"


def test_dockq_robust_to_chain_renaming_and_numbering_offset():
    # Same coordinates, but the model names its chains R/L and numbers
    # residues from 101: DockQ must still be 1 (correspondence comes from
    # sequence alignment + chain matching, not author bookkeeping).
    native = two_chain_complex()
    model = two_chain_complex(rec_name="R", lig_name="L",
                              rec_start=101, lig_start=201)
    r = compute_dockq(native, model, ["A"], ["B"])
    assert r.dockq == pytest.approx(1.0, abs=1e-6)
    assert ("A", "R") in r.receptor_mapping
    assert ("B", "L") in r.ligand_mapping


def test_dockq_homomultimer_permutation_recovers_swapped_chains():
    # Two identical ligand chains: B at the interface, C far away. The model
    # provides them with swapped names; the explicit (wrong) mapping must be
    # rescued by the fnat-maximizing permutation search.
    rec = make_chain("A", 30, (0.0, 0.0, 0.0), resname="ALA")
    lig_near = make_chain("B", 10, (0.0, 4.0, 0.0), resname="VAL")
    lig_far = make_chain("C", 10, (0.0, 60.0, 0.0), resname="VAL")
    native = make_structure([rec, lig_near, lig_far])

    rec_m = make_chain("A", 30, (0.0, 0.0, 0.0), resname="ALA")
    lig_near_m = make_chain("C", 10, (0.0, 4.0, 0.0), resname="VAL")   # swapped
    lig_far_m = make_chain("B", 10, (0.0, 60.0, 0.0), resname="VAL")
    model = make_structure([rec_m, lig_far_m, lig_near_m])

    r = compute_dockq(native, model, ["A"], ["B", "C"],
                      model_receptor_chains=["A"],
                      model_ligand_chains=["B", "C"])  # deliberately wrong order
    assert r.fnat == pytest.approx(1.0)
    assert r.dockq == pytest.approx(1.0, abs=1e-6)


def test_dockq_no_native_interface_raises():
    native = two_chain_complex(lig_origin=(0.0, 100.0, 0.0))
    model = two_chain_complex(lig_origin=(0.0, 100.0, 0.0))
    with pytest.raises(ValueError, match="native interface"):
        compute_dockq(native, model, ["A"], ["B"])


# --------------------------------------------------------------------------
# epitope / paratope
# --------------------------------------------------------------------------

def test_epitope_identity_perfect_scores():
    native = two_chain_complex()
    model = two_chain_complex()
    sites = epitope_metrics(native, model, ["A"], ["B"])
    for site in sites.values():
        assert site.precision == pytest.approx(1.0)
        assert site.recall == pytest.approx(1.0)
        assert site.f1 == pytest.approx(1.0)
    assert len(sites["epitope"].native_residues) > 0


def test_epitope_lost_when_ligand_moves_away():
    native = two_chain_complex()
    model = two_chain_complex(lig_origin=(0.0, 50.0, 0.0))
    sites = epitope_metrics(native, model, ["A"], ["B"])
    assert sites["epitope"].recall == pytest.approx(0.0)
    assert sites["epitope"].f1 == pytest.approx(0.0)


def test_antibody_wrapper_flags_mispose_with_correct_epitope():
    # Ligand slid 4 residues (15.2 A) along the receptor: every ligand
    # (epitope-side) residue is still in contact — so the epitope is
    # recovered — but all native residue PAIRS are lost and the ligand
    # placement is far off -> DockQ incorrect while epitope F1 stays high;
    # the report must carry the diagnostic note.
    native = two_chain_complex(lig_origin=(0.0, 4.0, 0.0))
    model = two_chain_complex(lig_origin=(4 * SPACING, 4.0, 0.0))
    res = evaluate_antibody_complex(native, model, ["A"], ["B"])
    assert res.dockq.capri == "incorrect"
    assert res.epitope.f1 > 0.5
    assert "mis-oriented" in res.report()
    d = res.to_dict()
    assert set(d) == {"dockq", "epitope", "paratope"}


# --------------------------------------------------------------------------
# pDockQ
# --------------------------------------------------------------------------

def test_pdockq_matches_published_sigmoid():
    import math
    st = two_chain_complex(b_iso=90.0)
    r = compute_pdockq(st, ["A"], ["B"])
    assert r.n_contacts > 0
    assert r.mean_interface_plddt == pytest.approx(90.0)
    x = 90.0 * math.log10(r.n_contacts)
    expected = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
    assert r.pdockq == pytest.approx(expected, abs=1e-12)


def test_pdockq_no_contacts_is_zero():
    st = two_chain_complex(lig_origin=(0.0, 100.0, 0.0))
    r = compute_pdockq(st, ["A"], ["B"])
    assert r.pdockq == 0.0
    assert r.n_contacts == 0


def test_pdockq_warns_on_non_plddt_bfactors():
    st = two_chain_complex(b_iso=0.5)  # not pLDDT-like
    with pytest.warns(UserWarning, match="pLDDT"):
        compute_pdockq(st, ["A"], ["B"])
