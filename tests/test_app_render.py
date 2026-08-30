"""End-to-end render test for the Streamlit app using Streamlit's AppTest.

Runs the actual section renderers (header, overview, 3D, per-residue, export)
against a real AlignmentResult — no browser required. Skipped when the optional
``[app]`` dependencies (streamlit, stmol, py3Dmol, plotly) are not installed.
"""
import os

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("stmol")
pytest.importorskip("py3Dmol")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")

_SCRIPT = f"""
import streamlit as st
from webapp import data as D
from webapp import sections as S

REF = {os.path.join(DATA, "ref.pdb")!r}
MOB = {os.path.join(DATA, "mob.pdb")!r}
OPTS = {{"seq_gap_open": -10, "seq_gap_extend": -0.5, "atoms": "CA", "min_plddt": 0.0}}

res = D.run_pairwise(REF, MOB, None, None, "auto", "auto", OPTS)
st.session_state["_ref_pdb_str"] = open(REF).read()

S.render_header(res)
tabs = st.tabs(["Overview", "3D", "Per-residue", "Export"])
with tabs[0]:
    S.render_overview(res)
with tabs[1]:
    S.render_3d(res)
with tabs[2]:
    S.render_per_residue(res)
with tabs[3]:
    S.render_export(res, is_ensemble=False)
"""


_ENSEMBLE_SCRIPT = f"""
import streamlit as st
from webapp import data as D
from webapp import sections as S

REF = {os.path.join(DATA, "ref.pdb")!r}
MOB = {os.path.join(DATA, "mob.pdb")!r}
OPTS = {{"seq_gap_open": -10, "seq_gap_extend": -0.5, "atoms": "CA", "min_plddt": 0.0}}

ens = D.run_ensemble(REF, [MOB, REF], None, {{}}, "auto", "auto", OPTS)
best = min(ens.results, key=lambda r: r.rmsd if r.rmsd is not None else 1e9)
st.session_state["_ref_pdb_str"] = open(REF).read()
S.render_header(best)
S.render_ensemble(ens)
S.render_export(ens, is_ensemble=True)
"""


def test_app_renders_verdict_and_all_tabs():
    at = AppTest.from_string(_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    blob = " ".join(str(el.value) for el in at.subheader)
    assert any(b in blob for b in ("EXCELLENT", "GOOD", "MODERATE", "POOR"))
    assert len(at.metric) == 5  # RMSD, TM, GDT_TS, lDDT-Ca, coverage


def test_app_renders_ensemble_without_crash():
    # Guards the 2-model cluster-slider edge case (min == max).
    at = AppTest.from_string(_ENSEMBLE_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
