"""Structure Alignment Suite — Streamlit front-end for pdb_align.

Run: streamlit run struct_pair_align.py
Built entirely on the public pdb_align API (no pdb_align.core).
"""
from __future__ import annotations

import streamlit as st

import pdb_align  # noqa: F401  (public API surface / version)
from webapp import data as D
from webapp import sections as S

st.set_page_config(page_title="Structure Alignment Suite", page_icon="🧬",
                   layout="wide")
st.title("🧬 Structure Alignment Suite")

if "uploads" not in st.session_state:
    st.session_state.uploads = {}      # name -> temp path (or remote ID)
if "results" not in st.session_state:
    st.session_state.results = {}      # input_key -> (kind, result/ensemble)

with st.sidebar:
    st.header("📤 Structures")
    files = st.file_uploader("Upload PDB/mmCIF", type=["pdb", "cif", "mmcif"],
                             accept_multiple_files=True)
    fetch = st.text_input("…or fetch (pdb:1ABC, af:P00533)")
    if st.button("Fetch") and fetch:
        for tid in [t.strip() for t in fetch.split(",") if t.strip()]:
            st.session_state.uploads[tid] = tid  # the API resolves IDs directly
    if files:
        for f in files:
            if f.name not in st.session_state.uploads:
                st.session_state.uploads[f.name] = D.save_upload_to_temp(
                    f.name, f.getvalue())

    names = list(st.session_state.uploads.keys())
    if len(names) < 2:
        st.info("Add at least two structures.")
        st.stop()

    ref_name = st.selectbox("Reference", names)
    mob_names = st.multiselect(
        "Mobile(s)", [n for n in names if n != ref_name],
        default=[n for n in names if n != ref_name][:1])
    ref_path = st.session_state.uploads[ref_name]
    try:
        ref_chain_opts = list(D.list_chains(ref_path).keys())
    except Exception as e:
        st.error(f"Could not read {ref_name}: {e}")
        st.stop()
    ref_chains = st.multiselect("Reference chains", ref_chain_opts,
                                default=ref_chain_opts) or None

    mode_label = st.selectbox(
        "Mode",
        ["auto", "seq_guided", "seq_free_shape", "seq_free_window", "flexible"])
    strategy = st.selectbox("Strategy", ["auto", "global", "local"])

    with st.expander("🧪 Model evaluation (interface)"):
        st.caption("Rank the mobile structure(s) as models of the reference; "
                   "pick the two sides of an interface for DockQ scoring.")
        eval_on = st.checkbox("Evaluate models vs reference", value=False)
        eval_receptor = st.multiselect(
            "Receptor / antibody chains (reference)", ref_chain_opts, default=[])
        eval_ligand = st.multiselect(
            "Ligand / antigen chains (reference)",
            [c for c in ref_chain_opts if c not in eval_receptor], default=[])
        eval_antibody = st.checkbox(
            "Antibody mode (epitope F1 + CDR-H3 RMSD)", value=False)
    with st.expander("Advanced"):
        opts = dict(
            seq_gap_open=st.slider("Gap open", -20, -1, -10),
            seq_gap_extend=st.slider("Gap extend", -20.0, -0.1, -0.5, 0.1),
            atoms=st.selectbox("Atoms", ["CA", "backbone", "all_heavy"]),
            min_plddt=st.number_input("Min pLDDT", 0.0, 100.0, 0.0),
        )
    run = st.button("🚀 Run", use_container_width=True)

if not mob_names:
    st.info("Select at least one mobile structure.")
    st.stop()

is_ensemble = len(mob_names) > 1
mob_paths = [st.session_state.uploads[n] for n in mob_names]

if run:
    eval_cfg = dict(on=eval_on, receptor=eval_receptor, ligand=eval_ligand,
                    antibody=eval_antibody)
    key = D.input_key(ref_path, mob_paths, ref_chains, eval_cfg,
                      mode_label, strategy, opts)
    if key not in st.session_state.results:
        with st.spinner("Aligning…"):
            try:
                if is_ensemble:
                    st.session_state.results[key] = (
                        "ensemble",
                        D.run_ensemble(ref_path, mob_paths, ref_chains, {},
                                       mode_label, strategy, opts))
                else:
                    res = D.run_pairwise(ref_path, mob_paths[0], ref_chains, None,
                                         mode_label, strategy, opts)
                    st.session_state.results[key] = ("pairwise", res)
                    # stash the reference PDB text for the 3D view (public API only)
                    st.session_state["_ref_pdb_str"] = (
                        open(ref_path).read()
                        if str(ref_path).lower().endswith(".pdb") else "")
            except Exception as e:
                st.error(f"Alignment failed: {e}")
                st.stop()
    if eval_on:
        with st.spinner("Evaluating models…"):
            try:
                st.session_state["_eval_" + key] = D.run_evaluation(
                    ref_path, mob_paths, mob_names,
                    eval_receptor or None, eval_ligand or None,
                    eval_antibody, mode_label, opts)
            except Exception as e:
                st.error(f"Model evaluation failed: {e}")
    st.session_state["_last_key"] = key

key = st.session_state.get("_last_key")
if not key or key not in st.session_state.results:
    st.info("Configure inputs in the sidebar and press Run.")
    st.stop()

kind, obj = st.session_state.results[key]
evaluation = st.session_state.get("_eval_" + key)

if kind == "ensemble":
    best = min(obj.results, key=lambda r: r.rmsd if r.rmsd is not None else 1e9)
    S.render_header(best)
    names = ["Overview", "3D", "Per-residue", "Ensemble"]
    names += ["Evaluation"] if evaluation is not None else []
    names += ["Export"]
    tabs = st.tabs(names)
    with tabs[0]:
        S.render_overview(best)
    with tabs[1]:
        S.render_3d(best)
    with tabs[2]:
        S.render_per_residue(best)
    with tabs[3]:
        S.render_ensemble(obj)
    if evaluation is not None:
        with tabs[4]:
            S.render_evaluation(evaluation)
    with tabs[-1]:
        S.render_export(obj, is_ensemble=True)
else:
    S.render_header(obj)
    names = ["Overview", "3D", "Per-residue"]
    names += ["Evaluation"] if evaluation is not None else []
    names += ["Export"]
    tabs = st.tabs(names)
    with tabs[0]:
        S.render_overview(obj)
    with tabs[1]:
        S.render_3d(obj)
    with tabs[2]:
        S.render_per_residue(obj)
    if evaluation is not None:
        with tabs[3]:
            S.render_evaluation(evaluation)
    with tabs[-1]:
        S.render_export(obj, is_ensemble=False)
