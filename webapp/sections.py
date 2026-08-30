"""Thin Streamlit tab renderers. Data/figures come from webapp.data/figures."""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from webapp import figures as F
from webapp import viewer as V

_BADGE = {"excellent": "🟢", "good": "🔵", "moderate": "🟠", "poor": "🔴"}


def render_header(res):
    q = res.quality
    s = res.summary_stats()
    st.subheader(f"{_BADGE.get(q.band, '⚪')} {q.band.upper()}  ·  {q.verdict}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSD (Å)", f"{s['rmsd']:.3f}" if s["rmsd"] is not None else "—")
    c2.metric("TM-score", f"{s['tm_score']:.3f}" if s["tm_score"] is not None else "—")
    c3.metric("GDT_TS", f"{s['gdt_ts']:.1f}" if s["gdt_ts"] is not None else "—",
              help="Single superposition, normalized by reference length")
    c4.metric("lDDT-Cα", f"{s['lddt_ca']:.3f}" if s.get("lddt_ca") is not None else "—",
              help="Superposition-free, matched residues")
    c5.metric("Coverage",
              f"{s['coverage_pct']:.0f}%" if s["coverage_pct"] is not None else "—")
    st.caption(f"Confidence: {q.confidence}")
    for w in q.warnings:
        st.warning(w)


def render_overview(res):
    s = res.summary_stats()
    st.write(f"**Method:** {s['method']}  ·  **Strategy:** {s['strategy']}")
    st.caption(s.get("reason") or "")
    if s.get("chain_mapping"):
        st.write("**Chain mapping (ref → mob, %id):**")
        st.dataframe(
            [{"ref": m["ref"], "mob": m["mob"], "identity": m["identity"]}
             for m in s["chain_mapping"]],
            hide_index=True, use_container_width=True)
    pc = res.per_chain
    if not pc.empty:
        st.write("**Per-chain RMSD:**")
        st.dataframe(pc, hide_index=True, use_container_width=True)
    fr = res.quality.flagged_regions
    if fr:
        st.write("**Flagged regions:**")
        st.dataframe([r.to_dict() for r in fr], hide_index=True,
                     use_container_width=True)
    else:
        st.info("No flagged high-deviation regions.")


def render_3d(res):
    from stmol import showmol

    color_by = st.selectbox("Colour by", ["rmsd", "plddt", "chain"], index=0)
    aligned = res.aligned_structure(
        color_by="rmsd" if color_by == "rmsd" else "bfactor")
    aligned_str = V.structure_to_pdb_string(aligned)
    ref_str = st.session_state.get("_ref_pdb_str", "")
    try:
        view = V.build_view(ref_str, aligned_str, color_by=color_by)
        showmol(view, height=520, width=760)
    except Exception as e:
        st.error(f"3D viewer unavailable: {e}")
    st.download_button("⬇️ Aligned structure (PDB)", data=aligned_str,
                       file_name="aligned.pdb")


def render_per_residue(res):
    df = res.get_rmsd_df(on="reference")
    st.plotly_chart(F.per_residue_figure(df, res.quality.flagged_regions),
                    use_container_width=True, key="per_res")
    with st.expander("Sequence alignment"):
        aln = res.get_sequence_alignment()
        st.code(aln if isinstance(aln, str) else str(aln))
    with st.expander("Distance matrices & histograms"):
        ref_c, mob_c = res.get_aligned_coords()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(F.distance_matrix_figure(ref_c, "Reference"),
                            use_container_width=True, key="dm_ref")
        with c2:
            st.plotly_chart(F.pair_distance_hist(ref_c, mob_c, "Cα–Cα"),
                            use_container_width=True, key="pdh")


def render_ensemble(ens):
    st.dataframe(ens.summary(), use_container_width=True, hide_index=True)
    st.plotly_chart(F.ensemble_rmsd_heatmap(ens.rmsd_matrix()),
                    use_container_width=True, key="ens_heat")
    if len(ens.results) >= 3:
        n = st.slider("Clusters", 2, len(ens.results), min(3, len(ens.results)))
        try:
            ens.cluster(n_clusters=n)
            st.pyplot(ens.plot_pca(color_by="cluster"))
            st.pyplot(ens.plot_dendrogram())
        except Exception as e:
            st.info(f"Clustering unavailable: {e}")
    else:
        st.info("Add 3+ mobile structures to enable clustering / PCA.")


def render_export(obj, is_ensemble):
    st.write("One-click reproducible bundle:")
    if st.button("📦 Build bundle (ZIP)"):
        out = os.path.join(tempfile.mkdtemp(), "bundle.zip")
        obj.export_bundle(out)
        with open(out, "rb") as f:
            st.download_button("⬇️ Download bundle", data=f.read(),
                               file_name="pdb_align_bundle.zip")
