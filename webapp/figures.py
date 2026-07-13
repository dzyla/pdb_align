"""Pure Plotly figure builders for the Streamlit app (no st.* calls)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def per_residue_figure(rmsd_df, flagged_regions=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(rmsd_df))), y=rmsd_df["RMSD"], mode="lines+markers",
        text=rmsd_df["Residue"], name="Cα RMSD",
        hovertemplate="%{text}<br>%{y:.2f} Å<extra></extra>"))
    labels = list(rmsd_df["Residue"])
    for fr in (flagged_regions or []):
        try:
            x0 = labels.index(fr.start_label)
            x1 = labels.index(fr.end_label)
        except ValueError:
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.12, line_width=0)
    fig.update_layout(xaxis_title="Residue index", yaxis_title="RMSD (Å)",
                      margin=dict(l=40, r=10, t=30, b=40), height=360)
    return fig


def distance_matrix_figure(coords, title) -> go.Figure:
    c = np.asarray(coords, dtype=float)
    dm = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    fig = go.Figure(go.Heatmap(z=dm, colorscale="Viridis",
                               colorbar=dict(title="Å")))
    fig.update_layout(title=title, height=420,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def pair_distance_hist(ref_coords, mob_coords, title) -> go.Figure:
    a = np.asarray(ref_coords, dtype=float)
    b = np.asarray(mob_coords, dtype=float)
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=-1)
    fig = go.Figure(go.Histogram(x=d, nbinsx=30))
    fig.update_layout(title=title, xaxis_title="Cα–Cα distance (Å)",
                      yaxis_title="count", height=320,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def ensemble_rmsd_heatmap(matrix_df) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix_df.values, x=list(matrix_df.columns), y=list(matrix_df.index),
        colorscale="Viridis", colorbar=dict(title="RMSD (Å)")))
    fig.update_layout(height=460, margin=dict(l=60, r=10, t=30, b=60))
    return fig
