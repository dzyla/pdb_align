import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdb_align import plotstyle


def test_palette_is_colorblind_safe_hexes():
    assert len(plotstyle.PALETTE) >= 8
    assert all(c.startswith("#") and len(c) == 7 for c in plotstyle.PALETTE)


def test_apply_nature_style_sets_rcparams():
    with plotstyle.apply_nature_style():
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["xtick.direction"] == "out"


def test_nature_figure_single_column_width_mm():
    fig, ax = plotstyle.nature_figure(width="single")
    w_in, _ = fig.get_size_inches()
    assert abs(w_in - 89 / 25.4) < 0.05
    plt.close(fig)


def test_panel_label_adds_text():
    fig, ax = plotstyle.nature_figure()
    plotstyle.panel_label(ax, "a")
    assert any(t.get_text() == "a" for t in ax.texts)
    plt.close(fig)


def test_plot_summary_returns_figure(tmp_path):
    import pdb_align
    r = pdb_align.align("tests/data/ref.pdb", "tests/data/mob.pdb")
    out = tmp_path / "summary.png"
    fig = r.plot_summary(filename=str(out))
    assert fig is not None
    assert out.exists()
    plt.close(fig)


def test_plot_summary_bar_chart_branch(tmp_path):
    import pandas as pd

    import pdb_align
    r = pdb_align.align("tests/data/ref.pdb", "tests/data/mob.pdb")
    r._per_chain = pd.DataFrame(
        [{"chain_ref": "A", "chain_mob": "A", "n_residues": 40, "rmsd": 0.68}],
        columns=["chain_ref", "chain_mob", "n_residues", "rmsd"])
    out = tmp_path / "bar.png"
    fig = r.plot_summary(filename=str(out))
    assert fig is not None
    assert out.exists()
    plt.close(fig)
