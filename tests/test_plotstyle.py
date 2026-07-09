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
