import os

import plotly.graph_objects as go
from pdb_align import PDBAligner
from webapp import figures as F

DATA = os.path.join(os.path.dirname(__file__), "data")


def _result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")


def test_per_residue_figure():
    r = _result()
    fig = F.per_residue_figure(r.get_rmsd_df(), r.quality.flagged_regions)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_distance_matrix_figure():
    r = _result()
    ref_c, mob_c = r.get_aligned_coords()
    fig = F.distance_matrix_figure(ref_c, "Reference")
    assert isinstance(fig, go.Figure)


def test_pair_distance_hist():
    r = _result()
    ref_c, mob_c = r.get_aligned_coords()
    fig = F.pair_distance_hist(ref_c, mob_c, "Pairs")
    assert isinstance(fig, go.Figure)


def test_ensemble_rmsd_heatmap():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    ens = al.align_ensemble([os.path.join(DATA, "mob.pdb"),
                             os.path.join(DATA, "ref.pdb")], mode="auto")
    fig = F.ensemble_rmsd_heatmap(ens.rmsd_matrix())
    assert isinstance(fig, go.Figure)
