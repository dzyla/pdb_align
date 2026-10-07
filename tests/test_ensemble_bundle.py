import os
import zipfile

from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")


def _ensemble():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    return al.align_ensemble([os.path.join(DATA, "mob.pdb"),
                              os.path.join(DATA, "ref.pdb")], mode="auto")


def test_ensemble_bundle_contains_tables(tmp_path):
    out = _ensemble().export_bundle(str(tmp_path / "ens.zip"))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.endswith("summary.csv") for n in names)
        assert any(n.endswith("rmsd_matrix.csv") for n in names)


def test_pca_point_labels_are_model_names_not_paths(tmp_path):
    """`align_ensemble` labels each model with the path it was given, and the
    PCA annotated every point with that full path — unreadable the moment the
    models live anywhere but the working directory. The table keeps the full
    path; the figure shows the name."""
    import matplotlib
    matplotlib.use("Agg")
    ens = _ensemble()

    fig = ens.plot_pca(color_by="rmsd")

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts, "every model should be labelled"
    assert all(os.sep not in t for t in texts), texts
    assert "mob.pdb" in texts
    # the summary table must still carry the full path
    assert any(os.sep in str(m) for m in ens.summary()["model"])


def test_dendrogram_leaf_labels_are_model_names_not_paths():
    import matplotlib
    matplotlib.use("Agg")
    ens = _ensemble()

    fig = ens.plot_dendrogram()

    leaves = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert leaves and all(os.sep not in t for t in leaves), leaves
