import pytest

pytest.importorskip("plotly", reason="the [app] extras are not installed")
pytest.importorskip("streamlit", reason="the [app] extras are not installed")


def test_webapp_modules_import():
    import webapp.data
    import webapp.figures
    import webapp.sections
    import webapp.viewer

    for fn in ("render_header", "render_overview", "render_3d",
               "render_per_residue", "render_ensemble", "render_export"):
        assert hasattr(webapp.sections, fn)
