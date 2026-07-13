def test_webapp_modules_import():
    import webapp.data  # noqa: F401
    import webapp.figures  # noqa: F401
    import webapp.viewer  # noqa: F401
    import webapp.sections  # noqa: F401

    for fn in ("render_header", "render_overview", "render_3d",
               "render_per_residue", "render_ensemble", "render_export"):
        assert hasattr(webapp.sections, fn)
