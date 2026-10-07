import glob
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _files():
    yield os.path.join(ROOT, "struct_pair_align.py")
    yield from glob.glob(os.path.join(ROOT, "webapp", "*.py"))


def test_app_does_not_import_core():
    for path in _files():
        with open(path) as f:
            src = f.read()
        assert not re.search(r"from\s+pdb_align\.core\s+import", src), path
        assert not re.search(r"import\s+pdb_align\.core", src), path


def test_the_example_notebook_imports_only_modules_that_exist():
    """The shipped notebook is documentation: it imported
    `pdb_align.structure`, a module that has not existed for several releases,
    so the demo died on its first cell."""
    import importlib.util
    import json

    nb_path = os.path.join(ROOT, "analysis_notebook.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)
    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")

    modules = set(re.findall(r"from\s+(pdb_align[\w.]*)\s+import", src))
    modules |= set(re.findall(r"^\s*import\s+(pdb_align[\w.]*)", src, re.M))
    assert modules, "the notebook should demonstrate the package"
    missing = [m for m in sorted(modules)
               if importlib.util.find_spec(m) is None]
    assert not missing, missing
