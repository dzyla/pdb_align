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
