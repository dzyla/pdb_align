import os
import re

import pytest

GUI = os.path.join(os.path.dirname(__file__), "..", "struct_pair_align.py")


@pytest.mark.xfail(reason="GUI still imports pdb_align.core; converged in SP2",
                   strict=False)
def test_gui_does_not_import_core():
    with open(GUI) as f:
        src = f.read()
    assert not re.search(r"from\s+pdb_align\.core\s+import", src)
    assert not re.search(r"import\s+pdb_align\.core", src)
