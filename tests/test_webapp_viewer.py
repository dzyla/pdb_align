import os

import gemmi
from webapp import viewer as V

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_remap_long_chain_ids():
    st = gemmi.read_structure(os.path.join(DATA, "ref.pdb"))
    st[0][0].name = "AAA"
    mapping = V.remap_long_chain_ids(st)
    assert mapping.get("AAA")
    assert all(len(c.name) == 1 for m in st for c in m)


def test_structure_to_pdb_string():
    st = gemmi.read_structure(os.path.join(DATA, "ref.pdb"))
    s = V.structure_to_pdb_string(st)
    assert "ATOM" in s and isinstance(s, str)
