import os

import pdb_align

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_inspect_structure_lists_chains_and_sequences():
    info = pdb_align.inspect_structure(os.path.join(DATA, "ref.pdb"))
    assert "chains" in info and "sequences" in info
    assert len(info["chains"]) >= 1
    chain = next(iter(info["chains"]))
    assert info["chains"][chain] > 0
    assert isinstance(info["sequences"][chain], str)
    assert len(info["sequences"][chain]) == info["chains"][chain]
