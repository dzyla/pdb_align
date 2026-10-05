import os

import pytest

from pdb_align import PDBAligner

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def result():
    al = PDBAligner()
    al.add_reference(os.path.join(DATA, "ref.pdb"))
    al.add_mobile(os.path.join(DATA, "mob.pdb"))
    return al.align(mode="auto")


def test_aligned_structure_bfactor_matches_rmsd_df(result):
    struct = result.aligned_structure(color_by="rmsd")
    df = result.get_rmsd_df(on="mobile")
    want = {row["Residue"]: row["RMSD"] for _, row in df.iterrows()}
    seen = 0
    for model in struct:
        for chain in model:
            for res in chain:
                label = f"{chain.name}:{res.seqid.num}"
                if label in want:
                    b = res[0].b_iso
                    assert b == pytest.approx(want[label], abs=1e-3)
                    seen += 1
    assert seen > 0


def test_aligned_structure_preserve_bfactor(result):
    struct = result.aligned_structure(color_by="bfactor")
    bvals = [a.b_iso for m in struct for c in m for r in c for a in r]
    assert any(b != 0.0 for b in bvals)
