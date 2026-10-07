#!/usr/bin/env python
"""Regenerate the figures used in README.md.

Runs the real library on a reproducible pair built from the trimmed 4HHB in
``tests/data``: the native, and a copy whose C+D chains are rotated as one
rigid body — a dimer-of-dimers motion, which is what haemoglobin actually
does. The app screenshots next to these are captures of the Streamlit front
end (``streamlit run struct_pair_align.py``) on the same two files.

    python docs/make_figures.py
"""
from __future__ import annotations

import math
import os
import tempfile

import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")

import pdb_align

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HHB = os.path.join(ROOT, "tests", "data", "4hhb_bb.pdb")
OUT = os.path.join(ROOT, "docs", "images")


def rotated(degrees: float, path: str, chains=("C", "D")) -> str:
    """4HHB with *chains* rotated about z as one rigid body."""
    st = gemmi.read_structure(HHB)
    st.setup_entities()
    a = math.radians(degrees)
    R = np.array([[math.cos(a), -math.sin(a), 0.0],
                  [math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    for chain in st[0]:
        if chain.name in chains:
            for res in chain:
                for atom in res:
                    atom.pos = gemmi.Position(*(R @ np.array(atom.pos.tolist())))
    st.write_pdb(path)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pdb_align_figs_")
    native = rotated(0.0, os.path.join(work, "hemoglobin_native.pdb"))
    model = rotated(25.0, os.path.join(work, "hemoglobin_model.pdb"))

    res = pdb_align.align(native, model)
    print(res.report())
    res.plot_summary(os.path.join(OUT, "summary.png"))
    res.plot_rmsd(filename=os.path.join(OUT, "per_residue.png"))

    aligner = pdb_align.PDBAligner()
    aligner.add_reference(native)
    models = [rotated(d, os.path.join(work, f"model_{d:02.0f}deg.pdb"))
              for d in (2, 4, 6, 12, 16, 20)]
    ens = aligner.align_ensemble(models)
    print(ens.summary().to_string(index=False))
    fig = ens.plot_pca(color_by="rmsd")
    fig.savefig(os.path.join(OUT, "ensemble_pca.png"), dpi=300,
                bbox_inches="tight")
    print("wrote:", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
