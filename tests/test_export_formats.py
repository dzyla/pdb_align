"""An export must not be defeated by the limits of the PDB format.

A chain name longer than one character cannot be written in the PDB format at
all — gemmi raises ``chain name too long`` — and such names are routine in
large assemblies and in the mmCIF files ``pdb:XXXX`` downloads from the RCSB.
Every export path therefore falls back to mmCIF, which keeps the chain names
the rest of the output (the RMSD table, the viewer scripts) refers to, and
says which file it actually wrote.
"""
import os
import zipfile

import gemmi
import pytest
from _synthetic import make_chain, make_structure

import pdb_align


def _long_chain_cif(path, name="AAA", n_res=30):
    st = make_structure([make_chain(name, n_res, (0.0, 0.0, 0.0))])
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))
    return str(path)


def _short_chain_pdb(path, name="A", n_res=30, origin=(0.0, 0.0, 0.0)):
    st = make_structure([make_chain(name, n_res, origin)])
    st.setup_entities()
    st.write_pdb(str(path))
    return str(path)


def test_save_aligned_pdb_falls_back_to_mmcif(tmp_path):
    src = _long_chain_cif(tmp_path / "long.cif")
    res = pdb_align.align(src, src)

    with pytest.warns(UserWarning, match="PDB format"):
        written = res.save_aligned_pdb(str(tmp_path / "aligned.pdb"))

    assert written.endswith(".cif")
    assert os.path.exists(written)
    # The fallback keeps the chain name; truncating or renaming it would
    # desynchronise the structure from the per-residue table and the scripts.
    assert [ch.name for ch in gemmi.read_structure(written)[0]] == ["AAA"]


def test_save_aligned_pdb_returns_the_path_it_wrote(tmp_path):
    src = _short_chain_pdb(tmp_path / "a.pdb")
    res = pdb_align.align(src, src)
    out = tmp_path / "aligned.pdb"

    written = res.save_aligned_pdb(str(out))

    assert written == str(out)
    assert out.exists()


def test_export_bundle_survives_long_chain_names(tmp_path):
    src = _long_chain_cif(tmp_path / "long.cif")
    res = pdb_align.align(src, src)

    with pytest.warns(UserWarning, match="PDB format"):
        out = res.export_bundle(str(tmp_path / "bundle.zip"))

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        pml = z.read("view.pml").decode()
        cxc = z.read("view.cxc").decode()
    assert "aligned.cif" in names and "reference.cif" in names
    # The scripts must point at the files the bundle really contains.
    assert "aligned.cif" in pml and "reference.cif" in pml
    assert "aligned.cif" in cxc and "reference.cif" in cxc


def test_subset_only_writes_just_the_aligned_residues(tmp_path):
    """``subset_only=True`` was accepted and ignored: the full mobile chain was
    written whatever the caller asked for."""
    ref = _short_chain_pdb(tmp_path / "ref.pdb", n_res=20)
    mob = _short_chain_pdb(tmp_path / "mob.pdb", n_res=30)
    res = pdb_align.align(ref, mob, mode="seq_guided")
    n_aligned = res.summary_stats()["n_aligned"]
    assert n_aligned == 20

    full = tmp_path / "full.pdb"
    subset = tmp_path / "subset.pdb"
    res.save_aligned_pdb(str(full))
    res.save_aligned_pdb(str(subset), subset_only=True)

    def n_residues(p):
        return sum(len(ch) for ch in gemmi.read_structure(str(p))[0])

    assert n_residues(full) == 30
    assert n_residues(subset) == n_aligned


def test_viewer_scripts_have_one_implementation(tmp_path):
    """There were two script generators: the one the bundle uses, and an older
    public pair that drifted apart from it — a different colour scale and one
    `alter` line per residue re-injecting a B-factor column the aligned file
    already carries (tens of thousands of lines on a complex). The public
    methods must emit the same scripts the bundle ships.
    """
    src = _short_chain_pdb(tmp_path / "a.pdb")
    res = pdb_align.align(src, src)
    pml, cxc = tmp_path / "v.pml", tmp_path / "v.cxc"

    res.save_pymol_script(str(pml), "aligned.pdb")
    res.save_chimerax_script(str(cxc), "aligned.pdb")

    pml_text, cxc_text = pml.read_text(), cxc.read_text()
    assert "alter" not in pml_text and "setattr" not in cxc_text
    assert "spectrum b, blue_white_red, mob, minimum=0, maximum=5" in pml_text
    assert "palette blue:white:red range 0,5" in cxc_text
    # Both scripts must load the reference and the aligned mobile.
    assert pml_text.count("load ") == 2 and "aligned.pdb" in pml_text
    assert cxc_text.count("open ") == 2 and "aligned.pdb" in cxc_text
    assert os.path.basename(res.ref_file) in pml_text
