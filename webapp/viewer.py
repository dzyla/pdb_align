"""Py3Dmol superposition view built from AlignmentResult.aligned_structure()."""
from __future__ import annotations

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def remap_long_chain_ids(struct) -> dict:
    """Rename chains whose gemmi name is >1 char to unique single chars (PDB limit)."""
    used = {c.name for m in struct for c in m if len(c.name) == 1}
    replacements = {}
    for model in struct:
        for chain in model:
            if len(chain.name) > 1:
                if chain.name not in replacements:
                    for c in _ALPHABET:
                        if c not in used:
                            replacements[chain.name] = c
                            used.add(c)
                            break
                chain.name = replacements.get(chain.name, chain.name[0])
    return replacements


def structure_to_pdb_string(struct) -> str:
    remap_long_chain_ids(struct)
    try:
        return struct.make_pdb_string()
    except Exception:
        import os
        import tempfile
        fd, p = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        struct.write_pdb(p)
        with open(p) as f:
            s = f.read()
        os.unlink(p)
        return s


def build_view(ref_pdb_str, aligned_pdb_str, color_by="rmsd"):
    """Ref as grey cartoon; mobile as cartoon coloured by B-factor (RMSD) or chain."""
    import py3Dmol
    view = py3Dmol.view(width=760, height=520)
    if ref_pdb_str:
        view.addModel(ref_pdb_str, "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": "lightgrey"}})
    view.addModel(aligned_pdb_str, "pdb")
    mob_model = 1 if ref_pdb_str else 0
    if color_by in ("rmsd", "plddt", "bfactor"):
        view.setStyle({"model": mob_model}, {"cartoon": {"colorscheme":
            {"prop": "b", "gradient": "roygb", "min": 0, "max": 5}}})
    else:  # by chain
        view.setStyle({"model": mob_model}, {"cartoon": {"colorscheme": "chainHetatm"}})
    view.zoomTo()
    return view
