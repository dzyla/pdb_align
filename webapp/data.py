"""IO, caching, and alignment wrappers for the Streamlit app.

Only the public pdb_align API is used here — never pdb_align.core.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

from pdb_align import PDBAligner, evaluate_models, inspect_structure


def save_upload_to_temp(name: str, data: bytes) -> str:
    suffix = os.path.splitext(name)[1].lower() or ".pdb"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="pdb_align_up_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def list_chains(path: str) -> dict:
    return inspect_structure(path)["chains"]


def _apply_opts(kwargs: dict, opts: dict) -> dict:
    kwargs.update(
        seq_gap_open=opts.get("seq_gap_open", -10),
        seq_gap_extend=opts.get("seq_gap_extend", -0.5),
        atoms=opts.get("atoms", "CA"),
        min_plddt=opts.get("min_plddt", 0.0),
    )
    return kwargs


def run_pairwise(ref_path, mob_path, ref_chains, mob_chains, mode, strategy, opts):
    al = PDBAligner()
    al.add_reference(ref_path, chains=ref_chains)
    al.add_mobile(mob_path, chains=mob_chains)
    return al.align(**_apply_opts(dict(mode=mode, strategy=strategy), opts))


def run_ensemble(ref_path, mob_paths, ref_chains, mob_chains_map, mode, strategy, opts):
    al = PDBAligner()
    al.add_reference(ref_path, chains=ref_chains)
    return al.align_ensemble(
        mob_list=list(mob_paths),
        **_apply_opts(dict(mode=mode), opts),
    )


def run_evaluation(ref_path, mob_paths, labels, receptor_chains, ligand_chains,
                   antibody_mode, mode, opts):
    """Rank the mobile structures against the reference (public API only).

    ``receptor_chains``/``ligand_chains`` may be None (fold metrics only).
    In antibody mode they are passed as antibody/antigen groups, adding
    epitope F1 and CDR columns.
    """
    kwargs = _apply_opts(dict(mode=mode), opts)
    if antibody_mode and receptor_chains and ligand_chains:
        return evaluate_models(ref_path, list(mob_paths), labels=list(labels),
                               antibody_chains=receptor_chains,
                               antigen_chains=ligand_chains, **kwargs)
    return evaluate_models(ref_path, list(mob_paths), labels=list(labels),
                           receptor_chains=receptor_chains,
                           ligand_chains=ligand_chains, **kwargs)


def input_key(ref, mobs, ref_chains, mob_chains_map, mode, strategy, opts) -> str:
    payload = json.dumps(dict(
        ref=ref, mobs=sorted(mobs), ref_chains=ref_chains,
        mob_chains=mob_chains_map, mode=mode, strategy=strategy, opts=opts,
    ), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()
