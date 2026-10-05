"""Rank predicted models against a reference structure (or template).

The daily loop of structure prediction: given one experimental structure (or
a homology template) and N predicted models (AlphaFold/Boltz/...), compute for
each model the fold metrics (TM-score, RMSD, GDT_TS, lDDT-Ca, coverage) and —
when an interface is specified — the DockQ family (fnat/iRMSD/LRMSD/DockQ/
CAPRI class), epitope/paratope diagnostics for immune complexes, and the
reference-free pDockQ when the models carry pLDDT.

Ranking: DockQ (descending) when an interface is scored, otherwise TM-score.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .aligner import PDBAligner, _map_parallel, _resolve_workers
from .confidence import compute_pdockq2, find_confidence_files, load_confidence
from .interface import (
    compute_dockq,
    compute_pdockq,
    epitope_metrics,
)


@dataclass
class ModelEvaluation:
    """Result of :func:`evaluate_models`: a ranked table plus per-model detail."""
    table: pd.DataFrame
    reference: str
    interface_scored: bool
    antibody_mode: bool
    details: List[dict] = field(default_factory=list)

    @property
    def best(self) -> Optional[str]:
        """Label of the top-ranked model (None if nothing scored)."""
        if self.table.empty:
            return None
        return str(self.table.iloc[0]["model"])

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "ranked_by": "dockq" if self.interface_scored else "tm_score",
            "best_model": self.best,
            "models": self.table.to_dict(orient="records"),
        }

    def report(self) -> str:
        lines = ["=" * 72,
                 " pdb_align - model evaluation vs reference",
                 "=" * 72,
                 f" Reference : {os.path.basename(self.reference)}",
                 f" Ranked by : {'DockQ (interface)' if self.interface_scored else 'TM-score'}",
                 "-" * 72]
        if self.table.empty:
            lines.append(" No models could be evaluated.")
            return "\n".join(lines)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            lines.append(self.table.to_string(index=False,
                                              float_format=lambda v: f"{v:.3f}"))
        lines.append("-" * 72)
        best_row = self.table.iloc[0]
        if self.interface_scored and not np.isnan(best_row.get("dockq", np.nan)):
            lines.append(f" Best model: {best_row['model']}  "
                         f"(DockQ {best_row['dockq']:.3f}, {best_row['capri']}; "
                         f"TM {best_row['tm_score']:.3f})")
        else:
            lines.append(f" Best model: {best_row['model']}  "
                         f"(TM {best_row['tm_score']:.3f}, RMSD {best_row['rmsd']:.2f} A)")
        if self.antibody_mode:
            lines.append(" Epitope F1 close to 1 with low DockQ = right epitope, "
                         "mis-oriented pose; low F1 = wrong antigen surface.")
        lines.append(" Refs: DockQ Basu&Wallner'16; pDockQ Bryant'22; "
                     "TM Zhang&Skolnick'04; lDDT Mariani'13.")
        lines.append("=" * 72)
        return "\n".join(lines)


def _evaluate_one_model(payload: dict):
    """Score one model against the reference.

    Module-level and self-contained so it can run unchanged in a worker
    process. Returns ``(row, detail)``: the ranked-table row and the full
    per-model detail. Every metric is attempted independently and a failure is
    recorded in ``detail`` with a NaN left in ``row``, so one unscorable model
    never removes the others from the ranking.

    Errors are returned rather than warned about: a ``warnings.warn`` inside a
    worker process never reaches the parent. :func:`evaluate_models` re-emits
    them once the results are back.
    """
    label = payload["label"]
    model = payload["model"]
    model_idx = payload["index"]
    ref_path = payload["ref_path"]
    receptor_chains = payload["receptor_chains"]
    ligand_chains = payload["ligand_chains"]
    interface_scored = payload["interface_scored"]
    antibody_mode = payload["antibody_mode"]
    mode = payload["mode"]
    atoms = payload["atoms"]
    min_plddt = payload["min_plddt"]
    with_pdockq = payload["with_pdockq"]
    with_cdr = payload["with_cdr"]
    confidence_files = payload["confidence_files"]
    align_kwargs = payload["align_kwargs"]

    aligner = PDBAligner()
    aligner.add_reference(payload["ref_path"])

    row = {"model": label, "rmsd": np.nan, "tm_score": np.nan,
           "gdt_ts": np.nan, "lddt_ca": np.nan, "coverage_pct": np.nan}
    detail = {"model": label, "path": str(model)}
    try:
        aligner.add_mobile(str(model))
        res = aligner.align(mode=mode, atoms=atoms, min_plddt=min_plddt,
                            **align_kwargs)
        s = res.summary_stats()
        for k in ("rmsd", "tm_score", "gdt_ts", "lddt_ca", "coverage_pct"):
            row[k] = s.get(k) if s.get(k) is not None else np.nan
        detail["summary"] = s
        model_path = aligner.mob_file
    except Exception as e:
        detail["error"] = f"failed fold alignment: {e}"
        return row, detail

    # --- confidence ingestion (ipTM / PAE), explicit or auto-discovered ---
    conf = None
    conf_paths = None
    if confidence_files is not None:
        entry = confidence_files[model_idx]
        if entry:
            conf_paths = [entry] if isinstance(entry, (str, os.PathLike)) else list(entry)
    else:
        conf_paths = find_confidence_files(str(model)) or None
    if conf_paths:
        try:
            conf = load_confidence(conf_paths)
            detail["confidence"] = conf.to_dict()
            row["iptm"] = conf.iptm if conf.iptm is not None else np.nan
        except Exception as e:
            warnings.warn(f"Model '{label}': could not parse confidence "
                          f"file(s) {conf_paths}: {e}", UserWarning, stacklevel=2)
            detail["confidence_error"] = str(e)

    if interface_scored:
        row.update({"dockq": np.nan, "fnat": np.nan, "irmsd": np.nan,
                    "lrmsd": np.nan, "capri": ""})
        try:
            dq = compute_dockq(ref_path, model_path,
                               receptor_chains, ligand_chains)
            row.update({"dockq": dq.dockq, "fnat": dq.fnat,
                        "irmsd": dq.irmsd, "lrmsd": dq.lrmsd,
                        "capri": dq.capri})
            detail["dockq"] = dq.to_dict()
        except Exception as e:
            warnings.warn(f"Model '{label}' failed DockQ: {e}",
                          UserWarning, stacklevel=2)
            detail["dockq_error"] = str(e)

        if antibody_mode:
            row["epitope_f1"] = np.nan
            try:
                sites = epitope_metrics(ref_path, model_path,
                                        receptor_chains, ligand_chains)
                row["epitope_f1"] = sites["epitope"].f1
                detail["epitope"] = sites["epitope"].to_dict()
                detail["paratope"] = sites["paratope"].to_dict()
            except Exception as e:
                detail["epitope_error"] = str(e)

        model_dq = detail.get("dockq", {})
        rec_model = [m.split("->")[1] for m in model_dq.get("receptor_mapping", [])]
        lig_model = [m.split("->")[1] for m in model_dq.get("ligand_mapping", [])]

        if with_pdockq:
            row["pdockq"] = np.nan
            try:
                if rec_model and lig_model:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        pq = compute_pdockq(model_path, rec_model, lig_model)
                    row["pdockq"] = pq.pdockq
                    detail["pdockq"] = pq.to_dict()
            except Exception as e:
                detail["pdockq_error"] = str(e)

        # PAE-based pDockQ2 (Zhu 2023) when a PAE matrix was ingested
        if conf is not None and conf.pae is not None:
            row["pdockq2"] = np.nan
            try:
                if rec_model and lig_model:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        pq2 = compute_pdockq2(model_path, rec_model,
                                              lig_model, conf.pae)
                    row["pdockq2"] = pq2.pdockq2
                    detail["pdockq2"] = pq2.to_dict()
            except Exception as e:
                detail["pdockq2_error"] = str(e)

        # per-CDR RMSD after framework superposition (needs ANARCI)
        if antibody_mode and with_cdr:
            row["cdr_h3"] = np.nan
            try:
                from .cdr import cdr_rmsd
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    cdrs = cdr_rmsd(ref_path, model_path, receptor_chains,
                                    model_antibody_chains=rec_model or None)
                detail["cdr"] = cdrs.to_dict()
                if cdrs.h3 is not None:
                    row["cdr_h3"] = cdrs.h3
            except RuntimeError as e:  # ANARCI unavailable/broken
                detail["cdr_error"] = str(e)
            except Exception as e:
                detail["cdr_error"] = str(e)

    return row, detail


def evaluate_models(
    reference: str,
    models: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    receptor_chains: Optional[Sequence[str]] = None,
    ligand_chains: Optional[Sequence[str]] = None,
    antibody_chains: Optional[Sequence[str]] = None,
    antigen_chains: Optional[Sequence[str]] = None,
    mode: str = "auto",
    atoms: str = "CA",
    min_plddt: float = 0.0,
    with_pdockq: bool = True,
    confidence_files: Optional[Sequence] = None,
    with_cdr: bool = True,
    workers: int = 1,
    **align_kwargs,
) -> ModelEvaluation:
    """
    Evaluate and rank N model structures against one reference.

    Parameters
    ----------
    reference, models : paths or remote IDs (``pdb:XXXX``, ``af:UniProtID``).
    receptor_chains, ligand_chains : reference chain names of the two sides
        of an interface to score with DockQ (optional).
    antibody_chains, antigen_chains : immune-complex shorthand — equivalent to
        receptor/ligand but additionally reports epitope/paratope
        precision/recall/F1 (antibody chains are merged as the receptor) and,
        when ANARCI is available and ``with_cdr``, per-CDR RMSD after
        framework superposition (``cdr_h3`` column; full set in ``details``).
    with_pdockq : also compute reference-free pDockQ per model when an
        interface is specified (needs pLDDT in the model B-factor column).
    workers : worker processes to score models with. ``1`` runs in-process,
        ``-1`` uses all cores. Models are independent and each one runs a fold
        alignment plus (optionally) DockQ, epitope, pDockQ and CDR metrics, so
        this is the setting that matters when ranking more than a handful.
    confidence_files : per-model confidence files (AF2/AF3/Boltz JSON, PAE
        npz/npy); each entry may be a path, a list of paths, or None. When the
        argument is omitted entirely, sibling files are auto-discovered via
        :func:`pdb_align.confidence.find_confidence_files`. Adds an ``iptm``
        column, and — with a PAE matrix and an interface — ``pdockq2``.
    Other keyword arguments are forwarded to :meth:`PDBAligner.align`.

    Notes
    -----
    A failing model is skipped with a ``UserWarning`` and appears in the
    table with NaN metrics, never silently dropped.
    """
    antibody_mode = antibody_chains is not None
    if antibody_mode:
        if receptor_chains is not None or ligand_chains is not None:
            raise ValueError("Give either receptor/ligand chains or "
                             "antibody/antigen chains, not both.")
        if antigen_chains is None:
            raise ValueError("antibody_chains requires antigen_chains.")
        receptor_chains, ligand_chains = antibody_chains, antigen_chains
    if (receptor_chains is None) != (ligand_chains is None):
        raise ValueError("receptor_chains and ligand_chains must be given together.")
    interface_scored = receptor_chains is not None

    if labels is None:
        labels = [os.path.splitext(os.path.basename(str(m)))[0] for m in models]
    if len(labels) != len(models):
        raise ValueError("labels must match models in length.")
    if confidence_files is not None and len(confidence_files) != len(models):
        raise ValueError("confidence_files must match models in length.")

    # Resolve the reference once (this also downloads a remote ID) so every
    # worker parses a local file instead of re-fetching it.
    ref_path = PDBAligner()._fetch_structure(str(reference))

    rows: List[dict] = []
    details: List[dict] = []
    payload_base = {
        "ref_path": ref_path,
        "receptor_chains": list(receptor_chains) if receptor_chains else None,
        "ligand_chains": list(ligand_chains) if ligand_chains else None,
        "interface_scored": interface_scored, "antibody_mode": antibody_mode,
        "mode": mode, "atoms": atoms, "min_plddt": min_plddt,
        "with_pdockq": with_pdockq, "with_cdr": with_cdr,
        "align_kwargs": dict(align_kwargs),
        "confidence_files": (list(confidence_files)
                             if confidence_files is not None else None),
    }
    payloads = [dict(payload_base, label=str(label), model=str(model), index=i)
                for i, (label, model) in enumerate(zip(labels, models))]

    rows: List[dict] = []
    details: List[dict] = []
    n_workers = _resolve_workers(workers, len(payloads))
    if n_workers > 1:
        outcomes = _map_parallel(_evaluate_one_model, payloads, n_workers,
                                 "evaluate_models")
    else:
        outcomes = [_evaluate_one_model(p) for p in payloads]
    for row, detail in outcomes:
        rows.append(row)
        details.append(detail)

    # Re-emit per-model problems in the parent: a model that could not be
    # scored must be visible, and a warning raised in a worker is lost.
    for detail in details:
        for key in ("error", "confidence_error", "dockq_error", "epitope_error",
                    "pdockq_error", "pdockq2_error", "cdr_error"):
            if detail.get(key):
                warnings.warn(f"Model {detail['model']!r}: {detail[key]}",
                              UserWarning, stacklevel=2)


    table = pd.DataFrame(rows)
    sort_key = "dockq" if (interface_scored and "dockq" in table.columns) else "tm_score"
    if not table.empty and sort_key in table.columns:
        table = table.sort_values(sort_key, ascending=False,
                                  na_position="last").reset_index(drop=True)
    return ModelEvaluation(table=table, reference=str(reference),
                           interface_scored=interface_scored,
                           antibody_mode=antibody_mode, details=details)
