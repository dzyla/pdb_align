"""pdb_align — high-performance protein structure alignment and model assessment."""

__version__ = "0.4.0"

from .aligner import (
    AlignmentFailedError,
    AlignmentResult,
    DomainResult,
    EnsembleResult,
    LoadedResult,
    PDBAligner,
    inspect_structure,
)
from .cdr import CDRAnnotation, CDRResult, annotate_cdrs, cdr_rmsd
from .chains import ChainMapping, align_multichain, match_chains
from .confidence import (
    ModelConfidence,
    PDockQ2Result,
    compute_pdockq2,
    find_confidence_files,
    load_confidence,
)
from .evaluate import ModelEvaluation, evaluate_models
from .exceptions import ChainNotFoundError, ParsingError
from .interface import (
    DockQResult,
    ImmuneComplexResult,
    PDockQResult,
    SiteComparison,
    capri_class,
    compute_dockq,
    compute_pdockq,
    dockq_formula,
    epitope_metrics,
    evaluate_antibody_complex,
)
from .interpretation import AlignmentQuality, FlaggedRegion
from .plotstyle import apply_nature_style

__all__ = [
    "align",
    "PDBAligner",
    "AlignmentResult",
    "AlignmentFailedError",
    "EnsembleResult",
    "DomainResult",
    "LoadedResult",
    "AlignmentQuality",
    "FlaggedRegion",
    "inspect_structure",
    "ParsingError",
    "ChainNotFoundError",
    "match_chains",
    "align_multichain",
    "ChainMapping",
    "apply_nature_style",
    # interface / model-assessment layer
    "compute_dockq",
    "DockQResult",
    "dockq_formula",
    "capri_class",
    "epitope_metrics",
    "SiteComparison",
    "evaluate_antibody_complex",
    "ImmuneComplexResult",
    "compute_pdockq",
    "PDockQResult",
    "evaluate_models",
    "ModelEvaluation",
    # confidence ingestion (PAE / ipTM) and CDR metrics
    "load_confidence",
    "find_confidence_files",
    "ModelConfidence",
    "compute_pdockq2",
    "PDockQ2Result",
    "annotate_cdrs",
    "cdr_rmsd",
    "CDRResult",
    "CDRAnnotation",
]


def align(
    ref: str,
    mob: str,
    chains_ref=None,
    chains_mob=None,
    verbose: bool = False,
    **kwargs,
) -> "AlignmentResult":
    """
    One-liner structural alignment.

    Parameters
    ----------
    ref : str
        Reference structure path or remote ID (``pdb:XXXX`` / ``af:UniProtID``).
    mob : str
        Mobile structure path or remote ID.
    chains_ref : list[str] | None
        Chains to use from the reference (e.g. ``["A"]``). ``None`` uses all chains.
    chains_mob : list[str] | None
        Chains to use from the mobile structure. ``None`` uses all chains.
    verbose : bool
        If ``True``, enable verbose logging in the underlying :class:`PDBAligner`.
        Default is ``False``.
    **kwargs
        Forwarded to :meth:`PDBAligner.align` (e.g. ``mode``, ``atoms``,
        ``seq_gap_open``, ``min_plddt``).

    Returns
    -------
    AlignmentResult
    """
    aligner = PDBAligner(verbose=verbose)
    aligner.add_reference(ref, chains=chains_ref)
    aligner.add_mobile(mob, chains=chains_mob)
    return aligner.align(**kwargs)
