"""pdb_align — high-performance protein structure alignment and model assessment."""

__version__ = "0.2.0"

from .aligner import PDBAligner, AlignmentResult, AlignmentFailedError, EnsembleResult, DomainResult, LoadedResult, inspect_structure
from .exceptions import ParsingError, ChainNotFoundError
from .chains import match_chains, align_multichain, ChainMapping
from .plotstyle import apply_nature_style
from .interpretation import AlignmentQuality, FlaggedRegion
from .interface import (
    compute_dockq, DockQResult, dockq_formula, capri_class,
    epitope_metrics, SiteComparison,
    evaluate_antibody_complex, ImmuneComplexResult,
    compute_pdockq, PDockQResult,
)
from .evaluate import evaluate_models, ModelEvaluation

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
