"""Plain-language interpretation of an alignment result.

Pure functions of numbers already produced by the aligner: no gemmi, no I/O.
Thresholds are module constants so they are reviewable and tunable in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

TM_EXCELLENT = 0.9
TM_GOOD = 0.5
TM_MODERATE = 0.3
RMSD_EXCELLENT = 1.0
RMSD_GOOD = 2.5
RMSD_MODERATE = 5.0
RMSD_FLAG_ABS = 2.0
RMSD_FLAG_REL = 2.0
LOW_COVERAGE = 50.0
CANDIDATE_DISAGREE = 1.0


@dataclass
class FlaggedRegion:
    chain: str
    start_label: str
    end_label: str
    n_residues: int
    max_rmsd: float
    mean_rmsd: float
    kind: str = "deviation"

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "start": self.start_label,
            "end": self.end_label,
            "n_residues": self.n_residues,
            "max_rmsd": round(self.max_rmsd, 3),
            "mean_rmsd": round(self.mean_rmsd, 3),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AlignmentQuality:
    band: str
    verdict: str
    confidence: str
    flagged_regions: List[FlaggedRegion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "flagged_regions": [fr.to_dict() for fr in self.flagged_regions],
            "warnings": list(self.warnings),
        }


def _band(tm_score: Optional[float], rmsd: Optional[float]) -> str:
    if tm_score is not None:
        if tm_score > TM_EXCELLENT:
            return "excellent"
        if tm_score > TM_GOOD:
            return "good"
        if tm_score > TM_MODERATE:
            return "moderate"
        return "poor"
    if rmsd is not None:
        if rmsd < RMSD_EXCELLENT:
            return "excellent"
        if rmsd < RMSD_GOOD:
            return "good"
        if rmsd < RMSD_MODERATE:
            return "moderate"
        return "poor"
    return "moderate"


def _flag_deviation_regions(per_residue) -> List[FlaggedRegion]:
    vals = [r for (_c, _l, r) in per_residue if r is not None]
    if not vals:
        return []
    thr = max(RMSD_FLAG_ABS, RMSD_FLAG_REL * median(vals))
    regions: List[FlaggedRegion] = []
    run: List[Tuple[str, str, float]] = []

    def flush():
        if not run:
            return
        rs = [x[2] for x in run]
        regions.append(FlaggedRegion(
            chain=run[0][0], start_label=run[0][1], end_label=run[-1][1],
            n_residues=len(run), max_rmsd=max(rs), mean_rmsd=sum(rs) / len(rs),
            kind="deviation"))

    prev_chain = None
    for chain, label, r in per_residue:
        if r is not None and r > thr:
            if prev_chain is not None and chain != prev_chain:
                flush()
                run = []
            run.append((chain, label, r))
        else:
            flush()
            run = []
        prev_chain = chain
    flush()
    return regions


def _confidence(coverage_pct, tm_pvalue, candidate_rmsds) -> str:
    score = 2  # start "high"
    if coverage_pct is not None and coverage_pct < LOW_COVERAGE:
        score -= 1
    rs = [r for r in (candidate_rmsds or []) if r is not None]
    if len(rs) >= 2 and (max(rs) - min(rs)) > CANDIDATE_DISAGREE:
        score -= 1
    if tm_pvalue is not None and tm_pvalue > 0.05:
        score -= 1
    return {2: "high", 1: "medium"}.get(max(score, 0), "low")


def _verdict(band, tm_score, rmsd, coverage_pct) -> str:
    phrases = {
        "excellent": "Near-identical structures",
        "good": "Same fold",
        "moderate": "Partial structural similarity",
        "poor": "Little structural similarity",
    }
    detail = []
    if coverage_pct is not None and rmsd is not None:
        detail.append(f"{coverage_pct:.0f}% of residues superimpose within "
                      f"{rmsd:.1f} A")
    if tm_score is not None:
        detail.append(f"TM={tm_score:.2f}")
    if detail:
        return f"{phrases[band]}: " + ", ".join(detail)
    return phrases[band]


def assess(*, tm_score, rmsd, coverage_pct, n_aligned, per_residue,
           chain_mapping, candidate_rmsds, hinge_regions=None,
           tm_pvalue=None, mapping_warnings=None,
           tm_scope=None, tm_normalization_length=None,
           tm_per_chain_available=False) -> AlignmentQuality:
    """Turn raw alignment numbers into a plain-language quality assessment.

    ``per_residue`` is a list of ``(chain, residue_label, rmsd)`` tuples;
    ``candidate_rmsds`` a list of the seq-guided/seq-free RMSDs (may hold None);
    ``hinge_regions`` an optional list of ``(chain, start_label, end_label)``;
    ``mapping_warnings`` the notes the chain-matching step produced, which
    belong in the verdict a user reads rather than only in the warning stream.
    """
    band = _band(tm_score, rmsd)
    regions = _flag_deviation_regions(per_residue)
    if hinge_regions:
        for (chain, start, end) in hinge_regions:
            regions.append(FlaggedRegion(chain=chain, start_label=start,
                                         end_label=end, n_residues=0,
                                         max_rmsd=0.0, mean_rmsd=0.0, kind="hinge"))
    warnings: List[str] = []
    if coverage_pct is not None and coverage_pct < LOW_COVERAGE:
        warnings.append(f"Low coverage: only {coverage_pct:.0f}% of residues aligned.")
    if tm_score is None:
        warnings.append("TM-score unavailable; quality band derived from RMSD.")
    if chain_mapping is not None and len(chain_mapping) == 1:
        warnings.append("Only one chain pair aligned; multi-chain agreement not assessed.")
    if tm_scope == "complex":
        length = (f" over {tm_normalization_length} residues"
                  if tm_normalization_length else "")
        hint = (" Per-chain TM-scores are reported alongside."
                if tm_per_chain_available else
                " A single-chain TM-score is not comparable with it.")
        warnings.append(
            f"TM-score is normalized by the whole reference selection{length}, "
            f"not by one chain, because the selection spans several chains."
            f"{hint}")
    for msg in (mapping_warnings or []):
        warnings.append(msg)
    confidence = _confidence(coverage_pct, tm_pvalue, candidate_rmsds)
    # A correspondence that sequence cannot support undermines every number
    # derived from it, so it caps confidence regardless of how good the fit is.
    if any(m.startswith("Chain correspondence rests") for m in (mapping_warnings or [])):
        confidence = "low"
    verdict = _verdict(band, tm_score, rmsd, coverage_pct)
    return AlignmentQuality(band=band, verdict=verdict, confidence=confidence,
                            flagged_regions=regions, warnings=warnings)
