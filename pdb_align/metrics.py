import numpy as np

# Minimum d0 used by TM-align. The raw d0 formula goes to zero (and even
# negative) for short chains, so it must be clamped for the score to stay
# physically meaningful.
_D0_MIN = 0.5


def compute_d0(length: int) -> float:
    """
    TM-score normalization distance d0 for a target of *length* residues.

    Uses the TM-align formula ``d0 = 1.24*(L-15)^(1/3) - 1.8`` and clamps the
    result to a minimum of 0.5 A, matching the reference TM-align behaviour.
    Without the clamp, d0 is negative for L in ~16-20, which makes the score
    meaningless.
    """
    if length <= 15:
        return _D0_MIN
    d0 = 1.24 * np.power(length - 15, 1.0 / 3.0) - 1.8
    return float(max(d0, _D0_MIN))


def calculate_tm_score(ref_coords: np.ndarray, mob_coords: np.ndarray, length: int) -> float:
    """
    Calculates the TM-score for two sets of aligned coordinates.
    
    Args:
        ref_coords: Reference coordinates (N, 3).
        mob_coords: Mobile coordinates aligned to reference (N, 3).
        length: The length of the target protein (usually reference length).
        
    Returns:
        TM-score (float between 0 and 1).
    """
    if len(ref_coords) != len(mob_coords) or len(ref_coords) == 0:
        return 0.0

    d0 = compute_d0(length)

    dists = np.linalg.norm(ref_coords - mob_coords, axis=1)
    score = np.sum(1 / (1 + (dists / d0)**2)) / length
    
    return float(score)

def calculate_lddt(ref_coords: np.ndarray, mob_coords: np.ndarray, threshold: float = 15.0) -> float:
    """
    Calculates the lDDT (Local Distance Difference Test) score.
    
    Args:
        ref_coords: Reference coordinates (N, 3).
        mob_coords: Mobile coordinates aligned to reference (N, 3).
        threshold: Distance inclusion threshold (default 15.0 A).
        
    Returns:
        lDDT score (float between 0 and 1).
    """
    if len(ref_coords) != len(mob_coords) or len(ref_coords) == 0:
        return 0.0
        
    n_atoms = len(ref_coords)
    if n_atoms <= 1:
        return 0.0

    # Calculate all pairwise distances
    ref_dists = np.linalg.norm(ref_coords[:, None, :] - ref_coords[None, :, :], axis=-1)
    mob_dists = np.linalg.norm(mob_coords[:, None, :] - mob_coords[None, :, :], axis=-1)
    
    # Create mask for pairs within threshold (excluding self-pairs)
    mask = (ref_dists < threshold) & (np.arange(n_atoms)[:, None] != np.arange(n_atoms)[None, :])
    
    if not np.any(mask):
        return 0.0
        
    # Calculate difference in distances
    diffs = np.abs(ref_dists - mob_dists)
    
    # Calculate fractions of distances preserved within thresholds: 0.5, 1.0, 2.0, 4.0
    preserved_05 = np.sum((diffs < 0.5) & mask)
    preserved_10 = np.sum((diffs < 1.0) & mask)
    preserved_20 = np.sum((diffs < 2.0) & mask)
    preserved_40 = np.sum((diffs < 4.0) & mask)
    
    total_pairs = np.sum(mask)
    
    lddt = (preserved_05 + preserved_10 + preserved_20 + preserved_40) / (4 * total_pairs)
    
    return float(lddt)

import math

# Parameters of the random-pair TM-score distribution.
#
# TM-score's d0 normalization was designed so that the TM-score of a pair of
# unrelated ("random") structures has a mean of ~0.17 that is essentially
# independent of chain length (Zhang & Skolnick, Proteins 2004, 57:702-710).
# Empirically that random-pair distribution is right-skewed and well described
# by an extreme-value (Gumbel) distribution with a spread of roughly 0.05.
_TM_RANDOM_MEAN = 0.17
_TM_RANDOM_STD = 0.05
_EULER_GAMMA = 0.5772156649015329


def calculate_tm_pvalue(tm_score: float, length: int) -> float:
    """
    Approximate p-value for an observed TM-score.

    Estimates P(TM_random >= tm_score): the probability that a pair of
    *unrelated* structures would reach at least this TM-score by chance. Small
    values indicate significant structural similarity.

    The random-pair TM-score distribution is modelled as a Gumbel (extreme
    value) distribution with mean ~0.17 and std ~0.05, which is approximately
    length-independent by TM-score's construction. This is an *approximation*
    intended to give a correctly-behaved significance signal (random matches
    score near 1, strong matches near 0); it is not a reproduction of any
    specific tool's exact p-value.

    Parameters
    ----------
    tm_score : float
        Observed TM-score in [0, 1].
    length : int
        Target chain length. Below 16 residues the score is not statistically
        meaningful and the p-value is reported as 1.0.
    """
    if length <= 15:
        return 1.0  # too short to be statistically meaningful
    if tm_score < 0.0:
        return 1.0
    if tm_score >= 1.0:
        return 0.0

    # Gumbel scale/location from the target mean and std.
    beta = _TM_RANDOM_STD * math.sqrt(6.0) / math.pi
    mu = _TM_RANDOM_MEAN - beta * _EULER_GAMMA

    # Upper-tail probability: P(X >= x) = 1 - exp(-exp(-(x-mu)/beta)).
    z = (tm_score - mu) / beta
    survival = -math.expm1(-math.exp(-z))
    return float(min(1.0, max(0.0, survival)))
