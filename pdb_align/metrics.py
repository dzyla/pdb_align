"""Standalone structure-comparison metrics (pure numpy, no gemmi/BioPython).

References
----------
- TM-score: Zhang & Skolnick, Proteins 2004, 57:702-710.
- TM-score significance (EVD parameters): Xu & Zhang, Bioinformatics 2010,
  26:889-895 ("How significant is a protein structure similarity with
  TM-score = 0.5?").
- lDDT: Mariani, Biasini, Barbato & Schwede, Bioinformatics 2013,
  29:2722-2728.
"""
import math

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

    Note: this evaluates the TM-score of the *given* superposition. The
    reported TM-score of an alignment should be the maximum over
    superpositions — see :func:`tm_optimal_superposition`.

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


def _kabsch_np(P: np.ndarray, Q: np.ndarray):
    """Minimal Kabsch superposition (local copy so metrics stays dependency-free).

    Returns (R, t) such that R @ Q + t best fits P in the least-squares sense.
    """
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (Q - cQ).T @ (P - cP)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cP - R @ cQ
    return R, t


# TM-score's search cutoff, clamped as in the reference implementation. The
# iterative refinement selects residues within this distance; the clamp keeps
# the search radius sane for very short and very long chains.
_D0_SEARCH_MIN = 4.5
_D0_SEARCH_MAX = 8.0


def tm_optimal_superposition(ref_coords: np.ndarray, mob_coords: np.ndarray,
                             length: int, max_iter: int = 20,
                             max_starts: int = 48):
    """
    TM-score-maximizing superposition for a *fixed* residue correspondence.

    TM-align/TM-score report the TM-score of the superposition that maximizes
    it, not of the RMSD-optimal (Kabsch) superposition; evaluating TM in the
    Kabsch frame systematically underestimates it whenever flexible tails or
    hinges drag the least-squares fit.

    This follows the search of the reference TM-score program: seed a
    superposition from every contiguous fragment of length L, L/2, L/4, ...
    down to 4 residues at every start position, then iteratively re-superpose
    on the residues within ``d0_search`` (clamped to [4.5, 8] A, grown by 0.5 A
    when fewer than three residues qualify) until the selected set stops
    changing, and keep the best TM-score seen.

    An earlier version used only seven non-overlapping seeds (full chain,
    halves, quarters) and a fixed cutoff of ``max(d0, 1)``. Both deviations
    lose TM-score on structures where the best superposition is driven by a
    sub-domain, because no seed starts inside it.

    Args:
        ref_coords: (N, 3) reference CA coordinates.
        mob_coords: (N, 3) mobile CA coordinates in any frame (the optimal
            rigid transform is re-derived internally).
        length: normalization length L for the TM-score (reference length).
        max_iter: refinement iterations per seed.
        max_starts: cap on seed start positions per fragment length, which
            bounds the search cost on large structures. Raising it cannot
            lower the score (the search only ever keeps the best seed).

    Returns:
        (tm_score, R, t): the maximal TM-score and its superposition, with
        ``R @ mob + t`` in the reference frame.
    """
    P = np.ascontiguousarray(ref_coords, dtype=float)
    Q = np.ascontiguousarray(mob_coords, dtype=float)
    N = len(P)
    if N == 0 or N != len(Q) or length <= 0:
        return 0.0, np.eye(3), np.zeros(3)
    if N < 3:
        t = P.mean(axis=0) - Q.mean(axis=0)
        d = np.linalg.norm(P - (Q + t), axis=1)
        d0 = compute_d0(length)
        return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / length), np.eye(3), t

    d0 = compute_d0(length)
    d0_sq = d0 * d0
    d0_search = min(max(d0, _D0_SEARCH_MIN), _D0_SEARCH_MAX)

    def tm_of(R, t):
        d_sq = np.sum((P - ((R @ Q.T).T + t)) ** 2, axis=1)
        return float(np.sum(1.0 / (1.0 + d_sq / d0_sq)) / length), d_sq

    # Fragment lengths L, L/2, L/4, ... >= 4, each at every start position.
    frag_lengths = []
    flen = N
    while flen >= 4:
        frag_lengths.append(flen)
        if flen == 4:
            break
        flen = max(4, flen // 2)

    best_tm, best_R, best_t = -1.0, np.eye(3), np.zeros(3)
    for flen in frag_lengths:
        # The reference program tries every start position, which is O(L^2)
        # seeds — fine in Fortran, 5.5 s at L = 3000 here. Starts are spread
        # evenly instead, at most `max_starts` per fragment length, so short
        # fragments still probe the whole chain (consecutive windows of the
        # same length overlap heavily and converge to the same refinement).
        n_pos = N - flen + 1
        if n_pos <= max_starts:
            starts = range(n_pos)
        else:
            starts = np.unique(np.linspace(0, n_pos - 1, max_starts).astype(int))
        for start in starts:
            R, t = _kabsch_np(P[start:start + flen], Q[start:start + flen])
            tm, d_sq = tm_of(R, t)
            if tm > best_tm:
                best_tm, best_R, best_t = tm, R, t
            prev_sel = None
            cut = d0_search
            for _ in range(max_iter):
                sel = d_sq < cut * cut
                n_sel = int(sel.sum())
                if n_sel < 3:
                    if cut > 50.0:
                        break
                    cut += 0.5
                    continue
                if prev_sel is not None and np.array_equal(sel, prev_sel):
                    break
                prev_sel = sel
                R, t = _kabsch_np(P[sel], Q[sel])
                tm, d_sq = tm_of(R, t)
                if tm > best_tm:
                    best_tm, best_R, best_t = tm, R, t
    return best_tm, best_R, best_t


def calculate_lddt(ref_coords: np.ndarray, mob_coords: np.ndarray,
                   threshold: float = 15.0, chunk: int = 512) -> float:
    """
    lDDT-Ca (Local Distance Difference Test on the given coordinate set).

    Superposition-free: compares the two internal distance matrices over all
    pairs whose *reference* distance is below ``threshold`` (15 A inclusion
    radius, as in Mariani et al. 2013), scoring the fraction preserved within
    0.5/1/2/4 A. Computed in row chunks so large complexes do not allocate
    full N x N matrices. When called with CA coordinates of matched residues
    this is the lDDT-Ca of the *matched* region; unmatched residues are not
    penalized (report coverage alongside).

    Args:
        ref_coords: Reference coordinates (N, 3).
        mob_coords: Model coordinates (N, 3), any frame.
        threshold: Inclusion radius on reference distances (default 15.0 A).
        chunk: Row-block size for the chunked computation.

    Returns:
        lDDT score in [0, 1].
    """
    if len(ref_coords) != len(mob_coords) or len(ref_coords) == 0:
        return 0.0

    n = len(ref_coords)
    if n <= 1:
        return 0.0

    P = np.asarray(ref_coords, dtype=float)
    Q = np.asarray(mob_coords, dtype=float)
    preserved = 0
    total = 0
    idx = np.arange(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ref_d = np.linalg.norm(P[s:e, None, :] - P[None, :, :], axis=-1)
        mob_d = np.linalg.norm(Q[s:e, None, :] - Q[None, :, :], axis=-1)
        mask = (ref_d < threshold) & (idx[s:e, None] != idx[None, :])
        if not mask.any():
            continue
        diffs = np.abs(ref_d - mob_d)[mask]
        for tol in (0.5, 1.0, 2.0, 4.0):
            preserved += int(np.sum(diffs < tol))
        total += int(mask.sum())

    if total == 0:
        return 0.0
    return float(preserved / (4 * total))


# Extreme-value distribution of the TM-score of random (unrelated) structure
# pairs, fitted by Xu & Zhang (Bioinformatics 2010, 26:889-895) on 7.2e7
# gapless comparisons of non-homologous PDB domains:
#     F(x) = exp(-exp(-(x - mu)/sigma)),  mu = 0.1512, sigma = 0.0242,
# length-independent by TM-score's construction. The paper's stated golden
# value P(TM >= 0.5) = 5.5e-7 follows directly from these parameters.
_TM_EVD_MU = 0.1512
_TM_EVD_SIGMA = 0.0242


def calculate_tm_pvalue(tm_score: float, length: int) -> float:
    """
    P-value of an observed TM-score under the random-pair null model.

    Estimates P(TM_random >= tm_score): the probability that a pair of
    *unrelated* structures reaches at least this TM-score by chance, using
    the extreme-value distribution fitted by Xu & Zhang (Bioinformatics 2010,
    26:889-895): mu = 0.1512, sigma = 0.0242, length-independent. At
    TM-score = 0.5 this gives 5.5e-7, matching the published value.

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

    # Upper-tail probability: P(X >= x) = 1 - exp(-exp(-(x-mu)/sigma)).
    z = (tm_score - _TM_EVD_MU) / _TM_EVD_SIGMA
    survival = -math.expm1(-math.exp(-z))
    return float(min(1.0, max(0.0, survival)))
