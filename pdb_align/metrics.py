import numpy as np

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
        
    if length <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * np.power(length - 15, 1/3) - 1.8

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
