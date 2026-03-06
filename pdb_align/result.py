import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .metrics import calculate_tm_score, calculate_lddt

@dataclass
class AlignmentResult:
    """Contains the results of a structural alignment operation."""
    method: str
    rmsd: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    aligned_pairs: int
    ref_length: int
    ref_coords: np.ndarray
    mob_coords: np.ndarray
    _tm_score: Optional[float] = None
    _lddt: Optional[float] = None
    
    @property
    def tm_score(self) -> float:
        if self._tm_score is None:
            self._tm_score = calculate_tm_score(self.ref_coords, self.mob_coords, self.ref_length)
        return self._tm_score
        
    @property
    def lddt(self) -> float:
        if self._lddt is None:
            self._lddt = calculate_lddt(self.ref_coords, self.mob_coords)
        return self._lddt

    def __str__(self):
        return f"<AlignmentResult Method: {self.method} | RMSD: {self.rmsd:.2f}Å | TM-score: {self.tm_score:.3f} | Pairs: {self.aligned_pairs}>"

    def __repr__(self):
        return self.__str__()
        
    def save_pdb(self, filename: str):
        """Saves the aligned mobile structure to a PDB file."""
        # TODO: Implement using gemmi
        pass
        
    def plot_rmsd(self, filename: str, style: str = "scientific"):
        """Plots the RMSD of the alignment."""
        # TODO: Port plotting code
        pass
