import os
from typing import List, Union, Optional
import numpy as np
import gemmi
from .exceptions import ParsingError, ChainNotFoundError

class StructureBase:
    """Wrapper around gemmi.Structure for flexible parsing and sub-domain access."""
    
    def __init__(self, filepath: str, chains: Optional[List[Union[str, int]]] = None):
        self.filepath = filepath
        self._struct = self._load_structure(filepath)
        self.chains = self._parse_chains(chains)
        self.model = self._struct[0] # assuming single model for now
        
    def _load_structure(self, filepath: str) -> gemmi.Structure:
        if not os.path.exists(filepath):
            raise ParsingError(f"File not found: {filepath}")
        try:
            return gemmi.read_structure(filepath)
        except Exception as e:
            raise ParsingError(f"Failed to parse {filepath}: {e}")
            
    def _parse_chains(self, chains: Optional[List[Union[str, int]]]) -> List[str]:
        if chains is None:
            return [chain.name for chain in self._struct[0]]
        
        parsed_chains = []
        for c in chains:
            if isinstance(c, int):
                if c >= len(self._struct[0]):
                    raise ChainNotFoundError(f"Chain index {c} out of bounds.")
                parsed_chains.append(self._struct[0][c].name)
            else:
                # Store string representations (could be "A" or "A:10-150")
                parsed_chains.append(str(c))
        return parsed_chains

    def get_coords(self, atoms: str = "CA") -> np.ndarray:
        """Extracts coordinates for the specified atom type and active chains."""
        coords = []
        for chain_def in self.chains:
            chain_name = chain_def.split(":")[0] if ":" in chain_def else chain_def
            
            # Sub-domain selection
            residue_range = None
            if ":" in chain_def:
                try:
                    start, end = map(int, chain_def.split(":")[1].split("-"))
                    residue_range = (start, end)
                except ValueError:
                    raise ParsingError(f"Invalid chain definition: {chain_def}")

            chain = self.model.find_chain(chain_name)
            if not chain:
                raise ChainNotFoundError(f"Chain {chain_name} not found in {self.filepath}")

            for residue in chain:
                # Check if residue falls within specified range
                if residue_range:
                    res_seq = residue.seqid.num
                    if not (residue_range[0] <= res_seq <= residue_range[1]):
                        continue

                for atom in residue:
                    if self._accept_atom(atom, atoms):
                        coords.append(atom.pos.tolist())
                        
        return np.array(coords)

    def _accept_atom(self, atom: gemmi.Atom, selection: str) -> bool:
        if selection == "CA":
            return atom.name == "CA"
        elif selection == "backbone":
            return atom.name in ["N", "CA", "C", "O"]
        elif selection == "all_heavy":
            return atom.element.name != "H"
        return False
        
    def info(self) -> str:
        return f"<Structure: {os.path.basename(self.filepath)} | Chains: {', '.join(self.chains)}>"

