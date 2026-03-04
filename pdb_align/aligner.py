import os
import tempfile
from typing import Optional, List, Union

from Bio.PDB import PDBParser, MMCIFParser

from .core import (
    extract_sequences_and_lengths, _parse_path,
    sequence_independent_alignment_joined_v2,
    perform_sequence_alignment, get_aligned_atoms_by_alignment,
    superimpose_atoms, pick_best_overall, compute_chain_similarity_matrix
)

class PDBAligner:
    """
    Object-oriented interface for protein structure alignment.
    """
    def __init__(self, ref_file: Optional[str] = None, chains_ref: Optional[List[Union[str, int]]] = None):
        self.ref_file = None
        self.ref_struct = None
        self.ref_seqs = {}
        self.ref_lens = {}
        self.chains_ref = chains_ref

        self.mob_file = None
        self.mob_struct = None
        self.mob_seqs = {}
        self.mob_lens = {}
        self.chains_mob = None

        self.last_result = None

        if ref_file:
            self.set_reference(ref_file, chains_ref)

    def set_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure."""
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference file not found: {ref_file}")
        self.ref_file = ref_file
        self.chains_ref = chains
        self.ref_struct = _parse_path(ref_file)
        self.ref_seqs, self.ref_lens = extract_sequences_and_lengths(self.ref_struct, os.path.basename(ref_file))

    def add_mobile(self, mob_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the mobile structure to align."""
        if not os.path.exists(mob_file):
            raise FileNotFoundError(f"Mobile file not found: {mob_file}")
        self.mob_file = mob_file
        self.chains_mob = chains
        self.mob_struct = _parse_path(mob_file)
        self.mob_seqs, self.mob_lens = extract_sequences_and_lengths(self.mob_struct, os.path.basename(mob_file))

    def align(self, mode: str = "auto", seq_gap_open: float = -10, seq_gap_extend: float = -0.5, **kwargs):
        """
        Runs the alignment process.

        mode:
            "auto" (or "Auto (best RMSD)"): best RMSD among seq-guided and seq-free
            "seq_guided" (or "Sequence-guided"): sequence-guided
            "seq_free_auto" (or "Sequence-free (auto)"): sequence-free auto
            "seq_free_shape" (or "Sequence-free (shape)"): sequence-free shape
            "seq_free_window" (or "Sequence-free (window)"): sequence-free window
        """
        if not self.ref_file or not self.mob_file:
            raise ValueError("Both reference and mobile structures must be set before alignment.")

        ref_chs = self.chains_ref if self.chains_ref else list(self.ref_seqs.keys())
        mob_chs = self.chains_mob if self.chains_mob else list(self.mob_seqs.keys())

        if not ref_chs or not mob_chs:
            raise ValueError("Select at least one chain per file.")

        seqguided = None
        seqfree = None

        if mode in ("auto", "Auto (best RMSD)", "seq_guided", "Sequence-guided"):
            seqA = "".join(str(self.ref_seqs[c].seq) for c in ref_chs if c in self.ref_seqs)
            seqB = "".join(str(self.mob_seqs[c].seq) for c in mob_chs if c in self.mob_seqs)
            aln = perform_sequence_alignment(seqA, seqB, seq_gap_open, seq_gap_extend)
            if aln:
                ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(self.ref_struct, ref_chs, self.mob_struct, mob_chs, aln)
                if ref_atoms and mob_atoms:
                    si = superimpose_atoms(ref_atoms, mob_atoms)
                    if si:
                        seqguided = dict(aln=aln, ref_atoms=ref_atoms, mob_atoms=mob_atoms, si=si)

        if mode in ("auto", "Auto (best RMSD)", "seq_free_auto", "Sequence-free (auto)", "seq_free_shape", "Sequence-free (shape)", "seq_free_window", "Sequence-free (window)"):
            sf_mode_map = {
                "seq_free_shape": "shape",
                "Sequence-free (shape)": "shape",
                "seq_free_window": "window",
                "Sequence-free (window)": "window"
            }
            sf_method = sf_mode_map.get(mode, "auto")

            try:
                res = sequence_independent_alignment_joined_v2(
                    file_ref=self.ref_file, file_mob=self.mob_file,
                    chains_ref=ref_chs, chains_mob=mob_chs,
                    method=sf_method, **kwargs
                )
                seqfree = res
            except Exception as e:
                pass

        if mode in ("auto", "Auto (best RMSD)"):
            best, reason = pick_best_overall(seqguided, seqfree, min_pairs=3)
            if best is None:
                raise RuntimeError("No alignment could be produced. Try different chains or mode.")
            chosen_name = best["name"]
            chosen = dict(name=chosen_name, reason=reason,
                          seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                          seqfree=seqfree if "Sequence-free" in chosen_name else None)
        else:
            chosen = dict(name=mode, reason="Manual mode.",
                          seqguided=seqguided if "seq_guided" in mode or "Sequence-guided" in mode else None,
                          seqfree=seqfree if "seq_free" in mode or "Sequence-free" in mode else None)

        self.last_result = dict(seqguided=seqguided, seqfree=seqfree, chosen=chosen)
        return self.last_result

    def batch_align(self, mob_dir: str, out_dir: str, mode: str = "auto"):
        """
        Aligns a directory of PDBs against the current reference structure.
        """
        if not self.ref_file:
            raise ValueError("Reference structure must be set for batch alignment.")

        os.makedirs(out_dir, exist_ok=True)
        results = {}
        for fname in os.listdir(mob_dir):
            if fname.lower().endswith((".pdb", ".cif", ".mmcif")):
                fpath = os.path.join(mob_dir, fname)
                try:
                    self.add_mobile(fpath)
                    res = self.align(mode=mode)
                    rmsd = self.get_rmsd()
                    results[fname] = {"rmsd": rmsd, "status": "success"}
                    out_pdb = os.path.join(out_dir, f"aligned_{fname}")
                    self.save_aligned_pdb(out_pdb)
                except Exception as e:
                    results[fname] = {"status": "error", "message": str(e)}
        return results

    def get_rmsd(self) -> Optional[float]:
        """Returns the RMSD of the best alignment."""
        if not self.last_result:
            return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]:
            return chosen["seqguided"]["si"]["rmsd"]
        elif chosen["seqfree"]:
            return chosen["seqfree"].rmsd
        return None

    def get_aligned_coords(self) -> Optional[tuple]:
        """Returns aligned coordinates depending on the chosen method."""
        if not self.last_result:
            return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]:
            return chosen["seqguided"]["si"]["ref_coords"], chosen["seqguided"]["si"]["mob_coords_transformed"]
        elif chosen["seqfree"]:
            return chosen["seqfree"].ref_subset_ca_coords, chosen["seqfree"].mob_subset_ca_coords_aligned
        return None

    def get_matched_pairs(self) -> Optional[list]:
        """Returns matched pairs."""
        if not self.last_result:
            return None
        chosen = self.last_result["chosen"]
        if chosen["seqfree"]:
            return chosen["seqfree"].pairs
        return None

    def get_sequence_alignment(self) -> Optional[tuple]:
        """Returns sequence alignment text if sequence-guided method was used."""
        if not self.last_result:
            return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]:
            aln = chosen["seqguided"]["aln"]
            return aln.seqA, aln.seqB, aln.score
        return None

    def get_similarity_matrix(self):
        """Returns the chain similarity matrices (Identity and BLOSUM62 scores)."""
        if not self.ref_file or not self.mob_file:
            raise ValueError("Both reference and mobile structures must be set.")
        return compute_chain_similarity_matrix(self.ref_seqs, self.mob_seqs)

    def save_aligned_pdb(self, filename: str, subset_only: bool = False):
        """Saves the aligned mobile structure to a PDB file."""
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        from Bio.PDB import PDBIO
        from .core import _AllAtomsSelect

        chosen = self.last_result["chosen"]
        R = None
        t = None
        if chosen["seqguided"]:
            R = chosen["seqguided"]["si"]["rotation"]
            t = chosen["seqguided"]["si"]["translation"]
        elif chosen["seqfree"]:
            R = chosen["seqfree"].rotation
            t = chosen["seqfree"].translation

        if R is not None and t is not None:
            io_obj = PDBIO()
            io_obj.set_structure(self.mob_struct)
            io_obj.save(filename, _AllAtomsSelect(R=R, t=t))

    def save_log(self, filename: str):
        """Saves alignment log summary."""
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")
        with open(filename, "w") as f:
            f.write("PDB Aligner Result Log\n")
            f.write("="*20 + "\n")
            f.write(f"Reference: {self.ref_file}\n")
            f.write(f"Mobile: {self.mob_file}\n")
            chosen = self.last_result["chosen"]
            f.write(f"Chosen method: {chosen['name']}\n")
            f.write(f"RMSD: {self.get_rmsd()}\n")
