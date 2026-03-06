import os
import tempfile
import concurrent.futures
from typing import Optional, List, Union

from Bio.PDB import PDBParser, MMCIFParser

from .core import (
    extract_sequences_and_lengths, _parse_path,
    sequence_independent_alignment_joined_v2,
    perform_sequence_alignment, get_aligned_atoms_by_alignment,
    superimpose_atoms, pick_best_overall, compute_chain_similarity_matrix
)
from .structure import StructureBase
from .result import AlignmentResult


class PDBAligner:
    """
    Object-oriented API for protein 3D structural alignment.

    This class supports single and batch alignments, offering various methods:
    - Sequence-guided structural superposition.
    - Sequence-free structural superposition (useful for low sequence identity).

    Attributes:
        verbose (bool): If True, prints status messages during operations.
    """
    def __init__(self, mode: str = "auto", atoms: str = "CA", verbose: bool = False):
        self.verbose = verbose
        self.mode = mode
        self.atoms = atoms

    def align(self, reference: StructureBase, mobile: StructureBase, seq_gap_open: float = -10, seq_gap_extend: float = -0.5, **kwargs) -> AlignmentResult:
        """
        Runs the alignment process. Returns an AlignmentResult stateful object mapping.
        """
        ref_chs = reference.chains
        mob_chs = mobile.chains

        if not ref_chs or not mob_chs:
            raise ValueError("Select at least one chain per file.")

        seqguided = None
        seqfree = None

        if self.mode in ("auto", "Auto (best RMSD)", "seq_guided", "Sequence-guided"):
            # Simple sequence extraction for backward compatibility structure
            # To be thoroughly correct with Gemmi, this can be moved to StructureBase
            ref_struct_bp = _parse_path(reference.filepath)
            mob_struct_bp = _parse_path(mobile.filepath)
            ref_seqs, _ = extract_sequences_and_lengths(ref_struct_bp, os.path.basename(reference.filepath))
            mob_seqs, _ = extract_sequences_and_lengths(mob_struct_bp, os.path.basename(mobile.filepath))

            seqA = "".join(str(ref_seqs[c].seq) for c in ref_chs if c in ref_seqs)
            seqB = "".join(str(mob_seqs[c].seq) for c in mob_chs if c in mob_seqs)
            aln = perform_sequence_alignment(seqA, seqB, seq_gap_open, seq_gap_extend)
            
            if aln:
                ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(ref_struct_bp, ref_chs, mob_struct_bp, mob_chs, aln)
                if ref_atoms and mob_atoms:
                    si = superimpose_atoms(ref_atoms, mob_atoms)
                    if si:
                        seqguided = dict(aln=aln, ref_atoms=ref_atoms, mob_atoms=mob_atoms, si=si)

        if self.mode in ("auto", "Auto (best RMSD)", "seq_free_auto", "Sequence-free (auto)", "seq_free_shape", "Sequence-free (shape)", "seq_free_window", "Sequence-free (window)"):
            sf_mode_map = {
                "seq_free_shape": "shape",
                "Sequence-free (shape)": "shape",
                "seq_free_window": "window",
                "Sequence-free (window)": "window"
            }
            sf_method = sf_mode_map.get(self.mode, "auto")

            try:
                res = sequence_independent_alignment_joined_v2(
                    file_ref=reference.filepath, file_mob=mobile.filepath,
                    chains_ref=ref_chs, chains_mob=mob_chs,
                    method=sf_method, **kwargs
                )
                seqfree = res
            except Exception as e:
                pass

        if self.mode in ("auto", "Auto (best RMSD)"):
            best, reason = pick_best_overall(seqguided, seqfree, min_pairs=3)
            if best is None:
                raise RuntimeError("No alignment could be produced. Try different chains or mode.")
            chosen_name = best["name"]
            chosen = dict(name=chosen_name, reason=reason,
                          seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                          seqfree=seqfree if "Sequence-free" in chosen_name else None)
        else:
            chosen = dict(name=self.mode, reason="Manual mode.",
                          seqguided=seqguided if "seq_guided" in self.mode or "Sequence-guided" in self.mode else None,
                          seqfree=seqfree if "seq_free" in self.mode or "Sequence-free" in self.mode else None)

        if self.verbose:
            print(f"\nAlignment Completed:")
            print(f"  Mode evaluated: {self.mode}")
            if seqguided:
                print(f"  Sequence-based RMSD: {seqguided['si']['rmsd']:.3f} Å")
            if seqfree:
                print(f"  Sequence-free RMSD: {seqfree.rmsd:.3f} Å")
            print(f"  Chosen method: {chosen['name']}")
            print(f"  Reason: {chosen['reason']}")

        # Map to AlignmentResult Dataclass
        if chosen["seqguided"]:
            rmsd = float(chosen["seqguided"]["si"]["rmsd"])
            rotation = chosen["seqguided"]["si"]["rotation"]
            translation = chosen["seqguided"]["si"]["translation"]
            pairs = len(chosen["seqguided"]["ref_atoms"])
            ref_coords = chosen["seqguided"]["si"]["ref_coords"]
            mob_coords = chosen["seqguided"]["si"]["mob_coords_transformed"]
            ref_length = len(ref_coords)
        else:
            rmsd = float(chosen["seqfree"].rmsd)
            rotation = chosen["seqfree"].rotation
            translation = chosen["seqfree"].translation
            pairs = int(chosen["seqfree"].kept_pairs)
            ref_coords = chosen["seqfree"].ref_subset_ca_coords
            mob_coords = chosen["seqfree"].mob_subset_ca_coords_aligned
            ref_length = len(ref_coords)
            
        return AlignmentResult(
            method=chosen["name"],
            rmsd=rmsd,
            rotation_matrix=rotation,
            translation_vector=translation,
            aligned_pairs=pairs,
            ref_length=ref_length,
            ref_coords=ref_coords,
            mob_coords=mob_coords
        )

    def _align_single(self, task_kwargs):
        reference = task_kwargs["reference"]
        fpath = task_kwargs["fpath"]
        fname = task_kwargs["fname"]
        out_pdb = task_kwargs["out_pdb"]
        kwargs = task_kwargs["kwargs"]
        
        try:
            mobile = StructureBase(fpath)
            res = self.align(reference, mobile, **kwargs)
            # Temporarily stubbed out native save_pdb - will assume output relies on Result mapping later
            # res.save_pdb(out_pdb) 
            return fname, {"rmsd": res.rmsd, "tm_score": res.tm_score, "status": "success"}
        except Exception as e:
            return fname, {"status": "error", "message": str(e)}

    def batch_align(self, reference: StructureBase, mob_dir: str, out_dir: str, workers: int = 4, **kwargs):
        """
        Aligns a directory of PDBs against the current reference structure concurrently.
        """
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        
        tasks = []
        for fname in os.listdir(mob_dir):
            if fname.lower().endswith((".pdb", ".cif", ".mmcif")):
                fpath = os.path.join(mob_dir, fname)
                out_pdb = os.path.join(out_dir, f"aligned_{fname}")
                tasks.append({
                    "reference": reference,
                    "fpath": fpath,
                    "fname": fname,
                    "out_pdb": out_pdb,
                    "kwargs": kwargs
                })
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for fname, result in executor.map(self._align_single, tasks):
                results[fname] = result
                if self.verbose:
                    print(f"Processed {fname}: {result}")
        
        return results
