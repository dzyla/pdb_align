import os
import tempfile
from typing import Optional, List, Union
import numpy.typing as npt

from Bio.PDB import PDBParser, MMCIFParser

from .core import (
    extract_sequences_and_lengths, _parse_path,
    sequence_independent_alignment_joined_v2,
    perform_sequence_alignment, get_aligned_atoms_by_alignment,
    superimpose_atoms, pick_best_overall, compute_chain_similarity_matrix
)

class AlignmentFailedError(Exception):
    """Raised when the alignment fails to produce a viable result."""
    pass

class ChainNotFoundError(Exception):
    """Raised when a requested chain is not found in the structure."""
    pass

class AlignmentResult:
    """
    Encapsulates the result of a structural alignment, providing a clean, stateless
    interface to properties like RMSD, matrices, and plotting tools.
    """
    def __init__(self, chosen: dict, seqguided: dict, seqfree: dict, ref_file: str, mob_file: str, mob_struct, ref_lens: dict, mob_lens: dict, verbose: bool = False):
        self._chosen = chosen
        self._seqguided = seqguided
        self._seqfree = seqfree
        self.ref_file = ref_file
        self.mob_file = mob_file
        self.mob_struct = mob_struct
        self.ref_lens = ref_lens
        self.mob_lens = mob_lens
        self.verbose = verbose

    @property
    def rmsd(self) -> Optional[float]:
        if self._chosen["seqguided"]:
            return self._chosen["seqguided"]["si"]["rmsd"]
        elif self._chosen["seqfree"]:
            return self._chosen["seqfree"].rmsd
        return None

    @property
    def tm_score(self) -> Optional[float]:
        """Wrapper for get_tm_score('reference') to maintain backwards compatibility."""
        return self.get_tm_score(normalize_by='reference')

    def get_tm_score(self, normalize_by: str = 'reference') -> Optional[float]:
        """
        Calculates the TM-score.

        normalize_by: 'reference', 'mobile', or 'min'.

        Warning: Standard TM-align always normalizes by the length of the *reference*
        protein. Changing the normalization length breaks TM-score comparability
        across different targets.
        """
        import numpy as np

        # Calculate TM-score dynamically
        if self._chosen["seqguided"]:
            ref_atoms = self._chosen["seqguided"]["ref_atoms"]
            mob_atoms = self._chosen["seqguided"]["mob_atoms"]
            per_res_rmsd = self._chosen["seqguided"]["si"]["per_residue_rmsd"]

            # Filter to CA only for TM-score if backbone/all_heavy was used
            ca_indices = [i for i, a in enumerate(ref_atoms) if a.get_name() == "CA"]
            if not ca_indices:
                return None
            ca_rmsd = per_res_rmsd[ca_indices]

            # The length should be the total length of the chains, not just the matched atoms
            L_ref = sum(self.ref_lens.values())
            L_mob = sum(self.mob_lens.values())

            if normalize_by == 'reference': L = L_ref
            elif normalize_by == 'mobile': L = L_mob
            elif normalize_by == 'min': L = min(L_ref, L_mob)
            else: raise ValueError("normalize_by must be 'reference', 'mobile', or 'min'")

            if L <= 15:
                return None
            d0 = 1.24 * (L - 15)**(1.0/3.0) - 1.8
            d0_sq = d0**2
            tm = np.sum(1.0 / (1.0 + (ca_rmsd**2) / d0_sq)) / L
            return float(tm)

        elif self._chosen["seqfree"]:
            ref_subset = self._chosen["seqfree"].ref_subset_infos
            mob_subset = self._chosen["seqfree"].mob_subset_infos
            pairs = self._chosen["seqfree"].pairs

            L_ref = sum(self.ref_lens.values())
            L_mob = sum(self.mob_lens.values())

            if normalize_by == 'reference': L = L_ref
            elif normalize_by == 'mobile': L = L_mob
            elif normalize_by == 'min': L = min(L_ref, L_mob)
            else: raise ValueError("normalize_by must be 'reference', 'mobile', or 'min'")

            if L <= 15:
                return None
            d0 = 1.24 * (L - 15)**(1.0/3.0) - 1.8
            d0_sq = d0**2

            sum_val = 0.0
            for (i, j) in pairs:
                diff = self._chosen["seqfree"].ref_subset_ca_coords[i] - self._chosen["seqfree"].mob_subset_ca_coords_aligned[j]
                dist_sq = np.sum(diff**2)
                sum_val += 1.0 / (1.0 + dist_sq / d0_sq)
            return float(sum_val / L)
        return None

    @property
    def tm_pvalue(self) -> Optional[float]:
        """
        Returns the p-value of the TM-score for the aligned structures.
        """
        tm = self.tm_score
        if tm is None:
            return None

        import pdb_align.metrics as metrics
        L = sum(self.ref_lens.values())
        return metrics.calculate_tm_pvalue(tm, L)

    @property
    def rotation_matrix(self) -> Optional[npt.NDArray]:
        if self._chosen["seqguided"]: return self._chosen["seqguided"]["si"]["rotation"]
        elif self._chosen["seqfree"]: return self._chosen["seqfree"].rotation
        return None

    @property
    def translation_vector(self) -> Optional[npt.NDArray]:
        if self._chosen["seqguided"]: return self._chosen["seqguided"]["si"]["translation"]
        elif self._chosen["seqfree"]: return self._chosen["seqfree"].translation
        return None

    def get_aligned_coords(self) -> Optional[tuple]:
        if self._chosen["seqguided"]:
            return self._chosen["seqguided"]["si"]["ref_coords"], self._chosen["seqguided"]["si"]["mob_coords_transformed"]
        elif self._chosen["seqfree"]:
            return self._chosen["seqfree"].ref_subset_ca_coords, self._chosen["seqfree"].mob_subset_ca_coords_aligned
        return None

    def get_matched_pairs(self) -> Optional[list]:
        if self._chosen["seqfree"]:
            return self._chosen["seqfree"].pairs
        return None

    def save_aligned_pdb(self, filename: str, subset_only: bool = False):
        """Saves the aligned mobile structure to a PDB file. Maps alignment distance into B-factor."""
        if not self._chosen:
            raise ValueError("No alignment results available. Run align() first.")

        chosen = self._chosen
        R = None
        t = None
        per_res_rmsd = None
        ref_atoms = []
        mob_atoms = []

        if chosen["seqguided"]:
            R = chosen["seqguided"]["si"]["rotation"]
            t = chosen["seqguided"]["si"]["translation"]
            per_res_rmsd = chosen["seqguided"]["si"]["per_residue_rmsd"]
            ref_atoms = chosen["seqguided"]["ref_atoms"]
            mob_atoms = chosen["seqguided"]["mob_atoms"]
        elif chosen["seqfree"]:
            R = chosen["seqfree"].rotation
            t = chosen["seqfree"].translation
            # Sequence-free aligns CA only usually
            ref_subset = chosen["seqfree"].ref_subset_ca_coords
            mob_subset = chosen["seqfree"].mob_subset_ca_coords_aligned
            pairs = chosen["seqfree"].pairs
            ref_infos = chosen["seqfree"].ref_subset_infos
            mob_infos = chosen["seqfree"].mob_subset_infos

            import numpy as np
            per_res_rmsd = []
            for (i, j) in pairs:
                dist = np.linalg.norm(ref_subset[i] - mob_subset[j])
                per_res_rmsd.append(dist)

            class PseudoAtom:
                def __init__(self, c_name, r_seq, r_ico):
                    self.chain_name = c_name
                    self.res_seq = r_seq
                    self.res_icode = r_ico

            # Since pairs are (i,j) indexes into ref_infos and mob_infos
            ref_atoms = []
            mob_atoms = []
            for (i, j) in pairs:
                ref_atoms.append(PseudoAtom(ref_infos[i].chain_id, ref_infos[i].resseq, ref_infos[i].icode))
                mob_atoms.append(PseudoAtom(mob_infos[j].chain_id, mob_infos[j].resseq, mob_infos[j].icode))

        if R is not None and t is not None:
            # We use gemmi to save the transformed structure
            import numpy as np
            out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, 'clone') else self.mob_struct.copy()

            # Create a lookup mapping for distances
            dist_map = {}
            if mob_atoms and per_res_rmsd is not None:
                for k in range(min(len(mob_atoms), len(per_res_rmsd))):
                    ma = mob_atoms[k]
                    # Handle pseudo atoms and normal atoms uniformly
                    c_name = getattr(ma, 'chain_name', getattr(ma, 'last_chain_name', 'A'))
                    # Usually get_id() for normal atoms
                    if hasattr(ma, 'get_id'):
                        het, r_seq, r_ico = ma.get_parent().get_id()
                    else:
                        r_seq = ma.res_seq
                        r_ico = ma.res_icode

                    key = (c_name, r_seq, r_ico.strip() if hasattr(r_ico, 'strip') else "")
                    dist_map[key] = float(per_res_rmsd[k])

            for model in out_struct:
                for chain in model:
                    for residue in chain:
                        resseq = residue.seqid.num
                        icode = residue.seqid.icode if hasattr(residue.seqid, 'has_icode') and residue.seqid.has_icode() else ""
                        if not icode and hasattr(residue.seqid, 'icode') and residue.seqid.icode != ' ':
                            icode = residue.seqid.icode

                        key = (chain.name, resseq, icode.strip() if hasattr(icode, 'strip') else "")
                        mapped_bfactor = dist_map.get(key, 0.0)

                        for atom in residue:
                            coord = np.array(atom.pos.tolist(), dtype=float)
                            new_coord = (R @ coord) + t
                            atom.pos.x = float(new_coord[0])
                            atom.pos.y = float(new_coord[1])
                            atom.pos.z = float(new_coord[2])

                            # Overwrite B-factor with local deviation distance
                            atom.b_iso = mapped_bfactor

            if filename.lower().endswith(".pdb"):
                out_struct.write_pdb(filename)
            elif filename.lower().endswith(".cif") or filename.lower().endswith(".mmcif"):
                out_struct.make_mmcif_document().write_file(filename)
            else:
                out_struct.write_pdb(filename)

    def get_log(self) -> str:
        lines = []
        lines.append("PDB Aligner Result Log")
        lines.append("="*20)
        lines.append(f"Reference: {self.ref_file}")
        lines.append(f"Mobile: {self.mob_file}")
        lines.append(f"Chosen method: {self._chosen['name']}")
        lines.append(f"RMSD: {self.rmsd:.3f} Å" if self.rmsd is not None else "RMSD: None")
        lines.append(f"Reason: {self._chosen['reason']}")
        return "\n".join(lines)

    def save_log(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.get_log() + "\n")

    def get_structure_based_sequence_alignment(self) -> Optional[tuple]:
        return self.get_sequence_alignment()

    def get_sequence_alignment(self) -> Optional[tuple]:
        if self._chosen["seqguided"]:
            aln = self._chosen["seqguided"]["aln"]
            return aln.seqA, aln.seqB, aln.score
        elif self._chosen["seqfree"]:
            ref_subset = self._chosen["seqfree"].ref_subset_infos
            mob_subset = self._chosen["seqfree"].mob_subset_infos
            pairs = self._chosen["seqfree"].pairs
            ref_aln = ""
            mob_aln = ""
            from Bio.PDB.Polypeptide import protein_letters_3to1
            def to_1l(resname): return protein_letters_3to1.get(resname, 'X')
            ref_idx_to_info = {i: info for i, info in enumerate(ref_subset)}
            mob_idx_to_info = {i: info for i, info in enumerate(mob_subset)}
            matched_ref = {r for r, m in pairs}
            matched_mob = {m for r, m in pairs}
            pair_dict = {r: m for r, m in pairs}
            r_idx, m_idx = 0, 0
            while r_idx < len(ref_subset) or m_idx < len(mob_subset):
                if r_idx in matched_ref and m_idx in matched_mob and pair_dict.get(r_idx) == m_idx:
                    ref_aln += to_1l(ref_idx_to_info[r_idx].resname)
                    mob_aln += to_1l(mob_idx_to_info[m_idx].resname)
                    r_idx += 1; m_idx += 1
                else:
                    if r_idx < len(ref_subset) and r_idx not in matched_ref:
                        ref_aln += to_1l(ref_idx_to_info[r_idx].resname)
                        mob_aln += "-"
                        r_idx += 1
                    elif m_idx < len(mob_subset) and m_idx not in matched_mob:
                        ref_aln += "-"
                        mob_aln += to_1l(mob_idx_to_info[m_idx].resname)
                        m_idx += 1
                    else:
                        ref_aln += "-"; mob_aln += "-"
                        if r_idx < len(ref_subset): r_idx += 1
                        if m_idx < len(mob_subset): m_idx += 1
            return ref_aln, mob_aln, None
        return None

    def get_sequence_alignment_fasta(self) -> str:
        aln_data = self.get_sequence_alignment()
        if not aln_data:
            raise ValueError("No alignment data available.")
        seqA, seqB, _ = aln_data
        fasta = f">Reference_{os.path.basename(self.ref_file)}\n{seqA}\n"
        fasta += f">Mobile_{os.path.basename(self.mob_file)}\n{seqB}\n"
        return fasta

    def save_sequence_alignment_fasta(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.get_sequence_alignment_fasta())

    def print_sequence_alignment(self, interval: int = 10):
        aln_data = self.get_sequence_alignment()
        if not aln_data:
            print("No alignment data available.")
            return
        seqA, seqB, score = aln_data
        ref_id = f"Ref ({os.path.basename(self.ref_file)})"
        mob_id = f"Mob ({os.path.basename(self.mob_file)})"
        def numline(aln: str, interval=10):
            line = [' '] * len(aln)
            c = 0; nxt = interval
            for i, ch in enumerate(aln):
                if ch != '-':
                    c += 1
                    if c == nxt:
                        s = str(nxt)
                        start = max(0, i - len(s) + 1)
                        for k, d in enumerate(s):
                            if start + k < len(aln): line[start + k] = d
                        nxt += interval
            return ''.join(line)
        pad = max(len(ref_id), len(mob_id))
        id1p = ref_id.ljust(pad)
        id2p = mob_id.ljust(pad)
        mp = "Match".ljust(pad)
        from Bio.Align import substitution_matrices
        try: blosum62 = substitution_matrices.load("BLOSUM62")
        except: blosum62 = {}
        match = ""
        for a, b in zip(seqA, seqB):
            if a == b and a != '-': match += "|"
            elif a != '-' and b != '-' and (blosum62.get((a, b), blosum62.get((b, a), 0)) > 0): match += ":"
            elif a == '-' or b == '-': match += " "
            else: match += "."
        loc1, loc2 = numline(seqA, interval), numline(seqB, interval)
        padsp = " " * (pad + 2)
        print("Pairwise Alignment:")
        print(f"{padsp}{loc1}")
        print(f"{id1p}: {seqA}")
        print(f"{mp}  {match}")
        print(f"{id2p}: {seqB}")
        print(f"{padsp}{loc2}")
        if score is not None: print(f"Alignment Score: {score}")

    def get_rmsd_df(self, on: str = 'reference'):
        import pandas as pd
        import numpy as np
        labels, chains, distances = [], [], []
        if self._chosen["seqguided"]:
            atoms = self._chosen["seqguided"]["ref_atoms"] if on == 'reference' else self._chosen["seqguided"]["mob_atoms"]
            per_res_rmsd = self._chosen["seqguided"]["si"]["per_residue_rmsd"]
            for idx, a in enumerate(atoms):
                if hasattr(a, 'chain_name'):
                    chain = getattr(a, 'chain_name', 'A')
                    r_seq = getattr(a, 'res_seq', '1')
                    r_ico = getattr(a, 'res_icode', '')
                    lbl = f"{chain}:{r_seq}{r_ico.strip()}" if str(r_ico).strip() else f"{chain}:{r_seq}"
                else:
                    p = a.get_parent()
                    chain = p.get_parent().id
                    rid = p.get_id()
                    lbl = f"{chain}:{rid[1]}{rid[2].strip()}" if str(rid[2]).strip() else f"{chain}:{rid[1]}"
                labels.append(lbl); chains.append(chain); distances.append(per_res_rmsd[idx])
        elif self._chosen["seqfree"]:
            ref_subset = self._chosen["seqfree"].ref_subset_infos
            mob_subset = self._chosen["seqfree"].mob_subset_infos
            pairs = self._chosen["seqfree"].pairs
            for (i, j) in pairs:
                r_info = ref_subset[i]
                m_info = mob_subset[j]
                diff = self._chosen["seqfree"].ref_subset_ca_coords[i] - self._chosen["seqfree"].mob_subset_ca_coords_aligned[j]
                dist = np.linalg.norm(diff)
                if on == 'reference':
                    lbl = f"{r_info.chain_id}:{r_info.resseq}{r_info.icode.strip()}"
                    chain = r_info.chain_id
                else:
                    lbl = f"{m_info.chain_id}:{m_info.resseq}{m_info.icode.strip()}"
                    chain = m_info.chain_id
                labels.append(lbl); chains.append(chain); distances.append(dist)
        df = pd.DataFrame({"Residue": labels, "Chain": chains, "RMSD": distances})
        return df

    def save_rmsd_csv(self, filename: str, on: str = 'reference'):
        df = self.get_rmsd_df(on=on)
        df.to_csv(filename, index=False)
        if self.verbose: print(f"Saved per-residue RMSD to {filename}")

    def report_peaks(self, on: str = 'reference', top_n: int = 5):
        try:
            df = self.get_rmsd_df(on=on)
            peaks = list(zip(df['Residue'], df['RMSD']))
        except Exception:
            return []
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:top_n] if top_n is not None else peaks
        if top_n is not None:
            print(f"Top {top_n} RMSD Peaks (on {on} numbering):")
            for lbl, dist in top_peaks:
                print(f"  Residue {lbl}: {dist:.3f} Å")
        return top_peaks

    def plot_rmsd(self, filename: str = "rmsd.pdf", style: str = "scientific", on: str = 'reference'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        try: df = self.get_rmsd_df(on=on)
        except Exception:
            print("No data to plot.")
            return
        if df.empty:
            print("No data to plot.")
            return
        with plt.style.context('default'):
            if style == "scientific":
                sns.set_style("whitegrid")
                sns.set_context("paper")
                plt.rcParams.update({
                    "font.family": "serif", "axes.titlesize": 14, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": 300,
                })
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x=df.index, y="RMSD", hue="Chain", marker='o', markersize=4, linestyle='-', linewidth=1, ax=ax)
            n_labels = len(df)
            step = max(1, n_labels // 10)
            ax.set_xticks(range(0, n_labels, step))
            ax.set_xticklabels(df["Residue"].iloc[::step], rotation=45, ha='right')
            ax.set_xlabel(f"Residue ({on.capitalize()})")
            ax.set_ylabel(r"C$\alpha$ RMSD ($\AA$)")
            ax.set_title("Per-Residue Structural Deviation")
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def save_pymol_script(self, filename: str, aligned_mobile_filename: str = "aligned_mobile.pdb"):
        """
        Generates a .pml script for PyMOL to easily visualize the alignment.
        This assumes you have saved the aligned mobile structure using `save_aligned_pdb`.
        """
        ref_basename = os.path.basename(self.ref_file)
        mob_basename = os.path.basename(self.mob_file)

        script = f"""# PyMOL Script for visualizing alignment
# Load structures
load {self.ref_file}, reference
load {aligned_mobile_filename}, mobile

# Hide defaults, show cartoons
hide everything
show cartoon, reference
show cartoon, mobile

# Color structures
color white, reference
color cyan, mobile

# Extract RMSD data and inject it into B-factors
# We map RMSD to the mobile structure for visualization
"""

        # Add B-factor injection logic
        df = self.get_rmsd_df(on="mobile")
        if not df.empty:
            script += "\n# Update B-factors with RMSD values for heatmapping\nalter mobile, b=0.0\n"
            for _, row in df.iterrows():
                try:
                    res_parts = row["Residue"].split(":")
                    if len(res_parts) == 2:
                        chain = res_parts[0]
                        res_id = res_parts[1]

                        # Handle insertion codes
                        import re
                        match = re.match(r"(\d+)([a-zA-Z]*)", res_id)
                        if match:
                            res_num = match.group(1)
                            # PyMOL alter syntax for specific residues
                            script += f"alter mobile and chain {chain} and resi {res_num}, b={row['RMSD']:.3f}\n"
                except Exception:
                    pass

            script += """
# Color by B-factor (RMSD)
spectrum b, blue_white_red, mobile, minimum=0, maximum=10
"""

        script += """
# Center and orient
zoom
center
"""
        with open(filename, "w") as f:
            f.write(script)
        if self.verbose:
            print(f"Saved PyMOL script to {filename}")

    def save_chimerax_script(self, filename: str, aligned_mobile_filename: str = "aligned_mobile.pdb"):
        """
        Generates a .cxc script for ChimeraX to easily visualize the alignment.
        This assumes you have saved the aligned mobile structure using `save_aligned_pdb`.
        """
        script = f"""# ChimeraX Script for visualizing alignment
# Load structures
open {self.ref_file}
open {aligned_mobile_filename}

# Hide atoms, show cartoon
hide atoms
show cartoons

# Color structures
color #1 white
color #2 cyan

# Update B-factors with RMSD values for heatmapping
"""

        df = self.get_rmsd_df(on="mobile")
        if not df.empty:
            for _, row in df.iterrows():
                try:
                    res_parts = row["Residue"].split(":")
                    if len(res_parts) == 2:
                        chain = res_parts[0]
                        res_id = res_parts[1]
                        import re
                        match = re.match(r"(\d+)([a-zA-Z]*)", res_id)
                        if match:
                            res_num = match.group(1)
                            # ChimeraX setattr syntax
                            script += f"setattr #2/{chain}:{res_num} atoms bfactor {row['RMSD']:.3f}\n"
                except Exception:
                    pass

            script += """
# Color by B-factor (RMSD)
color byattribute bfactor #2 palette blue:white:red range 0,10
"""

        script += """
# Center and orient
view
"""
        with open(filename, "w") as f:
            f.write(script)
        if self.verbose:
            print(f"Saved ChimeraX script to {filename}")

    def __repr__(self):
        return f"<AlignmentResult RMSD: {self.rmsd:.3f}Å, Method: {self._chosen['name']}>"

def _process_single_alignment(task_payload: dict):
    """
    Stateless module-level function for multiprocessing batch jobs.
    Avoids pickling heavy objects and drops references to prevent memory leaks.
    """
    try:
        aligner = PDBAligner(
            ref_file=task_payload["ref_file"],
            chains_ref=task_payload["chains_ref"],
            verbose=False
        )
        aligner.add_mobile(task_payload["fpath"], chains=task_payload["chains_mob"])

        kwargs = task_payload.get("kwargs", {})
        res = aligner.align(mode=task_payload["mode"], **kwargs)

        import os
        out_pdb = os.path.join(task_payload["out_dir"], f"aligned_{task_payload['fname']}")
        res.save_aligned_pdb(out_pdb)

        # Explicitly extract scalar metrics to avoid retaining heavy AlignmentResult/Structure references
        r = {"rmsd": float(res.rmsd) if res.rmsd is not None else None,
             "tm_score": float(res.tm_score) if res.tm_score is not None else None,
             "tm_pvalue": float(res.tm_pvalue) if hasattr(res, 'tm_pvalue') and res.tm_pvalue is not None else None,
             "status": "success"}

        del res
        del aligner
        return task_payload["fname"], r
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Batch alignment failed for {task_payload.get('fname', 'Unknown')}", exc_info=True)
        return task_payload["fname"], {"status": "error", "message": str(e)}

class PDBAligner:
    """
    Object-oriented API for protein 3D structural alignment.

    This class supports single and batch alignments, offering various methods:
    - Sequence-guided structural superposition.
    - Sequence-free structural superposition (useful for low sequence identity).

    Attributes:
        verbose (bool): If True, prints status messages during operations.
    """
    def __init__(self, ref_file: Optional[str] = None, chains_ref: Optional[List[Union[str, int]]] = None, verbose: bool = False):
        self.verbose = verbose
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
        self._struct_cache: dict = {}  # absolute path str -> gemmi.Structure; parse-once cache, no disk-change invalidation

        if ref_file:
            self.set_reference(ref_file, chains_ref)

    def add_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure. Alias for set_reference."""
        self.set_reference(ref_file, chains)

    def _fetch_structure(self, file_or_id: str) -> str:
        """Fetches a structure from PDB or AF-DB if a prefix is detected."""
        if file_or_id.lower().startswith("pdb:"):
            pdb_id = file_or_id[4:].strip()
            dest = f"{pdb_id}.cif"
            if not os.path.exists(dest):
                if self.verbose: print(f"Fetching {pdb_id} from RCSB PDB...")
                import requests
                r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif")
                if r.status_code == 200:
                    with open(dest, 'w') as f: f.write(r.text)
                else: raise ValueError(f"Could not fetch PDB {pdb_id}")
            return dest
        elif file_or_id.lower().startswith("af:"):
            af_id = file_or_id[3:].strip()
            dest = f"{af_id}.pdb"
            if not os.path.exists(dest):
                if self.verbose: print(f"Fetching {af_id} from AlphaFold DB...")
                import requests
                r = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v6.pdb")
                if r.status_code == 200:
                    with open(dest, 'w') as f: f.write(r.text)
                else: raise ValueError(f"Could not fetch AlphaFold model {af_id}")
            return dest
        return file_or_id

    def set_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure. Supports pdb:XXXX and af:XXXX fetching."""
        ref_file = self._fetch_structure(ref_file)
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference file not found: {ref_file}")
        ref_file = os.path.abspath(ref_file)
        self.ref_file = ref_file
        self.chains_ref = chains
        if ref_file not in self._struct_cache:
            self._struct_cache[ref_file] = _parse_path(ref_file)
        self.ref_struct = self._struct_cache[ref_file].clone()
        self.ref_seqs, self.ref_lens = extract_sequences_and_lengths(self.ref_struct, os.path.basename(ref_file))
        if self.verbose:
            print(f"Reference set to: {self.ref_file}")
            for ch in (self.chains_ref if self.chains_ref else self.ref_seqs.keys()):
                print(f"  Chain {ch}: {self.ref_lens.get(ch, 0)} aa")

    def add_mobile(self, mob_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the mobile structure to align. Supports pdb:XXXX and af:XXXX fetching."""
        mob_file = self._fetch_structure(mob_file)
        if not os.path.exists(mob_file):
            raise FileNotFoundError(f"Mobile file not found: {mob_file}")
        mob_file = os.path.abspath(mob_file)
        self.mob_file = mob_file
        self.chains_mob = chains
        if mob_file not in self._struct_cache:
            self._struct_cache[mob_file] = _parse_path(mob_file)
        self.mob_struct = self._struct_cache[mob_file].clone()
        self.mob_seqs, self.mob_lens = extract_sequences_and_lengths(self.mob_struct, os.path.basename(mob_file))
        if self.verbose:
            print(f"Mobile set to: {self.mob_file}")
            for ch in (self.chains_mob if self.chains_mob else self.mob_seqs.keys()):
                print(f"  Chain {ch}: {self.mob_lens.get(ch, 0)} aa")
            if self.ref_file:
                print("\nSimilarity Matrix:")
                id_mat, sc_mat = compute_chain_similarity_matrix(self.ref_seqs, self.mob_seqs)
                ref_chains = list(self.ref_seqs.keys())
                mob_chains = list(self.mob_seqs.keys())
                for i, r_ch in enumerate(ref_chains):
                    for j, m_ch in enumerate(mob_chains):
                        ident = id_mat.iloc[i, j] if hasattr(id_mat, "iloc") else id_mat[i, j]
                        if not __import__('numpy').isnan(ident):
                            print(f"  Chain {r_ch} (ref) - Chain {m_ch} (mobile): {ident:.1f}%")

    def set_reference_chains(self, chains: List[Union[str, int]]):
        """Changes the reference chains to use for alignment."""
        if not self.ref_file:
            raise ValueError("Reference structure must be set first.")
        self.chains_ref = chains
        if self.verbose:
            print(f"Reference chains updated to: {chains}")

    def set_mobile_chains(self, chains: List[Union[str, int]]):
        """Changes the mobile chains to use for alignment."""
        if not self.mob_file:
            raise ValueError("Mobile structure must be set first.")
        self.chains_mob = chains
        if self.verbose:
            print(f"Mobile chains updated to: {chains}")

    def align(self, mode: str = "auto", seq_gap_open: float = -10, seq_gap_extend: float = -0.5, atoms: str = "CA", min_plddt: float = 0.0, min_b_factor: float = 0.0, **kwargs):
        """
        Runs the alignment process.

        :param mode: Alignment mode to use. Options include:
            - "auto" (or "Auto (best RMSD)"): Automatically picks the best between seq-guided and seq-free.
            - "seq_guided" (or "Sequence-guided"): Forces sequence-guided alignment.
            - "seq_free_auto" (or "Sequence-free (auto)"): Sequence-free auto-selection between shape/window.
            - "seq_free_shape" (or "Sequence-free (shape)"): Sequence-free using shape matching.
            - "seq_free_window" (or "Sequence-free (window)"): Sequence-free using sliding windows.
        :type mode: str
        :param seq_gap_open: Gap open penalty for sequence alignment (default: -10).
        :type seq_gap_open: float
        :param seq_gap_extend: Gap extension penalty for sequence alignment (default: -0.5).
        :type seq_gap_extend: float
        :param atoms: Atoms to consider during superposition ("CA", "backbone", "all_heavy").
        :type atoms: str
        :param min_plddt: Minimum pLDDT (confidence) threshold to retain an atom (default: 0.0). Filter AF models.
        :type min_plddt: float
        :param min_b_factor: Minimum B-factor to retain an atom (default: 0.0).
        :type min_b_factor: float
        :returns: An AlignmentResult object containing transformation matrices, matched pairs, and metrics.
        :rtype: AlignmentResult
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
                ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(self.ref_struct, ref_chs, self.mob_struct, mob_chs, aln, atoms=atoms, min_b_factor=min_b_factor, min_plddt=min_plddt)
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
                    method=sf_method, atoms=atoms, min_b_factor=min_b_factor, min_plddt=min_plddt, **kwargs
                )
                seqfree = res
            except Exception as e:
                import traceback
                traceback.print_exc()

        if mode in ("auto", "Auto (best RMSD)"):
            best, reason = pick_best_overall(seqguided, seqfree, min_pairs=3)
            if best is None:
                raise AlignmentFailedError("No alignment could be produced. Try different chains or mode.")
            chosen_name = best["name"]
            chosen = dict(name=chosen_name, reason=reason,
                          seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                          seqfree=seqfree if "Sequence-free" in chosen_name else None)
        else:
            chosen = dict(name=mode, reason="Manual mode.",
                          seqguided=seqguided if "seq_guided" in mode or "Sequence-guided" in mode else None,
                          seqfree=seqfree if "seq_free" in mode or "Sequence-free" in mode else None)

        self.last_result = dict(seqguided=seqguided, seqfree=seqfree, chosen=chosen)

        # Calculate restricted lengths for TM-score based on specific chains aligned
        active_ref_lens = {c: self.ref_lens[c] for c in ref_chs if c in self.ref_lens}
        active_mob_lens = {c: self.mob_lens[c] for c in mob_chs if c in self.mob_lens}

        result_obj = AlignmentResult(
            chosen=chosen, seqguided=seqguided, seqfree=seqfree,
            ref_file=self.ref_file, mob_file=self.mob_file,
            mob_struct=self.mob_struct,
            ref_lens=active_ref_lens, mob_lens=active_mob_lens,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"\nAlignment Completed:")
            print(f"  Mode evaluated: {mode}")
            if seqguided:
                print(f"  Sequence-based RMSD: {seqguided['si']['rmsd']:.3f} Å")
            if seqfree:
                print(f"  Sequence-free RMSD: {seqfree.rmsd:.3f} Å")
            print(f"  Chosen method: {chosen['name']}")
            print(f"  Reason: {chosen['reason']}")

        return result_obj

    def find_binder_target_chain(self, binder_chains: List[str], candidate_chains: List[str]) -> str:
        """
        Identifies which candidate chain in the currently loaded mobile structure
        is physically closest to the specified binder chains.
        
        This is useful for multimeric complexes (like AlphaFold predictions) where
        a binder might stochastically attach to any of the symmetric chains.
        """
        if not self.mob_struct or not self.mob_file:
            raise ValueError("Mobile structure must be loaded before finding the binder target.")
            
        import numpy as np
        from scipy.spatial.distance import cdist
        from .core import _extract_ca_infos
        
        # Get coordinates for the binder chains
        binder_infos = _extract_ca_infos(self.mob_struct, chain_filter=binder_chains)
        if not binder_infos:
            raise ValueError(f"Could not extract CA atoms for binder chains {binder_chains} in {os.path.basename(self.mob_file)}")
        binder_coords = np.array([info.coord for info in binder_infos])
        
        min_dist = float('inf')
        best_chain = None
        
        # Check distance to each candidate chain
        for candidate in candidate_chains:
            candidate_infos = _extract_ca_infos(self.mob_struct, chain_filter=[candidate])
            if not candidate_infos:
                if self.verbose:
                    print(f"Warning: Candidate chain {candidate} not found or has no CA atoms.")
                continue
                
            candidate_coords = np.array([info.coord for info in candidate_infos])
            
            # Calculate all pairwise distances between binder CA atoms and candidate CA atoms
            distances = cdist(binder_coords, candidate_coords)
            
            # Find the minimum distance
            current_min = np.min(distances)
            
            if self.verbose:
                print(f"  Minimum distance to chain {candidate}: {current_min:.2f} Å")
                
            if current_min < min_dist:
                min_dist = current_min
                best_chain = candidate
                
        if best_chain is None:
            raise ValueError(f"Could not find any valid candidate chains from {candidate_chains} in {os.path.basename(self.mob_file)}")
            
        if self.verbose:
            print(f"Selected chain {best_chain} as the target for binder {binder_chains} (distance: {min_dist:.2f} Å)")
            
        return best_chain

    def align_with_binder(self, binder_chains: List[str], candidate_chains: List[str], 
                          mode: str = "auto", seq_gap_open: float = -10, seq_gap_extend: float = -0.5, 
                          atoms: str = "CA", **kwargs) -> AlignmentResult:
        """
        Calculates the target chain that the binder is bound to, sets it as the active mobile chain,
        and performs the alignment.
        """
        if self.verbose:
            print(f"Looking for target chain among {candidate_chains} bound by {binder_chains}...")
            
        best_chain = self.find_binder_target_chain(binder_chains, candidate_chains)
        self.set_mobile_chains([best_chain])
        
        return self.align(mode=mode, seq_gap_open=seq_gap_open, seq_gap_extend=seq_gap_extend, atoms=atoms, **kwargs)

    def batch_align_iter(self, mob_dir: str, out_dir: str, mode: str = "auto", workers: int = 1, **kwargs):
        """
        Generator that aligns a directory of PDBs and yields results sequentially or via multiprocessing.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor

        if not self.ref_file:
            raise ValueError("Reference structure must be set for batch alignment.")

        os.makedirs(out_dir, exist_ok=True)
        tasks = []

        for fname in os.listdir(mob_dir):
            if fname.lower().endswith((".pdb", ".cif", ".mmcif")):
                fpath = os.path.join(mob_dir, fname)
                tasks.append({
                    "ref_file": self.ref_file,
                    "chains_ref": self.chains_ref,
                    "chains_mob": self.chains_mob,
                    "fpath": fpath,
                    "fname": fname,
                    "mode": mode,
                    "out_dir": out_dir,
                    "kwargs": kwargs
                })

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process_single_alignment, task) for task in tasks]
                for future in futures:
                    fname, r = future.result()
                    if self.verbose:
                        status = r.get("status")
                        if status == "success":
                            print(f"Batch processed: {fname} (RMSD: {r.get('rmsd')})")
                        else:
                            print(f"Failed to process {fname}: {r.get('message')}")
                    yield fname, r
        else:
            for task in tasks:
                fname, r = _process_single_alignment(task)
                if self.verbose:
                    status = r.get("status")
                    if status == "success":
                        print(f"Batch processed: {fname} (RMSD: {r.get('rmsd')})")
                    else:
                        print(f"Failed to process {fname}: {r.get('message')}")
                yield fname, r

    def batch_align(self, mob_dir: str, out_dir: str, mode: str = "auto", workers: int = 1, **kwargs):
        """
        Aligns a directory of PDBs against the current reference structure.
        Returns a Pandas DataFrame.
        """
        import pandas as pd
        results = []
        for fname, r in self.batch_align_iter(mob_dir, out_dir, mode, workers, **kwargs):
            r["filename"] = fname
            results.append(r)
        return pd.DataFrame(results)

    def get_ensemble_statistics(self, df) -> dict:
        """
        Calculates summary statistics for a batch alignment ensemble (DataFrame).
        """
        if df.empty:
            return {}

        stats = {}
        if "rmsd" in df.columns:
            valid_rmsd = df["rmsd"].dropna()
            if not valid_rmsd.empty:
                stats["rmsd_mean"] = valid_rmsd.mean()
                stats["rmsd_median"] = valid_rmsd.median()
                stats["rmsd_std"] = valid_rmsd.std()

        if "tm_score" in df.columns:
            valid_tm = df["tm_score"].dropna()
            if not valid_tm.empty:
                stats["tm_score_mean"] = valid_tm.mean()
                stats["tm_score_median"] = valid_tm.median()
                stats["tm_score_max"] = valid_tm.max()

        return stats

    def get_general_sequence_alignment(self, ref_chain: str, mob_chain: str, gap_open: float = -10.0, gap_extend: float = -0.5) -> Optional[tuple]:
        """
        Computes a classic sequence alignment between two specified chains.
        Returns a tuple (seqA_aln, seqB_aln, score).
        """
        if not self.ref_file or not self.mob_file:
            raise ValueError("Reference and mobile structures must be loaded first.")

        if ref_chain not in self.ref_seqs:
            raise ChainNotFoundError(f"Reference chain '{ref_chain}' not found.")
        if mob_chain not in self.mob_seqs:
            raise ChainNotFoundError(f"Mobile chain '{mob_chain}' not found.")

        seqA = str(self.ref_seqs[ref_chain].seq)
        seqB = str(self.mob_seqs[mob_chain].seq)

        aln = perform_sequence_alignment(seqA, seqB, gap_open, gap_extend)
        if aln:
            return aln.seqA, aln.seqB, aln.score
        return None

    def print_general_sequence_alignment(self, ref_chain: str, mob_chain: str, interval: int = 10, gap_open: float = -10.0, gap_extend: float = -0.5):
        """
        Prints a classic sequence alignment directly between two specified chains.
        """
        aln_data = self.get_general_sequence_alignment(ref_chain, mob_chain, gap_open, gap_extend)
        if not aln_data:
            print("No general alignment data could be produced.")
            return

        seqA, seqB, score = aln_data
        ref_id = f"Ref ({os.path.basename(self.ref_file)} - {ref_chain})"
        mob_id = f"Mob ({os.path.basename(self.mob_file)} - {mob_chain})"

        def numline(aln: str, interval=10):
            line = [' '] * len(aln)
            c = 0
            nxt = interval
            for i, ch in enumerate(aln):
                if ch != '-':
                    c += 1
                    if c == nxt:
                        s = str(nxt)
                        start = max(0, i - len(s) + 1)
                        for k, d in enumerate(s):
                            if start + k < len(aln):
                                line[start + k] = d
                        nxt += interval
            return ''.join(line)

        pad = max(len(ref_id), len(mob_id))
        id1p = ref_id.ljust(pad)
        id2p = mob_id.ljust(pad)
        mp = "Match".ljust(pad)

        from Bio.Align import substitution_matrices
        try:
            blosum62 = substitution_matrices.load("BLOSUM62")
        except:
            blosum62 = {}

        match = ""
        for a, b in zip(seqA, seqB):
            if a == b and a != '-':
                match += "|"
            elif a != '-' and b != '-' and (blosum62.get((a, b), blosum62.get((b, a), 0)) > 0):
                match += ":"
            elif a == '-' or b == '-':
                match += " "
            else:
                match += "."

        loc1, loc2 = numline(seqA, interval), numline(seqB, interval)
        padsp = " " * (pad + 2)

        print("General Pairwise Alignment:")
        print(f"{padsp}{loc1}")
        print(f"{id1p}: {seqA}")
        print(f"{mp}  {match}")
        print(f"{id2p}: {seqB}")
        print(f"{padsp}{loc2}")
        print(f"Alignment Score: {score}")

    def get_similarity_matrix(self):
        """Returns the chain similarity matrices (Identity and BLOSUM62 scores)."""
        if not self.ref_file or not self.mob_file:
            raise ValueError("Both reference and mobile structures must be set.")
        return compute_chain_similarity_matrix(self.ref_seqs, self.mob_seqs)

    def save_aligned_pdb(self, filename: str, subset_only: bool = False):
        """Saves the aligned mobile structure to a PDB file. Maps alignment distance into B-factor."""
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        chosen = self.last_result["chosen"]
        R = None
        t = None
        per_res_rmsd = None
        ref_atoms = []
        mob_atoms = []

        if chosen["seqguided"]:
            R = chosen["seqguided"]["si"]["rotation"]
            t = chosen["seqguided"]["si"]["translation"]
            per_res_rmsd = chosen["seqguided"]["si"]["per_residue_rmsd"]
            ref_atoms = chosen["seqguided"]["ref_atoms"]
            mob_atoms = chosen["seqguided"]["mob_atoms"]
        elif chosen["seqfree"]:
            R = chosen["seqfree"].rotation
            t = chosen["seqfree"].translation
            # Sequence-free aligns CA only usually
            ref_subset = chosen["seqfree"].ref_subset_ca_coords
            mob_subset = chosen["seqfree"].mob_subset_ca_coords_aligned
            pairs = chosen["seqfree"].pairs
            ref_infos = chosen["seqfree"].ref_subset_infos
            mob_infos = chosen["seqfree"].mob_subset_infos

            import numpy as np
            per_res_rmsd = []
            for (i, j) in pairs:
                dist = np.linalg.norm(ref_subset[i] - mob_subset[j])
                per_res_rmsd.append(dist)

            class PseudoAtom:
                def __init__(self, c_name, r_seq, r_ico):
                    self.chain_name = c_name
                    self.res_seq = r_seq
                    self.res_icode = r_ico

            # Since pairs are (i,j) indexes into ref_infos and mob_infos
            ref_atoms = []
            mob_atoms = []
            for (i, j) in pairs:
                ref_atoms.append(PseudoAtom(ref_infos[i].chain_id, ref_infos[i].resseq, ref_infos[i].icode))
                mob_atoms.append(PseudoAtom(mob_infos[j].chain_id, mob_infos[j].resseq, mob_infos[j].icode))

        if R is not None and t is not None:
            # We use gemmi to save the transformed structure
            import numpy as np
            out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, 'clone') else self.mob_struct.copy()

            # Create a lookup mapping for distances
            dist_map = {}
            if mob_atoms and per_res_rmsd is not None:
                for k in range(min(len(mob_atoms), len(per_res_rmsd))):
                    ma = mob_atoms[k]
                    # Handle pseudo atoms and normal atoms uniformly
                    c_name = getattr(ma, 'chain_name', getattr(ma, 'last_chain_name', 'A'))
                    # Usually get_id() for normal atoms
                    if hasattr(ma, 'get_id'):
                        het, r_seq, r_ico = ma.get_parent().get_id()
                    else:
                        r_seq = ma.res_seq
                        r_ico = ma.res_icode

                    key = (c_name, r_seq, r_ico.strip() if hasattr(r_ico, 'strip') else "")
                    dist_map[key] = float(per_res_rmsd[k])

            for model in out_struct:
                for chain in model:
                    for residue in chain:
                        resseq = residue.seqid.num
                        icode = residue.seqid.icode if hasattr(residue.seqid, 'has_icode') and residue.seqid.has_icode() else ""
                        if not icode and hasattr(residue.seqid, 'icode') and residue.seqid.icode != ' ':
                            icode = residue.seqid.icode

                        key = (chain.name, resseq, icode.strip() if hasattr(icode, 'strip') else "")
                        mapped_bfactor = dist_map.get(key, 0.0)

                        for atom in residue:
                            coord = np.array(atom.pos.tolist(), dtype=float)
                            new_coord = (R @ coord) + t
                            atom.pos.x = float(new_coord[0])
                            atom.pos.y = float(new_coord[1])
                            atom.pos.z = float(new_coord[2])

                            # Overwrite B-factor with local deviation distance
                            atom.b_iso = mapped_bfactor

            if filename.lower().endswith(".pdb"):
                out_struct.write_pdb(filename)
            elif filename.lower().endswith(".cif") or filename.lower().endswith(".mmcif"):
                out_struct.make_mmcif_document().write_file(filename)
            else:
                out_struct.write_pdb(filename)

    def get_log(self) -> str:
        """Returns the alignment log summary as a string."""
        if not self._chosen:
            raise ValueError("No alignment results available. Run align() first.")
        lines = []
        lines.append("PDB Aligner Result Log")
        lines.append("="*20)
        lines.append(f"Reference: {self.ref_file}")
        lines.append(f"Mobile: {self.mob_file}")
        chosen = self._chosen
        lines.append(f"Chosen method: {chosen['name']}")
        lines.append(f"RMSD: {self.get_rmsd():.3f} Å" if self.get_rmsd() is not None else "RMSD: None")
        lines.append(f"Reason: {chosen['reason']}")
        return "\n".join(lines)

    def save_log(self, filename: str):
        """Saves alignment log summary."""
        with open(filename, "w") as f:
            f.write(self.get_log() + "\n")
