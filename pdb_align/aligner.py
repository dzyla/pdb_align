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

        if ref_file:
            self.set_reference(ref_file, chains_ref)

    def add_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure. Alias for set_reference."""
        self.set_reference(ref_file, chains)

    def set_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure."""
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference file not found: {ref_file}")
        self.ref_file = ref_file
        self.chains_ref = chains
        self.ref_struct = _parse_path(ref_file)
        self.ref_seqs, self.ref_lens = extract_sequences_and_lengths(self.ref_struct, os.path.basename(ref_file))
        if self.verbose:
            print(f"Reference set to: {self.ref_file}")
            for ch in (self.chains_ref if self.chains_ref else self.ref_seqs.keys()):
                print(f"  Chain {ch}: {self.ref_lens.get(ch, 0)} aa")

    def add_mobile(self, mob_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the mobile structure to align."""
        if not os.path.exists(mob_file):
            raise FileNotFoundError(f"Mobile file not found: {mob_file}")
        self.mob_file = mob_file
        self.chains_mob = chains
        self.mob_struct = _parse_path(mob_file)
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

        if self.verbose:
            print(f"\nAlignment Completed:")
            print(f"  Mode evaluated: {mode}")
            if seqguided:
                print(f"  Sequence-based RMSD: {seqguided['si']['rmsd']:.3f} Å")
            if seqfree:
                print(f"  Sequence-free RMSD: {seqfree.rmsd:.3f} Å")
            print(f"  Chosen method: {chosen['name']}")
            print(f"  Reason: {chosen['reason']}")

        return self.last_result

    def batch_align(self, mob_dir: str, out_dir: str, mode: str = "auto", **kwargs):
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
                    if self.verbose:
                        print(f"Batch processing: {fname}")
                    res = self.align(mode=mode, **kwargs)
                    rmsd = self.get_rmsd()
                    results[fname] = {"rmsd": rmsd, "status": "success"}
                    out_pdb = os.path.join(out_dir, f"aligned_{fname}")
                    self.save_aligned_pdb(out_pdb)
                except Exception as e:
                    results[fname] = {"status": "error", "message": str(e)}
                    if self.verbose:
                        print(f"Failed to process {fname}: {e}")
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

    def get_structure_based_sequence_alignment(self) -> Optional[tuple]:
        """Alias for get_sequence_alignment(). Returns the structure-derived sequence alignment (seqA, seqB, score)."""
        return self.get_sequence_alignment()

    def get_sequence_alignment(self) -> Optional[tuple]:
        """Returns structure-derived sequence alignment tuple (seqA, seqB, score)."""
        if not self.last_result:
            return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]:
            aln = chosen["seqguided"]["aln"]
            return aln.seqA, aln.seqB, aln.score
        elif chosen["seqfree"]:
            # Need to synthesize an alignment string from pairs
            ref_subset = chosen["seqfree"].ref_subset_infos
            mob_subset = chosen["seqfree"].mob_subset_infos
            pairs = chosen["seqfree"].pairs

            # This returns the synthesized alignment as shown in Streamlit
            ref_aln = ""
            mob_aln = ""

            from Bio.PDB.Polypeptide import protein_letters_3to1
            def to_1l(resname):
                return protein_letters_3to1.get(resname, 'X')

            ref_idx_to_info = {i: info for i, info in enumerate(ref_subset)}
            mob_idx_to_info = {i: info for i, info in enumerate(mob_subset)}

            matched_ref = {r for r, m in pairs}
            matched_mob = {m for r, m in pairs}

            pair_dict = {r: m for r, m in pairs}

            r_idx = 0
            m_idx = 0

            while r_idx < len(ref_subset) or m_idx < len(mob_subset):
                if r_idx in matched_ref and m_idx in matched_mob and pair_dict.get(r_idx) == m_idx:
                    ref_aln += to_1l(ref_idx_to_info[r_idx].resname)
                    mob_aln += to_1l(mob_idx_to_info[m_idx].resname)
                    r_idx += 1
                    m_idx += 1
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
                        ref_aln += "-"
                        mob_aln += "-"
                        if r_idx < len(ref_subset): r_idx += 1
                        if m_idx < len(mob_subset): m_idx += 1

            return ref_aln, mob_aln, None
        return None

    def get_sequence_alignment_fasta(self) -> str:
        """Returns sequence alignment in FASTA format."""
        aln_data = self.get_sequence_alignment()
        if not aln_data:
            raise ValueError("No alignment data available.")
        seqA, seqB, _ = aln_data

        fasta = f">Reference_{os.path.basename(self.ref_file)}\n{seqA}\n"
        fasta += f">Mobile_{os.path.basename(self.mob_file)}\n{seqB}\n"
        return fasta

    def save_sequence_alignment_fasta(self, filename: str):
        """Saves sequence alignment to a file in FASTA format."""
        fasta = self.get_sequence_alignment_fasta()
        with open(filename, "w") as f:
            f.write(fasta)

    def print_sequence_alignment(self, interval: int = 10):
        """Prints the sequence alignment in a formatted way."""
        aln_data = self.get_sequence_alignment()
        if not aln_data:
            print("No alignment data available.")
            return

        seqA, seqB, score = aln_data
        ref_id = f"Ref ({os.path.basename(self.ref_file)})"
        mob_id = f"Mob ({os.path.basename(self.mob_file)})"

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

        print("Pairwise Alignment:")
        print(f"{padsp}{loc1}")
        print(f"{id1p}: {seqA}")
        print(f"{mp}  {match}")
        print(f"{id2p}: {seqB}")
        print(f"{padsp}{loc2}")
        if score is not None:
            print(f"Alignment Score: {score}")

    def get_general_sequence_alignment(self, ref_chain: str, mob_chain: str, gap_open: float = -10.0, gap_extend: float = -0.5) -> Optional[tuple]:
        """
        Computes a classic sequence alignment between two specified chains.
        Returns a tuple (seqA_aln, seqB_aln, score).
        """
        if not self.ref_file or not self.mob_file:
            raise ValueError("Reference and mobile structures must be loaded first.")

        if ref_chain not in self.ref_seqs:
            raise ValueError(f"Reference chain '{ref_chain}' not found.")
        if mob_chain not in self.mob_seqs:
            raise ValueError(f"Mobile chain '{mob_chain}' not found.")

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

    def get_rotation(self) -> Optional['numpy.ndarray']:
        """Returns the rotation matrix of the best alignment."""
        if not self.last_result: return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]: return chosen["seqguided"]["si"]["rotation"]
        elif chosen["seqfree"]: return chosen["seqfree"].rotation
        return None

    def get_translation(self) -> Optional['numpy.ndarray']:
        """Returns the translation vector of the best alignment."""
        if not self.last_result: return None
        chosen = self.last_result["chosen"]
        if chosen["seqguided"]: return chosen["seqguided"]["si"]["translation"]
        elif chosen["seqfree"]: return chosen["seqfree"].translation
        return None

    def get_rmsd_df(self, on: str = 'reference'):
        """
        Returns a Pandas DataFrame containing per-residue RMSD.

        on: 'reference' or 'mobile' numbering.
        """
        import pandas as pd
        import numpy as np

        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        chosen = self.last_result["chosen"]

        labels = []
        chains = []
        distances = []

        if chosen["seqguided"]:
            atoms = chosen["seqguided"]["ref_atoms"] if on == 'reference' else chosen["seqguided"]["mob_atoms"]
            per_res_rmsd = chosen["seqguided"]["si"]["per_residue_rmsd"]

            for idx, a in enumerate(atoms):
                p = a.get_parent()
                chain = p.get_parent().id
                rid = p.get_id()
                lbl = f"{chain}:{rid[1]}{rid[2].strip()}" if str(rid[2]).strip() else f"{chain}:{rid[1]}"
                labels.append(lbl)
                chains.append(chain)
                distances.append(per_res_rmsd[idx])

        elif chosen["seqfree"]:
            ref_subset = chosen["seqfree"].ref_subset_infos
            mob_subset = chosen["seqfree"].mob_subset_infos
            pairs = chosen["seqfree"].pairs

            for (i, j) in pairs:
                r_info = ref_subset[i]
                m_info = mob_subset[j]

                diff = chosen["seqfree"].ref_subset_ca_coords[i] - chosen["seqfree"].mob_subset_ca_coords_aligned[j]
                dist = np.linalg.norm(diff)

                if on == 'reference':
                    lbl = f"{r_info.chain_id}:{r_info.resseq}{r_info.icode.strip()}"
                    chain = r_info.chain_id
                else:
                    lbl = f"{m_info.chain_id}:{m_info.resseq}{m_info.icode.strip()}"
                    chain = m_info.chain_id

                labels.append(lbl)
                chains.append(chain)
                distances.append(dist)

        df = pd.DataFrame({
            "Residue": labels,
            "Chain": chains,
            "RMSD": distances
        })
        return df

    def save_rmsd_csv(self, filename: str, on: str = 'reference'):
        """Saves the per-C-alpha RMSD to a CSV file."""
        df = self.get_rmsd_df(on=on)
        df.to_csv(filename, index=False)
        if self.verbose:
            print(f"Saved per-residue RMSD to {filename}")

    def report_peaks(self, on: str = 'reference', top_n: int = 5):
        """
        Reports the largest C-alpha RMSD peaks between the aligned structures.

        on: "reference" or "mobile" numbering.
        """
        import numpy as np

        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        chosen = self.last_result["chosen"]

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
        """
        Plots the per-residue C-alpha RMSD.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        # Get the DataFrame data
        try:
            df = self.get_rmsd_df(on=on)
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
                    "font.family": "serif",
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.dpi": 300,
                })

            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot as line + markers with hue by chain
            sns.lineplot(data=df, x=df.index, y="RMSD", hue="Chain", marker='o', markersize=4, linestyle='-', linewidth=1, ax=ax)

            # Formatting X-axis
            # Only show every Nth label to avoid crowding
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

    def get_log(self) -> str:
        """Returns the alignment log summary as a string."""
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")
        lines = []
        lines.append("PDB Aligner Result Log")
        lines.append("="*20)
        lines.append(f"Reference: {self.ref_file}")
        lines.append(f"Mobile: {self.mob_file}")
        chosen = self.last_result["chosen"]
        lines.append(f"Chosen method: {chosen['name']}")
        lines.append(f"RMSD: {self.get_rmsd():.3f} Å" if self.get_rmsd() is not None else "RMSD: None")
        lines.append(f"Reason: {chosen['reason']}")
        return "\n".join(lines)

    def save_log(self, filename: str):
        """Saves alignment log summary."""
        with open(filename, "w") as f:
            f.write(self.get_log() + "\n")
