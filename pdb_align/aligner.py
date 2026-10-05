import math
import os
import io
import logging
import tempfile
import warnings
from typing import Optional, List, Union
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

from .core import (
    Selection, extract_sequences_and_lengths, _parse_chain_selector, _parse_path,
    pairs_from_alignment, paired_atoms, perform_sequence_alignment,
    pick_best_overall, select_residues, sequence_independent_alignment_joined_v2,
    superimpose_atoms, compute_chain_similarity_matrix, compute_contact_overlap,
    _detect_hinges, _kabsch,
)
from .exceptions import ParsingError, ChainNotFoundError
from .metrics import compute_d0, tm_optimal_superposition, calculate_lddt

class AlignmentFailedError(ValueError):
    """Raised when the alignment cannot produce a usable result.

    Subclasses :class:`ValueError` so a caller can catch every "this request
    cannot be answered" condition — an empty selection, an unknown chain, a
    superposition with no residues — with one ``except ValueError``.
    """

@dataclass
class DomainResult:
    """Per-domain alignment result produced by mode='flexible'."""
    domain_id: int
    chain_id: str
    residue_start: int
    residue_end: int
    n_residues: int
    rmsd: float
    rotation: "np.ndarray"   # (3, 3)
    translation: "np.ndarray"  # (3,)

class AlignmentResult:
    """
    Encapsulates the result of a structural alignment, providing a clean, stateless
    interface to properties like RMSD, matrices, and plotting tools.
    """
    def __init__(self, chosen: dict, seqguided: dict, seqfree: dict, ref_file: str,
                 mob_file: str, mob_struct, ref_lens: dict, mob_lens: dict,
                 verbose: bool = False, domains: Optional[List["DomainResult"]] = None):
        self._chosen = chosen
        self._seqguided = seqguided
        self._seqfree = seqfree
        self.ref_file = ref_file
        self.mob_file = mob_file
        self.mob_struct = mob_struct
        self.ref_lens = ref_lens
        self.mob_lens = mob_lens
        self.verbose = verbose
        self.domains = domains  # List[DomainResult] or None
        self.strategy = "single"
        self.chain_mapping = None
        self._per_chain = None  # optional DataFrame set by multi-chain path
        self._tm_cache = {}     # normalize_by -> TM-optimal score
        self._lddt_cache = None
        self._contact_overlap_cache = None
        # The selections the comparison was actually made on (post-filter), so
        # every reported length, coverage and normalisation refers to them.
        self.ref_selection: Optional[Selection] = None
        self.mob_selection: Optional[Selection] = None

    @property
    def per_chain(self):
        import pandas as pd
        if self._per_chain is not None:
            return self._per_chain
        return pd.DataFrame(columns=["chain_ref", "chain_mob", "n_residues", "rmsd"])

    def summary_stats(self) -> dict:
        gdt = None
        sg = self._chosen.get("seqguided")
        sf = self._chosen.get("seqfree")
        if sg and sg.get("si"):
            gdt = sg["si"].get("gdt_ts")
        elif sf:
            gdt = getattr(sf, "gdt_ts", None)
        try:
            n_aligned = len(self.get_rmsd_df())
        except Exception:
            n_aligned = None
        l_ref = sum(self.ref_lens.values()) or None
        coverage = (n_aligned / l_ref * 100.0) if (n_aligned and l_ref) else None
        mapping = None
        if self.chain_mapping is not None:
            mapping = [
                {"ref": p[0], "mob": p[1], "identity": round(float(p[2]), 1)}
                for p in self.chain_mapping.pairs
            ]
        return {
            "method": self._chosen.get("name"),
            "strategy": self.strategy,
            "reason": self._chosen.get("reason"),
            "rmsd": self.rmsd,
            "tm_score": self.tm_score,
            "tm_score_min": self.get_tm_score("min"),
            "tm_pvalue": self.tm_pvalue,
            "gdt_ts": gdt,
            "lddt_ca": self.lddt_ca,
            "n_aligned": n_aligned,
            "coverage_pct": coverage,
            "chain_mapping": mapping,
            "ref_file": self.ref_file,
            "mob_file": self.mob_file,
        }

    @property
    def quality(self):
        """Plain-language :class:`~pdb_align.interpretation.AlignmentQuality`.

        A lazy, pure interpretation of the numbers already on this result:
        quality band, one-line verdict, confidence, flagged flexible regions,
        and warnings. Shared by :meth:`report`, :meth:`to_dict`, and the CLI.
        """
        from pdb_align.interpretation import assess
        s = self.summary_stats()
        try:
            df = self.get_rmsd_df(on="reference")
            per_residue = list(zip(df["Chain"], df["Residue"], df["RMSD"]))
        except Exception:
            per_residue = []
        # Compare the two raw candidates (both are populated even for the loser),
        # so a large seq-guided vs seq-free disagreement lowers confidence.
        cand = []
        if isinstance(self._seqguided, dict) and isinstance(self._seqguided.get("si"), dict):
            cand.append(self._seqguided["si"].get("rmsd"))
        if self._seqfree is not None:
            cand.append(getattr(self._seqfree, "rmsd", None))
        hinge = None
        if self.domains:
            hinge = [(d.chain_id, f"{d.chain_id}:{d.residue_start}",
                      f"{d.chain_id}:{d.residue_end}") for d in self.domains]
        return assess(
            tm_score=s.get("tm_score"), rmsd=s.get("rmsd"),
            coverage_pct=s.get("coverage_pct"), n_aligned=s.get("n_aligned"),
            per_residue=per_residue, chain_mapping=s.get("chain_mapping"),
            candidate_rmsds=cand, hinge_regions=hinge,
            tm_pvalue=self.tm_pvalue,
        )

    def to_dict(self) -> dict:
        d = self.summary_stats()
        d["per_chain"] = self.per_chain.to_dict(orient="records")
        d["quality"] = self.quality.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, default=float)

    def report(self, fmt: str = "text") -> str:
        if fmt == "json":
            return self.to_json()
        if fmt != "text":
            raise ValueError("fmt must be 'text' or 'json'")
        s = self.summary_stats()
        def fmt_num(v, spec):
            return format(v, spec) if v is not None else "n/a"
        lines = []
        lines.append("=" * 52)
        lines.append(" pdb_align - structural comparison")
        lines.append("=" * 52)
        import os
        lines.append(f" Reference : {os.path.basename(s['ref_file'])}")
        lines.append(f" Mobile    : {os.path.basename(s['mob_file'])}")
        lines.append(f" Method    : {s['method']}  (strategy: {s['strategy']})")
        lines.append("-" * 52)
        lines.append(f" RMSD          : {fmt_num(s['rmsd'], '.3f')} A")
        lines.append(f" TM-score      : {fmt_num(s['tm_score'], '.4f')}"
                     + (f"  (p = {s['tm_pvalue']:.2g})" if s.get('tm_pvalue') is not None else ""))
        lines.append(f" GDT_TS*       : {fmt_num(s['gdt_ts'], '.2f')}")
        lines.append(f" lDDT-Ca       : {fmt_num(s.get('lddt_ca'), '.3f')}  (matched residues)")
        lines.append(f" Aligned res   : {s['n_aligned'] if s['n_aligned'] is not None else 'n/a'}")
        lines.append(f" Coverage      : {fmt_num(s['coverage_pct'], '.1f')} %")
        if s["chain_mapping"]:
            lines.append("-" * 52)
            lines.append(" Chain mapping (ref -> mob, %id):")
            for m in s["chain_mapping"]:
                lines.append(f"   {m['ref']} -> {m['mob']}  ({m['identity']:.1f}%)")
        df = self.per_chain
        if not df.empty:
            lines.append("-" * 52)
            lines.append(" Per-chain RMSD:")
            for _, r in df.iterrows():
                lines.append(f"   {r['chain_ref']}->{r['chain_mob']}: "
                             f"{r['rmsd']:.3f} A ({int(r['n_residues'])} res)")
        q = self.quality
        lines.append("-" * 52)
        lines.append(f" Quality   : {q.band.upper()}  (confidence: {q.confidence})")
        lines.append(f" Verdict   : {q.verdict}")
        if q.flagged_regions:
            lines.append(" Flagged regions:")
            for fr in q.flagged_regions[:5]:
                lines.append(f"   {fr.chain} {fr.start_label}..{fr.end_label} "
                             f"({fr.kind}, max {fr.max_rmsd:.1f} A)")
        for w in q.warnings:
            lines.append(f" ! {w}")
        lines.append("-" * 52)
        lines.append(" * GDT_TS: single superposition, normalized by the")
        lines.append("   reference selection length (lower bound on CASP GDT).")
        lines.append(" Refs: TM-score Zhang&Skolnick'04 (TM-optimal superpos.);")
        lines.append("   p-value EVD Xu&Zhang'10; lDDT Mariani'13.")
        lines.append("=" * 52)
        return "\n".join(lines)

    @property
    def rmsd(self) -> Optional[float]:
        # Flexible mode: combine the per-domain RMSDs over all domain residues.
        # RMSDs are root-mean-*square* deviations, so they combine in
        # quadrature; an arithmetic mean of 1 A and 5 A reports 3.0 A where the
        # actual deviation over the union of both domains is 3.6 A.
        if self.domains is not None:
            total = sum(d.n_residues for d in self.domains)
            if total == 0:
                return None
            ss = sum((d.rmsd ** 2) * d.n_residues for d in self.domains)
            return math.sqrt(ss / total)
        if self._chosen["seqguided"]:
            return self._chosen["seqguided"]["si"]["rmsd"]
        elif self._chosen["seqfree"]:
            return self._chosen["seqfree"].rmsd
        return None

    @property
    def tm_score(self) -> Optional[float]:
        """Wrapper for get_tm_score('reference') to maintain backwards compatibility."""
        return self.get_tm_score(normalize_by='reference')

    def _matched_ca_coords(self):
        """Matched CA coordinate pairs (P_ref, Q_mob) for the chosen alignment.

        Q is returned in whatever frame is at hand (original or aligned) —
        callers that superpose re-derive the rigid transform themselves.
        Returns (None, None) when no matched CAs exist.
        """
        import numpy as np
        if self._chosen["seqguided"]:
            si = self._chosen["seqguided"]["si"]
            P, Q = si.get("ca_ref"), si.get("ca_mob")
            if P is None or len(P) == 0:
                return None, None
            return np.asarray(P), np.asarray(Q)
        elif self._chosen["seqfree"]:
            sf = self._chosen["seqfree"]
            if not sf.pairs:
                return None, None
            P = np.array([sf.ref_subset_ca_coords[i] for (i, j) in sf.pairs])
            Q = np.array([sf.mob_subset_ca_coords_aligned[j] for (i, j) in sf.pairs])
            return P, Q
        return None, None

    def get_tm_score(self, normalize_by: str = 'reference') -> Optional[float]:
        """
        TM-score of the alignment (Zhang & Skolnick 2004).

        Reported as TM-align/TM-score do: the *maximum* TM-score over rigid
        superpositions for the matched residue correspondence (via
        :func:`pdb_align.metrics.tm_optimal_superposition`), not the TM-score
        of the RMSD-optimal superposition — the latter systematically
        underestimates TM whenever flexible tails drag the least-squares fit.

        normalize_by: 'reference', 'mobile', or 'min'.

        Warning: Standard TM-align always normalizes by the length of the *reference*
        protein. Changing the normalization length breaks TM-score comparability
        across different targets.
        """
        if normalize_by in self._tm_cache:
            return self._tm_cache[normalize_by]

        L_ref = sum(self.ref_lens.values())
        L_mob = sum(self.mob_lens.values())
        if normalize_by == 'reference': L = L_ref
        elif normalize_by == 'mobile': L = L_mob
        elif normalize_by == 'min': L = min(L_ref, L_mob)
        else: raise ValueError("normalize_by must be 'reference', 'mobile', or 'min'")
        if L <= 15:
            return None

        P, Q = self._matched_ca_coords()
        if P is None or len(P) == 0:
            return None
        tm, _R, _t = tm_optimal_superposition(P, Q, L)
        self._tm_cache[normalize_by] = float(tm)
        return float(tm)

    @property
    def lddt_ca(self) -> Optional[float]:
        """lDDT-Ca over the matched residues (Mariani et al. 2013).

        Superposition-free: compares internal CA-CA distance matrices with a
        15 A inclusion radius and 0.5/1/2/4 A tolerance thresholds. Computed
        over matched residues only — read it together with ``coverage_pct``.
        """
        if self._lddt_cache is not None:
            return self._lddt_cache
        P, Q = self._matched_ca_coords()
        if P is None or len(P) < 2:
            return None
        self._lddt_cache = float(calculate_lddt(P, Q))
        return self._lddt_cache

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

    def save_aligned_pdb(self, filename: str, subset_only: bool = False, preserve_bfactor: bool = False):
        """Saves the aligned mobile structure to a PDB file.

        By default the per-residue alignment distance is written into the
        B-factor column (useful for heat-map colouring). Pass
        ``preserve_bfactor=True`` to keep the input B-factors (e.g. AlphaFold
        pLDDT) untouched.
        """
        color_by = "bfactor" if preserve_bfactor else "rmsd"
        out_struct = self._build_aligned_structure(color_by=color_by)
        if filename.lower().endswith(".cif") or filename.lower().endswith(".mmcif"):
            out_struct.make_mmcif_document().write_file(filename)
        else:
            out_struct.write_pdb(filename)

    def _build_aligned_structure(self, color_by: str = "rmsd"):
        """Return a transformed clone of the mobile structure.

        ``color_by="rmsd"`` writes the per-residue alignment deviation (Å) into
        every atom's ``b_iso``; ``color_by`` in ``{"bfactor", "plddt"}`` leaves the
        original B-factors untouched. Shared by :meth:`save_aligned_pdb` and
        :meth:`aligned_structure` so file output and in-memory views are identical.
        """
        import numpy as np
        if not self._chosen:
            raise ValueError("No alignment results available. Run align() first.")

        chosen = self._chosen
        R = t = per_res_rmsd = None
        mob_atoms = []
        dist_map = {}

        if chosen["seqguided"]:
            si = chosen["seqguided"]["si"]
            R = si["rotation"]
            t = si["translation"]
            per_res_rmsd = si["per_residue_rmsd"]
            # Residue-keyed directly: no re-derivation from atom objects.
            keys = si.get("mob_residue_keys") or []
            dist_map = {k: float(v) for k, v in zip(keys, per_res_rmsd) if k}
            mob_atoms = chosen["seqguided"]["mob_atoms"]
        elif chosen["seqfree"]:
            R = chosen["seqfree"].rotation
            t = chosen["seqfree"].translation
            ref_subset = chosen["seqfree"].ref_subset_ca_coords
            mob_subset = chosen["seqfree"].mob_subset_ca_coords_aligned
            pairs = chosen["seqfree"].pairs
            mob_infos = chosen["seqfree"].mob_subset_infos
            per_res_rmsd = [float(np.linalg.norm(ref_subset[i] - mob_subset[j]))
                            for (i, j) in pairs]

            class PseudoAtom:
                def __init__(self, c_name, r_seq, r_ico):
                    self.chain_name = c_name
                    self.res_seq = r_seq
                    self.res_icode = r_ico

            mob_atoms = [PseudoAtom(mob_infos[j].chain_id, mob_infos[j].resseq,
                                    mob_infos[j].icode) for (i, j) in pairs]

        if R is None or t is None:
            raise ValueError("Chosen alignment has no transform.")

        out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, 'clone') \
            else self.mob_struct.copy()

        write_rmsd = (color_by == "rmsd")
        if write_rmsd and not dist_map and mob_atoms and per_res_rmsd is not None:
            for k in range(min(len(mob_atoms), len(per_res_rmsd))):
                ma = mob_atoms[k]
                c_name = getattr(ma, 'chain_name', 'A')
                r_ico = getattr(ma, 'res_icode', '')
                key = (c_name, ma.res_seq,
                       r_ico.strip() if hasattr(r_ico, 'strip') else "")
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
                        if write_rmsd:
                            atom.b_iso = mapped_bfactor
        return out_struct

    def aligned_structure(self, color_by: str = "rmsd"):
        """In-memory transformed mobile structure for 3D viewing/export.

        Returns a :class:`gemmi.Structure` moved onto the reference frame. With
        ``color_by="rmsd"`` (default) the per-residue deviation is stored in the
        B-factor column; ``"bfactor"``/``"plddt"`` preserve the input B-factors.
        """
        return self._build_aligned_structure(color_by=color_by)

    def _write_pymol_script(self, path, aligned_name, ref_name):
        lines = [
            f"# pdb_align: reference (grey) + mobile coloured by per-residue "
            f"deviation (B-factor column, 0-5 A)",
            f"load {ref_name}, ref",
            f"load {aligned_name}, mob",
            "hide everything",
            "show cartoon",
            "color grey70, ref",
            "spectrum b, blue_white_red, mob, minimum=0, maximum=5",
            "set cartoon_transparency, 0.1, ref",
            "zoom",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_chimerax_script(self, path, aligned_name, ref_name):
        lines = [
            "# pdb_align: reference (grey) + mobile coloured by per-residue "
            "deviation (B-factor column, 0-5 A)",
            f"open {ref_name}",
            f"open {aligned_name}",
            "hide atoms",
            "show cartoons",
            "color #1 grey",
            "color byattribute bfactor #2 palette blue:white:red range 0,5",
            "view",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def export_bundle(self, path, include=None, fmt="zip"):
        """Write a reproducible bundle of alignment outputs.

        Components (``include``, default all): ``aligned`` (transformed mobile
        structure coloured by per-residue RMSD), ``rmsd_csv``, ``plots``
        (summary + per-residue figures), ``pymol`` (.pml), ``chimerax`` (.cxc),
        and ``report`` (text + JSON, both carrying the quality verdict).
        ``fmt="zip"`` writes a ``.zip``; ``fmt="dir"`` a folder. Returns the path.
        """
        import os
        import tempfile
        import zipfile
        import shutil
        components = include or ["aligned", "reference", "rmsd_csv", "plots",
                                 "pymol", "chimerax", "report"]
        # The viewer scripts load the reference *and* the aligned mobile; the
        # bundle must therefore contain both. It previously shipped only the
        # aligned mobile and loaded it twice, so the side-by-side view the
        # scripts promise was impossible to reproduce.
        ref_name = "reference.pdb"
        workdir = tempfile.mkdtemp(prefix="pdb_align_bundle_")
        try:
            if "aligned" in components:
                self.aligned_structure(color_by="rmsd").write_pdb(
                    os.path.join(workdir, "aligned.pdb"))
            if "reference" in components:
                self._write_reference_copy(os.path.join(workdir, ref_name))
            if "rmsd_csv" in components:
                self.get_rmsd_df().to_csv(os.path.join(workdir, "rmsd.csv"),
                                          index=False)
            if "plots" in components:
                try:
                    self.plot_summary(os.path.join(workdir, "summary.png"))
                    self.plot_rmsd(filename=os.path.join(workdir, "rmsd.png"))
                except Exception:
                    pass
            if "pymol" in components:
                self._write_pymol_script(os.path.join(workdir, "view.pml"),
                                         "aligned.pdb", ref_name)
            if "chimerax" in components:
                self._write_chimerax_script(os.path.join(workdir, "view.cxc"),
                                            "aligned.pdb", ref_name)
            if "report" in components:
                with open(os.path.join(workdir, "report.txt"), "w") as f:
                    f.write(self.report(fmt="text") + "\n")
                with open(os.path.join(workdir, "report.json"), "w") as f:
                    f.write(self.to_json())

            if fmt == "dir":
                if os.path.isdir(path):
                    shutil.rmtree(path)
                shutil.copytree(workdir, path)
                return path
            zpath = path if path.endswith(".zip") else path + ".zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(workdir)):
                    z.write(os.path.join(workdir, fn), fn)
            return zpath
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _write_reference_copy(self, path):
        """Write the reference structure into a bundle as PDB.

        Re-reading and re-writing (rather than copying bytes) normalises mmCIF
        input to the PDB the viewer scripts expect and drops anything the
        viewers cannot use.
        """
        import gemmi
        struct = gemmi.read_structure(self.ref_file)
        struct.setup_entities()
        struct.write_pdb(path)

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
        if self._chosen["seqguided"] and self._chosen["seqguided"].get("aln") is None and not self._chosen["seqfree"]:
            return None
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
        """Per-**residue** deviation table (``Residue``, ``Chain``, ``RMSD``).

        One row per residue in every atom mode. With ``atoms="backbone"`` or
        ``"all_heavy"`` the value is the RMS over that residue's matched atoms;
        emitting one row per atom (as before) multiplied the residue count by
        ~4 or ~8 and pushed the reported coverage to 792%.
        """
        import pandas as pd
        import numpy as np
        labels, chains, distances = [], [], []
        if self._chosen["seqguided"]:
            si = self._chosen["seqguided"]["si"]
            per_res_rmsd = si["per_residue_rmsd"]
            if on == 'reference':
                labels = list(si["residue_labels"])
                chains = list(si["residue_chains"])
            else:
                labels = list(si.get("mob_residue_labels") or si["residue_labels"])
                chains = list(si.get("mob_residue_chains") or si["residue_chains"])
            distances = list(per_res_rmsd)
            n = min(len(labels), len(distances))
            labels, chains, distances = labels[:n], chains[:n], distances[:n]
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

    def plot_rmsd(self, filename: str = "rmsd.pdf", style: str = "scientific",
                  on: str = 'reference'):
        """Per-residue deviation plot, one line per chain.

        Drawn with matplotlib only. It used to require seaborn, which is
        declared in the ``[app]`` extra, so ``pdb_align --plot`` raised
        ImportError on a core install.
        """
        import contextlib
        import matplotlib.pyplot as plt
        from . import plotstyle
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
                style_ctx = plotstyle.apply_nature_style()
                figsize = (89 / 25.4 * 2, 89 / 25.4 * 1.1)
                markersize, linewidth = 3, 0.9
            else:
                style_ctx = contextlib.nullcontext()
                figsize = (10, 4)
                markersize, linewidth = 4, 1
            with style_ctx:
                fig, ax = plt.subplots(figsize=figsize)
                multi = df["Chain"].nunique() > 1
                for i, (chain, grp) in enumerate(df.groupby("Chain", sort=False)):
                    ax.plot(grp.index, grp["RMSD"], marker='o',
                            markersize=markersize, linestyle='-',
                            linewidth=linewidth,
                            color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)],
                            label=str(chain))
                if multi:
                    ax.legend(frameon=False, title="Chain")
                n_labels = len(df)
                step = max(1, n_labels // 10)
                ax.set_xticks(range(0, n_labels, step))
                ax.set_xticklabels(df["Residue"].iloc[::step], rotation=45,
                                   ha='right')
                ax.set_xlabel(f"Residue ({on.capitalize()})")
                ax.set_ylabel(r"C$\alpha$ deviation ($\AA$)")
                ax.set_title("Per-residue structural deviation")
                fig.tight_layout()
                fig.savefig(filename, bbox_inches='tight')
                plt.close(fig)

    def plot_summary(self, filename: str = None, show: bool = False):
        """Compact multi-panel Nature-style summary: per-residue RMSD + per-chain bar + scores."""
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from . import plotstyle
        df = self.get_rmsd_df()
        stats = self.summary_stats()
        with plotstyle.apply_nature_style():
            fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 183/25.4*0.4))
            ax0, ax1 = axes
            if not df.empty:
                for i, (chain, g) in enumerate(df.groupby("Chain")):
                    ax0.plot(range(len(g)), g["RMSD"], lw=0.9,
                             color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)], label=str(chain))
                ax0.set_xlabel("Residue index"); ax0.set_ylabel(r"C$\alpha$ deviation ($\AA$)")
                if df["Chain"].nunique() > 1:
                    ax0.legend(frameon=False)
            plotstyle.panel_label(ax0, "a")
            pc = self.per_chain
            if not pc.empty:
                labels = [f"{a}->{b}" for a, b in zip(pc["chain_ref"], pc["chain_mob"])]
                ax1.bar(labels, pc["rmsd"], color=plotstyle.PALETTE[0])
                ax1.set_ylabel(r"RMSD ($\AA$)"); ax1.tick_params(axis="x", rotation=45)
            else:
                txt = (f"RMSD {stats['rmsd']:.2f} A\n"
                       f"TM {stats['tm_score']:.3f}" if stats['rmsd'] is not None else "n/a")
                ax1.text(0.5, 0.5, txt, ha="center", va="center", transform=ax1.transAxes)
                ax1.axis("off")
            plotstyle.panel_label(ax1, "b")
            fig.tight_layout()
            if filename:
                fig.savefig(filename)
            if show:
                plt.show()
        return fig

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

    _SAVE_VERSION = 1

    def save(self, path: str):
        """Persist all computed data to a versioned .npz (no gemmi needed to reload)."""
        import numpy as np, json
        df = self.get_rmsd_df()
        per_chain = self.per_chain
        meta = self.summary_stats()
        meta["_save_version"] = self._SAVE_VERSION
        coords = self.get_aligned_coords()
        ref_c, mob_c = (coords if coords is not None else (np.empty((0, 3)), np.empty((0, 3))))
        np.savez_compressed(
            path,
            meta_json=json.dumps(meta, default=float),
            rmsd_residues=np.array(df["Residue"].tolist(), dtype=object),
            rmsd_chains=np.array(df["Chain"].tolist(), dtype=object),
            rmsd_values=df["RMSD"].to_numpy(dtype=float),
            per_chain_json=per_chain.to_json(orient="records"),
            ref_coords=np.asarray(ref_c, dtype=float),
            mob_coords=np.asarray(mob_c, dtype=float),
        )

    @staticmethod
    def load(path: str) -> "LoadedResult":
        return LoadedResult._from_npz(path)


class LoadedResult:
    """A replayable, gemmi-free view of a saved AlignmentResult."""

    def __init__(self, meta, rmsd_df, per_chain, ref_coords, mob_coords):
        self._meta = meta
        self._rmsd_df = rmsd_df
        self._per_chain = per_chain
        self.ref_coords = ref_coords
        self.mob_coords_aligned = mob_coords
        self.strategy = meta.get("strategy")
        self.rmsd = meta.get("rmsd")
        self.tm_score = meta.get("tm_score")

    @classmethod
    def _from_npz(cls, path):
        import numpy as np, json, pandas as pd
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        if meta.get("_save_version") != AlignmentResult._SAVE_VERSION:
            raise ValueError(
                f"Unsupported save version {meta.get('_save_version')}; "
                f"expected {AlignmentResult._SAVE_VERSION}."
            )
        rmsd_df = pd.DataFrame({
            "Residue": list(z["rmsd_residues"]),
            "Chain": list(z["rmsd_chains"]),
            "RMSD": z["rmsd_values"],
        })
        per_chain = pd.read_json(io.StringIO(str(z["per_chain_json"])), orient="records")
        return cls(meta, rmsd_df, per_chain, z["ref_coords"], z["mob_coords"])

    def summary_stats(self):
        return dict(self._meta)

    def get_rmsd_df(self, on="reference"):
        return self._rmsd_df.copy()

    @property
    def per_chain(self):
        return self._per_chain

    # Reuse the exact rendering logic from AlignmentResult by delegation.
    report = AlignmentResult.report
    plot_rmsd = AlignmentResult.plot_rmsd
    plot_summary = AlignmentResult.plot_summary


class EnsembleResult:
    """
    Holds multiple AlignmentResult objects from an ensemble run against a common
    reference. Provides PCA, clustering, and summary analysis.
    """

    def __init__(self, results: List["AlignmentResult"], labels: List[str]):
        if len(results) != len(labels):
            raise ValueError("results and labels must have the same length.")
        self.results = results
        self.labels = labels
        self._cluster_labels: Optional[np.ndarray] = None
        self._feature_matrix: Optional[np.ndarray] = None

    def _get_feature_matrix(self) -> np.ndarray:
        """N_models x N_common_residues matrix of per-residue RMSD (cached)."""
        if self._feature_matrix is not None:
            return self._feature_matrix

        dfs = [r.get_rmsd_df(on="reference") for r in self.results]
        common = set(dfs[0]["Residue"].tolist())
        for df in dfs[1:]:
            common &= set(df["Residue"].tolist())
        common_sorted = sorted(common)
        if not common_sorted:
            raise ValueError(
                "No common residue identifiers found across all models. "
                "Ensure all structures are aligned to the same reference numbering."
            )

        matrix = []
        for df in dfs:
            indexed = df.set_index("Residue")["RMSD"]
            row = [float(indexed[res]) if res in indexed.index else 0.0
                   for res in common_sorted]
            matrix.append(row)

        self._feature_matrix = np.array(matrix, dtype=float)
        return self._feature_matrix

    def summary(self) -> pd.DataFrame:
        """DataFrame with columns: model, rmsd, tm_score, gdt_ts, n_aligned."""
        rows = []
        for label, result in zip(self.labels, self.results):
            gdt_ts = None
            sg = result._chosen.get("seqguided")
            sf = result._chosen.get("seqfree")
            if sg and sg.get("si"):
                gdt_ts = sg["si"].get("gdt_ts")
            elif sf:
                gdt_ts = getattr(sf, "gdt_ts", None)

            try:
                n_aligned = len(result.get_rmsd_df())
            except Exception:
                n_aligned = None

            rows.append({
                "model": label,
                "rmsd": result.rmsd,
                "tm_score": result.tm_score,
                "gdt_ts": gdt_ts,
                "n_aligned": n_aligned,
            })
        return pd.DataFrame(rows)

    def rmsd_matrix(self) -> pd.DataFrame:
        """NxN DataFrame of pairwise per-residue RMSD-vector distances between models."""
        N = len(self.labels)
        mat = self._get_feature_matrix()
        pairwise = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                diff = mat[i] - mat[j]
                d = float(np.sqrt(np.mean(diff ** 2)))
                pairwise[i, j] = d
                pairwise[j, i] = d
        return pd.DataFrame(pairwise, index=self.labels, columns=self.labels)

    def cluster(self, n_clusters: int = None) -> np.ndarray:
        """
        K-means clustering on per-residue RMSD vectors.

        If n_clusters is None, auto-selects via elbow method (k=2..8).
        Stores labels internally for plot_pca(color_by='cluster').
        Returns integer label array of length N.
        """
        from sklearn.cluster import KMeans

        mat = self._get_feature_matrix()
        N = len(self.results)

        if n_clusters is None:
            max_k = min(8, N - 1)
            if max_k < 2:
                self._cluster_labels = np.zeros(N, dtype=int)
                return self._cluster_labels
            ks = list(range(2, max_k + 1))
            inertias = []
            for k in ks:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(mat)
                inertias.append(km.inertia_)
            if len(inertias) >= 2:
                # Standard elbow: k where the drop in inertia is greatest
                n_clusters = ks[int(np.argmax(-np.diff(inertias)))]
            else:
                n_clusters = 2

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._cluster_labels = km.fit_predict(mat)
        return self._cluster_labels

    def plot_pca(self, color_by: str = "cluster", save_path: str = None):
        """
        2D PCA of per-residue RMSD vectors, one point per model.

        color_by: 'cluster' | 'rmsd' | 'tm_score'
        If color_by='cluster' but cluster() not called, falls back to 'rmsd' with a warning.
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        mat = self._get_feature_matrix()
        n_components = min(2, mat.shape[0], mat.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(mat)
        if coords.shape[1] < 2:
            coords = np.hstack([coords, np.zeros((len(coords), 2 - coords.shape[1]))])

        if color_by == "cluster" and self._cluster_labels is None:
            warnings.warn(
                "plot_pca(color_by='cluster') called before cluster() — "
                "falling back to color_by='rmsd'.",
                UserWarning,
                stacklevel=2,
            )
            color_by = "rmsd"

        if color_by == "cluster":
            colors, cmap, clabel = self._cluster_labels, "tab10", "Cluster"
        elif color_by == "rmsd":
            colors, cmap, clabel = [r.rmsd or 0.0 for r in self.results], "viridis", "RMSD (Å)"
        elif color_by == "tm_score":
            colors, cmap, clabel = [r.tm_score or 0.0 for r in self.results], "plasma", "TM-score"
        else:
            raise ValueError(f"color_by must be 'cluster', 'rmsd', or 'tm_score', got {color_by!r}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=60, alpha=0.85)
        plt.colorbar(sc, ax=ax, label=clabel)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)" if len(var) > 0 else "PC1")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)" if len(var) > 1 else "PC2")
        ax.set_title("Structural Ensemble PCA")
        for i, lbl in enumerate(self.labels):
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.6,
                        ha="center", va="bottom")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_dendrogram(self, save_path: str = None):
        """Hierarchical clustering dendrogram using Ward linkage on per-residue RMSD vectors."""
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        mat = self._get_feature_matrix()
        Z = linkage(mat, method="ward")
        fig, ax = plt.subplots(figsize=(max(8, len(self.labels) * 0.4), 5))
        dendrogram(Z, labels=self.labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
        ax.set_title("Structural Ensemble Dendrogram (Ward)")
        ax.set_ylabel("Distance")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def export_bundle(self, path, fmt="zip"):
        """Write a reproducible ensemble bundle: ``summary.csv``,
        ``rmsd_matrix.csv``, ``clusters.csv``, ``pca.png``, ``dendrogram.png``.

        ``fmt="zip"`` writes a ``.zip``; ``fmt="dir"`` a folder. Returns the path.
        """
        import os
        import tempfile
        import zipfile
        import shutil
        workdir = tempfile.mkdtemp(prefix="pdb_align_ens_")
        try:
            self.summary().to_csv(os.path.join(workdir, "summary.csv"), index=False)
            self.rmsd_matrix().to_csv(os.path.join(workdir, "rmsd_matrix.csv"))
            try:
                import pandas as pd
                labels = self.cluster()
                pd.DataFrame({"model": self.labels, "cluster": labels}).to_csv(
                    os.path.join(workdir, "clusters.csv"), index=False)
            except Exception:
                pass
            for meth, fn in ((self.plot_pca, "pca.png"),
                             (self.plot_dendrogram, "dendrogram.png")):
                try:
                    meth(save_path=os.path.join(workdir, fn))
                except Exception:
                    pass
            if fmt == "dir":
                if os.path.isdir(path):
                    shutil.rmtree(path)
                shutil.copytree(workdir, path)
                return path
            zpath = path if path.endswith(".zip") else path + ".zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(workdir)):
                    z.write(os.path.join(workdir, fn), fn)
            return zpath
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def __repr__(self) -> str:
        n = len(self.results)
        preview = self.labels[:3]
        suffix = "..." if n > 3 else ""
        return f"<EnsembleResult n_models={n} labels={preview}{suffix}>"


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
        # absolute path str -> (gemmi.Structure, mtime, size); parse-once cache
        # keyed on file metadata so an edited file on disk is re-parsed.
        self._struct_cache: dict = {}
        # Directory for remote (pdb:/af:) downloads. Override with the
        # PDB_ALIGN_CACHE_DIR env var; never pollutes the current directory.
        self._fetch_cache_dir = os.environ.get("PDB_ALIGN_CACHE_DIR") or \
            os.path.join(os.path.expanduser("~"), ".cache", "pdb_align")

        if ref_file:
            self.set_reference(ref_file, chains_ref)

    def add_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure. Alias for set_reference."""
        self.set_reference(ref_file, chains)

    #: Parsed structures kept in memory. Bounded because evaluating N models
    #: through one aligner would otherwise retain all N structures (tens of MB
    #: each for a large complex).
    _CACHE_MAX = 4

    def _load_cached_structure(self, abspath: str):
        """Return a fresh clone of the parsed structure, re-parsing if the file
        on disk changed since it was cached (keyed on mtime + size)."""
        try:
            stat = os.stat(abspath)
            sig = (stat.st_mtime, stat.st_size)
        except OSError:
            sig = None
        cached = self._struct_cache.get(abspath)
        if cached is None or cached[1] != sig:
            while len(self._struct_cache) >= self._CACHE_MAX:
                self._struct_cache.pop(next(iter(self._struct_cache)))
            self._struct_cache[abspath] = (_parse_path(abspath), sig)
        else:  # mark as most recently used
            self._struct_cache[abspath] = self._struct_cache.pop(abspath)
        return self._struct_cache[abspath][0].clone()

    # Network timeout (seconds) for remote structure fetches; without it a
    # stalled connection would hang the whole alignment indefinitely.
    _FETCH_TIMEOUT = 30

    # AlphaFold DB model versions to try, newest first. The DB retires old
    # versions and not every entry exists at the newest one, so we fall back.
    _AF_MODEL_VERSIONS = (6, 5, 4)

    def _download(self, url: str, dest: str, what: str):
        """Download *url* to *dest* atomically, raising ValueError on failure.

        Writes to a temporary file in the same directory and renames on success
        so a stalled/failed download never leaves a truncated file behind.
        """
        import requests
        try:
            r = requests.get(url, timeout=self._FETCH_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Could not fetch {what}: {exc}") from exc
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".part")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(r.text)
            os.replace(tmp, dest)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @staticmethod
    def _is_usable(path: str) -> bool:
        """True if *path* exists and is non-empty (i.e. a complete download)."""
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _fetch_structure(self, file_or_id: str) -> str:
        """Fetches a structure from PDB or AF-DB if a prefix is detected.

        Downloads are cached under ``self._fetch_cache_dir`` rather than the
        current working directory.
        """
        cache = self._fetch_cache_dir
        if file_or_id.lower().startswith("pdb:"):
            pdb_id = file_or_id[4:].strip()
            dest = os.path.join(cache, f"{pdb_id}.cif")
            if not self._is_usable(dest):
                if self.verbose: print(f"Fetching {pdb_id} from RCSB PDB...")
                self._download(f"https://files.rcsb.org/download/{pdb_id}.cif", dest, f"PDB {pdb_id}")
            return dest
        elif file_or_id.lower().startswith("af:"):
            af_id = file_or_id[3:].strip()
            dest = os.path.join(cache, f"{af_id}.pdb")
            if self._is_usable(dest):
                return dest
            if self.verbose: print(f"Fetching {af_id} from AlphaFold DB...")
            last_exc = None
            for ver in self._AF_MODEL_VERSIONS:
                url = f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v{ver}.pdb"
                try:
                    self._download(url, dest, f"AlphaFold model {af_id} (v{ver})")
                    return dest
                except ValueError as exc:
                    last_exc = exc
                    if self.verbose: print(f"  v{ver} unavailable: {exc}")
            raise ValueError(
                f"Could not fetch AlphaFold model {af_id} at any known version "
                f"{self._AF_MODEL_VERSIONS}: {last_exc}"
            )
        return file_or_id

    def set_reference(self, ref_file: str, chains: Optional[List[Union[str, int]]] = None):
        """Sets the reference structure. Supports pdb:XXXX and af:XXXX fetching."""
        ref_file = self._fetch_structure(ref_file)
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference file not found: {ref_file}")
        ref_file = os.path.abspath(ref_file)
        self.ref_file = ref_file
        self.ref_struct = self._load_cached_structure(ref_file)
        self.ref_seqs, self.ref_lens = extract_sequences_and_lengths(self.ref_struct, os.path.basename(ref_file))
        # Validate AFTER the sequences are read, so an unknown chain can be named against
        # what the file actually contains. See _validate_chain_selection.
        self.chains_ref = self._validate_chain_selection(chains, self.ref_seqs, "Reference")
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
        self.mob_struct = self._load_cached_structure(mob_file)
        self.mob_seqs, self.mob_lens = extract_sequences_and_lengths(self.mob_struct, os.path.basename(mob_file))
        # Validate AFTER the sequences are read, so an unknown chain can be named against
        # what the file actually contains. See _validate_chain_selection.
        self.chains_mob = self._validate_chain_selection(chains, self.mob_seqs, "Mobile")
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

    @staticmethod
    def _validate_chain_selection(chains, available, side: str):
        """Reject a chain selection that cannot mean what the caller intended.

        A DUPLICATE entry silently aligns one chain against two different partners, which
        produces a plausible-looking but meaningless RMSD rather than an error. This is
        easy to hit when a caller maps reference chains onto mobile chains through a
        correspondence table and falls back to the identity for entries the table omits:
        two distinct reference chains then land on the same mobile chain. Observed in the
        wild on a two-copy assembly where only one copy was modelled -- the joint fit
        reported 23.65 A for a structure that superposes at 4.82 A, with the giveaway
        being a per-chain breakdown showing identical values for chains that are not
        identical. Failing loudly here costs one exception; failing silently costs a wrong
        number that looks right.
        """
        if chains is None:
            return None
        seq = list(chains)
        dupes = sorted({str(c) for c in seq if seq.count(c) > 1})
        # A selector may carry a residue range ("A:10-150"); validate the chain
        # part only. Comparing the whole selector against chain names rejected
        # the documented range syntax outright.
        def _chain_part(c):
            if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
                return c
            return _parse_chain_selector(c)[0]
        if dupes:
            raise ValueError(
                f"{side} chain selection contains duplicate chain(s) {dupes}: {seq}. "
                f"Each chain may appear at most once -- a repeated chain would be aligned "
                f"against two different partners and the resulting RMSD would be "
                f"meaningless. If you built this list from a chain-correspondence map, "
                f"drop the entries the map does not resolve instead of falling back to "
                f"the identity for them."
            )
        if available:
            known = {str(a) for a in available}
            unknown = [str(c) for c in seq
                       if not isinstance(_chain_part(c), (int, np.integer))
                       and str(_chain_part(c)) not in known]
            if unknown:
                raise ValueError(
                    f"{side} chain(s) {unknown} are not present in the structure "
                    f"(available: {sorted(known)}). Residue ranges are written "
                    f"'CHAIN:start-end', e.g. 'A:10-150'."
                )
        return seq

    def set_reference_chains(self, chains: List[Union[str, int]]):
        """Changes the reference chains to use for alignment."""
        if not self.ref_file:
            raise ValueError("Reference structure must be set first.")
        self.chains_ref = self._validate_chain_selection(
            chains, getattr(self, "ref_seqs", None), "Reference")
        if self.verbose:
            print(f"Reference chains updated to: {chains}")

    def set_mobile_chains(self, chains: List[Union[str, int]]):
        """Changes the mobile chains to use for alignment."""
        if not self.mob_file:
            raise ValueError("Mobile structure must be set first.")
        self.chains_mob = self._validate_chain_selection(
            chains, getattr(self, "mob_seqs", None), "Mobile")
        if self.verbose:
            print(f"Mobile chains updated to: {chains}")

    # --- residue selection -------------------------------------------------

    @staticmethod
    def _looks_like_plddt(struct, chains) -> bool:
        """True when the B-factor column plausibly holds pLDDT.

        pLDDT is a confidence in [0, 100] that is high for well-predicted
        residues; a crystallographic B-factor is a disorder measure that is
        *low* for well-ordered ones and routinely exceeds 100 for flexible
        loops. The two therefore demand opposite cutoffs, and applying a
        pLDDT floor to an experimental reference discards exactly the ordered
        core one wants to keep (and, at a typical --min-plddt 70, discards the
        entire structure).

        Heuristic, deliberately conservative: a predicted model has a high
        mean, nothing above 100, and no exact zeros.
        """
        from .core import select_residues
        try:
            sel = select_residues(struct, chains, with_atoms=False)
        except ValueError:
            return False
        b = np.array([r.b_iso for r in sel.residues], dtype=float)
        if b.size == 0:
            return False
        return bool(b.max() <= 100.0 and b.mean() >= 50.0 and b.min() > 0.0)

    def _build_selections(self, ref_chs, mob_chs, min_b_factor, min_plddt):
        """Build both selections, applying ``min_plddt`` only where it means
        something. Returns (ref_sel, mob_sel)."""
        ref_plddt = mob_plddt = 0.0
        if min_plddt > 0.0:
            ref_is_pred = self._looks_like_plddt(self.ref_struct, ref_chs)
            mob_is_pred = self._looks_like_plddt(self.mob_struct, mob_chs)
            ref_plddt = min_plddt if ref_is_pred else 0.0
            mob_plddt = min_plddt if mob_is_pred else 0.0
            skipped = [name for name, pred in
                       (("reference", ref_is_pred), ("mobile", mob_is_pred))
                       if not pred]
            if skipped:
                warnings.warn(
                    f"min_plddt={min_plddt:g} was not applied to the "
                    f"{' and '.join(skipped)} structure: its B-factor column "
                    f"does not look like pLDDT (predicted models carry 0-100 "
                    f"confidences, experimental structures carry B-factors, "
                    f"where low means well ordered). Use min_b_factor to "
                    f"filter on B-factors explicitly.",
                    UserWarning, stacklevel=3)
        ref_sel = select_residues(self.ref_struct, ref_chs,
                                  min_b_factor=min_b_factor, min_plddt=ref_plddt,
                                  source=os.path.basename(self.ref_file or ""))
        mob_sel = select_residues(self.mob_struct, mob_chs,
                                  min_b_factor=min_b_factor, min_plddt=mob_plddt,
                                  source=os.path.basename(self.mob_file or ""))
        return ref_sel, mob_sel

    def align(self, mode: str = "auto", seq_gap_open: float = -10,
              seq_gap_extend: float = -0.5, atoms: str = "CA",
              min_plddt: float = 0.0, min_b_factor: float = 0.0,
              hinge_threshold: float = 3.0, hinge_window: int = 15,
              domain_min_residues: int = 30, strategy: str = "auto", **kwargs):
        """
        Run the alignment.

        :param mode: ``"auto"`` (compare sequence-guided and sequence-free and
            keep the better), ``"seq_guided"``, ``"seq_free_shape"``,
            ``"seq_free_window"``, ``"seq_free_auto"``, or ``"flexible"``
            (rigid domains separated by hinges).
        :param seq_gap_open: gap-open score for the sequence alignment.
        :param seq_gap_extend: gap-extend score for the sequence alignment.
        :param atoms: atoms to superpose: ``"CA"``, ``"backbone"`` (N, CA, C, O)
            or ``"all_heavy"``. Side chains are only paired between residues of
            the same type.
        :param min_plddt: pLDDT floor for *predicted* structures. Applied only
            to a side whose B-factor column looks like pLDDT; a warning names
            any side it was skipped for.
        :param min_b_factor: B-factor floor, applied symmetrically.
        :param strategy: multi-chain superposition strategy, ``"auto"``,
            ``"global"`` or ``"local"``.
        :returns: an :class:`AlignmentResult`.
        :raises AlignmentFailedError: when the requested mode cannot produce an
            alignment. It never returns a result whose metrics are all ``None``.
        """
        if mode == "flexible":
            return self._align_flexible(
                seq_gap_open=seq_gap_open, seq_gap_extend=seq_gap_extend,
                atoms=atoms, min_plddt=min_plddt, min_b_factor=min_b_factor,
                hinge_threshold=hinge_threshold, hinge_window=hinge_window,
                domain_min_residues=domain_min_residues, strategy=strategy,
                **kwargs)

        if not self.ref_file or not self.mob_file:
            raise ValueError("Both reference and mobile structures must be set "
                             "before alignment.")

        ref_chs = self.chains_ref if self.chains_ref else list(self.ref_seqs.keys())
        mob_chs = self.chains_mob if self.chains_mob else list(self.mob_seqs.keys())
        if not ref_chs or not mob_chs:
            raise ValueError("Select at least one chain per file.")

        ref_sel, mob_sel = self._build_selections(ref_chs, mob_chs,
                                                  min_b_factor, min_plddt)

        # Multi-chain dispatch: only when both sides really have several chains.
        if mode in ("auto", "Auto (best RMSD)") and \
                len(ref_sel.chain_order) > 1 and len(mob_sel.chain_order) > 1:
            from .chains import match_chains, align_multichain, _chain_selections
            mapping = match_chains(self.ref_seqs, self.mob_seqs,
                                   self.ref_struct, self.mob_struct,
                                   ref_sel.chain_order, mob_sel.chain_order)
            if mapping.pairs:
                ref_subs = {c: ref_sel.sub([i for i, r in enumerate(ref_sel.residues)
                                            if r.chain_id == c])
                            for c in ref_sel.chain_order}
                mob_subs = {c: mob_sel.sub([i for i, r in enumerate(mob_sel.residues)
                                            if r.chain_id == c])
                            for c in mob_sel.chain_order}
                mc = align_multichain(self.ref_struct, self.mob_struct, mapping,
                                      strategy=strategy, atoms=atoms,
                                      ref_selections=ref_subs,
                                      mob_selections=mob_subs)
                result_obj = self._multichain_to_result(mc, ref_sel, mob_sel)
                self.last_result = {"seqguided": None, "seqfree": None,
                                    "chosen": result_obj._chosen}
                if self.verbose:
                    print(result_obj.report())
                return result_obj

        seqguided = None
        seqfree = None
        failures = []

        if mode in ("auto", "Auto (best RMSD)", "seq_guided", "Sequence-guided"):
            # One sequence per selection, so alignment columns map onto
            # residues by index and cannot drift (see core module docstring).
            aln = perform_sequence_alignment(ref_sel.sequence, mob_sel.sequence,
                                             seq_gap_open, seq_gap_extend)
            pairs = pairs_from_alignment(aln)
            if pairs:
                ref_atoms, mob_atoms = paired_atoms(ref_sel, mob_sel, pairs,
                                                    atoms=atoms)
                si = superimpose_atoms(
                    ref_atoms, mob_atoms,
                    recycles=int(kwargs.get("recycles", 0)),
                    keep_fraction=float(kwargs.get("keep_fraction", 1.0)),
                    n_total=ref_sel.n_residues)
                if si:
                    seqguided = dict(aln=aln, ref_atoms=ref_atoms,
                                     mob_atoms=mob_atoms, si=si,
                                     ref_selection=ref_sel,
                                     mob_selection=mob_sel)
                else:
                    failures.append("sequence-guided: no superposable atoms")
            else:
                failures.append("sequence-guided: the sequences share no "
                                "alignable residues")

        seqfree_modes = ("auto", "Auto (best RMSD)", "seq_free_auto",
                         "Sequence-free (auto)", "seq_free_shape",
                         "Sequence-free (shape)", "seq_free_window",
                         "Sequence-free (window)")
        if mode in seqfree_modes:
            sf_method = {"seq_free_shape": "shape", "Sequence-free (shape)": "shape",
                         "seq_free_window": "window",
                         "Sequence-free (window)": "window"}.get(mode, "auto")
            try:
                seqfree = sequence_independent_alignment_joined_v2(
                    file_ref=self.ref_file, file_mob=self.mob_file,
                    method=sf_method, atoms=atoms,
                    ref_selection=ref_sel, mob_selection=mob_sel, **kwargs)
            except Exception as exc:
                logger.warning("Sequence-free alignment failed: %s", exc,
                               exc_info=True)
                failures.append(f"sequence-free: {exc}")

        if mode in ("auto", "Auto (best RMSD)"):
            best, reason = pick_best_overall(seqguided, seqfree, min_pairs=3)
            if best is None:
                raise AlignmentFailedError(
                    "No alignment could be produced. "
                    + "; ".join(failures or ["no candidate strategy applied"]))
            chosen_name = best["name"]
            chosen = dict(name=chosen_name, reason=reason,
                          seqguided=seqguided if "Sequence-guided" in chosen_name else None,
                          seqfree=seqfree if "Sequence-free" in chosen_name else None)
        else:
            picked_sg = seqguided if ("seq_guided" in mode or "Sequence-guided" in mode) else None
            picked_sf = seqfree if ("seq_free" in mode or "Sequence-free" in mode) else None
            if picked_sg is None and picked_sf is None:
                # Explicit modes must fail loudly. Returning a result whose
                # rmsd/tm_score/report are all None pushes the failure into the
                # caller's output instead of raising it here.
                raise AlignmentFailedError(
                    f"mode={mode!r} produced no alignment. "
                    + "; ".join(failures or ["unknown mode"]))
            chosen = dict(name=mode, reason="Manual mode.",
                          seqguided=picked_sg, seqfree=picked_sf)

        # A non-finite RMSD means the chosen strategy never managed a
        # superposition (fewer than three matched residues). Returning it would
        # put "inf" in a report and in every derived metric.
        chosen_rmsd = (chosen["seqguided"]["si"]["rmsd"] if chosen["seqguided"]
                       else chosen["seqfree"].rmsd if chosen["seqfree"] else None)
        if chosen_rmsd is None or not np.isfinite(chosen_rmsd):
            detail = "; ".join(failures) if failures else \
                "fewer than the 3 matched residues a superposition needs"
            raise AlignmentFailedError(
                f"mode={mode!r} could not superpose the selections "
                f"({chosen['name']}): {detail}.")

        self.last_result = dict(seqguided=seqguided, seqfree=seqfree, chosen=chosen)

        result_obj = AlignmentResult(
            chosen=chosen, seqguided=seqguided, seqfree=seqfree,
            ref_file=self.ref_file, mob_file=self.mob_file,
            mob_struct=self.mob_struct,
            ref_lens=dict(ref_sel.lens), mob_lens=dict(mob_sel.lens),
            verbose=self.verbose)
        result_obj.ref_selection = ref_sel
        result_obj.mob_selection = mob_sel

        if self.verbose:
            print("\nAlignment completed:")
            print(f"  Mode evaluated: {mode}")
            if seqguided:
                print(f"  Sequence-guided RMSD: {seqguided['si']['rmsd']:.3f} Å")
            if seqfree:
                print(f"  Sequence-free RMSD:   {seqfree.rmsd:.3f} Å")
            print(f"  Chosen method: {chosen['name']}")
            print(f"  Reason: {chosen['reason']}")

        return result_obj

    def _align_flexible(self, hinge_threshold, hinge_window,
                        domain_min_residues, **align_kwargs):
        """Rigid-body decomposition: align, find hinges, refit per domain.

        Hinge detection runs per chain. A chain boundary is not a hinge, and a
        "domain" spanning one is not a rigid body: fitting it mixes two
        independent motions and reports a per-domain RMSD that describes
        neither (observed: domains labelled 'chain A 42-21' with a 4.6 A fit on
        a structure whose chains are individually rigid).
        """
        initial = self.align(mode="auto", **align_kwargs)
        sg = initial._chosen.get("seqguided")
        if sg is None:
            warnings.warn(
                "mode='flexible': the chosen alignment has no residue-level "
                "correspondence, so no domain decomposition was attempted; "
                "returning the rigid-body result.",
                UserWarning, stacklevel=3)
            return initial

        si = sg["si"]
        ca_rmsd = np.asarray(si["per_residue_rmsd"], dtype=float)
        keys = si["residue_keys"]
        chains = si["residue_chains"]
        ca_ref = np.asarray(si["ca_ref"])
        ca_mob = np.asarray(si["ca_mob"])
        if len(ca_ref) != len(ca_rmsd) or len(ca_ref) < 3:
            warnings.warn(
                "mode='flexible': no CA-level correspondence available; "
                "returning the rigid-body result.", UserWarning, stacklevel=3)
            return initial

        chain_starts = [i for i in range(len(chains))
                        if i == 0 or chains[i] != chains[i - 1]]
        splits = _detect_hinges(ca_rmsd, window=hinge_window,
                                threshold=hinge_threshold,
                                min_segment=domain_min_residues,
                                chain_starts=chain_starts)
        boundaries = sorted(set([0, *splits, len(ca_rmsd)]))
        segments = [(s, e) for s, e in zip(boundaries[:-1], boundaries[1:])
                    if e - s >= 3 and len(set(chains[s:e])) == 1]
        segments = self._merge_rigid_segments(segments, chains, ca_ref, ca_mob,
                                              hinge_threshold)

        domains = []
        for start, end in segments:
            R, t, rmsd = _kabsch(ca_ref[start:end], ca_mob[start:end])
            domains.append(DomainResult(
                domain_id=len(domains), chain_id=chains[start],
                residue_start=int(keys[start][1]),
                residue_end=int(keys[end - 1][1]),
                n_residues=end - start, rmsd=float(rmsd),
                rotation=R, translation=t))
        initial.domains = domains or None
        return initial

    @staticmethod
    def _merge_rigid_segments(segments, chains, ca_ref, ca_mob, threshold):
        """Merge consecutive same-chain segments that are one rigid body.

        Hinges are detected on deviations measured in the *initial* whole-
        structure frame, which is a compromise fit: a chain that moved as one
        rigid body still shows a deviation ramp across its length and picks up
        spurious splits (a hinged haemoglobin decomposed into 9 "domains" for 4
        rigid chains). A split is only real if the two sides cannot be fitted
        together, so adjacent segments are merged while their union still
        superposes within the same threshold that declared the hinge.
        """
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            p_start, p_end = merged[-1]
            s_start, s_end = seg
            if p_end == s_start and chains[p_start] == chains[s_start]:
                _R, _t, rmsd = _kabsch(ca_ref[p_start:s_end], ca_mob[p_start:s_end])
                if rmsd <= threshold:
                    merged[-1] = (p_start, s_end)
                    continue
            merged.append(seg)
        return merged

    def _multichain_to_result(self, mc, ref_sel, mob_sel):
        from .core import compute_gdt_ts

        ref_atoms = mc.ref_infos
        mob_atoms = mc.mob_infos
        si = superimpose_atoms(ref_atoms, mob_atoms, n_total=ref_sel.n_residues)
        if si is None:
            raise AlignmentFailedError(
                "The multi-chain superposition produced no matched atoms.")
        # Keep the transform the chain layer chose (it may be the local fit).
        si["rotation"] = mc.rotation
        si["translation"] = mc.translation
        si["rmsd"] = mc.rmsd
        seqguided = {"aln": None, "ref_atoms": ref_atoms, "mob_atoms": mob_atoms,
                     "si": si, "ref_selection": ref_sel, "mob_selection": mob_sel}
        chosen = {"name": f"Multi-chain ({mc.strategy})",
                  "reason": f"Chain-aware {mc.strategy} superposition over "
                            f"{len(mc.mapping.pairs)} chain pair(s)."
                            + (" Correspondence refined geometrically "
                               "(sequence-identical chains)."
                               if getattr(mc.mapping, "refined", False) else ""),
                  "seqguided": seqguided, "seqfree": None}
        res = AlignmentResult(chosen=chosen, seqguided=seqguided, seqfree=None,
                              ref_file=self.ref_file, mob_file=self.mob_file,
                              mob_struct=self.mob_struct,
                              ref_lens=dict(ref_sel.lens),
                              mob_lens=dict(mob_sel.lens),
                              verbose=self.verbose)
        res.strategy = mc.strategy
        res.chain_mapping = mc.mapping
        res.ref_selection = ref_sel
        res.mob_selection = mob_sel
        res._per_chain = pd.DataFrame(
            mc.per_chain, columns=["chain_ref", "chain_mob", "n_residues", "rmsd"])
        return res

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

    def align_ensemble(
        self,
        mob_list: List[str],
        mode: str = "auto",
        atoms: str = "CA",
        workers: int = 1,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> "EnsembleResult":
        """
        Align a list of mobile structures against the already-loaded reference.

        Parameters
        ----------
        mob_list : list[str]
            Paths to mobile PDB/CIF files, or remote IDs (``pdb:XXXX``, ``af:UniProtID``).
        mode : str
            Alignment mode forwarded to :meth:`align`. Default ``"auto"``.
        atoms : str
            Atom selection forwarded to :meth:`align`. Default ``"CA"``.
        workers : int
            Reserved for future parallel execution. Currently unused. Default ``1``.
        out_dir : str | None
            If given, each aligned mobile PDB is saved here as ``aligned_<filename>``.
        **kwargs
            Additional keyword arguments forwarded to :meth:`align`.

        Returns
        -------
        EnsembleResult
        """
        if not self.ref_file:
            raise ValueError("Reference structure must be loaded before calling align_ensemble().")

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        results = []
        labels = []

        for mob_path in mob_list:
            label = mob_path  # preserve original input as label
            try:
                self.add_mobile(mob_path)
                # Use the resolved local path for filesystem operations
                safe_fname = os.path.basename(self.mob_file or mob_path)
                res = self.align(mode=mode, atoms=atoms, **kwargs)
                if out_dir:
                    out_pdb = os.path.join(out_dir, f"aligned_{safe_fname}")
                    res.save_aligned_pdb(out_pdb)
                results.append(res)
                labels.append(label)
                if self.verbose:
                    print(f"align_ensemble: {label} → RMSD={res.rmsd:.3f} Å")
            except Exception as exc:
                warnings.warn(
                    f"align_ensemble: skipping '{label}' — {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                if self.verbose:
                    print(f"align_ensemble: {label} failed — {exc}")

        return EnsembleResult(results=results, labels=labels)

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

    def save_aligned_pdb(self, filename: str, subset_only: bool = False,
                         preserve_bfactor: bool = False):
        """Saves the aligned mobile structure to a PDB file. Maps alignment
        distance into the B-factor column (unless ``preserve_bfactor=True``).

        Delegates to :meth:`AlignmentResult.save_aligned_pdb` so there is a
        single implementation of the transform/B-factor logic.
        """
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")

        result = AlignmentResult(
            chosen=self.last_result["chosen"],
            seqguided=self.last_result["seqguided"],
            seqfree=self.last_result["seqfree"],
            ref_file=self.ref_file, mob_file=self.mob_file,
            mob_struct=self.mob_struct,
            ref_lens=self.ref_lens, mob_lens=self.mob_lens,
            verbose=self.verbose,
        )
        result.save_aligned_pdb(filename, subset_only=subset_only,
                                preserve_bfactor=preserve_bfactor)

    def get_log(self) -> str:
        """Returns the alignment log summary as a string."""
        if not self.last_result:
            raise ValueError("No alignment results available. Run align() first.")
        chosen = self.last_result["chosen"]
        seqguided = chosen.get("seqguided")
        seqfree = chosen.get("seqfree")
        if seqguided:
            rmsd = seqguided["si"]["rmsd"]
        elif seqfree:
            rmsd = seqfree.rmsd
        else:
            rmsd = None
        lines = []
        lines.append("PDB Aligner Result Log")
        lines.append("="*20)
        lines.append(f"Reference: {self.ref_file}")
        lines.append(f"Mobile: {self.mob_file}")
        lines.append(f"Chosen method: {chosen['name']}")
        lines.append(f"RMSD: {rmsd:.3f} Å" if rmsd is not None else "RMSD: None")
        lines.append(f"Reason: {chosen['reason']}")
        return "\n".join(lines)

    def save_log(self, filename: str):
        """Saves alignment log summary."""
        with open(filename, "w") as f:
            f.write(self.get_log() + "\n")


def inspect_structure(path_or_id, cache_dir=None):
    """List chains, residue counts, and sequences for a structure.

    Accepts a local file path or a remote ID (``pdb:XXXX`` / ``af:UniProtID``).
    Returns ``{"chains": {chain: n_residues}, "sequences": {chain: seq_str}}``.
    A public accessor so callers (e.g. the GUI) need no ``pdb_align.core``.
    """
    al = PDBAligner()
    if cache_dir:
        al._fetch_cache_dir = cache_dir
    al.add_reference(path_or_id)
    chains = {c: int(n) for c, n in al.ref_lens.items()}
    sequences = {c: str(rec.seq) for c, rec in al.ref_seqs.items()}
    return {"chains": chains, "sequences": sequences}
