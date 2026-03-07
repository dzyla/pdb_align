import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# I need to update _extract_ca_infos to accept min_b_factor
new_extract = """def _extract_ca_infos(
    struct: gemmi.Structure, chain_filter: Optional[List[str]], min_b_factor: float = 0.0
) -> List[ResidueInfo]:
    model = struct[0]
    infos = []
    idx = 0

    # Parse bounds
    bounds_map = {}
    valid_chain_ids = set()
    if chain_filter is not None:
        for c in chain_filter:
            cid, start, end = _parse_chain_selector(c)
            valid_chain_ids.add(cid)
            if cid not in bounds_map:
                bounds_map[cid] = []
            if start is not None or end is not None:
                bounds_map[cid].append((start, end))

    for chain in model:
        if chain_filter is not None and chain.name not in valid_chain_ids:
            continue

        c_bounds = bounds_map.get(chain.name, [])

        for res in chain:
            if res.name not in AA_DICT:
                continue

            resseq = res.seqid.num
            icode = res.seqid.icode if hasattr(res.seqid, 'has_icode') and res.seqid.has_icode() or (hasattr(res.seqid, 'icode') and res.seqid.icode != ' ') else ""

            # Sub-domain filtering
            if c_bounds:
                in_bounds = False
                for start, end in c_bounds:
                    if (start is None or resseq >= start) and (
                        end is None or resseq <= end
                    ):
                        in_bounds = True
                        break
                if not in_bounds:
                    continue

            ca = None
            for atom in res:
                if atom.name == "CA":
                    ca = atom
                    break

            if ca is None:
                continue

            if ca.b_iso < min_b_factor:
                continue

            coord = np.array(ca.pos.tolist(), dtype=float)
            infos.append(
                ResidueInfo(
                    idx=idx,
                    chain_id=chain.name,
                    resseq=int(resseq),
                    icode=icode.strip() if isinstance(icode, str) else "",
                    resname=res.name,
                    coord=coord,
                )
            )
            idx += 1

    if not infos:
        raise ValueError(f"No C-alpha atoms found for selected chains.")
    return infos"""

text = re.sub(r'def _extract_ca_infos.*?(?=\n\n\ndef _pairwise_dists)', new_extract, text, flags=re.DOTALL)


# Update sequence_independent_alignment_joined_v2 to accept min_b_factor
text = re.sub(r'def sequence_independent_alignment_joined_v2\(\n    file_ref: str,\n    file_mob: str,\n    chains_ref: Optional\[List\[Union\[str, int\]\]\] = None,\n    chains_mob: Optional\[List\[Union\[str, int\]\]\] = None,\n    method: str = "auto",\n    shape_nbins: int = 24,\n    shape_gap_penalty: float = 2\.0,\n    shape_band_frac: float = 0\.20,\n    inlier_rmsd_cut: float = 3\.0,\n    inlier_quantile: float = 0\.85,\n    refinement_iters: int = 2,\n    atoms: str = "CA",\n\) -> AlignmentResultSF:',
r'''def sequence_independent_alignment_joined_v2(
    file_ref: str,
    file_mob: str,
    chains_ref: Optional[List[Union[str, int]]] = None,
    chains_mob: Optional[List[Union[str, int]]] = None,
    method: str = "auto",
    shape_nbins: int = 24,
    shape_gap_penalty: float = 2.0,
    shape_band_frac: float = 0.20,
    inlier_rmsd_cut: float = 3.0,
    inlier_quantile: float = 0.85,
    refinement_iters: int = 2,
    atoms: str = "CA",
    min_b_factor: float = 0.0,
) -> AlignmentResultSF:''', text)


text = text.replace("ref_infos = _extract_ca_infos(ref_struct, ref_ids)", "ref_infos = _extract_ca_infos(ref_struct, ref_ids, min_b_factor)")
text = text.replace("mob_infos = _extract_ca_infos(mob_struct, mob_ids)", "mob_infos = _extract_ca_infos(mob_struct, mob_ids, min_b_factor)")

with open("pdb_align/core.py", "w") as f:
    f.write(text)


with open("pdb_align/aligner.py", "r") as f:
    aligner_text = f.read()

# Update align to accept min_plddt / min_b_factor
aligner_text = re.sub(r'def align\(\n        self,\n        mode: str = "auto",\n        seq_gap_open: float = -10,\n        seq_gap_extend: float = -0\.5,\n        atoms: str = "CA",\n        \*\*kwargs,\n    \) -> AlignmentResult:',
r'''def align(
        self,
        mode: str = "auto",
        seq_gap_open: float = -10,
        seq_gap_extend: float = -0.5,
        atoms: str = "CA",
        min_plddt: float = 0.0,
        min_b_factor: float = 0.0,
        **kwargs,
    ) -> AlignmentResult:
        min_b_val = max(min_plddt, min_b_factor)''', aligner_text)

# pass min_b_val to sequence_independent_alignment_joined_v2
aligner_text = aligner_text.replace("method=sf_method, atoms=atoms, **kwargs", "method=sf_method, atoms=atoms, min_b_factor=min_b_val, **kwargs")


# Also need to add min_b_factor filtering to sequence-guided alignment.
# We modify get_aligned_atoms_by_alignment
with open("pdb_align/core.py", "r") as f:
    text = f.read()

text = re.sub(r'def get_aligned_atoms_by_alignment\(\n    ref_struct: gemmi\.Structure, ref_chains, mob_struct: gemmi\.Structure, mob_chains, alignment, atoms: str = "CA"\n\):',
r'''def get_aligned_atoms_by_alignment(
    ref_struct: gemmi.Structure, ref_chains, mob_struct: gemmi.Structure, mob_chains, alignment, atoms: str = "CA", min_b_factor: float = 0.0
):''', text)

text = text.replace("if atoms == \"CA\" and not has_ca:\n                            continue\n                        residues.append(res)",
"""if atoms == "CA" and not has_ca:
                            continue

                        # Filter by B-factor for CA atoms if requested
                        if min_b_factor > 0.0:
                            b_factor_ok = False
                            for atom in res:
                                if atom.name == "CA" and atom.b_iso >= min_b_factor:
                                    b_factor_ok = True
                                    break
                            if not b_factor_ok:
                                continue

                        residues.append(res)""")

with open("pdb_align/core.py", "w") as f:
    f.write(text)

aligner_text = aligner_text.replace(
    "ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(\n                    self.ref_struct,\n                    ref_chs,\n                    self.mob_struct,\n                    mob_chs,\n                    aln,\n                    atoms=atoms,\n                )",
    "ref_atoms, mob_atoms = get_aligned_atoms_by_alignment(\n                    self.ref_struct,\n                    ref_chs,\n                    self.mob_struct,\n                    mob_chs,\n                    aln,\n                    atoms=atoms,\n                    min_b_factor=min_b_val,\n                )"
)

with open("pdb_align/aligner.py", "w") as f:
    f.write(aligner_text)
