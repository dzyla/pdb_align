import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# I see my patch_b_factor.py substitution for _extract_ca_infos failed because of whitespace mismatch.
# Let me explicitly rewrite it since I know what it needs to look like.
new_extract_ca = """def _extract_ca_infos(
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
            icode = res.seqid.icode if hasattr(res.seqid, 'has_icode') and res.seqid.has_icode() else ""
            if not icode and hasattr(res.seqid, 'icode') and res.seqid.icode != ' ':
                icode = res.seqid.icode

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

            if min_b_factor > 0.0 and ca.b_iso < min_b_factor:
                continue

            coord = np.array(ca.pos.tolist(), dtype=float)
            infos.append(
                ResidueInfo(
                    idx=idx,
                    chain_id=chain.name,
                    resseq=int(resseq),
                    icode=icode.strip() if hasattr(icode, 'strip') else "",
                    resname=res.name,
                    coord=coord,
                )
            )
            idx += 1

    if not infos:
        raise ValueError(f"No C-alpha atoms found for selected chains.")
    return infos"""

text = re.sub(r'def _extract_ca_infos\(struct:_Structure\.Structure, chain_filter:Optional\[List\[str\]\]\)->List\[ResidueInfo\]:.*?(?=\ndef _pairwise_dists)', new_extract_ca, text, flags=re.DOTALL)

with open("pdb_align/core.py", "w") as f:
    f.write(text)
