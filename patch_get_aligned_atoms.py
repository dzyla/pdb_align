import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

new_get_aligned = """def get_aligned_atoms_by_alignment(ref_struct: gemmi.Structure, ref_chains, mob_struct: gemmi.Structure, mob_chains, alignment, atoms: str = "CA", min_b_factor: float = 0.0):
    if not alignment: return [], []
    seqA, seqB = alignment.seqA, alignment.seqB

    if atoms == "backbone":
        target_atoms = {"N", "CA", "C", "O"}
    elif atoms == "all_heavy":
        target_atoms = None # All non-hydrogen
    else:
        target_atoms = {"CA"}

    def get_res_list(struct, chains):
        residues=[]
        model=struct[0]
        bounds_map = {}
        for c in chains:
            cid, start, end = _parse_chain_selector(c)
            if cid not in bounds_map: bounds_map[cid] = []
            if start is not None or end is not None: bounds_map[cid].append((start, end))

        for ch in chains:
            cid, _, _ = _parse_chain_selector(ch)
            chain = model.find_chain(cid)
            if chain:
                c_bounds = bounds_map.get(cid, [])
                for res in chain:
                    if res.name in AA_DICT:
                        resseq = res.seqid.num

                        if c_bounds:
                            in_bounds = False
                            for (start, end) in c_bounds:
                                if (start is None or resseq >= start) and (end is None or resseq <= end):
                                    in_bounds = True
                                    break
                            if not in_bounds:
                                continue

                        has_ca = False
                        for atom in res:
                            if atom.name == "CA":
                                has_ca = True
                                break
                        if atoms == "CA" and not has_ca:
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

                        residues.append(res)
        return residues

    ref_res=get_res_list(ref_struct, ref_chains); mob_res=get_res_list(mob_struct, mob_chains)
    ref_idx=0; mob_idx=0; ref_atoms=[]; mob_atoms=[]

    class PseudoAtom:
        def __init__(self, coord, name, chain_name, res_seq, res_icode):
            self.coord = coord
            self.name = name
            self.chain_name = chain_name
            self.res_seq = res_seq
            self.res_icode = res_icode
        def get_coord(self):
            return self.coord
        def get_name(self):
            return self.name

    for a,b in zip(seqA, seqB):
        r_match=None; m_match=None
        if a!='-':
            while ref_idx<len(ref_res):
                r=ref_res[ref_idx]
                if AA_DICT.get(r.name)==a:
                    r_match=r; ref_idx+=1; break
                ref_idx+=1
        if b!='-':
            while mob_idx<len(mob_res):
                m=mob_res[mob_idx]
                if AA_DICT.get(m.name)==b:
                    m_match=m; mob_idx+=1; break
                mob_idx+=1

        if r_match is not None and m_match is not None:
            r_parent_chain = r_match.last_chain_name if hasattr(r_match, 'last_chain_name') else "A"
            m_parent_chain = m_match.last_chain_name if hasattr(m_match, 'last_chain_name') else "A"
            for atA in r_match:
                if atA.element.name == "H": continue
                if target_atoms is not None and atA.name not in target_atoms: continue
                for atB in m_match:
                    if atB.name == atA.name:
                        ref_atoms.append(PseudoAtom(np.array(atA.pos.tolist(), dtype=float), atA.name, "ref", r_match.seqid.num, r_match.seqid.icode if hasattr(r_match.seqid, 'has_icode') and r_match.seqid.has_icode() else ""))
                        mob_atoms.append(PseudoAtom(np.array(atB.pos.tolist(), dtype=float), atB.name, "mob", m_match.seqid.num, m_match.seqid.icode if hasattr(m_match.seqid, 'has_icode') and m_match.seqid.has_icode() else ""))
                        break

    return ref_atoms, mob_atoms"""

text = re.sub(r'def get_aligned_atoms_by_alignment\(.*?(?=\n\ndef superimpose_atoms)', new_get_aligned, text, flags=re.DOTALL)

with open("pdb_align/core.py", "w") as f:
    f.write(text)
