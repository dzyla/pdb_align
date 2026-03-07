import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# I see my standardize2.py didn't match and replace properly.
text = re.sub(r'def _parse_path\(path:str\) -> _Structure.Structure:\n\s+parser = MMCIFParser\(QUIET=True\) if path.lower\(\).endswith\(\("\.cif","\.mmcif"\)\) else PDBParser\(QUIET=True\)\n\s+return parser.get_structure\(os.path.basename\(path\), path\)',
              r'def _parse_path(path: str) -> gemmi.Structure:\n    return gemmi.read_structure(path)', text)

text = re.sub(r'def _chain_ids\(struct:_Structure.Structure\) -> List\[str\]:\n\s+return \[ch.id for ch in list\(struct\)\[0\]\]',
              r'def _chain_ids(struct: gemmi.Structure) -> List[str]:\n    return [ch.name for ch in struct[0]]', text)


text = re.sub(r'def _resolve_selectors\(struct: _Structure.Structure, sel: Optional\[List\[Union\[str, int\]\]\]\) -> Optional\[List\[str\]\]:',
              r'def _resolve_selectors(struct: gemmi.Structure, sel: Optional[List[Union[str, int]]]) -> Optional[List[str]]:', text)

text = re.sub(r'def extract_sequences_and_lengths\(struct: Structure.Structure, fname: str\):.*?return seqs, lens',
r'''def extract_sequences_and_lengths(struct: gemmi.Structure, fname: str):
    seqs: Dict[str, SeqRecord] = {}
    lens: Dict[str, int] = {}
    try:
        model = struct[0]
    except Exception:
        return {}, {}
    for chain in model:
        seq = []
        ca_count = 0
        for res in chain:
            resname = res.name
            if resname in AA_DICT:
                seq.append(AA_DICT[resname])
                has_ca = False
                for atom in res:
                    if atom.name == "CA":
                        has_ca = True
                        break
                if has_ca:
                    ca_count += 1
        if seq:
            seqs[chain.name] = SeqRecord(
                Seq("".join(seq)),
                id=f"{fname}_{chain.name}",
                description=f"Chain {chain.name}",
            )
            lens[chain.name] = int(ca_count)
    return seqs, lens''', text, flags=re.DOTALL)

with open("pdb_align/core.py", "w") as f:
    f.write(text)
