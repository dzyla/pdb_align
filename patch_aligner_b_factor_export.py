import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# We need to map the actual alignment distance into the B-factor column
new_save_pdb = """def save_aligned_pdb(self, filename: str, subset_only: bool = False):
        \"\"\"Saves the aligned mobile structure to a PDB file. Maps alignment distance into B-factor.\"\"\"
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
            out_struct = self.mob_struct.clone()

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
                out_struct.write_minimal_cif(filename)
            else:
                out_struct.write_pdb(filename)"""

text = re.sub(r'def save_aligned_pdb\(self, filename: str, subset_only: bool = False\):.*?(?=\n\n    def get_log)', new_save_pdb, text, flags=re.DOTALL)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
