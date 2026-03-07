import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Add min_plddt and min_b_factor to align signature
text = re.sub(
    r'def align\(self, mode: str = "auto", seq_gap_open: float = -10, seq_gap_extend: float = -0\.5, atoms: str = "CA", \*\*kwargs\):',
    r'def align(self, mode: str = "auto", seq_gap_open: float = -10, seq_gap_extend: float = -0.5, atoms: str = "CA", min_plddt: float = 0.0, min_b_factor: float = 0.0, **kwargs):\n        min_b_val = max(min_plddt, min_b_factor)',
    text
)

# And make sure get_aligned_atoms_by_alignment uses min_b_val
text = re.sub(
    r'get_aligned_atoms_by_alignment\(self\.ref_struct, ref_chs, self\.mob_struct, mob_chs, aln, atoms=atoms\)',
    r'get_aligned_atoms_by_alignment(self.ref_struct, ref_chs, self.mob_struct, mob_chs, aln, atoms=atoms, min_b_factor=min_b_val)',
    text
)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
