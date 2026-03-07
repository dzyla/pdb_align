import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Replace sequence_independent_alignment_joined_v2 call
text = re.sub(
    r'res = sequence_independent_alignment_joined_v2\(\n\s+file_ref=self\.ref_file, file_mob=self\.mob_file,\n\s+chains_ref=ref_chs, chains_mob=mob_chs,\n\s+method=sf_method, atoms=atoms, \*\*kwargs\n\s+\)',
    r'res = sequence_independent_alignment_joined_v2(\n                    file_ref=self.ref_file, file_mob=self.mob_file,\n                    chains_ref=ref_chs, chains_mob=mob_chs,\n                    method=sf_method, atoms=atoms, min_b_factor=min_b_val, **kwargs\n                )',
    text
)

# And fix the try except
text = re.sub(r'except Exception as e:\n\s+pass', r'except Exception as e:\n                import traceback\n                traceback.print_exc()', text)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
