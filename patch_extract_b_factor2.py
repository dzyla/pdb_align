import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# Fix the third call site inside sequence_independent_alignment_joined_v2
text = text.replace("mob_all_infos=_extract_ca_infos(_parse_path(file_mob), chain_filter=None)",
"mob_all_infos=_extract_ca_infos(_parse_path(file_mob), chain_filter=None, min_b_factor=min_b_factor)")

with open("pdb_align/core.py", "w") as f:
    f.write(text)
