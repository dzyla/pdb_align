import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Since self.mob_struct is apparently a Biopython Structure here, let's just make it generic. Wait, why is it Biopython?
# Let's see what self.mob_struct is in aligner.py:
# def add_mobile:
# self.mob_struct = _parse_path(mob_file)
# We modified _parse_path to return gemmi.read_structure!
# Let's check _parse_path in core.py.
