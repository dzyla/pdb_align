import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Replace clone() with clone() is fine if gemmi Structure has it, wait gemmi structure does have clone(), let's check what mob_struct is
# It seems mob_struct is a Biopython Structure maybe? No, we replaced it with gemmi. But maybe it's still being parsed via some old method.

# Wait, the structure parsing in aligner.py line ~630:
# self.mob_struct = _parse_path(mob_file)
# which returns gemmi.Structure. gemmi.Structure DOES have a clone() method. Let's see what it is.
