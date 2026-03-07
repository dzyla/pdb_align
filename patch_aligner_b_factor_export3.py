import re

with open("pdb_align/result.py", "r") as f:
    text = f.read()

text = text.replace("out_struct = self.mob_struct.clone()", "out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, 'clone') else self.mob_struct.copy()")

with open("pdb_align/result.py", "w") as f:
    f.write(text)

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

text = text.replace("out_struct = self.mob_struct.clone()", "out_struct = self.mob_struct.clone() if hasattr(self.mob_struct, 'clone') else self.mob_struct.copy()")
text = text.replace("self._chosen:", "self._chosen") # fixing any previous issues

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
