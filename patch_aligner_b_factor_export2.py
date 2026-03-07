import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# Fix self.last_result
text = text.replace("if not self.last_result:", "if not self._chosen:")
text = text.replace("chosen = self.last_result[\"chosen\"]", "chosen = self._chosen")

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
