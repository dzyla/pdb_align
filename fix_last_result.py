import re

with open("pdb_align/result.py", "r") as f:
    text = f.read()

text = text.replace("if not self.last_result:\n            raise ValueError(\"No alignment results available. Run align() first.\")\n\n        chosen = self.last_result[\"chosen\"]",
"if not self._chosen:\n            raise ValueError(\"No alignment results available. Run align() first.\")\n\n        chosen = self._chosen")

with open("pdb_align/result.py", "w") as f:
    f.write(text)
