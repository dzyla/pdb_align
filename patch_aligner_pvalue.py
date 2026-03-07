import re

with open("pdb_align/aligner.py", "r") as f:
    text = f.read()

# I want to add tm_pvalue property
new_property = """    @property
    def tm_pvalue(self) -> Optional[float]:
        \"\"\"
        Returns the p-value of the TM-score for the aligned structures.
        \"\"\"
        tm = self.tm_score
        if tm is None:
            return None

        import pdb_align.metrics as metrics
        L = sum(self.ref_lens.values())
        return metrics.calculate_tm_pvalue(tm, L)
"""

text = re.sub(
    r'    @property\n    def rotation_matrix',
    new_property + '\n    @property\n    def rotation_matrix',
    text
)

with open("pdb_align/aligner.py", "w") as f:
    f.write(text)
